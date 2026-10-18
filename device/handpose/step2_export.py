"""Step 2: export MediaPipe-Hand (palm detector + landmark) and RTMPose-hand to ExecuTorch on the VENTUNO Q.

  * XNNPACK fp32 (CPU reference / fallback)
  * QNN HTP V75 (QCS8275 NPU), PTQ with on-device calibration data from Step 1

Then runs every .pte with the on-device runners, compares against PyTorch, and runs the full two-hand
pipeline end to end on each backend.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from export_lib import compare, delegation_summary, export_qnn, export_xnnpack, run_qnn, run_xnnpack
from hand_models import INPUT_SHAPE, ROOT, draw_hands, load_models, run_pipeline, torch_fns

REF = ROOT / "export" / "reference"
CALIB = ROOT / "export" / "calibration"
PTE_DIR = ROOT / "export" / "pte"
OUT = ROOT / "export" / "step2"

OUTPUT_SHAPES = {
    "mp_palm_detector": [(1, 2944, 18), (1, 2944, 1)],
    "mp_hand_landmark": [(1,), (1,), (1, 21, 3)],
    "rtmpose_hand": [(1, 21, 512), (1, 21, 512)],
}
CALIB_SET = {"mp_palm_detector": "detector.pt", "mp_hand_landmark": "landmark.pt", "rtmpose_hand": "landmark.pt"}
RUNNERS = {"xnnpack": run_xnnpack, "qnn": run_qnn}


def pte_path(name: str, backend: str, quant: str) -> Path:
    return PTE_DIR / (f"{name}_xnnpack_fp32.pte" if backend == "xnnpack" else f"{name}_qnn_{quant}.pte")


def runner_fn(name: str, backend: str, quant: str):
    pte, run = pte_path(name, backend, quant), RUNNERS[backend]

    def fn(x: torch.Tensor):
        outs, _ = run(pte, [x.numpy()], iterations=1)
        return tuple(torch.from_numpy(o.reshape(s)) for o, s in zip(outs, OUTPUT_SHAPES[name]))
    return fn


def mean_px(a: list[dict], b: list[dict], key: str) -> float | None:
    if len(a) != len(b) or not a:
        return None
    return round(float(np.mean([np.linalg.norm(x[key] - y[key], axis=1).mean() for x, y in zip(a, b)])), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", default=["xnnpack", "qnn"])
    ap.add_argument("--models", nargs="+", default=list(OUTPUT_SHAPES))
    ap.add_argument("--quant", default="16a8w", choices=["8a8w", "16a8w", "16a16w"])
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args()
    PTE_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    models = load_models()
    example = (torch.rand(INPUT_SHAPE),)
    report: dict = {"quant": args.quant, "models": {}}

    for name in args.models:
        model = models[name]
        r = report["models"][name] = {}
        for backend in args.backends:
            pte = pte_path(name, backend, args.quant)
            if not args.skip_export:
                t0 = time.time()
                if backend == "xnnpack":
                    export_xnnpack(model, example, pte)
                else:
                    calib = torch.load(CALIB / CALIB_SET[name])
                    export_qnn(model, example, list(calib.split(1)), pte, quant=args.quant)
                print(f"[export] {pte.name} {pte.stat().st_size / 1e6:.1f} MB in {time.time() - t0:.0f}s", flush=True)

            entry = {"pte": pte.name, "size_MB": round(pte.stat().st_size / 1e6, 2)}
            if backend == "qnn":
                entry["delegation"] = delegation_summary(pte)

            per_input = []
            for ref_in in sorted(REF.glob(f"{name}_in?.raw")):
                x = np.fromfile(ref_in, dtype=np.float32)
                outs, _ = RUNNERS[backend](pte, [x], iterations=1)
                cmp = [compare(np.fromfile(REF / f"{ref_in.stem}_out{j}.raw", dtype=np.float32), o)
                       for j, o in enumerate(outs)]
                per_input.append({k: [round(c[k], 6) for c in cmp] for k in ("cosine", "max_abs", "mean_abs")})
            entry["vs_pytorch"] = per_input
            _, ms = RUNNERS[backend](pte, [np.random.rand(*INPUT_SHAPE).astype(np.float32)], iterations=50)
            entry["latency_ms"] = round(ms, 3)
            r[backend] = entry
            print(f"[{name}/{backend}] latency={ms:.2f}ms cosine={[p['cosine'] for p in per_input]}", flush=True)

    # End-to-end two-hand pipeline on each backend vs PyTorch.
    if set(args.models) == set(OUTPUT_SHAPES):
        sample = cv2.cvtColor(cv2.imread(str(ROOT / "assets" / "sample_input.jpeg")), cv2.COLOR_BGR2RGB)
        frame = np.concatenate([sample, sample[:, ::-1]], axis=1).copy()
        ref_hands = run_pipeline(frame, *torch_fns(models))
        report["pipeline"] = {}
        for backend in args.backends:
            fns = [runner_fn(n, backend, args.quant) for n in ("mp_palm_detector", "mp_hand_landmark", "rtmpose_hand")]
            hands = run_pipeline(frame, *fns)
            report["pipeline"][backend] = {
                "hands": [{"hand": "right" if h["handedness_right"] > 0.5 else "left",
                           "palm_score": round(h["palm_score"], 4), "hand_flag": round(h["hand_flag"], 4)} for h in hands],
                "mediapipe_kpt_err_px_vs_torch": mean_px(hands, ref_hands, "mp_keypoints"),
                "rtmpose_kpt_err_px_vs_torch": mean_px(hands, ref_hands, "rtm_keypoints"),
            }
            cv2.imwrite(str(OUT / f"two_hands_{backend}_mediapipe.jpg"), draw_hands(frame, hands, "mp_keypoints"))
            cv2.imwrite(str(OUT / f"two_hands_{backend}_rtmpose.jpg"), draw_hands(frame, hands, "rtm_keypoints"))
            print(f"[pipeline/{backend}] {report['pipeline'][backend]}", flush=True)

    (OUT / f"step2_report_{args.quant}.json").write_text(json.dumps(report, indent=2))
    print("STEP2_DONE")


if __name__ == "__main__":
    main()
