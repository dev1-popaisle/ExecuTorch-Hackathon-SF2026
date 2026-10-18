# ====================================================================================================
# Sign & Say - Smart Home Mobility Project
# Author: Umit D. Sami
# Open source project - ExecuTorch Hackathon 2026 (San Francisco) - 11.18.2026
#
# This is a smart home mobility project for physically challenged individuals. The project is an IoT
# system running on the Arduino VENTUNO Q without the use of any cloud resources (everything runs inside
# the device, including advanced AI computation tasks such as speech and vision recognition).
# The project allows a physically challenged person to set the temperature in the house, turn the lights
# on and off, and change TV channels in two ways:
#   1. VENTUNO Q Voice AI assistant module
#   2. VENTUNO Q Hand Recognition with ExecuTorch running in the device
# ====================================================================================================
"""Step 1: load MediaPipe-Hand (Qualcomm AI Hub pin) + RTMPose-hand, sanity-check them in PyTorch,
and save reference inputs/outputs + calibration tensors for Step 2."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from hand_models import INPUT_SHAPE, ROOT, draw_hands, load_models, run_pipeline, to_input, torch_fns

OUT = ROOT / "export" / "reference"
CALIB = ROOT / "export" / "calibration"


def two_hand_image(img_rgb: np.ndarray) -> np.ndarray:
    """Original + mirrored copy side by side -> one left and one right hand in a single frame."""
    return np.concatenate([img_rgb, img_rgb[:, ::-1]], axis=1).copy()


def augment(img_rgb: np.ndarray, rng: np.random.Generator, n: int) -> list[np.ndarray]:
    h, w = img_rgb.shape[:2]
    out = []
    for _ in range(n):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-45, 45), rng.uniform(0.7, 1.2))
        m[:, 2] += rng.uniform(-0.12, 0.12, 2) * (w, h)
        a = cv2.warpAffine(img_rgb, m, (w, h), borderMode=cv2.BORDER_REFLECT)
        if rng.random() < 0.5:
            a = a[:, ::-1]
        a = np.clip(a.astype(np.float32) * rng.uniform(0.6, 1.3) + rng.uniform(-30, 30), 0, 255)
        out.append(a.astype(np.uint8))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CALIB.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    models = load_models()
    report = {}
    for name, m in models.items():
        n_params = sum(p.numel() for p in m.parameters())
        x = torch.rand(INPUT_SHAPE)
        with torch.no_grad():
            outs = m(x)
            t0 = time.perf_counter()
            for _ in range(5):
                m(x)
            ms = (time.perf_counter() - t0) / 5 * 1000
        report[name] = {"params_M": round(n_params / 1e6, 3), "outputs": [list(o.shape) for o in outs],
                        "torch_cpu_ms": round(ms, 1)}
        print(f"[{name}] params={n_params / 1e6:.2f}M outputs={[tuple(o.shape) for o in outs]} cpu={ms:.1f}ms")

    # Qualcomm reference tensors (same inputs AI Hub uses to validate this model).
    det_in = torch.from_numpy(np.load(ROOT / "assets" / "sample_detector_inputs.npy")).float()
    lm_in = torch.from_numpy(np.load(ROOT / "assets" / "sample_landmark_inputs.npy")).float()

    palm_fn, lm_fn, rtm_fn = torch_fns(models)
    sample = cv2.cvtColor(cv2.imread(str(ROOT / "assets" / "sample_input.jpeg")), cv2.COLOR_BGR2RGB)
    test_images = {"sample_one_hand": sample, "sample_two_hands": two_hand_image(sample)}
    cam = ROOT / "assets" / "camera_frame.jpg"
    if cam.exists():
        test_images["camera_frame"] = cv2.cvtColor(cv2.imread(str(cam)), cv2.COLOR_BGR2RGB)

    report["pipeline"] = {}
    crops = [lm_in]
    for name, img in test_images.items():
        hands = run_pipeline(img, palm_fn, lm_fn, rtm_fn)
        report["pipeline"][name] = [
            {"hand": "right" if h["handedness_right"] > 0.5 else "left", "palm_score": round(h["palm_score"], 3),
             "hand_flag": round(h["hand_flag"], 3),
             "mp_vs_rtm_mean_px": round(float(np.linalg.norm(h["mp_keypoints"] - h["rtm_keypoints"], axis=1).mean()), 2)}
            for h in hands]
        print(f"[pipeline:{name}] {report['pipeline'][name]}")
        cv2.imwrite(str(OUT / f"{name}_mediapipe.jpg"), draw_hands(img, hands, "mp_keypoints"))
        cv2.imwrite(str(OUT / f"{name}_rtmpose.jpg"), draw_hands(img, hands, "rtm_keypoints"))

    # Reference inputs for Step 2 numeric validation (raw float32, NCHW).
    from blazebase import resize_pad
    det_ref = to_input(resize_pad(test_images["sample_two_hands"])[0])
    refs = {"mp_palm_detector": [det_in, det_ref], "mp_hand_landmark": [lm_in], "rtmpose_hand": [lm_in]}
    for name, inputs in refs.items():
        for i, x in enumerate(inputs):
            x.numpy().astype(np.float32).tofile(OUT / f"{name}_in{i}.raw")
            with torch.no_grad():
                for j, o in enumerate(models[name](x)):
                    o.numpy().astype(np.float32).tofile(OUT / f"{name}_in{i}_out{j}.raw")

    # Calibration sets: full frames for the detector, upright hand crops for the landmark models.
    rng = np.random.default_rng(0)
    det_cal, lm_cal = [det_in, det_ref], list(crops)
    sources = [sample, two_hand_image(sample)] + ([test_images["camera_frame"]] if cam.exists() else [])
    for src in sources:
        for a in augment(src, rng, 24):
            det_cal.append(to_input(resize_pad(a)[0]))
    palm_net = models["mp_palm_detector"].net
    for a in augment(sample, rng, 48) + augment(two_hand_image(sample), rng, 24):
        hands_crops = []

        def grab(c):
            hands_crops.append(c.clone())
            return lm_fn(c)

        run_pipeline(a, palm_fn, grab, None, palm_net=palm_net)
        lm_cal.extend(hands_crops)
    torch.save(torch.cat(det_cal), CALIB / "detector.pt")
    torch.save(torch.cat(lm_cal), CALIB / "landmark.pt")
    report["calibration"] = {"detector_frames": len(det_cal), "hand_crops": len(lm_cal)}
    print(f"[calibration] detector frames={len(det_cal)} hand crops={len(lm_cal)}")

    (OUT / "step1_report.json").write_text(json.dumps(report, indent=2))
    print("STEP1_OK")


if __name__ == "__main__":
    main()
