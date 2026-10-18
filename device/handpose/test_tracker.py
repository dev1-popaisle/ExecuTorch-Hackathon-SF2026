"""Check the live NPU tracker (numpy pipeline) against the PyTorch reference pipeline on the two-hand test image."""

import cv2
import numpy as np

from hand_models import ROOT, draw_hands, load_models, run_pipeline, torch_fns
from hand_tracker import SHM, HandTracker

sample = cv2.cvtColor(cv2.imread(str(ROOT / "assets" / "sample_input.jpeg")), cv2.COLOR_BGR2RGB)
frame = np.concatenate([sample, sample[:, ::-1]], axis=1).copy()
frame = cv2.resize(frame, (1280, 640), interpolation=cv2.INTER_CUBIC)
h, w = frame.shape[:2]

ref = run_pipeline(frame, *torch_fns(load_models()))
ref_by_x = sorted(ref, key=lambda r: r["mp_keypoints"][:, 0].mean())

SHM.mkdir(parents=True, exist_ok=True)
tracker = HandTracker(mirrored_input=True)  # report the model's raw labels, like the reference pipeline
try:
    for i in range(5):  # first frame detects; later frames run purely from tracked ROIs
        res = tracker.process(frame)
        hands = sorted(res["hands"], key=lambda r: np.mean(r["pts"], axis=0)[0])
        errs = []
        for got, want in zip(hands, ref_by_x):
            pts = np.array(got["pts"]) * (w, h)
            errs.append(float(np.linalg.norm(pts - want["mp_keypoints"], axis=1).mean()))
        print(f"frame {i}: hands={[(x['label'], x['score']) for x in hands]} ref={[('Right' if r['handedness_right'] > 0.5 else 'Left') for r in ref_by_x]} "
              f"mean_err_px={np.round(errs, 2).tolist()} ms={res['ms']}")
    vis = [{"mp_keypoints": np.array(x["pts"]) * (w, h), "handedness_right": 1.0 if x["label"] == "Right" else 0.0,
            "hand_flag": x["score"], "roi_box": np.zeros((4, 2))} for x in hands]
    cv2.imwrite(str(ROOT / "export" / "step2" / "live_tracker_test.jpg"), draw_hands(frame, vis))
finally:
    tracker.close()
