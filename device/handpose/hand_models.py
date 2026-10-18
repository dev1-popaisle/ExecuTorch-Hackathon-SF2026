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
"""Model factory + reference two-stage pipeline (palm detector -> per-hand landmarks).

All exported models share one input contract: RGB float32 (1, 3, 256, 256) in [0, 1].
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
MEDIAPIPE_DIR = ROOT / "third_party" / "MediaPipePyTorch"  # zmurez/MediaPipePyTorch @ 65f2549 (Qualcomm AI Hub pin)
RTMPOSE_CKPT = ROOT / "models" / "rtmpose-m_hand5_256.pth"
sys.path.insert(0, str(MEDIAPIPE_DIR))

from blazebase import resize_pad, denormalize_detections  # noqa: E402
from blazehand_landmark import BlazeHandLandmark  # noqa: E402
from blazepalm import BlazePalm  # noqa: E402

from rtmpose_hand import RTMPoseHand, decode_simcc  # noqa: E402

INPUT_SHAPE = (1, 3, 256, 256)
MAX_HANDS = 2

# MediaPipe hand topology: wrist=0, thumb 1-4, index 5-8, middle 9-12, ring 13-16, pinky 17-20.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


class PalmDetector(torch.nn.Module):
    """BlazePalm raw head outputs: boxes (1, 2944, 18), score logits (1, 2944, 1)."""

    def __init__(self, net: BlazePalm):
        super().__init__()
        self.net = net

    def forward(self, image):
        boxes, scores = self.net(image)
        return boxes, scores


class HandLandmark(torch.nn.Module):
    """BlazeHandLandmark: hand_flag (1,), handedness (1,), landmarks (1, 21, 3) normalized to the crop."""

    def __init__(self, net: BlazeHandLandmark):
        super().__init__()
        self.net = net

    def forward(self, image):
        hand_flag, handed, landmarks = self.net(image)
        return hand_flag, handed, landmarks


def load_palm_net() -> BlazePalm:
    palm = BlazePalm()
    palm.load_state_dict(torch.load(MEDIAPIPE_DIR / "blazepalm.pth", map_location="cpu"))
    palm.load_anchors(str(MEDIAPIPE_DIR / "anchors_palm.npy"))
    palm.min_score_thresh = 0.75
    return palm.eval()


def load_models() -> dict[str, torch.nn.Module]:
    palm = load_palm_net()
    landmark = BlazeHandLandmark()
    landmark.load_state_dict(torch.load(MEDIAPIPE_DIR / "blazehand_landmark.pth", map_location="cpu"))
    rtm = RTMPoseHand().load_mmpose_checkpoint(RTMPOSE_CKPT)
    return {
        "mp_palm_detector": PalmDetector(palm).eval(),
        "mp_hand_landmark": HandLandmark(landmark.eval()).eval(),
        "rtmpose_hand": rtm.eval(),
    }


def to_input(img_rgb_u8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img_rgb_u8).permute(2, 0, 1).unsqueeze(0).float().div(255.0).contiguous()


RunFn = Callable[[torch.Tensor], tuple]


def run_pipeline(frame_rgb: np.ndarray, palm_fn: RunFn, landmark_fn: RunFn, rtm_fn: RunFn | None = None,
                 palm_net: BlazePalm | None = None, max_hands: int = MAX_HANDS, min_hand_flag: float = 0.5):
    """Full-frame reference pipeline. Each *_fn maps a (1,3,256,256) tensor to that model's raw outputs,
    so the same code validates PyTorch, XNNPACK and QNN results."""
    palm_net = palm_net or load_palm_net()
    lm_net = BlazeHandLandmark()  # only used for its (weight-free) ROI helpers

    img256, _, scale, pad = resize_pad(frame_rgb)
    boxes, scores = palm_fn(to_input(img256))
    dets = palm_net._tensors_to_detections(torch.as_tensor(boxes), torch.as_tensor(scores), palm_net.anchors)[0]
    dets = palm_net._weighted_non_max_suppression(dets)
    dets = torch.stack(dets)[:max_hands] if dets else torch.zeros((0, 19))
    palm_scores = dets[:, -1].clone()  # denormalize_detections also rescales the trailing score column
    dets = denormalize_detections(dets.clone(), scale, pad)

    hands = []
    if len(dets):
        xc, yc, roi_scale, theta = palm_net.detection2roi(dets)
        crops, affines, roi_box = lm_net.extract_roi(frame_rgb, xc, yc, theta, roi_scale)
        for i in range(len(crops)):
            crop = crops[i:i + 1].contiguous()
            flag, handed, lms = landmark_fn(crop)
            flag = float(torch.as_tensor(flag).reshape(-1)[0])
            if flag < min_hand_flag:
                continue
            lms = torch.as_tensor(lms).reshape(1, 21, 3).clone()
            pts = lm_net.denormalize_landmarks(lms, affines[i:i + 1])[0]
            hand = {
                "palm_score": float(palm_scores[i]),
                "hand_flag": flag,
                "handedness_right": float(torch.as_tensor(handed).reshape(-1)[0]),
                "mp_keypoints": pts[:, :2].numpy(),
                "mp_depth": pts[:, 2].numpy(),
                "roi_box": roi_box[i].T.numpy(),
            }
            if rtm_fn is not None:
                sx, sy = rtm_fn(crop)
                kp, sc = decode_simcc(torch.as_tensor(sx), torch.as_tensor(sy))
                kp3 = torch.cat([kp / 256.0, torch.zeros(1, 21, 1)], dim=-1)
                hand["rtm_keypoints"] = lm_net.denormalize_landmarks(kp3, affines[i:i + 1])[0][:, :2].numpy()
                hand["rtm_scores"] = sc[0].numpy()
            hands.append(hand)
    return hands


def torch_fns(models: dict[str, torch.nn.Module]):
    def wrap(m):
        def fn(x):
            with torch.no_grad():
                return m(x)
        return fn
    return wrap(models["mp_palm_detector"]), wrap(models["mp_hand_landmark"]), wrap(models["rtmpose_hand"])


def draw_hands(frame_rgb: np.ndarray, hands: list[dict], key: str = "mp_keypoints") -> np.ndarray:
    out = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR).copy()
    for h in hands:
        if key not in h:
            continue
        right = h["handedness_right"] > 0.5
        color = (60, 200, 255) if right else (255, 160, 60)
        pts = h[key].astype(int)
        cv2.polylines(out, [h["roi_box"][[0, 1, 3, 2]].astype(int)], True, (80, 80, 80), 1)
        for a, b in HAND_CONNECTIONS:
            cv2.line(out, tuple(pts[a]), tuple(pts[b]), color, 2, cv2.LINE_AA)
        for j, p in enumerate(pts):
            cv2.circle(out, tuple(p), 4 if j in (4, 8, 12, 16, 20) else 3, (255, 255, 255), -1, cv2.LINE_AA)
        label = f"{'Right' if right else 'Left'} {h['hand_flag']:.2f}"
        cv2.putText(out, label, tuple(pts[0] + [5, 15]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out
