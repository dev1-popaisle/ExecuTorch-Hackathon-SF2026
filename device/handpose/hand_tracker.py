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
"""Live two-hand tracker on the VENTUNO Q NPU (numpy + OpenCV only, no torch).

MediaPipe-style loop:
  * palm detector (QNN) finds new hands; it only runs while fewer than MAX_HANDS hands are tracked
    (plus a periodic re-check),
  * the landmark model (QNN) runs on a rotated crop per hand; the next frame's crop comes from the current
    landmarks, so tracking stays stable and the detector stays mostly idle,
  * a One Euro filter smooths the 21 keypoints per hand.

Each model runs in a persistent qnn_executor_runner (`--input_list_path=-`), which keeps the QNN context
loaded and reads one frame request per stdin line.
"""

from __future__ import annotations

import math
import os
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PTE_DIR = ROOT / "export" / "pte"
ANCHORS = ROOT / "third_party" / "MediaPipePyTorch" / "anchors_palm.npy"
ET_ROOT = Path("/home/arduino/QualcommDeveloper/executorch")
QNN_RUNNER = ET_ROOT / "examples" / "qualcomm" / "executor_runner" / "qnn_executor_runner"
QNN_LIB_DIR = ET_ROOT / "build-x86" / "backends" / "qualcomm"
SHM = Path("/dev/shm/handpose")

RES = 256
MAX_HANDS = 2
DETECT_SCORE = 0.7
NMS_IOU = 0.3
HAND_FLAG_MIN = 0.5
REDETECT_EVERY = 15
# BlazePalm boxes fists and thumbs-up hands (mostly knuckles) too small for the landmark crop; new detections the
# landmark model rejects are retried with larger crops.
RETRY_SCALES = (1.6, 2.5)
# BlazePalm detection -> hand ROI (mediapipe hand_detection_cpu.pbtxt)
PALM_DY, PALM_DSCALE = -0.5, 2.6
# Landmarks -> next-frame ROI (mediapipe hand_landmark_landmarks_to_roi)
ROI_LANDMARKS = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18]
ROI_SCALE, ROI_SHIFT_Y = 2.0, -0.1
# The palm + landmark models (quantized) take the area around an eye for a hand with flag ~0.98, so hand scores
# can't tell them apart. BlazeFace finds faces; a "hand" whose landmarks mostly sit inside a face is dropped.
# A real hand held over the face hides it from BlazeFace, so it isn't filtered.
FACE_PTE = PTE_DIR / "mp_face_detector_qnn_16a8w.pte"
FACE_ANCHORS = ROOT / "third_party" / "MediaPipePyTorch" / "anchors_face_back.npy"
FACE_SCORE = 0.6
FACE_EVERY = 2          # frames between face detections (faces move slowly)
FACE_INSIDE_MAX = 0.6   # max fraction of a hand's landmarks inside a face box
FACE_GROW_X, FACE_GROW_UP = 0.1, 0.35


class PersistentRunner:
    def __init__(self, pte: Path, name: str, output_shapes: list[tuple[int, ...]]):
        self.dir = SHM / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.input = self.dir / "input_0.raw"
        self.shapes = output_shapes
        env = dict(os.environ,
                   LD_LIBRARY_PATH=f"{QNN_LIB_DIR}:/usr/lib:{os.environ.get('LD_LIBRARY_PATH', '')}",
                   ADSP_LIBRARY_PATH="/usr/lib/dsp/cdsp;/usr/lib/rfsa/adsp;/dsp")
        self.proc = subprocess.Popen(
            [str(QNN_RUNNER), f"--model_path={pte}", "--input_list_path=-", f"--output_folder_path={self.dir}",
             "--iteration=1", "--warm_up=0", f"--performance_output_path={self.dir / 'perf.txt'}"],
            # The runner logs every frame to stderr; keeping it would fill /dev/shm over time.
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1, env=env)

    def __call__(self, x: np.ndarray) -> list[np.ndarray]:
        np.ascontiguousarray(x, dtype=np.float32).tofile(self.input)
        self.proc.stdin.write(f"{self.input}\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"qnn_executor_runner for {self.dir.name} exited ({self.proc.poll()})")
            if line.startswith("FRAME_READY"):
                break
        return [np.fromfile(self.dir / f"output_0_{i}.raw", dtype=np.float32).reshape(s)
                for i, s in enumerate(self.shapes)]

    def close(self, force: bool = False):
        if self.proc.poll() is None:
            if force:  # a hung runner may not act on SIGTERM
                self.proc.kill()
                return
            self.proc.stdin.close()
            self.proc.terminate()


class OneEuro:
    def __init__(self, min_cutoff=1.0, beta=30.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        if self.x is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-3)
        dx = (x - self.x) / dt
        self.dx += self._alpha(self.d_cutoff, dt) * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        self.x += self._alpha(cutoff, dt) * (x - self.x)
        self.t = t
        return self.x.copy()


class Track:
    _next_id = 0

    def __init__(self):
        self.id = Track._next_id
        Track._next_id += 1
        self.filter = OneEuro()
        self.right_prob = None
        self.roi = None

    def update(self, pts_norm: np.ndarray, right_prob: float, t: float) -> np.ndarray:
        self.right_prob = right_prob if self.right_prob is None else 0.8 * self.right_prob + 0.2 * right_prob
        return self.filter(pts_norm, t)


def resize_pad(img: np.ndarray):
    """Letterbox to RES x RES; returns padded image, scale (orig px per padded px) and (pad_y, pad_x) in orig px."""
    h, w = img.shape[:2]
    if h >= w:
        h1, w1, scale = RES, RES * w // h, w / (RES * w // h)
    else:
        h1, w1, scale = RES * h // w, RES, h / (RES * h // w)
    py, px = RES - h1, RES - w1
    out = np.zeros((RES, RES, 3), np.uint8)
    out[py // 2:py // 2 + h1, px // 2:px // 2 + w1] = cv2.resize(img, (w1, h1), interpolation=cv2.INTER_AREA)
    return out, scale, (py // 2 * scale, px // 2 * scale)


def to_nchw(img_u8: np.ndarray) -> np.ndarray:
    return (img_u8.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]


def weighted_nms(dets: np.ndarray) -> list[np.ndarray]:
    """BlazeFace-style blending NMS. dets: (N, 19) = ymin,xmin,ymax,xmax, 7 keypoints (x,y), score."""
    out = []
    order = np.argsort(-dets[:, -1])
    while len(order):
        boxes = dets[order, :4]
        first = boxes[0]
        iy1, ix1 = np.maximum(first[0], boxes[:, 0]), np.maximum(first[1], boxes[:, 1])
        iy2, ix2 = np.minimum(first[2], boxes[:, 2]), np.minimum(first[3], boxes[:, 3])
        inter = np.clip(iy2 - iy1, 0, None) * np.clip(ix2 - ix1, 0, None)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iou = inter / (area[0] + area - inter + 1e-9)
        group = order[iou > NMS_IOU]
        order = order[iou <= NMS_IOU]
        g = dets[group]
        w = g[:, -1:]
        blended = (g[:, :-1] * w).sum(0) / w.sum()
        out.append(np.concatenate([blended, [g[:, -1].mean()]]))
    return out


def decode_faces(raw_boxes: np.ndarray, raw_scores: np.ndarray, anchors: np.ndarray, scale: float, pad) -> list:
    """BlazeFace (back model) heads -> face boxes (x0, y0, x1, y1) in frame px."""
    scores = 1 / (1 + np.exp(-np.clip(raw_scores[0, :, 0], -100, 100)))
    keep = scores >= FACE_SCORE
    if not keep.any():
        return []
    rb, a = raw_boxes[0][keep], anchors[keep]
    xc, yc = rb[:, 0] / RES * a[:, 2] + a[:, 0], rb[:, 1] / RES * a[:, 3] + a[:, 1]
    w, h = rb[:, 2] / RES * a[:, 2], rb[:, 3] / RES * a[:, 3]
    dets = np.stack([yc - h / 2, xc - w / 2, yc + h / 2, xc + w / 2, scores[keep]], axis=1).astype(np.float32)
    pad_y, pad_x = pad
    faces = []
    for d in weighted_nms(dets):
        y0, x0, y1, x1 = d[:4] * scale * RES - (pad_y, pad_x, pad_y, pad_x)
        faces.append((round(float(x0), 1), round(float(y0), 1), round(float(x1), 1), round(float(y1), 1)))
    return faces


def inside_faces(pts: np.ndarray, faces: list) -> float:
    """Fraction of landmarks (frame px) inside any face box, grown to cover the forehead (BlazeFace boxes run
    brows to chin, and the false skeletons spread from the eye upward)."""
    inside = np.zeros(len(pts), bool)
    for x0, y0, x1, y1 in faces:
        w, h = x1 - x0, y1 - y0
        x0, x1, y0, y1 = x0 - FACE_GROW_X * w, x1 + FACE_GROW_X * w, y0 - FACE_GROW_UP * h, y1 + FACE_GROW_X * h
        inside |= (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
    return float(inside.mean())


def roi_corners(roi):
    xc, yc, scale, theta = roi
    c, s = math.cos(theta), math.sin(theta)
    local = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], np.float32) * scale / 2
    return local @ np.array([[c, s], [-s, c]], np.float32) + (xc, yc)


def roi_from_landmarks(pts: np.ndarray):
    """pts: (21, 2) frame px -> (xc, yc, scale, theta) for the next frame's crop."""
    theta = math.atan2(pts[0, 1] - pts[9, 1], pts[0, 0] - pts[9, 0]) - math.pi / 2
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([[c, -s], [s, c]], np.float32)
    sel = pts[ROI_LANDMARKS]
    c0 = sel.mean(0)
    local = (sel - c0) @ rot
    lo, hi = local.min(0), local.max(0)
    size = float((hi - lo).max()) * ROI_SCALE
    center_local = (lo + hi) / 2 + (0.0, ROI_SHIFT_Y * size)
    xc, yc = c0 + center_local @ rot.T
    return float(xc), float(yc), size, theta


def same_hand(a, b) -> bool:
    return math.hypot(a[0] - b[0], a[1] - b[1]) < 0.35 * max(a[2], b[2])


class HandTracker:
    def __init__(self, pte_dir: Path = PTE_DIR, mirrored_input: bool = False):
        self.palm = PersistentRunner(pte_dir / "mp_palm_detector_qnn_16a8w.pte", "palm", [(1, 2944, 18), (1, 2944, 1)])
        self.landmark = PersistentRunner(pte_dir / "mp_hand_landmark_qnn_16a8w.pte", "landmark", [(1,), (1,), (1, 21, 3)])
        self.anchors = np.load(ANCHORS).astype(np.float32)
        self.face = None
        if FACE_PTE.exists():
            self.face = PersistentRunner(FACE_PTE, "face", [(1, 896, 16), (1, 896, 1)])
            self.face_anchors = np.load(FACE_ANCHORS).astype(np.float32)
        # MediaPipe's handedness assumes a mirrored (selfie) image; the webcam feed is not mirrored.
        self.swap_handedness = not mirrored_input
        self.tracks: list[Track] = []
        self.faces: list = []
        self.frame_idx = 0

    def reset(self):
        """Forgets tracked hands; the next frame runs the palm detector from scratch."""
        self.tracks = []
        self.faces = []
        self.frame_idx = 0

    def close(self, force: bool = False):
        self.palm.close(force)
        self.landmark.close(force)
        if self.face:
            self.face.close(force)

    def detect_faces(self, padded) -> list:
        img, scale, pad = padded
        raw_boxes, raw_scores = self.face(to_nchw(img))
        return decode_faces(raw_boxes, raw_scores, self.face_anchors, scale, pad)

    def detect(self, padded):
        img, scale, (pad_y, pad_x) = padded
        raw_boxes, raw_scores = self.palm(to_nchw(img))
        raw_boxes, a = raw_boxes[0], self.anchors
        scores = 1 / (1 + np.exp(-np.clip(raw_scores[0, :, 0], -80, 80)))
        keep = scores >= DETECT_SCORE
        if not keep.any():
            return []
        rb, a, scores = raw_boxes[keep], a[keep], scores[keep]
        xs = rb[:, 0::2] / RES * a[:, 2:3] + a[:, 0:1]  # box center x, then 7 keypoint x
        ys = rb[:, 1::2] / RES * a[:, 3:4] + a[:, 1:2]
        w, h = rb[:, 2] / RES * a[:, 2], rb[:, 3] / RES * a[:, 3]
        dets = np.zeros((len(rb), 19), np.float32)
        dets[:, 0], dets[:, 1] = ys[:, 0] - h / 2, xs[:, 0] - w / 2
        dets[:, 2], dets[:, 3] = ys[:, 0] + h / 2, xs[:, 0] + w / 2
        dets[:, 4:18:2], dets[:, 5:18:2] = xs[:, 2:], ys[:, 2:]
        dets[:, 18] = scores
        rois = []
        for d in weighted_nms(dets)[:MAX_HANDS]:
            ymin, xmin, ymax, xmax = (d[[0, 1, 2, 3]] * scale * RES) - (pad_y, pad_x, pad_y, pad_x)
            kx, ky = d[4:18:2] * scale * RES - pad_x, d[5:18:2] * scale * RES - pad_y
            box = xmax - xmin
            theta = math.atan2(ky[0] - ky[2], kx[0] - kx[2]) - math.pi / 2
            rois.append(((xmin + xmax) / 2, (ymin + ymax) / 2 + PALM_DY * box, box * PALM_DSCALE, theta))
        return rois

    def landmarks(self, rgb: np.ndarray, roi):
        corners = roi_corners(roi)
        m = cv2.getAffineTransform(corners[:3].astype(np.float32),
                                   np.array([[0, 0], [0, RES - 1], [RES - 1, 0]], np.float32))
        crop = cv2.warpAffine(rgb, m, (RES, RES), flags=cv2.INTER_LINEAR)
        flag, handed, lms = self.landmark(to_nchw(crop))
        inv = cv2.invertAffineTransform(m)
        pts = lms[0, :, :2] * RES @ inv[:, :2].T + inv[:, 2]
        return float(flag[0]), float(handed[0]), pts, lms[0, :, 2] * roi[2]

    def process(self, rgb: np.ndarray) -> dict:
        t0 = time.perf_counter()
        h, w = rgb.shape[:2]
        candidates = [(t.roi, t) for t in self.tracks]
        ms_detect = ms_face = 0.0
        padded = None
        if len(self.tracks) < MAX_HANDS or self.frame_idx % REDETECT_EVERY == 0:
            td = time.perf_counter()
            padded = resize_pad(rgb)
            for roi in self.detect(padded):
                if not any(same_hand(roi, r) for r, _ in candidates):
                    candidates.append((roi, None))
            ms_detect = (time.perf_counter() - td) * 1000
        new_candidate = any(t is None for _, t in candidates)
        if self.face and candidates and (new_candidate or self.frame_idx % FACE_EVERY == 0):
            tf = time.perf_counter()
            self.faces = self.detect_faces(padded or resize_pad(rgb))
            ms_face = (time.perf_counter() - tf) * 1000
        elif not candidates:
            self.faces = []

        now = time.monotonic()
        tl = time.perf_counter()
        tracks, hands = [], []
        for roi, track in candidates:
            flag, right, pts, depth = self.landmarks(rgb, roi)
            for k in RETRY_SCALES if track is None else ():
                if flag >= HAND_FLAG_MIN:
                    break
                flag, right, pts, depth = self.landmarks(rgb, (roi[0], roi[1], roi[2] * k, roi[3]))
            if flag < HAND_FLAG_MIN:
                continue
            if self.faces and inside_faces(pts, self.faces) > FACE_INSIDE_MAX:
                continue
            next_roi = roi_from_landmarks(pts)
            if any(same_hand(next_roi, t.roi) for t in tracks):
                continue
            track = track or Track()
            smooth = track.update(pts / (w, h), right, now)
            track.roi = next_roi
            tracks.append(track)
            p_right = 1 - track.right_prob if self.swap_handedness else track.right_prob
            hands.append({"id": track.id, "label": "Right" if p_right > 0.5 else "Left",
                          "handedness": round(max(p_right, 1 - p_right), 3), "score": round(flag, 3),
                          "pts": np.round(smooth, 4).tolist(), "depth": np.round(depth / w, 4).tolist()})
        self.tracks = tracks[:MAX_HANDS]
        self.frame_idx += 1
        return {"w": w, "h": h, "hands": hands[:MAX_HANDS], "faces": self.faces,
                "ms": {"detect": round(ms_detect, 2), "face": round(ms_face, 2),
                       "landmarks": round((time.perf_counter() - tl) * 1000, 2),
                       "total": round((time.perf_counter() - t0) * 1000, 2)}}
