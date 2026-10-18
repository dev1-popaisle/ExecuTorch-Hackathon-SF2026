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
"""Export MediaPipe BlazeFace (back-camera / longer-range model, 256x256) to QNN HTP V75 for the hand tracker's
face filter, calibrated on frames from the live camera, and check it against PyTorch.

    python export_face.py      # camera_stream.py must be serving 8090 (calibration frames)
"""

from __future__ import annotations

import time
import urllib.request

import cv2
import numpy as np
import torch

from export_lib import delegation_summary, export_qnn, run_qnn
from hand_models import MEDIAPIPE_DIR, ROOT
from hand_tracker import FACE_PTE, decode_faces, resize_pad, to_nchw

from blazeface import BlazeFace  # noqa: E402  (MEDIAPIPE_DIR is on sys.path via hand_models)


class FaceDetector(torch.nn.Module):
    """BlazeFace raw heads: boxes + 6 keypoints (1, 896, 16), score logits (1, 896, 1)."""

    def __init__(self, net: BlazeFace):
        super().__init__()
        self.net = net

    def forward(self, image):
        r1, r2, c1, c2 = self.net(image)
        return torch.cat((r1, r2), dim=1), torch.cat((c1, c2), dim=1)


def load_face() -> FaceDetector:
    net = BlazeFace(back_model=True)
    net.load_state_dict(torch.load(MEDIAPIPE_DIR / "blazefaceback.pth", map_location="cpu"))
    return FaceDetector(net.eval()).eval()


def camera_frames(n: int, every: int = 4, url: str = "http://127.0.0.1:8090/stream.mjpg") -> list[np.ndarray]:
    out, buf, i = [], b"", 0
    with urllib.request.urlopen(url, timeout=5) as r:
        while len(out) < n:
            buf += r.read(65536)
            a, b = buf.find(b"\xff\xd8"), buf.find(b"\xff\xd9")
            if a != -1 and b > a:
                jpg, buf = buf[a:b + 2], buf[b + 2:]
                i += 1
                if i % every == 0:
                    out.append(cv2.cvtColor(cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR),
                                            cv2.COLOR_BGR2RGB))
    return out


def main():
    model = load_face()
    frames = camera_frames(48)
    # Mirrored and darker / brighter copies widen the calibration range beyond this one scene.
    variants = frames + [f[:, ::-1].copy() for f in frames[::2]] + \
        [np.clip(f.astype(np.float32) * g, 0, 255).astype(np.uint8) for f in frames[::3] for g in (0.5, 1.4)]
    calib = [torch.from_numpy(np.ascontiguousarray(to_nchw(resize_pad(f)[0]))) for f in variants]
    t0 = time.time()
    export_qnn(model, (torch.rand(1, 3, 256, 256),), calib, FACE_PTE, quant="16a8w")
    print(f"[export] {FACE_PTE.name} {FACE_PTE.stat().st_size / 1e6:.2f} MB in {time.time() - t0:.0f}s "
          f"delegation={delegation_summary(FACE_PTE)}")

    anchors = np.load(MEDIAPIPE_DIR / "anchors_face_back.npy").astype(np.float32)
    for f in frames[::12]:
        img, scale, pad = resize_pad(f)
        x = to_nchw(img)
        with torch.no_grad():
            tb, ts = (t.numpy() for t in model(torch.from_numpy(x)))
        outs, ms = run_qnn(FACE_PTE, [x], iterations=20)
        qb, qs = outs[0].reshape(1, 896, 16), outs[1].reshape(1, 896, 1)
        print(f"latency {ms:.2f} ms  torch faces {decode_faces(tb, ts, anchors, scale, pad)}  "
              f"qnn faces {decode_faces(qb, qs, anchors, scale, pad)}")
    cv2.imwrite(str(ROOT / "export" / "step2" / "face_calib_sample.jpg"), cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
