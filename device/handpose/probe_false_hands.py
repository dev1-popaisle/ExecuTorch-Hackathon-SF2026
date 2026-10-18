"""Records what the tracker accepts on the live camera feed (palm score, hand flag, handedness, retry scale, track
age) and saves annotated frames, to tell real hands from faces. Run while camera_stream.py is serving 8090.

    python probe_false_hands.py [seconds] [outdir]
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

import hand_tracker as ht
from hand_tracker import SHM, HandTracker, same_hand

SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 60
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/probe")
OUT.mkdir(parents=True, exist_ok=True)


def frames(url="http://127.0.0.1:8090/stream.mjpg"):
    buf = b""
    with urllib.request.urlopen(url, timeout=5) as r:
        while True:
            buf += r.read(65536)
            a, b = buf.find(b"\xff\xd8"), buf.find(b"\xff\xd9")
            if a != -1 and b != -1 and b > a:
                jpg, buf = buf[a:b + 2], buf[b + 2:]
                img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    yield img


class Probe(HandTracker):
    """HandTracker.process with every candidate's scores recorded (accepted or not)."""

    def process_debug(self, rgb):
        h, w = rgb.shape[:2]
        candidates = [(t.roi, t, None) for t in self.tracks]
        if len(self.tracks) < ht.MAX_HANDS or self.frame_idx % ht.REDETECT_EVERY == 0:
            for roi, score in self.detect_scored(rgb):
                if not any(same_hand(roi, r) for r, _, _ in candidates):
                    candidates.append((roi, None, score))
        out, tracks = [], []
        for roi, track, palm in candidates:
            flag, right, pts, _ = self.landmarks(rgb, roi)
            scale = 1.0
            for k in ht.RETRY_SCALES if track is None else ():
                if flag >= ht.HAND_FLAG_MIN:
                    break
                scale = k
                flag, right, pts, _ = self.landmarks(rgb, (roi[0], roi[1], roi[2] * k, roi[3]))
            rec = {"palm": None if palm is None else round(palm, 3), "flag": round(flag, 3),
                   "handed": round(max(right, 1 - right), 3), "retry": scale, "tracked": track is not None,
                   "age": getattr(track, "age", 0), "size": round(roi[2] / w, 3), "pts": pts.tolist()}
            if flag >= ht.HAND_FLAG_MIN:
                next_roi = ht.roi_from_landmarks(pts)
                if not any(same_hand(next_roi, t.roi) for t in tracks):
                    track = track or ht.Track()
                    track.age = getattr(track, "age", 0) + 1
                    track.update(pts / (w, h), right, time.monotonic())
                    track.roi = next_roi
                    tracks.append(track)
                    rec["accepted"] = True
            out.append(rec)
        self.tracks = tracks[:ht.MAX_HANDS]
        self.frame_idx += 1
        return out

    def detect_scored(self, rgb):
        img, scale, (pad_y, pad_x) = ht.resize_pad(rgb)
        raw_boxes, raw_scores = self.palm(ht.to_nchw(img))
        scores = 1 / (1 + np.exp(-np.clip(raw_scores[0, :, 0], -80, 80)))
        keep = scores >= 0.5  # below the live threshold too, to see the margin
        if not keep.any():
            return []
        rb, a, sc = raw_boxes[0][keep], self.anchors[keep], scores[keep]
        xs = rb[:, 0::2] / ht.RES * a[:, 2:3] + a[:, 0:1]
        ys = rb[:, 1::2] / ht.RES * a[:, 3:4] + a[:, 1:2]
        bw, bh = rb[:, 2] / ht.RES * a[:, 2], rb[:, 3] / ht.RES * a[:, 3]
        dets = np.zeros((len(rb), 19), np.float32)
        dets[:, 0], dets[:, 1] = ys[:, 0] - bh / 2, xs[:, 0] - bw / 2
        dets[:, 2], dets[:, 3] = ys[:, 0] + bh / 2, xs[:, 0] + bw / 2
        dets[:, 4:18:2], dets[:, 5:18:2] = xs[:, 2:], ys[:, 2:]
        dets[:, 18] = sc
        out = []
        for d in ht.weighted_nms(dets)[:ht.MAX_HANDS]:
            ymin, xmin, ymax, xmax = (d[[0, 1, 2, 3]] * scale * ht.RES) - (pad_y, pad_x, pad_y, pad_x)
            kx, ky = d[4:18:2] * scale * ht.RES - pad_x, d[5:18:2] * scale * ht.RES - pad_y
            box = xmax - xmin
            theta = np.arctan2(ky[0] - ky[2], kx[0] - kx[2]) - np.pi / 2
            out.append((((xmin + xmax) / 2, (ymin + ymax) / 2 + ht.PALM_DY * box, box * ht.PALM_DSCALE, float(theta)),
                        float(d[18])))
        return out


ht.SHM = SHM.with_name("handpose_probe")  # the live server's runners use SHM
ht.SHM.mkdir(parents=True, exist_ok=True)
probe = Probe()
log = open(OUT / "log.jsonl", "w")
t0, n, saved = time.time(), 0, 0
try:
    for bgr in frames():
        if time.time() - t0 > SECONDS:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        recs = probe.process_debug(rgb)
        n += 1
        if recs:
            log.write(json.dumps({"frame": n, "t": round(time.time() - t0, 2),
                                  "recs": [{k: v for k, v in r.items() if k != "pts"} for r in recs]}) + "\n")
            if n % 5 == 0 and saved < 60:
                vis = bgr.copy()
                for r in recs:
                    color = (0, 255, 0) if r.get("accepted") else (0, 0, 255)
                    for x, y in np.array(r["pts"]).astype(int):
                        cv2.circle(vis, (x, y), 4, color, -1)
                    x, y = np.array(r["pts"]).min(0).astype(int)
                    cv2.putText(vis, f"palm={r['palm']} flag={r['flag']} hd={r['handed']} k={r['retry']} age={r['age']}",
                                (max(0, x), max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.imwrite(str(OUT / f"f{n:05d}.jpg"), cv2.resize(vis, (640, 360)))
                saved += 1
finally:
    probe.close()
    log.close()
print(f"{n} frames in {time.time() - t0:.0f}s, {saved} saved to {OUT}")
