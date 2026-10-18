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
"""Hand gestures from the NPU hand landmarks (21 MediaPipe keypoints per hand).

Per frame, each gesture is a geometric test on the 3D landmarks (x, y in pixels, z from the landmark model scaled
like x), so it doesn't depend on hand size or distance. GestureState turns noisy per-frame hits into a stable
state: a gesture must hold for a few frames to trigger, then the state stays set for HOLD_S seconds and resets to
"none". The same gesture re-triggers only after it was released, so holding a thumbs up doesn't repeat; a different
gesture (thumbs up -> thumbs down) takes over immediately.

Landmarks: 0 wrist; thumb 1-4; index 5-8; middle 9-12; ring 13-16; pinky 17-20 (MCP, PIP, DIP, TIP).
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque

import numpy as np

WRIST = 0
THUMB = (1, 2, 3, 4)  # CMC, MCP, IP, TIP
FINGERS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}

HOLD_S = 3.0           # how long a triggered gesture stays the state
TRIGGER_WINDOW = 10    # frames considered for triggering (~0.33 s at 30 fps)
TRIGGER_HITS = 5       # ...of which this many must match (fist landmarks are noisy frame to frame)
RELEASE_S = 0.4        # gesture must be absent this long before it can trigger again
NONE = "none"


def hand_xyz(hand: dict, w: int, h: int) -> np.ndarray:
    pts = np.asarray(hand["pts"], np.float32) * (w, h)
    z = np.asarray(hand.get("depth") or [0.0] * 21, np.float32)[:, None] * w
    return np.hstack([pts, z])


def bend(p: np.ndarray, a: int, b: int, c: int) -> float:
    """Bend at joint b in degrees: 0 = straight, 90 = right angle."""
    u, v = p[b] - p[a], p[c] - p[b]
    cos = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-6))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def dist(p: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(p[a] - p[b]))


def finger_curled(p: np.ndarray, chain: tuple[int, int, int, int]) -> bool:
    mcp, pip, dip, tip = chain
    folded = bend(p, mcp, pip, dip) + bend(p, pip, dip, tip) > 100
    tip_in = dist(p, tip, WRIST) < dist(p, pip, WRIST)  # the tip comes back toward the palm
    return folded or tip_in


def finger_extended(p: np.ndarray, chain: tuple[int, int, int, int], palm: float) -> bool:
    """Clearly pointing out (straight and long) - unlike a loosely folded finger in a relaxed fist."""
    mcp, pip, dip, tip = chain
    straight = bend(p, mcp, pip, dip) + bend(p, pip, dip, tip) < 40
    return straight and dist(p, mcp, tip) > 0.6 * palm and dist(p, tip, WRIST) > dist(p, pip, WRIST)


def thumb_vertical(p: np.ndarray, direction: int) -> tuple[bool, dict]:
    """Thumbs up (direction=-1) / down (+1): a fist (loose or tight, anywhere in the frame) with the thumb straight
    and pointing up / down, its tip beyond the knuckles, the wrist and the middle of the fist."""
    palm = dist(p, WRIST, 9) + 1e-6
    cmc, mcp, ip, tip = THUMB
    thumb_straight = bend(p, mcp, ip, tip) < 45 and bend(p, cmc, mcp, ip) < 55
    thumb_long = dist(p, mcp, tip) > 0.45 * palm
    v = p[tip, :2] - p[mcp, :2]
    # 0 = exactly up (direction -1) or down (+1) in the image (y grows down); a relaxed thumb leans ~50°
    angle = math.degrees(math.atan2(abs(v[0]), direction * v[1]))
    # Folded finger joints inside a fist are guessed loosely by the landmark model (a pinky joint can land past the
    # thumb), so "beyond" uses the reliable knuckles + wrist and the middle of the fist.
    base = [c[0] for c in FINGERS.values()] + [WRIST]
    ys = direction * p[base, 1]  # larger = further in the thumb's direction
    fist = direction * p[[i for c in FINGERS.values() for i in c], 1].mean()
    tip_y = direction * p[tip, 1]
    tip_beyond = tip_y > ys.max() and tip_y - fist > 0.3 * palm
    curled = {name: finger_curled(p, chain) for name, chain in FINGERS.items()}
    extended = [name for name, chain in FINGERS.items() if finger_extended(p, chain, palm)]
    # Loose fists: one finger may read as not folded, but none may point out (that's a different gesture).
    fist_ok = sum(curled.values()) >= 3 and not extended
    ok = thumb_straight and thumb_long and angle < 60 and tip_beyond and fist_ok
    return ok, {"thumb_straight": thumb_straight, "thumb_long": thumb_long, "angle": round(angle, 1),
                "tip_beyond": bool(tip_beyond), "curled": curled, "extended": extended,
                "thumb_len": round(dist(p, mcp, tip) / palm, 2),
                "margin": round(float(tip_y - ys.max()) / palm, 2),
                "beyond_fist": round(float(tip_y - fist) / palm, 2)}


def thumbs_up(p: np.ndarray) -> tuple[bool, dict]:
    return thumb_vertical(p, -1)


def thumbs_down(p: np.ndarray) -> tuple[bool, dict]:
    return thumb_vertical(p, +1)


CLASSIFIERS = {"thumbs_up": thumbs_up, "thumbs_down": thumbs_down}


def classify(hand: dict, w: int, h: int) -> tuple[str, dict]:
    """Per-frame gesture of one hand ("none" if no classifier matches) and the features behind it."""
    p = hand_xyz(hand, w, h)
    details = {}
    for name, fn in CLASSIFIERS.items():
        ok, details[name] = fn(p)
        if ok:
            return name, details
    return NONE, details


class GestureState:
    """Debounced, timed gesture state shared by all hands. Listeners get the state dict on every change."""

    def __init__(self, hold_s: float = HOLD_S):
        self.hold_s = hold_s
        self.lock = threading.Lock()
        self.listeners: list = []
        self.reset()

    def reset(self):
        with self.lock:
            self.recent: deque[str] = deque(maxlen=TRIGGER_WINDOW)
            self.last_seen: dict[str, float] = {}  # gesture -> last frame time it was detected
            self.armed: dict[str, bool] = {}
            old = getattr(self, "state", {})
            self.state = {"gesture": NONE, "hand": None, "since": time.time(), "until": 0.0,
                          "count": old.get("count", 0)}
            out = self.snapshot()
        if old.get("gesture", NONE) != NONE:
            for fn in self.listeners:
                fn(out)

    def held(self) -> str | None:
        """The gesture held steadily in the last frames (even if it triggered long ago), or None."""
        with self.lock:
            for g in CLASSIFIERS:
                if sum(1 for x in self.recent if x == g) >= TRIGGER_HITS:
                    return g
        return None

    def snapshot(self, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        s = dict(self.state)
        s["remaining"] = round(max(0.0, s["until"] - now), 2) if s["gesture"] != NONE else 0.0
        return s

    def update(self, hands: list[dict], w: int, h: int, now: float | None = None) -> dict:
        """Feed one frame of tracked hands; returns the current state plus per-hand frame gestures."""
        now = time.time() if now is None else now
        frame = [(hand, *classify(hand, w, h)) for hand in hands]
        seen = next(((hand, g) for hand, g, _ in frame if g != NONE), (None, NONE))
        changed = False
        with self.lock:
            self.recent.append(seen[1])
            for g in CLASSIFIERS:
                if g == seen[1]:
                    self.last_seen[g] = now
                elif now - self.last_seen.get(g, 0.0) > RELEASE_S:
                    self.armed[g] = True
            if self.state["gesture"] != NONE and now >= self.state["until"]:
                self.state.update(gesture=NONE, hand=None, since=now, until=0.0)
                changed = True
            g = seen[1]
            # A different gesture replaces the current one right away (thumbs up -> down) with a fresh hold.
            if (g != NONE and g != self.state["gesture"] and self.armed.get(g, True)
                    and sum(1 for x in self.recent if x == g) >= TRIGGER_HITS):
                self.armed[g] = False
                self.state.update(gesture=g, hand=seen[0].get("label"), since=now, until=now + self.hold_s,
                                  count=self.state["count"] + 1)
                changed = True
            out = self.snapshot(now)
        if changed:
            for fn in self.listeners:
                fn(out)
        out["frame"] = [{"id": hand.get("id"), "gesture": g} for hand, g, _ in frame]
        return out
