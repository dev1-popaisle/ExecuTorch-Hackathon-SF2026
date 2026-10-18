"""Gesture classifier checks.

  python test_gestures.py img.jpg [img2.jpg:rot180 ...] [--dump]
      NPU hand tracker + classifier on still images. Each image is fed several times (detect, then track) like
      consecutive video frames; ":rot180" tests the upside-down image. Uses its own /dev/shm folder, so it can run
      next to the live camera_stream server.
  python test_gestures.py --live [seconds]
      Prints the live server's per-frame gesture features whenever they change (for tuning thresholds).
  python test_gestures.py --rules
      Classifier + 3 s state machine on fixed landmark sets (thumbs up read off two live screenshots, and
      negatives); exits non-zero on a wrong answer.
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

import cv2

import hand_tracker
from gestures import GestureState, classify

hand_tracker.SHM = Path("/dev/shm/handpose_test")


def fmt(name: str, d: dict) -> str:
    curled = "".join("x" if v else "." for v in d["curled"].values())
    return (f"[{name}] straight={d['thumb_straight']!s:5s} long={d['thumb_long']!s:5s} angle={d['angle']:5.1f} "
            f"beyond={d['tip_beyond']!s:5s} margin={d['margin']:5.2f} beyond_fist={d['beyond_fist']:5.2f} "
            f"len={d['thumb_len']:4.2f} curled(i,m,r,p)={curled} extended={','.join(d['extended']) or '-'}")


def live(seconds: float):
    t0, last, d = time.time(), None, {}
    while time.time() - t0 < seconds:
        d = json.load(urllib.request.urlopen("http://127.0.0.1:8090/hands"))
        for hand in d["hands"]:
            g, det = classify(hand, d["w"], d["h"])
            # show the thumb test (up or down) the hand is closer to
            name = min(("thumbs_up", "thumbs_down"), key=lambda k: det[k]["angle"])
            line = f"{hand['label']:5s} frame={g:11s} state={d['gesture']['gesture']:11s} {fmt(name, det[name])}"
            if line != last:
                print(f"{time.time() - t0:5.1f}s {line}", flush=True)
                last = line
        time.sleep(0.1)
    print("gestures triggered so far:", d.get("gesture", {}).get("count"))


# Pixel landmarks read off live tracker screenshots (1024 px wide) and hand-made negatives.
THUMBS_UP_1 = [(370, 460), (400, 372), (468, 296), (525, 236), (540, 178), (520, 312), (618, 358), (580, 373),
               (547, 357), (522, 370), (615, 418), (578, 424), (545, 410), (521, 435), (603, 467), (567, 472),
               (538, 463), (513, 500), (585, 512), (555, 513), (524, 506)]
THUMBS_UP_2 = [(343, 345), (390, 260), (471, 197), (553, 149), (592, 87), (558, 216), (636, 290), (595, 302),
               (565, 285), (565, 275), (621, 355), (575, 361), (547, 343), (547, 340), (604, 410), (562, 412),
               (535, 395), (527, 415), (580, 460), (544, 460), (512, 446)]
OPEN_HAND = [(500, 500), (440, 470), (400, 430), (370, 390), (345, 355), (450, 380), (440, 310), (435, 270),
             (432, 235), (500, 370), (500, 295), (500, 250), (500, 212), (545, 380), (552, 310), (556, 268),
             (559, 232), (585, 400), (600, 345), (608, 312), (614, 282)]
FIST_THUMB_TUCKED = [(500, 500), (450, 460), (430, 420), (450, 385), (490, 370), (460, 360), (470, 300), (500, 330),
                     (495, 365), (500, 355), (510, 295), (535, 325), (530, 360), (540, 360), (550, 305), (565, 335),
                     (560, 365), (575, 375), (585, 325), (595, 350), (590, 375)]
# Thumbs down read off live screenshots: sideways fist with the thumb hanging down-right, and the same in front of a face.
THUMBS_DOWN_1 = [(438, 183), (467, 262), (533, 327), (587, 380), (613, 438), (546, 260), (637, 273), (580, 258),
                 (575, 277), (537, 215), (641, 225), (600, 225), (569, 213), (532, 180), (634, 171), (578, 172),
                 (565, 165), (543, 133), (617, 127), (575, 128), (568, 125)]
THUMBS_DOWN_2 = [(460, 269), (495, 347), (564, 410), (623, 467), (662, 535), (593, 345), (692, 341), (625, 332),
                 (619, 358), (571, 308), (686, 291), (598, 291), (612, 301), (564, 269), (671, 236), (606, 248),
                 (590, 247), (573, 215), (645, 192), (599, 203), (590, 200)]


def with_index(pts, index):
    return pts[:5] + index + pts[9:]


def rules() -> int:
    w, h = 1024, 600
    flip = lambda pts: [(x, h - y) for x, y in pts]  # noqa: E731
    sideways = lambda pts: [(500 - (y - 300), 300 + (x - 500)) for x, y in pts]  # noqa: E731  rotate 90°
    as_hand = lambda pts: {"pts": [(x / w, y / h) for x, y in pts], "depth": [0.0] * 21, "label": "Left"}  # noqa: E731
    loose = with_index(THUMBS_UP_1, [(520, 312), (590, 320), (630, 345), (650, 375)])  # index half open
    gun = with_index(THUMBS_UP_1, [(520, 312), (600, 310), (650, 308), (700, 306)])    # index pointing out
    cases = [("thumbs up 1", THUMBS_UP_1, "thumbs_up"), ("thumbs up 2", THUMBS_UP_2, "thumbs_up"),
             ("thumbs up, loose fist", loose, "thumbs_up"),
             ("thumbs down 1", THUMBS_DOWN_1, "thumbs_down"), ("thumbs down 2", THUMBS_DOWN_2, "thumbs_down"),
             ("up 1 upside down", flip(THUMBS_UP_1), "thumbs_down"),
             ("up 2 upside down", flip(THUMBS_UP_2), "thumbs_down"),
             ("down 1 upside down", flip(THUMBS_DOWN_1), "thumbs_up"),
             ("thumbs down, loose", flip(loose), "thumbs_down"),
             ("thumb sideways", sideways(THUMBS_UP_1), "none"),
             ("index pointing (gun)", gun, "none"), ("gun upside down", flip(gun), "none"),
             ("open hand", OPEN_HAND, "none"), ("fist, thumb tucked", FIST_THUMB_TUCKED, "none")]
    bad = 0
    for name, pts, want in cases:
        got, details = classify(as_hand(pts), w, h)
        bad += got != want
        key = got if got != "none" else min(("thumbs_up", "thumbs_down"), key=lambda k: details[k]["angle"])
        print(f"{'ok ' if got == want else 'BAD'} {name:22s} -> {got:11s} {fmt(key, details[key])}")

    def timeline_of(script):
        state, t, out = GestureState(), 0.0, []
        for seconds, hands in script:
            for _ in range(int(seconds * 30)):
                g = state.update(hands, w, h, now=t)["gesture"]
                if not out or out[-1][1] != g:
                    out.append((round(t, 2), g))
                t += 1 / 30
        return out

    up, down = [as_hand(THUMBS_UP_1)], [as_hand(THUMBS_DOWN_1)]
    # held thumbs up: 3 s then reset, no repeat; released and shown again: triggers again
    tl = timeline_of(((5, up), (1, []), (1, up), (4, [])))
    ok = [g for _, g in tl] == ["none", "thumbs_up", "none", "thumbs_up", "none"] and abs(tl[2][0] - tl[1][0] - 3) < .05
    bad += not ok
    print(f"{'ok ' if ok else 'BAD'} hold/reset timeline: {tl}")
    # thumbs up -> thumbs down after 1 s switches right away (not after 3 s), then down holds its own 3 s
    tl = timeline_of(((1, up), (1, down), (1, up), (5, [])))
    ok = ([g for _, g in tl] == ["none", "thumbs_up", "thumbs_down", "thumbs_up", "none"]
          and tl[2][0] - 1.0 < 0.3 and tl[3][0] - 2.0 < 0.3 and abs(tl[4][0] - tl[3][0] - 3) < .05)
    bad += not ok
    print(f"{'ok ' if ok else 'BAD'} up -> down -> up switch timeline: {tl}")
    return bad


def fusion() -> int:
    """Voice "light" + thumbs up / down within 3 s on a real HomeState (temporary state file, real timers)."""
    import tempfile

    from camera_stream import HomeState

    bad = 0

    def check(name, cond, detail=""):
        nonlocal bad
        bad += not cond
        print(f"{'ok ' if cond else 'BAD'} {name} {detail}")

    home = HomeState(Path(tempfile.mkdtemp()) / "state.json")
    held = {"g": None}
    home.gesture_held = lambda: held["g"]
    gesture = lambda g: home.set_gesture({"gesture": g, "hand": "Left", "since": 0, "until": 0, "count": 1})  # noqa: E731
    lights = lambda: home.snapshot()["lighting"]["on"]  # noqa: E731

    home.update({"lighting": {"on": False}})
    r = home.voice_intent("lighting", "Hey Arduino, light.")
    time.sleep(1.0)
    gesture("thumbs_up")
    check("voice, then 👍 1 s later -> on", r["status"] == "waiting" and lights() is True, home.snapshot()["voice"]["result"])

    time.sleep(3.5)  # old gesture/intent out of the window
    gesture("thumbs_down")
    time.sleep(1.5)
    r = home.voice_intent("lighting", "Hey Arduino, light.")
    check("👎, then voice 1.5 s later -> off", r["status"] == "done" and lights() is False, r["result"])

    time.sleep(3.5)
    r = home.voice_intent("lighting", "Hey Arduino, light.")
    time.sleep(3.3)
    v = home.snapshot()["voice"]
    check("voice, no gesture -> expires, light unchanged", lights() is False and v["status"] == "failed", v["result"])

    gesture("thumbs_up")
    time.sleep(3.2)
    r = home.voice_intent("lighting", "Hey Arduino, light.")
    check("👍 3.2 s before voice -> not paired", r["status"] == "waiting" and lights() is False)
    time.sleep(3.3)

    held["g"] = "thumbs_up"
    r = home.voice_intent("lighting", "Hey Arduino, light.")
    check("👍 held since long ago + voice -> on", r["status"] == "done" and lights() is True, r["result"])
    held["g"] = None

    time.sleep(0.2)
    r = home.voice_intent("lighting", "Hey Arduino, light.")
    time.sleep(0.5)
    gesture("thumbs_down")
    check("voice, then 👎 0.5 s later -> off", lights() is False and home.snapshot()["change"]["source"] == "voice",
          "(last change: voice status)")
    check("light change came from voice+gesture", home.snapshot()["voice"]["result"].startswith("Lights off"),
          home.snapshot()["voice"]["result"])
    return bad


def main():
    if "--rules" in sys.argv:
        sys.exit(1 if rules() else 0)
    if "--fusion" in sys.argv:
        sys.exit(1 if fusion() else 0)
    if "--live" in sys.argv:
        rest = [a for a in sys.argv[sys.argv.index("--live") + 1:] if not a.startswith("--")]
        return live(float(rest[0]) if rest else 30)
    tracker = hand_tracker.HandTracker()
    try:
        for arg in [a for a in sys.argv[1:] if not a.startswith("--")]:
            path, _, mode = arg.partition(":")
            bgr = cv2.imread(path)
            if mode == "rot180":
                bgr = cv2.rotate(bgr, cv2.ROTATE_180)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tracker.reset()
            state = GestureState()
            for i in range(12):
                res = tracker.process(rgb)
                out = state.update(res["hands"], res["w"], res["h"], now=i / 30)
            print(f"\n{Path(path).name}{' (rotated 180)' if mode else ''}: {len(res['hands'])} hand(s), "
                  f"state={out['gesture']}")
            for hand in res["hands"]:
                g, details = classify(hand, res["w"], res["h"])
                print(f"  {hand['label']} score={hand['score']}: frame gesture={g}")
                for name in ("thumbs_up", "thumbs_down"):
                    print("   ", fmt(name, details[name]))
                if "--dump" in sys.argv:
                    print("    pts", json.dumps(hand["pts"]), "\n    depth", json.dumps(hand["depth"]))
    finally:
        tracker.close()


if __name__ == "__main__":
    main()
