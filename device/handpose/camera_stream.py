"""Ventuno home server on the VENTUNO Q: Logitech camera MJPEG, NPU hand tracking and the home state (stdlib only;
--hands needs the handpose venv).

GStreamer owns the camera and emits re-encoded JPEGs (the Brio's raw MJPEG lacks Huffman tables, which some
decoders reject). One reader thread keeps the latest frame; every HTTP client gets frames from it.

  GET /stream.mjpg   multipart/x-mixed-replace live stream
  GET /snapshot.jpg  latest frame
  GET /hands         latest hand keypoints (JSON), with --hands
  GET /hands.sse     hand keypoints as server-sent events, with --hands
  GET /api/state     home state (lighting / temperature / tv) + voice assistant status
  GET /api/events    home state pushed as server-sent events on every change
  POST /api/state    partial update, e.g. {"tv": {"channel": 2}, "source": "voice"}; refusals (channel change
                     while the TV is off) come back in change.notice
  POST /api/voice    voice assistant status, e.g. {"status": "listening"}
  GET /health        JSON status
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STREAM_VERSION = "14"
BOUNDARY = b"frame"
STATE_FILE = Path(os.environ.get("VENTUNO_STATE", Path(__file__).resolve().parent.parent / "home_state.json"))
DEFAULT_STATE = {
    "lighting": {"on": False, "brightness": 70},
    "temperature": {"on": True, "current": 70.7, "target": 72.0, "mode": "auto", "unit": "F"},
    "tv": {"on": False, "channel": 1, "volume": 12},
}
# url: Pluto TV web page (/live-tv/<guide id>/); stream: Pluto channel id for the HLS stitcher the UI plays;
# aliases: what can be said instead of the channel number ("go to sci fi").
FUSION_S = 3.0  # voice + gesture must happen within this many seconds of each other
GESTURE_ACTIONS = {"lighting": {"thumbs_up": {"on": True}, "thumbs_down": {"on": False}}}
GESTURE_ICONS = {"thumbs_up": "👍", "thumbs_down": "👎"}
TARGET_NAMES = {"lighting": "Lights", "temperature": "Temperature", "tv": "TV"}
TV_CHANNELS = [
    {"number": 1, "name": "Comedy", "pluto_name": "Comedy Central Pluto TV",
     "url": "https://pluto.tv/us/watch/live-tv/28668/", "stream": "5ca671f215a62078d2ec0abf",
     "aliases": ["comedy", "comedies", "comedy central", "funny"]},
    {"number": 2, "name": "Sci Fi", "pluto_name": "Pluto TV Sci-Fi",
     "url": "https://pluto.tv/us/watch/live-tv/18447/", "stream": "5b4fc274694c027be6ed3eea",
     "aliases": ["sci fi", "scifi", "syfy", "sy fy", "science fiction", "sigh fi", "psy fi", "sky fi", "sci fy"]},
    {"number": 3, "name": "Action", "pluto_name": "Pluto TV Action",
     "url": "https://pluto.tv/us/watch/live-tv/29215/", "stream": "561d7d484dc7c8770484914a",
     "aliases": ["action", "action movies"]},
]


class HomeState:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.RLock()  # on_tv_off -> gesture reset -> set_gesture re-enters it
        self.changed = threading.Condition(self.lock)
        self.seq = 0
        self.last_change = {"sections": [], "source": "", "t": 0.0}
        self.data = json.loads(json.dumps(DEFAULT_STATE))
        self.voice = {"status": "offline", "heard": "", "result": "", "ok": None, "t": 0.0}
        self.on_tv_off = None  # called when the TV switches off (the UI goes back to the camera view)
        self.gesture = {"gesture": "none", "hand": None, "since": 0.0, "until": 0.0, "count": 0}
        self.last_gesture: tuple[str, float] | None = None  # (gesture, time it triggered)
        self.pending: dict | None = None                     # voice intent waiting for a gesture
        self.gesture_held = None                             # () -> gesture held steadily right now, or None
        if path.exists():
            try:
                saved = json.loads(path.read_text())
                temp = saved.get("temperature", {})
                if temp and temp.get("unit") != "F":  # state saved before the switch to Fahrenheit
                    for key in ("current", "target"):
                        if isinstance(temp.get(key), (int, float)):
                            temp[key] = round(temp[key] * 9 / 5 + 32, 1)
                    temp["unit"] = "F"
                for k in self.data:
                    self.data[k].update({kk: v for kk, v in saved.get(k, {}).items() if kk in self.data[k]})
            except (ValueError, OSError):
                pass
        if not 1 <= self.data["tv"]["channel"] <= len(TV_CHANNELS):
            self.data["tv"]["channel"] = 1

    def _snapshot(self) -> dict:
        return json.loads(json.dumps({**self.data, "tv_channels": TV_CHANNELS, "voice": self.voice,
                                      "gesture": self.gesture, "seq": self.seq, "change": self.last_change}))

    def snapshot(self) -> dict:
        with self.lock:
            return self._snapshot()

    def _publish(self, sections: list[str], source: str, notice: dict | None = None):
        self.seq += 1
        self.last_change = {"sections": sections, "source": source, "t": time.time(), "notice": notice or {}}
        self.changed.notify_all()

    def _check_tv(self, tv: dict) -> str | None:
        """Drops a channel change the TV can't make; returns the message to show in the TV room."""
        if "channel" not in tv:
            return None
        ch = tv["channel"]
        if not tv.get("on", self.data["tv"]["on"]):
            del tv["channel"]
            return "TV is off"
        if not isinstance(ch, int) or not 1 <= ch <= len(TV_CHANNELS):
            del tv["channel"]
            return f"No channel {ch} (1–{len(TV_CHANNELS)})"
        return None

    def update(self, patch: dict, source: str = "ui") -> dict:
        with self.lock:
            notice = {}
            if isinstance(patch.get("tv"), dict):
                patch["tv"] = dict(patch["tv"])
                msg = self._check_tv(patch["tv"])
                if msg:
                    notice["tv"] = msg
            tv_was_on = self.data["tv"]["on"]
            changed, touched = [], []
            for section, values in patch.items():
                if section in self.data and isinstance(values, dict):
                    touched.append(section)
                    for k, v in values.items():
                        if k not in self.data[section] or k == "unit":
                            continue
                        cur = self.data[section][k]
                        if type(v) is type(cur) or (type(cur) is float and type(v) is int):
                            new = float(v) if type(cur) is float else v
                            if new != cur and section not in changed:
                                changed.append(section)
                            self.data[section][k] = new
            if changed:
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text(json.dumps(self.data, indent=2))
                tmp.replace(self.path)
            if tv_was_on and not self.data["tv"]["on"] and self.on_tv_off:
                self.on_tv_off()
            if changed or touched:
                # Also publish no-op commands ("turn on the TV" while it's on) so the UI acknowledges them.
                self._publish(touched, source, notice)
            return self._snapshot()

    def set_gesture(self, gesture: dict):
        """Gesture state changes (gestures.GestureState) go out on /api/events like voice and UI changes."""
        with self.lock:
            self.gesture = {k: gesture[k] for k in ("gesture", "hand", "since", "until", "count")}
            self._publish(["gesture"], "gesture")
            if gesture["gesture"] != "none":
                self.last_gesture = (gesture["gesture"], time.time())
                self._fuse()

    # Voice + gesture: "Hey Arduino, light" names the device, a thumbs up / down within FUSION_S (before or after)
    # says on / off. Explicit voice commands ("turn on the light") don't come through here.
    def voice_intent(self, target: str, heard: str) -> dict:
        with self.lock:
            if target not in GESTURE_ACTIONS:
                return {"status": "failed", "result": f"No gesture actions for {target}"}
            now = time.time()
            self.pending = {"target": target, "heard": heard, "t": now, "id": now}
            held = self.gesture_held() if self.gesture_held else None
            if held:  # a thumb held up since before (no fresh trigger) counts as "now"
                self.last_gesture = (held, now)
            done = self._fuse()
            if done:
                return {"status": "done", "result": done}
            threading.Timer(FUSION_S, self._expire_intent, args=(now,)).start()
            result = f"{TARGET_NAMES[target]}: show 👍 for on or 👎 for off"
            self.voice.update(status="waiting", heard=heard, result=result, ok=None, t=now)
            self._publish([], "voice")
            return {"status": "waiting", "result": result}

    def _fuse(self) -> str | None:
        p, g = self.pending, self.last_gesture
        if not p or not g or abs(g[1] - p["t"]) > FUSION_S:
            return None
        values = GESTURE_ACTIONS[p["target"]].get(g[0])
        if values is None:
            return None
        self.pending, self.last_gesture = None, None
        self.update({p["target"]: dict(values)}, source="voice+gesture")
        result = f"{TARGET_NAMES[p['target']]} {'on' if values.get('on') else 'off'} (voice + {GESTURE_ICONS[g[0]]})"
        self.voice.update(status="done", heard=p["heard"], result=result, ok=True, t=time.time())
        self._publish([], "voice")
        return result

    def _expire_intent(self, intent_id: float):
        with self.lock:
            if self.pending and self.pending["id"] == intent_id:
                target = self.pending["target"]
                self.pending = None
                self.voice.update(status="failed", ok=False, t=time.time(),
                                  result=f"{TARGET_NAMES[target]}: no 👍 or 👎 within {FUSION_S:g} s")
                self._publish([], "voice")

    def set_voice(self, status: dict) -> dict:
        with self.lock:
            self.voice.update({k: v for k, v in status.items() if k in self.voice}, t=time.time())
            self._publish([], "voice")
            return self._snapshot()

    def wait(self, last_seq: int, timeout: float = 5.0):
        with self.lock:
            self.changed.wait_for(lambda: self.seq != last_seq, timeout)
            return self.seq, self._snapshot()


class Camera:
    def __init__(self, device: str, width: int, height: int, fps: int, quality: int):
        self.cmd = [
            "gst-launch-1.0", "-q",
            "v4l2src", f"device={device}", "do-timestamp=true", "!",
            f"image/jpeg,width={width},height={height},framerate={fps}/1", "!",
            "jpegdec", "!", "videoconvert", "!", "jpegenc", f"quality={quality}", "!",
            "multipartmux", f"boundary={BOUNDARY.decode()}", "!", "fdsink", "fd=1", "sync=false",
        ]
        self.size = (width, height)
        self.frame: bytes | None = None
        self.frame_t = 0.0
        self.seq = 0
        self.fps = 0.0
        self.error = ""
        self.proc: subprocess.Popen | None = None
        self.cond = threading.Condition()
        threading.Thread(target=self._run, daemon=True).start()

    def ensure_live(self, max_age: float = 2.0):
        """Restarts the capture pipeline if it stopped delivering frames."""
        if self.proc and time.time() - self.frame_t > max_age:
            print("camera stalled, restarting pipeline", flush=True)
            self.proc.kill()

    def _run(self):
        while True:
            proc = self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
            try:
                self._read(proc.stdout)
            except Exception as e:  # noqa: BLE001
                self.error = str(e)
            proc.kill()
            err = proc.stderr.read().decode(errors="replace").strip()
            self.error = err[-300:] or self.error or "camera pipeline exited"
            with self.cond:
                self.frame = None
                self.cond.notify_all()
            time.sleep(2)

    def _read(self, f):
        t0, n = time.time(), 0
        while True:
            line = f.readline()
            if not line:
                return
            if not line.startswith(b"--" + BOUNDARY):
                continue
            length = None
            while True:
                h = f.readline()
                if not h:
                    return
                if h in (b"\r\n", b"\n"):
                    break
                k, _, v = h.decode(errors="replace").partition(":")
                if k.strip().lower() == "content-length":
                    length = int(v.strip())
            if length is None:
                continue
            data = b""
            while len(data) < length:
                chunk = f.read(length - len(data))
                if not chunk:
                    return
                data += chunk
            n += 1
            if time.time() - t0 >= 1.0:
                self.fps, t0, n = n / (time.time() - t0), time.time(), 0
            with self.cond:
                self.frame, self.error, self.frame_t = data, "", time.time()
                self.seq += 1
                self.cond.notify_all()

    def wait_frame(self, last_seq: int, timeout: float = 5.0):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != last_seq or self.frame is None, timeout)
            return self.seq, self.frame


class Hands:
    """Runs the NPU hand tracker on the newest camera frame, dropping frames it can't keep up with."""

    def __init__(self, cam: Camera):
        self.cam = cam
        self.result: dict | None = None
        self.seq = 0
        self.fps = 0.0
        self.error = ""
        self.epoch = 0  # bumped by reset(); results carry it so clients can drop pre-reset hands
        self.reset_pending = threading.Event()
        self.tracker = None
        from gestures import GestureState
        self.gestures = GestureState()
        self.cond = threading.Condition()
        threading.Thread(target=self._run, daemon=True).start()

    def reset(self):
        """Fresh start for gesture tracking (e.g. the TV overlay went away): drop tracked hands, restart a stalled
        camera or NPU tracker."""
        self.epoch += 1
        self.reset_pending.set()
        self.gestures.reset()
        self.cam.ensure_live()
        last = self.result["t"] if self.result else 0.0
        if self.tracker is not None and time.time() - last > 3.0:
            print("hand tracker stalled, restarting it", flush=True)
            self.tracker.close(force=True)  # the blocked NPU call fails and _run starts a new tracker

    def _run(self):
        import cv2
        import numpy as np
        from hand_tracker import HandTracker

        while True:
            tracker = None
            try:
                tracker = self.tracker = HandTracker()
                frame_seq, t0, n = -1, time.time(), 0
                while True:
                    if self.reset_pending.is_set():
                        self.reset_pending.clear()
                        tracker.reset()
                        frame_seq = self.cam.seq  # only frames captured after the reset
                    frame_seq, jpeg = self.cam.wait_frame(frame_seq)
                    if jpeg is None:
                        continue
                    epoch = self.epoch
                    td = time.perf_counter()
                    bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    decode_ms = (time.perf_counter() - td) * 1000
                    res = tracker.process(rgb)
                    res["ms"]["decode"] = round(decode_ms, 2)
                    res["gesture"] = self.gestures.update(res["hands"], res["w"], res["h"])
                    n += 1
                    if time.time() - t0 >= 1.0:
                        self.fps, t0, n = n / (time.time() - t0), time.time(), 0
                    if self.reset_pending.is_set():
                        continue  # a reset arrived mid-frame; this result belongs to the old session
                    res.update(frame=frame_seq, t=time.time(), fps=round(self.fps, 1), epoch=epoch)
                    with self.cond:
                        self.result, self.error = res, ""
                        self.seq += 1
                        self.cond.notify_all()
            except Exception:  # noqa: BLE001
                self.error = traceback.format_exc()[-500:]
                print(self.error, flush=True)
            finally:
                self.tracker = None
                if tracker:
                    tracker.close()
            time.sleep(1)

    def wait(self, last_seq: int, timeout: float = 5.0):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != last_seq, timeout)
            return self.seq, self.result


def make_handler(cam: Camera, hands: Hands | None, home: HomeState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            path = self.path.split("?")[0]
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            except ValueError:
                return self._send(400, "application/json", b'{"error": "bad json"}')
            if path == "/api/state":
                source = body.get("source") if isinstance(body.get("source"), str) else "ui"
                self._send(200, "application/json", json.dumps(home.update(body, source)).encode())
            elif path == "/api/voice":
                self._send(200, "application/json", json.dumps(home.set_voice(body)).encode())
            elif path == "/api/intent":
                out = home.voice_intent(str(body.get("target", "")), str(body.get("heard", "")))
                self._send(200, "application/json", json.dumps(out).encode())
            else:
                self._send(404, "text/plain", b"not found")

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/api/state":
                self._send(200, "application/json", json.dumps(home.snapshot()).encode())
            elif path == "/api/events":
                self._sse(home.wait)
            elif path == "/health":
                body = json.dumps({"ok": cam.frame is not None, "fps": round(cam.fps, 1), "size": cam.size,
                                   "frames": cam.seq, "error": cam.error, "version": STREAM_VERSION,
                                   "hands": None if hands is None else {
                                       "ok": hands.result is not None and not hands.error,
                                       "fps": round(hands.fps, 1), "error": hands.error, "epoch": hands.epoch,
                                       "age": round(time.time() - hands.result["t"], 2) if hands.result else None},
                                   "frame_age": round(time.time() - cam.frame_t, 2)}).encode()
                self._send(200, "application/json", body)
            elif path == "/gesture" and hands is not None:
                self._send(200, "application/json", json.dumps(hands.gestures.snapshot()).encode())
            elif path == "/hands" and hands is not None:
                self._send(200, "application/json", json.dumps(hands.result).encode())
            elif path == "/hands.sse" and hands is not None:
                self._sse(hands.wait)
            elif path == "/snapshot.jpg":
                _, frame = cam.wait_frame(-1, 3.0)
                self._send(200, "image/jpeg", frame) if frame else self._send(503, "text/plain", b"no frame")
            elif path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY.decode()}")
                self.send_header("Cache-Control", "no-cache, no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                seq = -1
                try:
                    while True:
                        seq, frame = cam.wait_frame(seq)
                        if frame is None:
                            continue
                        self.wfile.write(b"--" + BOUNDARY + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self._send(404, "text/plain", b"not found")

        def _sse(self, wait):
            """Server-sent events: `wait(last_seq)` blocks until new data (or times out -> keepalive)."""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            seq = -1
            try:
                while True:
                    new_seq, data = wait(seq)
                    if new_seq == seq or data is None:
                        self.wfile.write(b": keepalive\n\n")
                    else:
                        self.wfile.write(b"data: " + json.dumps(data).encode() + b"\n\n")
                    seq = new_seq
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--hands", action="store_true", help="run the NPU hand tracker (needs the handpose venv)")
    ap.add_argument("--no-docker", action="store_true", help="don't also listen on the docker0 bridge")
    a = ap.parse_args()
    cam = Camera(a.device, a.width, a.height, a.fps, a.quality)
    hands = Hands(cam) if a.hands else None
    home = HomeState(STATE_FILE)
    home.on_tv_off = hands.reset if hands else cam.ensure_live
    if hands:
        hands.gestures.listeners.append(home.set_gesture)
        home.gesture_held = hands.gestures.held
    handler = make_handler(cam, hands, home)
    server = ThreadingHTTPServer((a.host, a.port), handler)
    server.daemon_threads = True
    print(f"camera_stream on http://{a.host}:{a.port} ({a.width}x{a.height}@{a.fps})", flush=True)
    if not a.no_docker:
        threading.Thread(target=serve_on_docker_bridge, args=(handler, a.port), daemon=True).start()
    server.serve_forever()


def docker_bridge_ip() -> str | None:
    out = subprocess.run(["ip", "-4", "-o", "addr", "show", "docker0"], capture_output=True, text=True).stdout
    parts = out.split()
    return parts[parts.index("inet") + 1].split("/")[0] if "inet" in parts else None


def serve_on_docker_bridge(handler, port: int):
    """App Lab containers reach the board via Docker's host gateway (docker0), not 127.0.0.1. docker0 is not on the
    LAN, so this exposes the API to local containers only. Docker can start after us at boot, so keep retrying."""
    while True:
        ip = docker_bridge_ip()
        if ip:
            try:
                server = ThreadingHTTPServer((ip, port), handler)
            except OSError as e:
                print(f"docker bridge bind {ip}:{port} failed: {e}", flush=True)
            else:
                server.daemon_threads = True
                print(f"also serving on http://{ip}:{port} for App Lab containers", flush=True)
                server.serve_forever()
        time.sleep(10)


if __name__ == "__main__":
    main()
