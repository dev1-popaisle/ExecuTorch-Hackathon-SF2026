#!/usr/bin/python3
"""Ventuno Home UI (macOS, stdlib only, Python 3.9+).

Watches USB/adb for the Arduino VENTUNO Q. When it connects:
  1. forwards the board's camera stream port over adb,
  2. pushes + starts camera_stream.py (with the NPU hand tracker) on the board,
  3. starts the UI server on http://127.0.0.1:9999 (first connection) and opens it in the browser.
"""

import json
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
UI_HOST, UI_PORT = "127.0.0.1", 9999
LOCAL_STREAM_PORT, DEVICE_STREAM_PORT = 18090, 8090
DEVICE_DIR = "/home/arduino/handpose/src"
DEVICE_SCRIPTS = ("camera_stream.py", "hand_tracker.py", "gestures.py", "run_home_server.sh")
POLL_SECONDS = 2.0


def find_device_script(name):
    for p in (HERE / name, HERE.parent / "device" / "handpose" / name):
        if p.exists():
            return p
    raise FileNotFoundError(f"{name} not found next to ventuno_ui.py")


def local_stream_version():
    m = re.search(r'^STREAM_VERSION = "([^"]+)"', find_device_script("camera_stream.py").read_text(), re.M)
    return m.group(1) if m else None


def find_adb():
    for p in (shutil.which("adb"), "/opt/homebrew/bin/adb", "/usr/local/bin/adb"):
        if p and Path(p).exists():
            return p
    raise FileNotFoundError("adb not found (brew install android-platform-tools)")


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class Device:
    """Tracks the VENTUNO Q over adb and keeps the camera stream running on it."""

    def __init__(self):
        self.adb = find_adb()
        self.serial = None
        self.connected = False
        self.hostname = ""
        self._identity = {}
        self.lock = threading.Lock()

    def _run(self, *args, timeout=10):
        return subprocess.run([self.adb, *args], capture_output=True, text=True, timeout=timeout)

    def _shell(self, cmd, timeout=10):
        return self._run("-s", self.serial, "shell", cmd, timeout=timeout)

    def find_ventuno(self):
        try:
            out = self._run("devices", timeout=5).stdout
        except (subprocess.TimeoutExpired, OSError):
            return None
        for line in out.splitlines()[1:]:
            parts = line.split()
            if len(parts) < 2 or parts[1] != "device":
                continue
            serial = parts[0]
            if serial not in self._identity:
                r = subprocess.run([self.adb, "-s", serial, "shell", "hostname; cat /sys/devices/soc0/machine"],
                                   capture_output=True, text=True, timeout=10)
                self._identity[serial] = r.stdout.strip().replace("\n", " ")
            ident = self._identity[serial].lower()
            if "ventuno" in ident or "qcs8275" in ident:
                return serial
        return None

    def on_connect(self):
        self.hostname = self._identity.get(self.serial, "")
        self._run("-s", self.serial, "forward", f"tcp:{LOCAL_STREAM_PORT}", f"tcp:{DEVICE_STREAM_PORT}")
        self._shell(f"mkdir -p {DEVICE_DIR}")
        for name in DEVICE_SCRIPTS:
            self._run("-s", self.serial, "push", str(find_device_script(name)), f"{DEVICE_DIR}/{name}")
        health = device_health()
        if health and health.get("version") != local_stream_version():
            log(f"restarting outdated camera stream (v{health.get('version')} -> v{local_stream_version()})")
            # Kept as its own command: the pattern must not match this shell's own command line.
            self._shell("pkill -f '[c]amera_stream.py'; sleep 1")
        self.ensure_stream()

    def ensure_stream(self):
        # The board also runs this from cron, so the home server works without the Mac.
        self._shell(f"bash {DEVICE_DIR}/run_home_server.sh")

    def poll(self):
        serial = self.find_ventuno()
        with self.lock:
            was = self.connected
            self.serial = serial or self.serial
            self.connected = serial is not None
        if self.connected and not was:
            log(f"VENTUNO Q connected ({serial})")
            try:
                self.on_connect()
            except (subprocess.SubprocessError, OSError) as e:
                log(f"device setup failed, will retry from the UI: {e}")
            return "connected"
        if was and not self.connected:
            log("VENTUNO Q disconnected")
            return "disconnected"
        return None


def device_health():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{LOCAL_STREAM_PORT}/health", timeout=1.5) as r:
            return json.loads(r.read())
    except Exception:  # noqa: BLE001
        return None


def device_request(path, body=None):
    """The home state lives on the board (voice commands change it there); the UI only relays."""
    req = urllib.request.Request(f"http://127.0.0.1:{LOCAL_STREAM_PORT}{path}", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3) as r:
        return r.read()


def make_handler(device):
    index_html = HERE / "index.html"

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, "application/json", json.dumps(obj).encode())

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", index_html.read_bytes())
            elif path == "/hls.min.js":
                self._send(200, "text/javascript; charset=utf-8", (HERE / "hls.min.js").read_bytes())
            elif path == "/api/status":
                self._json({"device": {"connected": device.connected, "serial": device.serial,
                                       "name": device.hostname},
                            "camera": device_health() if device.connected else None,
                            "ui_version": index_html.stat().st_mtime_ns})
            elif path == "/api/state":
                self._relay_state()
            elif path == "/stream.mjpg":
                self._proxy_stream("/stream.mjpg")
            elif path == "/api/hands.sse":
                self._proxy_stream("/hands.sse")
            elif path == "/api/events":
                self._proxy_stream("/api/events")
            else:
                self._send(404, "text/plain", b"not found")

        def do_POST(self):
            if self.path.split("?")[0] != "/api/state":
                return self._send(404, "text/plain", b"not found")
            self._relay_state(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")

        def _relay_state(self, body=None):
            if not device.connected:
                return self._json({"error": "device not connected"}, 503)
            try:
                self._send(200, "application/json", device_request("/api/state", body))
            except urllib.error.HTTPError as e:
                self._send(e.code, "application/json", e.read())
            except Exception as e:  # noqa: BLE001
                self._json({"error": f"home server on the board unreachable: {e}"}, 503)

        def _proxy_stream(self, upstream_path):
            if not device.connected:
                return self._send(503, "text/plain", b"device not connected")
            try:
                upstream = urllib.request.urlopen(f"http://127.0.0.1:{LOCAL_STREAM_PORT}{upstream_path}", timeout=10)
            except Exception:  # noqa: BLE001
                device.ensure_stream()
                return self._send(503, "text/plain", b"camera stream not ready")
            self.send_response(200)
            self.send_header("Content-Type", upstream.headers.get("Content-Type"))
            self.send_header("Cache-Control", "no-cache, no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            try:
                while True:
                    chunk = upstream.read1(64 * 1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                upstream.close()

    return Handler


def main():
    device = Device()
    server = None
    last_open = 0.0
    failures = 0
    log(f"watching for VENTUNO Q (adb={device.adb})")
    while True:
        try:
            event = device.poll()
        except Exception as e:  # noqa: BLE001
            log(f"poll error: {e}")
            event = None
        if device.connected and server is None:
            try:
                server = ThreadingHTTPServer((UI_HOST, UI_PORT), make_handler(device))
            except OSError as e:
                log(f"cannot bind {UI_HOST}:{UI_PORT} ({e}); retrying")
                time.sleep(POLL_SECONDS)
                continue
            server.daemon_threads = True
            threading.Thread(target=server.serve_forever, daemon=True).start()
            log(f"UI running at http://{UI_HOST}:{UI_PORT}")
            event = "connected"
        if device.connected and event != "connected":
            # An adb server restart drops port forwards without ever looking like a disconnect.
            failures = 0 if device_health() else failures + 1
            if failures >= 2:
                log("camera stream unreachable; re-applying adb forward and restarting stream")
                try:
                    device.on_connect()
                except (subprocess.SubprocessError, OSError) as e:
                    log(f"device setup failed: {e}")
                failures = 0
        if event == "connected":
            failures = 0
            if "--no-browser" not in sys.argv and time.time() - last_open > 30:
                webbrowser.open(f"http://{UI_HOST}:{UI_PORT}")
                last_open = time.time()
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
