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
"""Generates architecture_diagram.svg (system + data flow of Sign & Say).

    python3 build_diagram.py      # writes architecture_diagram.svg next to this file
"""

from pathlib import Path
from xml.sax.saxutils import escape

W, H = 2200, 1310
FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"

C = {
    "cam": ("#f4f4f5", "#52525b"),
    "board": ("#f2fafa", "#00878f"),
    "build": ("#f5f5f5", "#8a8a8a"),
    "hand": ("#e9f1fd", "#2f6fd6"),
    "hand_box": ("#ffffff", "#2f6fd6"),
    "npu": ("#fff3e6", "#e8710a"),
    "npu_box": ("#ffdcb5", "#d35f00"),
    "npu_core": ("#f59a3c", "#b04f00"),
    "voice": ("#eaf7ee", "#2e9d4f"),
    "voice_box": ("#ffffff", "#2e9d4f"),
    "api": ("#ffffff", "#00666c"),
    "mac": ("#f4effc", "#7a4bd1"),
    "mac_box": ("#ffffff", "#7a4bd1"),
    "mcu": ("#e6f6f7", "#00878f"),
}

out: list[str] = []


def text(x, y, s, size=12, weight="normal", color="#222", anchor="start", rotate=None, italic=False):
    tr = f' transform="rotate({rotate} {x} {y})"' if rotate else ""
    st = ' font-style="italic"' if italic else ""
    out.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}" '
               f'text-anchor="{anchor}"{tr}{st}>{escape(s)}</text>')


def rect(x, y, w, h, kind, rx=10, dashed=False, sw=2):
    fill, stroke = C[kind]
    dash = ' stroke-dasharray="7 5"' if dashed else ""
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" '
               f'stroke-width="{sw}"{dash}/>')


def box(x, y, w, h, kind, lines, dashed=False, title_color=None):
    """Rounded box with a bold first line and smaller detail lines, centered."""
    rect(x, y, w, h, kind, rx=8, dashed=dashed, sw=1.6)
    stroke = title_color or C[kind][1]
    total = 17 + 15 * (len(lines) - 1)
    ty = y + h / 2 - total / 2 + 13
    text(x + w / 2, ty, lines[0], 14, "bold", stroke if kind not in ("npu_core",) else "#fff", "middle")
    for i, line in enumerate(lines[1:]):
        text(x + w / 2, ty + 17 + 15 * i, line, 12, "normal", "#333" if kind != "npu_core" else "#fff", "middle")


def arrow(points, color="#444", both=False, dashed=False, width=2.2):
    pts = " ".join(f"{x},{y}" for x, y in points)
    marker = color.lstrip("#")
    dash = ' stroke-dasharray="8 6"' if dashed else ""
    start = f' marker-start="url(#s{marker})"' if both else ""
    out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}"{dash}{start} '
               f'marker-end="url(#e{marker})" stroke-linejoin="round"/>')


ARROW_COLORS = ["#444", "#2f6fd6", "#2e9d4f", "#d35f00", "#7a4bd1", "#00666c", "#8a8a8a"]
BLUE, GREEN, ORANGE, PURPLE, TEAL, GRAY = ARROW_COLORS[1:]

# ---------------------------------------------------------------- title
text(W / 2, 48, "Sign & Say — Smart Home Mobility Project", 30, "bold", "#111", "middle")
text(W / 2, 78, "System architecture & data flow  ·  all AI runs on the Arduino VENTUNO Q (Qualcomm QCS8275)  ·  "
     "the laptop only displays the UI", 15, "normal", "#555", "middle")

# ---------------------------------------------------------------- Logitech camera unit
rect(30, 280, 210, 830, "cam", rx=14)
text(135, 306, "📷 LOGITECH CAMERA", 15, "bold", C["cam"][1], "middle")
text(135, 324, "USB, plugged into the board", 11.5, "normal", "#555", "middle")
box(45, 340, 180, 90, "cam", ["Image sensor", "1280×720 MJPEG", "30 fps · /dev/video0"])
box(45, 925, 180, 90, "cam", ["Built-in microphone", "ALSA capture", "always listening"])

# ---------------------------------------------------------------- VENTUNO Q board
rect(270, 110, 1440, 1070, "board", rx=18, sw=3)
text(292, 140, "ARDUINO VENTUNO Q", 21, "bold", C["board"][1])
text(548, 140, "Qualcomm QCS8275 SoC  ·  Kryo CPU (Ubuntu 24.04)  ·  Hexagon NPU HTP V75  ·  "
     "on-board MCU + LED matrix", 14, "normal", "#2b5f62")

# build-time strip (right to left: PyTorch -> .pte)
rect(320, 155, 1200, 100, "build", rx=10, dashed=True, sw=1.6)
text(335, 174, "BUILD TIME — PyTorch → ExecuTorch export (run once, on the board)", 13, "bold", "#666")
box(1300, 183, 210, 64, "build", ["PyTorch 2.13 models", "MediaPipePyTorch (AI Hub pin)"])
box(1110, 183, 170, 64, "build", ["torch.export", "capture the graph"])
box(880, 183, 210, 64, "build", ["QnnQuantizer 16a8w", "calibrated on real frames"])
box(640, 183, 220, 64, "build", ["QnnPartitioner", "target QCS8275 · HTP V75"])
box(340, 183, 280, 64, "build", [".pte files  (export/pte/)", "100% NPU-delegated · 0.5–3.4 MB"])
for x0, x1 in ((1300, 1280), (1110, 1090), (880, 860), (640, 620)):
    arrow([(x0, 215), (x1, 215)], GRAY)
arrow([(340, 228), (300, 228), (300, 735), (340, 735)], GRAY, dashed=True)
text(292, 520, "deploy .pte models to the NPU", 12, "bold", "#777", "middle", rotate=-90)

# ---------------------------------------------------------------- hand gesture lane (CPU)
rect(320, 275, 1200, 275, "hand", rx=12)
text(340, 465, "✋ HAND GESTURE PIPELINE", 15, "bold", BLUE)
text(340, 484, "Home server · camera_stream.py :8090", 12, "normal", "#333")
text(340, 501, "Kryo CPU · Python 3.12 · NumPy + OpenCV", 12, "normal", "#333")
text(340, 518, "started at boot by cron (run_home_server.sh)", 12, "normal", "#333")

box(340, 320, 170, 80, "hand_box", ["Camera capture", "OpenCV V4L2", "latest-frame thread"])
box(530, 320, 190, 80, "hand_box", ["Hand tracker", "hand_tracker.py", "letterbox · crops · NMS"])
box(740, 320, 160, 80, "hand_box", ["Face filter", "drop hands with >60%", "keypoints in a face"])
box(920, 320, 150, 80, "hand_box", ["One Euro filter", "smooth 21 keypoints", "up to 2 hands"])
box(1090, 320, 160, 80, "hand_box", ["Gesture rules 👍 👎", "gestures.py", "3D thumb + fist geometry"])
box(1270, 320, 160, 80, "hand_box", ["GestureState", "5 of 10 frames", "3 s hold · instant switch"])
for x0, x1 in ((510, 530), (720, 740), (900, 920), (1070, 1090), (1250, 1270)):
    arrow([(x0, 360), (x1, 360)], BLUE)

box(1090, 440, 190, 80, "hand_box", ["Voice + gesture fusion", "'light' + 👍/👎 within 3 s", "→ lights on / off"],
    title_color="#7a3fb0")
box(1300, 440, 200, 80, "hand_box", ["Home state", "Lighting · Temperature °F · TV", "home_state.json"])
arrow([(1350, 400), (1185, 440)], BLUE)
arrow([(1280, 480), (1300, 480)], BLUE)
arrow([(1500, 480), (1540, 480)], TEAL)

arrow([(425, 320), (425, 292), (1540, 292)], TEAL)
text(560, 287, "MJPEG video (passthrough, no re-encode)", 11.5, "bold", TEAL)
arrow([(995, 320), (995, 305), (1540, 305)], TEAL)
text(1090, 317, "21 keypoints / hand → /hands.sse", 11, "bold", TEAL)

# ---------------------------------------------------------------- NPU band
rect(320, 575, 1200, 255, "npu", rx=12, sw=2.5)
text(900, 600, "⚡ QUALCOMM HEXAGON NPU (HTP V75)", 17, "bold", ORANGE)
text(1345, 600, "NPU software stack", 12.5, "bold", ORANGE)

box(340, 612, 540, 50, "npu_box", ["ExecuTorch qnn_executor_runner — 1 persistent process per model",
                                   "QNN context stays loaded · frames in/out via /dev/shm"])
box(340, 685, 170, 100, "npu_box", ["Palm detector", "MediaPipe BlazePalm", "16a8w · 1.42 ms", "(CPU: 116 ms)"])
box(525, 685, 170, 100, "npu_box", ["Hand landmark", "21 × 3D keypoints", "16a8w · 1.39 ms", "(CPU: 75 ms)"])
box(710, 685, 170, 100, "npu_box", ["Face detector", "MediaPipe BlazeFace", "16a8w · 1.3 ms", "every 2nd frame"])
for x in (425, 610, 795):
    arrow([(x, 662), (x, 685)], ORANGE)

box(920, 612, 400, 50, "npu_box", ["Arduino App Lab inference containers",
                                   "ei-qnn-models-runner · audio-analytics (QNN)"])
box(920, 685, 190, 100, "npu_box", ["Wake-word model", "Edge Impulse KWS", "'Hey Arduino'", "keyword spotting"])
box(1130, 685, 190, 100, "npu_box", ["Whisper Small", "speech → text", "English", "fully offline"])
for x in (1015, 1225):
    arrow([(x, 662), (x, 685)], ORANGE)

box(1340, 612, 170, 40, "npu_box", ["QNN delegate", "ExecuTorch backend"])
box(1340, 664, 170, 40, "npu_box", ["QAIRT / QNN 2.49"])
box(1340, 716, 170, 40, "npu_box", ["FastRPC (CPU ↔ DSP)"])
box(1340, 768, 170, 48, "npu_core", ["Hexagon HTP V75", "tensor cores"])
for y0, y1 in ((652, 664), (704, 716), (756, 768)):
    arrow([(1425, y0), (1425, y1)], ORANGE)
arrow([(1320, 650), (1330, 650), (1330, 684), (1340, 684)], ORANGE, width=1.6)

arrow([(680, 400), (680, 612)], ORANGE, both=True)
text(690, 452, "frames → NPU", 11.5, "bold", ORANGE)
text(690, 468, "← boxes / keypoints", 11.5, "bold", ORANGE)
arrow([(820, 400), (820, 612)], ORANGE, both=True)
text(830, 452, "face boxes", 11.5, "bold", ORANGE)

# ---------------------------------------------------------------- voice lane
rect(320, 855, 1200, 300, "voice", rx=12)
text(340, 880, "🎙 VOICE AI PIPELINE", 15, "bold", GREEN)
text(340, 898, "Arduino App Lab app 'hey-arduino-home' · runs in Docker containers", 12, "normal", "#333")

box(340, 920, 220, 90, "voice_box", ["Microphone + pre-roll", "PrerollMicrophone", "8 s rolling buffer"])
box(920, 920, 190, 90, "voice_box", ["KeywordSpotting brick", "'Hey Arduino' ≥ 0.90", "'Hey Ardy' ≥ 0.30"])
box(1130, 920, 190, 90, "voice_box", ["ASR brick (Whisper)", "asr_session.py", "6 s deadline flush"])
box(1340, 920, 170, 90, "voice_box", ["commands.py", "wake confirm + parse", "lights · temp · TV"])
arrow([(560, 965), (920, 965)], GREEN)
text(740, 958, "audio stream", 11.5, "bold", GREEN, "middle")
arrow([(1110, 965), (1130, 965)], GREEN)
arrow([(1320, 965), (1340, 965)], GREEN)
arrow([(1015, 920), (1015, 785)], ORANGE, both=True)
arrow([(1225, 920), (1225, 785)], ORANGE, both=True)

box(340, 1040, 540, 95, "voice_box", ["Arduino App Lab platform", "arduino-app-cli 0.13 · bricks: arduino:keyword_spotting, arduino:asr",
                                      "containers: python-apps-base 0.12 · ei-qnn-models-runner 0.12.1",
                                      "audio-analytics 1.0.2"], dashed=True)
box(1340, 1045, 170, 90, "mcu", ["On-board MCU", "sketch.ino via Bridge", "LED matrix: listening,", "done, not understood"])
arrow([(1425, 1010), (1425, 1045)], GREEN)

arrow([(1510, 985), (1540, 985)], GREEN)
arrow([(1510, 945), (1528, 945), (1528, 538), (1185, 538), (1185, 520)], "#7a4bd1")
text(1200, 534, "voice 'light' → /api/intent → wait for 👍 / 👎", 11, "bold", PURPLE)

# ---------------------------------------------------------------- HTTP API column
rect(1540, 275, 150, 880, "api", rx=10, sw=2.5)
text(1615, 300, "HTTP API", 16, "bold", TEAL, "middle")
text(1615, 318, "port 8090", 12, "normal", "#333", "middle")
for i, s in enumerate(["/stream.mjpg", "/hands.sse", "/gesture", "/api/events", "/api/state", "/health"]):
    text(1615, 420 + 22 * i, s, 12.5, "normal", "#222", "middle")
text(1615, 920, "from voice app:", 12, "bold", GREEN, "middle")
for i, s in enumerate(["/api/state", "/api/intent", "/api/voice"]):
    text(1615, 942 + 20 * i, s, 12.5, "normal", "#222", "middle")
text(1615, 1080, "also bound on docker0", 11, "normal", "#555", "middle")
text(1615, 1095, "for App Lab containers", 11, "normal", "#555", "middle")

# ---------------------------------------------------------------- laptop
rect(1760, 275, 410, 760, "mac", rx=16, sw=3)
text(1780, 302, "💻 MACBOOK LAPTOP", 17, "bold", PURPLE)
text(1780, 321, "UI only — no AI runs here", 12.5, "normal", "#555")
box(1780, 335, 370, 66, "mac_box", ["adb forward  tcp:18090 → board :8090", "over the USB-C cable"])
box(1780, 425, 370, 82, "mac_box", ["ventuno_ui.py  →  http://127.0.0.1:9999", "launchd agent com.ventuno.home-ui",
                                    "USB watcher · deploys scripts · HTTP proxy"])
rect(1780, 530, 370, 480, "mac_box", rx=10, sw=1.6)
text(1965, 554, "🌐 Browser — index.html", 14, "bold", PURPLE, "middle")
text(1965, 572, "'Sign & Say' house UI", 12, "normal", "#333", "middle")
box(1800, 588, 330, 92, "mac", ["Camera view", "live MJPEG + hand skeleton overlay", "👍 / 👎 gesture badge"])
box(1800, 692, 330, 50, "mac", ["💡 Lighting — on/off, brightness"])
box(1800, 752, 330, 50, "mac", ["🌡 Temperature — actual | target °F"])
box(1800, 812, 330, 50, "mac", ["📺 TV — Comedy · Sci Fi · Action"])
box(1800, 872, 330, 50, "mac", ["Status: board · camera fps · gesture · voice"])
box(1800, 932, 330, 64, "mac", ["hls.js player", "TV stream inside the camera window"])
arrow([(1690, 368), (1780, 368)], TEAL, both=True, width=3)
text(1735, 358, "USB-C", 13, "bold", TEAL, "middle")
arrow([(1965, 401), (1965, 425)], PURPLE, both=True)
arrow([(1965, 507), (1965, 530)], PURPLE, both=True)

box(1820, 1065, 290, 60, "build", ["☁ Pluto TV (internet)", "HLS live channel streams"])
arrow([(1965, 1065), (1965, 996)], GRAY, dashed=True)

# ---------------------------------------------------------------- camera links
arrow([(225, 375), (340, 375)], "#52525b")
text(282, 368, "USB video", 11, "bold", "#52525b", "middle")
arrow([(225, 965), (340, 965)], "#52525b")
text(282, 958, "USB audio", 11, "bold", "#52525b", "middle")

# ---------------------------------------------------------------- legend
LY = 1215
legend = [("cam", "Logitech camera unit"), ("board", "Arduino VENTUNO Q board"), ("hand", "Hand gesture pipeline (CPU)"),
          ("npu_box", "Qualcomm Hexagon NPU"), ("voice", "Voice AI (Arduino App Lab)"),
          ("build", "Build time (PyTorch → ExecuTorch)"), ("mac", "Laptop UI")]
x = 40
for kind, label in legend:
    rect(x, LY, 26, 18, kind, rx=4, sw=1.6)
    text(x + 34, LY + 14, label, 13, "normal", "#222")
    x += 34 + len(label) * 7.2 + 26
arrow([(40, LY + 55), (100, LY + 55)], "#444")
text(110, LY + 60, "live data", 13)
arrow([(210, LY + 55), (270, LY + 55)], "#444", both=True)
text(280, LY + 60, "request / response", 13)
arrow([(440, LY + 55), (500, LY + 55)], GRAY, dashed=True)
text(510, LY + 60, "deploy / external", 13)
text(700, LY + 60, "Hand tracking ~30 fps  ·  NPU: palm 1.42 ms, landmarks 1.39 ms, face 1.3 ms "
     "(CPU fp32: 75–116 ms)  ·  Voice: wake word + Whisper on the NPU, no cloud", 13, "bold", "#444")

# ---------------------------------------------------------------- assemble
defs = []
for col in ARROW_COLORS + ["#52525b"]:
    m = col.lstrip("#")
    defs.append(f'<marker id="e{m}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{col}"/></marker>')
    defs.append(f'<marker id="s{m}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{col}"/></marker>')
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
       f'font-family="{FONT}">\n<defs>{"".join(defs)}</defs>\n'
       f'<rect width="{W}" height="{H}" fill="#ffffff"/>\n' + "\n".join(out) + "\n</svg>\n")
dest = Path(__file__).with_name("architecture_diagram.svg")
dest.write_text(svg, encoding="utf-8")
print(f"wrote {dest}")
