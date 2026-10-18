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
"""Hey Arduino home commands: wake word -> Whisper (NPU) -> phrase rules -> Ventuno home server on the board.

Based on the Arduino "Local AI Voice Assistant" tutorial (keyword spotting + ASR bricks), without the LLM and TTS:
commands only switch lighting / temperature / TV in the home state served by camera_stream.py on port 8090.
"""

import json
import os
import time
import urllib.request

from arduino.app_bricks.asr import AutomaticSpeechRecognition
from arduino.app_bricks.keyword_spotting import KeywordSpotting
from arduino.app_peripherals.microphone import Microphone
from arduino.app_utils import App, Bridge

from asr_session import transcribe_command
from commands import interpret, wake_confirmed
from mic_preroll import PrerollMicrophone

# The app runs in a container; App Lab maps this host alias to the board (Docker host gateway, docker0), where
# camera_stream.py also listens.
HOME_URL = os.environ.get("VENTUNO_HOME_URL", "http://msgpack-rpc-router:8090")
WAKE_WORD = "hey_arduino"  # the keyword model's label; it covers both wake phrases (see spotter below)
WAKE_SURE = 0.90  # at or above: "Hey Arduino", act on it as before
WAKE_MIN = 0.30   # weaker matches ("Hey Ardy") must be confirmed by the transcript (commands.wake_confirmed)
LISTEN_TIMEOUT_S = 6
# LED matrix states understood by the sketch
IDLE, LISTENING, PROCESSING, DONE, FAILED = 0, 1, 2, 3, 4


def home(path, body=None):
    req = urllib.request.Request(HOME_URL + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3) as r:
        return json.loads(r.read())


def report(status, **extra):
    try:
        home("/api/voice", {"status": status, **extra})
    except OSError as e:
        print(f"home server unreachable: {e}")


def show(led_state):
    try:
        Bridge.call("set_state", led_state)
    except Exception as e:  # noqa: BLE001
        print(f"LED matrix update failed: {e}")


mic_spotter = Microphone(device=Microphone.USB_MIC_1, shared=True)
mic_asr = PrerollMicrophone(Microphone(device=Microphone.USB_MIC_1, shared=True))
asr = AutomaticSpeechRecognition(mic_asr, language="en")
# The keyword model is trained on "Hey Arduino" only. "Hey Ardy" scores lower, so weaker matches also start
# listening, but only count if the transcript confirms them: a wake phrase, or a home command right away.
spotter = KeywordSpotting(mic=mic_spotter, confidence=WAKE_MIN, debounce_sec=2.0)
wake = {"score": 0.0, "weak": False, "pending": None}

app_state = "IDLE"


def best_match_with_score(item, confidence):
    match = KeywordSpotting.get_best_match(item, confidence)
    if match:
        wake["score"] = match[1] / 100
    return match


spotter.get_best_match = best_match_with_score


def on_wake_word():
    global app_state
    score = wake["score"]
    if app_state == "IDLE":
        mic_asr.mark()
        wake["weak"] = score < WAKE_SURE
        print(f"\nWake word detected (score {score:.2f}{', checking transcript' if wake['weak'] else ''})")
        app_state = "LISTENING"
    else:  # handled right after the current command, replaying the audio since that command ended
        wake["pending"] = score


spotter.on_detect(WAKE_WORD, on_wake_word)
mic_spotter.start()
asr.start()
spotter.start()
show(IDLE)
report("idle", heard="", result="", ok=None)
print("Ready. Say 'Hey Arduino' or 'Hey Ardy', then a command like 'turn on the lights'.")


def listen(state: dict, weak: bool) -> str:
    def is_command(text: str) -> bool:
        if weak and not wake_confirmed(text, state):
            return False
        try:
            patch, _, target = interpret(text, state)
        except KeyError:  # home server was unreachable: keep listening until the deadline
            return False
        return bool(patch or target)

    def give_up(text: str) -> bool:
        return weak and not wake_confirmed(text, state)  # chatter that tripped the keyword model

    mic_asr.start()
    try:
        return transcribe_command(asr, LISTEN_TIMEOUT_S, is_command, give_up=give_up)
    except Exception as e:  # noqa: BLE001
        print(f"ASR error: {e}")
        return ""
    finally:
        mic_asr.stop()


def loop():
    global app_state
    if app_state != "LISTENING":
        time.sleep(0.05)
        return

    show(LISTENING)
    report("listening", heard="", result="")
    try:
        state = home("/api/state")
    except OSError:
        state = {}
    weak = wake["weak"]
    text = listen(state, weak)
    print(f"Heard: {text!r}")
    if weak and not wake_confirmed(text, state):
        print("Ignored: no 'Hey Ardy' / 'Hey Arduino' in the transcript (keyword model matched chatter)")
        show(IDLE)
        report("idle")
        next_wake()
        return

    show(PROCESSING)
    ok, result = False, "Didn't catch a command"
    if text:
        try:
            patch, done, target = interpret(text, home("/api/state"))
            if patch:
                resp = home("/api/state", {**patch, "source": "voice"})
                refused = list((resp.get("change") or {}).get("notice", {}).values())
                ok, result = not refused, ", ".join(refused or done)
            elif target:
                # "Hey Arduino, light" + thumbs up / down: the home server pairs it with a gesture within 3 s.
                resp = home("/api/intent", {"target": target, "heard": text})
                ok, result = {"done": True, "waiting": None}.get(resp["status"], False), resp["result"]
            else:
                result = "Not a home command"
        except OSError as e:
            result = f"Home server unreachable ({e})"
    print({True: "OK: ", None: "Waiting for gesture: "}.get(ok, "Ignored: ") + result)
    if ok is not None:  # while waiting for a gesture the home server owns the voice status
        report("done" if ok else "failed", heard=text, result=result, ok=ok)

    show({True: DONE, None: PROCESSING}.get(ok, FAILED))
    time.sleep(1.2)
    show(IDLE)
    report("idle")
    next_wake()


def next_wake():
    """Back to idle, or straight into a wake word that came in while busy."""
    global app_state
    score, wake["pending"] = wake["pending"], None
    if score is None:
        app_state = "IDLE"
        return
    wake["weak"] = score < WAKE_SURE
    print(f"\nWake word detected while busy (score {score:.2f}), listening")
    app_state = "LISTENING"


try:
    App.run(user_loop=loop)
finally:
    spotter.stop()
    mic_spotter.stop()
    mic_asr.stop()
    asr.stop()
    show(IDLE)
    report("offline")
