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
"""Transcripts heard on the board (Whisper on the NPU, noisy room) -> expected outcome. Run: python3 test_commands.py"""

import sys

from commands import has_wake_phrase, interpret, is_wake_only, wake_confirmed

STATE = {
    "lighting": {"on": False, "brightness": 70},
    "temperature": {"on": True, "target": 72.0},
    "tv": {"on": True, "channel": 1, "volume": 12},
    "tv_channels": [{"number": 1, "name": "Comedy", "aliases": ["comedy"]},
                    {"number": 2, "name": "Sci Fi", "aliases": ["sci fi", "scifi"]},
                    {"number": 3, "name": "Action", "aliases": ["action"]}],
}
G = "gesture"  # bare "light": waits for a thumbs up / down
CASES = [
    # bare "light" + chatter picked up in the same window
    ("Hey Arduino, light.", G),
    ("Hey, Arduino lights.", G),
    ("Hey Arduino. Light. Light. And the first person to get five lights.", G),
    ("Hey Arduino light? Light? It should work. job.", G),
    ("supposed to say it. Hey Arduino, light.", G),
    ("Hey Arduino, light. Hi Kim. Hi Twitter. These young people bro, what turned on", G),
    ("Hey, are you doing light? somebody smart.", G),
    ("Hey, how are doing all? Light?", G),
    ("Light. They are Tweeno. Light.", G),
    ("Hey Arduino, like it?", G),
    ("like it? I can do it as well.", G),
    # chatter only
    ("I see two lights. I see four. I see four. I did not get any attention from you yet.", None),
    ("Hey Arduino, we can remove that. What does he think about that? Hey Arduino, would you perform?", None),
    ("It's not good enough.", None),
    ("But you're gonna come by train though so... I'll come by train.", None),
    # spoken commands, unchanged
    ("Hey Arduino, turn on the light.", "Lights on"),
    ("Hey Arduino, turn off the lights.", "Lights off"),
    ("Hey Arduino turn off the light. Yeah that's it.", "Lights off"),
    ("Light. How you doing? Turn on the light. love.", "Lights on"),
    ("Light. Hey, what do we know? Turn off the lights. [", "Lights off"),
    ("- I'm gonna push all the changes. - Light. Turn on the light.", "Lights on"),
    ("Hey Arduino, lights 40.", "Lights at 40%"),
    ("Hey Arduino, set the lights to 30 percent.", "Lights at 30%"),
    ("Hey Arduino, dim the lights.", "Lights at 50%"),
    ("Hey Arduino, turn on the lights and the TV.", "Lights on, TV on"),
    ("Hey Arduino, go to channel 2.", "TV channel 2 (Sci Fi)"),
    ("Hey Arduino, sci fi.", "TV channel 2 (Sci Fi)"),
    ("Hey Arduino, set the temperature to 70 degrees.", "Temperature set to 70°F"),
    # "Hey Ardy" works like "Hey Arduino" for every command
    ("Hey Ardy, light.", G),
    ("Hey Artie, lights.", G),
    ("Hey, Hardy. Light.", G),
    ("Hey Ardy, turn on the light.", "Lights on"),
    ("Hey Ardie, turn off the lights.", "Lights off"),
    ("Hey Arty, lights 40.", "Lights at 40%"),
    ("Hey Ardy, turn on the TV.", "TV on"),
    ("Hey Ardy, go to channel 3.", "TV channel 3 (Action)"),
    ("Hey Ardy, comedy.", "TV channel 1 (Comedy)"),
    ("Hey Ardy, set the temperature to 68.", "Temperature set to 68°F"),
    ("Hey R.D., turn off the TV.", "TV off"),
    ("Hey, Ardee. Turn on the light.", "Lights on"),   # heard on the board
    ("Hey, Arby, turn off the light.", "Lights off"),  # heard on the board
]
WAKE_ONLY = ["", "Hey Arduino!", "Hey, Arduino.", "Hey Arduino", "Hey Arduino light.",
             "Hey Ardy!", "Hey, Artie.", "Hey Ardy, light."]
# Wake phrase check for weak keyword-spotter triggers ("Hey Ardy" scores lower than "Hey Arduino").
WAKE_PHRASE = [("Hey Ardy, light.", True), ("Hey Artie, turn on the TV.", True), ("Hey, Hardy.", True),
               ("Hey Ardie lights", True), ("Hey Arduino, light.", True), ("Hey, are you doing light?", True),
               ("Hey everybody, let me pull all the changes.", False), ("Laurel and Hardy were funny.", False),
               ("That party was great.", False), ("Turn on the light.", False), ("", False)]
# Weak triggers heard on the board for spoken "Hey Ardy ..." (True) and room chatter (False).
WAKE_CONFIRMED = [("Turn off the light.", True), ("ARD, turn off the light.", True), ("Hey Addy, light.", True),
                  ("Hey, Arby, turn off the light.", True), ("", False),
                  ("Hey everybody, let me pull all the changes.", False),
                  ("I see two lights. I see four.", False), ("It's not good enough.", False),
                  ("You can push out the changes. Turn on the lights.", False)]


def main() -> int:
    bad = 0
    for text, want in CASES:
        patch, done, target = interpret(text, STATE)
        got = ", ".join(done) if patch else (G if target else None)
        ok = got == want
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {text[:72]!r:76s} -> {got}" + ("" if ok else f"   (want {want})"))
    for text in WAKE_ONLY:
        want = "light" not in text
        ok = is_wake_only(text) == want
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} wake only {text!r:24s} -> {is_wake_only(text)}")
    for text, want in WAKE_PHRASE:
        ok = has_wake_phrase(text) == want
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} wake phrase {text!r:46s} -> {has_wake_phrase(text)}")
    for text, want in WAKE_CONFIRMED:
        ok = wake_confirmed(text, STATE) == want
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} weak wake confirmed {text!r:46s} -> {wake_confirmed(text, STATE)}")
    total = len(CASES) + len(WAKE_ONLY) + len(WAKE_PHRASE) + len(WAKE_CONFIRMED)
    print(f"{total - bad}/{total} passed")
    return bad


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
