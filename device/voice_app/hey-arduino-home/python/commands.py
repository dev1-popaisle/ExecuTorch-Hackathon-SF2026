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
"""Turns a Whisper transcript into home-state changes (lighting / temperature / TV) with simple phrase rules."""

from __future__ import annotations

import re

TARGETS = {
    "lighting": ["lights", "light", "lighting", "lamps", "lamp", "lite"],
    "temperature": ["temperature", "thermostat", "heating", "heater", "heat", "air conditioning", "aircon", "ac",
                    "climate", "cooling"],
    "tv": ["tv", "television", "telly"],
}
NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
# Whisper's usual spellings of words we care about.
FIXES = [(r"\bt\s*\.?\s*v\b\.?", "tv"), (r"\bteevee\b|\btivi\b", "tv"), (r"\bchann?el+s?\b", "channel"),
         (r"\bchannel (to|too)\b", "channel 2"), (r"\bchannel (won|want)\b", "channel 1"),
         (r"\bchannel (tree|free)\b", "channel 3"),
         (r"\ba\s*\.?\s*c\b\.?", "ac"), (r"\bswitch(ed)?\b", "turn"), (r"\bshut\b", "turn off"),
         (r"\bnumber\b", ""), (r"°\s*c\b|\bcentigrade\b", " degrees celsius"),
         (r"\bdegrees?\b|°\s*f?\b|°", " degrees"), (r"\bfahrenheit\b", "")]
# The home runs in Fahrenheit; "celsius" in a command is converted.
TEMP_MIN, TEMP_MAX = 50.0, 86.0


def normalize(text: str) -> str:
    t = text.lower().replace("-", " ")
    t = re.sub(r"(\d)\s*°", r"\1 °", t)
    t = re.sub(r"(\d)\s*([fc])\b", r"\1 °\2", t)  # "72F", "22 C"
    for pat, rep in FIXES:
        t = re.sub(pat, rep, t)
    t = re.sub(r"[^a-z0-9. ]+", " ", t)
    t = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", t)
    words, out, i = t.split(), [], 0
    while i < len(words):
        w = words[i]
        if w in NUMBER_WORDS:
            n = NUMBER_WORDS[w]
            if n >= 20 and i + 1 < len(words) and words[i + 1] in NUMBER_WORDS and NUMBER_WORDS[words[i + 1]] < 10:
                n += NUMBER_WORDS[words[i + 1]]
                i += 1
            out.append(str(n))
        else:
            out.append(w)
        i += 1
    return re.sub(r"\b(\d+) point (\d)\b", r"\1.\2", " ".join(out))


def channel_by_name(clause: str, channels: list[dict] | None) -> int | None:
    """Channel number for a spoken channel name ("sci fi", "comedy"), from the home server's channel list."""
    hits = [(m.start(), c["number"]) for c in channels or [] for a in c.get("aliases", [c["name"].lower()])
            for m in re.finditer(rf"\b{re.escape(a)}\b", clause)]
    return min(hits)[1] if hits else None


def find_target(clause: str, channels: list[dict] | None = None) -> str | None:
    hits = [(m.start(), name) for name, words in TARGETS.items() for w in words
            for m in re.finditer(rf"\b{w}\b", clause)]
    if hits:
        return min(hits)[1]
    if re.search(r"\bchannel\b|\bvolume\b", clause) or channel_by_name(clause, channels):
        return "tv"
    if re.search(r"\bdegrees\b|\bwarmer\b|\bcooler\b|\bcolder\b|\bhotter\b", clause):
        return "temperature"
    if re.search(r"\bbrighter\b|\bdimmer\b|\bdim\b|\bbrightness\b", clause):
        return "lighting"
    return None


def find_on_off(clause: str) -> bool | None:
    if re.search(r"\boff\b|\bdisable\b|\bdeactivate\b", clause):
        return False
    if re.search(r"\bon\b|\benable\b|\bactivate\b|\bstart\b", clause):
        return True
    return None


def number_in(clause: str) -> float | None:
    m = re.search(r"\b(\d+(?:\.\d+)?)\b", clause)
    return float(m.group(1)) if m else None


def clause_action(target: str, clause: str, state: dict) -> tuple[dict, str] | None:
    """Returns ({section: {key: value}}, description) or None if the clause holds no command for target."""
    up = re.search(r"\bup\b|\bnext\b|\bincrease\b|\braise\b|\bhigher\b|\blouder\b|\bwarmer\b|\bhotter\b|\bbrighter\b"
                   r"|\bplus\b", clause)
    down = re.search(r"\bdown\b|\bprevious\b|\bback\b|\bdecrease\b|\blower\b|\bquieter\b|\bcooler\b|\bcolder\b"
                     r"|\bdimmer\b|\bdim\b|\bminus\b", clause)
    n = number_in(clause)
    on = find_on_off(clause)

    if target == "tv":
        tv = state["tv"]
        if re.search(r"\bvolume\b|\blouder\b|\bquieter\b", clause):
            if n is not None:
                v = int(max(0, min(100, n)))
            elif up or down:
                v = max(0, min(100, tv["volume"] + (2 if up else -2)))
            else:
                return None
            return {"tv": {"volume": v, "on": True}}, f"TV volume {v}"
        named = channel_by_name(clause, state.get("tv_channels"))
        if named or re.search(r"\bchannel\b", clause) or (n is not None and on is None):
            count = len(state.get("tv_channels") or []) or 3
            if named:
                ch = named
            elif n is not None:
                ch = int(n)
            elif up or down:
                ch = (tv["channel"] - 1 + (1 if up else -1)) % count + 1
            else:
                return None
            # The home server refuses the change while the TV is off ("TV is off" in the TV room).
            change = {"channel": ch, **({"on": True} if re.search(r"\bturn on\b", clause) else {})}
            names = {c["number"]: c["name"] for c in state.get("tv_channels") or []}
            return {"tv": change}, f"TV channel {ch}" + (f" ({names[ch]})" if ch in names else "")
        if on is not None:
            return {"tv": {"on": on}}, f"TV {'on' if on else 'off'}"
        return None

    if target == "temperature":
        t = state["temperature"]
        if n is not None and on is not False:
            f = n * 9 / 5 + 32 if re.search(r"\bcelsius\b", clause) else n
            target_f = round(max(TEMP_MIN, min(TEMP_MAX, f)), 1)
            return {"temperature": {"target": target_f, "on": True}}, f"Temperature set to {target_f:g}°F"
        if (up or down) and on is None:
            target_f = max(TEMP_MIN, min(TEMP_MAX, t["target"] + (1.0 if up else -1.0)))
            return {"temperature": {"target": target_f, "on": True}}, f"Temperature set to {target_f:g}°F"
        if on is not None:
            return {"temperature": {"on": on}}, f"Temperature control {'on' if on else 'off'}"
        return None

    if target == "lighting":
        light = state["lighting"]
        # "lights 40" / "set the lights to 40", but not overheard chatter like "I see two lights".
        brightness_cmd = len(clause.split()) <= 3 or re.search(
            r"\b(set|turn|dim|make|change|put|brightness|percent|bright)\b", clause)
        if n is not None and on is not False and brightness_cmd:
            b = int(max(0, min(100, n)))
            return {"lighting": {"brightness": b, "on": b > 0}}, f"Lights at {b}%"
        if (up or down) and on is None:
            b = max(5, min(100, light["brightness"] + (20 if up else -20)))
            return {"lighting": {"brightness": b, "on": True}}, f"Lights at {b}%"
        if on is not None:
            return {"lighting": {"on": on}}, f"Lights {'on' if on else 'off'}"
        return None
    return None


GESTURE_TARGETS = {"lighting"}  # devices that "Hey Arduino, <device>" + thumbs up / down can switch
# Whisper's takes on a lone "light" after the wake word. Only used for the gesture intent, which does nothing
# unless a thumbs up / down follows within 3 s, so a false "like" costs nothing.
LIGHT_SOUNDALIKES = r"\b(like|likes|liked|like it|light it|lie it|lied|lights?|lite|lighting|lighten)\b"
FILLER = r"\b(hey|hi|okay|ok|please|the|um|uh|so|oh|now|a|all)\b"
# The wake phrases "Hey Arduino" and "Hey Ardy" as Whisper writes them, including mishearings seen on this board.
# The short "Ardy" forms need the "hey" so words like "hardy" in a sentence don't count.
WAKE_RE = re.compile(r"\b(?:hey|hi|hay|okay|ok)?[\s,.!?]*(?:ar\s*du\s*ino|ardweeno|arduino|our\s*dweeno"
                     r"|are\s+you\s+doing|how\s+are\s+(?:you\s+)?doing|they\s+are\s+tweeno|tweeno)\b"
                     r"|\b(?:hey|hi|hay|a)[\s,.!?]+(?:ardy|ardie|ardi|ardey|arty|artie|arby|arbie|archie|hardy|addy"
                     r"|ar\s*dee|r\.?\s*d\.?|audi|eddie)(?![a-z])"
                     r"|^[\s\-]*(?:ard|ardy|ardie|arty|artie|addy)(?![a-z])", re.I)


def has_wake_phrase(text: str) -> bool:
    return bool(WAKE_RE.search(text))


def starts_with_command(text: str, state: dict) -> bool:
    """The first sentence after any wake phrase is a home command ('Turn off the light.'), as when Whisper drops a
    softly spoken 'Hey Ardy'. Chatter that trips the keyword model rarely opens with one."""
    sentences = command_sentences(text)
    if not sentences:
        return False
    patch, _, target = interpret(sentences[0], state)
    return bool(patch or target)


def wake_confirmed(text: str, state: dict) -> bool:
    """Whether a weak keyword-spotter trigger was really 'Hey Ardy' / 'Hey Arduino'."""
    try:
        return has_wake_phrase(text) or starts_with_command(text, state)
    except KeyError:  # no home state (server unreachable): only the wake phrase can confirm
        return has_wake_phrase(text)


def is_wake_only(text: str) -> bool:
    """'Hey Arduino!' / '' : the command hasn't been said (yet)."""
    return not re.search(r"[a-z0-9]", WAKE_RE.sub(" ", text.lower()))


def gesture_target(text: str) -> str | None:
    """'Hey Arduino, light.' -> 'lighting': a device named with no action, to be completed by a thumbs up / down.
    Only used when parse() found no command, so spoken commands keep priority."""
    norm = normalize(text)
    target = find_target(norm)
    if target is None and len(re.sub(FILLER, " ", norm).split()) <= 3 and re.search(LIGHT_SOUNDALIKES, norm):
        target = "lighting"
    if target not in GESTURE_TARGETS or find_on_off(norm) is not None or number_in(norm) is not None:
        return None
    if re.search(r"\bup\b|\bdown\b|\bbrighter\b|\bdimmer\b|\bdim\b", norm):
        return None
    return target


def command_sentences(text: str) -> list[str]:
    """Sentences most likely to hold the command, best first: the ones right after each wake phrase, then the rest.
    The ASR window also picks up whoever is talking nearby, so the command is usually the first thing after
    'Hey Arduino' and anything later is chatter."""
    starts = [m.end() for m in WAKE_RE.finditer(text)] or [0]
    out: list[str] = []
    for s in starts + [0]:
        for sentence in re.split(r"[.?!;]+", text[s:]):
            sentence = WAKE_RE.sub(" ", sentence).strip(" ,")
            if sentence and sentence not in out:
                out.append(sentence)
    return out


def interpret(text: str, state: dict) -> tuple[dict, list[str], str | None]:
    """(patch, descriptions, gesture target) from the first sentence that is a command. A spoken command anywhere
    wins over a bare device name ('light') waiting for a thumbs up / down: Whisper sometimes writes a stray
    'Light.' before 'Turn on the light.'"""
    sentences = command_sentences(text)
    for sentence in sentences:
        patch, done = parse(sentence, state)
        if patch:
            return patch, done, None
    for sentence in sentences:
        target = gesture_target(sentence)
        if target:
            return {}, [], target
    return {}, [], None


def parse(text: str, state: dict) -> tuple[dict, list[str]]:
    """Returns (merged state patch, descriptions). Handles 'turn on the lights and the TV'."""
    norm = normalize(text)
    patch: dict = {}
    done: list[str] = []
    last_target, last_on = None, None
    for clause in re.split(r"\band\b|\bthen\b|\balso\b|,", norm):
        clause = clause.strip()
        if not clause:
            continue
        channels = state.get("tv_channels")
        target = find_target(clause, channels) or last_target
        on = find_on_off(clause)
        if target is None:
            last_on = on if on is not None else last_on
            continue
        if on is None and last_on is not None and find_target(clause, channels) and not number_in(clause) \
                and not channel_by_name(clause, channels) \
                and not re.search(r"\bup\b|\bdown\b|\bnext\b|\bprevious\b|\bwarmer\b|\bcooler\b|\bcolder\b", clause):
            clause = f"{clause} {'on' if last_on else 'off'}"  # "turn on the lights and the tv"
        merged_state = {k: {**v, **patch.get(k, {})} for k, v in state.items() if isinstance(v, dict)}
        merged_state["tv_channels"] = state.get("tv_channels")
        result = clause_action(target, clause, merged_state)
        if result:
            p, desc = result
            for section, values in p.items():
                patch.setdefault(section, {}).update(values)
            done.append(desc)
        last_target = target
        last_on = find_on_off(clause) if find_on_off(clause) is not None else last_on
    return patch, done
