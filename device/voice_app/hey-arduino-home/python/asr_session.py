"""Command-oriented listening on top of the ASR brick.

`transcribe_sentence(timeout)` ends a session at its first sentence boundary (700 ms of silence) or, at the timeout,
closes it and drops whatever Whisper hasn't returned yet. With people talking nearby there is rarely 700 ms of
silence, so commands said in a noisy room came back empty. Here the session keeps collecting sentences until they
hold a command, and at the deadline it is flushed (Whisper transcribes the buffered audio) before closing.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

GRACE_S = 2.0  # time for Whisper to return the flushed audio


def transcribe_command(asr, deadline_s: float, is_command: Callable[[str], bool], start_stream=None,
                       give_up: Callable[[str], bool] | None = None) -> str:
    """Text heard until it holds a command (is_command), give_up(text) after a sentence, or deadline_s passes;
    '' if nothing was said."""
    stop = threading.Event()

    def watchdog():
        if stop.wait(deadline_s):
            return
        session = asr._active_session
        if session is not None:
            asr._flush_transcription_session(session.session_id)
        if not stop.wait(GRACE_S):
            asr.cancel()

    threading.Thread(target=watchdog, daemon=True, name="asr-deadline").start()
    sentences: list[str] = []
    partial = ""
    t0 = time.monotonic()
    try:
        stream = start_stream() if start_stream else asr.transcribe_stream(duration=0)
        with stream:
            for ev in stream:
                if ev.type == "partial_text" and ev.data.strip():
                    partial = ev.data.strip()
                elif ev.type == "full_text" and ev.data.strip():
                    sentences.append(ev.data.strip())
                    partial = ""
                    text = " ".join(sentences)
                    if is_command(text) or (give_up and give_up(text)):
                        return text
                    if time.monotonic() - t0 > deadline_s:  # the flushed remainder: nothing more is coming
                        return text
    finally:
        stop.set()
    return " ".join(sentences + ([partial] if partial else []))
