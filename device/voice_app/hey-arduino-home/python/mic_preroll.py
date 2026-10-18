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
"""Microphone that replays audio from just before the wake word, then continues live.

Keyword spotting fires ~0.5-1 s after "Hey Arduino" ends, so an ASR session started at that moment misses a command
spoken in the same breath ("Hey Arduino turn off the light"). This keeps a rolling buffer from its own shared tap
on the USB mic and, when started, first yields the audio since `mark()` minus PREROLL_S.
"""

import collections
import queue
import threading
import time

import numpy as np
from arduino.app_peripherals.microphone import BaseMicrophone, Microphone

PREROLL_S = 1.5
HISTORY_S = 8.0


class PrerollMicrophone(BaseMicrophone):
    def __init__(self, source: Microphone):
        self.source = source
        super().__init__(sample_rate=source.sample_rate, channels=source.channels, format=source.format,
                         buffer_size=source.buffer_size, auto_reconnect=False)
        self.name = f"Preroll({source.name})"
        self._history: collections.deque[tuple[float, np.ndarray]] = collections.deque()
        self._history_lock = threading.Lock()
        self._live: queue.Queue[np.ndarray] | None = None
        self._mark_t = 0.0
        self._preroll_s = PREROLL_S
        self._consumed_t = 0.0
        self.source.start()
        threading.Thread(target=self._tap, daemon=True).start()

    def mark(self, preroll_s: float = PREROLL_S):
        """Call when the wake word is detected; the next start() replays from preroll_s before this moment."""
        self._mark_t = time.monotonic()
        self._preroll_s = preroll_s

    def _tap(self):
        for chunk in self.source.stream():
            now = time.monotonic()
            with self._history_lock:
                self._history.append((now, chunk))
                while self._history and now - self._history[0][0] > HISTORY_S:
                    self._history.popleft()
                if self._live is not None:
                    self._live.put(chunk)

    def _open_microphone(self) -> None:
        # Never replay audio the previous session already transcribed (a wake word heard mid-command).
        since = max((self._mark_t or time.monotonic()) - self._preroll_s, self._consumed_t)
        live: queue.Queue[np.ndarray] = queue.Queue()
        with self._history_lock:
            for t, chunk in self._history:
                if t >= since:
                    live.put(chunk)
            self._live = live

    def _close_microphone(self) -> None:
        with self._history_lock:
            self._live = None
        self._consumed_t = time.monotonic()
        self._mark_t = 0.0

    def _read_audio(self) -> np.ndarray | None:
        live = self._live
        if live is None:
            return None
        try:
            return live.get(timeout=0.05)
        except queue.Empty:
            return None
