"""
Countdown timers. Deliberately has no pygame dependency - it's plain
wall-clock math, so it can be unit-tested and reasoned about on its own.
"""

import time


class QuestionTimer:
    """Per-question countdown."""

    def __init__(self):
        self._start = None
        self._limit = 0.0

    def start(self, limit_seconds):
        self._start = time.time()
        self._limit = limit_seconds

    def elapsed(self):
        return (time.time() - self._start) if self._start is not None else 0.0

    def remaining(self):
        return max(0.0, self._limit - self.elapsed())

    def expired(self):
        return self.remaining() <= 0.0


class SessionTimer:
    """Overall session countdown. duration_seconds=None means endless."""

    def __init__(self, duration_seconds=None):
        self._start = time.time()
        self.duration = duration_seconds

    def elapsed(self):
        return time.time() - self._start

    def remaining(self):
        if self.duration is None:
            return None
        return max(0.0, self.duration - self.elapsed())

    def is_over(self):
        return self.duration is not None and self.remaining() <= 0.0
