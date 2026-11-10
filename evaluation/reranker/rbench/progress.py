"""Progress lines for long steps: done / total, rate, elapsed and time left.

Reranking prints its own (rerankers.score); ingest, depth measurement and simulation use this.
A line is printed at most every EVERY seconds, and at the end.
"""

from __future__ import annotations

import time

EVERY = 30.0  # seconds between progress lines


def _clock(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m" if seconds >= 3600 else f"{seconds // 60}m{seconds % 60:02d}s"


class Progress:
    def __init__(self, label: str, total: int, unit: str = "") -> None:
        self.label, self.total, self.unit = label, total, unit
        self.done = 0
        self.start = self.last = time.monotonic()

    def step(self, n: int = 1, note: str = "") -> None:
        self.done += n
        now = time.monotonic()
        if now - self.last >= EVERY or self.done >= self.total:
            self.last = now
            elapsed = now - self.start
            rate = self.done / elapsed if elapsed > 0 else 0.0
            left = (self.total - self.done) / rate if rate > 0 else float("inf")
            print(f"{self.label}: {self.done}/{self.total}{(' ' + self.unit) if self.unit else ''}, "
                  f"{elapsed and rate * 60:.1f}/min, elapsed {_clock(elapsed)}"
                  + (f", about {_clock(left)} left" if self.done < self.total and left != float("inf") else "")
                  + (f" ({note})" if note else ""), flush=True)


def elapsed(start: float) -> str:
    return _clock(time.monotonic() - start)
