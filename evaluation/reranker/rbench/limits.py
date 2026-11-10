"""Concurrency and retries for hosted models: as fast as DeepInfra allows, at the same cost.

DeepInfra limits concurrent requests per model (200 by default: docs.deepinfra.com/account/
rate-limits, 2026-11-09), answering 429 "Rate limited" past it, and sometimes under heavy load
below it. Price is per token, so more requests in flight cost nothing extra. Each hosted
model (every reranker and embedder on DeepInfra separately) gets one Adaptive limit for the
process: it starts at START, grows while requests succeed (x1.5 per window of successes until
the first 429, then +2 per window), is cut x0.7 on a 429 (at most once per window, so one
burst counts once), and never passes CAP (below DeepInfra's 200, for headroom). The local
llama-server keeps its fixed limit (embedding.EmbedderSpec.in_flight): it is shared.

Retries (`attempts`): a 429, a 5xx, a timeout or a transport error is retried after a backoff
that doubles to BACKOFF_MAX, for up to PATIENCE per request; then the run stops with GaveUp
("rerun to resume; finished work is kept"). Out of credits (credits.py) and non-retryable
errors stop at once.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field

START, CAP, FLOOR = 16, 150, 2
BACKOFF_MAX = 300.0  # seconds between attempts, at most
PATIENCE = 3600.0  # seconds one request may keep failing before the run stops


class GaveUp(RuntimeError):
    """A request kept failing (retryably) for PATIENCE: DeepInfra is down or overloaded."""

    def __init__(self, what: str, last: str) -> None:
        super().__init__(f"DeepInfra has not answered {what} for {PATIENCE / 60:.0f} min (last: {last}): rerun to "
                         "resume; finished work is kept")


@dataclass
class Adaptive:
    """One hosted model's concurrency limit (state only; Gate applies it inside an event loop)."""

    name: str
    limit: float = START
    cap: int = CAP
    adaptive: bool = True
    throttled_once: bool = False  # a 429 seen: additive growth from now on
    window: int = 0  # successes since the last change
    last_cut: float = field(default=0.0)
    peak: float = START
    cuts: int = 0  # times a 429 cut the limit

    def success(self) -> None:
        if not self.adaptive:
            return
        self.window += 1
        if self.window >= max(1, int(self.limit)):
            self.limit = min(self.cap, self.limit + 2 if self.throttled_once else self.limit * 1.5)
            self.peak = max(self.peak, self.limit)
            self.window = 0

    def throttled(self) -> None:
        if not self.adaptive:
            return
        self.throttled_once = True
        now = time.monotonic()
        if now - self.last_cut > 2.0:  # one burst of 429s is one signal
            self.limit = max(FLOOR, self.limit * 0.7)
            self.last_cut, self.window = now, 0
            self.cuts += 1


_LIMITS: dict[str, Adaptive] = {}


def limit_for(name: str, start: int = START, adaptive: bool = True) -> Adaptive:
    """The process-wide limit of one hosted model (created on first use)."""
    if name not in _LIMITS:
        _LIMITS[name] = Adaptive(name, limit=start, cap=max(start, CAP) if adaptive else start, adaptive=adaptive,
                                 peak=start)
    return _LIMITS[name]


class Gate:
    """An Adaptive limit applied within one event loop: `async with gate:` waits for a slot."""

    def __init__(self, state: Adaptive) -> None:
        self.state = state
        self.active = 0
        self._cond = asyncio.Condition()

    async def __aenter__(self) -> Gate:
        async with self._cond:
            await self._cond.wait_for(lambda: self.active < max(1, int(self.state.limit)))
            self.active += 1
        return self

    async def __aexit__(self, *exc) -> None:
        async with self._cond:
            self.active -= 1
            self._cond.notify_all()

    async def wake(self) -> None:
        """Let waiters re-check a limit that grew."""
        async with self._cond:
            self._cond.notify_all()


def backoff(attempt: int) -> float:
    """Seconds before retry number `attempt` (1, 2, ...): doubling with jitter, at most BACKOFF_MAX."""
    return min(BACKOFF_MAX, 2.0 ** min(attempt, 12)) * (0.75 + 0.5 * random.random())
