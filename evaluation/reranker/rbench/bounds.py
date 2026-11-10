"""True bounds: the cheapest reading of a pool that any ranker could achieve, per reading.

The user's request (2026-11-09: "yes if no downside"): free, from the data already simulated;
every line drawn as a bound is a valid bound (no simulated ranker beats it on any question,
checked over lme-test by tests/test_bounds.py); the real reader cost model (reading.py:
overlaps once, headers); per reading, over the same top-L pool every ranker at L reorders.

How a ranker reads (reading.admission): contexts in its order, each admitted whole (nearest
its seed first), so what it has read when an event happens is some whole contexts plus a
prefix of one more, rendered with every pool context in the query result.

Seeds alone (expand_context 0): every context is one seed, its own component and its own run,
so a set S of seeds costs  sum(inner + run - NEWLINE) + SEPARATOR x (|S| - 1):  additive. The
bounds are exact optimums over subsets of the pool:
  first        the cheapest evidence span read alone (all its segments in the pool)
  first_touch  the cheapest evidence segment in the pool
  every        all segments of the cheapest answer
  every_touch  per span of an answer, its cheapest segment; spans sharing segments make the
               choice a set cover, and then the most expensive span's cheapest segment is used
               instead (valid, not exact)
  share        spans in cheapest-first order (picking the cheapest k spans is optimal for
               additive costs): the k-th turn is complete at the sum of the k cheapest
  share_touch  the same with each span's cheapest segment
With windows (expand_context > 0) the rendered cost is not monotone in what is read: a
segment that joins two components can save a header, so an exact search would have to
consider contexts that hold no evidence. The bound there is the evidence's own text tokens
plus the cheapest single header in the pool: valid for every ranker, but loose (no rendering,
no other headers, nothing between). Rows record which kind each is (`exact`).

A span with a segment outside the pool (the top L seeds, or the union of their windows) cannot
be read in full: inf. Touching it needs one of its segments in the pool.
"""

from __future__ import annotations

import math

import numpy as np

from rbench.metrics import INF, Gold
from rbench.reading import constants


def _alternatives(g: Gold) -> list[list[list[int]]]:
    """Each answer's spans as segment-id lists."""
    return [[[int(c) for c, _ in span] for span in alt] for alt in g.alternatives]


def bound(g: Gold, pool: set[int], inner: np.ndarray, run: np.ndarray, expand: int) -> dict:
    """{first, every, first_touch, every_touch: cost; share_full, share_touch: costs whose
    readings.hit at any budget is the best share any ranker reaches; exact: bool} for one pool."""
    _, newline, separator = constants()
    alts = _alternatives(g)
    if expand == 0:  # a seed read alone is its own component and run
        unit = {s: int(inner[s]) + int(run[s]) - newline for s in pool}
        floor, sep = 0.0, separator
    else:  # text only; at least one header (the pool's cheapest) for the whole reading
        unit = {s: int(inner[s]) for s in pool}
        floor, sep = float(min((int(run[s]) for s in pool), default=0) - newline), 0

    def cost(segments: set[int]) -> float:
        return sum(unit[s] for s in segments) + sep * max(0, len(segments) - 1)

    exact = expand == 0
    full = [[cost(set(span)) if span and all(s in pool for s in span) else INF for span in alt] for alt in alts]
    touch = [[min((unit[s] for s in span if s in pool), default=INF) for span in alt] for alt in alts]
    first = min((c for alt in full for c in alt), default=INF) + floor
    first_touch = min((c for alt in touch for c in alt), default=INF) + floor
    every = min((cost({s for span in alt for s in span}) if all(np.isfinite(f)) else INF
                 for alt, f in zip(alts, full, strict=True)), default=INF) + floor
    every_touch = INF
    curves_full, curves_touch = [], []
    for alt, f, t in zip(alts, full, touch, strict=True):
        disjoint = len({s for span in alt for s in span}) == sum(len(span) for span in alt)
        if not disjoint:
            exact = False  # spans share segments: a set cover; use valid floors instead
        if all(np.isfinite(t)):
            value = sum(t) + sep * max(0, len(t) - 1) if disjoint else max(t)
            every_touch = min(every_touch, value + floor)
        # spans sharing segments may cost less together than apart: completing k of them costs
        # at least the k-th cheapest span alone (a valid floor)
        curves_full.append(_completions(sorted(f), sep, floor) if disjoint else [c + floor for c in sorted(f)])
        curves_touch.append(_completions(sorted(t), sep, floor) if disjoint
                            else [min(t) + floor if np.isfinite(min(t)) else INF] * len(t))
    return {"first": first, "every": every, "first_touch": first_touch, "every_touch": every_touch,
            "share_full": _best(curves_full), "share_touch": _best(curves_touch), "exact": exact}


def _completions(costs: list[float], separator: float, floor: float) -> list[float]:
    """The cost at which the 1st, 2nd, ... span of an answer is complete when its spans are
    read cheapest first (separators between seeds; the windows' header floor once)."""
    out, total = [], 0.0
    for k, c in enumerate(costs):
        total = total + c + (separator if k else 0.0) if np.isfinite(c) and np.isfinite(total) else INF
        out.append(total + floor if np.isfinite(total) else INF)
    return out


def _best(curves: list[list[float]]) -> list[float]:
    """One cost list whose share at every budget is the best of the answers' (a ranker's share
    follows one answer, so the bound is their pointwise maximum): over a common denominator N,
    the j-th entry is the soonest any answer has at least j/N of its spans."""
    curves = [c for c in curves if c]
    if not curves:
        return []
    n = math.lcm(*(len(c) for c in curves))
    return [min(c[math.ceil(j * len(c) / n) - 1] for c in curves) for j in range(1, n + 1)]
