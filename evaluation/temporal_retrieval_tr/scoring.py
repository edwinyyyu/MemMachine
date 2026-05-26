"""Scoring: pair_overlap + flat-list final_score.

A query plan is a list of `IntervalSet` targets. Each target is a set of
allowed moments (one or more half-open intervals). The score for a doc
is the arithmetic mean of per-target bests:

    final_score = mean over targets of (max over anchors of pair_overlap)

The two semantic levels collapse into one:
- Intra-target OR (multiple intervals in one target) is handled by
  pair_overlap via the doc anchor being inside ANY of the target's
  intervals.
- Cross-target aggregation (multiple targets) is mean — a doc matching
  more targets scores higher (graded coverage).

The planner's only decision is one-multi-interval-target vs multiple-targets.
One target with multi-intervals = "set membership" (the user describes one
allowed region with holes). Multiple targets = "graded coverage" (the user
lists separate periods the doc should match). See planner.py.

Boundary cases:
- Empty plan (no temporal constraint) → 1.0 (semantic / rerank decides).
- Timeless doc (no doc anchors) → 1.0 (semantic / rerank decides).
"""
from __future__ import annotations

from .time_range import (
    SENTINEL_THRESHOLD,
    IntervalSet,
    intersect,
    is_empty,
    measure,
)


def pair_overlap(A: IntervalSet, B: IntervalSet) -> float:
    """Frac-min overlap. Returns value in [0, 1].

    Order matters:
    1. Empty intersection → 0.0 (HARD GATE).
    2. Both sides infinite measure → 1.0 (BOTH-INFINITE SHORTCUT).
    3. Otherwise → min(1.0, |A∩B| / min(|A|, |B|)).
    """
    inter = intersect(A, B)
    if is_empty(inter):
        return 0.0

    a_w = measure(A)
    b_w = measure(B)
    inter_w = measure(inter)

    a_inf = a_w >= SENTINEL_THRESHOLD
    b_inf = b_w >= SENTINEL_THRESHOLD
    if a_inf and b_inf:
        return 1.0

    denom = min(a_w, b_w)
    if denom <= 0:
        return 0.0
    frac = inter_w / denom
    if frac > 1.0:
        return 1.0
    return frac


def best_per_target(target: IntervalSet, anchors: list[IntervalSet]) -> float:
    """max over doc anchors of pair_overlap(target, anchor)."""
    best = 0.0
    for anchor in anchors:
        f = pair_overlap(target, anchor)
        if f > best:
            best = f
            if best >= 1.0:
                return 1.0
    return best


def final_score(
    targets: list[IntervalSet],
    anchors: list[IntervalSet],
) -> float:
    """Arithmetic mean of per-target bests → [0, 1].

    Mean ranks docs by TOTAL overlap mass with the query targets — the
    unique cross-target combiner with no distributional bias. The full
    power-mean family (exponent p) was swept 0.3..5 and rejected:
    p=1 (mean) is the joint R@1/R@5 optimum; p<1 regresses R@1 at the
    extreme, p>1 regresses R@5. noisy-OR likewise tied-or-lost. The
    benches are full-or-nothing overlaps so the combiner is not a
    lever; the controlled partial-overlap test confirms mean is the
    principled choice anyway.

    Boundary cases:
    - No targets (no temporal constraint) → 1.0.
    - No anchors (timeless doc) → 1.0; semantic / rerank decides.
    """
    if not targets:
        return 1.0
    if not anchors:
        return 1.0
    total = 0.0
    for t in targets:
        total += best_per_target(t, anchors)
    return total / len(targets)


def temporal_pass(targets: list[IntervalSet], anchors: list[IntervalSet]) -> bool:
    return final_score(targets, anchors) > 0.0
