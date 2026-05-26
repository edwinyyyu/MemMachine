"""V7 scoring: pair_overlap + flat-list final_score.

A query plan is a list of `TimeRange` refs. Each ref is a set of allowed
moments (one or more half-open intervals). The score for a doc is the
arithmetic mean of per-ref bests:

    final_score = mean over refs of (max over doc-refs of pair_overlap)

The two semantic levels collapse into one:
- Intra-ref OR (multiple intervals in one ref) is handled by pair_overlap
  via the doc anchor being inside ANY of the ref's intervals.
- Cross-ref aggregation (multiple refs) is mean — a doc matching more
  refs scores higher (graded coverage).

The planner's only decision is one-multi-interval-ref vs multiple-refs.
One ref with multi-intervals = "set membership" (the user describes one
allowed region with holes). Multiple refs = "graded coverage" (the user
lists separate periods the doc should match). See planner_direct.py.

Boundary cases:
- Empty plan (no temporal constraint) → 1.0 (semantic / rerank decides).
- Timeless doc (no doc refs) → 1.0 (semantic / rerank decides).
"""
from __future__ import annotations

from .time_range import (
    SENTINEL_THRESHOLD,
    TimeRange,
    intersect,
    is_empty,
    measure,
)


def pair_overlap(A: TimeRange, B: TimeRange) -> float:
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


def best_per_ref(ref: TimeRange, doc_refs: list[TimeRange]) -> float:
    """max over doc-refs of pair_overlap(ref, dref)."""
    best = 0.0
    for dref in doc_refs:
        f = pair_overlap(ref, dref)
        if f > best:
            best = f
            if best >= 1.0:
                return 1.0
    return best


def final_score(
    refs: list[TimeRange],
    doc_refs: list[TimeRange],
) -> float:
    """Arithmetic mean of per-ref bests → [0, 1].

    Mean ranks docs by TOTAL overlap mass with the query refs — the
    unique cross-ref combiner with no distributional bias. The full
    power-mean family (exponent p) was swept 0.3..5 and rejected:
    p=1 (mean) is the joint R@1/R@5 optimum; p<1 regresses R@1 at the
    extreme, p>1 regresses R@5. noisy-OR likewise tied-or-lost. The
    benches are full-or-nothing overlaps so the combiner is not a
    lever; the controlled partial-overlap test confirms mean is the
    principled choice anyway. See research/_power_mean_ab.py,
    research/_noisy_or_ab.py, research/_partial_overlap_test.py.

    Boundary cases:
    - No refs (no temporal constraint) → 1.0.
    - No doc refs (timeless doc) → 1.0; semantic / rerank decides.
    """
    if not refs:
        return 1.0
    if not doc_refs:
        return 1.0
    total = 0.0
    for r in refs:
        total += best_per_ref(r, doc_refs)
    return total / len(refs)


def temporal_pass(refs: list[TimeRange], doc_refs: list[TimeRange]) -> bool:
    return final_score(refs, doc_refs) > 0.0
