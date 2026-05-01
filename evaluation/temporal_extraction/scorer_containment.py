"""Containment-aware interval scorer.

Extends jaccard_composite with two width-asymmetric terms that fire when one
interval's BEST point lies within the other's [earliest, latest] bracket.
This fixes the A3 fuzzy-modifier case: a 1-day query can match a 2-year
doc bracket because the query's best point is contained within the doc.

The combined score takes max over:
    - jaccard_composite (current default)
    - q_in_s: fires when Q.best in [S.earliest, S.latest]
    - s_in_q: fires when S.best in [Q.earliest, Q.latest]

Each containment score decays with the width-ratio between the containing and
contained intervals so a massive bracket (e.g. "ancient history") doesn't
dominate everything it happens to contain.

Three decay formulas are exposed:
    - log2:  1 / (1 + log2(1 + ratio))      [gentle — default]
    - sqrt:  1 / (1 + sqrt(ratio))          [harsher]
    - dice:  min(1, smaller_span / larger_span)  [Dice-like]

Where ratio = max(wide_span / narrow_span, 1.0).
"""

from __future__ import annotations

import math
from typing import Literal

from scorer import Interval, score_jaccard_composite

DecayMode = Literal["log2", "sqrt", "dice"]


def _span(iv: Interval) -> float:
    return max(float(iv.latest_us - iv.earliest_us), 1.0)


def _best(iv: Interval) -> float:
    """Return a representative best-point (us). Falls back to midpoint."""
    if iv.best_us is not None:
        return float(iv.best_us)
    return 0.5 * (iv.earliest_us + iv.latest_us)


def _ratio_decay(wide_span: float, narrow_span: float, mode: DecayMode) -> float:
    if narrow_span <= 0:
        return 1.0
    ratio = max(wide_span / narrow_span, 1.0)
    if mode == "log2":
        return 1.0 / (1.0 + math.log2(1.0 + ratio))
    if mode == "sqrt":
        return 1.0 / (1.0 + math.sqrt(ratio))
    if mode == "dice":
        return min(1.0, narrow_span / wide_span)
    raise ValueError(f"unknown decay mode: {mode}")


def q_in_s_score(q: Interval, s: Interval, decay: DecayMode = "log2") -> float:
    """If Q.best lies within [S.earliest, S.latest], return a decayed match."""
    qb = _best(q)
    if not (s.earliest_us <= qb <= s.latest_us):
        return 0.0
    q_span = _span(q)
    s_span = _span(s)
    # Use wider span as the denominator's wide; contained span = min.
    wide = max(q_span, s_span)
    narrow = min(q_span, s_span)
    return _ratio_decay(wide, narrow, decay)


def s_in_q_score(q: Interval, s: Interval, decay: DecayMode = "log2") -> float:
    """Symmetric: if S.best lies within [Q.earliest, Q.latest]."""
    sb = _best(s)
    if not (q.earliest_us <= sb <= q.latest_us):
        return 0.0
    q_span = _span(q)
    s_span = _span(s)
    wide = max(q_span, s_span)
    narrow = min(q_span, s_span)
    return _ratio_decay(wide, narrow, decay)


def score_jaccard_with_containment(
    q: Interval, s: Interval, decay: DecayMode = "log2"
) -> float:
    """max(jaccard_composite, q_in_s_score, s_in_q_score)."""
    j = score_jaccard_composite(q, s)
    qs = q_in_s_score(q, s, decay=decay)
    sq = s_in_q_score(q, s, decay=decay)
    return max(j, qs, sq)


def score_pair_with_containment(
    q: Interval,
    s: Interval,
    mode: str = "jaccard_with_containment",
    decay: DecayMode = "log2",
) -> float:
    """Opt-in entry point. Preserves existing modes; adds containment-aware."""
    if mode == "jaccard_with_containment":
        return score_jaccard_with_containment(q, s, decay=decay)
    # Fall through to existing scorer.
    from scorer import score_pair

    return score_pair(q, s, mode=mode)  # type: ignore[arg-type]
