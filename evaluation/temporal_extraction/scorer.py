"""Per-interval scoring + per-doc aggregation.

Supports multiple pairwise scoring functions (Jaccard composite, Gaussian,
Gaussian integrated, hard-overlap) and per-doc aggregation rules
(sum, max, top_k, log_sum).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from schema import GRANULARITY_ORDER

ScoreMode = Literal[
    "jaccard_composite", "gaussian", "gaussian_integrated", "hard_overlap"
]
AggMode = Literal["sum", "max", "top_k", "log_sum"]


@dataclass
class Interval:
    earliest_us: int
    latest_us: int
    best_us: int | None
    granularity: str


def overlaps(q: Interval, s: Interval) -> bool:
    return q.earliest_us < s.latest_us and s.earliest_us < q.latest_us


def _gran_score(q_gran: str, s_gran: str) -> float:
    gq = GRANULARITY_ORDER.get(q_gran, 3)
    gs = GRANULARITY_ORDER.get(s_gran, 3)
    gap = abs(gq - gs)
    return max(0.0, 1.0 - gap / 5.0)


def _center_us(iv: Interval) -> float:
    if iv.best_us is not None:
        return float(iv.best_us)
    return 0.5 * (iv.earliest_us + iv.latest_us)


def _sigma_us(iv: Interval) -> float:
    # σ = (latest − earliest) / 4 (half-bracket ≈ 2σ)
    span = iv.latest_us - iv.earliest_us
    return max(span / 4.0, 1.0)


def score_jaccard_composite(q: Interval, s: Interval) -> float:
    if not overlaps(q, s):
        return 0.0
    overlap = min(q.latest_us, s.latest_us) - max(q.earliest_us, s.earliest_us)
    union = max(q.latest_us, s.latest_us) - min(q.earliest_us, s.earliest_us)
    jaccard = overlap / union if union > 0 else 0.0
    if q.best_us is not None and s.best_us is not None:
        span = max(
            q.latest_us - q.earliest_us,
            s.latest_us - s.earliest_us,
            1_000_000,
        )
        proximity = max(0.0, 1.0 - abs(q.best_us - s.best_us) / span)
    else:
        proximity = 0.5
    return (
        0.5 * jaccard
        + 0.3 * proximity
        + 0.2 * _gran_score(q.granularity, s.granularity)
    )


def score_gaussian(q: Interval, s: Interval) -> float:
    """exp(−(μq−μs)² / (2(σq² + σs²))).

    Does NOT gate on bracket overlap — a Gaussian has infinite support —
    so tiny bracket mismatches still score > 0.
    """
    mu_q = _center_us(q)
    mu_s = _center_us(s)
    sigma_q = _sigma_us(q)
    sigma_s = _sigma_us(s)
    denom = 2.0 * (sigma_q * sigma_q + sigma_s * sigma_s)
    if denom <= 0:
        return 0.0
    d = mu_q - mu_s
    return math.exp(-(d * d) / denom)


def score_gaussian_integrated(q: Interval, s: Interval) -> float:
    """Normalized closed-form product-integral:

        ∫ N(t; μq, σq²) · N(t; μs, σs²) dt = N(μq; μs, σq² + σs²)

    Normalize by dividing by the max possible value (when μq = μs), i.e.
    1 / √(2π(σq² + σs²)). This normalization makes it identical to
    score_gaussian above — we expose it separately to confirm.
    """
    mu_q = _center_us(q)
    mu_s = _center_us(s)
    var = _sigma_us(q) ** 2 + _sigma_us(s) ** 2
    if var <= 0:
        return 0.0
    d = mu_q - mu_s
    raw = math.exp(-(d * d) / (2.0 * var)) / math.sqrt(2.0 * math.pi * var)
    max_val = 1.0 / math.sqrt(2.0 * math.pi * var)
    return raw / max_val if max_val > 0 else 0.0


def score_hard_overlap(q: Interval, s: Interval) -> float:
    return 1.0 if overlaps(q, s) else 0.0


def score_pair(
    q: Interval, s: Interval, mode: ScoreMode = "jaccard_composite"
) -> float:
    if mode == "jaccard_composite":
        return score_jaccard_composite(q, s)
    if mode == "gaussian":
        return score_gaussian(q, s)
    if mode == "gaussian_integrated":
        return score_gaussian_integrated(q, s)
    if mode == "hard_overlap":
        return score_hard_overlap(q, s)
    raise ValueError(f"unknown score mode: {mode}")


def aggregate_pair_scores(
    scores: list[float], mode: AggMode = "sum", top_k: int = 3
) -> float:
    """Aggregate a list of pairwise scores into a single doc score."""
    if not scores:
        return 0.0
    if mode == "sum":
        return sum(scores)
    if mode == "max":
        return max(scores)
    if mode == "top_k":
        return sum(sorted(scores, reverse=True)[:top_k])
    if mode == "log_sum":
        return math.log1p(sum(scores))
    raise ValueError(f"unknown agg mode: {mode}")


def aggregate(per_interval_hits: list[tuple[str, float]]) -> dict[str, float]:
    """Sum scores per doc_id (legacy helper for base pipeline)."""
    out: dict[str, float] = {}
    for doc_id, s in per_interval_hits:
        out[doc_id] = out.get(doc_id, 0.0) + s
    return out
