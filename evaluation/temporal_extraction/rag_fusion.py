"""Fusion primitives for multi-retriever RAG.

- rrf: Reciprocal Rank Fusion (k=60 classical)
- min_max_normalize: per-retriever normalization to [0, 1] within top-K
- score_blend: linear combination of normalized scores
"""

from __future__ import annotations


def rrf(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion across multiple ranked lists.

    Each list is an ordered [doc_id]. Docs not in a list get 0 contribution
    from that list. Output: ranked [(doc_id, score)] desc by score.
    """
    scores: dict[str, float] = {}
    for rl in ranked_lists:
        for i, d in enumerate(rl):
            scores[d] = scores.get(d, 0.0) + 1.0 / (k + i + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _channel_cv(vals: list[float]) -> tuple[float, float, float, float]:
    """Compute (lo, hi, cv, span) for a list of scores.

    cv = std / |mean| (coefficient of variation), the measure of how
    informative this channel is on the current query's candidate pool.
    """
    import math

    lo, hi = min(vals), max(vals)
    span = hi - lo
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    std = math.sqrt(var)
    scale = max(abs(mean), 1e-9)
    cv = std / scale
    return lo, hi, cv, span


def min_max_normalize(
    scored: dict[str, float],
    top_k: int | None = None,
    dispersion_min_abs: float = 1e-6,
) -> tuple[dict[str, float], float] | None:
    """Normalize scores to [0, 1] via min-max and return (normed, cv).

    Returns None when span is below numerical-precision threshold
    (`dispersion_min_abs`) — channel has literally no signal.

    Otherwise returns the normalized dict AND the channel's coefficient
    of variation (CV = std / |mean|), which `score_blend` uses to
    dynamically scale the channel's contribution in the fusion. Higher
    CV = more informative channel = larger contribution.
    """
    if not scored:
        return None
    items = sorted(scored.items(), key=lambda x: x[1], reverse=True)
    if top_k is not None:
        items = items[:top_k]
    vals = [v for _, v in items]
    if not vals:
        return None
    lo, hi, cv, span = _channel_cv(vals)

    if span < dispersion_min_abs:
        return None

    return {d: (v - lo) / span for d, v in items}, cv


def score_blend(
    per_retriever_scores: dict[str, dict[str, float]],
    weights: dict[str, float],
    top_k_per: int = 40,
    dispersion_min_abs: float = 1e-6,
    dispersion_cv_ref: float = 0.20,
) -> list[tuple[str, float]]:
    """Linear blend of per-retriever scores with CV-scaled channel weights.

    Each channel's effective weight is scaled by its dispersion (CV =
    std/|mean|) on the current query's candidate pool. A channel with low
    CV (uninformative — scores cluster around a single value) gets a
    proportionally smaller contribution; a channel with CV ≥ `cv_ref` keeps
    its full user-specified weight. This avoids the "stretched-noise from
    a flat channel competing with a real signal" failure mode seen on the
    dense-time-cluster benchmark, without the cliff of a hard gate.

    Channels with span below `dispersion_min_abs` (truly zero variance)
    still get dropped, since min-max normalization is undefined.

    After CV-scaling, surviving weights are renormalized so they sum to
    the original total — preserving the caller's intended mixing scale.

    Tuning:
    - `dispersion_cv_ref` is the CV at which a channel reaches full weight.
      Default 0.50 was set so that dense-cluster T (CV ≈ 0.11) gets ~22%
      of full weight while TempReason T (CV > 1.0) gets full weight.
    """
    normed: dict[str, dict[str, float]] = {}
    cv_by_channel: dict[str, float] = {}
    for name, scored in per_retriever_scores.items():
        result = min_max_normalize(
            scored,
            top_k=top_k_per,
            dispersion_min_abs=dispersion_min_abs,
        )
        if result is None:
            continue  # span below numerical-precision threshold
        n_dict, cv = result
        normed[name] = n_dict
        cv_by_channel[name] = cv

    # CV-scaled effective weights — linear ramp from 0 to user weight as CV
    # grows from 0 to cv_ref, capped at user weight beyond cv_ref.
    effective_weights: dict[str, float] = {}
    for name in normed:
        user_w = weights.get(name, 0.0)
        cv = cv_by_channel[name]
        scale = min(1.0, cv / dispersion_cv_ref) if dispersion_cv_ref > 0 else 1.0
        effective_weights[name] = user_w * scale

    # Renormalize so the active channels' weights sum to the original total.
    orig_total = sum(weights.get(name, 0.0) for name in per_retriever_scores)
    eff_total = sum(effective_weights.values())
    if eff_total > 0 and orig_total > 0:
        for name in effective_weights:
            effective_weights[name] *= orig_total / eff_total

    all_ids: set[str] = set()
    for name in normed:
        all_ids |= set(normed[name].keys())

    combined: dict[str, float] = {}
    for d in all_ids:
        s = 0.0
        for name, w in effective_weights.items():
            s += w * normed.get(name, {}).get(d, 0.0)
        combined[d] = s
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)


def scores_to_ranked(scored: dict[str, float]) -> list[str]:
    return [d for d, _ in sorted(scored.items(), key=lambda x: x[1], reverse=True)]
