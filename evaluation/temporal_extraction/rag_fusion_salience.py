"""Salience-modulated score-blend fusion.

Wraps the same dispersion-gated CV-scaled blend as `rag_fusion.score_blend`
but applies a per-doc, per-channel salience multiplier BEFORE min-max
normalization. The intent is to silence channels for docs whose intrinsic
salience for that channel is near zero (e.g., a content-only doc gets T*0
on the temporal channel, so its T-score never lifts it on temporal queries).

Salience comes from `salience_extractor.SalienceExtractor` and is a dict
{doc_id: {"S": float, "T": float, "L": float, "E": float}} with values
summing to ~1.0.

`channel_to_key` maps each retriever's name to a salience-vector key:
    {"T": "T", "S": "S", "L": "L", "E": "E"} by default.
A retriever named differently can be passed in `channel_to_key` so the
caller controls the mapping (we never fabricate a salience axis for an
unmapped channel — it gets multiplier 1.0).

API mirrors rag_fusion.score_blend with one extra arg.
"""

from __future__ import annotations

import math


def _channel_cv(vals: list[float]) -> tuple[float, float, float, float]:
    lo, hi = min(vals), max(vals)
    span = hi - lo
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    std = math.sqrt(var)
    scale = max(abs(mean), 1e-9)
    cv = std / scale
    return lo, hi, cv, span


def _min_max_normalize(
    scored: dict[str, float],
    top_k: int | None = None,
    dispersion_min_abs: float = 1e-6,
) -> tuple[dict[str, float], float] | None:
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


DEFAULT_CHANNEL_TO_KEY = {"T": "T", "S": "S", "L": "L", "E": "E"}


def score_blend_with_salience(
    per_retriever_scores: dict[str, dict[str, float]],
    weights: dict[str, float],
    doc_salience: dict[str, dict[str, float]],
    *,
    channel_to_key: dict[str, str] | None = None,
    salience_floor: float = 0.0,
    salience_temperature: float = 1.0,
    top_k_per: int = 40,
    dispersion_min_abs: float = 1e-6,
    dispersion_cv_ref: float = 0.20,
) -> list[tuple[str, float]]:
    """Salience-modulated CV-scaled score-blend fusion.

    Per (doc, channel): effective_raw = salience[doc][key]^salience_temperature
                                       * raw_score[doc]
    Then dispersion-gate + min-max normalize + CV-scaled global weight blend.

    salience_floor: minimum multiplier; if the doc has 0.0 salience for a
        channel, we still let it pass through with this floor (default 0.0
        = full silencing). 0.05 keeps a token contribution.
    salience_temperature: 1.0 = linear modulation. 0.5 sqrt-flattens
        (less aggressive). 2.0 sharpens. Default 1.0.
    """
    if channel_to_key is None:
        channel_to_key = DEFAULT_CHANNEL_TO_KEY

    # Step 1: apply per-doc salience modulation to raw scores.
    modulated: dict[str, dict[str, float]] = {}
    for chan, scored in per_retriever_scores.items():
        key = channel_to_key.get(chan)
        new_scored: dict[str, float] = {}
        for did, raw in scored.items():
            if key is None:
                mult = 1.0
            else:
                sal = doc_salience.get(did, {}).get(key, 0.25)
                mult = max(salience_floor, sal)
                if salience_temperature != 1.0:
                    mult = mult**salience_temperature
            new_scored[did] = raw * mult
        modulated[chan] = new_scored

    # Step 2: dispersion-gate + normalize per channel.
    normed: dict[str, dict[str, float]] = {}
    cv_by_channel: dict[str, float] = {}
    for chan, scored in modulated.items():
        result = _min_max_normalize(
            scored, top_k=top_k_per, dispersion_min_abs=dispersion_min_abs
        )
        if result is None:
            continue
        n_dict, cv = result
        normed[chan] = n_dict
        cv_by_channel[chan] = cv

    # Step 3: CV-scaled effective weights.
    effective_weights: dict[str, float] = {}
    for chan in normed:
        user_w = weights.get(chan, 0.0)
        cv = cv_by_channel[chan]
        scale = min(1.0, cv / dispersion_cv_ref) if dispersion_cv_ref > 0 else 1.0
        effective_weights[chan] = user_w * scale

    orig_total = sum(weights.get(chan, 0.0) for chan in per_retriever_scores)
    eff_total = sum(effective_weights.values())
    if eff_total > 0 and orig_total > 0:
        for chan in effective_weights:
            effective_weights[chan] *= orig_total / eff_total

    all_ids: set[str] = set()
    for chan in normed:
        all_ids |= set(normed[chan].keys())

    combined: dict[str, float] = {}
    for d in all_ids:
        s = 0.0
        for chan, w in effective_weights.items():
            s += w * normed.get(chan, {}).get(d, 0.0)
        combined[d] = s
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)


def score_blend_with_salience_post(
    per_retriever_scores: dict[str, dict[str, float]],
    weights: dict[str, float],
    doc_salience: dict[str, dict[str, float]],
    *,
    channel_to_key: dict[str, str] | None = None,
    salience_floor: float = 0.05,
    salience_temperature: float = 1.0,
    top_k_per: int = 40,
    dispersion_min_abs: float = 1e-6,
    dispersion_cv_ref: float = 0.20,
) -> list[tuple[str, float]]:
    """Variant: apply salience AFTER normalization, at the weight level.

    Decouples salience from CV gate and channel statistics. Pipeline:

    1. Min-max normalize each channel's raw scores (CV gate fires here on
       undistorted signal).
    2. CV-scale global weights as in plain `score_blend`.
    3. At blend time, multiply each (doc, channel) contribution by the
       doc's salience for that channel — i.e., per-doc effective weight.

    This avoids the failure mode where multiplying raw scores by salience
    distorts the channel's distribution and interferes with the gate. A
    doc with low salience for a channel simply contributes less of that
    channel's normalized signal — no statistical side effects.

    salience_floor (default 0.05) prevents complete silencing — even a doc
    with ~0 salience still gets a token contribution, so the channel can
    pull it up if it's a strong match.
    """
    if channel_to_key is None:
        channel_to_key = DEFAULT_CHANNEL_TO_KEY

    # Step 1: dispersion-gate + normalize per channel ON RAW SCORES
    normed: dict[str, dict[str, float]] = {}
    cv_by_channel: dict[str, float] = {}
    for chan, scored in per_retriever_scores.items():
        result = _min_max_normalize(
            scored, top_k=top_k_per, dispersion_min_abs=dispersion_min_abs
        )
        if result is None:
            continue
        n_dict, cv = result
        normed[chan] = n_dict
        cv_by_channel[chan] = cv

    # Step 2: CV-scaled effective weights (same as plain score_blend)
    effective_weights: dict[str, float] = {}
    for chan in normed:
        user_w = weights.get(chan, 0.0)
        cv = cv_by_channel[chan]
        scale = min(1.0, cv / dispersion_cv_ref) if dispersion_cv_ref > 0 else 1.0
        effective_weights[chan] = user_w * scale

    orig_total = sum(weights.get(chan, 0.0) for chan in per_retriever_scores)
    eff_total = sum(effective_weights.values())
    if eff_total > 0 and orig_total > 0:
        for chan in effective_weights:
            effective_weights[chan] *= orig_total / eff_total

    all_ids: set[str] = set()
    for chan in normed:
        all_ids |= set(normed[chan].keys())

    # Step 3: blend with per-doc salience as weight modulator
    combined: dict[str, float] = {}
    for d in all_ids:
        s = 0.0
        for chan, w in effective_weights.items():
            key = channel_to_key.get(chan)
            sal = doc_salience.get(d, {}).get(key, 0.25) if key else 1.0
            sal_eff = max(salience_floor, sal)
            if salience_temperature != 1.0:
                sal_eff = sal_eff**salience_temperature
            s += w * sal_eff * normed.get(chan, {}).get(d, 0.0)
        combined[d] = s
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)
