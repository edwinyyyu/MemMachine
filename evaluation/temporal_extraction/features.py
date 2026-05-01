"""F1 — feature extraction for learned-scorer experiment.

For a (query, doc) pair, produce a dense feature vector combining:
  * interval-level structural features (Jaccard, proximity, granularity)
  * hand-crafted scorer scores (jaccard_composite, gaussian) as baselines
  * lightweight semantic and lexical features

All features are deterministic and reuse the existing scorer helpers. No
LLM calls and no new embeddings beyond what is already in
``cache/embedding_cache.json``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from schema import (
    GRANULARITY_ORDER,
    TimeExpression,
    to_us,
)
from scorer import (
    Interval,
    score_gaussian,
    score_jaccard_composite,
)

FEATURE_NAMES: list[str] = [
    "jaccard_bracket",
    "best_proximity_sec",
    "best_proximity_log",
    "granularity_gap",
    "granularity_compat",
    "semantic_cosine",
    "num_q_exprs",
    "num_d_exprs",
    "has_anchor_match",
    "has_recurrence_instance",
    "max_pair_score_jaccard",
    "max_pair_score_gaussian",
    "query_length_chars",
    "doc_length_chars",
]

# ---------------------------------------------------------------------------
# Interval helpers
# ---------------------------------------------------------------------------
_CAP_SECONDS = 10 * 365 * 24 * 3600  # 10 years


def flatten_to_intervals(
    tes: Sequence[TimeExpression],
    *,
    recurrence_horizon_years: int = 2,
) -> list[Interval]:
    """Flatten a list of TimeExpressions to a flat list of Intervals.

    For recurrences we expand instances within ``[now-10y, now+2y]`` exactly
    like ``eval.flatten_query_intervals``. Duration-only expressions contribute
    nothing (no anchor).
    """
    out: list[Interval] = []
    for te in tes:
        if te.kind == "instant" and te.instant:
            out.append(
                Interval(
                    earliest_us=to_us(te.instant.earliest),
                    latest_us=to_us(te.instant.latest),
                    best_us=to_us(te.instant.best) if te.instant.best else None,
                    granularity=te.instant.granularity,
                )
            )
        elif te.kind == "interval" and te.interval:
            g_start = te.interval.start.granularity
            g_end = te.interval.end.granularity
            g = (
                g_start
                if GRANULARITY_ORDER.get(g_start, 3) >= GRANULARITY_ORDER.get(g_end, 3)
                else g_end
            )
            best = te.interval.start.best or te.interval.start.earliest
            out.append(
                Interval(
                    earliest_us=to_us(te.interval.start.earliest),
                    latest_us=to_us(te.interval.end.latest),
                    best_us=to_us(best),
                    granularity=g,
                )
            )
        elif te.kind == "recurrence" and te.recurrence:
            # Expand via the live expander for parity with eval.py. Doing it
            # inline keeps the module independent of eval import order.
            try:
                from expander import expand  # local import
            except Exception:
                continue
            now = datetime.now(tz=timezone.utc)
            anchor = te.recurrence.dtstart.best or te.recurrence.dtstart.earliest
            start = min(
                now.replace(year=now.year - 10),
                anchor.replace(year=max(anchor.year - 1, 1)),
            )
            end = now.replace(year=now.year + recurrence_horizon_years)
            if te.recurrence.until is not None:
                end_bound = te.recurrence.until.latest or te.recurrence.until.earliest
                if end_bound < end:
                    end = end_bound
            try:
                expanded = list(expand(te.recurrence, start, end))
            except Exception:
                expanded = []
            for inst in expanded:
                out.append(
                    Interval(
                        earliest_us=to_us(inst.earliest),
                        latest_us=to_us(inst.latest),
                        best_us=to_us(inst.best) if inst.best else None,
                        granularity=inst.granularity,
                    )
                )
    return out


def best_bracket_jaccard(q_ivs: list[Interval], d_ivs: list[Interval]) -> float:
    best = 0.0
    for qi in q_ivs:
        for di in d_ivs:
            lo = max(qi.earliest_us, di.earliest_us)
            hi = min(qi.latest_us, di.latest_us)
            if hi <= lo:
                continue
            overlap = hi - lo
            union = max(qi.latest_us, di.latest_us) - min(
                qi.earliest_us, di.earliest_us
            )
            if union <= 0:
                continue
            j = overlap / union
            if j > best:
                best = j
    return best


def best_proximity_seconds(q_ivs: list[Interval], d_ivs: list[Interval]) -> float:
    best = _CAP_SECONDS
    for qi in q_ivs:
        q_center = (
            qi.best_us
            if qi.best_us is not None
            else (qi.earliest_us + qi.latest_us) // 2
        )
        for di in d_ivs:
            d_center = (
                di.best_us
                if di.best_us is not None
                else (di.earliest_us + di.latest_us) // 2
            )
            delta = abs(q_center - d_center) / 1_000_000.0
            if delta < best:
                best = delta
    return min(best, _CAP_SECONDS)


def best_granularity_gap(q_ivs: list[Interval], d_ivs: list[Interval]) -> int:
    if not q_ivs or not d_ivs:
        return 5
    best = 9
    for qi in q_ivs:
        gq = GRANULARITY_ORDER.get(qi.granularity, 3)
        for di in d_ivs:
            gs = GRANULARITY_ORDER.get(di.granularity, 3)
            gap = abs(gq - gs)
            if gap < best:
                best = gap
    return best


def has_anchor_match(q_ivs: list[Interval], d_ref: datetime | None) -> int:
    """1 if doc's reference_time (utterance anchor) overlaps any query bracket."""
    if d_ref is None or not q_ivs:
        return 0
    ref_us = to_us(d_ref)
    if ref_us is None:
        return 0
    for qi in q_ivs:
        if qi.earliest_us <= ref_us < qi.latest_us:
            return 1
    return 0


def has_recurrence_instance(d_tes: Sequence[TimeExpression]) -> int:
    for te in d_tes:
        if te.kind == "recurrence" and te.recurrence is not None:
            return 1
    return 0


def max_hand_pair_score(
    q_ivs: list[Interval],
    d_ivs: list[Interval],
    mode: str,
) -> float:
    if not q_ivs or not d_ivs:
        return 0.0
    fn = score_jaccard_composite if mode == "jaccard" else score_gaussian
    best = 0.0
    for qi in q_ivs:
        for di in d_ivs:
            s = fn(qi, di)
            if s > best:
                best = s
    return best


def cosine(a, b) -> float:

    na = float((a * a).sum()) ** 0.5
    nb = float((b * b).sum()) ** 0.5
    denom = na * nb
    if denom <= 0:
        return 0.0
    return float((a * b).sum() / denom)


@dataclass
class PairContext:
    q_text: str
    d_text: str
    q_tes: Sequence[TimeExpression]
    d_tes: Sequence[TimeExpression]
    q_emb: object | None = None  # numpy array
    d_emb: object | None = None
    d_ref_time: datetime | None = None


def extract_features(ctx: PairContext) -> list[float]:
    q_ivs = flatten_to_intervals(ctx.q_tes)
    d_ivs = flatten_to_intervals(ctx.d_tes)

    bracket_j = best_bracket_jaccard(q_ivs, d_ivs)
    prox_sec = best_proximity_seconds(q_ivs, d_ivs)
    prox_log = math.log1p(prox_sec)
    gran_gap = best_granularity_gap(q_ivs, d_ivs)
    gran_compat = max(0.0, 1.0 - gran_gap / 5.0)

    if ctx.q_emb is not None and ctx.d_emb is not None:
        sem_cos = cosine(ctx.q_emb, ctx.d_emb)
    else:
        sem_cos = 0.0

    anchor_match = has_anchor_match(q_ivs, ctx.d_ref_time)
    recur_inst = has_recurrence_instance(ctx.d_tes)
    max_j = max_hand_pair_score(q_ivs, d_ivs, "jaccard")
    max_g = max_hand_pair_score(q_ivs, d_ivs, "gaussian")

    return [
        float(bracket_j),
        float(prox_sec),
        float(prox_log),
        float(gran_gap),
        float(gran_compat),
        float(sem_cos),
        float(len(ctx.q_tes)),
        float(len(ctx.d_tes)),
        float(anchor_match),
        float(recur_inst),
        float(max_j),
        float(max_g),
        float(len(ctx.q_text)),
        float(len(ctx.d_text)),
    ]


def feature_names() -> list[str]:
    return list(FEATURE_NAMES)
