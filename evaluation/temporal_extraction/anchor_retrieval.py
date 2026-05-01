"""Retrieval that queries both referent intervals and utterance anchors,
then merges doc-level scores.

Exposed `retrieve(...)` takes a flattened query-interval list plus modes
for how to combine anchor/referent scores (R1 union / R2 aggregation
variants).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

from anchor_store import UtteranceAnchorStore
from scorer import Interval, score_pair
from store import IntervalStore

# Which indexes to consult.
SourceMode = Literal[
    "anchor_only",  # R1a
    "referent_only",  # R1b (current system)
    "union",  # R1c
]

# How to combine anchor_score with referent_scores within union mode.
AggMode = Literal[
    "sum",  # R2a: anchor_score + Σ ref_scores
    "max",  # R2b: max(anchor_score, best_ref_score)
    "weighted",  # R2c/d: α * best_ref_score + β * anchor_score
    "sum_weighted",  # α * Σ ref_scores + β * anchor_score
    "routed",  # R2e: choose α,β based on query intent
]

Intent = Literal["utterance", "referent", "ambiguous"]


def _score_referents_per_doc(
    store: IntervalStore,
    q_intervals: list[Interval],
) -> tuple[dict[str, float], dict[str, float]]:
    """Return (sum_per_doc, max_per_doc) referent scores."""
    sum_per_doc: dict[str, float] = defaultdict(float)
    max_per_doc: dict[str, float] = defaultdict(float)
    for qi in q_intervals:
        rows = store.query_overlap(qi.earliest_us, qi.latest_us)
        best_per_doc: dict[str, float] = {}
        for expr_id, doc_id, e_us, l_us, b_us, gran in rows:
            s = Interval(
                earliest_us=e_us,
                latest_us=l_us,
                best_us=b_us,
                granularity=gran,
            )
            sc = score_pair(qi, s)
            if sc > best_per_doc.get(doc_id, 0.0):
                best_per_doc[doc_id] = sc
        for d, sc in best_per_doc.items():
            sum_per_doc[d] += sc
            if sc > max_per_doc[d]:
                max_per_doc[d] = sc
    return dict(sum_per_doc), dict(max_per_doc)


def _score_anchors_per_doc(
    astore: UtteranceAnchorStore,
    q_intervals: list[Interval],
) -> dict[str, float]:
    """Best anchor-score across query-intervals, per doc. One row per
    doc by design, so multiple query-intervals take max (not sum) — the
    anchor represents a single doc-utterance event."""
    best_per_doc: dict[str, float] = {}
    for qi in q_intervals:
        rows = astore.query_overlap(qi.earliest_us, qi.latest_us)
        for doc_id, e_us, l_us, b_us, gran in rows:
            s = Interval(
                earliest_us=e_us,
                latest_us=l_us,
                best_us=b_us,
                granularity=gran,
            )
            sc = score_pair(qi, s)
            if sc > best_per_doc.get(doc_id, 0.0):
                best_per_doc[doc_id] = sc
    return best_per_doc


def retrieve(
    store: IntervalStore,
    astore: UtteranceAnchorStore,
    q_intervals: list[Interval],
    *,
    source: SourceMode = "union",
    agg: AggMode = "sum",
    alpha: float = 0.5,
    beta: float = 0.5,
    intent: Intent | None = None,
    intent_weights: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Return doc_id -> aggregate score.

    R1 variants (source):
      - anchor_only: ignore `store`; score only via anchors.
      - referent_only: ignore `astore`; legacy `temporal_retrieve`-equivalent
          (sum-of-bests, matching eval.py's aggregation).
      - union: union doc-ids from both; combine via `agg`.

    R2 variants (agg) — only used with source=="union":
      - sum: score = anchor_score + Σ ref_scores
      - max: score = max(anchor_score, best_ref_score)
      - weighted: α*best_ref_score + β*anchor_score
      - sum_weighted: α*Σ ref_scores + β*anchor_score
      - routed: pick (α, β) from intent_weights[intent]; default to weighted.

    intent_weights: dict mapping intent label ("utterance", "referent",
      "ambiguous") -> (alpha, beta) used only when agg=="routed".
    """
    ref_sum, ref_max = _score_referents_per_doc(store, q_intervals)
    anchor_scores = _score_anchors_per_doc(astore, q_intervals)

    if source == "referent_only":
        return dict(ref_sum)
    if source == "anchor_only":
        return dict(anchor_scores)

    # union
    doc_ids = set(ref_sum) | set(anchor_scores)
    out: dict[str, float] = {}

    if agg == "routed":
        if intent_weights is None:
            intent_weights = {
                "utterance": (0.3, 0.7),
                "referent": (0.7, 0.3),
                "ambiguous": (0.5, 0.5),
            }
        a, b = intent_weights.get(intent or "ambiguous", (alpha, beta))
        for d in doc_ids:
            out[d] = a * ref_max.get(d, 0.0) + b * anchor_scores.get(d, 0.0)
        return out

    for d in doc_ids:
        a_s = anchor_scores.get(d, 0.0)
        r_sum = ref_sum.get(d, 0.0)
        r_max = ref_max.get(d, 0.0)
        if agg == "sum":
            out[d] = r_sum + a_s
        elif agg == "max":
            out[d] = max(r_max, a_s)
        elif agg == "weighted":
            out[d] = alpha * r_max + beta * a_s
        elif agg == "sum_weighted":
            out[d] = alpha * r_sum + beta * a_s
        else:
            raise ValueError(f"unknown agg mode: {agg}")
    return out
