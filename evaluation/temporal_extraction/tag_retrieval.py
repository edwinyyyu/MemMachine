"""Tag-based retrieval over a hierarchical-tag index.

Scoring:
    Jaccard:            |Q ∩ D| / |Q ∪ D|
    Rarity-weighted:    Σ w(t) for t in Q ∩ D    /    Σ w(t) for t in Q ∪ D

Where ``w(t)`` is either a static granularity weight (see
``hierarchical_tags.tag_weight``) or an IDF-style rarity score computed
from the corpus.

Per-(query, doc) pair scoring aggregates over expression pairs with
either ``max`` (strictest signal) or ``sum`` (many aligned expressions).
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Literal

from hierarchical_tags import tag_weight, tags_for_expression
from schema import TimeExpression
from tag_store import TagStore

TagScoreMode = Literal["jaccard", "weighted"]
TagAggMode = Literal["sum", "max"]


# ---------------------------------------------------------------------------
# Per-expression pair scoring
# ---------------------------------------------------------------------------
def _jaccard(q_tags: set[str], d_tags: set[str]) -> float:
    if not q_tags or not d_tags:
        return 0.0
    inter = q_tags & d_tags
    if not inter:
        return 0.0
    union = q_tags | d_tags
    return len(inter) / len(union)


def _weighted(
    q_tags: set[str],
    d_tags: set[str],
    weights: dict[str, float],
) -> float:
    """Rarity-weighted Jaccard."""
    if not q_tags or not d_tags:
        return 0.0
    inter = q_tags & d_tags
    if not inter:
        return 0.0
    union = q_tags | d_tags
    num = sum(weights.get(t, 1.0) for t in inter)
    denom = sum(weights.get(t, 1.0) for t in union)
    if denom <= 0:
        return 0.0
    return num / denom


# ---------------------------------------------------------------------------
# Rarity weights from corpus
# ---------------------------------------------------------------------------
def compute_idf_weights(store: TagStore, base: float = 1.0) -> dict[str, float]:
    """w(t) = base * log((N + 1) / (df + 1)) + 1, like BM25 smoothing.

    Combined with static granularity priors so very-common decade tags
    still contribute less than a rare-day tag even when df is similar.
    """
    n = max(1, store.num_docs())
    weights: dict[str, float] = {}
    for tag, hits in store.inverted.items():
        df = len({d for d, _ in hits})
        idf = math.log((n + 1.0) / (df + 1.0)) + 1.0
        prior = tag_weight(tag)  # granularity prior
        weights[tag] = base * idf * prior
    return weights


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def retrieve(
    store: TagStore,
    query_exprs: list[TimeExpression],
    score_mode: TagScoreMode = "jaccard",
    agg_mode: TagAggMode = "sum",
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Tag-based retrieval. Returns doc_id -> aggregate score.

    1. Expand each query TimeExpression to its tag set.
    2. Gather candidate (doc_id, expr_id) pairs via the inverted index.
    3. Score each (query-expr, stored-expr) pair.
    4. Aggregate across pairs per doc using ``agg_mode``.

    Within a given query expression, each stored expression contributes at
    most once (we take the raw per-pair score and later aggregate across
    all (query-expr, stored-expr) pairs).
    """
    if weights is None:
        weights = {t: tag_weight(t) for t in store.inverted}

    # Aggregate per doc_id across pairs.
    # For each (q_idx, stored (doc,expr)), compute one pair score.
    # Then aggregate across pairs for each doc.
    per_doc: dict[str, list[float]] = defaultdict(list)

    for q_i, q_te in enumerate(query_exprs):
        q_tags = tags_for_expression(q_te)
        if not q_tags:
            continue
        candidates = store.candidates_for_tags(q_tags)
        # Group by doc -> list of expr_ids (we'll dedupe (doc,expr) below).
        for doc_id, expr_id in candidates:
            d_tags = store.forward.get((doc_id, expr_id), set())
            if not d_tags:
                continue
            if score_mode == "jaccard":
                s = _jaccard(q_tags, d_tags)
            else:
                s = _weighted(q_tags, d_tags, weights)
            if s <= 0:
                continue
            per_doc[doc_id].append(s)

    # Aggregate across pairs.
    out: dict[str, float] = {}
    for d, scores in per_doc.items():
        if not scores:
            continue
        if agg_mode == "max":
            out[d] = max(scores)
        else:  # sum
            out[d] = sum(scores)
    return out


def rank(
    store: TagStore,
    query_exprs: list[TimeExpression],
    score_mode: TagScoreMode = "jaccard",
    agg_mode: TagAggMode = "sum",
    weights: dict[str, float] | None = None,
) -> list[tuple[str, float]]:
    scores = retrieve(
        store,
        query_exprs,
        score_mode=score_mode,
        agg_mode=agg_mode,
        weights=weights,
    )
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
