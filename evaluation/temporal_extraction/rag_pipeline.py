"""RAG pipeline variants: all 9 fusion strategies.

Each variant takes per-query retrieval data (T/S/A/E scores, router
intents, optional LLM rerank) and returns a ranked list of doc_ids.

Retriever names used throughout:
- "T" = temporal (multi-axis)
- "S" = semantic (cosine)
- "A" = allen relational
- "E" = era

CASCADE (V1): temporal top-20 -> semantic rerank.
TEMPORAL-ONLY (V2): T only.
SEMANTIC-ONLY (V3): S only.
RRF-ALL (V4): RRF(k=60) across T, S, A, E (only those that fired).
ROUTED-SINGLE (V5): pick one retriever from router intent.
ROUTED-MULTI (V6): pick 1-N retrievers from router, RRF between them.
SCORE-BLEND (V7): min-max normalized weighted sum 0.4/0.4/0.1/0.1.
LLM-RERANK (V8): union top-20 from each retriever, LLM rerank.
HYBRID-CASCADE-RRF (V9): CASCADE if T fires well, else RRF-ALL if ambiguous,
    else pure SEMANTIC if T fires nothing.
"""

from __future__ import annotations

from rag_fusion import rrf, score_blend, scores_to_ranked

# ---------------------------------------------------------------------------
# Variant container
# ---------------------------------------------------------------------------
INTENT_TO_RETRIEVER = {
    "temporal": "T",
    "semantic": "S",
    "relational": "A",
    "era": "E",
}


def _top_k(scored: dict[str, float], k: int) -> list[str]:
    return [d for d, _ in sorted(scored.items(), key=lambda x: x[1], reverse=True)[:k]]


def _rank_list(scored: dict[str, float]) -> list[str]:
    return scores_to_ranked(scored)


# ---------------------------------------------------------------------------
# V1 CASCADE
# ---------------------------------------------------------------------------
def v1_cascade(
    t_scores: dict[str, float],
    s_scores: dict[str, float],
    all_doc_ids: list[str],
    t_top_cand: int = 20,
) -> list[str]:
    t_ranked = _rank_list(t_scores)
    # If temporal fired nothing for this query, fall back to semantic.
    if not t_ranked or max(t_scores.values(), default=0.0) <= 0.0:
        return _rank_list(s_scores)
    cand = t_ranked[:t_top_cand]
    # Semantic rerank within cand.
    sem = {d: s_scores.get(d, 0.0) for d in cand}
    reranked = _rank_list(sem)
    # Append remainder of S's ranking excluding the reranked set.
    seen = set(reranked)
    tail = [d for d in _rank_list(s_scores) if d not in seen]
    return reranked + tail


# V2 TEMPORAL-ONLY
def v2_temporal_only(
    t_scores: dict[str, float], s_scores: dict[str, float]
) -> list[str]:
    ranked = _rank_list(t_scores)
    if not ranked:
        return _rank_list(s_scores)
    # Append rest by semantic fallback (stable tail)
    seen = set(ranked)
    tail = [d for d in _rank_list(s_scores) if d not in seen]
    return ranked + tail


# V3 SEMANTIC-ONLY
def v3_semantic_only(s_scores: dict[str, float]) -> list[str]:
    return _rank_list(s_scores)


# V4 RRF-ALL
def v4_rrf_all(
    t_scores: dict[str, float],
    s_scores: dict[str, float],
    a_scores: dict[str, float],
    e_scores: dict[str, float],
    k: int = 60,
) -> list[str]:
    lists = []
    for scored in (t_scores, s_scores, a_scores, e_scores):
        rl = _rank_list(scored)
        if rl:
            lists.append(rl)
    fused = rrf(lists, k=k)
    return [d for d, _ in fused]


# V5 ROUTED-SINGLE
def v5_routed_single(
    intents: list[str],
    t_scores: dict[str, float],
    s_scores: dict[str, float],
    a_scores: dict[str, float],
    e_scores: dict[str, float],
) -> list[str]:
    if not intents:
        return _rank_list(s_scores)
    name = INTENT_TO_RETRIEVER.get(intents[0], "S")
    lookup = {"T": t_scores, "S": s_scores, "A": a_scores, "E": e_scores}
    chosen = lookup.get(name, s_scores)
    ranked = _rank_list(chosen)
    if not ranked:
        return _rank_list(s_scores)
    seen = set(ranked)
    tail = [d for d in _rank_list(s_scores) if d not in seen]
    return ranked + tail


# V6 ROUTED-MULTI
def v6_routed_multi(
    intents: list[str],
    t_scores: dict[str, float],
    s_scores: dict[str, float],
    a_scores: dict[str, float],
    e_scores: dict[str, float],
    k: int = 60,
) -> list[str]:
    if not intents:
        return _rank_list(s_scores)
    lookup = {"T": t_scores, "S": s_scores, "A": a_scores, "E": e_scores}
    chosen_names = []
    for it in intents:
        nm = INTENT_TO_RETRIEVER.get(it)
        if nm and nm not in chosen_names:
            chosen_names.append(nm)
    if not chosen_names:
        return _rank_list(s_scores)
    lists = []
    for nm in chosen_names:
        rl = _rank_list(lookup[nm])
        if rl:
            lists.append(rl)
    if not lists:
        return _rank_list(s_scores)
    fused = rrf(lists, k=k)
    ranked = [d for d, _ in fused]
    seen = set(ranked)
    tail = [d for d in _rank_list(s_scores) if d not in seen]
    return ranked + tail


# V7 SCORE-BLEND
def v7_score_blend(
    t_scores: dict[str, float],
    s_scores: dict[str, float],
    a_scores: dict[str, float],
    e_scores: dict[str, float],
    weights: dict[str, float] = None,
    top_k_per: int = 40,
) -> list[str]:
    if weights is None:
        weights = {"T": 0.4, "S": 0.4, "A": 0.1, "E": 0.1}
    per_ret = {
        "T": t_scores,
        "S": s_scores,
        "A": a_scores,
        "E": e_scores,
    }
    fused = score_blend(per_ret, weights, top_k_per=top_k_per)
    ranked = [d for d, _ in fused]
    # Tail with anything missed from S
    seen = set(ranked)
    tail = [d for d in _rank_list(s_scores) if d not in seen]
    return ranked + tail


# V8 LLM-RERANK (caller supplies reranked list over top-20 union)
def v8_llm_rerank(
    reranked_top: list[str],
    s_scores: dict[str, float],
) -> list[str]:
    seen = set(reranked_top)
    tail = [d for d in _rank_list(s_scores) if d not in seen]
    return reranked_top + tail


# V9 HYBRID
def v9_hybrid_cascade_rrf(
    intents: list[str],
    t_scores: dict[str, float],
    s_scores: dict[str, float],
    a_scores: dict[str, float],
    e_scores: dict[str, float],
    rich_threshold: float = 0.5,
) -> list[str]:
    """Routing logic:
    - If T fires nothing (max score == 0) -> pure SEMANTIC.
    - If router gave a single strong intent in {temporal} AND T is rich
      (at least one doc with T score >= rich_threshold relative to max),
      -> CASCADE.
    - Else -> RRF-ALL (ambiguous / multi-intent).
    """
    t_max = max(t_scores.values(), default=0.0)
    if t_max <= 0.0:
        return v3_semantic_only(s_scores)

    single_temporal = intents == ["temporal"]
    # Use cascade when router says temporal-only AND T actually produced
    # a usable top candidate set.
    if single_temporal and t_max > 0:
        return v1_cascade(t_scores, s_scores, list(s_scores.keys()))
    # Otherwise RRF-ALL across all that fired.
    return v4_rrf_all(t_scores, s_scores, a_scores, e_scores)
