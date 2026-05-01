"""Co-temporal graph-expansion retrieval.

Wraps a direct-retrieval scoring dict with graph-expansion scores and
blends the two with semantic cosine.

Usage:
    ranked = cotemporal_rerank(
        direct_scores=direct_scores,
        semantic_scores=sem_scores,
        graph=graph,                  # CotemporalGraph instance
        all_doc_ids=all_doc_ids,
        use_cotemporal=True,
        K_seed=20, M_neighbors=10,
        alpha=0.6, beta=0.25, gamma=0.15,
        decay=0.5,
    )

Returns list[(doc_id, final_score)] sorted desc.
"""

from __future__ import annotations

from collections.abc import Iterable

from cotemporal_graph import CotemporalGraph


def _minmax(d: dict[str, float]) -> dict[str, float]:
    if not d:
        return {}
    vals = list(d.values())
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1e-9
    return {k: (v - lo) / span for k, v in d.items()}


def cotemporal_rerank(
    direct_scores: dict[str, float],
    semantic_scores: dict[str, float],
    graph: CotemporalGraph | None,
    all_doc_ids: Iterable[str],
    use_cotemporal: bool = True,
    K_seed: int = 20,
    M_neighbors: int = 10,
    alpha: float = 0.6,
    beta: float = 0.25,
    gamma: float = 0.15,
    decay: float = 0.5,
) -> tuple[list[tuple[str, float]], dict[str, dict]]:
    """Return (ranked, diagnostics).

    diagnostics[doc_id] = {
        "direct": float, "expansion": float, "semantic": float,
        "final": float, "via_expansion_only": bool,
    }
    """
    all_ids = set(all_doc_ids)

    # 1. First hop (normalize direct scores)
    direct_n = _minmax(direct_scores)

    # 2. Expansion via graph
    expansion_raw: dict[str, float] = {}
    if use_cotemporal and graph is not None:
        seeds = sorted(direct_n.items(), key=lambda x: x[1], reverse=True)[:K_seed]
        for seed_id, seed_score in seeds:
            if seed_score <= 0.0:
                continue
            neighbors = graph.neighbors(seed_id, limit=M_neighbors)
            for nb_id, w, _reason in neighbors:
                contrib = seed_score * w * decay
                expansion_raw[nb_id] = expansion_raw.get(nb_id, 0.0) + contrib

    expansion_n = _minmax(expansion_raw)
    sem_n = _minmax(semantic_scores)

    # 3. Blend
    final: dict[str, float] = {}
    diag: dict[str, dict] = {}
    direct_set = {d for d, s in direct_n.items() if s > 0}
    for d in all_ids:
        dsc = direct_n.get(d, 0.0)
        esc = expansion_n.get(d, 0.0)
        ssc = sem_n.get(d, 0.0)
        fsc = alpha * dsc + beta * esc + gamma * ssc
        final[d] = fsc
        diag[d] = {
            "direct": dsc,
            "expansion": esc,
            "semantic": ssc,
            "final": fsc,
            "via_expansion_only": (d not in direct_set and esc > 0.0),
        }

    ranked = sorted(final.items(), key=lambda x: x[1], reverse=True)
    return ranked, diag


def describe_expansion_source(
    diag: dict[str, dict], top_k: int = 5, ranked: list[tuple[str, float]] | None = None
) -> dict[str, int]:
    """Count how many of the top-K retrieved docs came from expansion-only."""
    if ranked is None:
        ranked = sorted(diag.items(), key=lambda x: x[1]["final"], reverse=True)
        ranked = [(d, info["final"]) for d, info in ranked]
    top = ranked[:top_k]
    counts = {"direct_or_mixed": 0, "expansion_only": 0}
    for d, _ in top:
        info = diag.get(d, {})
        if info.get("via_expansion_only"):
            counts["expansion_only"] += 1
        else:
            counts["direct_or_mixed"] += 1
    return counts
