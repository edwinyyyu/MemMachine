"""RRF recency ranking — characterize vs additive (absolute) and Copeland (relative).

RRF combines source rankings via score = sum over sources of 1/(k + rank).
For extremum queries, two sources:
- sim ranking: by (base + match), descending (most-relevant first)
- recency ranking: by ref_time, direction-aware

For non-extremum queries, fall back to current scoring.

The key question: where does RRF sit on the absolute/relative spectrum?
- Additive uses doc's ABSOLUTE rank position in recency ordering (W·r_v)
- Copeland uses PAIRWISE binary comparisons (more-recent gets +bonus)
- RRF uses rank position too, but combined across sources via 1/(k+rank)

Run this A/B implements RRF directly in the harness (bypassing the
retriever's main score path) so we don't need to modify shipped code
just to characterize.

Run from `evaluation/`:
    uv run python -m temporal_retrieval_tr.research._rrf_recency_ab
"""
from __future__ import annotations

import asyncio
import gc

from temporal_retrieval_tr import Doc, TemporalRetriever

from temporal_retrieval.research._common import make_embed_fn, setup_env
from temporal_retrieval_tr.research.bench import (
    BENCH_NAMES,
    load_bench,
    make_cached_embed_fn,
    make_cosine_rerank_fn,
    metrics,
)

setup_env()

# RRF k values to sweep
RRF_KS = [60]  # fix k at standard value; alpha is the more interesting knob
# Per-source weights (alpha = weight on recency source vs sim source)
# Sweep alpha to find the right recency weight.
RRF_ALPHAS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

# Reference arms
REFERENCE_ARMS = [
    ("add_W0.5",  "additive",  0.5,  None),
    ("cope_0.40", "copeland",  None, 0.40),
]


async def rrf_query(
    vd: TemporalRetriever, query: str, ref_time: str, k_top: int,
    rrf_k: int, alpha: float = 1.0,
) -> list[str]:
    """Run query with RRF combining sim and recency rankings.

    Uses the retriever's pool building / match / base computation,
    but replaces the final scoring with RRF.
    """
    # Step through retriever's query logic up to the scoring step.
    # We exploit that the retriever's setup is deterministic — call
    # query with a non-extremum config first to grab the pool, then
    # rerank by RRF.
    # Easier path: temporarily disable recency, get the pool, recompute
    # rankings by hand using cached state.
    from temporal_retrieval_min.schema import parse_iso, to_us
    plan = await vd._planner.plan(query, ref_time)
    has_extremum = plan.latest_intent or plan.earliest_intent
    if not has_extremum:
        # Non-extremum: fall back to current scoring (use cope_0.40 default)
        vd.copeland_bonus = 0.40
        results = await vd.query(query, ref_time, k=k_top)
        return [r.doc_id for r in results]

    # Compute pool via the retriever's normal logic, but score with RRF.
    import numpy as np
    from temporal_retrieval_tr.scoring import final_score
    from temporal_retrieval_tr.time_range import is_infinite_measure
    from temporal_retrieval_min.core import build_pool, normalize_rerank_full

    query_targets = plan.targets
    bounded_target_present = any(
        not is_infinite_measure(t) for t in query_targets
    )
    q_emb = (await vd.embed_fn([query]))[0]
    q_emb = np.asarray(q_emb, dtype=np.float32)
    sem_scores = vd._cosine_all(q_emb)
    all_dids = list(vd._doc_ref_us.keys())

    match_all = {}
    eligible = []
    for did in all_dids:
        d_anchors = vd._doc_anchors.get(did, [])
        if not d_anchors:
            match_all[did] = 1.0
            if not query_targets or not bounded_target_present:
                eligible.append(did)
        else:
            s = final_score(query_targets, d_anchors)
            match_all[did] = s
            if s > 0.0:
                eligible.append(did)

    pool = build_pool(sem_scores, all_dids, eligible, vd.pool_size)
    if not pool:
        return []

    # Rerank for sim ranking
    pool_texts = [vd._docs[did].text for did in pool]
    rerank_scores = await vd.rerank_fn(query, pool_texts)
    rerank_pool = dict(zip(pool, rerank_scores, strict=False))
    # sim_score = raw rerank + match (we're using raw cosine reranker)
    sim_score = {did: rerank_pool[did] + match_all[did] for did in pool}

    # Source 1: rank by sim (descending)
    sim_ranked = sorted(pool, key=lambda d: -sim_score[d])
    rank_sim = {did: i + 1 for i, did in enumerate(sim_ranked)}

    # Source 2: rank by recency anchor (direction-aware)
    direction_latest = plan.latest_intent
    anchors = {}
    for did in pool:
        ivs = vd._doc_ivs.get(did, [])
        anchor = None
        for iv in ivs:
            mid = (iv.earliest_us + iv.latest_us) // 2
            if anchor is None:
                anchor = mid
            elif direction_latest and mid > anchor:
                anchor = mid
            elif (not direction_latest) and mid < anchor:
                anchor = mid
        anchors[did] = anchor if anchor is not None else vd._doc_ref_us[did]
    rec_sorted = sorted(pool, key=lambda d: -anchors[d] if direction_latest else anchors[d])
    rank_rec = {did: i + 1 for i, did in enumerate(rec_sorted)}

    # RRF score
    rrf_score = {
        did: 1.0 / (rrf_k + rank_sim[did]) + alpha / (rrf_k + rank_rec[did])
        for did in pool
    }
    ranked = sorted(pool, key=lambda d: -rrf_score[d])
    return ranked[:k_top]


async def run_bench(bench: str, embed_fn, rerank_fn) -> dict | None:
    loaded = load_bench(bench)
    if loaded[0] is None:
        return None
    docs_jsonl, queries, gold = loaded
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]
    vd = TemporalRetriever(embed_fn=embed_fn, rerank_fn=rerank_fn)
    await vd.index(docs)
    out = {}
    # Reference arms
    for label, mode, w, b in REFERENCE_ARMS:
        if mode == "additive":
            vd.recency_weight = w
            vd.copeland_bonus = None
        else:
            vd.copeland_bonus = b
        rk = {}
        for q in queries:
            r = await vd.query(q["text"], q["ref_time"], k=10)
            rk[q["query_id"]] = [x.doc_id for x in r]
        out[label] = metrics(rk, gold)
    # RRF arms
    for rrf_k in RRF_KS:
        for alpha in RRF_ALPHAS:
            label = f"rrf_k{rrf_k}_a{alpha}"
            rk = {}
            for q in queries:
                rank_ids = await rrf_query(vd, q["text"], q["ref_time"], 10, rrf_k, alpha)
                rk[q["query_id"]] = rank_ids
            out[label] = metrics(rk, gold)
    del vd, docs, docs_jsonl
    gc.collect()
    return out


async def main() -> None:
    raw_embed = await make_embed_fn()
    embed_fn = make_cached_embed_fn(raw_embed)
    rerank_fn = make_cosine_rerank_fn(embed_fn)
    labels = [a[0] for a in REFERENCE_ARMS] + [
        f"rrf_k{k}_a{a}" for k in RRF_KS for a in RRF_ALPHAS
    ]
    print(f"=== RRF recency A/B ({len(BENCH_NAMES)} benches) ===\n", flush=True)
    print(f"Arms: {labels}\n", flush=True)
    header = "  ".join(f"{L:>14s}" for L in labels)
    hdr = f"{'bench':28s}  {header}    n"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    rows = {}
    key = {"composition", "cotemporal", "same_topic_recency",
           "same_topic_recency_hard", "recency_stress_deep", "recency_vs_rerank",
           "latest_recent"}
    for bench in BENCH_NAMES:
        try:
            res = await run_bench(bench, embed_fn, rerank_fn)
        except Exception as e:
            print(f"{bench:28s}  ERROR: {e}", flush=True)
            continue
        if res is None:
            continue
        rows[bench] = res
        mark = ">" if bench in key else " "
        cells = "  ".join(f"{res[L]['R@1']:>14.3f}" for L in labels)
        print(f"{mark} {bench:26s}  {cells}  {res[labels[0]]['n']:>4d}",
              flush=True)
    if rows:
        n = len(rows)
        print("-" * len(hdr), flush=True)
        for k_metric in ("R@1", "R@5", "R@10"):
            macro = {L: sum(r[L].get(k_metric, 0) for r in rows.values()) / n
                     for L in labels}
            cells = "  ".join(f"{macro[L]:>14.4f}" for L in labels)
            print(f"  {'MACRO ' + k_metric:26s}  {cells}  n={n}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
