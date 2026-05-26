"""RRF noise discrimination — add RRF arms to the noise injection test.

How RRF should behave theoretically:
- RRF uses RANKS, not score magnitudes. Sim noise perturbs the sim
  ranking but ranks are bounded integers — noise can shift gold by
  a few positions but not unbounded.
- Recency ranks are unchanged by sim noise.
- RRF's contribution per source = 1/(k+rank), so small rank shifts
  have small impact.

Question: is RRF noise-robust like Copeland, or noise-fragile like additive?
"""
from __future__ import annotations

import asyncio
import gc

import numpy as np

from temporal_retrieval_tr import Doc, TemporalRetriever
from temporal_retrieval.research._common import make_embed_fn, setup_env
from temporal_retrieval_tr.research.bench import (
    load_bench, make_cached_embed_fn, make_cosine_rerank_fn, metrics,
)

setup_env()

NOISE_SCALES = [0.0, 0.05, 0.10, 0.20, 0.50]
POOL_SIZES = [40, 200]
RRF_ALPHA = 0.5
RRF_K = 60


async def rrf_query_noisy(
    vd: TemporalRetriever, query: str, ref_time: str, k_top: int,
    noise_scale: float,
) -> list[str]:
    """RRF with the SAME noise injection that the retriever uses for
    additive/Copeland, so all arms see identical noise."""
    import hashlib
    import random
    from temporal_retrieval_tr.scoring import final_score
    from temporal_retrieval_tr.time_range import is_infinite_measure
    from temporal_retrieval_min.core import build_pool

    plan = await vd._planner.plan(query, ref_time)
    has_extremum = plan.latest_intent or plan.earliest_intent
    if not has_extremum:
        # Non-extremum: fall back to Copeland default
        r = await vd.query(query, ref_time, k=k_top)
        return [x.doc_id for x in r]

    query_targets = plan.targets
    bounded = any(not is_infinite_measure(t) for t in query_targets)
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
            if not query_targets or not bounded:
                eligible.append(did)
        else:
            s = final_score(query_targets, d_anchors)
            match_all[did] = s
            if s > 0.0:
                eligible.append(did)

    pool = build_pool(sem_scores, all_dids, eligible, vd.pool_size)
    if not pool:
        return []

    pool_texts = [vd._docs[did].text for did in pool]
    rerank_scores = await vd.rerank_fn(query, pool_texts)
    rerank_pool = dict(zip(pool, rerank_scores, strict=False))

    # Apply the same noise as in the retriever's main path
    if noise_scale > 0.0:
        q_seed = f"42|{query}".encode("utf-8")
        seed_int = int(hashlib.sha256(q_seed).hexdigest()[:16], 16)
        rng = random.Random(seed_int)
        half = noise_scale / 2.0
        rerank_pool = {d: rerank_pool[d] + rng.uniform(-half, half) for d in pool}

    sim_score = {d: rerank_pool[d] + match_all[d] for d in pool}
    sim_ranked = sorted(pool, key=lambda d: -sim_score[d])
    rank_sim = {d: i + 1 for i, d in enumerate(sim_ranked)}

    direction_latest = plan.latest_intent
    anchors = {}
    for d in pool:
        ivs = vd._doc_ivs.get(d, [])
        anchor = None
        for iv in ivs:
            mid = (iv.earliest_us + iv.latest_us) // 2
            if anchor is None:
                anchor = mid
            elif direction_latest and mid > anchor:
                anchor = mid
            elif (not direction_latest) and mid < anchor:
                anchor = mid
        anchors[d] = anchor if anchor is not None else vd._doc_ref_us[d]
    rec_sorted = sorted(pool, key=lambda d: -anchors[d] if direction_latest else anchors[d])
    rank_rec = {d: i + 1 for i, d in enumerate(rec_sorted)}

    rrf_score = {
        d: 1.0 / (RRF_K + rank_sim[d]) + RRF_ALPHA / (RRF_K + rank_rec[d])
        for d in pool
    }
    ranked = sorted(pool, key=lambda d: -rrf_score[d])
    return ranked[:k_top]


async def main() -> None:
    raw_embed = await make_embed_fn()
    embed_fn = make_cached_embed_fn(raw_embed)
    rerank_fn = make_cosine_rerank_fn(embed_fn)

    print(f"=== RRF noise test (recency_stress_deep, α=0.5) ===\n", flush=True)

    docs_jsonl, queries, gold = load_bench("recency_stress_deep")
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]

    for ps in POOL_SIZES:
        print(f"\n--- pool_size = {ps} ---", flush=True)
        hdr = (f"  {'noise':>6s} | {'add W=0.5':>10s} {'cope_0.40':>10s} {'rrf_α0.5':>10s}")
        print(hdr, flush=True)
        print("  " + "-" * (len(hdr) - 2), flush=True)
        for noise in NOISE_SCALES:
            vd = TemporalRetriever(
                embed_fn=embed_fn, rerank_fn=rerank_fn,
                pool_size=ps, noise_scale=noise,
            )
            await vd.index(docs)
            # Additive
            vd.recency_weight = 0.5
            vd.copeland_bonus = None
            rk = {}
            for q in queries:
                r = await vd.query(q["text"], q["ref_time"], k=10)
                rk[q["query_id"]] = [x.doc_id for x in r]
            add_r1 = metrics(rk, gold)["R@1"]
            # Copeland
            vd.copeland_bonus = 0.40
            rk = {}
            for q in queries:
                r = await vd.query(q["text"], q["ref_time"], k=10)
                rk[q["query_id"]] = [x.doc_id for x in r]
            cope_r1 = metrics(rk, gold)["R@1"]
            # RRF (with same noise applied)
            rk = {}
            for q in queries:
                ids = await rrf_query_noisy(vd, q["text"], q["ref_time"], 10, noise)
                rk[q["query_id"]] = ids
            rrf_r1 = metrics(rk, gold)["R@1"]
            print(f"  {noise:>6.3f} | {add_r1:>10.3f} {cope_r1:>10.3f} {rrf_r1:>10.3f}",
                  flush=True)
            del vd
            gc.collect()


if __name__ == "__main__":
    asyncio.run(main())
