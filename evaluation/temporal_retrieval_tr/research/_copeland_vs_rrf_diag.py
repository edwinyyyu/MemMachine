"""Per-query diagnostic: find queries where Copeland and RRF disagree
on composition and recency_vs_rerank benches. Show the docs + cosines
+ ref_times to understand the mechanism difference concretely.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np

from temporal_retrieval_tr import Doc, TemporalRetriever
from temporal_retrieval.research._common import (
    DATA_DIR, make_embed_fn, setup_env,
)
from temporal_retrieval_tr.research.bench import (
    load_bench, make_cached_embed_fn, make_cosine_rerank_fn,
)

setup_env()


async def rrf_top1(
    vd: TemporalRetriever, query: str, ref_time: str, rrf_k: int, alpha: float,
) -> tuple[str, list[tuple[str, dict]]]:
    """Run RRF on a single query and return top-1 + pool details for inspection."""
    import numpy as np
    from temporal_retrieval_tr.scoring import final_score
    from temporal_retrieval_tr.time_range import is_infinite_measure
    from temporal_retrieval_min.core import build_pool

    plan = await vd._planner.plan(query, ref_time)
    has_extremum = plan.latest_intent or plan.earliest_intent
    if not has_extremum:
        # Non-extremum: just use Copeland default
        vd.copeland_bonus = 0.40
        r = await vd.query(query, ref_time, k=10)
        return r[0].doc_id if r else None, []

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
        return None, []

    pool_texts = [vd._docs[did].text for did in pool]
    rerank_scores = await vd.rerank_fn(query, pool_texts)
    rerank_pool = dict(zip(pool, rerank_scores, strict=False))
    sim_score = {did: rerank_pool[did] + match_all[did] for did in pool}
    sim_ranked = sorted(pool, key=lambda d: -sim_score[d])
    rank_sim = {did: i + 1 for i, did in enumerate(sim_ranked)}

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

    rrf_score = {
        did: 1.0 / (rrf_k + rank_sim[did]) + alpha / (rrf_k + rank_rec[did])
        for did in pool
    }
    ranked = sorted(pool, key=lambda d: -rrf_score[d])
    # Pool detail for top 8
    details = []
    for did in ranked[:8]:
        details.append((did, {
            "cos": rerank_pool[did],
            "match": match_all[did],
            "rank_sim": rank_sim[did],
            "rank_rec": rank_rec[did],
            "rrf": rrf_score[did],
            "anchor_year": anchors[did] // (1_000_000 * 86400 * 365),
        }))
    return ranked[0], details


async def analyze(bench: str, embed_fn, rerank_fn, alpha: float) -> None:
    print(f"\n=========== {bench} (RRF α={alpha}) ===========\n")
    docs_jsonl, queries, gold = load_bench(bench)
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]
    doc_text = {d.id: d.text for d in docs}
    doc_ref = {d.id: d.ref_time[:10] for d in docs}
    vd = TemporalRetriever(embed_fn=embed_fn, rerank_fn=rerank_fn)
    await vd.index(docs)

    # Copeland baseline
    vd.copeland_bonus = 0.40
    cope_top1 = {}
    for q in queries:
        r = await vd.query(q["text"], q["ref_time"], k=10)
        cope_top1[q["query_id"]] = r[0].doc_id if r else None

    # RRF
    rrf_top1_map = {}
    rrf_pool_detail = {}
    for q in queries:
        top, detail = await rrf_top1(vd, q["text"], q["ref_time"], 60, alpha)
        rrf_top1_map[q["query_id"]] = top
        rrf_pool_detail[q["query_id"]] = detail

    # Find queries where they DIFFER
    for q in queries:
        qid = q["query_id"]
        cope_doc = cope_top1[qid]
        rrf_doc = rrf_top1_map[qid]
        if cope_doc == rrf_doc:
            continue
        golds = set(gold.get(qid, set()))
        cope_hit = cope_doc in golds
        rrf_hit = rrf_doc in golds
        if not (rrf_hit and not cope_hit):
            # Only show cases where RRF wins (rrf hit, copeland miss)
            continue
        print(f"\n  qid={qid}")
        print(f"  Q: {q['text']}")
        print(f"  golds: {sorted(golds)}")
        print(f"  Copeland top1: {cope_doc} ({'HIT' if cope_hit else 'miss'})")
        print(f"    text: {doc_text[cope_doc][:130]}")
        print(f"    ref_time: {doc_ref[cope_doc]}")
        print(f"  RRF top1:      {rrf_doc} ({'HIT' if rrf_hit else 'miss'})")
        print(f"    text: {doc_text[rrf_doc][:130]}")
        print(f"    ref_time: {doc_ref[rrf_doc]}")
        # Show top-8 pool with RRF detail
        print(f"  Pool (RRF top-8):")
        for did, info in rrf_pool_detail[qid]:
            mark = "*GOLD*" if did in golds else ""
            print(f"    {mark:6s} {did:30s} cos={info['cos']:.3f} m={info['match']:.3f} "
                  f"rs={info['rank_sim']:>2d} rr={info['rank_rec']:>2d} "
                  f"ref={doc_ref.get(did, '')}")


async def main() -> None:
    raw_embed = await make_embed_fn()
    embed_fn = make_cached_embed_fn(raw_embed)
    rerank_fn = make_cosine_rerank_fn(embed_fn)
    # Composition: RRF α=0.5 wins (+2 over cope_0.40)
    await analyze("composition", embed_fn, rerank_fn, alpha=0.5)
    # recency_vs_rerank: RRF α=2.0 wins big
    await analyze("recency_vs_rerank", embed_fn, rerank_fn, alpha=2.0)


if __name__ == "__main__":
    asyncio.run(main())
