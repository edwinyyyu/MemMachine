"""A/B: production planner vs retrieval-oriented no-anaphora planner."""
from __future__ import annotations

import asyncio, gc

from temporal_retrieval_tr import Doc, TemporalRetriever
from temporal_retrieval.research._common import make_embed_fn, setup_env
from temporal_retrieval_tr.research.bench import (
    BENCH_NAMES, load_bench, make_cached_embed_fn, make_cosine_rerank_fn, metrics,
)
from temporal_retrieval_tr.research._retrieval_oriented_planner import (
    RetrievalOrientedPlanner,
)

setup_env()


async def run_bench(bench, embed_fn, rerank_fn):
    loaded = load_bench(bench)
    if loaded[0] is None:
        return None
    docs_jsonl, queries, gold = loaded
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]
    vd_b = TemporalRetriever(embed_fn=embed_fn, rerank_fn=rerank_fn)
    await vd_b.index(docs)
    rk_b = {q["query_id"]: [x.doc_id for x in
            await vd_b.query(q["text"], q["ref_time"], k=10)]
            for q in queries}
    m_b = metrics(rk_b, gold)
    vd_r = TemporalRetriever(embed_fn=embed_fn, rerank_fn=rerank_fn,
                             planner=RetrievalOrientedPlanner())
    await vd_r.index(docs)
    rk_r = {q["query_id"]: [x.doc_id for x in
            await vd_r.query(q["text"], q["ref_time"], k=10)]
            for q in queries}
    m_r = metrics(rk_r, gold)
    del vd_b, vd_r, docs
    gc.collect()
    return m_b, m_r


async def main():
    raw = await make_embed_fn()
    embed_fn = make_cached_embed_fn(raw)
    rerank_fn = make_cosine_rerank_fn(embed_fn)
    print(f"\n=== Retrieval-oriented prompt A/B over {len(BENCH_NAMES)} benches ===\n",
          flush=True)
    hdr = (f"{'bench':30s}  {'base R@1':>8s} {'ro R@1':>8s} {'ΔR@1':>7s}  "
           f"{'base R@5':>8s} {'ro R@5':>8s}  {'n':>4s}")
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    rows = {}
    for bench in BENCH_NAMES:
        try:
            r = await run_bench(bench, embed_fn, rerank_fn)
        except Exception as e:
            print(f"{bench:30s}  ERROR: {e}", flush=True)
            continue
        if r is None:
            continue
        mb, mr = r
        rows[bench] = (mb, mr)
        d = mr["R@1"] - mb["R@1"]
        mark = ">" if abs(d) >= 0.02 else " "
        print(f"{mark} {bench:28s}  {mb['R@1']:>8.3f} {mr['R@1']:>8.3f} {d:>+7.3f}  "
              f"{mb['R@5']:>8.3f} {mr['R@5']:>8.3f}  {mb['n']:>4d}", flush=True)
    if rows:
        n = len(rows)
        mb_r1 = sum(r[0]["R@1"] for r in rows.values()) / n
        mr_r1 = sum(r[1]["R@1"] for r in rows.values()) / n
        mb_r5 = sum(r[0]["R@5"] for r in rows.values()) / n
        mr_r5 = sum(r[1]["R@5"] for r in rows.values()) / n
        print("-" * len(hdr), flush=True)
        print(f"  {'MACRO':28s}  {mb_r1:>8.4f} {mr_r1:>8.4f} {mr_r1 - mb_r1:>+7.4f}  "
              f"{mb_r5:>8.4f} {mr_r5:>8.4f}  n={n}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
