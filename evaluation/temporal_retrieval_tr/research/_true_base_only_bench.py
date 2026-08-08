"""True base-only floor: pool filtering + base rank, NO extremum Copeland.

The original `base_only` mode was misleading — it still ran the extremum
Copeland tournament for queries with latest_intent/earliest_intent.
`base_only_strict` skips that too, so this is the true "what does the
pool itself give us via cosine alone" floor.

Compare:
  pure-semantic baseline:     R@1=0.6241  R@5=0.9064
  base_only (extremum runs):  R@1=0.7989  R@5=0.9611
  base_only_strict (THIS):    R@1=?       R@5=?
  copeland_pairwise:          R@1=0.8044  R@5=0.9618

If base_only_strict ≈ pure-semantic → pool filtering displaces nothing;
all the gap above pure-semantic comes from extremum Copeland.

If base_only_strict significantly above pure-semantic → pool filtering
DOES displace some high-cosine non-eligibles (e.g., when only ~half
of build_pool's slots can pull from raw semantic).
"""
from __future__ import annotations

import asyncio, gc

from temporal_retrieval_tr import Doc, TemporalExtractor, TemporalRetriever
from temporal_retrieval_tr.planner import QueryPlanner
from temporal_retrieval.research._common import make_embed_fn, setup_env
from temporal_retrieval_tr.research.bench import (
    BENCH_NAMES, load_bench, make_cached_embed_fn, make_cosine_rerank_fn, metrics,
)

setup_env()


async def run_bench(bench, extractor, planner, embed_fn, rerank_fn):
    loaded = load_bench(bench)
    if loaded[0] is None:
        return None
    docs_jsonl, queries, gold = loaded
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]
    vd = TemporalRetriever(
        embed_fn=embed_fn, rerank_fn=rerank_fn,
        extractor=extractor, planner=planner,
        ranking_method="base_only_strict",
    )
    await vd.index(docs)
    rk = {q["query_id"]: [x.doc_id for x in
          await vd.query(q["text"], q["ref_time"], k=10)]
          for q in queries}
    m = metrics(rk, gold)
    del vd, docs
    gc.collect()
    return m


async def main():
    raw = await make_embed_fn()
    embed_fn = make_cached_embed_fn(raw)
    rerank_fn = make_cosine_rerank_fn(embed_fn)
    extractor = TemporalExtractor()
    planner = QueryPlanner()
    print("\n=== base_only_strict — true no-temporal-scoring floor ===\n",
          flush=True)
    rows = []
    for bench in BENCH_NAMES:
        try:
            m = await run_bench(bench, extractor, planner, embed_fn, rerank_fn)
        except Exception as e:
            print(f"  ERROR {bench}: {e}", flush=True)
            continue
        if m is None:
            continue
        rows.append((bench, m))
        print(
            f"  {bench:30s}  R@1={m['R@1']:.3f}  R@5={m['R@5']:.3f}  n={m['n']}",
            flush=True,
        )
    n = len(rows)
    r1 = sum(m["R@1"] for _, m in rows) / n
    r5 = sum(m["R@5"] for _, m in rows) / n
    print(
        f"\nMACRO base_only_strict:    R@1={r1:.4f}  R@5={r5:.4f}  n={n}",
        flush=True,
    )
    print("MACRO pure-semantic:       R@1=0.6241  R@5=0.9064", flush=True)
    print("MACRO base_only (extr on): R@1=0.7989  R@5=0.9611", flush=True)
    print("MACRO copeland_pairwise:   R@1=0.8044  R@5=0.9618", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
