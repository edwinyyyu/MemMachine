"""Single-arm bench for the new TemporalExtractor.

Compare against historical baseline (shipped planner v5-neutral-examples
+ v3.3 extractor, macro R@1 = 0.8041, R@5 = 0.9605) measured in
b4h72t3bv. We only need to compute the new arm's macro.
"""
from __future__ import annotations

import asyncio, gc

from temporal_retrieval_tr import Doc, TemporalExtractor, TemporalRetriever
from temporal_retrieval.research._common import make_embed_fn, setup_env
from temporal_retrieval_tr.research.bench import (
    BENCH_NAMES, load_bench, make_cached_embed_fn, make_cosine_rerank_fn, metrics,
)

setup_env()

# Historical baseline (b4h72t3bv "ship" column: planner v5 + v3.3 extractor)
BASELINE_R1 = 0.8041
BASELINE_R5 = 0.9605


async def run_bench(bench, embed_fn, rerank_fn):
    loaded = load_bench(bench)
    if loaded[0] is None:
        return None
    docs_jsonl, queries, gold = loaded
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]

    vd = TemporalRetriever(embed_fn=embed_fn, rerank_fn=rerank_fn,
                           extractor=TemporalExtractor())
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
    print(f"\n=== New TemporalExtractor (multi-interval) bench, vs historical "
          f"v3.3 baseline (R@1={BASELINE_R1}, R@5={BASELINE_R5}) ===\n",
          flush=True)
    hdr = f"{'bench':30s}  {'new R@1':>9s}  {'new R@5':>9s}  {'n':>4s}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    rows = []
    for bench in BENCH_NAMES:
        try:
            m = await run_bench(bench, embed_fn, rerank_fn)
        except Exception as e:
            print(f"{bench:30s}  ERROR: {e}", flush=True)
            continue
        if m is None:
            continue
        rows.append((bench, m))
        print(f"  {bench:28s}  {m['R@1']:>9.3f}  {m['R@5']:>9.3f}  {m['n']:>4d}",
              flush=True)
    if rows:
        n = len(rows)
        new_r1 = sum(m["R@1"] for _, m in rows) / n
        new_r5 = sum(m["R@5"] for _, m in rows) / n
        print("-" * len(hdr), flush=True)
        print(f"  {'MACRO (new)':28s}  {new_r1:>9.4f}  {new_r5:>9.4f}  n={n}",
              flush=True)
        print(f"  {'MACRO (v3.3 historical)':28s}  {BASELINE_R1:>9.4f}  "
              f"{BASELINE_R5:>9.4f}", flush=True)
        print(f"  {'Δ':28s}  {new_r1 - BASELINE_R1:>+9.4f}  "
              f"{new_r5 - BASELINE_R5:>+9.4f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
