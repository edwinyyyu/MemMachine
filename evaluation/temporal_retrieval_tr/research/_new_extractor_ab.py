"""A/B: shipped planner + v3.3 (via adapter) vs shipped planner + new
TemporalExtractor (multi-interval).

Both arms use the same (shipped) planner, isolating the impact of the
extractor change: legacy singleton-only vs new multi-interval-capable.
"""
from __future__ import annotations

import asyncio, gc

from temporal_retrieval_tr import Doc, TemporalExtractor, TemporalRetriever
from temporal_retrieval.research._common import make_embed_fn, setup_env
from temporal_retrieval_tr.research.bench import (
    BENCH_NAMES, load_bench, make_cached_embed_fn, make_cosine_rerank_fn, metrics,
)
from temporal_retrieval_tr.research._v33_legacy_adapter import (
    V33LegacyExtractorAdapter,
)

setup_env()


async def run_bench(bench, embed_fn, rerank_fn):
    loaded = load_bench(bench)
    if loaded[0] is None:
        return None
    docs_jsonl, queries, gold = loaded
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]

    # A = shipped planner + legacy v3.3 (singletons-only) via research adapter
    vd_a = TemporalRetriever(embed_fn=embed_fn, rerank_fn=rerank_fn,
                             extractor=V33LegacyExtractorAdapter())
    await vd_a.index(docs)
    rk_a = {q["query_id"]: [x.doc_id for x in
            await vd_a.query(q["text"], q["ref_time"], k=10)]
            for q in queries}
    m_a = metrics(rk_a, gold)

    # B = shipped planner + new shipped TemporalExtractor (multi-interval)
    vd_b = TemporalRetriever(embed_fn=embed_fn, rerank_fn=rerank_fn,
                             extractor=TemporalExtractor())
    await vd_b.index(docs)
    rk_b = {q["query_id"]: [x.doc_id for x in
            await vd_b.query(q["text"], q["ref_time"], k=10)]
            for q in queries}
    m_b = metrics(rk_b, gold)

    del vd_a, vd_b, docs
    gc.collect()
    return m_a, m_b


async def main():
    raw = await make_embed_fn()
    embed_fn = make_cached_embed_fn(raw)
    rerank_fn = make_cosine_rerank_fn(embed_fn)
    print(f"\n=== Legacy v3.3 (singletons) vs new TemporalExtractor "
          f"(multi-interval) over {len(BENCH_NAMES)} benches ===\n", flush=True)
    hdr = (f"{'bench':30s}  {'v3.3 R@1':>9s} {'new R@1':>9s} {'ΔR@1':>7s}  "
           f"{'v3.3 R@5':>9s} {'new R@5':>9s}  {'n':>4s}")
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
        ma, mb = r
        rows[bench] = (ma, mb)
        d = mb["R@1"] - ma["R@1"]
        mark = ">" if abs(d) >= 0.02 else " "
        print(f"{mark} {bench:28s}  {ma['R@1']:>9.3f} {mb['R@1']:>9.3f} "
              f"{d:>+7.3f}  {ma['R@5']:>9.3f} {mb['R@5']:>9.3f}  {ma['n']:>4d}",
              flush=True)
    if rows:
        n = len(rows)
        ma_r1 = sum(r[0]["R@1"] for r in rows.values()) / n
        mb_r1 = sum(r[1]["R@1"] for r in rows.values()) / n
        ma_r5 = sum(r[0]["R@5"] for r in rows.values()) / n
        mb_r5 = sum(r[1]["R@5"] for r in rows.values()) / n
        print("-" * len(hdr), flush=True)
        print(f"  {'MACRO':28s}  {ma_r1:>9.4f} {mb_r1:>9.4f} "
              f"{mb_r1 - ma_r1:>+7.4f}  {ma_r5:>9.4f} {mb_r5:>9.4f}  n={n}",
              flush=True)


if __name__ == "__main__":
    asyncio.run(main())
