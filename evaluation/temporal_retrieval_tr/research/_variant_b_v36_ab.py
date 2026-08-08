"""A/B: shipped (variant B planner v4-recurring-enum-tod + v3.3 extractor)
vs combined (shipped planner + v3.6 surgical extractor).

Tests whether the surgical v3.6 prompt (minimal patch to v3.3) avoids
the LLM drift on non-recurring benches that v3.5_tod caused, while
still delivering past-3+future-1 + TOD-band recurring envelopes.
"""
from __future__ import annotations

import asyncio, gc

from temporal_retrieval_tr import Doc, TemporalRetriever
from temporal_retrieval.research._common import make_embed_fn, setup_env
from temporal_retrieval_tr.research.bench import (
    BENCH_NAMES, load_bench, make_cached_embed_fn, make_cosine_rerank_fn, metrics,
)
from temporal_retrieval_tr.research._extractor_v3_6_surgical import (
    TemporalExtractorV3_6,
)

setup_env()


async def run_bench(bench, embed_fn, rerank_fn):
    loaded = load_bench(bench)
    if loaded[0] is None:
        return None
    docs_jsonl, queries, gold = loaded
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]

    # A = current ship: variant B planner (default) + v3.3 extractor (default)
    vd_a = TemporalRetriever(embed_fn=embed_fn, rerank_fn=rerank_fn)
    await vd_a.index(docs)
    rk_a = {q["query_id"]: [x.doc_id for x in
            await vd_a.query(q["text"], q["ref_time"], k=10)]
            for q in queries}
    m_a = metrics(rk_a, gold)

    # B = variant B planner (default) + v3.6 surgical extractor
    vd_b = TemporalRetriever(embed_fn=embed_fn, rerank_fn=rerank_fn,
                             extractor=TemporalExtractorV3_6())
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
    print(f"\n=== Shipped planner + v3.6 surgical extractor vs shipped "
          f"(v3.3) over {len(BENCH_NAMES)} benches ===\n", flush=True)
    hdr = (f"{'bench':30s}  {'ship R@1':>9s} {'v3.6 R@1':>9s} {'ΔR@1':>7s}  "
           f"{'ship R@5':>9s} {'v3.6 R@5':>9s}  {'n':>4s}")
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
