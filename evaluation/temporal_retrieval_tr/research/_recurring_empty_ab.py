"""A/B: pure-subtract no-anaphora (current ship) vs recurring-empty (variant A).

Isolates the effect of the RECURRING PATTERNS → empty rule. Comparison
to original anaphora baseline is via prior bvvhe3qqe.output table.
"""
from __future__ import annotations

import asyncio, gc

from temporal_retrieval_tr import Doc, TemporalRetriever
from temporal_retrieval.research._common import make_embed_fn, setup_env
from temporal_retrieval_tr.research.bench import (
    BENCH_NAMES, load_bench, make_cached_embed_fn, make_cosine_rerank_fn, metrics,
)
from temporal_retrieval_tr.research._no_anaphora_planner import NoAnaphoraPlanner
from temporal_retrieval_tr.research._recurring_empty_planner import (
    RecurringEmptyPlanner,
)

setup_env()


async def run_bench(bench, embed_fn, rerank_fn):
    loaded = load_bench(bench)
    if loaded[0] is None:
        return None
    docs_jsonl, queries, gold = loaded
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]

    # A = pure-subtract no-anaphora (current ship-side state)
    vd_a = TemporalRetriever(embed_fn=embed_fn, rerank_fn=rerank_fn,
                             planner=NoAnaphoraPlanner())
    await vd_a.index(docs)
    rk_a = {q["query_id"]: [x.doc_id for x in
            await vd_a.query(q["text"], q["ref_time"], k=10)]
            for q in queries}
    m_a = metrics(rk_a, gold)

    # B = recurring-empty (variant A — adds RECURRING PATTERNS → empty)
    vd_b = TemporalRetriever(embed_fn=embed_fn, rerank_fn=rerank_fn,
                             planner=RecurringEmptyPlanner())
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
    print(f"\n=== Recurring-empty (variant A) vs pure-subtract no-anaphora "
          f"over {len(BENCH_NAMES)} benches ===\n", flush=True)
    hdr = (f"{'bench':30s}  {'no-ana R@1':>10s} {'recur R@1':>10s} {'ΔR@1':>7s}  "
           f"{'no-ana R@5':>10s} {'recur R@5':>10s}  {'n':>4s}")
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
        print(f"{mark} {bench:28s}  {ma['R@1']:>10.3f} {mb['R@1']:>10.3f} "
              f"{d:>+7.3f}  {ma['R@5']:>10.3f} {mb['R@5']:>10.3f}  {ma['n']:>4d}",
              flush=True)
    if rows:
        n = len(rows)
        ma_r1 = sum(r[0]["R@1"] for r in rows.values()) / n
        mb_r1 = sum(r[1]["R@1"] for r in rows.values()) / n
        ma_r5 = sum(r[0]["R@5"] for r in rows.values()) / n
        mb_r5 = sum(r[1]["R@5"] for r in rows.values()) / n
        print("-" * len(hdr), flush=True)
        print(f"  {'MACRO':28s}  {ma_r1:>10.4f} {mb_r1:>10.4f} {mb_r1 - ma_r1:>+7.4f}  "
              f"{ma_r5:>10.4f} {mb_r5:>10.4f}  n={n}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
