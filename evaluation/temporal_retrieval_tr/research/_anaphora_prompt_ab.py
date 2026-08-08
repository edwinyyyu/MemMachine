"""A/B: production planner (with anaphora) vs no-anaphora planner.

Tests whether the anaphora prompt section silently helps target extraction
or is dead weight (anaphora field is not consumed by the retriever).

Run from `evaluation/`:
    uv run python -m temporal_retrieval_tr.research._anaphora_prompt_ab
"""
from __future__ import annotations

import asyncio
import gc

from temporal_retrieval_tr import Doc, TemporalRetriever
from temporal_retrieval.research._common import make_embed_fn, setup_env
from temporal_retrieval_tr.research.bench import (
    BENCH_NAMES, load_bench, make_cached_embed_fn, make_cosine_rerank_fn, metrics,
)
from temporal_retrieval_tr.research._no_anaphora_planner import NoAnaphoraPlanner

setup_env()


async def run_bench(bench: str, embed_fn, rerank_fn) -> tuple[dict, dict] | None:
    loaded = load_bench(bench)
    if loaded[0] is None:
        return None
    docs_jsonl, queries, gold = loaded
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]

    # Baseline: production planner (with anaphora)
    vd_base = TemporalRetriever(embed_fn=embed_fn, rerank_fn=rerank_fn)
    await vd_base.index(docs)
    rk_base = {}
    for q in queries:
        r = await vd_base.query(q["text"], q["ref_time"], k=10)
        rk_base[q["query_id"]] = [x.doc_id for x in r]
    m_base = metrics(rk_base, gold)

    # No-anaphora arm
    vd_na = TemporalRetriever(
        embed_fn=embed_fn,
        rerank_fn=rerank_fn,
        planner=NoAnaphoraPlanner(),
    )
    await vd_na.index(docs)
    rk_na = {}
    for q in queries:
        r = await vd_na.query(q["text"], q["ref_time"], k=10)
        rk_na[q["query_id"]] = [x.doc_id for x in r]
    m_na = metrics(rk_na, gold)

    del vd_base, vd_na, docs
    gc.collect()
    return m_base, m_na


async def main() -> None:
    raw = await make_embed_fn()
    embed_fn = make_cached_embed_fn(raw)
    rerank_fn = make_cosine_rerank_fn(embed_fn)
    print(f"\n=== Anaphora prompt A/B over {len(BENCH_NAMES)} benches ===\n", flush=True)
    hdr = (f"{'bench':30s}  {'base R@1':>8s} {'na R@1':>8s} {'ΔR@1':>7s}  "
           f"{'base R@5':>8s} {'na R@5':>8s}  {'n':>4s}")
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    rows = {}
    for bench in BENCH_NAMES:
        try:
            result = await run_bench(bench, embed_fn, rerank_fn)
        except Exception as e:
            print(f"{bench:30s}  ERROR: {e}", flush=True)
            continue
        if result is None:
            continue
        mb, mn = result
        rows[bench] = (mb, mn)
        d_r1 = mn["R@1"] - mb["R@1"]
        mark = ">" if abs(d_r1) >= 0.02 else " "
        print(f"{mark} {bench:28s}  {mb['R@1']:>8.3f} {mn['R@1']:>8.3f} {d_r1:>+7.3f}  "
              f"{mb['R@5']:>8.3f} {mn['R@5']:>8.3f}  {mb['n']:>4d}",
              flush=True)
    if rows:
        n = len(rows)
        macro_b_r1 = sum(r[0]["R@1"] for r in rows.values()) / n
        macro_n_r1 = sum(r[1]["R@1"] for r in rows.values()) / n
        macro_b_r5 = sum(r[0]["R@5"] for r in rows.values()) / n
        macro_n_r5 = sum(r[1]["R@5"] for r in rows.values()) / n
        print("-" * len(hdr), flush=True)
        print(f"  {'MACRO':28s}  {macro_b_r1:>8.4f} {macro_n_r1:>8.4f} "
              f"{macro_n_r1 - macro_b_r1:>+7.4f}  "
              f"{macro_b_r5:>8.4f} {macro_n_r5:>8.4f}  n={n}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
