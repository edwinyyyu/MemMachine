"""Fine W sweep: find the best balance point in additive linear-rank.

From the same-topic diagnostic and the q_cot_2_a flip math, the
plausible balance is W ∈ [1.0, 1.2) — captures cheap+medium
composition flips while staying below the cotemporal threshold
(W ≈ 1.198 from gold's tied-anchor r=0.222 vs competitor r=1.0).

This sweep at fine granularity to find the cusp.

Run from `evaluation/`:
    uv run python -m temporal_retrieval_tr.research._w_fine_sweep
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

WEIGHTS = [0.5, 0.75, 1.0, 1.10, 1.15, 1.20, 1.25, 1.30, 1.40, 1.50]


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
    for w in WEIGHTS:
        vd.recency_weight = w
        vd.copeland_bonus = None
        rk = {}
        for q in queries:
            r = await vd.query(q["text"], q["ref_time"], k=10)
            rk[q["query_id"]] = [x.doc_id for x in r]
        out[w] = metrics(rk, gold)
    del vd, docs, docs_jsonl
    gc.collect()
    return out


async def main() -> None:
    raw_embed = await make_embed_fn()
    embed_fn = make_cached_embed_fn(raw_embed)
    rerank_fn = make_cosine_rerank_fn(embed_fn)
    print(f"=== Fine W sweep {WEIGHTS} over {len(BENCH_NAMES)} benches ===\n",
          flush=True)
    wcols = "  ".join(f"{w:>5.2f}" for w in WEIGHTS)
    hdr = f"{'bench':28s}  {wcols}  {'n':>4s}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    rows = {}
    key = {"composition", "cotemporal"}
    for bench in BENCH_NAMES:
        try:
            res = await run_bench(bench, embed_fn, rerank_fn)
        except Exception as e:
            print(f"{bench:28s}  ERROR: {e}", flush=True)
            continue
        if res is None:
            print(f"{bench:28s}  SKIPPED", flush=True)
            continue
        rows[bench] = res
        vals = "  ".join(f"{res[w]['R@1']:>5.3f}" for w in WEIGHTS)
        mark = ">" if bench in key else " "
        print(f"{mark} {bench:26s}  {vals}  {res[WEIGHTS[0]]['n']:>4d}",
              flush=True)
    if rows:
        n = len(rows)
        print("-" * len(hdr), flush=True)
        for k_metric in ("R@1", "R@5", "R@10"):
            macro = [sum(r[w].get(k_metric, 0) for r in rows.values()) / n
                     for w in WEIGHTS]
            vals = "  ".join(f"{v:>5.3f}" for v in macro)
            print(f"  {'MACRO ' + k_metric:26s}  {vals}  n={n}", flush=True)
        # Key bench detail at R@1 AND R@5 (R@K is the user's real target)
        print("\n=== Key benches ===")
        for kb in ("composition", "cotemporal"):
            if kb not in rows:
                continue
            for k_metric in ("R@1", "R@5", "R@10"):
                cells = "  ".join(f"{rows[kb][w].get(k_metric, 0):>5.3f}"
                                  for w in WEIGHTS)
                print(f"  {kb + ' ' + k_metric:26s}  {cells}"
                      f"  n={rows[kb][WEIGHTS[0]]['n']}", flush=True)
        # Pick best W on R@1, R@5, R@10 separately
        for k_metric in ("R@1", "R@5", "R@10"):
            macro = [sum(r[w].get(k_metric, 0) for r in rows.values()) / n
                     for w in WEIGHTS]
            best_i = max(range(len(WEIGHTS)), key=lambda i: macro[i])
            print(f"\nBest W on macro {k_metric} = {WEIGHTS[best_i]}: "
                  f"{macro[best_i]:.4f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
