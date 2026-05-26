"""Pool size × scoring mechanism sweep — discriminate additive vs Copeland.

Theoretical prediction:
- Additive uses rank-based recency r_v ∈ [0, 1] over the pool. Per-pair
  gap between adjacent recency ranks = W/(N-1). At W=0.5, N=100, that's
  ~0.005 — vanishes as N grows.
- Copeland uses fixed per-pair bonus, independent of N.

Stress hypothesis: with many same-topic competitors compressed in the
pool, additive should degrade as N grows while Copeland holds.

Test bench: recency_stress_deep — 17 scenarios, 8-10 same-topic docs
each (147 total), identical text within scenario, dates spanning 2-4
years. Plus same_topic_recency_hard for cross-validation.

Run from `evaluation/`:
    uv run python -m temporal_retrieval_tr.research._pool_size_sweep
"""
from __future__ import annotations

import asyncio
import gc

from temporal_retrieval_tr import Doc, TemporalRetriever

from temporal_retrieval.research._common import make_embed_fn, setup_env
from temporal_retrieval_tr.research.bench import (
    load_bench,
    make_cached_embed_fn,
    make_cosine_rerank_fn,
    metrics,
)

setup_env()

POOL_SIZES = [40, 80, 120, 200]

ARMS: list[tuple[str, str, float]] = [
    ("add_W0.5",  "additive", 0.5),
    ("add_W1.0",  "additive", 1.0),
    ("add_W1.5",  "additive", 1.5),
    ("cope_0.20", "copeland", 0.20),
    ("cope_0.30", "copeland", 0.30),
    ("cope_0.50", "copeland", 0.50),
]

# Stress-relevant benches
TARGET_BENCHES = [
    "recency_vs_rerank",
    "recency_stress_deep",
    "same_topic_recency_hard",
    "same_topic_recency",
    "composition",
    "cotemporal",
]


async def run_bench(bench: str, pool_size: int, embed_fn, rerank_fn) -> dict | None:
    loaded = load_bench(bench)
    if loaded[0] is None:
        return None
    docs_jsonl, queries, gold = loaded
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]
    vd = TemporalRetriever(
        embed_fn=embed_fn, rerank_fn=rerank_fn,
        pool_size=pool_size,
    )
    await vd.index(docs)
    out = {}
    for label, mode, value in ARMS:
        if mode == "additive":
            vd.recency_weight = value
            vd.copeland_bonus = None
        else:
            vd.copeland_bonus = value
        rk = {}
        for q in queries:
            r = await vd.query(q["text"], q["ref_time"], k=10)
            rk[q["query_id"]] = [x.doc_id for x in r]
        out[label] = metrics(rk, gold)
    del vd, docs, docs_jsonl
    gc.collect()
    return out


async def main() -> None:
    raw_embed = await make_embed_fn()
    embed_fn = make_cached_embed_fn(raw_embed)
    rerank_fn = make_cosine_rerank_fn(embed_fn)
    labels = [a[0] for a in ARMS]
    print(f"=== Pool-size × scoring sweep ===", flush=True)
    print(f"Pool sizes: {POOL_SIZES}", flush=True)
    print(f"Arms: {labels}\n", flush=True)

    results: dict[tuple[str, int], dict] = {}
    for bench in TARGET_BENCHES:
        for ps in POOL_SIZES:
            res = await run_bench(bench, ps, embed_fn, rerank_fn)
            if res is None:
                continue
            results[(bench, ps)] = res

    # Layout: for each bench, table of pool_size × arm
    for bench in TARGET_BENCHES:
        print(f"\n=== {bench} ===")
        bench_rows = [(ps, results[(bench, ps)]) for ps in POOL_SIZES
                      if (bench, ps) in results]
        if not bench_rows:
            print("  (no data)")
            continue
        n = bench_rows[0][1][labels[0]]["n"]
        print(f"  n = {n}\n")
        for metric in ("R@1", "R@5", "R@10"):
            cells = "  ".join(f"{L:>9s}" for L in labels)
            print(f"  {metric} | pool_size  {cells}")
            print(f"  {'-' * (12 + 11 * len(labels))}")
            for ps, row in bench_rows:
                vals = "  ".join(f"{row[L].get(metric, 0):>9.3f}" for L in labels)
                print(f"  {metric:5s}|   {ps:>5d}    {vals}")
            print()

    # Highlight differentiating cases
    print("\n=== Discrimination summary ===")
    print("Cases where additive(W=0.5) and Copeland(0.20) differ:")
    for bench in TARGET_BENCHES:
        for ps in POOL_SIZES:
            key = (bench, ps)
            if key not in results:
                continue
            add = results[key]["add_W0.5"]
            cop = results[key]["cope_0.20"]
            diff = cop["R@1"] - add["R@1"]
            if abs(diff) > 0.001:
                print(f"  {bench:30s} pool={ps:4d}  add={add['R@1']:.3f}  "
                      f"cope={cop['R@1']:.3f}  Δ={diff:+.3f}")


if __name__ == "__main__":
    asyncio.run(main())
