"""Measure per-stage latency of the temporal_retrieval pipeline.

Splits cost into:
  - Index-time: doc extraction (v3 single-pass extractor per doc).
  - Query-time:
    - planner LLM call
    - per-leaf extractor call (the anchor resolver)
    - retrieval (cosine + rerank + scoring)

Each phase is timed with cold and warm cache for honest cost picture.
Uses the small speculative_anchors bench (28 docs / 12 queries) and
isolated cache subdir so the cold pass actually triggers LLM calls.

Run from ``evaluation/``::

    uv run python -m temporal_retrieval.research._measure_latency
"""

from __future__ import annotations

import asyncio
import shutil
import time

from temporal_retrieval import Doc, TemporalRetriever
from temporal_retrieval.extractor_v3 import TemporalExtractorV3
from temporal_retrieval.planner import QueryPlanner

from ._common import (
    ROOT,
    load_bench_jsonl,
    make_embed_fn,
    make_rerank_fn,
    setup_env,
)

setup_env()


def fresh_cache(subdir: str) -> None:
    """Wipe the named cache subdir so the next run is cold."""
    cache_dir = ROOT / "cache" / subdir
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


async def measure(label: str, cache_subdir: str) -> dict:
    print(f"\n=== {label} ===", flush=True)
    embed_fn = await make_embed_fn()
    rerank_fn = await make_rerank_fn()

    docs_jsonl, queries, _ = load_bench_jsonl(
        "speculative_anchors_docs.jsonl",
        "speculative_anchors_queries.jsonl",
        "speculative_anchors_gold.jsonl",
    )
    docs = [
        Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
        for d in docs_jsonl
    ]

    planner = QueryPlanner(cache_subdir=f"{cache_subdir}/planner")
    extractor = TemporalExtractorV3()
    retriever = TemporalRetriever(
        embed_fn=embed_fn,
        rerank_fn=rerank_fn,
        planner=planner,
        extractor=extractor,
    )

    # ----- Indexing -----
    t0 = time.perf_counter()
    await retriever.index(docs)
    index_total = time.perf_counter() - t0
    print(
        f"index({len(docs)} docs): {index_total:.2f}s total = "
        f"{index_total / len(docs) * 1000:.0f} ms/doc",
        flush=True,
    )

    # ----- Querying -----
    per_query_total: list[float] = []
    per_query_plan: list[float] = []
    per_query_retrieve: list[float] = []

    for q in queries:
        t_q0 = time.perf_counter()

        tp0 = time.perf_counter()
        await planner.plan(q["text"], q["ref_time"])
        tp1 = time.perf_counter()
        per_query_plan.append(tp1 - tp0)

        # Full query covers plan+leaf-extract+retrieve, but the plan is
        # cached from above so this measures the remainder.
        tr0 = time.perf_counter()
        await retriever.query(q["text"], q["ref_time"], k=10)
        tr1 = time.perf_counter()
        per_query_retrieve.append(tr1 - tr0)

        per_query_total.append(time.perf_counter() - t_q0)

    def stats(name: str, vals: list[float]) -> None:
        n = len(vals)
        if not n:
            print(f"{name}: n=0", flush=True)
            return
        vals_sorted = sorted(vals)
        mean = sum(vals) / n
        median = vals_sorted[n // 2]
        p95 = vals_sorted[min(n - 1, int(n * 0.95))]
        print(
            f"{name}: n={n} mean={mean * 1000:.0f}ms "
            f"median={median * 1000:.0f}ms p95={p95 * 1000:.0f}ms",
            flush=True,
        )

    stats("  plan        ", per_query_plan)
    stats("  retrieve    ", per_query_retrieve)
    stats("  TOTAL/query ", per_query_total)
    return {
        "label": label,
        "index_total_s": index_total,
        "n_docs": len(docs),
        "n_queries": len(queries),
        "plan_mean_ms": sum(per_query_plan) / max(1, len(per_query_plan)) * 1000,
        "retrieve_mean_ms": sum(per_query_retrieve) / max(1, len(per_query_retrieve)) * 1000,
    }


async def main() -> None:
    sub = "latency_measure"
    # Wipe to get a cold pass.
    fresh_cache(sub)
    await measure("COLD (no cache)", sub)
    # Re-run with warm cache.
    await measure("WARM (cache hits)", sub)


if __name__ == "__main__":
    asyncio.run(main())
