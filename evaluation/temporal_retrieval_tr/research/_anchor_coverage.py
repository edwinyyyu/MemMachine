"""How many docs have extracted anchors vs are timeless, per bench?

Answers: in Copeland-pairwise's 2x2 rule, the heterogeneous case
(timed-vs-timeless) only matters if there are timeless docs in the pool.
If anchor coverage is near 100%, the 2x2 rule collapses to "always use
base+match" — which would make it indistinguishable from additive.

If anchor coverage is mixed, the heterogeneous rule structurally
differs from additive (and from base_only).
"""
from __future__ import annotations

import asyncio

from temporal_retrieval_tr import Doc, TemporalExtractor
from temporal_retrieval.research._common import setup_env
from temporal_retrieval_tr.research.bench import BENCH_NAMES, load_bench

setup_env()


async def main():
    extractor = TemporalExtractor()
    total_docs = 0
    total_timed = 0
    print(f"{'bench':30s}  {'n':>4s}  {'timed':>5s}  {'%timed':>7s}")
    print("-" * 55)
    for bench in BENCH_NAMES:
        loaded = load_bench(bench)
        if loaded[0] is None:
            continue
        docs_jsonl, _, _ = loaded
        from temporal_retrieval_min.schema import parse_iso
        async def _ext(d):
            return await extractor.extract_anchors(
                d["text"], parse_iso(d["ref_time"])
            )
        anchors_list = await asyncio.gather(*(_ext(d) for d in docs_jsonl))
        n = len(docs_jsonl)
        n_timed = sum(1 for a in anchors_list if a)
        pct = 100.0 * n_timed / n if n else 0.0
        print(f"{bench:30s}  {n:>4d}  {n_timed:>5d}  {pct:>6.1f}%")
        total_docs += n
        total_timed += n_timed
    pct_total = 100.0 * total_timed / total_docs
    print("-" * 55)
    print(f"{'TOTAL':30s}  {total_docs:>4d}  {total_timed:>5d}  {pct_total:>6.1f}%")
    extractor.save_caches()


if __name__ == "__main__":
    asyncio.run(main())
