"""Measure actual pool cosine range to validate the 2× amplification estimate.

For each query in our same-topic benches, compute the top-40 pool from
raw cosine, then report (pool_max_cos - pool_min_cos). The amplification
factor for normalize_dict is 1 / pool_range.
"""
from __future__ import annotations

import asyncio
import json
import statistics

import numpy as np

from temporal_retrieval.research._common import DATA_DIR, make_embed_fn, setup_env
from temporal_retrieval_tr.research.bench import make_cached_embed_fn

setup_env()

BENCHES = [
    "composition", "same_topic_recency", "same_topic_recency_hard",
    "recency_stress_deep", "recency_vs_rerank",
]


def cosine(a, b) -> float:
    na = float(np.linalg.norm(a)) or 1e-9
    nb = float(np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / (na * nb))


async def main() -> None:
    raw_embed = await make_embed_fn()
    embed_fn = make_cached_embed_fn(raw_embed)
    pool_size = 40

    all_ranges = []
    bench_ranges = {}
    for bench in BENCHES:
        try:
            with open(DATA_DIR / f"{bench}_docs.jsonl") as f:
                docs = [json.loads(line) for line in f]
            with open(DATA_DIR / f"{bench}_queries.jsonl") as f:
                queries = [json.loads(line) for line in f]
        except FileNotFoundError:
            continue
        doc_text = [d["text"] for d in docs]
        doc_embs = await embed_fn(doc_text)
        q_text = [q["text"] for q in queries]
        q_embs = await embed_fn(q_text)
        ranges = []
        for qe in q_embs:
            sims = sorted([cosine(qe, de) for de in doc_embs], reverse=True)
            pool_sims = sims[:pool_size]
            if len(pool_sims) >= 2:
                ranges.append(pool_sims[0] - pool_sims[-1])
        bench_ranges[bench] = ranges
        all_ranges.extend(ranges)

    print("=== Pool cosine range (max − min, pool_size=40) ===\n")
    print(f"{'bench':28s}  {'min':>6s}  {'p25':>6s}  {'p50':>6s}  {'p75':>6s}  "
          f"{'max':>6s}  {'mean':>6s}  {'avg amp':>8s}")
    print("-" * 90)
    for bench, ranges in bench_ranges.items():
        if not ranges:
            continue
        r_min, r_max = min(ranges), max(ranges)
        med = statistics.median(ranges)
        mean = statistics.mean(ranges)
        if len(ranges) >= 4:
            p25 = statistics.quantiles(ranges, n=4)[0]
            p75 = statistics.quantiles(ranges, n=4)[2]
        else:
            p25 = p75 = med
        amp = 1.0 / mean
        print(f"{bench:28s}  {r_min:>6.3f}  {p25:>6.3f}  {med:>6.3f}  {p75:>6.3f}  "
              f"{r_max:>6.3f}  {mean:>6.3f}  {amp:>8.2f}x")

    print("-" * 90)
    if all_ranges:
        r_min, r_max = min(all_ranges), max(all_ranges)
        med = statistics.median(all_ranges)
        mean = statistics.mean(all_ranges)
        p25 = statistics.quantiles(all_ranges, n=4)[0]
        p75 = statistics.quantiles(all_ranges, n=4)[2]
        amp = 1.0 / mean
        print(f"{'ALL':28s}  {r_min:>6.3f}  {p25:>6.3f}  {med:>6.3f}  {p75:>6.3f}  "
              f"{r_max:>6.3f}  {mean:>6.3f}  {amp:>8.2f}x")


if __name__ == "__main__":
    asyncio.run(main())
