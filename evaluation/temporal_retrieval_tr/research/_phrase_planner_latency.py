"""Latency probe for the minimal phrase planner."""
from __future__ import annotations

import asyncio, statistics

from temporal_retrieval.research._common import setup_env
from temporal_retrieval_tr.research._phrase_planner import PhrasePlanner
from temporal_retrieval_tr.research.bench import load_bench

setup_env()


SAMPLE_QUERY_BENCHES = [
    "hard_bench", "composition", "negation_temporal",
    "edge_conjunctive_temporal", "same_topic_recency", "open_ended_date",
]


def _build_sample():
    samples = []
    seen = set()
    for bench in SAMPLE_QUERY_BENCHES:
        loaded = load_bench(bench)
        if loaded[0] is None:
            continue
        _, queries, _ = loaded
        n = 0
        for q in queries:
            if q["text"] in seen:
                continue
            seen.add(q["text"])
            samples.append((q["text"], q["ref_time"]))
            n += 1
            if n >= 4:
                break
    return samples[:24]


async def time_planner(label, planner, samples):
    print(f"\n=== {label} ===", flush=True)
    latencies = []
    fails = 0
    for q, ref in samples:
        import time
        t0 = time.perf_counter()
        try:
            await planner.plan(q, ref)
        except Exception:
            fails += 1
        latencies.append(time.perf_counter() - t0)
    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[int(0.95 * len(latencies))]
    mean = sum(latencies) / len(latencies)
    print(
        f"  p50={p50*1000:.0f}ms  p95={p95*1000:.0f}ms  mean={mean*1000:.0f}ms  "
        f"fails={fails}/{len(latencies)}",
        flush=True,
    )


async def main():
    samples = _build_sample()
    print(f"Probe: {len(samples)} queries (sequential, no cache, includes Duckling+code roundtrip)\n",
          flush=True)
    for label, model, effort in [
        ("PhrasePlanner gpt-5-nano @ minimal", "gpt-5-nano", "minimal"),
        ("PhrasePlanner gpt-4.1-nano",         "gpt-4.1-nano", None),
        ("PhrasePlanner gpt-4o-mini",          "gpt-4o-mini", None),
    ]:
        # Use fresh planner per config to avoid cache hits.
        from pathlib import Path
        cache_file = Path(__file__).resolve().parent.parent / "cache" / "phrase_planner" / f"{model}_{effort}_cache.json"
        if cache_file.exists():
            cache_file.unlink()
        planner = PhrasePlanner(model=model, reasoning_effort=effort)
        await time_planner(label, planner, samples)


if __name__ == "__main__":
    asyncio.run(main())
