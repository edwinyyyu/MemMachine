"""Latency probe: planner LLM across OpenAI model choices.

Reports per-model p50 / p95 / mean latency and any parse failures
on a fixed 20-query sample. Cold-start; no cache.

Models tested:
- gpt-5-mini @ medium (current production)
- gpt-5-mini @ low
- gpt-5-nano @ medium
- gpt-5-nano @ low
- gpt-5-nano @ minimal
- gpt-4.1-mini
- gpt-4.1-nano
- gpt-4o-mini
"""
from __future__ import annotations

import asyncio, json, os, statistics, time
from pathlib import Path

from openai import AsyncOpenAI
from openai.types.responses.response_format_text_json_schema_config_param import (
    ResponseFormatTextJSONSchemaConfigParam,
)
from openai.types.responses import ResponseTextConfigParam

from temporal_retrieval.research._common import setup_env
from temporal_retrieval_tr.planner import _PLAN_JSON_SCHEMA, PROMPT as _PROMPT
from temporal_retrieval_tr.research.bench import BENCH_NAMES, load_bench

setup_env()

if not os.environ.get("OPENAI_API_KEY"):
    from dotenv import load_dotenv
    load_dotenv()


# 20 representative queries covering simple dates, extremum, negation,
# quarters, conjunctions, and non-temporal patterns.
SAMPLE_QUERY_BENCHES = [
    "hard_bench", "composition", "negation_temporal",
    "edge_conjunctive_temporal", "same_topic_recency", "open_ended_date",
]


def _build_sample() -> list[tuple[str, str]]:
    """Return ~20 (query, ref_time) pairs covering varied difficulty."""
    samples: list[tuple[str, str]] = []
    seen = set()
    for bench in SAMPLE_QUERY_BENCHES:
        loaded = load_bench(bench)
        if loaded[0] is None:
            continue
        _, queries, _ = loaded
        # Pick first 4 unique queries from each
        n = 0
        for q in queries:
            key = q["text"]
            if key in seen:
                continue
            seen.add(key)
            samples.append((q["text"], q["ref_time"]))
            n += 1
            if n >= 4:
                break
    return samples[:24]  # ~24 queries


CONFIGS = [
    # (label, model, reasoning_effort)
    ("gpt-5-mini @ medium (current)", "gpt-5-mini",        "medium"),
    ("gpt-5-mini @ low",              "gpt-5-mini",        "low"),
    ("gpt-5-nano @ medium",           "gpt-5-nano",        "medium"),
    ("gpt-5-nano @ low",              "gpt-5-nano",        "low"),
    ("gpt-5-nano @ minimal",          "gpt-5-nano",        "minimal"),
    ("gpt-4.1-mini",                  "gpt-4.1-mini",      None),
    ("gpt-4.1-nano",                  "gpt-4.1-nano",      None),
    ("gpt-4o-mini",                   "gpt-4o-mini",       None),
]


async def time_one(client, model, reasoning_effort, query, ref_time) -> tuple[float, bool]:
    prompt = _PROMPT.format(query=query, ref_time=ref_time)
    format_config: ResponseFormatTextJSONSchemaConfigParam = {
        "type": "json_schema", "name": "plan", "strict": True,
        "schema": _PLAN_JSON_SCHEMA,
    }
    text_config: ResponseTextConfigParam = {"format": format_config}
    kwargs = dict(model=model, input=prompt, text=text_config)
    if reasoning_effort is not None:
        kwargs["reasoning"] = {"effort": reasoning_effort}
    t0 = time.perf_counter()
    try:
        resp = await client.responses.create(**kwargs)
        json.loads(resp.output_text)
        return (time.perf_counter() - t0), True
    except Exception as e:
        return (time.perf_counter() - t0), False


async def probe_model(client, label, model, effort, samples):
    print(f"\n=== {label} ===", flush=True)
    latencies = []
    failures = 0
    # Run sequentially (sequential = realistic per-query latency).
    for q, ref in samples:
        lat, ok = await time_one(client, model, effort, q, ref)
        latencies.append(lat)
        if not ok:
            failures += 1
    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[int(0.95 * len(latencies))]
    mean = sum(latencies) / len(latencies)
    print(
        f"  p50={p50*1000:.0f}ms  p95={p95*1000:.0f}ms  mean={mean*1000:.0f}ms  "
        f"failures={failures}/{len(latencies)}",
        flush=True,
    )
    return label, p50, p95, mean, failures


async def main():
    samples = _build_sample()
    print(f"Probe: {len(samples)} queries (sequential, cold cache)\n", flush=True)
    client = AsyncOpenAI()
    summary = []
    for label, model, effort in CONFIGS:
        try:
            r = await probe_model(client, label, model, effort, samples)
            summary.append(r)
        except Exception as e:
            print(f"  ERROR ({label}): {e}", flush=True)
            summary.append((label, 0, 0, 0, len(samples)))
    print("\n=== Summary (sorted by p50) ===")
    summary.sort(key=lambda r: r[1])
    print(f"  {'Config':35s}  {'p50':>8s}  {'p95':>8s}  {'mean':>8s}  {'fail':>6s}")
    for label, p50, p95, mean, fail in summary:
        print(
            f"  {label:35s}  {p50*1000:>6.0f}ms  {p95*1000:>6.0f}ms  "
            f"{mean*1000:>6.0f}ms  {fail:>6d}",
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(main())
