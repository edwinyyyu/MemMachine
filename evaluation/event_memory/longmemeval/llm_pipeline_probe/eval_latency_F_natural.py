"""Latency benchmark for F-natural across model x reasoning x input size.

For each cell, run on three input sizes:
  - small  (~600 chars, single LLM call)
  - medium (~2500 chars, single LLM call)
  - large  (~6000 chars, single LLM call, near the windowing threshold)
  - huge   (~13000 chars, triggers windowing → parallel sub-calls)

Reports per-cell median, p95, and max wall-clock seconds. Repeats k times
per (cell, size) so we see variance, not a one-shot anecdote.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import time

import openai
from dotenv import load_dotenv
from probe_segmenter_F_natural import segment
from sample_chunks import collect

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


def pick_inputs() -> dict[str, str]:
    """Return one input per size bucket from the corpus."""
    bins = collect()
    pool = sorted({t for ts in bins.values() for t in ts}, key=len)
    by_target = {}
    for tag, target in [("small", 700), ("medium", 2500), ("large", 6000)]:
        # closest input ≥ target
        candidates = [t for t in pool if len(t) >= target]
        by_target[tag] = candidates[0] if candidates else pool[-1]
    # huge: concatenate to exceed window threshold
    long_chunks = sorted([t for t in pool if len(t) > 1500], key=len, reverse=True)
    by_target["huge"] = "\n\n".join(long_chunks[:6])[:13000]
    return by_target


CELLS = [
    ("gpt-5-nano", "low"),
    ("gpt-5-nano", "medium"),
    ("gpt-5.4-nano", "low"),
    ("gpt-5.4-nano", "medium"),
    ("gpt-4o-mini", ""),
    ("gpt-4.1-nano", ""),
]


async def time_one(client, model, reasoning, text):
    t0 = time.monotonic()
    segs = await segment(client, model, text, reasoning)
    return time.monotonic() - t0, len(segs)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()

    inputs = pick_inputs()
    print("Inputs: " + ", ".join(f"{k}={len(v)}ch" for k, v in inputs.items()))
    print(f"Reps per (cell, size): {args.reps}")
    print()
    print(
        f"{'cell':24s} | {'size':6s} | {'med_s':>5s} {'p95_s':>5s} {'max_s':>5s} {'segs':>4s}"
    )
    print("-" * 72)

    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    for model, reasoning in CELLS:
        for size_label, text in inputs.items():
            durations = []
            n_segs = 0
            for _ in range(args.reps):
                d, n = await time_one(client, model, reasoning, text)
                durations.append(d)
                n_segs = n
            med = statistics.median(durations)
            mx = max(durations)
            p95 = sorted(durations)[max(0, int(0.95 * len(durations)) - 1)]
            cell = f"{model} {reasoning or '(default)'}"
            print(
                f"{cell:24s} | {size_label:6s} | "
                f"{med:>5.2f} {p95:>5.2f} {mx:>5.2f} "
                f"{n_segs:>4d}"
            )
        print("-" * 72)
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
