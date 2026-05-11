"""Quantify structural-content preservation with the pre-escape variant
of v33. Same cases as probe_segmenter_structural_quality.py for
direct comparison.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import time

import openai
from dotenv import load_dotenv
from probe_segmenter_v33_preescape import segment as segment_pre

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


CELLS = [
    ("gpt-5-nano", "low"),
    ("gpt-5-nano", "medium"),
    ("gpt-5.4-nano", "low"),
    ("gpt-5.4-nano", "medium"),
    ("gpt-5-mini", "low"),
]


ASCII_ART_SOURCE = (
    "I drew this for my daughter today:\n  /\\_/\\\n ( o.o )\n  > ^ <\nShe loved it."
)

ASCII_TABLE_SOURCE = (
    "Q3 model benchmarks:\n"
    "| Model | Task A | Task B |\n"
    "| --- | --- | --- |\n"
    "| GPT-4 | 0.85 | 0.78 |\n"
    "| Claude | 0.91 | 0.82 |\n"
    "| Gemini | 0.79 | 0.81 |\n"
    "Claude won on both tasks."
)

PYTHON_CODE_SOURCE = (
    "Here is the helper I wrote yesterday:\n"
    "def find_max(node):\n"
    "    if node is None:\n"
    "        return float('-inf')\n"
    "    return max(node.value, find_max(node.left), find_max(node.right))\n"
    "It handles the empty-tree case via -inf."
)

ASCII_ARROW_SOURCE = (
    "The pipeline flow:\n"
    "ingest --> segmenter --> deriver --> embedder --> vector_store\n"
    "Everything downstream of segmenter is parallel."
)

MORSE_SOURCE = (
    "The full message was:\n"
    ".... . .-.. .-.. --- / .-- --- .-. .-.. -..\n"
    "That's HELLO WORLD in morse."
)


CASES = [
    ("ascii_art", ASCII_ART_SOURCE, "/\\_/\\"),
    ("ascii_table", ASCII_TABLE_SOURCE, "GPT-4 | 0.85"),
    ("python_code", PYTHON_CODE_SOURCE, "def find_max(node):"),
    ("ascii_arrow", ASCII_ARROW_SOURCE, "ingest --> segmenter --> deriver"),
    ("morse", MORSE_SOURCE, ".... . .-.. .-.."),
]


async def run_one(client, model, reasoning, source, signature):
    t0 = time.monotonic()
    segs = await segment_pre(client, model, source, reasoning)
    elapsed = time.monotonic() - t0
    joined = " ".join(segs)
    return signature in joined, elapsed, len(segs)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    print(f"# variant=v33+preescape reps={args.reps}")
    print()

    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(args.concurrency)

    async def go(model, reasoning, source, signature):
        async with sem:
            return await run_one(client, model, reasoning, source, signature)

    header = f"{'cell':28s} | {'case':12s} | {'pass/reps':>10s} | {'med_s':>6s}"
    print(header)
    print("-" * len(header))
    for model, reasoning in CELLS:
        cell = f"{model} @ {reasoning}"
        for case_id, source, signature in CASES:
            tasks = [go(model, reasoning, source, signature) for _ in range(args.reps)]
            results = await asyncio.gather(*tasks)
            passes = sum(1 for ok, _, _ in results if ok)
            times = [t for _, t, _ in results]
            med = statistics.median(times)
            print(
                f"{cell:28s} | {case_id:12s} | "
                f"{passes:>4d}/{args.reps:<4d}  | {med:>6.2f}"
            )
        print("-" * len(header))
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
