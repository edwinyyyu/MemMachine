"""Latency benchmark for the v30 segmenter prompt across the 3 candidate
models x 2 reasoning levels.

Quality is already known-equivalent at low reasoning (135/140 on each of
gpt-5-nano, gpt-5.4-nano, gpt-5-mini per the feedback bench). This
probe measures wall-clock latency on representative input sizes so the
production default can be picked on latency, not just quality.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import time

import openai
from dotenv import load_dotenv
from probe_segmenter_F_natural_v30 import segment as segment_v30

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


SHORT_TEXT = (
    "Hi team, As discussed, please find attached the proposal. "
    "Project budget is $45k. Deadline is March 30. Let me know if "
    "any questions. Thanks, Sarah"
)

MEDIUM_TEXT = (
    "Yesterday I drove out to the new Tesla store in Burlingame to take a "
    "Model Y for a test drive. The salesperson was a guy named Marcus -- "
    "he had just transferred from the Fremont store after a reorg. We "
    "took the long way through Highway 280 because he wanted to show me "
    "how the autopilot handled curves at 65mph. It actually held the "
    "lane really well, smoother than I expected. The acceleration off "
    "the line is honestly absurd -- 0-60 in 4.8 seconds for the "
    "Long Range trim. Marcus said most people who test drive end up "
    "ordering the Performance trim but he didn't push it. Total visit "
    "was about 90 minutes. I'm seriously considering the order, but I "
    "want to think about the interest rate -- 7.99% on the 72-month "
    "loan is a lot more than I'd like. Going to sleep on it for a week "
    "and re-evaluate after we see the next Fed meeting on March 19."
)


CELLS = [
    ("gpt-5-nano", "low"),
    ("gpt-5-nano", "medium"),
    ("gpt-5.4-nano", "low"),
    ("gpt-5.4-nano", "medium"),
    ("gpt-5-mini", "low"),
    ("gpt-5-mini", "medium"),
]


async def time_one(client, model, reasoning, text):
    t0 = time.monotonic()
    segs = await segment_v30(client, model, text, reasoning)
    return time.monotonic() - t0, len(segs)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()

    inputs = {"short": SHORT_TEXT, "medium": MEDIUM_TEXT}
    print("Inputs: " + ", ".join(f"{k}={len(v)}ch" for k, v in inputs.items()))
    print(f"Reps per (cell, size): {args.reps}")
    print()
    print(
        f"{'cell':28s} | {'size':6s} | "
        f"{'med_s':>6s} {'p95_s':>6s} {'max_s':>6s} {'segs':>4s}"
    )
    print("-" * 76)

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
            cell = f"{model} @ {reasoning}"
            print(
                f"{cell:28s} | {size_label:6s} | "
                f"{med:>6.2f} {p95:>6.2f} {mx:>6.2f} {n_segs:>4d}"
            )
        print("-" * 76)
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
