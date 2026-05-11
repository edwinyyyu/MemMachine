"""Side-by-side qualitative comparison of gpt-5-nano vs gpt-5-mini at low
reasoning on v64 prompt.

Both passed 52/52 on the bench. This compares output STYLE: derivative
count, specific wording, compound-id preservation, atomization behavior.
"""

from __future__ import annotations

import asyncio
import os

import openai
from dotenv import load_dotenv
from probe_deriver_v64_compound_ids import derive

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


SEGMENTS = [
    (
        "B1_trip_with_coactors",
        "Last March I went to Tokyo with my wife Anne and our dog Mochi. "
        "We stayed at the Park Hyatt for 5 nights and ate ramen at Ichiran in Shibuya.",
    ),
    (
        "D1_long_single_subject",
        "We finally finished restoring the 1967 Ford Mustang my grandfather left me. "
        "It took almost three years from the day we towed it out of the barn in "
        "upstate New York. The engine block had completely seized up from a decade "
        "of moisture, so we ended up rebuilding the 289 V8 from scratch, with new "
        "pistons, rings, and a Holley four-barrel carburetor. Bodywork was the "
        "worst part. We went with the original Wimbledon White over a black "
        "interior. We rebuilt the suspension with new bushings and put in a "
        "Borg-Warner T-10 four-speed transmission. Last weekend I finally drove "
        "it to the Hudson Valley cars-and-coffee.",
    ),
    (
        "PR2_treehouse_with_friend",
        "My friend Sarah and I spent Saturday building a treehouse in her backyard. "
        "We used pine for the frame and cedar for the panels. We finished in the "
        "late afternoon and her dog wouldn't stop barking at it.",
    ),
    (
        "P1_preference",
        "I've always preferred pour-over coffee from Stumptown over chain shops, "
        "especially their Hair Bender blend. I find chain shop espresso bitter and "
        "over-extracted; Stumptown's hand-pour technique with a 30-second bloom "
        "brings out chocolate and citrus notes I actually look forward to in "
        "the morning.",
    ),
    (
        "F2_focused_statement",
        "In chess, the move 25.g4 is played to gain space on the kingside "
        "and restrict the opponent's pawn structure.",
    ),
    (
        "FIL3_affirmative_with_content",
        "ok, leaving Tuesday at 5 for the Boston trip",
    ),
    (
        "BARE_term",
        "Tokyo",
    ),
]


CONFIGS = [
    ("gpt-5-nano", "low"),
    ("gpt-5-mini", "low"),
]

N_TRIALS_PER_SEGMENT = 2  # check determinism / variance


async def main():
    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(6)

    async def one(seg_label, segment, model, reasoning, trial):
        async with sem:
            derivs = await derive(client, segment, model=model, reasoning=reasoning)
            return (seg_label, model, reasoning, trial, derivs)

    tasks = []
    for seg_label, segment in SEGMENTS:
        for model, reasoning in CONFIGS:
            for trial in range(N_TRIALS_PER_SEGMENT):
                tasks.append(one(seg_label, segment, model, reasoning, trial))

    results = await asyncio.gather(*tasks)

    # Group by (seg, model)
    by_key: dict = {}
    for seg_label, model, reasoning, trial, derivs in results:
        by_key.setdefault((seg_label, model, reasoning), []).append((trial, derivs))

    for seg_label, segment in SEGMENTS:
        print(f"\n{'=' * 78}")
        print(f"SEGMENT: {seg_label}")
        print(f"{'=' * 78}")
        print(f"Input: {segment[:200]}{'...' if len(segment) > 200 else ''}")
        print()
        for model, reasoning in CONFIGS:
            trials = by_key.get((seg_label, model, reasoning), [])
            for trial, derivs in trials:
                config = f"{model}@{reasoning}"
                print(f"--- {config} (trial {trial}) [n={len(derivs)}] ---")
                for d in derivs:
                    print(f"  - {d}")
                print()

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
