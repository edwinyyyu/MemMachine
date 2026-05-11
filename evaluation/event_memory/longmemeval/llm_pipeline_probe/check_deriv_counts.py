"""Quick check of v65 derivative counts across diverse segments.

Tests whether the model is:
  - Returning ~1 derivative ~= the verbatim segment (lazy / wasteful)
  - Splitting multi-subject content into independent derivatives (good)
  - Compressing long narratives to one detail-preserving derivative (good)
"""

from __future__ import annotations

import asyncio
import os

import openai
from dotenv import load_dotenv
from probe_deriver_v65_completeness import derive

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


SEGMENTS = [
    # Single-subject, short
    (
        "ST1_single_short",
        "Last March I went to Tokyo with my wife Anne and our dog Mochi. "
        "We stayed at the Park Hyatt for 5 nights and ate ramen at Ichiran in Shibuya.",
    ),
    # Single-subject, long (D1)
    (
        "LO1_long_narrative",
        "We finally finished restoring the 1967 Ford Mustang my grandfather left me. "
        "It took almost three years from the day we towed it out of the barn in "
        "upstate New York. We rebuilt the 289 V8 with new pistons, rings, and a "
        "Holley four-barrel carburetor. We went with the original Wimbledon White "
        "over a black interior. Last weekend I finally drove it to the Hudson Valley "
        "cars-and-coffee.",
    ),
    # Multi-subject (deliberate)
    (
        "MS1_multi_subject_two_topics",
        "My team launched the new product on Friday. Also, I've been meaning to tell "
        "you I'm interviewing at Google next month for a staff engineer role.",
    ),
    # Multi-subject (deliberate, three)
    (
        "MS2_multi_subject_three_items",
        "Highlights from yesterday: closed the deal with Acme for 800K, fixed the "
        "auth bug that was breaking SSO, and finally booked the Hawaii trip for "
        "Sarah's 40th birthday in June.",
    ),
    # Multi-subject (conversational)
    (
        "MS3_message_with_multiple_facts",
        "Tomorrow's standup is moved to 11am because Priya has a dentist appointment. "
        "Don't forget to bring the demo laptop. And the QA team merged in the new "
        "test harness over the weekend so we should rebase before pushing.",
    ),
    # Focused statement
    (
        "FS1_focused_statement",
        "In chess, the move 25.g4 is played to gain space on the kingside and "
        "restrict the opponent's pawn structure.",
    ),
    # Bare term
    ("BT1_bare", "Tokyo"),
    # Filler
    ("FI1_filler", "yes"),
    # Filler with content
    (
        "FI2_filler_with_content",
        "ok, leaving Tuesday at 5 for the Boston trip",
    ),
]


def edit_overlap(a: str, b: str) -> float:
    """Rough overlap ratio: shared word tokens / max(len)."""
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


async def main():
    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(6)

    N_TRIALS = 3

    async def one(seg_label, segment, trial):
        async with sem:
            derivs = await derive(client, segment, model="gpt-5-nano", reasoning="low")
            return (seg_label, segment, trial, derivs)

    tasks = []
    for seg_label, segment in SEGMENTS:
        for trial in range(N_TRIALS):
            tasks.append(one(seg_label, segment, trial))

    results = await asyncio.gather(*tasks)

    # Group by segment label
    by_seg: dict = {}
    for seg_label, segment, trial, derivs in results:
        by_seg.setdefault(seg_label, []).append((segment, trial, derivs))

    for seg_label, trials in by_seg.items():
        first = trials[0]
        segment = first[0]
        seg_words = len(segment.split())
        print(f"\n{'=' * 78}")
        print(f"{seg_label}  (segment: {seg_words} words)")
        print(f"{'=' * 78}")
        print(f"SOURCE: {segment}")
        print()

        for segment, trial, derivs in trials:
            avg_overlap = (
                (sum(edit_overlap(d, segment) for d in derivs) / max(1, len(derivs)))
                if derivs
                else 0.0
            )
            total_words = sum(len(d.split()) for d in derivs)
            print(
                f"  trial {trial}: n={len(derivs)}, "
                f"total_words={total_words}, "
                f"word_overlap_w_source~{avg_overlap:.0%}"
            )
            for d in derivs:
                print(f"    - {d}")

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
