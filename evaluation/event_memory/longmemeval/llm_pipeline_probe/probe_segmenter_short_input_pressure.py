"""Test the short-input-pressure hypothesis on the segmenter.

Hypothesis: when given a passage of only ~3 sentences, the model feels
pressure to produce a non-trivial list and over-segments. Embedded in a
longer passage, the same sentences would merge into one segment because
context disambiguates that they form one unit.

Method: take the four F4 over-fragmentation cases that fail under v33
when isolated. Prepend / append varied surrounding context (vacation
narrative + future plans + ambient news) and pass the whole thing
through the v33 segmenter. Measure how the F4 sentences are segmented:
do they fuse into one segment of the larger output, or do they still
get split as if they were isolated?
"""

from __future__ import annotations

import asyncio
import os

import openai
from dotenv import load_dotenv
from probe_segmenter_F_natural_v33 import segment as segment_v33

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


# The four F4 passages that v33 over-fragments when isolated.
F4_ISOLATED: list[tuple[str, str]] = [
    (
        "reaction",
        "Wow! That sounds amazing! The connection to nature must be incredible.",
    ),
    (
        "acknowledgment",
        "No problem, Pat! Let me know whenever you need assistance. Take care!",
    ),
    (
        "praise",
        "Wow, Riley - you and Casey are so in tune! It's clear you both rock on stage.",
    ),
    (
        "thanks",
        "Thanks, Jordan! It's a mix of drama and romance!",
    ),
]


# Surrounding context (~200 chars before and ~200 chars after) covering
# a different topic from the F4 passages. Picked to neither overlap with
# the F4 content nor obviously cue topic shift just before/after.
PREFIX = (
    "I spent last weekend at Crater Lake -- the water was bluer than "
    "I expected and the hike to the rim took about two hours. "
    "I'm planning to head back in late August with the kids. "
)
SUFFIX = (
    " On a separate note, I'm starting a new role at Vercel on June 9. "
    "It's a staff engineer position on the runtime team. "
    "I'll be working with Frieda and Markus, who I met at the offsite."
)


def _count_segments_overlapping(segments: list[str], target: str) -> int:
    """Count how many returned segments contain any portion of `target`.

    Approximate: split target into sentences and check sentence-by-sentence
    membership. A clean fusion gives 1; a clean split gives N.
    """
    sentences = [s.strip() for s in target.split(". ") if s.strip()]
    # for each sentence, find which segment (if any) contains it
    hits: set[int] = set()
    for sent in sentences:
        # the sentence may have a trailing punctuation eaten by split
        # search with reasonable tolerance
        needle = sent.rstrip(".!?")
        for i, seg in enumerate(segments):
            if needle in seg:
                hits.add(i)
                break
    return len(hits)


async def main() -> None:
    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    results: list[dict] = []
    for label, f4_passage in F4_ISOLATED:
        embedded = PREFIX + f4_passage + SUFFIX

        iso_segs = await segment_v33(client, "gpt-5.4-nano", f4_passage, "low")
        emb_segs = await segment_v33(client, "gpt-5.4-nano", embedded, "low")

        emb_f4_segs = _count_segments_overlapping(emb_segs, f4_passage)

        results.append(
            {
                "label": label,
                "iso_n": len(iso_segs),
                "iso_segs": iso_segs,
                "emb_n": len(emb_segs),
                "emb_f4_segs": emb_f4_segs,
                "emb_segs": emb_segs,
            }
        )

    print("# Short-input-pressure test, gpt-5.4-nano @ low, prompt v33")
    print()
    print(f"# PREFIX ({len(PREFIX)} chars): {PREFIX[:60]}...")
    print(f"# SUFFIX ({len(SUFFIX)} chars): {SUFFIX[:60]}...")
    print()
    print(
        "| label | F4 isolated n_segs | F4 embedded — segs in F4 region | total embedded n_segs |"
    )
    print("|---|---|---|---|")
    for r in results:
        print(f"| {r['label']} | {r['iso_n']} | {r['emb_f4_segs']} | {r['emb_n']} |")
    print()
    for r in results:
        print(f"\n## {r['label']}")
        print(f"\nISOLATED ({r['iso_n']} segs):")
        for i, s in enumerate(r["iso_segs"]):
            print(f"  [{i}] {s!r}")
        print(f"\nEMBEDDED ({r['emb_n']} segs total, {r['emb_f4_segs']} in F4 region):")
        for i, s in enumerate(r["emb_segs"]):
            print(f"  [{i}] {s!r}")

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
