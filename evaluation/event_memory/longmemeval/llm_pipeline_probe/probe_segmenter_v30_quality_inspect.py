"""Side-by-side qualitative segment output across all 6 model x reasoning
cells. The feedback bench measures keep/drop substring hits; this probe
shows the actual segments so quality can be inspected directly.
"""

from __future__ import annotations

import asyncio
import os

import openai
from dotenv import load_dotenv
from probe_segmenter_F_natural_v30 import segment as segment_v30

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


CASES = [
    (
        "1_email_with_envelope",
        (
            "Hi team, As discussed, please find attached the proposal. "
            "Project budget is $45k. Deadline is March 30. Let me know if "
            "any questions. Thanks, Sarah"
        ),
    ),
    (
        "2_multi_topic_narrative",
        (
            "Yesterday I drove out to the new Tesla store in Burlingame to "
            "take a Model Y for a test drive. Marcus, the salesperson, "
            "had just transferred from Fremont. We took Highway 280 -- "
            "autopilot held the lane really well at 65mph. Acceleration "
            "is absurd: 0-60 in 4.8s for the Long Range. After that I "
            "met my sister Anne for lunch at Tacolicious in Palo Alto. "
            "She just got back from a 3-week trip to Tokyo with her "
            "husband Daniel."
        ),
    ),
    (
        "3_short_reply_with_content",
        "ok, leaving Tuesday at 5 for the Boston trip",
    ),
    (
        "4_pure_filler",
        "yes",
    ),
    (
        "5_ascii_art_inline",
        (
            "I drew this for my daughter today:\n"
            "  /\\_/\\\n"
            " ( o.o )\n"
            "  > ^ <\n"
            "She loved it."
        ),
    ),
    (
        "6_single_focused_fact",
        "Honestly, I think Sleep is overrated.",
    ),
    (
        "7_email_with_specifics",
        (
            "Hey Bob, just confirming: meeting is moved to Thursday 3pm "
            "in Conference Room 4. Per Alice, we're going to talk about "
            "the Q3 budget overrun ($12k on AWS) and the hiring plan "
            "for the new platform team. Cheers, Marcus"
        ),
    ),
]


CELLS = [
    ("gpt-5-nano", "low"),
    ("gpt-5-nano", "medium"),
    ("gpt-5.4-nano", "low"),
    ("gpt-5.4-nano", "medium"),
    ("gpt-5-mini", "low"),
    ("gpt-5-mini", "medium"),
]


async def main() -> None:
    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    for case_id, source in CASES:
        print(f"\n{'=' * 76}")
        print(f"CASE {case_id}")
        print(f"{'=' * 76}")
        print("SOURCE:")
        print(source)
        print()
        # Run all 6 cells in parallel for this case.
        tasks = [
            segment_v30(client, model, source, reasoning) for model, reasoning in CELLS
        ]
        results = await asyncio.gather(*tasks)
        for (model, reasoning), segs in zip(CELLS, results, strict=True):
            print(f"--- {model} @ {reasoning} ({len(segs)} segs) ---")
            for i, s in enumerate(segs):
                # Show repr to expose any leading/trailing whitespace.
                print(f"  [{i}] {s!r}")
            print()
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
