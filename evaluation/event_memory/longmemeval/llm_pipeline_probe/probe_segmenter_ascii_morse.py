"""Quick check: does v30 keep ASCII art and morse code?

Both are highly specific to a passage (distinctive phrasing / unique
content), so rule 2 KEEP side should pick them up. But the LLM might
mis-classify long whitespace-heavy content as "boilerplate" or trim it
at unusual boundaries. Test empirically.
"""

from __future__ import annotations

import asyncio
import os

import openai
from dotenv import load_dotenv
from probe_segmenter_F_natural import call
from probe_segmenter_F_natural_v30 import PROMPT_F_NATURAL_V30

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


CASES = [
    (
        "ascii_art_cat_inline",
        (
            "I drew this cat for my daughter today:\n"
            "  /\\_/\\\n"
            " ( o.o )\n"
            "  > ^ <\n"
            "She loved it."
        ),
    ),
    (
        "ascii_art_only",
        ("  /\\_/\\\n ( o.o )\n  > ^ <"),
    ),
    (
        "morse_code_sos",
        "I sent the SOS signal: ... --- ... and the rescuer arrived.",
    ),
    (
        "morse_code_message",
        (
            "The full message was: .... . .-.. .-.. --- / .-- --- .-. .-.. -..\n"
            "(That's HELLO WORLD in morse.)"
        ),
    ),
    (
        "ascii_table",
        (
            "Q3 results:\n"
            "+--------+-------+\n"
            "| metric | value |\n"
            "+--------+-------+\n"
            "| revenue| $1.2M |\n"
            "| growth | 14%   |\n"
            "+--------+-------+\n"
            "Decent quarter."
        ),
    ),
]


async def main() -> None:
    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = "gpt-5-mini"
    reasoning = "low"
    print(f"# model={model} reasoning={reasoning} prompt=v30\n")

    for case_id, source in CASES:
        prompt = PROMPT_F_NATURAL_V30.format(passage=source)
        segs = await call(client, model, prompt, reasoning)
        joined = "\n--SEG--\n".join(segs)
        print(f"=== {case_id} ===")
        print("SOURCE:")
        print(source)
        print("\nSEGMENTS:")
        print(joined)
        # Reconstruction via "".join (no source-stitching applied here,
        # just LLM-raw to see what the model emits before stitching).
        print(f"\nN_SEGS: {len(segs)}")
        print()

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
