"""Dump v33 ascii_art residual failures on gpt-5-nano @ low to diagnose
whether the model is dropping art, paraphrasing it, or producing mangled
whitespace.
"""

from __future__ import annotations

import asyncio
import os

import openai
from dotenv import load_dotenv
from probe_segmenter_F_natural_v33 import segment as segment_v33

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")

SOURCE = (
    "I drew this for my daughter today:\n  /\\_/\\\n ( o.o )\n  > ^ <\nShe loved it."
)
SIGNATURE = "/\\_/\\"


async def main() -> None:
    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(8)

    async def go():
        async with sem:
            return await segment_v33(client, "gpt-5-nano", SOURCE, "low")

    runs = await asyncio.gather(*(go() for _ in range(30)))
    fails = [
        (i, segs) for i, segs in enumerate(runs) if SIGNATURE not in " ".join(segs)
    ]
    passes = [(i, segs) for i, segs in enumerate(runs) if SIGNATURE in " ".join(segs)]
    print(f"passes: {len(passes)}/30")
    print(f"failures: {len(fails)}/30")
    print()
    print("=== FAILURES ===")
    for i, segs in fails[:10]:
        print(f"\n--- rep {i + 1} ({len(segs)} segs) ---")
        for j, s in enumerate(segs):
            print(f"  [{j}] {s!r}")
    print()
    print("=== SAMPLE PASS ===")
    if passes:
        i, segs = passes[0]
        print(f"--- rep {i + 1} ({len(segs)} segs) ---")
        for j, s in enumerate(segs):
            print(f"  [{j}] {s!r}")
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
