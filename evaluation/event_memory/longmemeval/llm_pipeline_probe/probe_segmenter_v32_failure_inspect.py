"""Inspect what's actually happening on the v32 regressions for gpt-5-nano
on table and code cases. Run a few reps and dump every segment so we can
tell content-loss from segmentation-difference.
"""

from __future__ import annotations

import asyncio
import os

import openai
from dotenv import load_dotenv
from probe_segmenter_F_natural_v32 import segment as segment_v32

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


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


async def main() -> None:
    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    for case_id, source, signature in [
        ("ascii_table", ASCII_TABLE_SOURCE, "GPT-4 | 0.85"),
        ("python_code", PYTHON_CODE_SOURCE, "def find_max(node):"),
    ]:
        print(f"\n{'=' * 76}\nCASE {case_id}\n{'=' * 76}")
        print(f"signature: {signature!r}")
        print()
        for rep in range(5):
            segs = await segment_v32(client, "gpt-5-nano", source, "low")
            joined = " ".join(segs)
            verdict = "PASS" if signature in joined else "FAIL"
            print(f"--- rep {rep + 1} [{verdict}] ({len(segs)} segs) ---")
            for i, s in enumerate(segs):
                print(f"  [{i}] {s!r}")
            print()
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
