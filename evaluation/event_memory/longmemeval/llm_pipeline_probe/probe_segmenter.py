"""Probe an LLM-based semantic segmenter.

Strategy: never have the LLM rewrite or repeat the text — that risks
hallucination and content loss. Instead, label every line with a number
and ask the LLM to return cut points (line numbers where a NEW segment
should begin). We then split the text by those cuts, guaranteeing all
content is preserved verbatim.

This is what the architecture wants from a "good for an LLM reader"
segmenter: each segment should be a coherent, self-contained piece of
evidence. The LLM picks the boundaries; we apply them mechanically.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re

import openai
from dotenv import load_dotenv
from sample_chunks import collect

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


PROMPT_S1 = """\
You are a semantic chunker. You will be shown a passage with each line \
prefixed by a number in square brackets like [42]. Your job is to decide \
where to CUT the passage so each resulting segment is a coherent, \
self-contained unit a downstream reader can use as evidence on its own.

Cut where:
  - the topic shifts to something a reader would index separately
  - a complete thought, table, code block, or list ends
Do NOT cut:
  - inside a single sentence
  - between rows of the same table
  - inside the same code block, list, or step-by-step instruction
  - just to satisfy a length target — coherence trumps balance

Reply with a single line containing only the cut points: the line numbers \
where a NEW segment begins, comma-separated, no spaces. Always include 1 \
as the first cut. If the whole passage should be one segment, reply "1".

Example reply formats: "1" or "1,14,27,55"

PASSAGE:
{passage}"""


def _annotate(text: str) -> tuple[str, list[str]]:
    lines = text.splitlines() or [""]
    annotated = "\n".join(f"[{i + 1}] {line}" for i, line in enumerate(lines))
    return annotated, lines


def _parse_cuts(reply: str, n_lines: int) -> list[int]:
    nums = [int(m) for m in re.findall(r"\d+", reply)]
    nums = [n for n in nums if 1 <= n <= n_lines]
    if not nums:
        return [1]
    nums = sorted(set(nums))
    if nums[0] != 1:
        nums.insert(0, 1)
    return nums


def _apply_cuts(lines: list[str], cuts: list[int]) -> list[str]:
    bounds = [*cuts, len(lines) + 1]
    return [
        "\n".join(lines[bounds[i] - 1 : bounds[i + 1] - 1])
        for i in range(len(bounds) - 1)
    ]


async def segment(
    client: openai.AsyncOpenAI, model: str, text: str, reasoning: str | None
) -> list[str]:
    annotated, lines = _annotate(text)
    kwargs: dict = {"model": model, "input": PROMPT_S1.format(passage=annotated)}
    if reasoning and model.startswith("gpt-5"):
        kwargs["reasoning"] = {"effort": reasoning}
    resp = await client.responses.create(**kwargs)
    reply = (resp.output_text or "").strip()
    cuts = _parse_cuts(reply, len(lines))
    return _apply_cuts(lines, cuts), cuts, reply


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.4-nano")
    parser.add_argument("--reasoning", default="low")
    parser.add_argument("--per-bucket", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--min-length",
        type=int,
        default=600,
        help="Only test on chunks at least this long; shorter "
        "chunks rarely need splitting.",
    )
    args = parser.parse_args()

    bins = collect()
    samples: list[tuple[str, str]] = []
    for b, ts in bins.items():
        long_ts = [t for t in ts if len(t) >= args.min_length]
        for t in long_ts[: args.per_bucket]:
            samples.append((b, t))

    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(args.concurrency)

    async def go(text: str):
        async with sem:
            return await segment(client, args.model, text, args.reasoning)

    results = await asyncio.gather(*(go(t) for _, t in samples))
    await client.close()

    print(f"# model={args.model} reasoning={args.reasoning}")
    for (bucket, text), (segs, cuts, reply) in zip(samples, results, strict=False):
        print()
        print(
            f"=== [{bucket}] (len={len(text)}, lines={len(text.splitlines())}, "
            f"cuts={cuts}, llm-reply={reply!r}) ==="
        )
        # Verify lossless: joining segments with "\n" should equal text
        # (modulo trailing newline differences).
        rejoined = "\n".join(segs)
        ok = rejoined.rstrip("\n") == text.rstrip("\n")
        print(f"-- lossless: {ok}, n_segments={len(segs)}")
        for i, s in enumerate(segs):
            head = s.splitlines()[0] if s.splitlines() else "(empty)"
            head = head[:120] + ("…" if len(head) > 120 else "")
            print(f"  seg[{i}] len={len(s)}: {head}")


if __name__ == "__main__":
    asyncio.run(main())
