"""Stress-test the LLM segmenter against the failures that actually matter:
mid-table cuts, mid-code-block cuts, and broken markdown structure.

We auto-flag any cut that lands inside a fenced code block or a markdown
table block. The model must produce ZERO such cuts to be ship-grade.
"""

from __future__ import annotations

import argparse
import asyncio
import os

import openai
from dotenv import load_dotenv
from probe_segmenter import PROMPT_S1, _annotate, _parse_cuts
from sample_chunks import collect

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


def _line_in_code_block(lines: list[str]) -> list[bool]:
    """Mark lines that are INSIDE a fenced code block (``` start/end)."""
    inside = [False] * len(lines)
    in_block = False
    for i, ln in enumerate(lines):
        if ln.strip().startswith("```"):
            in_block = not in_block
            inside[i] = True  # the fence itself counts as "inside"
            continue
        inside[i] = in_block
    return inside


def _line_in_table(lines: list[str]) -> list[bool]:
    """Mark lines that are part of a contiguous markdown-table block.

    A table block is 2+ adjacent lines that contain at least 2 pipes.
    """
    has_pipe = [ln.count("|") >= 2 for ln in lines]
    in_table = [False] * len(lines)
    i = 0
    while i < len(lines):
        if has_pipe[i]:
            j = i
            while j < len(lines) and has_pipe[j]:
                j += 1
            if j - i >= 2:
                for k in range(i, j):
                    in_table[k] = True
            i = j
        else:
            i += 1
    return in_table


def _bad_cuts(text: str, cuts: list[int]) -> list[tuple[int, str]]:
    """Cuts that fall strictly INSIDE a code block or table.

    A cut at line k means "segment 2 starts at line k". So lines
    k-1 (end of prev segment) and k (start of next) sit on the boundary.
    A cut is bad when both lines are inside the same structural block.
    """
    lines = text.splitlines()
    in_code = _line_in_code_block(lines)
    in_tbl = _line_in_table(lines)
    bad = []
    for c in cuts:
        if c == 1 or c > len(lines):
            continue
        prev = c - 2  # line index just before the cut
        nxt = c - 1  # line index at the cut (start of next segment)
        if prev >= 0 and nxt < len(lines):
            if in_code[prev] and in_code[nxt]:
                bad.append((c, "code-block"))
            elif in_tbl[prev] and in_tbl[nxt]:
                bad.append((c, "table"))
    return bad


async def segment_one(
    client: openai.AsyncOpenAI, model: str, text: str, reasoning: str | None
):
    annotated, lines = _annotate(text)
    kwargs: dict = {"model": model, "input": PROMPT_S1.format(passage=annotated)}
    if reasoning and model.startswith("gpt-5"):
        kwargs["reasoning"] = {"effort": reasoning}
    resp = await client.responses.create(**kwargs)
    reply = (resp.output_text or "").strip()
    cuts = _parse_cuts(reply, len(lines))
    return cuts, _bad_cuts(text, cuts)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.4-nano")
    parser.add_argument("--reasoning", default="low")
    parser.add_argument("--n-table", type=int, default=15)
    parser.add_argument("--n-code", type=int, default=15)
    parser.add_argument("--concurrency", type=int, default=12)
    args = parser.parse_args()

    bins = collect()
    cases = [("table", t) for t in bins["T table"][: args.n_table]] + [
        ("code", t) for t in bins["C code"][: args.n_code]
    ]

    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(args.concurrency)

    async def go(text: str):
        async with sem:
            return await segment_one(client, args.model, text, args.reasoning)

    results = await asyncio.gather(*(go(t) for _, t in cases))
    await client.close()

    bad_total = 0
    bad_examples: list[tuple[str, int, str, str]] = []
    for (kind, text), (cuts, bad) in zip(cases, results, strict=False):
        if bad:
            bad_total += len(bad)
            for cut, reason in bad:
                preview = "\n".join(text.splitlines()[max(0, cut - 3) : cut + 1])
                bad_examples.append((kind, cut, reason, preview))

    print(f"# model={args.model} reasoning={args.reasoning}")
    print(f"  total cases: {len(cases)} (table={args.n_table}, code={args.n_code})")
    print(f"  bad-cut occurrences: {bad_total}")
    if bad_examples:
        print()
        print("BAD CUTS (cut line, reason, surrounding context):")
        for kind, cut, reason, preview in bad_examples[:20]:
            print(f"\n  --- [{kind}] cut at line {cut} ({reason}) ---")
            for ln in preview.splitlines():
                print(f"    {ln[:140]}")


if __name__ == "__main__":
    asyncio.run(main())
