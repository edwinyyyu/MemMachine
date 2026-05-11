"""LLM segmenter, unstructured modes U1 and U2.

U1 - LLM is asked to rewrite the kept content as plain prose. It is NOT
     told anything about splitting. A deterministic splitter
     (RecursiveCharacterTextSplitter) cuts the result.

U2 - LLM is told the splitting strategy directly: "write the kept
     content as paragraphs separated by a blank line, each paragraph at
     most ~N chars". Splitter is trivial: split on `\\n\\s*\\n`.

Both modes share the same fidelity rule as Mode F: verbatim quoting,
no synonym swaps, drop scaffolding by exclusion.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re

import openai
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sample_chunks import collect

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


PROMPT_U1 = """\
Compress this passage to the parts a human would still remember weeks \
later, written as plain text in the original speaker's voice.

Rules:
  1. VERBATIM. Each kept span must be a contiguous verbatim quote from \
the passage. Do NOT change wording: "fabulous" stays "fabulous", \
"gobsmacked" stays "gobsmacked". Connotation matters — never swap a \
word for a synonym, never paraphrase, never abstract.
  2. DROP generic scaffolding by simply not including it: "Hi!", "What \
a great question!", "I hope this helps!", "Let me know if you have any \
other questions", a polite restatement of what the other party just \
asked.
  3. KEEP every named entity, place, person, brand, work, date, price, \
preference, plan, decision, and specific factual claim, plus eccentric \
or distinctive phrasing.
  4. PRESERVE original order — kept spans must appear in the same order \
as their source quotes.

Output: the kept content as plain text. No preface, no header, no \
explanation of what you did.

PASSAGE:
{passage}"""


PROMPT_U2 = """\
Compress this passage to the parts a human would still remember weeks \
later, formatted as a list of standalone memory segments separated by a \
single blank line (i.e. two consecutive newlines: \\n\\n).

Rules:
  1. VERBATIM. Each segment is a contiguous verbatim quote from the \
passage. Do NOT change wording: "fabulous" stays "fabulous", \
"gobsmacked" stays "gobsmacked". Never swap a word for a synonym, \
never paraphrase, never abstract.
  2. DROP generic scaffolding by simply not including it in any segment: \
"Hi!", "What a great question!", "I hope this helps!", "Let me know if \
you have any other questions", a polite restatement of what the other \
party just asked.
  3. KEEP every named entity, place, person, brand, work, date, price, \
preference, plan, decision, and specific factual claim, plus eccentric \
or distinctive phrasing.
  4. PRESERVE original order — segments must appear in the same order \
as their source quotes.
  5. STANDALONE — each segment should read on its own. If a quote \
depends on a referent introduced earlier, widen the quote to start \
where the referent is named.
  6. Each segment ≤ {max_chars} characters. Use a blank line — two \
consecutive newlines — to separate one segment from the next, and use \
no other formatting.

Output: the segments separated by blank lines, and nothing else. No \
preface, no JSON, no markdown headers, no numbered list markers.

PASSAGE:
{passage}"""


async def call(client, model, prompt, reasoning):
    kwargs: dict = {"model": model, "input": prompt}
    if reasoning and model.startswith("gpt-5"):
        kwargs["reasoning"] = {"effort": reasoning}
    resp = await client.responses.create(**kwargs)
    return (resp.output_text or "").strip()


def split_u1(text: str, max_chars: int) -> list[str]:
    """Deterministic splitter for U1 output.

    Same as the one used in compare_R_vs_RS / R-mode: prefer paragraph
    breaks, then sentence punctuation, then commas, then spaces.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=0,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
        keep_separator="end",
    )
    return splitter.split_text(text)


def split_u2(text: str) -> list[str]:
    """Trivial paragraph splitter for U2 output: cut on blank lines."""
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5-nano")
    parser.add_argument("--reasoning", default="low")
    parser.add_argument("--mode", default="U1", choices=["U1", "U2"])
    parser.add_argument("--max-chars", type=int, default=500)
    parser.add_argument("--per-bucket", type=int, default=2)
    parser.add_argument("--min-length", type=int, default=600)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    bins = collect()
    samples = []
    for b, ts in bins.items():
        for t in [x for x in ts if len(x) >= args.min_length][: args.per_bucket]:
            samples.append((b, t))

    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(args.concurrency)

    async def go(text):
        prompt = (PROMPT_U1 if args.mode == "U1" else PROMPT_U2).format(
            passage=text, max_chars=args.max_chars
        )
        async with sem:
            return await call(client, args.model, prompt, args.reasoning)

    rep = await asyncio.gather(*(go(t) for _, t in samples))
    await client.close()

    print(
        f"# model={args.model} reasoning={args.reasoning} mode={args.mode} "
        f"max_chars={args.max_chars}"
    )
    for (bucket, src), reply in zip(samples, rep, strict=False):
        segs = split_u1(reply, args.max_chars) if args.mode == "U1" else split_u2(reply)
        max_len = max((len(s) for s in segs), default=0)
        print()
        print(
            f"=== [{bucket}] src_len={len(src)} -> {len(segs)} segs, max_len={max_len} ==="
        )
        for i, s in enumerate(segs[:6]):
            print(
                f"  seg[{i}] ({len(s)} chars): {s[:200]}{'…' if len(s) > 200 else ''}"
            )
        if len(segs) > 6:
            print(f"  …and {len(segs) - 6} more segments")


if __name__ == "__main__":
    asyncio.run(main())
