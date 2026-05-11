"""Probe an LLM-based deriver that rewrites segments into embedding targets.

The deriver's output is what we'd embed. The original segment text is kept
verbatim for the LLM reader at retrieval time. So our goal is to produce
text that:
  - matches semantic queries about the topic ("show me the benchmark table")
  - matches questions about specific values that are IN the original
    ("what was the Q3 revenue?")
  - drops noise (raw numbers, table separators, syntactic clutter)

Two prompt strategies tried:
  D1 - "describe what this is" (the user's example phrasing)
  D2 - D1 + "include the salient entities/topics by name; include numeric
       values inline in prose, not as tables"

We run both at small-model tier and eyeball outputs.
"""

from __future__ import annotations

import argparse
import asyncio
import os

import openai
from dotenv import load_dotenv
from sample_chunks import collect

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


PROMPT_D1 = """\
You produce embedding-target descriptions for a long-term retrieval index. \
The original text below will be kept verbatim for an LLM reader at \
retrieval time, but its EMBEDDING will come from your description.

Goal: write 1-2 sentences describing what this text IS — the kind of \
content (table, code, list, statement, story, plan, etc.), what it is \
about (topic, entities, what's being compared or shown), and what kinds \
of questions it could answer. Do NOT reproduce the text verbatim, \
especially raw numbers, table separators, code, or formatting. Use \
prose only.

Reply with the description and nothing else.

TEXT:
{text}"""


PROMPT_D2 = """\
You produce embedding-target descriptions for a long-term retrieval index. \
The original text below will be kept verbatim for an LLM reader at \
retrieval time, but its EMBEDDING will come from your description.

Goal: write 1-2 sentences describing what this text IS — what kind of \
content (table, code, list, statement, story, plan, etc.) and what it \
is about. Name the salient entities, brands, places, people, and \
specific topics so a query that mentions them will match. If the text \
contains numeric values, mention them in prose only (e.g., "shows \
quarterly revenue of about $4M, $5M, and $4M"); do NOT reproduce raw \
tables, code, or formatting.

Reply with the description and nothing else.

TEXT:
{text}"""


PROMPTS = {"D1": PROMPT_D1, "D2": PROMPT_D2}


async def describe(
    client: openai.AsyncOpenAI,
    model: str,
    prompt_template: str,
    text: str,
    reasoning: str | None,
) -> str:
    kwargs: dict = {"model": model, "input": prompt_template.format(text=text)}
    if reasoning and model.startswith("gpt-5"):
        kwargs["reasoning"] = {"effort": reasoning}
    resp = await client.responses.create(**kwargs)
    return (resp.output_text or "").strip()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.4-nano")
    parser.add_argument("--reasoning", default="low")
    parser.add_argument("--prompt", default="D2", choices=["D1", "D2"])
    parser.add_argument("--per-bucket", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=12)
    args = parser.parse_args()

    bins = collect()
    samples = [(b, t) for b, ts in bins.items() for t in ts[: args.per_bucket]]
    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(args.concurrency)
    template = PROMPTS[args.prompt]

    async def go(text: str) -> str:
        async with sem:
            return await describe(client, args.model, template, text, args.reasoning)

    descs = await asyncio.gather(*(go(t) for _, t in samples))
    await client.close()

    print(f"# model={args.model} reasoning={args.reasoning} prompt={args.prompt}")
    for (bucket, text), desc in zip(samples, descs, strict=False):
        print()
        print(f"=== [{bucket}] (len={len(text)}) ===")
        snippet = text if len(text) <= 400 else text[:400] + " …"
        print("-- ORIGINAL:")
        print(snippet)
        print(f"-- DESCRIPTION ({len(desc)} chars):")
        print(desc)


if __name__ == "__main__":
    asyncio.run(main())
