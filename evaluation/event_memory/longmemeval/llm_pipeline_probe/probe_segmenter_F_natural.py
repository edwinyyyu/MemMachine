"""LLM segmenter, Mode F-natural — let the LLM segment by coherence alone.

Differences from Mode F:
  - No `maxLength` constraint in the JSON schema.
  - Prompt does not mention any length budget.
  - For inputs longer than a `WINDOW_CHARS` threshold (default 8000), a
    deterministic RecursiveCharacterTextSplitter cuts the input into
    large windows; we then run the LLM per window and concatenate the
    resulting segments. This is the "whole-book" safety hatch — it only
    triggers on extreme inputs.

The goal: the LLM picks segment boundaries based on what coheres
together, not on a per-segment char count. We trust the prompt to
keep segments "small enough" while the absence of a hard cap removes
the LLM's incentive to truncate widening quotes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

import openai
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from probe_segmenter_F import verbatim_ratio
from sample_chunks import collect

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


# Default window size for the pre-pass. Anything beyond this gets
# deterministically chunked; well within the model's context but
# big enough that natural segments inside the window are cohesive.
WINDOW_CHARS = 8000


SCHEMA_NATURAL = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "segments": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["segments"],
}


PROMPT_F_NATURAL = """\
Compress this passage into the parts a human would still remember weeks \
later, broken into a list of standalone memory segments.

Rules:
  1. VERBATIM. Each segment must be a contiguous verbatim quote from \
the passage. Do NOT change wording: "fabulous" stays "fabulous", \
"gobsmacked" stays "gobsmacked". Connotation matters — never swap a \
word for a synonym, never paraphrase, never abstract. The only edits \
allowed are starting and ending the quote at sentence/clause boundaries.
  2. DROP generic scaffolding by simply not including it in any segment: \
"Hi!", "What a great question!", "I hope this helps!", "Let me know if \
you have any other questions", a polite restatement of what the other \
party just asked.
  3. KEEP every named entity, place, person, brand, work, date, price, \
preference, plan, decision, and specific factual claim, plus eccentric \
or distinctive phrasing.
  4. PRESERVE original order — segments must appear in the same order \
as their source quotes.
  5. SEGMENT NATURALLY — break where topics or sub-topics shift. \
Coherence trumps balance: a long passage that stays on one topic is \
one segment; a passage that covers several topics gets one segment per \
topic. Do not artificially split a coherent unit, and do not artificially \
merge unrelated ones.
  6. STANDALONE — each segment should read on its own. If a quote \
depends on a referent introduced earlier (e.g., uses "the trip" for \
"the Tokyo trip in May"), widen the quote to start where the referent \
is named.

Output: a JSON object {{ "segments": [...] }} and nothing else.

PASSAGE:
{passage}"""


def _splitter_for_windows(window_chars: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=window_chars,
        chunk_overlap=0,
        separators=[
            "\n\n\n",
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ": ",
            ", ",
            " ",
            "",
        ],
        keep_separator="end",
    )


async def call(client, model, prompt, reasoning):
    kwargs: dict = {
        "model": model,
        "input": prompt,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "segments_response",
                "schema": SCHEMA_NATURAL,
                "strict": True,
            }
        },
    }
    if reasoning and model.startswith("gpt-5"):
        kwargs["reasoning"] = {"effort": reasoning}
    resp = await client.responses.create(**kwargs)
    raw = (resp.output_text or "").strip()
    parsed = json.loads(raw)
    return parsed.get("segments", [])


async def segment(
    client: openai.AsyncOpenAI,
    model: str,
    text: str,
    reasoning: str | None = "low",
    window_chars: int = WINDOW_CHARS,
) -> list[str]:
    """Run F-natural; transparently pre-window if input is huge."""
    if len(text) <= window_chars:
        prompt = PROMPT_F_NATURAL.format(passage=text)
        return await call(client, model, prompt, reasoning)

    splitter = _splitter_for_windows(window_chars)
    windows = splitter.split_text(text)
    sub_results = await asyncio.gather(
        *(
            call(client, model, PROMPT_F_NATURAL.format(passage=w), reasoning)
            for w in windows
        )
    )
    flat: list[str] = []
    for r in sub_results:
        flat.extend(r)
    return flat


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5-nano")
    parser.add_argument("--reasoning", default="low")
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
        async with sem:
            return await segment(client, args.model, text, args.reasoning)

    rep = await asyncio.gather(*(go(t) for _, t in samples))
    await client.close()

    print(f"# model={args.model} reasoning={args.reasoning} mode=F-natural")
    for (bucket, src), segs in zip(samples, rep, strict=False):
        avg_verb = sum(verbatim_ratio(s, src) for s in segs) / max(len(segs), 1)
        max_len = max((len(s) for s in segs), default=0)
        print()
        print(
            f"=== [{bucket}] src={len(src)} -> {len(segs)} segs, "
            f"max_len={max_len}, verbatim={avg_verb:.0%} ==="
        )
        for i, s in enumerate(segs):
            v = verbatim_ratio(s, src)
            print(
                f"  seg[{i}] (len={len(s)}, verbatim={v:.0%}): {s[:200]}{'…' if len(s) > 200 else ''}"
            )


if __name__ == "__main__":
    asyncio.run(main())
