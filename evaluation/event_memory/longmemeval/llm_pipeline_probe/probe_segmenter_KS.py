"""LLM segmenter, Mode KS — keep-spans by unit index.

The robustness recipe:
  1. Pre-tokenize the passage into addressable units (sentences within
     paragraphs, plus paragraph breaks for code/tables).
  2. Show the LLM the units numbered, [1] [2] [3] ...
  3. The LLM emits ONLY integer indices, grouped into segments, via the
     OpenAI Responses API with strict json_schema enforcement.
  4. Code looks up the units verbatim and joins them into segment text.

Properties this gets us "for free":
  - Zero hallucination — the LLM never generates new text. Output is a
    verbatim subset/rearrangement of source units.
  - Zero paraphrase — "fabulous" stays "fabulous", "gobsmacked" stays
    "gobsmacked". No connotation drift possible.
  - Structure guaranteed — strict json_schema means the response WILL be
    a list of integer arrays. No JSON parse failures, no marker emission
    failures.
  - Generic scaffolding still dropped — the LLM can omit a unit like
    "What a great question!" while keeping the content unit.
  - Standalone segments — the LLM may include a referent unit in a later
    segment to anchor pronouns ("the trip" → also include the earlier
    "Tokyo trip planned for May" unit).

The only thing the LLM has to do well is pick indices. That's a much
narrower task than "rewrite text following 6 rules" and tends to be
robust across model sizes / reasoning levels.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re

import openai
from dotenv import load_dotenv
from sample_chunks import collect

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


# Schema enforced by the Responses API in strict mode. The LLM is forced
# to emit {"segments": [{"include": [int, ...]}, ...]}.
SEGMENT_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "include": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                },
                "required": ["include"],
            },
        },
    },
    "required": ["segments"],
}


PROMPT_KS = """\
You are grouping a passage into standalone memory segments. The passage \
has been split into UNITS (sentences, code blocks, table blocks, list \
items). Each unit is prefixed with its number AND its character length \
in the form [N|Lch] — e.g. [42|187ch] means unit 42 is 187 characters \
long. Use the lengths to budget each segment.

Your job:
  1. DROP units that are pure conversational scaffolding with no content \
of their own — bare acknowledgements, generic greetings, "What a great \
question!", "I hope this helps!", "Let me know if you have any other \
questions", a polite restatement of what the other party just asked.
  2. KEEP every other unit unchanged. Eccentric or distinctive openings, \
named entities, places, brands, works, dates, prices, preferences, plans, \
decisions, factual claims, distinctive phrasing — all stay.
  3. GROUP the kept units into segments that read coherently together. \
The SUM of unit lengths in any one segment must not exceed {max_chars} \
characters. If a single unit is longer than {max_chars}, it stands as \
its own segment — that is allowed.
  4. STANDALONE: if a unit in a segment depends on something introduced \
earlier (e.g., uses "the trip" to refer to "the Tokyo trip in May"), \
include the earlier unit that names the referent in the same segment. \
You may include the same unit in more than one segment if needed.
  5. NEVER reorder units within a segment — preserve original order.

You do NOT rewrite or paraphrase any text. You only choose which unit \
numbers belong to each segment.

Output: a JSON object with key "segments". Each element has key \
"include" — a list of unit numbers (integers) for that segment.

PASSAGE:
{passage}"""


def split_units(text: str, max_unit_chars: int = 400) -> list[str]:
    """Paragraph-then-sentence tokenization.

    A unit is either a sentence (within a normal paragraph) or a whole
    paragraph block (kept intact for code/tables/list items). Any unit
    longer than `max_unit_chars` gets recursively cut on natural
    boundaries so the LLM has finer-grained options to budget with;
    this prevents one giant code/table block from forcing a segment
    way over its size cap.
    """
    units: list[str] = []
    paragraphs = re.split(r"\n\s*\n", text)
    # Sentence-split: punctuation + whitespace + capital-starting next sentence,
    # BUT not when the period is preceded by a digit (so we don't slice
    # "**1. Educate yourself:**" or "It cost $5. Then..." mid-marker).
    sent_split = re.compile(r"(?<=\D[.!?])\s+(?=[A-Z\"“‘\(\[*])")
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # If the paragraph contains code / table / list markers / multi-line
        # structure, keep it as one unit unless it's bigger than the cap.
        is_structured = (
            "```" in para
            or para.count("|") >= 4
            or re.match(r"^\s*[\-\*\+]\s", para)
            # Numbered list item, optionally wrapped in markdown bold/italic:
            # matches "1.", "1)", "**1.", "*1)*", etc.
            or re.match(r"^\s*[*_]{0,2}\d+[\.\)][*_]{0,2}\s", para)
            or "\n" in para
        )
        if is_structured:
            if len(para) <= max_unit_chars:
                units.append(para)
            else:
                # Subdivide on newlines (table rows / code lines / list items).
                lines = para.splitlines()
                buf: list[str] = []
                buf_len = 0
                for ln in lines:
                    if buf and buf_len + len(ln) + 1 > max_unit_chars:
                        units.append("\n".join(buf))
                        buf = [ln]
                        buf_len = len(ln)
                    else:
                        buf.append(ln)
                        buf_len += len(ln) + 1
                if buf:
                    units.append("\n".join(buf))
            continue
        for s in sent_split.split(para):
            s = s.strip()
            if not s:
                continue
            if len(s) <= max_unit_chars:
                units.append(s)
            else:
                # Long unstructured sentence (rare): chunk by clauses.
                pieces = re.split(r"(?<=[,;:])\s+", s)
                buf, buf_len = [], 0
                for p in pieces:
                    if buf and buf_len + len(p) + 1 > max_unit_chars:
                        units.append(" ".join(buf))
                        buf = [p]
                        buf_len = len(p)
                    else:
                        buf.append(p)
                        buf_len += len(p) + 1
                if buf:
                    units.append(" ".join(buf))
    return units


def annotate(units: list[str]) -> str:
    return "\n".join(f"[{i + 1}|{len(u)}ch] {u}" for i, u in enumerate(units))


def assemble(units: list[str], segments_spec: list[dict]) -> list[str]:
    out: list[str] = []
    for seg in segments_spec:
        include = seg.get("include", []) or []
        chosen: list[str] = []
        for idx in include:
            if 1 <= idx <= len(units):
                chosen.append(units[idx - 1])
        if chosen:
            out.append(
                "\n\n".join(chosen)
                if any("\n" in c for c in chosen)
                else " ".join(chosen)
            )
    return out


async def call(
    client: openai.AsyncOpenAI, model: str, prompt: str, reasoning: str | None
) -> dict | None:
    kwargs: dict = {
        "model": model,
        "input": prompt,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "segments_response",
                "schema": SEGMENT_SCHEMA,
                "strict": True,
            }
        },
    }
    if reasoning and model.startswith("gpt-5"):
        kwargs["reasoning"] = {"effort": reasoning}
    resp = await client.responses.create(**kwargs)
    raw = (resp.output_text or "").strip()
    try:
        return json.loads(raw)
    except Exception:
        return None


async def segment_one(
    client: openai.AsyncOpenAI,
    model: str,
    text: str,
    reasoning: str | None,
    max_chars: int,
):
    units = split_units(text)
    annotated = annotate(units)
    prompt = PROMPT_KS.format(passage=annotated, max_chars=max_chars)
    parsed = await call(client, model, prompt, reasoning)
    if parsed is None:
        return units, [], None
    segments = assemble(units, parsed.get("segments", []))
    return units, segments, parsed


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5-nano")
    parser.add_argument("--reasoning", default="low")
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

    async def go(text: str):
        async with sem:
            return await segment_one(
                client, args.model, text, args.reasoning, args.max_chars
            )

    rep = await asyncio.gather(*(go(t) for _, t in samples))
    await client.close()

    print(f"# model={args.model} reasoning={args.reasoning} max_chars={args.max_chars}")
    for (bucket, src), (units, segs, parsed) in zip(samples, rep, strict=False):
        print()
        print(
            f"=== [{bucket}] src_len={len(src)}, units={len(units)}, "
            f"segs={len(segs)}, schema_ok={parsed is not None} ==="
        )
        print("-- ORIG (first 250):")
        print(src[:250])
        for i, s in enumerate(segs):
            print(
                f"-- seg[{i}] ({len(s)} chars): {s[:200]}{'…' if len(s) > 200 else ''}"
            )


if __name__ == "__main__":
    asyncio.run(main())
