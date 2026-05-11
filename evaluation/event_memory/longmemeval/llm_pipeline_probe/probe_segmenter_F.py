"""LLM segmenter, Mode F — LLM emits segments directly, no pre-split.

Architecture:
  1. Pass the FULL passage to the LLM. No sentence-splitting, no
     line-numbering, no pre-tokenization.
  2. LLM emits a JSON list of segments via strict json_schema.
     Each segment is a verbatim (or as-verbatim-as-possible) quote
     from the passage. Generic scaffolding is dropped by NOT being
     included in any segment.
  3. json_schema enforces structure AND per-segment maxLength so the
     model cannot exceed the cap.

Properties this gets us:
  - The LLM controls boundaries — it knows where to split coherently
    and never has to deal with a garbage unit produced by a regex.
  - Strict json_schema → 100% parse, 100% per-segment length compliance.
  - Verbatim is a prompt obligation, not a structural one. We measure
    fidelity post-hoc as "what fraction of segment chars appear in the
    source as a contiguous substring" (longest-substring coverage).

If fidelity holds at >=99% across model/reasoning cells, this Mode F
satisfies all three robustness goals without sentence-splitting.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

import openai
from dotenv import load_dotenv
from sample_chunks import collect

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


def build_schema(max_chars: int) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "segments": {
                "type": "array",
                "items": {
                    "type": "string",
                    "maxLength": max_chars,
                },
            }
        },
        "required": ["segments"],
    }


PROMPT_F = """\
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
  5. STANDALONE — each segment should read on its own. If a quote \
depends on a referent introduced earlier (e.g., uses "the trip" for \
"the Tokyo trip in May"), include the earlier sentence that names the \
referent in the same segment by widening the quote to start there.
  6. Each segment's length is capped automatically at {max_chars} \
characters by the response schema.

Output: a JSON object {{ "segments": [...] }} and nothing else.

PASSAGE:
{passage}"""


async def call(client, model, prompt, reasoning, schema):
    kwargs: dict = {
        "model": model,
        "input": prompt,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "segments_response",
                "schema": schema,
                "strict": True,
            }
        },
    }
    if reasoning and model.startswith("gpt-5"):
        kwargs["reasoning"] = {"effort": reasoning}
    resp = await client.responses.create(**kwargs)
    raw = (resp.output_text or "").strip()
    return json.loads(raw)


async def segment_one(client, model, text, reasoning, max_chars):
    schema = build_schema(max_chars)
    prompt = PROMPT_F.format(passage=text, max_chars=max_chars)
    parsed = await call(client, model, prompt, reasoning, schema)
    return parsed.get("segments", [])


# Fidelity check: longest contiguous substring of segment that appears
# in source. We average this fraction over all segment chars.
def verbatim_ratio(seg: str, src: str) -> float:
    """Greedy LCS-by-anchor: find the longest run of seg that's in src,
    repeat on the leftover, sum.

    A pure verbatim segment scores 1.0; one with paraphrased pieces
    scores < 1.0 in proportion to non-verbatim spans.
    """
    if not seg:
        return 1.0
    remaining = seg
    matched = 0
    while remaining:
        # Find the longest prefix of `remaining` that appears in src.
        best = 0
        for L in range(len(remaining), 0, -1):
            if remaining[:L] in src:
                best = L
                break
        if best == 0:
            # Skip one char and continue
            remaining = remaining[1:]
            continue
        matched += best
        remaining = remaining[best:]
    return matched / len(seg)


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

    async def go(text):
        async with sem:
            return await segment_one(
                client, args.model, text, args.reasoning, args.max_chars
            )

    rep = await asyncio.gather(*(go(t) for _, t in samples))
    await client.close()

    print(f"# model={args.model} reasoning={args.reasoning} max_chars={args.max_chars}")
    for (bucket, src), segs in zip(samples, rep, strict=False):
        avg_verb = sum(verbatim_ratio(s, src) for s in segs) / max(len(segs), 1)
        max_len = max((len(s) for s in segs), default=0)
        print()
        print(
            f"=== [{bucket}] src_len={len(src)} -> {len(segs)} segs, "
            f"max_len={max_len}, avg_verbatim={avg_verb:.0%} ==="
        )
        print("-- ORIG (first 300):")
        print(src[:300])
        for i, s in enumerate(segs):
            v = verbatim_ratio(s, src)
            print(
                f"-- seg[{i}] (len={len(s)}, verbatim={v:.0%}): {s[:200]}{'…' if len(s) > 200 else ''}"
            )


if __name__ == "__main__":
    asyncio.run(main())
