"""LLM-based semantic segmenter v2: rewrite-then-split as memory consolidation.

Two modes are compared:
  R  - LLM rewrites the passage in the original speaker's voice, dropping
       generic scaffolding while keeping distinctive wording verbatim.
       Splitting is then handled deterministically by RecursiveCharacterTextSplitter.
  RS - LLM does rewrite AND split in one step, emitting a JSON array of
       standalone segments. The LLM may add tiny in-voice clarifications
       to make each segment self-contained, since it knows the boundaries.

Shared principles (same as the cue-worthiness filter):
  - Keep what a human would remember weeks later: named entities, places,
    brands, works, dates, prices, preferences, plans, decisions, specific
    factual claims, AND distinctive phrasing (eccentric greetings, oddly
    chosen words). Do NOT swap synonyms — "fabulous" ≠ "awesome".
  - Drop only generic scaffolding: a bland "Hi", "great question!", agent
    boilerplate, restated user questions, "I hope this helps!".
  - Stay in the SAME perspective the source used. If first-person, keep
    first-person; do not narrate. (Mixing perspectives across segments
    breaks the assumption that segments inherit source ProducerContext.)

This probe outputs the rewrite (mode R) or list (mode RS) so we can
eyeball fidelity, scaffolding-drop, and any hallucination.
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


# Mode R: rewrite only. Splitting is the segmenter's downstream concern.
PROMPT_R = """\
You compress a passage into the parts a human would still remember weeks \
later, then output the compressed text in the speaker's original voice. \
The output is later embedded as a memory.

Rules:
  1. PRESERVE distinctive wording verbatim. "Fabulous" is not "awesome", \
"gobsmacked" is not "surprised" — connotation matters. Do not swap synonyms.
  2. KEEP every named entity, place, person, brand, work, date, price, \
preference, plan, decision, and specific factual claim.
  3. KEEP eccentric or memorable openings/closings ("Yo yo yo", "Cheers, \
mate"). DROP only generic scaffolding ("Hi!", "What a great question!", \
"I hope this helps!", "Let me know if you have any other questions", \
restating what the other party just asked).
  4. Stay in the SAME perspective and voice the source used. Do not \
narrate or summarize in third person. Do not generalize. Quote and \
condense; do not abstract.
  5. Output the compressed passage and nothing else. No preface, no \
headers, no explanation of what you did.

PASSAGE:
{passage}"""


# Mode H: rewrite + emit explicit split markers in plain text.
# A trivial splitter splits on the sentinel only; no other markdown/format
# heuristic, so "4." or a stray bullet can never become a 2-char segment.
SEGMENT_SPLIT_MARKER = "\n<<<SPLIT>>>\n"

PROMPT_H = """\
You compress a passage into the parts a human would still remember weeks \
later, then mark the boundaries between standalone memory segments.

Rules:
  1. PRESERVE distinctive wording verbatim. "Fabulous" is not "awesome", \
"gobsmacked" is not "surprised" — connotation matters. Do not swap synonyms.
  2. KEEP every named entity, place, person, brand, work, date, price, \
preference, plan, decision, and specific factual claim.
  3. KEEP eccentric or memorable openings/closings. DROP only generic \
scaffolding ("Hi!", "What a great question!", "I hope this helps!", \
"Let me know if you have any other questions", restating what the other \
party just asked).
  4. Stay in the SAME perspective and voice the source used. Do not \
narrate or summarize in third person. Do not generalize. Quote and \
condense; do not abstract.
  5. To mark a segment boundary, emit EXACTLY this on its own line:
     <<<SPLIT>>>
     Use it only between coherent standalone memory segments. Each segment \
between markers must be readable on its own; if it depends on something \
introduced earlier in the passage, restate the referent in the same voice. \
Do NOT introduce facts not in the passage.
  6. Each segment between markers should be at most ~{max_chars} characters.
  7. Output the compressed passage, with <<<SPLIT>>> markers, and nothing \
else. No preface, no JSON, no headers, no markdown fences.

PASSAGE:
{passage}"""


# Mode RS: rewrite + split in one step. JSON array output.
PROMPT_RS = """\
You compress a passage into the parts a human would still remember weeks \
later, AND split the result into a list of standalone memory segments. \
Each segment is later embedded and may be retrieved on its own.

Rules:
  1. PRESERVE distinctive wording verbatim. "Fabulous" is not "awesome", \
"gobsmacked" is not "surprised" — connotation matters. Do not swap synonyms.
  2. KEEP every named entity, place, person, brand, work, date, price, \
preference, plan, decision, and specific factual claim.
  3. KEEP eccentric or memorable openings/closings ("Yo yo yo", "Cheers, \
mate"). DROP only generic scaffolding ("Hi!", "What a great question!", \
"I hope this helps!", "Let me know if you have any other questions", \
restating what the other party just asked).
  4. Stay in the SAME perspective and voice the source used. Do not \
narrate or summarize in third person. Do not generalize. Quote and \
condense; do not abstract.
  5. Each segment must be readable standalone. If a segment depends on \
something introduced earlier in the passage (e.g., "the trip"), restate \
the referent in the same voice ("the Tokyo trip planned for May"). Do \
NOT introduce facts that are not in the passage.
  6. Each segment should be at most ~{max_chars} characters.
  7. Output a JSON array of strings, in order, and nothing else. No \
preface, no markdown fences. Do not include keys or any other structure.

PASSAGE:
{passage}"""


async def call(
    client: openai.AsyncOpenAI, model: str, prompt: str, reasoning: str | None
) -> str:
    kwargs: dict = {"model": model, "input": prompt}
    if reasoning and model.startswith("gpt-5"):
        kwargs["reasoning"] = {"effort": reasoning}
    resp = await client.responses.create(**kwargs)
    return (resp.output_text or "").strip()


def parse_h(reply: str) -> list[str]:
    """Split on the <<<SPLIT>>> marker; trim each segment."""
    raw = reply.strip()
    parts = [p.strip() for p in raw.split("<<<SPLIT>>>")]
    return [p for p in parts if p]


def parse_rs(reply: str) -> list[str] | None:
    s = reply.strip()
    # Tolerate occasional ```json fences.
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        s = s.removesuffix("```")
    try:
        out = json.loads(s)
    except Exception:
        return None
    if isinstance(out, list) and all(isinstance(x, str) for x in out):
        return out
    return None


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.4-nano")
    parser.add_argument("--reasoning", default="low")
    parser.add_argument("--mode", default="RS", choices=["R", "RS"])
    parser.add_argument("--max-chars", type=int, default=500)
    parser.add_argument("--per-bucket", type=int, default=2)
    parser.add_argument("--min-length", type=int, default=400)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    bins = collect()
    samples = []
    for b, ts in bins.items():
        long_ts = [t for t in ts if len(t) >= args.min_length]
        for t in long_ts[: args.per_bucket]:
            samples.append((b, t))

    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(args.concurrency)

    async def go(text: str) -> str:
        prompt = (PROMPT_R if args.mode == "R" else PROMPT_RS).format(
            passage=text, max_chars=args.max_chars
        )
        async with sem:
            return await call(client, args.model, prompt, args.reasoning)

    replies = await asyncio.gather(*(go(t) for _, t in samples))
    await client.close()

    print(
        f"# model={args.model} reasoning={args.reasoning} "
        f"mode={args.mode} max_chars={args.max_chars}"
    )
    for (bucket, text), reply in zip(samples, replies, strict=False):
        print()
        print(f"=== [{bucket}] (input len={len(text)}) ===")
        snippet = text if len(text) <= 350 else text[:350] + " …"
        print("-- INPUT:")
        print(snippet)
        print("-- OUTPUT:")
        if args.mode == "R":
            print(f"(len={len(reply)})")
            print(reply)
        else:
            segs = parse_rs(reply)
            if segs is None:
                print(f"!! PARSE FAILED. raw reply ({len(reply)} chars):")
                print(reply[:600])
            else:
                total = sum(len(s) for s in segs)
                print(f"({len(segs)} segments, total chars={total})")
                for i, s in enumerate(segs):
                    flag = "  TOO LONG" if len(s) > args.max_chars * 1.2 else ""
                    print(f"  seg[{i}] (len={len(s)}){flag}: {s}")


if __name__ == "__main__":
    asyncio.run(main())
