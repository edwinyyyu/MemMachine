"""Two-stage LLM deriver: classify case, then dispatch to a case-specific rewriter.

Stage 1 (CLASSIFIER): one terse call decides which of 4 cases the segment is:
    focused | overloaded | encoded | structured
Stage 2 (REWRITER): a case-specific prompt produces 0-N derivative strings.

Run: uv run python probe_deriver_v2_two_stage.py
"""

from __future__ import annotations

import asyncio
import json
import os

import openai
from dotenv import load_dotenv

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


# =============================================================================
# Stage 1: classify the segment into one of four cases
# =============================================================================
STAGE1_PROMPT = """\
You are a router for a retrieval-index ingestion pipeline. Your only job is to \
classify a SEGMENT of text into exactly one of four cases. Pick the case that \
best fits the WHOLE segment as written.

CASES:
- "focused": natural-language prose centered on a single fact, statement, \
identity, or short claim. A query about it would rephrase that one fact.
- "overloaded": natural-language prose that packs multiple independent facts \
(distinct events, entities, transactions, attributes) into one passage. Each \
fact could be queried independently.
- "encoded": the surface form is not directly readable as plain natural \
language — it is a cipher, transliteration, abbreviation system, escape \
sequence, base-N blob, or otherwise transformed text whose meaning is hidden \
under the encoding.
- "structured": the segment is non-prose: tables, code, markup, raw numeric \
grids, logs, schemas, key-value dumps. Its information lives in structure, \
not sentences.

Pick the single best fit. If multiple seem to apply, pick the one whose \
HANDLING differs most from plain reading: structured > encoded > overloaded > \
focused.

SEGMENT:
{text}"""

STAGE1_SCHEMA = {
    "name": "classify",
    "schema": {
        "type": "object",
        "properties": {
            "case": {
                "type": "string",
                "enum": ["focused", "overloaded", "encoded", "structured"],
            },
        },
        "required": ["case"],
        "additionalProperties": False,
    },
    "strict": True,
}


# =============================================================================
# Stage 2: case-specific rewriters
# =============================================================================
# Shared principle (kept short, repeated in each prompt so each is self-contained):
# A derivative is text we will EMBED. The original segment is kept verbatim and
# shown to the reader at retrieval time. So derivatives must contain the
# distinctive query-words a future user is likely to use: named entities,
# places, people, brands, works, dates, numbers, prices, distinctive concept
# terms. Connective tissue ("I went to", "this is because") may be normalized.

STAGE2_FOCUSED = """\
You produce embedding-target text for a retrieval index. The original segment \
is already kept verbatim for the reader; you write what gets embedded.

This segment expresses a SINGLE focused fact, identity, or claim. Emit \
exactly ONE derivative that captures that fact as a complete, well-formed \
sentence (or, for a bare identity, the identity itself). The derivative \
should read like a normal sentence a human would write — not a keyword \
list. Distinctive query-words must appear verbatim: named entities, people, \
places, brands, works, titles, dates, numbers, prices, concept terms. You \
may drop or normalize purely-connective phrasing and pure affect, but the \
derivative must remain grammatical and self-contained. Default to staying \
close to the segment's wording rather than compressing.

Return JSON: {{"derivatives": ["..."]}}.

SEGMENT:
{text}"""

STAGE2_OVERLOADED = """\
You produce embedding-target text for a retrieval index. The original segment \
is already kept verbatim for the reader; you write what gets embedded.

This segment packs MULTIPLE independent facts into one passage. Split them. \
Emit one derivative per fact a future user might query independently \
(distinct events, entities, transactions, attributes, places, prices, dates). \
Each derivative is its own short sentence focused on one fact. Keep every \
distinctive query-word verbatim in the derivative that carries that fact: \
named entities, people, places, brands, works, dates, numbers, prices, \
concept terms. Do not invent facts not present in the segment, and do not \
duplicate the same fact across derivatives.

Return JSON: {{"derivatives": ["...", "...", ...]}}.

SEGMENT:
{text}"""

STAGE2_ENCODED = """\
You produce embedding-target text for a retrieval index. The original segment \
is already kept verbatim for the reader; you write what gets embedded.

This segment is in an ENCODED surface form (cipher, transliteration, \
abbreviation system, escape sequence, base-N blob, or similar). Decode it \
into readable language if you can identify the encoding with reasonable \
confidence. Emit:
  - one derivative that is the decoded plain-language content (so queries \
about the underlying meaning match), AND
  - one derivative that names the encoding type and notes that this segment \
is an instance of it (so queries about the encoding itself match).
If you cannot decode confidently, emit only the encoding-name derivative. \
Keep distinctive query-words from the decoded content verbatim.

Return JSON: {{"derivatives": ["...", "..."]}}.

SEGMENT:
{text}"""

STAGE2_STRUCTURED = """\
You produce embedding-target text for a retrieval index. The original segment \
is already kept verbatim for the reader; you write what gets embedded.

This segment is STRUCTURED non-prose (table, code, markup, log, schema, raw \
numeric grid, key-value dump). Embedding the raw structure matches poorly \
against natural-language queries. Instead, write a short prose description \
that names: (a) what the artifact IS (e.g., "Python recursive function", \
"markdown comparison table", "SQL schema"), and (b) the distinctive \
query-words inside it that a future user might ask about — function/class/ \
column/identifier names, model names, metric names, units, headings, key \
strings, and any standout numbers. Keep those query-words verbatim. You may \
emit additional derivatives if the artifact contains multiple independently \
queryable items (e.g., a table where each row is its own fact).

Return JSON: {{"derivatives": ["...", ...]}}.

SEGMENT:
{text}"""


STAGE2_PROMPTS = {
    "focused": STAGE2_FOCUSED,
    "overloaded": STAGE2_OVERLOADED,
    "encoded": STAGE2_ENCODED,
    "structured": STAGE2_STRUCTURED,
}

STAGE2_SCHEMA = {
    "name": "rewrite",
    "schema": {
        "type": "object",
        "properties": {
            "derivatives": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["derivatives"],
        "additionalProperties": False,
    },
    "strict": True,
}


# =============================================================================
# pipeline
# =============================================================================
DEFAULT_MODEL = "gpt-5.4-nano"
DEFAULT_REASONING = "low"


async def _call_json(
    client: openai.AsyncOpenAI,
    model: str,
    prompt: str,
    schema: dict,
    reasoning: str | None,
) -> dict:
    kwargs: dict = {
        "model": model,
        "input": prompt,
        "text": {"format": {"type": "json_schema", **schema}},
    }
    if reasoning and model.startswith("gpt-5"):
        kwargs["reasoning"] = {"effort": reasoning}
    resp = await client.responses.create(**kwargs)
    return json.loads(resp.output_text)


async def classify(
    client: openai.AsyncOpenAI,
    segment: str,
    model: str = DEFAULT_MODEL,
    reasoning: str = DEFAULT_REASONING,
) -> str:
    out = await _call_json(
        client,
        model,
        STAGE1_PROMPT.format(text=segment),
        STAGE1_SCHEMA,
        reasoning,
    )
    return out["case"]


async def rewrite(
    client: openai.AsyncOpenAI,
    segment: str,
    case: str,
    model: str = DEFAULT_MODEL,
    reasoning: str = DEFAULT_REASONING,
) -> list[str]:
    prompt = STAGE2_PROMPTS[case].format(text=segment)
    out = await _call_json(client, model, prompt, STAGE2_SCHEMA, reasoning)
    return out["derivatives"]


async def derive(
    client: openai.AsyncOpenAI,
    segment: str,
    model: str = DEFAULT_MODEL,
    reasoning: str = DEFAULT_REASONING,
) -> tuple[str, list[str]]:
    case = await classify(client, segment, model, reasoning)
    derivatives = await rewrite(client, segment, case, model, reasoning)
    return case, derivatives


# =============================================================================
# critical + adversarial cases
# =============================================================================
CRITICAL_CASES: list[tuple[str, str, str]] = [
    (
        "C1 already-focused",
        "focused",
        "I went to Tokyo last March with my wife Anne.",
    ),
    (
        "C2 overloaded",
        "overloaded",
        "Last March I went to Tokyo with my wife Anne, stayed at the Park Hyatt "
        "for 5 nights at $400/night, and had ramen at Ichiran in Shibuya.",
    ),
    (
        "C3 caesar cipher",
        "encoded",
        "Khoor, zruog! Wklv lv d phvvdjh.",
    ),
    (
        "C4 markdown table",
        "structured",
        "| Model | Task A | Task B |\n"
        "| --- | --- | --- |\n"
        "| GPT-4 | 0.85 | 0.78 |\n"
        "| Claude | 0.91 | 0.82 |\n"
        "| Gemini | 0.79 | 0.81 |",
    ),
    (
        "C5 identity",
        "focused",
        "Tokyo",
    ),
    (
        "C6 code block",
        "structured",
        "def find_max(node):\n"
        "    if node is None: return float('-inf')\n"
        "    return max(node.value, find_max(node.left), find_max(node.right))",
    ),
]

ADVERSARIAL_CASES: list[tuple[str, str]] = [
    # A1: prose that names entities but fundamentally one fact (book recommendation).
    (
        "A1 single recommendation with multiple entities",
        "I'm reading 'The Three-Body Problem' by Liu Cixin and loving it.",
    ),
    # A2: log line — structured but content-bearing.
    (
        "A2 log line",
        "2026-04-12T03:14:07Z ERROR payments.processor: charge_id=ch_18ABc declined "
        "(insufficient_funds) user_id=usr_9912 amount_cents=4500",
    ),
    # A3: morse code.
    (
        "A3 morse",
        "... --- ...",
    ),
    # A4: borderline overloaded — two clauses, same trip, related entities.
    (
        "A4 borderline two-clause",
        "After the meeting with Priya about the Q4 forecast, I drove to Half Moon Bay "
        "and watched the sunset.",
    ),
    # A5: prose-like but base64.
    (
        "A5 base64 blob",
        "VGhlIHF1aWNrIGJyb3duIGZveCBqdW1wcyBvdmVyIHRoZSBsYXp5IGRvZy4=",
    ),
    # A6: a per-row queryable table (scores by player).
    (
        "A6 per-row table",
        "| Player | Goals | Assists |\n"
        "| --- | --- | --- |\n"
        "| Messi | 12 | 8 |\n"
        "| Mbappé | 14 | 5 |\n"
        "| Haaland | 18 | 3 |",
    ),
]


# =============================================================================
# runner
# =============================================================================
async def main() -> None:
    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print("=" * 72)
    print("TWO-STAGE DERIVER PROBE")
    print(f"model={DEFAULT_MODEL} reasoning={DEFAULT_REASONING}")
    print("=" * 72)

    print("\n## CRITICAL CASES (C1-C6)\n")
    crit_results = await asyncio.gather(
        *(derive(client, seg) for _, _, seg in CRITICAL_CASES)
    )
    for (label, expected, seg), (case, derivs) in zip(
        CRITICAL_CASES, crit_results, strict=False
    ):
        passed_stage1 = case == expected
        print(f"--- {label} ---")
        snippet = seg if len(seg) <= 200 else seg[:200] + " …"
        print(f"SEGMENT: {snippet!r}")
        print(
            f"STAGE1: case={case!r} (expected={expected!r}) "
            f"{'PASS' if passed_stage1 else 'FAIL'}"
        )
        print(f"STAGE2: {len(derivs)} derivative(s)")
        for i, d in enumerate(derivs):
            print(f"  [{i}] {d}")
        print()

    print("\n## ADVERSARIAL CASES\n")
    adv_results = await asyncio.gather(
        *(derive(client, seg) for _, seg in ADVERSARIAL_CASES)
    )
    for (label, seg), (case, derivs) in zip(
        ADVERSARIAL_CASES, adv_results, strict=False
    ):
        print(f"--- {label} ---")
        snippet = seg if len(seg) <= 200 else seg[:200] + " …"
        print(f"SEGMENT: {snippet!r}")
        print(f"STAGE1: case={case!r}")
        print(f"STAGE2: {len(derivs)} derivative(s)")
        for i, d in enumerate(derivs):
            print(f"  [{i}] {d}")
        print()

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
