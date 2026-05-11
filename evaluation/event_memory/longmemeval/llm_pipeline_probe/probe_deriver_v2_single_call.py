"""Single-call deriver feasibility probe.

A SEGMENT (50-500 chars typical) is fed to one LLM call that returns a list
of DERIVATIVE strings. Each derivative will be embedded as a retrieval target
for that segment. The original segment stays verbatim in storage and is shown
to the reader at query time.

Goals of this probe:
  - Determine whether ONE LLM call with a principle-based prompt can handle
    all four typology cases (already-focused, overloaded, encoded,
    structured) plus identity/edge cases reliably.
  - Iterate prompt; diagnose failures.

Run:
    uv run python probe_deriver_v2_single_call.py
"""

from __future__ import annotations

import asyncio
import json
import os
import textwrap
from typing import Any

import openai
from dotenv import load_dotenv

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


# --------------------------------------------------------------------------
# PROMPT
# --------------------------------------------------------------------------

PROMPT_DERIVER = """\
You produce DERIVATIVES of a SEGMENT for a semantic retrieval index. The \
original SEGMENT is stored verbatim and shown to the reader; your \
derivatives are the strings that get embedded. A derivative succeeds when \
its embedding shares vocabulary and meaning with the kind of query a \
future user would write to find this segment.

Output a JSON list of derivative strings (0, 1, or many).

GUIDING QUESTION (INTERNAL ONLY). Before writing each derivative, think \
about the kind of query it would match. Do NOT write that query into the \
derivative; do not include words like \"Query:\", \"Answer:\", or \
\"Question:\". Each derivative is a plain statement (or, for bare-entity \
segments, a plain phrase) — what someone might write down as a fact, \
not a Q&A pair.

PRINCIPLES.

(P1) QUERY-LIKELY TOKENS APPEAR VERBATIM. The tokens carrying retrieval \
signal — named people, places, organizations, brands, products, works, \
dates, amounts, distinctive concept words, technical terms, identifiers \
— must appear verbatim in any derivative that covers them. Never invent \
entities the segment does not contain.

(P2) MATCH THE SHAPE OF A NATURAL QUERY. A derivative is itself a piece \
of natural text that an embedder will see. It should read like something \
a person could plausibly write or ask. Do NOT produce keyword-only \
fragments stripped of grammar; do NOT preserve formatting, code syntax, \
table delimiters, log brackets, or other artifacts that a query would \
not contain. Connective glue (\"I went to ...\", \"as I mentioned ...\") \
may be kept or dropped — whichever yields more natural query-shaped \
text.

(P3) ONE DERIVATIVE PER ATOMIC FACT, NO OVERLAP. If the segment contains \
several independent facts that would each answer a different query, \
emit one derivative per fact — each self-contained, and none repeating \
content already covered by another (two derivatives that paraphrase the \
same underlying fact are redundant; drop one). If the segment is a \
single focused fact, emit ONE derivative that stays close to the \
original sentence (paraphrasing a clean fact severs the entity \
relationships that make it findable). If the segment is a single bare \
term or entity, emit just that term unmodified.

(P4) WHEN THE SURFACE IS NOT QUERY-SHAPED, REPLACE IT. Some segments are \
not natural prose to begin with: encoded text (cipher, base64, morse), \
heavy abbreviation, code, tables, lists of raw numbers, log lines, \
markup, etc. For these, the surface itself will not match plain-language \
queries. Produce derivatives that re-express the content as natural \
prose:
   - If the encoding is recognizable (e.g., a Caesar cipher or a known \
     abbreviation), DECODE/EXPAND it and put the decoded form in a \
     derivative — that is the form a query will use.
   - Always include at least one ARTIFACT-DESCRIPTION derivative: one \
     fluent sentence that names what the segment IS (e.g., \"Python \
     function\", \"benchmark comparison table\", \"server error log \
     line\") and what it is ABOUT, listing the entities or labels it \
     covers and, if useful, representative values. This is what matches \
     queries like \"the benchmark table\" or \"the auth error\".
   - For tables or lists with several rows, you may also emit one \
     row-level derivative per row when each row is itself an answerable \
     fact. Keep these self-contained sentences, not pipe-delimited \
     fragments.
   - Do NOT preserve raw formatting: no pipes, no code syntax, no \
     bracketed timestamps. Render values in prose.

Do not duplicate derivatives. Do not invent content not in the segment.

SEGMENT:
{segment}
"""


# --------------------------------------------------------------------------
# OPENAI CALL
# --------------------------------------------------------------------------

DERIVATIVES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["derivatives"],
    "properties": {
        "derivatives": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
}


async def derive(
    client: openai.AsyncOpenAI,
    segment: str,
    *,
    model: str = "gpt-5.4-nano",
    reasoning: str = "low",
) -> list[str]:
    resp = await client.responses.create(
        model=model,
        input=PROMPT_DERIVER.format(segment=segment),
        reasoning={"effort": reasoning},
        text={
            "format": {
                "type": "json_schema",
                "name": "derivatives",
                "schema": DERIVATIVES_SCHEMA,
                "strict": True,
            }
        },
    )
    payload = json.loads(resp.output_text)
    return list(payload.get("derivatives", []))


# --------------------------------------------------------------------------
# CASES
# --------------------------------------------------------------------------

CRITICAL_CASES: list[tuple[str, str]] = [
    (
        "C1 already-focused",
        "I went to Tokyo last March with my wife Anne.",
    ),
    (
        "C2 overloaded",
        (
            "Last March I went to Tokyo with my wife Anne, stayed at the "
            "Park Hyatt for 5 nights at $400/night, and had ramen at "
            "Ichiran in Shibuya."
        ),
    ),
    (
        "C3 caesar cipher",
        "Khoor, zruog! Wklv lv d phvvdjh.",
    ),
    (
        "C4 markdown table",
        textwrap.dedent(
            """\
            | Model | Task A | Task B |
            | --- | --- | --- |
            | GPT-4 | 0.85 | 0.78 |
            | Claude | 0.91 | 0.82 |
            | Gemini | 0.79 | 0.81 |
            """
        ).strip(),
    ),
    (
        "C5 identity bare entity",
        "Tokyo",
    ),
    (
        "C6 code block",
        textwrap.dedent(
            """\
            def find_max(node):
                if node is None:
                    return float('-inf')
                return max(node.value, find_max(node.left), find_max(node.right))
            """
        ).strip(),
    ),
]

ADVERSARIAL_CASES: list[tuple[str, str]] = [
    (
        "A1 ISBN-bearing fact",
        "Picked up The Pragmatic Programmer (ISBN 978-0135957059) at Powell's for $35.",
    ),
    (
        "A2 metaphorical statement",
        "My PhD advisor was a lighthouse during the storm of my third year.",
    ),
    (
        "A3 mixed-language",
        'Tonight at the Brasserie Lipp we had cassoulet and creme brulee; the chef said "bon appétit" and we said arigato.',
    ),
    (
        "A4 abbreviation-dense",
        "TIL JFK was POTUS during the CMC in '62; SAC went to DEFCON 2.",
    ),
    (
        "A5 raw log line",
        "[2026-04-12T03:14:22Z] ERROR auth.service: token validation failed for user_id=42 reason=expired_jwt",
    ),
]


# --------------------------------------------------------------------------
# RUNNER
# --------------------------------------------------------------------------


async def run_case(
    client: openai.AsyncOpenAI, label: str, segment: str
) -> tuple[str, str, list[str]]:
    derivs = await derive(client, segment)
    return label, segment, derivs


def _short(s: str, n: int = 120) -> str:
    s = s.replace("\n", " \\n ")
    return s if len(s) <= n else s[: n - 1] + "…"


def print_block(label: str, segment: str, derivs: list[str]) -> None:
    print()
    print(f"=== {label} ===")
    print(f"  SEGMENT: {_short(segment, 200)}")
    print(f"  N_DERIVATIVES: {len(derivs)}")
    for i, d in enumerate(derivs):
        print(f"    [{i}] {d}")


async def main() -> None:
    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print("# DERIVER v2 SINGLE-CALL PROBE")
    print("# model=gpt-5.4-nano reasoning=low")

    print("\n## CRITICAL CASES")
    sem = asyncio.Semaphore(8)

    async def go(label: str, seg: str) -> tuple[str, str, list[str]]:
        async with sem:
            return await run_case(client, label, seg)

    crit_results = await asyncio.gather(*(go(label, s) for label, s in CRITICAL_CASES))
    for label, seg, derivs in crit_results:
        print_block(label, seg, derivs)

    print("\n## ADVERSARIAL CASES")
    adv_results = await asyncio.gather(
        *(go(label, s) for label, s in ADVERSARIAL_CASES)
    )
    for label, seg, derivs in adv_results:
        print_block(label, seg, derivs)

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
