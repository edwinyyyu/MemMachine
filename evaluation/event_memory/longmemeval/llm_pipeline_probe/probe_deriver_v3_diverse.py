"""Diverse-bench deriver probe (iter-2).

Expands the iter-1 (probe_deriver_v2_single_call.py) bench with 16 new cases
spanning length, encoding/format, and ambiguity/genre axes. Re-runs the
6 critical and 5 adversarial cases for regression check.

Run:
    uv run python probe_deriver_v3_diverse.py
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
# PROMPT (unchanged from iter-1)
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

(P5) IF THE TOKENS A QUERY WOULD USE DIFFER FROM THE TOKENS IN THE \
SEGMENT, INCLUDE BOTH FORMS. The retrieval target must contain whatever \
form a future query will plausibly use. When the segment uses tokens \
that a likely query would NOT use — heavy acronym shorthand (JFK, CMC, \
POTUS), highly compressed jargon, or text written in a script or \
language different from the one a likely query would be written in — \
emit at least one derivative that contains the EXPANDED or TRANSLATED \
form alongside the original. Keep the segment's original tokens in some \
derivative as well, so queries written either way can match. This is \
about making the index findable; it is not translation for its own \
sake. Apply this only when the gap between segment-form and query-form \
is real; for content already in plain English with full names, do \
nothing extra.

(P6) PRESERVE LITERAL QUOTED SPEECH WHEN PRESENT. If the segment shows \
text inside quotation marks (e.g., something a person said), keep at \
least one derivative that retains the quoted text verbatim with its \
attribution. Indirect-speech paraphrases lose the exact tokens a query \
might cite.

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
    prompt: str = PROMPT_DERIVER,
    model: str = "gpt-5.4-nano",
    reasoning: str = "low",
) -> list[str]:
    resp = await client.responses.create(
        model=model,
        input=prompt.format(segment=segment),
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

# Each case: (label, segment, expected_behavior_note)

CRITICAL_CASES: list[tuple[str, str, str]] = [
    (
        "C1 already-focused",
        "I went to Tokyo last March with my wife Anne.",
        "1 derivative close to original; preserves Tokyo, Anne, March.",
    ),
    (
        "C2 overloaded",
        (
            "Last March I went to Tokyo with my wife Anne, stayed at the "
            "Park Hyatt for 5 nights at $400/night, and had ramen at "
            "Ichiran in Shibuya."
        ),
        "~3 derivatives: Tokyo trip with Anne; Park Hyatt $400/5nt; ramen at Ichiran/Shibuya.",
    ),
    (
        "C3 caesar cipher",
        "Khoor, zruog! Wklv lv d phvvdjh.",
        "Decoded 'Hello, world! This is a message.' + artifact desc.",
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
        "Artifact desc + per-row prose facts (no pipes).",
    ),
    (
        "C5 identity bare entity",
        "Tokyo",
        "Single derivative: 'Tokyo'.",
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
        "Artifact desc: Python function find_max returning max value in tree.",
    ),
]

ADVERSARIAL_CASES: list[tuple[str, str, str]] = [
    (
        "A1 ISBN-bearing fact",
        "Picked up The Pragmatic Programmer (ISBN 978-0135957059) at Powell's for $35.",
        "Preserve title, ISBN, Powell's, $35.",
    ),
    (
        "A2 metaphorical statement",
        "My PhD advisor was a lighthouse during the storm of my third year.",
        "Preserve metaphor; describes PhD advisor's role in third year.",
    ),
    (
        "A3 mixed-language",
        'Tonight at the Brasserie Lipp we had cassoulet and creme brulee; the chef said "bon appétit" and we said arigato.',
        "Preserve Brasserie Lipp, cassoulet, creme brulee, chef quote, arigato.",
    ),
    (
        "A4 abbreviation-dense",
        "TIL JFK was POTUS during the CMC in '62; SAC went to DEFCON 2.",
        "Expand JFK/POTUS/CMC/SAC AND keep verbatim acronyms.",
    ),
    (
        "A5 raw log line",
        "[2026-04-12T03:14:22Z] ERROR auth.service: token validation failed for user_id=42 reason=expired_jwt",
        "Artifact desc as auth error log; render values in prose.",
    ),
]


# --- DIVERSE NEW CASES (D1-D16) ---

DIVERSE_CASES: list[tuple[str, str, str]] = [
    # ----- LENGTH AXIS -----
    (
        "D1 long single-topic",
        textwrap.dedent("""\
            We finally finished restoring the 1967 Ford Mustang my grandfather \
            left me. It took almost three years from the day we towed it out \
            of the barn in upstate New York. The engine block had completely \
            seized up from a decade of moisture, so we ended up rebuilding the \
            289 V8 from scratch, with new pistons, rings, and a Holley \
            four-barrel carburetor. Bodywork was the worst part — the rear \
            quarter panels were so corroded we had to fabricate replacements \
            from sheet metal at a friend's shop in Poughkeepsie. We went with \
            the original Wimbledon White over a black interior, which was the \
            color my grandfather always said he wanted but never got around \
            to repainting. The chrome trim was sent out to a specialty \
            re-plater in Pennsylvania. We rebuilt the suspension with new \
            bushings, replaced the rear leaf springs, and put in a Borg-Warner \
            T-10 four-speed transmission to match the original spec. Last \
            weekend I finally drove it down to the Hudson Valley cars-and-coffee \
            and parked it next to my friend Marcus's '69 Camaro. It still pulls \
            slightly to the right under hard braking and the radio is a modern \
            replacement, but the rest is as close to factory as we could get. \
            My grandmother cried when she saw it.\
        """).strip(),
        "Should NOT decompose per paragraph; either ~1 long derivative or a few atomic facts (car identity, engine rebuild, paint, transmission, debut), NOT one per sentence.",
    ),
    (
        "D2 long multi-topic",
        textwrap.dedent("""\
            This week was packed. On Monday, my sister Priya gave birth to a \
            baby girl named Aisha at Mount Sinai in New York; she's 7 lbs 2 oz \
            and they're all doing well. Wednesday the renovation crew finally \
            finished tiling our master bathroom — we went with the navy hex \
            tiles from Clé and a Kohler Purist faucet, and the total came in \
            around $14,200, about $1,800 over our original budget. Thursday I \
            had a long phone call with Dr. Henley about my dad's cardiology \
            results: his ejection fraction has improved from 38% to 47% since \
            starting the new beta blocker, and they're cautiously optimistic. \
            Friday I got the offer letter from Stripe — Senior Staff \
            Engineer, base $310k, signing bonus $80k, start date June 22nd. \
            And tomorrow we're hosting eight people for our delayed Diwali \
            dinner; I'm making the lamb biryani my mother taught me before \
            she passed.\
        """).strip(),
        "~5 derivatives: Aisha's birth; bathroom reno cost; dad's cardiology improvement; Stripe offer details; Diwali dinner.",
    ),
    # ----- ENCODING / FORMAT AXIS -----
    (
        "D3 LaTeX equation",
        r"$$\int_0^{\pi} \sin(x)\, dx = 2$$",
        "Artifact desc: LaTeX integral; restate value (integral of sin from 0 to pi equals 2).",
    ),
    (
        "D4 JSON dump",
        '{"event": "user_signup", "user_id": 42, "ts": "2024-03-15", "source": "ads"}',
        "Artifact desc as JSON event record + atomic facts (user_signup, user_id 42, 2024-03-15, source=ads).",
    ),
    (
        "D5 ASCII art / diagram",
        textwrap.dedent("""\
            +----------+      +-----------+      +---------+
            | Producer | ---> |  Kafka    | ---> | Consumer |
            +----------+      +-----------+      +---------+
        """).strip(),
        "Artifact desc: pipeline diagram Producer -> Kafka -> Consumer; do NOT preserve box characters.",
    ),
    (
        "D6 base64 of plain text",
        # base64 of "The launch is scheduled for July 14 at Cape Canaveral."
        "VGhlIGxhdW5jaCBpcyBzY2hlZHVsZWQgZm9yIEp1bHkgMTQgYXQgQ2FwZSBDYW5hdmVyYWwu",
        "Decode + emit decoded form: launch scheduled July 14 at Cape Canaveral.",
    ),
    (
        "D7 nested code with comments",
        textwrap.dedent("""\
            def normalize_email(addr):
                # Strip trailing whitespace and lowercase the domain.
                # Gmail-style dots in the local part are preserved (we no longer
                # canonicalize them after the 2024 incident).
                local, _, domain = addr.partition('@')
                return f\"{local}@{domain.lower().strip()}\"
        """).strip(),
        "Artifact desc reflecting the comment intent (lowercase domain, preserve gmail dots after 2024 incident); should NOT ignore comments.",
    ),
    (
        "D8 Cyrillic prose",
        "Анна живёт в Санкт-Петербурге и работает архитектором.",
        "Preserve Anna / Saint Petersburg / architect; transliteration optional but English query forms helpful.",
    ),
    (
        "D9 CJK prose",
        "山田さんは京都で寿司屋を経営しています。",
        "Preserve Mr. Yamada / Kyoto / sushi restaurant; English form for retrieval.",
    ),
    (
        "D10 chemistry formula",
        "H2SO4 reacts with NaOH at 25°C to produce Na2SO4 and water.",
        "Preserve H2SO4, NaOH, Na2SO4, 25°C; could expand to sulfuric acid / sodium hydroxide.",
    ),
    (
        "D11 git-commit-message",
        "Fix race in event_memory's encode_events; partition lock now acquired before deriver dispatch",
        "1 derivative restating the fix near-verbatim (preserves event_memory, encode_events, partition lock, deriver).",
    ),
    (
        "D12 dense numeric list",
        "Q1: $1.2M, Q2: $1.4M, Q3: $1.1M, Q4: $1.5M, total $5.2M for 2024",
        "Per-quarter facts (or summary + total) preserving 2024 and dollar amounts in prose.",
    ),
    # ----- AMBIGUITY / GENRE AXIS -----
    (
        "D13 metaphor double-meaning",
        "He's the lighthouse in our team's storm.",
        "Preserve lighthouse/storm metaphor; describes a teammate's stabilizing role.",
    ),
    (
        "D14 attributed quote",
        'Anne said "I\'d love to try sushi in Tokyo someday."',
        "Preserve attribution to Anne and the literal quote; preserve sushi/Tokyo.",
    ),
    (
        "D15 hedged claim",
        "I think they might be moving to Berlin in autumn, possibly September.",
        "Preserve the hedging (I think / might / possibly); preserve Berlin/autumn/September.",
    ),
    (
        "D16 list of preferences",
        "Pinot Noir, dark chocolate, jazz, sailing",
        "Either keep as a list-derivative or one item each as plain phrases; do NOT invent context.",
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


def print_block(label: str, segment: str, expected: str, derivs: list[str]) -> None:
    print()
    print(f"=== {label} ===")
    print(f"  SEGMENT: {_short(segment, 220)}")
    print(f"  EXPECTED: {expected}")
    print(f"  N_DERIVATIVES: {len(derivs)}")
    for i, d in enumerate(derivs):
        print(f"    [{i}] {d}")


async def main() -> None:
    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print("# DERIVER v3 DIVERSE-BENCH PROBE")
    print("# model=gpt-5.4-nano reasoning=low (Responses API + json_schema)")
    print(
        f"# critical={len(CRITICAL_CASES)}  adversarial={len(ADVERSARIAL_CASES)}  diverse={len(DIVERSE_CASES)}"
    )

    sem = asyncio.Semaphore(8)

    async def go(label: str, seg: str, exp: str) -> tuple[str, str, str, list[str]]:
        async with sem:
            lbl, s, d = await run_case(client, label, seg)
            return lbl, s, exp, d

    print("\n## CRITICAL CASES (regression check)")
    crit_results = await asyncio.gather(
        *(go(label, s, e) for label, s, e in CRITICAL_CASES)
    )
    for label, seg, exp, derivs in crit_results:
        print_block(label, seg, exp, derivs)

    print("\n## ADVERSARIAL CASES (regression check)")
    adv_results = await asyncio.gather(
        *(go(label, s, e) for label, s, e in ADVERSARIAL_CASES)
    )
    for label, seg, exp, derivs in adv_results:
        print_block(label, seg, exp, derivs)

    print("\n## DIVERSE CASES (D1-D16)")
    div_results = await asyncio.gather(
        *(go(label, s, e) for label, s, e in DIVERSE_CASES)
    )
    for label, seg, exp, derivs in div_results:
        print_block(label, seg, exp, derivs)

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
