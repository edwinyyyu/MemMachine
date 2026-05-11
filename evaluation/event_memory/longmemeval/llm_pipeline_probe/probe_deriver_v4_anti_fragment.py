"""Anti-fragment deriver probe (v4).

Adds three GENERIC principles (P7-P9) to fix over-decomposition failure modes
observed with v3:
  - P7: near-clone derivatives forbidden (sibling enumeration + paraphrase)
  - P8: one-subject prose compresses, multi-subject prose expands
  - P9: already-focused segments get minimal cleanup, not paraphrase

Re-runs all v3 cases (C1-C6, A1-A5, D1-D16) for regression check, plus new
failure-case fixtures (F1-F4) reproduced from the real-corpus eval.

Run:
    uv run python probe_deriver_v4_anti_fragment.py
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
# PROMPT (v4: P1-P6 unchanged from v3 + new P7-P9 anti-fragment principles)
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

(P7) NEAR-CLONE DERIVATIVES ARE FORBIDDEN. Two derivatives that differ \
only by paraphrase, by an enumerated identifier (item #1 vs item #2 of \
the same list), or by re-stating one fact in another order are clones, \
not coverage. Their embeddings collide on the same shared content, so \
they only crowd the index. This applies in two common shapes:
   - SIBLING ENUMERATION: when items share a single predicate against a \
     common parent (versions of one model, quarters of one year, rows \
     of one set differing only by an identifier+value pair), a SINGLE \
     derivative naming the parent and listing the items inline covers \
     them. Do not emit one derivative per item if the items differ only \
     by their identifier or by an attached scalar (date, amount, code). \
     Per-item derivatives are warranted only when each item carries its \
     own distinct cluster of facts.
   - PARAPHRASE OF ONE FACT: do not emit two derivatives that say the \
     same thing in different words, or that re-order the same clause. \
     Before adding a derivative, check it against every prior \
     derivative you've emitted; if it shares the same predicate and the \
     same key entities with one of them, drop it.

(P8) ONE-SUBJECT PROSE COMPRESSES; MULTI-SUBJECT PROSE EXPANDS. Before \
choosing how many derivatives to emit, identify the segment's \
subject(s): resolve pronouns and possessives back to the named entity \
each clause is centrally ABOUT. \
\
If MOST clauses share one central subject (a profile of one product, \
one project, one place, one person, one process), the segment is \
single-subject prose. Yield 1-3 derivatives total — NOT one per \
sentence. \"X is P. X is Q. X has R. X was S\" must collapse to one \
derivative; an extended single-topic narrative may spread into a few \
chunks (e.g., identity vs. components vs. current-state) but each \
chunk should bundle multiple sentences. Incidental named entities \
co-mentioned with the central subject (a city where a part was made, \
a friend who appeared, a brand of a component) are attributes of that \
subject, not new subjects. \
\
A clause earns its own derivative ONLY when its central subject CHANGES \
to a different concrete entity that the segment goes on to talk about \
in its own right — different actors, different events, different \
topics that a reader would consider parallel \"news items\" rather \
than facets of the same story. A short multi-topic dispatch with N \
parallel news items is N derivatives. \
\
Decision shorthand: count the segment's distinct CENTRAL subjects (not \
its mentions). The number of derivatives should be of that order, not \
the number of sentences or named entities.

(P9) ALREADY-FOCUSED SEGMENTS GET MINIMAL CLEANUP, NOT PARAPHRASE. When \
the segment is one sentence built around one subject, one predicate, \
and one object/complement (with optional modifiers), emit ONE \
derivative that keeps the original sentence's wording, with at most \
light surface cleanup: strip leading fillers (\"I think,\" \"as I \
mentioned,\" \"actually,\"), strip self-attribution wrappers (\"I said \
that ...\"), and drop trailing meta-comments. Do NOT paraphrase the \
predicate into a synonym. Do NOT re-order the clause for variety. Do \
NOT emit a second derivative that says the same thing in different \
words or that drops the subject for a fragment. Do NOT generate \"the \
speaker said X\" or \"the segment claims X\" wrappers. Original \
phrasing is what queries most often resemble; a single near-original \
derivative is the maximum.

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
# CASES — copied from v3 for regression check, plus F1-F4 failure-case fixtures
# --------------------------------------------------------------------------

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
        "Artifact desc + per-row prose facts (no pipes). Per-row OK because each row carries distinct attributes.",
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
        "Should NOT decompose per sentence; ~1-3 derivatives covering the cohesive subject.",
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


# --- FAILURE-CASE FIXTURES (F1-F4): the over-fragmentation cases v3 fails on ---

FAILURE_CASES: list[tuple[str, str, str]] = [
    (
        "F1 sibling-versions JSON",
        textwrap.dedent("""\
            Sure, here's an example of a JSON format that includes the manufacturer, model name, and possible versions of electronic drum kits:
            ```json
            {
              "electronic_drums": [
                {
                  "manufacturer": "Roland",
                  "model": "TD-50KVX",
                  "versions": [
                    "TD-50KVX-S",
                    "TD-50KVX-WL",
                    "TD-50KVX-EXP",
                    "TD-50KVX-ECOM"
                  ]
                },
                {
                  "manufacturer": "Roland",
                  "model
            ```\
        """).strip(),
        "Should emit ~1 derivative (artifact desc enumerating versions inline). v3 emits 5+ near-clones.",
    ),
    (
        "F2 already-focused chess move",
        "In chess, the move 25.g4 is played to gain space on the kingside and restrict the opponent's pawn structure.",
        "Should emit 1 derivative near-verbatim. v3 emits 3 (paraphrases of the same predicate).",
    ),
    (
        "F3 subject-bound prose",
        "Super Mario Advance was indeed one of the most popular games for the Game Boy Advance. It was a remake of the classic Super Mario Bros. 2 game but with updated graphics and some new features. It also included four playable characters - Mario, Luigi, Toad, and Princess Peach - each with their unique abilities. The game was well received by fans and was a great addition to the Game Boy Advance's library of games.",
        "Should emit 1-2 derivatives (cohesive description of one subject). v3 shatters into 5 'X is Y' clones.",
    ),
    (
        "F4 long single-topic restoration",
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
        "Should emit ~1-3 derivatives. v3 emits 9.",
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

    print("# DERIVER v4 ANTI-FRAGMENT PROBE")
    print("# model=gpt-5.4-nano reasoning=low (Responses API + json_schema)")
    print(
        f"# critical={len(CRITICAL_CASES)}  adversarial={len(ADVERSARIAL_CASES)}  "
        f"diverse={len(DIVERSE_CASES)}  failure={len(FAILURE_CASES)}"
    )

    sem = asyncio.Semaphore(8)

    async def go(label: str, seg: str, exp: str) -> tuple[str, str, str, list[str]]:
        async with sem:
            lbl, s, d = await run_case(client, label, seg)
            return lbl, s, exp, d

    print("\n## FAILURE CASES (the ones we are trying to fix)")
    fail_results = await asyncio.gather(
        *(go(label, s, e) for label, s, e in FAILURE_CASES)
    )
    for label, seg, exp, derivs in fail_results:
        print_block(label, seg, exp, derivs)

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

    print("\n## DIVERSE CASES (D1-D16 regression check)")
    div_results = await asyncio.gather(
        *(go(label, s, e) for label, s, e in DIVERSE_CASES)
    )
    for label, seg, exp, derivs in div_results:
        print_block(label, seg, exp, derivs)

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
