"""v8 deriver probe — v7 with explicit dedupe enforcement.

v7 trade-off vs v5: ~53% shorter prompt, but produces near-clone pairs on
focused segments (C1, F2, FOCUS-F2) and over-splits acronym dense case (A4).

Targeted fix in v8:
  (a) Rule 4: "EXACTLY ONE" + explicit "do NOT add a second derivative
      paraphrasing or shortening the first" for focused segments.
  (b) Rule 5: explicit final-check clause — "before finalizing the list,
      compare every pair; if both would match the same query, drop one."

No other changes. Goal: same length budget (~4.7k chars), no near-clones.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import textwrap
from typing import Any

import openai
from dotenv import load_dotenv

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


PROMPT_DERIVER = """\
You produce DERIVATIVES of a SEGMENT for a semantic retrieval index. The \
SEGMENT is shown verbatim to the reader; your DERIVATIVES are what gets \
embedded. A derivative succeeds when its embedding shares vocabulary and \
meaning with the query a future user would write to find this segment.

Output a JSON list of derivative strings. Each is a plain statement, not a \
\"Query:/Answer:\" pair. Never invent content the segment lacks.

RULES.

1. VERBATIM TOKENS. Named entities, places, organizations, brands, \
products, works, dates, amounts, distinctive concept words, technical \
terms, and identifiers appear verbatim in any derivative that covers \
them. First-person voice (\"I\", \"we\", \"my\", \"our\") is preserved \
as-is — do NOT rewrite to \"the speaker\", \"they\", \"the user\", or \
\"their\".

2. QUERY-SHAPED PROSE. Each derivative reads like a full grammatical \
sentence a person could plausibly write. NO keyword fragments, NO \
noun-phrase-only fragments, NO formatting artifacts (pipes, code syntax, \
log brackets, markup).

3. COUNT CENTRAL SUBJECTS, NOT SENTENCES OR ENTITIES. If most clauses \
share one central subject (a profile of one product, place, person, \
project, process, event, or change), the segment is SINGLE-SUBJECT — \
yield 1–3 derivatives total, NOT one per sentence. \"X is P. X is Q. X \
has R. X was S.\" MUST collapse to ONE derivative. An extended \
single-topic narrative may spread into 2–3 chunks (e.g., identity vs. \
components vs. current-state) but each chunk bundles multiple sentences. \
Co-mentioned entities (a city where it was made, a friend who appeared, \
a brand of a component, a place it was tested) are attributes of the \
central subject, not new subjects. A clause earns its own derivative \
ONLY when its central subject CHANGES to another entity the segment \
treats as a parallel news item.

4. ONE DERIVATIVE PER ATOMIC FACT. For genuinely independent facts that \
would each answer a different query, emit one self-contained derivative \
per fact. For a FOCUSED segment (one clause OR one topical scope — a \
commit message, a chess move, a definition, a single observation), emit \
EXACTLY ONE near-original derivative and STOP: strip leading fillers \
(\"I think,\", \"actually,\"), self-attribution wrappers (\"I said \
that...\"), trailing meta-comments; do NOT paraphrase the predicate, \
re-order the clause, add \"the speaker said X\" / \"the user said X\" \
wrappers, split the topic into multiple derivatives, OR add a second \
derivative that rephrases or shortens the first. For a bare standalone \
entity or term (a single noun like \"Tokyo\"), emit just that term \
unmodified.

5. NO NEAR-CLONES. Before finalizing the list, compare every pair of \
derivatives. If both would match the same query — same predicate and \
same key entities, even if word order or surface form differs — DROP \
one. Items differing only by an identifier+value (versions of one model, \
quarters of one year, rows of one set) collapse into ONE derivative \
naming the parent and listing the items inline; per-item derivatives \
are warranted only when each carries its own distinct cluster of facts. \
Original-form + expanded-form for acronyms or translations belong in ONE \
derivative that includes both, not two separate derivatives.

6. SCOPE TRAVELS WHEN SPLITTING. When rule 4 actually splits (multiple \
distinct atomic facts), shared SCOPING CONTEXT — the trip, project, \
period, person, or artifact the facts are ABOUT — must be repeated in \
every sibling derivative whose fact lives under that scope. Test: would \
a natural query for THIS fact plausibly include this token to \
disambiguate it from similar facts in other scopes? If yes, carry it. \
Co-actors and per-clause attributes (a brand inside one clause, a \
person who appears in only one clause) do not propagate. Rule 6 does \
NOT override rules 3 or 4: when those collapse to a small count, there \
is no sibling to propagate to.

7. REPLACE NON-PROSE SURFACES. For encoded text, code, tables, lists of \
raw numbers, log lines, or markup: (a) decode recognizable encodings \
(cipher, base64) into a derivative; (b) always emit at least one \
ARTIFACT-DESCRIPTION derivative — one fluent sentence naming what the \
segment IS (e.g., \"Python function\", \"benchmark comparison table\", \
\"server error log line\") and what it is ABOUT, listing the entities \
or labels it covers; (c) for multi-row tables/lists, you may also emit \
one prose derivative per row when each row is itself an answerable \
fact; (d) render values in prose — never keep pipes, code syntax, or \
brackets.

8. BRIDGE FORM GAPS. When likely query tokens differ from segment \
tokens — heavy acronyms (JFK, POTUS, CMC), compressed jargon, or text \
in a script or language different from the one a likely query would \
use — emit a derivative with the expanded/translated form alongside one \
that keeps the original tokens. Skip this for plain English with full \
names already spelled out.

9. PRESERVE QUOTED SPEECH. If the segment shows text inside quotation \
marks with attribution, keep at least one derivative with the quoted \
text verbatim and its attribution.

SEGMENT:
{segment}
"""


DERIVATIVES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["derivatives"],
    "properties": {"derivatives": {"type": "array", "items": {"type": "string"}}},
}


async def derive(
    client: openai.AsyncOpenAI,
    segment: str,
    *,
    prompt: str = PROMPT_DERIVER,
    model: str = "gpt-5.4-nano",
    reasoning: str = "medium",
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


EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536


async def embed_many(client, texts):
    if not texts:
        return []
    resp = await client.embeddings.create(
        model=EMBED_MODEL, input=texts, dimensions=EMBED_DIM
    )
    return [d.embedding for d in resp.data]


def cos(a, b):
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


CRITICAL_CASES = [
    (
        "C1 already-focused",
        "I went to Tokyo last March with my wife Anne.",
        "1 derivative; first-person preserved.",
    ),
    (
        "C2 overloaded (Tokyo trip)",
        (
            "Last March I went to Tokyo with my wife Anne, stayed at the "
            "Park Hyatt for 5 nights at $400/night, and had ramen at "
            "Ichiran in Shibuya."
        ),
        "~3 derivatives with Tokyo propagated.",
    ),
    (
        "C3 caesar cipher",
        "Khoor, zruog! Wklv lv d phvvdjh.",
        "Decoded + artifact desc.",
    ),
    (
        "C4 markdown table",
        textwrap.dedent("""\
        | Model | Task A | Task B |
        | --- | --- | --- |
        | GPT-4 | 0.85 | 0.78 |
        | Claude | 0.91 | 0.82 |
        | Gemini | 0.79 | 0.81 |
        """).strip(),
        "Artifact desc + per-row prose facts (no pipes).",
    ),
    ("C5 identity bare entity", "Tokyo", "Single derivative."),
    (
        "C6 code block",
        textwrap.dedent("""\
        def find_max(node):
            if node is None:
                return float('-inf')
            return max(node.value, find_max(node.left), find_max(node.right))
        """).strip(),
        "Artifact desc.",
    ),
]

ANTI_FRAGMENT_CASES = [
    (
        "F1 drum-kit JSON",
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
        "~1 derivative.",
    ),
    (
        "F2 chess move",
        "In chess, the move 25.g4 is played to gain space on the kingside and restrict the opponent's pawn structure.",
        "1 derivative near-verbatim.",
    ),
    (
        "F3 Mario prose",
        "Super Mario Advance was indeed one of the most popular games for the Game Boy Advance. It was a remake of the classic Super Mario Bros. 2 game but with updated graphics and some new features. It also included four playable characters - Mario, Luigi, Toad, and Princess Peach - each with their unique abilities. The game was well received by fans and was a great addition to the Game Boy Advance's library of games.",
        "1-2 derivatives.",
    ),
    (
        "D1 Mustang restoration",
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
        "~1-3 derivatives.",
    ),
]

CONTROL_CASES = [
    (
        "FOCUS-F1 chess move",
        "In chess, the move 25.g4 is played to gain space on the kingside and restrict the opponent's pawn structure.",
        "1 derivative.",
    ),
    (
        "FOCUS-F2 git commit",
        "Fix race in event_memory's encode_events; partition lock now acquired before deriver dispatch",
        "1 derivative.",
    ),
]

ABBREV_CASE = (
    "A4 abbreviation-dense",
    "TIL JFK was POTUS during the CMC in '62; SAC went to DEFCON 2.",
    "Both forms in one or two derivs.",
)

O1_CASE = (
    "O1 lodging-anchor",
    (
        "Last March I went to Tokyo with my wife Anne, stayed at the "
        "Park Hyatt for 5 nights at $400/night, and had ramen at "
        "Ichiran in Shibuya."
    ),
)
O1_QUERY = "where did I stay in Tokyo"


def _short(s, n=200):
    s = s.replace("\n", " \\n ")
    return s if len(s) <= n else s[: n - 1] + "…"


def print_block(label, segment, expected, derivs):
    print()
    print(f"=== {label} ===")
    print(f"  SEGMENT: {_short(segment, 220)}")
    print(f"  EXPECTED: {expected}")
    print(f"  N_DERIVATIVES: {len(derivs)}")
    for i, d in enumerate(derivs):
        print(f"    [{i}] {d}")


async def run_case(client, label, segment):
    return await derive(client, segment)


async def main():
    print(
        f"# PROMPT LENGTH: {len(PROMPT_DERIVER)} chars / {len(PROMPT_DERIVER.split())} words"
    )
    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(8)

    async def go(label, seg, exp):
        async with sem:
            d = await run_case(client, label, seg)
            return label, seg, exp, d

    print("# DERIVER v8 COMPACT-DEDUPE PROBE")
    print("# model=gpt-5.4-nano reasoning=medium")

    print("\n## O1 TARGET CASE")
    label, seg = O1_CASE
    print(f"  SEGMENT: {seg}")
    print(f"  QUERY:   {O1_QUERY}")
    rep_derivs = await asyncio.gather(*(run_case(client, label, seg) for _ in range(5)))

    flat = [O1_QUERY, seg]
    rep_offsets = []
    for derivs in rep_derivs:
        start = len(flat)
        flat.extend(derivs)
        rep_offsets.append((start, len(flat)))
    embs = await embed_many(client, flat)
    q_emb = embs[0]
    seg_cos = cos(q_emb, embs[1])
    print(f"\n  cos(query, segment_verbatim) = {seg_cos:.4f}")
    wins = 0
    for rep_idx, (derivs, (a, b)) in enumerate(
        zip(rep_derivs, rep_offsets, strict=False)
    ):
        print(f"\n  REP {rep_idx + 1}: {len(derivs)} derivatives")
        cos_scores = [cos(q_emb, embs[i]) for i in range(a, b)]
        for i, (d, c) in enumerate(zip(derivs, cos_scores, strict=False)):
            mark = " <-- best" if cos_scores and c == max(cos_scores) else ""
            print(f"    [{i}] cos={c:.4f}  {d}{mark}")
        best_c = max(cos_scores) if cos_scores else 0.0
        verdict = "WIN " if best_c > seg_cos else "loss"
        if verdict.startswith("WIN"):
            wins += 1
        print(
            f"    -> best cos={best_c:.4f}  delta={best_c - seg_cos:+.4f}  [{verdict}]"
        )
    print(f"\n  ## O1 SUMMARY: {wins}/5 reps")

    print("\n## CRITICAL CASES")
    for label, seg, exp, derivs in await asyncio.gather(
        *(go(label, s, e) for label, s, e in CRITICAL_CASES)
    ):
        print_block(label, seg, exp, derivs)

    print("\n## ANTI-FRAGMENT CASES")
    for label, seg, exp, derivs in await asyncio.gather(
        *(go(label, s, e) for label, s, e in ANTI_FRAGMENT_CASES)
    ):
        print_block(label, seg, exp, derivs)

    print("\n## CONTROL CASES")
    for label, seg, exp, derivs in await asyncio.gather(
        *(go(label, s, e) for label, s, e in CONTROL_CASES)
    ):
        print_block(label, seg, exp, derivs)

    print("\n## ABBREVIATION CASE")
    a4_label, a4_seg, a4_exp = ABBREV_CASE
    a4_derivs = await run_case(client, a4_label, a4_seg)
    print_block(a4_label, a4_seg, a4_exp, a4_derivs)

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
