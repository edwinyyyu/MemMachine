"""v11 deriver probe — same rules as v10, plain-English terms.

Hypothesis: SEGMENT/DERIVATIVE are codebase jargon. Replacing them with
everyday retrieval terms (passage, search phrasing) should perform as well
or better, since those terms are more common in the model's training data.

Changes vs v10:
  SEGMENT  -> PASSAGE
  DERIVATIVE -> SEARCH PHRASING (or just PHRASING after introduction)
  "semantic retrieval index" -> "semantic search index"
  JSON key: "derivatives" -> "phrasings"  (downstream code can map both)

Everything else identical to v10.
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
Given a PASSAGE, write the SEARCH PHRASINGS that should match it in a \
semantic search index. The PASSAGE is shown verbatim to the reader; your \
PHRASINGS are the strings that get embedded and matched against future \
search queries. A phrasing succeeds when it shares vocabulary and meaning \
with the query a future user would write to find this passage.

Each phrasing is a full grammatical sentence a person could plausibly \
write — no telegraph fragments, no formatting artifacts.

Rules:
  1. VERBATIM TOKENS. Named entities, places, brands, products, works, \
dates, amounts, distinctive concept words, technical terms, identifiers, \
and quoted speech (with attribution) appear verbatim in any phrasing \
that covers them. Never invent content the passage lacks.
  2. COUNT CENTRAL SUBJECTS, NOT SENTENCES. If most clauses share one \
central subject (one product, place, person, project, process, event), \
yield 1–3 phrasings total — not one per sentence. \"X is P. X is Q. X \
has R.\" collapses to ONE. Co-mentioned entities (a city, a friend, a \
component brand) are attributes of that subject, not new subjects.
  3. ATOMIC FACTS GET ONE PHRASING EACH. For genuinely independent facts \
(different central subjects), emit one self-contained phrasing per \
fact. For a FOCUSED passage (one clause or one topical scope — a commit \
message, a chess move, a definition), emit EXACTLY ONE near-original \
phrasing and STOP. Do not paraphrase the predicate, re-order, add \"the \
speaker said X\" wrappers, or split the topic. For a bare standalone \
term, emit just that term.
  4. NO NEAR-CLONES. Before finalizing, compare every pair: if both \
would match the same query, drop one. Items differing only by an \
id+value (versions of one model, rows of one set) collapse to ONE \
phrasing naming the parent and listing items inline. Original-form + \
expanded-form (acronyms, translations) belong in ONE phrasing, not two.
  5. CARRY SCOPE WHEN SPLITTING. If rule 3 splits into multiple \
phrasings, repeat shared scoping context (the trip, project, period, \
person, artifact the facts are ABOUT) in every sibling whose fact lives \
under that scope. Co-actors and per-clause attributes do not propagate.
  6. NON-PROSE GETS REPLACED WITH PROSE. For encoded text, code, tables, \
lists of raw numbers, log lines, or markup: decode recognizable \
encodings (cipher, base64); ALWAYS emit at least one DESCRIPTION \
phrasing — a full sentence naming what the passage IS (e.g., \"Python \
function returning max value in a binary tree\", \"benchmark table \
comparing GPT-4, Claude, and Gemini on Task A and Task B\") and what it \
is ABOUT; for multi-row tables, optionally emit one prose phrasing per \
row when each is independently answerable. NEVER keep pipes, code \
syntax, or brackets — render values in prose.
  7. BRIDGE FORM GAPS. When likely query tokens differ from the passage \
tokens (heavy acronyms like JFK/POTUS/CMC, or text in a different \
script/language than queries will use), emit a phrasing including the \
expanded/translated form alongside the original. Skip for plain English \
with full names.

Output: a JSON object {{ \"phrasings\": [...] }} and nothing else.

PASSAGE:
{segment}
"""


PHRASINGS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["phrasings"],
    "properties": {"phrasings": {"type": "array", "items": {"type": "string"}}},
}


async def derive(
    client, segment, *, prompt=PROMPT_DERIVER, model="gpt-5.4-nano", reasoning="medium"
):
    resp = await client.responses.create(
        model=model,
        input=prompt.format(segment=segment),
        reasoning={"effort": reasoning},
        text={
            "format": {
                "type": "json_schema",
                "name": "phrasings",
                "schema": PHRASINGS_SCHEMA,
                "strict": True,
            }
        },
    )
    payload = json.loads(resp.output_text)
    return list(payload.get("phrasings", []))


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
        "1 phrasing.",
    ),
    (
        "C2 overloaded (Tokyo trip)",
        (
            "Last March I went to Tokyo with my wife Anne, stayed at the "
            "Park Hyatt for 5 nights at $400/night, and had ramen at "
            "Ichiran in Shibuya."
        ),
        "~3 phrasings with Tokyo propagated.",
    ),
    ("C3 caesar cipher", "Khoor, zruog! Wklv lv d phvvdjh.", "Decoded + description."),
    (
        "C4 markdown table",
        textwrap.dedent("""\
        | Model | Task A | Task B |
        | --- | --- | --- |
        | GPT-4 | 0.85 | 0.78 |
        | Claude | 0.91 | 0.82 |
        | Gemini | 0.79 | 0.81 |
        """).strip(),
        "Description + per-row prose facts.",
    ),
    ("C5 identity bare entity", "Tokyo", "Single phrasing."),
    (
        "C6 code block",
        textwrap.dedent("""\
        def find_max(node):
            if node is None:
                return float('-inf')
            return max(node.value, find_max(node.left), find_max(node.right))
        """).strip(),
        "Description.",
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
        "~1 phrasing.",
    ),
    (
        "F2 chess move",
        "In chess, the move 25.g4 is played to gain space on the kingside and restrict the opponent's pawn structure.",
        "1 phrasing near-verbatim.",
    ),
    (
        "F3 Mario prose",
        "Super Mario Advance was indeed one of the most popular games for the Game Boy Advance. It was a remake of the classic Super Mario Bros. 2 game but with updated graphics and some new features. It also included four playable characters - Mario, Luigi, Toad, and Princess Peach - each with their unique abilities. The game was well received by fans and was a great addition to the Game Boy Advance's library of games.",
        "1-2 phrasings.",
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
        "~1-3 phrasings.",
    ),
]

CONTROL_CASES = [
    (
        "FOCUS-F1 chess move",
        "In chess, the move 25.g4 is played to gain space on the kingside and restrict the opponent's pawn structure.",
        "1 phrasing.",
    ),
    (
        "FOCUS-F2 git commit",
        "Fix race in event_memory's encode_events; partition lock now acquired before deriver dispatch",
        "1 phrasing.",
    ),
]

ABBREV_CASE = (
    "A4 abbreviation-dense",
    "TIL JFK was POTUS during the CMC in '62; SAC went to DEFCON 2.",
    "Both forms in one or two phrasings.",
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
    print(f"  PASSAGE: {_short(segment, 220)}")
    print(f"  EXPECTED: {expected}")
    print(f"  N_PHRASINGS: {len(derivs)}")
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

    print("# DERIVER v11 PLAIN-TERMS PROBE (passage/phrasing)")
    print("# model=gpt-5.4-nano reasoning=medium")

    print("\n## O1 TARGET CASE")
    label, seg = O1_CASE
    print(f"  PASSAGE: {seg}")
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
    print(f"\n  cos(query, passage_verbatim) = {seg_cos:.4f}")
    wins = 0
    for rep_idx, (derivs, (a, b)) in enumerate(
        zip(rep_derivs, rep_offsets, strict=False)
    ):
        print(f"\n  REP {rep_idx + 1}: {len(derivs)} phrasings")
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
