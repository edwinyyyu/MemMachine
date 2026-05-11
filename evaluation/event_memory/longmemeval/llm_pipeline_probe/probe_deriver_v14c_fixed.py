"""v14c — goal-first + compression-prior with 3 surgical fixes from v14b.

v14b regressions diagnosed:
  - C1 = 0 derivs: logically correct under goal-first (segment already serves
    query, segment is in index) but recall@K validation shows near-original
    deriv adds lift. Fix: explicit "at least one derivative per segment".
  - C3 cipher = 1 (just decoded): "ALWAYS emit description" not strong
    enough; treated as either-or with decode. Fix: explicit separation.
  - F1 drum-kit = 6: I accidentally dropped v14's "items differing only
    by id+value collapse into ONE" clause from v14b. Restore it.

Hypothesis still under test: goal-first framing is more model-robust than
rule-based wording, once fully specified.
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
A SEGMENT is stored verbatim in a retrieval system. Your DERIVATIVES are \
additional strings that get embedded for semantic search alongside it.

Goal: a future user searching for anything in this segment should find \
it via cosine similarity to at least one derivative. Generate the FEWEST \
derivatives that cover every searchable thing in the segment — every \
extra derivative slot wastes index space and dilutes ranking. Splitting \
a focused thought, paraphrasing the same fact, listing items already \
inline, or atomizing a single-topic narrative are FAILURES, not coverage. \
Emit at least one derivative per segment.

How many derivatives? Count the distinct searchable things from the \
user's perspective, NOT the sentences or named entities:

  - A passage about ONE topic — one product, place, person, project, \
process, event — even across many sentences, is ONE searchable thing. \
Output ONE derivative (occasionally 2-3 if the topic is rich and one \
sentence cannot hold it). Co-mentioned things (a city it visited, a \
friend who appeared, a brand of a component) are attributes of the \
topic, not new topics.
  - Several genuinely independent facts a future query could ask \
separately → one derivative per fact.
  - A single focused statement (one fact, one chess move, one \
definition, one commit message) → exactly ONE near-original derivative \
preserving the segment's wording. Do NOT split its clauses. Do NOT \
paraphrase.
  - A bare standalone entity (a single noun like \"Tokyo\") → exactly \
ONE derivative containing just that term.
  - A list of items sharing the same predicate (versions of one model, \
quarters of one year, rows of one set, members of one series) → ONE \
derivative naming the parent and listing the items inline. Per-item \
derivatives waste slots because all items share the same future query.

EACH DERIVATIVE:
  - Is a full grammatical sentence (except bare-entity derivatives).
  - Uses the segment's wording for identifying terms verbatim — names, \
places, dates, amounts, brands, distinctive concept words.
  - Is self-contained — readable as if it were the only memory in the \
index about its topic. If the segment names a scope only once (the \
trip, project, period, person, artifact the facts are ABOUT), repeat \
that scope in EVERY derivative whose fact lives under it. Per-clause \
attributes (a brand or person mentioned in only one clause) stay where \
they are.

NON-PROSE OR PARTLY-FORMATTED CONTENT (code, tables, encoded text, log \
lines, JSON, markup, lists of raw numbers): you MUST emit AT LEAST ONE \
DESCRIPTION derivative — a sentence saying what the segment IS (e.g., \
\"Python function returning max in a binary tree\", \"benchmark table \
comparing GPT-4, Claude, and Gemini on Task A and B\", \"Caesar cipher \
text\") and what it is ABOUT. In ADDITION:
  - Decode recognizable encodings (Caesar, base64) into prose, in a \
separate derivative.
  - For multi-row tables, emit per-row prose sentences when each row is \
independently searchable, in addition to the description.
  - Render values in prose. Never preserve pipes, code syntax, or \
brackets.

HEAVY ACRONYMS (JFK, POTUS, CMC) or text in a script/language a query \
won't use: ONE derivative includes both the original and the expanded/\
translated form. Skip for plain English with full names.

Never invent content the segment lacks.

Output: a JSON object {{ \"derivatives\": [...] }} and nothing else.

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
    client, segment, *, prompt=PROMPT_DERIVER, model="gpt-5.4-nano", reasoning="medium"
):
    kwargs: dict[str, Any] = {
        "model": model,
        "input": prompt.format(segment=segment),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "derivatives",
                "schema": DERIVATIVES_SCHEMA,
                "strict": True,
            }
        },
    }
    if reasoning:
        kwargs["reasoning"] = {"effort": reasoning}
    resp = await client.responses.create(**kwargs)
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
        "1 derivative.",
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
    ("C5 identity bare entity", "Tokyo", "Single derivative."),
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

    print("# DERIVER v14c GOAL-FIRST + COMPRESSION + FIXES")
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
