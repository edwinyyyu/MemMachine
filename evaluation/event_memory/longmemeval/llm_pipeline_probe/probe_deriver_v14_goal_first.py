"""v14 deriver probe — goal-first framing, fresh seed.

User feedback: v10's compactness was reached by local trim/merge perturbations
that nudge gpt-5.4-nano specifically. The result is a set of rules that happen
to work, not a clearly-stated task any model would interpret the same way.

This is a fresh seed. Instead of numbered rules, state the optimization target
upfront and structure the prompt around the principles that follow from it
(coverage, granularity, self-containment, no redundancy). The model should be
able to derive correct behavior from the goal without memorizing rule wording.

Hypothesis: a goal-first prompt generalizes across models better than a
rule-set tuned to one model's quirks.

Run:
    uv run python probe_deriver_v14_goal_first.py
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

Goal: a future user's natural-language search query about anything in \
this segment should match either the segment itself or at least one \
derivative (top-K cosine similarity). Generate the set of derivatives \
that achieves this with the least redundancy.

Coverage. Each searchable thing in the segment is covered by at least \
one derivative or by the segment itself. Use the segment's own phrasing \
for entity names, places, dates, amounts, and distinctive terms a query \
would include — verbatim.

Granularity. How many derivatives? Ask: how many distinct things could a \
future query be about?
  - A passage about ONE topic across multiple sentences is ONE searchable \
thing → one derivative (or a few, each bundling several sentences).
  - Several genuinely independent facts → one derivative per fact.
  - A single focused statement (a fact, definition, commit message, chess \
move) → one derivative close to the original wording.
  - A bare standalone term → just that term.

Self-containment. Each derivative reads as if it were the only thing in \
the index about its topic. The scope it depends on (the trip, project, \
period, person, artifact a fact is ABOUT) is included in the derivative \
itself, not left implicit. Per-clause attributes (a brand or person \
mentioned in only one clause) stay where they are.

No redundancy. Two derivatives that would match the same query waste \
index slots — drop or merge such pairs. An acronym and its expansion \
belong in ONE derivative; items differing only by an id+value (model \
versions, table rows) belong in ONE derivative listing them inline.

Edge cases:
  - Non-prose surfaces (code, tables, encoded text, lists of raw numbers, \
log lines, markup): write a natural sentence describing what the segment \
IS and what it is ABOUT; decode recognizable encodings (cipher, base64) \
into prose. For multi-row tables, also emit per-row prose sentences when \
each row is independently searchable. Never keep pipes, code syntax, or \
brackets.
  - Heavy acronyms (JFK, POTUS, CMC) or text in a script/language a \
query won't use: include both forms in one derivative.

Each derivative is a full grammatical sentence. Never invent content the \
segment lacks.

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

    print("# DERIVER v14 GOAL-FIRST PROBE (fresh seed, principles not rules)")
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
