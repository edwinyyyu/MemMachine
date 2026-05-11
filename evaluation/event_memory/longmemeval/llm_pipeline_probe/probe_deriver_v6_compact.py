"""Compact deriver probe (v6).

Goal: shorten v5's 9937-char prompt while preserving the load-bearing rules
validated by iters 1-8:
  - P1 verbatim tokens, P2 query-shaped prose
  - P3+P9 atomic facts vs already-focused (single near-original)
  - P4 non-prose surfaces -> decode + artifact-description
  - P5 acronym/translation bridge
  - P6 quoted speech verbatim
  - P7 no near-clones (sibling enum + paraphrase)
  - P8 count central subjects, not sentences
  - P10 scope travels when splitting

Consolidations vs v5:
  - P3 + P9 -> rule 4 (single 'atomic fact, focused = one near-original').
  - P8 moved up to rule 3 (subject-counting drives every other rule).
  - P10 reframed in two sentences with the same disambiguation test.
  - Removed the meta 'GUIDING QUESTION (INTERNAL ONLY)' paragraph (its
    'do not write Query:/Answer:' is folded into the header).
  - Removed trailing 'Do not duplicate / Do not invent' — implied by 1, 5.

Run:
    uv run python probe_deriver_v6_compact.py
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


# --------------------------------------------------------------------------
# PROMPT v6 — compact (target ~2.5k chars)
# --------------------------------------------------------------------------

PROMPT_DERIVER = """\
You produce DERIVATIVES of a SEGMENT for a semantic retrieval index. The \
SEGMENT is shown verbatim to the reader; your DERIVATIVES are what gets \
embedded. A derivative succeeds when its embedding shares vocabulary and \
meaning with the query a future user would write to find this segment.

Output a JSON list of derivative strings. Each is a plain statement, not a \
\"Query:/Answer:\" pair. Never invent content the segment lacks.

RULES.

1. VERBATIM TOKENS. Named entities, places, organizations, brands, products, \
works, dates, amounts, distinctive concept words, technical terms, and \
identifiers appear verbatim in any derivative that covers them.

2. QUERY-SHAPED PROSE. Each derivative reads like text a person could \
plausibly write — full grammar, no keyword fragments, no formatting \
artifacts (pipes, code syntax, log brackets, markup).

3. COUNT CENTRAL SUBJECTS, NOT SENTENCES OR ENTITIES. If most clauses share \
one central subject (a profile of one product, place, person, project, or \
process), yield 1–3 derivatives total — not one per sentence. Co-mentioned \
entities (a city where it was made, a friend who appeared, a brand of a \
component) are attributes of that subject, not new subjects. A clause earns \
its own derivative only when the central subject CHANGES to another entity \
the segment treats as a parallel news item.

4. ONE DERIVATIVE PER ATOMIC FACT. For genuinely independent facts, emit one \
self-contained derivative per fact. For a focused single-fact segment, emit \
ONE near-original derivative — strip leading fillers (\"I think,\", \
\"actually,\"), self-attribution wrappers (\"I said that...\"), and trailing \
meta-comments; do NOT paraphrase the predicate, re-order the clause, or add \
\"the speaker said X\" wrappers. For a bare entity or term, emit just that \
term.

5. NO NEAR-CLONES. Drop any derivative that shares the same predicate and \
key entities with another. Items differing only by an identifier+value \
(versions of one model, quarters of one year, rows of one set) collapse \
into ONE derivative naming the parent and listing the items inline; \
per-item derivatives are warranted only when each carries its own distinct \
cluster of facts.

6. SCOPE TRAVELS WHEN SPLITTING. When you split under rule 4, shared \
SCOPING CONTEXT — the trip, project, period, person, or artifact the facts \
are ABOUT — must be repeated in every sibling derivative whose fact lives \
under that scope. Test: would a natural query for THIS fact plausibly \
include this token to disambiguate it from similar facts in other scopes? \
If yes, carry it. Co-actors and per-clause attributes (a brand inside one \
clause, a person who appears in only one clause) do not propagate.

7. REPLACE NON-PROSE SURFACES. For encoded text, code, tables, lists of raw \
numbers, log lines, or markup: (a) decode recognizable encodings (cipher, \
base64) into a derivative; (b) always emit at least one ARTIFACT-DESCRIPTION \
derivative — one fluent sentence naming what the segment IS (e.g., \"Python \
function\", \"benchmark comparison table\", \"server error log line\") and \
what it is ABOUT, listing the entities or labels it covers; (c) for \
multi-row tables/lists, you may also emit one prose derivative per row when \
each row is itself an answerable fact; (d) render values in prose — never \
keep pipes, code syntax, or brackets.

8. BRIDGE FORM GAPS. When likely query tokens differ from segment tokens — \
heavy acronyms (JFK, POTUS, CMC), compressed jargon, or text in a script or \
language different from the one a likely query would use — emit a \
derivative with the expanded/translated form alongside one that keeps the \
original tokens. Skip this for plain English with full names already \
spelled out.

9. PRESERVE QUOTED SPEECH. If the segment shows text inside quotation marks \
with attribution, keep at least one derivative with the quoted text \
verbatim and its attribution.

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


# --------------------------------------------------------------------------
# EMBEDDING + COSINE
# --------------------------------------------------------------------------

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536


async def embed_many(client: openai.AsyncOpenAI, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    resp = await client.embeddings.create(
        model=EMBED_MODEL, input=texts, dimensions=EMBED_DIM
    )
    return [d.embedding for d in resp.data]


def cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# --------------------------------------------------------------------------
# CASES — mirror v5's probe so results are comparable
# --------------------------------------------------------------------------

CRITICAL_CASES: list[tuple[str, str, str]] = [
    (
        "C1 already-focused",
        "I went to Tokyo last March with my wife Anne.",
        "1 derivative close to original; preserves Tokyo, Anne, March.",
    ),
    (
        "C2 overloaded (Tokyo trip)",
        (
            "Last March I went to Tokyo with my wife Anne, stayed at the "
            "Park Hyatt for 5 nights at $400/night, and had ramen at "
            "Ichiran in Shibuya."
        ),
        "~3 derivatives: Tokyo trip with Anne; Park Hyatt $400/5nt in Tokyo; ramen at Ichiran/Shibuya in Tokyo. Scope (Tokyo) SHOULD propagate.",
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

ANTI_FRAGMENT_CASES: list[tuple[str, str, str]] = [
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
        "~1 derivative (artifact desc enumerating versions inline).",
    ),
    (
        "F2 chess move",
        "In chess, the move 25.g4 is played to gain space on the kingside and restrict the opponent's pawn structure.",
        "1 derivative near-verbatim (already-focused).",
    ),
    (
        "F3 Mario prose",
        "Super Mario Advance was indeed one of the most popular games for the Game Boy Advance. It was a remake of the classic Super Mario Bros. 2 game but with updated graphics and some new features. It also included four playable characters - Mario, Luigi, Toad, and Princess Peach - each with their unique abilities. The game was well received by fans and was a great addition to the Game Boy Advance's library of games.",
        "1-2 derivatives (cohesive description of one subject).",
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
        "~1-3 derivatives (cohesive single-subject restoration story).",
    ),
]

CONTROL_CASES: list[tuple[str, str, str]] = [
    (
        "FOCUS-F1 chess move (already-focused)",
        "In chess, the move 25.g4 is played to gain space on the kingside and restrict the opponent's pawn structure.",
        "1 derivative; minimal cleanup.",
    ),
    (
        "FOCUS-F2 git commit (already-focused)",
        "Fix race in event_memory's encode_events; partition lock now acquired before deriver dispatch",
        "1 derivative near-verbatim.",
    ),
]

ABBREV_CASE: tuple[str, str, str] = (
    "A4 abbreviation-dense",
    "TIL JFK was POTUS during the CMC in '62; SAC went to DEFCON 2.",
    "Expand JFK/POTUS/CMC/SAC AND keep verbatim acronyms.",
)

O1_CASE: tuple[str, str] = (
    "O1 lodging-anchor",
    (
        "Last March I went to Tokyo with my wife Anne, stayed at the "
        "Park Hyatt for 5 nights at $400/night, and had ramen at "
        "Ichiran in Shibuya."
    ),
)
O1_QUERY = "where did I stay in Tokyo"


# --------------------------------------------------------------------------
# RUNNER
# --------------------------------------------------------------------------


def _short(s: str, n: int = 200) -> str:
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


async def run_case(client: openai.AsyncOpenAI, label: str, segment: str) -> list[str]:
    return await derive(client, segment)


async def main() -> None:
    print(
        f"# PROMPT LENGTH: {len(PROMPT_DERIVER)} chars / {len(PROMPT_DERIVER.split())} words"
    )
    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(8)

    async def go(label: str, seg: str, exp: str) -> tuple[str, str, str, list[str]]:
        async with sem:
            d = await run_case(client, label, seg)
            return label, seg, exp, d

    print("# DERIVER v6 COMPACT PROBE")
    print("# model=gpt-5.4-nano reasoning=medium (Responses API + json_schema)")

    print("\n## O1 TARGET CASE — 5 reps of Tokyo trip with lodging-anchor query")
    label, seg = O1_CASE
    print(f"  SEGMENT: {seg}")
    print(f"  QUERY:   {O1_QUERY}")

    rep_derivs = await asyncio.gather(*(run_case(client, label, seg) for _ in range(5)))

    flat: list[str] = [O1_QUERY, seg]
    rep_offsets: list[tuple[int, int]] = []
    for derivs in rep_derivs:
        start = len(flat)
        flat.extend(derivs)
        rep_offsets.append((start, len(flat)))
    embs = await embed_many(client, flat)
    q_emb = embs[0]
    seg_emb = embs[1]
    seg_cos = cos(q_emb, seg_emb)
    print(f"\n  cos(query, segment_verbatim) = {seg_cos:.4f}")

    rep_summaries: list[tuple[int, list[float], int, float, str]] = []
    for rep_idx, (derivs, (a, b)) in enumerate(
        zip(rep_derivs, rep_offsets, strict=False)
    ):
        print(f"\n  REP {rep_idx + 1}: {len(derivs)} derivatives")
        cos_scores = [cos(q_emb, embs[i]) for i in range(a, b)]
        for i, (d, c) in enumerate(zip(derivs, cos_scores, strict=False)):
            mark = " <-- best" if cos_scores and c == max(cos_scores) else ""
            print(f"    [{i}] cos={c:.4f}  {d}{mark}")
        if cos_scores:
            best_i = max(range(len(cos_scores)), key=lambda j: cos_scores[j])
            best_c = cos_scores[best_i]
        else:
            best_i, best_c = -1, 0.0
        delta = best_c - seg_cos
        verdict = "WIN " if best_c > seg_cos else "loss"
        print(
            f"    -> best cos={best_c:.4f}  delta vs verbatim={delta:+.4f}  [{verdict}]"
        )
        rep_summaries.append((rep_idx + 1, cos_scores, best_i, best_c, verdict))

    wins = sum(1 for r in rep_summaries if r[4].startswith("WIN"))
    print(
        f"\n  ## O1 SUMMARY: {wins}/5 reps where best derivative > segment verbatim "
        f"(seg cos={seg_cos:.4f})"
    )

    print("\n## CRITICAL CASES (regression check)")
    crit_results = await asyncio.gather(
        *(go(label, s, e) for label, s, e in CRITICAL_CASES)
    )
    for label, seg, exp, derivs in crit_results:
        print_block(label, seg, exp, derivs)

    print("\n## ANTI-FRAGMENT CASES (must hold)")
    af_results = await asyncio.gather(
        *(go(label, s, e) for label, s, e in ANTI_FRAGMENT_CASES)
    )
    for label, seg, exp, derivs in af_results:
        print_block(label, seg, exp, derivs)

    print("\n## CONTROL CASES (already-focused, must NOT bloat)")
    ctrl_results = await asyncio.gather(
        *(go(label, s, e) for label, s, e in CONTROL_CASES)
    )
    for label, seg, exp, derivs in ctrl_results:
        print_block(label, seg, exp, derivs)

    print("\n## ABBREVIATION CASE (must still expand)")
    a4_label, a4_seg, a4_exp = ABBREV_CASE
    a4_derivs = await run_case(client, a4_label, a4_seg)
    print_block(a4_label, a4_seg, a4_exp, a4_derivs)

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
