"""Anchor-propagation deriver probe (v5).

Adds ONE GENERIC principle (P10) on top of v4's P1-P9 to fix the anchor-drop
failure mode discovered by surgical eval (eval_retrieval_mechanism.py):

  Segment: "Last March I went to Tokyo with my wife Anne, stayed at the
            Park Hyatt for 5 nights at $400/night, and had ramen at
            Ichiran in Shibuya."
  Query:   "where did I stay in Tokyo"
  v4 atomization:
    - "I went to Tokyo last March with my wife Anne"
    - "I stayed at the Park Hyatt for 5 nights at $400/night"   <-- lost "Tokyo"
    - "I had ramen at Ichiran in Shibuya"
  -> The lodging derivative dropped the location anchor that scopes the fact.

P10 (anchor propagation): when a multi-fact segment is split by P3,
shared scoping context (the surrounding context that makes each fact
*about* something findable) must be carried into each child derivative.
This is distinct from P3's "no overlap on FACTS" — anchors are not facts;
they are the address under which a fact is stored. P10 also explicitly
does NOT push duplication on already-focused / single-subject prose
(those are governed by P8/P9 which still cap derivative count and
forbid bloat).

Run:
    uv run python probe_deriver_v5_anchor_propagation.py
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
# PROMPT (v5: P1-P9 unchanged from v4 + P10 anchor-propagation)
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

(P10) WHEN SPLITTING, CARRY THE SHARED SCOPING CONTEXT INTO EACH CHILD. \
P3 forbids repeating the same FACT across derivatives, but it does NOT \
forbid repeating the shared SCOPING CONTEXT under which each fact lives \
— the context the parent segment used (perhaps just once, at the top) \
to tell the reader what these facts are ABOUT. A scoping context is \
whatever a query would naturally need to specify in order to ask about \
this kind of fact at all: the trip the events happened on, the \
project the decisions belong to, the person the details describe, the \
period the items occurred in, the artifact the parts compose. When you \
split a segment under P3 and a piece of scoping context appears in the \
parent but only in ONE clause's surface, you must carry that context \
into each sibling derivative whose fact lives under that same scope. \
Otherwise the sibling becomes an orphan: structurally a fact, but no \
longer findable from a query that names the scope. \
\
This is NOT a license to copy every named entity into every \
derivative. The test is strict: would a natural query for THIS \
particular fact plausibly include this token to disambiguate it from \
similar facts in other scopes? If yes, carry it. If the token is just \
a co-mentioned attribute of a sibling (a brand inside one clause, a \
person who only appears in one clause), do not propagate it. Co-actors \
and incidental modifiers stay where they are; scope travels. \
\
P10 changes nothing for single-subject prose (P8) or already-focused \
segments (P9): those still collapse to a small number of derivatives, \
because there is no split happening and so nothing to propagate \
across. P10 only fires when P3 has produced multiple sibling \
derivatives and there is a scope they share that one of them would \
otherwise drop.

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
# CASES (subset of v4 used for the regression matrix)
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
        "~3 derivatives: Tokyo trip with Anne; Park Hyatt $400/5nt in Tokyo; ramen at Ichiran/Shibuya in Tokyo. P10: Tokyo SHOULD propagate.",
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

# F1 drum-kit, F2 chess, F3 Mario, D1 Mustang — iter-3a fixes that must hold.
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

# F1/F2 controls = already-focused, must NOT bloat.
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

# Abbreviation dense — must still expand JFK/POTUS/CMC/SAC.
ABBREV_CASE: tuple[str, str, str] = (
    "A4 abbreviation-dense",
    "TIL JFK was POTUS during the CMC in '62; SAC went to DEFCON 2.",
    "Expand JFK/POTUS/CMC/SAC AND keep verbatim acronyms.",
)

# Target: O1 — Tokyo trip, lodging derivative dropped Tokyo anchor under v4.
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
    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(8)

    async def go(label: str, seg: str, exp: str) -> tuple[str, str, str, list[str]]:
        async with sem:
            d = await run_case(client, label, seg)
            return label, seg, exp, d

    print("# DERIVER v5 ANCHOR-PROPAGATION PROBE")
    print("# model=gpt-5.4-nano reasoning=medium (Responses API + json_schema)")

    # ---------- O1: 5 reps of the Tokyo trip + retrieval litmus ----------
    print("\n## O1 TARGET CASE — 5 reps of Tokyo trip with lodging-anchor query")
    label, seg = O1_CASE
    print(f"  SEGMENT: {seg}")
    print(f"  QUERY:   {O1_QUERY}")

    rep_derivs = await asyncio.gather(*(run_case(client, label, seg) for _ in range(5)))

    # Embed query, full segment, and union of all derivatives.
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

    # ---------- Regression matrix ----------
    print("\n## CRITICAL CASES (regression check)")
    crit_results = await asyncio.gather(
        *(go(label, s, e) for label, s, e in CRITICAL_CASES)
    )
    for label, seg, exp, derivs in crit_results:
        print_block(label, seg, exp, derivs)

    print("\n## ANTI-FRAGMENT CASES (iter-3a fixes — must hold)")
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
