"""query_planner_v4 — DNF (disjunctive normal form) over temporal constraints.

Lets the LLM emit the FULL boolean structure of the query rather than relying
on a hardcoded aggregation (AND vs OR). Outer list = disjunction of clauses;
each inner clause = conjunction of leaf constraints.

Any propositional formula reduces to DNF, so this is expressively complete.

Examples:
  "in Q4 2023"
    expr = [[{phrase: "Q4 2023", direction: "in"}]]

  "in Q3 2023 after the launch"            (AND of two leaves)
    expr = [[{Q3 2023, in}, {the launch, after}]]

  "in Q1 OR Q4 of 2023"                    (OR of two singletons)
    expr = [[{Q1 2023, in}], [{Q4 2023, in}]]

  "in 2024 not in summer"                  (AND with negation leaf)
    expr = [[{2024, in}, {summer 2024, not_in}]]

  "what did I do recently"                 (no temporal scope; extremum only)
    expr = []
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from query_planner_v2 import (
    CONCURRENCY,
    DIRECTION_VALUES,
    MODEL,
    PER_CALL_TIMEOUT_S,
    AsyncOpenAI,
    Constraint,
    _clean_constraint,
    _clean_extremum,
)

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache" / "planner_v4"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "llm_plan_cache.json"


_PLAN_V4_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["expr", "extremum"],
    "properties": {
        "expr": {
            "type": "array",
            "description": "DNF: outer list = OR clauses, inner = AND of leaves",
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["phrase", "direction"],
                    "properties": {
                        "phrase": {"type": "string"},
                        "direction": {
                            "type": "string",
                            "enum": list(DIRECTION_VALUES),
                        },
                    },
                },
            },
        },
        "extremum": {
            "type": ["string", "null"],
            "enum": ["latest", "earliest", None],
        },
    },
}


# =============================================================================
# Prompt versions — DO NOT modify a prompt in place. Add a new version below
# and append to PROMPTS / bump DEFAULT_PROMPT_VERSION. The cache key includes
# the version, so every change is a fresh extraction; old versions can be
# rerun by passing `prompt_version=` to the planner.
# =============================================================================

PLAN_PROMPT_V4_0 = """For each TEMPORAL EXPRESSION in this query, classify its direction
AND express the OVERALL boolean structure as DNF (disjunctive normal form).

Query: {query}
Reference time: {ref_time}

DNF SHAPE
=========
The output `expr` is a list of lists.
  - Outer list = OR (disjunction) of clauses
  - Each inner list = AND (conjunction) of leaf constraints
  - Each leaf = {{"phrase": ..., "direction": "in"|"after"|"before"|"not_in"}}

99% of queries have ONE clause (no explicit "or"). Default: one inner list
containing all the leaves AND-ed together. Only use multiple outer clauses
when the query says "or" / "either ... or" / "in X or Y".

LEAF EXTRACTION
===============
A leaf has:
  - phrase: a CALENDAR-CONCRETE date/period text the extractor can resolve
    (e.g., "Q4 2023", "March 2024", "October 13 2020", "summer 2024"). OR
    an anaphoric event reference if you don't know its calendar date (e.g.,
    "the launch", "the migration"); the pipeline resolves these via corpus
    retrieval.

  Sources of phrases:
    (a) Direct date phrases — copy verbatim.
        "in Q4 2023" -> {{"phrase": "Q4 2023", "direction": "in"}}
        "March 2024" -> {{"phrase": "March 2024", "direction": "in"}}

    (b) Relative deictic phrases — resolve against `Reference time`.
        "last quarter", "two weeks ago", "yesterday", "back in college":
        KEEP AS-IS — the downstream extractor handles deictic resolution.
        Use direction "in".

    (c) Event-anchor + offset (you know the event's date) — RESOLVE IN-PLACE
        using world knowledge + arithmetic.
        "four days after Election Day 2020" -> "November 7, 2020", "in"
        "two months after the iPhone launched in 2007" -> "August 29, 2007", "in"
        "three weeks before the Berlin Wall fell" -> "October 19, 1989", "in"
        "the day Kennedy was shot" -> "November 22, 1963", "in"

        FUZZY QUANTITIES — when the user says "about", "around",
        "approximately", "a few" — WIDEN to coarser-precision interval.
        "about two months after iPhone 2007" -> "August or September 2007", "in"
        "around three weeks before Berlin Wall" -> "October 1989", "in"

    (d) Event-anchor (no offset, you know the date) — resolve.
        "the year iPhone launched" -> {{"phrase": "2007", "direction": "in"}}
        "in the month JFK was killed" -> {{"phrase": "November 1963", "direction": "in"}}

    (e) Anaphoric event reference WITH a direction cue (you DON'T know the
        date — refers to a corpus event) — emit as a leaf with the cue's
        direction. The pipeline resolves the phrase via corpus retrieval.
        "after the launch"          -> {{"phrase": "the launch", "direction": "after"}}
        "before the migration"       -> {{"phrase": "the migration", "direction": "before"}}
        "during the offsite"         -> {{"phrase": "the offsite", "direction": "in"}}
        "since the redesign shipped" -> {{"phrase": "the redesign", "direction": "after"}}

        EXCEPTION — purely topical event references (no direction cue) →
        SKIP. "What did Maya say about the launch?", "Who attended the
        offsite?" — no temporal scoping intent.

    (f) GENERIC TIME VOCABULARY USED NON-DEICTICALLY — SKIP.
        "What happens during the day in a beehive?" → no leaf
        "How do I plan my morning routine?" → no leaf
        "When does spring usually start?" → no leaf
        "past and future verb tenses" → no leaf
        "the future of AI" → no leaf

        DO emit for deictic uses: "this morning", "last spring", "spring 2024".

DIRECTION ENUM
==============
  "in"     — the date phrase NAMES the time of interest (DEFAULT). Cues:
             "in", "during", "of", "from <date>", or no cue. Also for
             relative deictic phrases ("two weeks ago", "back in college")
             and resolved event-anchor + offset expressions.
  "after"  — strictly AFTER the resolved date. Cues: "after", "since",
             "post" — only when the user wants OPEN-ENDED search.
  "before" — strictly BEFORE the resolved date. Cues: "before", "until",
             "prior to".
  "not_in" — matches OUTSIDE this date phrase. Cues: "not in", "outside",
             "excluding", "except".

When in doubt, use "in".

COMPOSITION RULE — relative phrase inside a window
====================================================
When a relative date phrase (a season, month, quarter without year) appears
WITHOUT an explicit year, resolve it against the year/period named by the
OTHER constraints in the SAME clause, NOT against `Reference time`.

  "in 2024 not in summer"
    -> [[{{"phrase": "2024", "direction": "in"}},
         {{"phrase": "summer 2024", "direction": "not_in"}}]]

  "in 2024 excluding the spring semester"
    -> [[{{"phrase": "2024", "direction": "in"}},
         {{"phrase": "spring 2024", "direction": "not_in"}}]]

  "in Q1 2023 outside of February"
    -> [[{{"phrase": "Q1 2023", "direction": "in"}},
         {{"phrase": "February 2023", "direction": "not_in"}}]]

  Already-qualified phrases need no composition:
  "What I did since 2022 outside of Q1 2023"
    -> [[{{"phrase": "2022", "direction": "after"}},
         {{"phrase": "Q1 2023", "direction": "not_in"}}]]

EXTREMUM
========
extremum: set ONLY when the query asks the system to PICK the most-recent /
oldest from MULTIPLE candidates the user knows exist. "latest" or
"earliest", else null.

  "Most recent meeting in March 2024"     -> "latest"
  "What's my latest budget review"        -> "latest"
  "What was my earliest job"              -> "earliest"

DO NOT set extremum when "first/last" describes a SPECIFIC event:
  "When did Marcus host his first dinner party?"   -> null
  "When did Aiden have his first child?"           -> null

EXAMPLES
========

Query: "in Q4 2023"
{{"expr":[[{{"phrase":"Q4 2023","direction":"in"}}]],"extremum":null}}

Query: "after 2020"
{{"expr":[[{{"phrase":"2020","direction":"after"}}]],"extremum":null}}

Query: "Four days after Election Day 2020, what state did AP call?"
{{"expr":[[{{"phrase":"November 7, 2020","direction":"in"}}]],"extremum":null}}

Query: "About two months after the iPhone launched in 2007, what price cut?"
{{"expr":[[{{"phrase":"August 29, 2007","direction":"in"}}]],"extremum":null}}

Query: "in 2024 not in summer"
{{"expr":[[{{"phrase":"2024","direction":"in"}},{{"phrase":"summer 2024","direction":"not_in"}}]],"extremum":null}}

Query: "What did I do in Q3 2023 after the launch?"
{{"expr":[[{{"phrase":"Q3 2023","direction":"in"}},{{"phrase":"the launch","direction":"after"}}]],"extremum":null}}

Query: "Most recent change since the redesign shipped"
{{"expr":[[{{"phrase":"the redesign","direction":"after"}}]],"extremum":"latest"}}

Query: "in Q1 or Q4 of 2023"
{{"expr":[[{{"phrase":"Q1 2023","direction":"in"}}],[{{"phrase":"Q4 2023","direction":"in"}}]],"extremum":null}}

Query: "What movie was popular the year iPhone launched"
{{"expr":[[{{"phrase":"2007","direction":"in"}}]],"extremum":null}}

Query: "latest budget review in Q2 2024"
{{"expr":[[{{"phrase":"Q2 2024","direction":"in"}}]],"extremum":"latest"}}

Query: "what did I do recently"
{{"expr":[],"extremum":"latest"}}
"""


# =============================================================================
# v4.1 — tighten rule (e) to "the X" patterns only.
#
# v4.0 over-emitted personal-era phrases ("grad school", "my parental leave",
# "back when I worked at Acme") as constraints, which extracted to empty/weak
# intervals and tanked era_refs (0.333 → 0.083). v4.1 narrows rule (e) to
# anaphoric phrases that EXPLICITLY START WITH "the" — only those have a
# clear corpus-anchor doc to look up. Personal-era phrases without "the"
# fall through to SKIP, matching v3.1 behavior on era-style queries.
# =============================================================================
_RULE_E_V4_0 = """    (e) Anaphoric event reference WITH a direction cue (you DON'T know the
        date — refers to a corpus event) — emit as a leaf with the cue's
        direction. The pipeline resolves the phrase via corpus retrieval.
        "after the launch"          -> {{"phrase": "the launch", "direction": "after"}}
        "before the migration"       -> {{"phrase": "the migration", "direction": "before"}}
        "during the offsite"         -> {{"phrase": "the offsite", "direction": "in"}}
        "since the redesign shipped" -> {{"phrase": "the redesign", "direction": "after"}}
        "the day after the kickoff" -> {{"phrase": "the kickoff", "direction": "after"}}

        When emitting these, output the cleanest noun-phrase form of
        the event reference (drop articles like "the" only if they're
        weakly attached; keep them when removing would make the phrase
        ambiguous).

        EXCEPTION — event references WITHOUT a direction cue (purely
        topical, e.g. "What did Maya say about the launch?", "Who
        attended the offsite?") — SKIP. No temporal scoping intent.

        For events the LLM DOES know the date for (rule c, d above),
        prefer in-place resolution to a calendar phrase ("the year
        iPhone launched" -> "2007"). Only fall back to corpus_anchor
        emission when you don't know the event's date."""

_RULE_E_V4_1 = """    (e) Anaphoric CORPUS EVENT with a direction cue — emit ONLY when the
        phrase EXPLICITLY STARTS WITH "the" + a concrete noun-phrase
        naming a SINGULAR corpus event the LLM doesn't know the date of.
        The phrase MUST begin with "the". Direction is taken from the cue
        ("after", "before", "since", "during", "until").

        EMIT:
        "after the launch"          -> {{"phrase": "the launch", "direction": "after"}}
        "before the migration"       -> {{"phrase": "the migration", "direction": "before"}}
        "during the offsite"         -> {{"phrase": "the offsite", "direction": "in"}}
        "since the redesign shipped" -> {{"phrase": "the redesign", "direction": "after"}}
        "the day after the kickoff"  -> {{"phrase": "the kickoff", "direction": "after"}}
        "before the year-end review" -> {{"phrase": "the year-end review", "direction": "before"}}

        SKIP all other vague era / personal-history phrases — even if
        they have a direction cue. Personal eras don't have a clear
        corpus anchor doc, and emitting weak constraints hurts retrieval:
        "during grad school"            -> SKIP
        "back when I worked at Acme"    -> SKIP
        "while living in Sweden"        -> SKIP
        "during my parental leave"      -> SKIP
        "back in college"               -> SKIP
        "right after I graduated"       -> SKIP
        "during my fitness phase"       -> SKIP
        "while training for the Olympics" -> SKIP
        "during the pandemic year"      -> emit "2020" if known (rule c/d), else SKIP

        EXCEPTION — event references WITHOUT a direction cue (purely
        topical, e.g. "What did Maya say about the launch?") — SKIP.

        For events the LLM DOES know the date for (rule c, d above),
        prefer in-place resolution to a calendar phrase."""


PLAN_PROMPT_V4_1 = PLAN_PROMPT_V4_0.replace(_RULE_E_V4_0, _RULE_E_V4_1)
assert _RULE_E_V4_0 not in PLAN_PROMPT_V4_1, "v4.1 rule (e) substitution failed"


# Registry of prompt versions. Add new entries; never edit older ones.
PROMPTS: dict[str, str] = {
    "v4.0": PLAN_PROMPT_V4_0,
    "v4.1": PLAN_PROMPT_V4_1,
}

DEFAULT_PROMPT_VERSION = "v4.1"


@dataclass
class QueryPlanV4:
    """DNF query plan: outer = OR, inner = AND. Empty expr = no temporal scope."""

    expr: list[list[Constraint]] = field(default_factory=list)
    extremum: str | None = None
    raw: str | None = field(default=None, repr=False)
    parse_error: str | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "expr": [[c.to_dict() for c in clause] for clause in self.expr],
            "extremum": self.extremum,
        }

    @classmethod
    def from_obj(cls, d: dict[str, Any], raw: str = "") -> QueryPlanV4:
        expr_raw = d.get("expr") or []
        expr: list[list[Constraint]] = []
        for clause in expr_raw:
            if not isinstance(clause, list):
                continue
            leaves: list[Constraint] = []
            for c in clause:
                cc = _clean_constraint(c)
                if cc is not None:
                    leaves.append(cc)
            if leaves:
                expr.append(leaves)
        return cls(
            expr=expr,
            extremum=_clean_extremum(d.get("extremum")),
            raw=raw,
        )

    @property
    def all_leaves(self) -> list[Constraint]:
        out = []
        for clause in self.expr:
            out.extend(clause)
        return out

    @property
    def latest_intent(self) -> bool:
        return self.extremum == "latest"

    @property
    def earliest_intent(self) -> bool:
        return self.extremum == "earliest"


def _cache_key(query: str, ref_time: str, version: str) -> str:
    h = hashlib.sha256()
    h.update(MODEL.encode())
    h.update(b"|")
    h.update(version.encode())
    h.update(b"|")
    h.update(query.encode())
    h.update(b"|")
    h.update(ref_time.encode())
    return h.hexdigest()


class QueryPlannerV4:
    """LLM planner that emits a DNF boolean expression of constraints.

    The prompt version is selectable; older versions remain accessible by
    passing `prompt_version=` and stay registered in `PROMPTS`. Cache keys
    include the version so changes don't blow away history.
    """

    def __init__(self, prompt_version: str = DEFAULT_PROMPT_VERSION):
        if prompt_version not in PROMPTS:
            raise ValueError(
                f"unknown prompt_version {prompt_version!r}; "
                f"available: {sorted(PROMPTS)}"
            )
        self.prompt_version = prompt_version
        self.prompt_template = PROMPTS[prompt_version]
        self._client = AsyncOpenAI(timeout=PER_CALL_TIMEOUT_S)
        self._sem = asyncio.Semaphore(CONCURRENCY)
        self._calls = 0
        self._cache_hits = 0
        self._parse_failures = 0
        self._total = 0
        self._cache = self._load_cache()

    def _load_cache(self) -> dict:
        if not CACHE_FILE.exists():
            return {}
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            return {}

    def _save_cache(self):
        try:
            CACHE_FILE.write_text(json.dumps(self._cache))
        except Exception:
            pass

    async def _plan_one(self, qid: str, query: str, ref_time: str) -> QueryPlanV4:
        self._total += 1
        key = _cache_key(query, ref_time, self.prompt_version)
        if key in self._cache:
            self._cache_hits += 1
            try:
                return QueryPlanV4.from_obj(
                    self._cache[key],
                    raw=json.dumps(self._cache[key]),
                )
            except Exception:
                pass

        prompt = self.prompt_template.format(query=query, ref_time=ref_time)
        async with self._sem:
            try:
                resp = await self._client.responses.create(
                    model=MODEL,
                    input=prompt,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "query_plan_v4",
                            "strict": True,
                            "schema": _PLAN_V4_JSON_SCHEMA,
                        }
                    },
                )
                self._calls += 1
                raw = resp.output_text
                obj = json.loads(raw)
                plan = QueryPlanV4.from_obj(obj, raw=raw)
                self._cache[key] = obj
                self._save_cache()
                return plan
            except Exception as e:
                self._parse_failures += 1
                return QueryPlanV4(parse_error=str(e), raw="")

    async def plan_many(self, items) -> dict[str, QueryPlanV4]:
        items = list(items)
        coros = [self._plan_one(qid, q, rt) for qid, q, rt in items]
        plans = await asyncio.gather(*coros)
        return {qid: plan for (qid, _, _), plan in zip(items, plans)}

    def stats(self) -> dict:
        return {
            "model": MODEL,
            "prompt_version": self.prompt_version,
            "total_queries": self._total,
            "calls": self._calls,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": self._cache_hits / max(1, self._total),
            "parse_failures": self._parse_failures,
        }


def evaluate_dnf_mask(
    plan: QueryPlanV4,
    doc_ivs: list,
    leaf_anchor_resolver,  # callable: (clause_idx, leaf_idx, leaf) -> list[Interval]
) -> float:
    """Evaluate the DNF expression against a doc's intervals.

    Returns a score in [0, 1]. 1.0 if the expression is empty (no temporal
    scope). For each clause (AND of leaves) compute the min over leaves; the
    final score is the max over clauses.

    `leaf_anchor_resolver` returns the anchor intervals for a leaf — typically
    the calendar extraction of leaf.phrase, falling back to corpus-anchor
    retrieval when extraction is empty.

    For "not_in" leaves, we use 1 - excluded_containment(...) which is in
    [0, 1] (fractional containment of the doc's intervals inside the
    excluded anchor). For "in"/"after"/"before" leaves we use the binary
    constraint_factor_for_doc(...).
    """
    from composition_eval_v3 import constraint_factor_for_doc
    from negation import excluded_containment

    if not plan.expr:
        return 1.0
    or_max = 0.0
    for ci, clause in enumerate(plan.expr):
        and_min = 1.0
        for li, leaf in enumerate(clause):
            anchor_ivs = leaf_anchor_resolver(ci, li, leaf)
            if not anchor_ivs:
                # Leaf without resolvable intervals contributes 1.0
                # (treat as no-op rather than zeroing out the clause).
                f = 1.0
            elif leaf.direction == "not_in":
                cont = excluded_containment(doc_ivs, anchor_ivs)
                f = max(0.0, 1.0 - cont)
            else:
                f = constraint_factor_for_doc(doc_ivs, anchor_ivs, leaf.direction)
            if f < and_min:
                and_min = f
        if and_min > or_max:
            or_max = and_min
    return or_max
