"""LLM-based query planner: produces a DNF expression of temporal
constraints.

A plan is a list of clauses (outer = OR), each clause is a list of leaves
(inner = AND), and each leaf is `(phrase, relation)` where relation is
one of `intersect / after / before / disjoint`. An optional `extremum`
selector captures `latest / earliest` intent.

Wraps a single LLM call (gpt-5-mini, JSON-schema strict) with a persistent
file cache keyed on (model, prompt_version, query, ref_time).

The match evaluator `evaluate_dnf_match(plan, doc_ivs, resolver)` consumes
the plan + a doc's TE intervals + a per-leaf anchor resolver and returns
a fractional score in [0, 1]: max-over-clauses, min-over-leaves.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from openai.types.responses import ResponseTextConfigParam
from openai.types.responses.response_format_text_json_schema_config_param import (
    ResponseFormatTextJSONSchemaConfigParam,
)

if not os.environ.get("OPENAI_API_KEY"):
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Constants & cache location
# ---------------------------------------------------------------------------
MODEL = "gpt-5-mini"
PER_CALL_TIMEOUT_S = 45.0
CONCURRENCY = 8
PROMPT_VERSION = "v4.3"

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache" / "planner"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "llm_plan_cache.json"

RELATION_VALUES = ("intersect", "after", "before", "disjoint")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
PLAN_PROMPT = """For each TEMPORAL EXPRESSION in this query, classify its relation
AND express the OVERALL boolean structure as DNF (disjunctive normal form).

Query: {query}
Reference time: {ref_time}

DNF SHAPE
=========
The output `expr` is a list of lists.
  - Outer list = OR (disjunction) of clauses
  - Each inner list = AND (conjunction) of leaf constraints
  - Each leaf = {{"phrase": ..., "relation": "intersect"|"after"|"before"|"disjoint"}}

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
    (a) Direct date phrases — copy VERBATIM.
        "in Q4 2023" -> {{"phrase": "Q4 2023", "relation": "intersect"}}
        "March 2024" -> {{"phrase": "March 2024", "relation": "intersect"}}

        Do NOT add information that isn't in the user's text. If the
        phrase has NO year and NO deictic cue ("last"/"next"/"this"/
        "previous"/"current"), keep it bare — do NOT splice in a year
        from `Reference time`. The user may genuinely mean any year.
            "in March"  -> {{"phrase": "March",  "relation": "intersect"}}
            "in summer" -> {{"phrase": "summer", "relation": "intersect"}}
            "Q1 retros" -> {{"phrase": "Q1",     "relation": "intersect"}}
        The extractor refuses bare period words (no calendar anchor),
        so the query falls through to semantic+rerank across all
        candidate years. Splicing in `Reference time`'s year would
        incorrectly mask out matches from other years.

    (b) Relative deictic phrases — resolve against `Reference time`.
        "last quarter", "two weeks ago", "yesterday", "back in college":
        KEEP AS-IS — the downstream extractor handles deictic resolution.
        Use relation "intersect".

    (c) Event-anchor + offset (you know the event's date) — RESOLVE IN-PLACE
        using world knowledge + arithmetic.
        "four days after Election Day 2020" -> "November 7, 2020", "intersect"
        "two months after the iPhone launched in 2007" -> "August 29, 2007", "intersect"
        "three weeks before the Berlin Wall fell" -> "October 19, 1989", "intersect"
        "the day Kennedy was shot" -> "November 22, 1963", "intersect"

        FUZZY QUANTITIES — when the user says "about", "around",
        "approximately", "a few" — WIDEN to coarser-precision interval.
        "about two months after iPhone 2007" -> "August or September 2007", "intersect"
        "around three weeks before Berlin Wall" -> "October 1989", "intersect"

    (d) Event-anchor (no offset, you know the date) — resolve.
        "the year iPhone launched" -> {{"phrase": "2007", "relation": "intersect"}}
        "in the month JFK was killed" -> {{"phrase": "November 1963", "relation": "intersect"}}

    (e) Anaphoric event reference WITH a relation cue (you DON'T know the
        date — refers to a corpus event) — emit as a leaf with the cue's
        relation. The pipeline resolves the phrase via corpus retrieval.
        "after the launch"          -> {{"phrase": "the launch", "relation": "after"}}
        "before the migration"       -> {{"phrase": "the migration", "relation": "before"}}
        "during the offsite"         -> {{"phrase": "the offsite", "relation": "intersect"}}
        "since the redesign shipped" -> {{"phrase": "the redesign", "relation": "after"}}

        EXCEPTION — purely topical event references (no relation cue) →
        SKIP. "What did Maya say about the launch?", "Who attended the
        offsite?" — no temporal scoping intent.

        TEMPORAL-LOOKING FRAMINGS THAT ARE NOT SCOPING (skip → expr=[]):
        These words look temporal but only NAME a topic or PROVENANCE; the
        user is asking ABOUT the event, not for content scoped relative to it.
        - "from": "notes from the offsite", "lessons from the launch" —
          provenance/topic, not scope. → expr=[]
        - "of" (when the head is "aftermath", "outcomes", "lessons",
          "story of", "review of", "recap of", "wake of"): topical. → expr=[]
          "aftermath of the launch", "lessons of the migration"
        - "look back at", "looking back at", "thinking back to":
          narrative framing, not scope. → expr=[]
        - "behind" / "story behind": topical. → expr=[]
        - "when did X happen?", "when was X?": user wants the DATE OF X;
          retrieving docs about X is the answer, not filtering BY X. → expr=[]
        - "how did X go?", "what was X like?": narrative, not scope. → expr=[]

        The test: does this phrasing NARROW the time window of the answer,
        or just NAME what the answer is about? If it only names the topic,
        emit expr=[].

    (f) GENERIC TIME VOCABULARY USED NON-DEICTICALLY — SKIP.
        "What happens during the day in a beehive?" → no leaf
        "How do I plan my morning routine?" → no leaf
        "When does spring usually start?" → no leaf
        "past and future verb tenses" → no leaf
        "the future of AI" → no leaf

        DO emit for deictic uses: "this morning", "last spring", "spring 2024".

RELATION ENUM
=============
  "intersect"     — the date phrase NAMES the time of interest (DEFAULT). Cues:
             "in", "during", "of", "from <date>", or no cue. Also for
             relative deictic phrases ("two weeks ago", "back in college")
             and resolved event-anchor + offset expressions.
  "after"  — strictly AFTER the resolved date. Cues: "after", "since",
             "post" — only when the user wants OPEN-ENDED search.
  "before" — strictly BEFORE the resolved date. Cues: "before", "until",
             "prior to".
  "disjoint" — matches OUTSIDE this date phrase. Cues: "not in", "outside",
             "excluding", "except".

When in doubt, use "intersect".

COMPOSITION RULE — relative phrase inside a window
=================================================
When a relative date phrase (a season, month, quarter without year) appears
WITHOUT an explicit year, resolve it against the year/period named by the
OTHER constraints in the SAME clause, NOT against `Reference time`.

  "in 2024 not in summer"
    -> [[{{"phrase": "2024", "relation": "intersect"}},
         {{"phrase": "summer 2024", "relation": "disjoint"}}]]

  "in 2024 excluding the spring semester"
    -> [[{{"phrase": "2024", "relation": "intersect"}},
         {{"phrase": "spring 2024", "relation": "disjoint"}}]]

  "in Q1 2023 outside of February"
    -> [[{{"phrase": "Q1 2023", "relation": "intersect"}},
         {{"phrase": "February 2023", "relation": "disjoint"}}]]

  Already-qualified phrases need no composition:
  "What I did since 2022 outside of Q1 2023"
    -> [[{{"phrase": "2022", "relation": "after"}},
         {{"phrase": "Q1 2023", "relation": "disjoint"}}]]

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
{{"expr":[[{{"phrase":"Q4 2023","relation":"intersect"}}]],"extremum":null}}

Query: "after 2020"
{{"expr":[[{{"phrase":"2020","relation":"after"}}]],"extremum":null}}

Query: "What did I work on in March?"
{{"expr":[[{{"phrase":"March","relation":"intersect"}}]],"extremum":null}}

Query: "What trips did I take in summer?"
{{"expr":[[{{"phrase":"summer","relation":"intersect"}}]],"extremum":null}}

Query: "What were my Q1 retros?"
{{"expr":[[{{"phrase":"Q1","relation":"intersect"}}]],"extremum":null}}

Query: "Four days after Election Day 2020, what state did AP call?"
{{"expr":[[{{"phrase":"November 7, 2020","relation":"intersect"}}]],"extremum":null}}

Query: "About two months after the iPhone launched in 2007, what price cut?"
{{"expr":[[{{"phrase":"August 29, 2007","relation":"intersect"}}]],"extremum":null}}

Query: "in 2024 not in summer"
{{"expr":[[{{"phrase":"2024","relation":"intersect"}},{{"phrase":"summer 2024","relation":"disjoint"}}]],"extremum":null}}

Query: "What did I do in Q3 2023 after the launch?"
{{"expr":[[{{"phrase":"Q3 2023","relation":"intersect"}},{{"phrase":"the launch","relation":"after"}}]],"extremum":null}}

Query: "Most recent change since the redesign shipped"
{{"expr":[[{{"phrase":"the redesign","relation":"after"}}]],"extremum":"latest"}}

Query: "in Q1 or Q4 of 2023"
{{"expr":[[{{"phrase":"Q1 2023","relation":"intersect"}}],[{{"phrase":"Q4 2023","relation":"intersect"}}]],"extremum":null}}

Query: "What movie was popular the year iPhone launched"
{{"expr":[[{{"phrase":"2007","relation":"intersect"}}]],"extremum":null}}

Query: "latest budget review in Q2 2024"
{{"expr":[[{{"phrase":"Q2 2024","relation":"intersect"}}]],"extremum":"latest"}}

Query: "what did I do recently"
{{"expr":[],"extremum":"latest"}}

Query: "Notes from the team retreat"
{{"expr":[],"extremum":null}}

Query: "Lessons from the v3 launch"
{{"expr":[],"extremum":null}}

Query: "Aftermath of the v3 launch"
{{"expr":[],"extremum":null}}

Query: "Look back at the regression"
{{"expr":[],"extremum":null}}

Query: "When did the v3 launch happen?"
{{"expr":[],"extremum":null}}

Query: "How did the migration go?"
{{"expr":[],"extremum":null}}

Query: "Recent migration plan"
{{"expr":[],"extremum":"latest"}}
"""


_PLAN_JSON_SCHEMA: dict[str, object] = {
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
                    "required": ["phrase", "relation"],
                    "properties": {
                        "phrase": {"type": "string"},
                        "relation": {
                            "type": "string",
                            "enum": list(RELATION_VALUES),
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


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Constraint:
    """A single leaf in a DNF plan."""

    phrase: str
    relation: str  # "intersect" | "after" | "before" | "disjoint"

    def to_dict(self) -> dict[str, Any]:
        return {"phrase": self.phrase, "relation": self.relation}


@dataclass
class QueryPlan:
    """DNF query plan: outer = OR over clauses, inner = AND over leaves.
    Empty `expr` = no temporal scope (recency-only or pure semantic)."""

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
    def from_obj(cls, d: dict[str, Any], raw: str = "") -> QueryPlan:
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


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------
def _clean_str(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() == "null":
        return None
    return s


def _clean_extremum(v: Any) -> str | None:
    s = _clean_str(v)
    if s is None:
        return None
    s_l = s.lower()
    if s_l in ("latest", "earliest"):
        return s_l
    return None


def _clean_constraint(v: Any) -> Constraint | None:
    if not isinstance(v, dict):
        return None
    phrase = _clean_str(v.get("phrase"))
    relation = _clean_str(v.get("relation"))
    if not phrase or not relation:
        return None
    relation_l = relation.lower()
    if relation_l not in RELATION_VALUES:
        return None
    return Constraint(phrase=phrase, relation=relation_l)


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------
def _cache_key(query: str, ref_time: str) -> str:
    h = hashlib.sha256()
    h.update(MODEL.encode())
    h.update(b"|")
    h.update(PROMPT_VERSION.encode())
    h.update(b"|")
    h.update(query.encode())
    h.update(b"|")
    h.update(ref_time.encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
class QueryPlanner:
    """LLM planner that emits a DNF boolean expression of constraints.

    Cache lives at `<package>/cache/planner/llm_plan_cache.json`. Cache
    keys include the prompt version so prompt changes invalidate cleanly.

    `prompt_template` and `cache_subdir` allow A/B-testing prompt variants
    without invalidating the production cache.
    """

    def __init__(
        self,
        prompt_template: str | None = None,
        cache_subdir: str | None = None,
    ) -> None:
        self._client = AsyncOpenAI(timeout=PER_CALL_TIMEOUT_S)
        self._sem = asyncio.Semaphore(CONCURRENCY)
        self._calls = 0
        self._cache_hits = 0
        self._parse_failures = 0
        self._total = 0
        self._prompt_template = prompt_template or PLAN_PROMPT
        if cache_subdir is None:
            self._cache_file = CACHE_FILE
        else:
            cache_dir = ROOT / "cache" / cache_subdir
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_file = cache_dir / "llm_plan_cache.json"
        self._cache = self._load_cache()

    def _load_cache(self) -> dict:
        if not self._cache_file.exists():
            return {}
        try:
            return json.loads(self._cache_file.read_text())
        except Exception:
            return {}

    def _save_cache(self):
        # Multi-process safe: hold an exclusive lock on a sidecar file
        # across the read-merge-write so concurrent runners don't clobber
        # each other's plans.
        import fcntl
        with contextlib.suppress(Exception):
            lock_path = self._cache_file.with_suffix(self._cache_file.suffix + ".lock")
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(lock_path, "w") as lf:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
                try:
                    disk: dict = {}
                    if self._cache_file.exists():
                        try:
                            disk = json.loads(self._cache_file.read_text())
                        except Exception:
                            disk = {}
                    disk.update(self._cache)
                    self._cache = disk
                    tmp = self._cache_file.with_suffix(self._cache_file.suffix + ".tmp")
                    tmp.write_text(json.dumps(self._cache))
                    tmp.replace(self._cache_file)
                finally:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

    async def plan(self, query: str, ref_time: str) -> QueryPlan:
        self._total += 1
        key = _cache_key(query, ref_time)
        if key in self._cache:
            self._cache_hits += 1
            try:
                return QueryPlan.from_obj(
                    self._cache[key],
                    raw=json.dumps(self._cache[key]),
                )
            except Exception:
                pass

        prompt = self._prompt_template.format(query=query, ref_time=ref_time)
        format_config: ResponseFormatTextJSONSchemaConfigParam = {
            "type": "json_schema",
            "name": "query_plan",
            "strict": True,
            "schema": _PLAN_JSON_SCHEMA,
        }
        text_config: ResponseTextConfigParam = {"format": format_config}
        async with self._sem:
            try:
                resp = await self._client.responses.create(
                    model=MODEL,
                    input=prompt,
                    text=text_config,
                )
                self._calls += 1
                raw = resp.output_text
                obj = json.loads(raw)
                plan = QueryPlan.from_obj(obj, raw=raw)
                self._cache[key] = obj
                self._save_cache()
                return plan
            except Exception as e:
                self._parse_failures += 1
                return QueryPlan(parse_error=str(e), raw="")

    async def plan_many(self, items) -> dict[str, QueryPlan]:
        items = list(items)
        coros = [self.plan(q, rt) for _qid, q, rt in items]
        plans = await asyncio.gather(*coros)
        return {qid: plan for (qid, _, _), plan in zip(items, plans, strict=False)}

    def stats(self) -> dict:
        return {
            "model": MODEL,
            "prompt_version": PROMPT_VERSION,
            "total_queries": self._total,
            "calls": self._calls,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": self._cache_hits / max(1, self._total),
            "parse_failures": self._parse_failures,
        }


# ---------------------------------------------------------------------------
# DNF match evaluator
# ---------------------------------------------------------------------------
def evaluate_dnf_match(
    plan: QueryPlan,
    doc_ivs: list,
    leaf_anchor_resolver,  # callable: (clause_idx, leaf_idx, leaf) -> list[Interval]
) -> float:
    """Evaluate the DNF expression against a doc's intervals.

    Returns a temporal-match score in [0, 1] representing how well the
    doc's envelopes satisfy the planner's DNF constraint. 1.0 if the
    expression is empty (no temporal scope). For each clause (AND of
    leaves) compute the min over leaves; the final score is the max
    over clauses.

    `leaf_anchor_resolver` returns the anchor intervals for a leaf —
    the v3 extractor's resolution of leaf.phrase against ref_time.

    Leaf factors:
    - "intersect" / "after" / "before": binary 0/1 via
      `constraint_factor_for_doc` (does any doc interval satisfy the
      relation relative to the anchor).
    - "disjoint": `1 - excluded_containment`, where containment is the
      strict max-pair fraction of any doc interval inside the
      excluded window. The aggregate alternative was tested and lost.
    """
    from .core import constraint_factor_for_doc, excluded_containment

    if not plan.expr:
        return 1.0
    or_max = 0.0
    for ci, clause in enumerate(plan.expr):
        and_min = 1.0
        for li, leaf in enumerate(clause):
            anchor_ivs = leaf_anchor_resolver(ci, li, leaf)
            if not anchor_ivs:
                f = 1.0
            elif leaf.relation == "disjoint":
                cont = excluded_containment(doc_ivs, anchor_ivs)
                f = max(0.0, 1.0 - cont)
            else:
                f = constraint_factor_for_doc(doc_ivs, anchor_ivs, leaf.relation)
            if f < and_min:
                and_min = f
        if and_min > or_max:
            or_max = and_min
    return or_max
