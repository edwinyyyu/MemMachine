"""Boolean-tree LLM query planner — must/should variant of planner_tree.

Identical structure and evaluator to `planner_tree`, but exposes nodes
named "must" / "should" instead of "and" / "or". The semantics are the
same (strict AND/OR with min/max), so this is purely a prompt-vocabulary
ablation: does swapping the operator labels to BM25-style terminology
make the LLM emit different (better/worse) trees?

Leaf semantics, relation enum, "not", and "disjoint" are unchanged.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

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


MODEL = "gpt-5-mini"
PER_CALL_TIMEOUT_S = 45.0
CONCURRENCY = 8
PROMPT_VERSION = "tree_mustshould_v1"

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache" / "planner_tree_mustshould"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "llm_plan_cache.json"

RELATION_VALUES = ("intersect", "after", "before", "disjoint")
NODE_TYPES = ("leaf", "must", "should", "not")


# ---------------------------------------------------------------------------
# Prompt — operators renamed from and/or to must/should.
# Same semantics as planner_tree.py: must = ALL children required (min);
# should = ANY child suffices (max). The vocabulary swap is the ablation.
# ---------------------------------------------------------------------------
TREE_PLAN_PROMPT = """For each TEMPORAL EXPRESSION in this query, classify its relation
AND express the OVERALL boolean structure as a RECURSIVE NODE TREE.

Query: {query}
Reference time: {ref_time}

TREE SHAPE
==========
The output `expr` is either null (no temporal scope) OR a recursive
NODE. Each node has one of four `type` values:

  - "leaf"   : a single constraint with `phrase` + `relation`. The
               extractor resolves `phrase` to a calendar window.
               Other fields ignored.
  - "must"   : ALL children required. `children` is a non-empty list
               of sub-nodes. Doc must satisfy every child.
  - "should" : ANY child suffices. `children` is a non-empty list of
               sub-nodes. Doc satisfies the node if it satisfies any
               child.
  - "not"    : negation. `child` is a single sub-node. Doc satisfies if
               it does NOT satisfy the child.

Every node MUST set every field, using null/empty defaults for fields
not used by its type:
  - "leaf"   : phrase=<str>, relation=<str>, children=[], child=null
  - "must"   : phrase=null, relation=null, children=[<node>, ...], child=null
  - "should" : phrase=null, relation=null, children=[<node>, ...], child=null
  - "not"    : phrase=null, relation=null, children=[], child=<node>

99% of queries fold to a single leaf. Use `must` / `should` ONLY when the
query has multiple constraints or an explicit "or" / "either ... or".
Use `not` only for genuinely nested negation; for simple "not in X",
prefer a leaf with relation="disjoint".

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
        "in Q4 2023" -> leaf("Q4 2023", "intersect")
        "March 2024" -> leaf("March 2024", "intersect")

        Do NOT add information that isn't in the user's text. If the
        phrase has NO year and NO deictic cue ("last"/"next"/"this"/
        "previous"/"current"), keep it bare — do NOT splice in a year
        from `Reference time`. The user may genuinely mean any year.
            "in March"  -> leaf("March",  "intersect")
            "in summer" -> leaf("summer", "intersect")
            "Q1 retros" -> leaf("Q1",     "intersect")
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
        "four days after Election Day 2020" -> leaf("November 7, 2020", "intersect")
        "two months after the iPhone launched in 2007" -> leaf("August 29, 2007", "intersect")
        "three weeks before the Berlin Wall fell" -> leaf("October 19, 1989", "intersect")
        "the day Kennedy was shot" -> leaf("November 22, 1963", "intersect")

        FUZZY QUANTITIES — when the user says "about", "around",
        "approximately", "a few" — WIDEN to coarser-precision interval.
        "about two months after iPhone 2007" -> leaf("August or September 2007", "intersect")
        "around three weeks before Berlin Wall" -> leaf("October 1989", "intersect")

    (d) Event-anchor (no offset, you know the date) — resolve.
        "the year iPhone launched" -> leaf("2007", "intersect")
        "in the month JFK was killed" -> leaf("November 1963", "intersect")

    (e) Anaphoric event reference WITH a relation cue (you DON'T know the
        date — refers to a corpus event) — emit as a leaf with the cue's
        relation. The pipeline resolves the phrase via corpus retrieval.
        "after the launch"          -> leaf("the launch", "after")
        "before the migration"       -> leaf("the migration", "before")
        "during the offsite"         -> leaf("the offsite", "intersect")
        "since the redesign shipped" -> leaf("the redesign", "after")

        EXCEPTION — purely topical event references (no relation cue) →
        SKIP. "What did Maya say about the launch?", "Who attended the
        offsite?" — no temporal scoping intent.

        TEMPORAL-LOOKING FRAMINGS THAT ARE NOT SCOPING (skip → expr=null):
        These words look temporal but only NAME a topic or PROVENANCE; the
        user is asking ABOUT the event, not for content scoped relative to it.
        - "from": "notes from the offsite", "lessons from the launch" —
          provenance/topic, not scope. → expr=null
        - "of" (when the head is "aftermath", "outcomes", "lessons",
          "story of", "review of", "recap of", "wake of"): topical. → expr=null
          "aftermath of the launch", "lessons of the migration"
        - "look back at", "looking back at", "thinking back to":
          narrative framing, not scope. → expr=null
        - "behind" / "story behind": topical. → expr=null
        - "when did X happen?", "when was X?": user wants the DATE OF X;
          retrieving docs about X is the answer, not filtering BY X. → expr=null
        - "how did X go?", "what was X like?": narrative, not scope. → expr=null

        The test: does this phrasing NARROW the time window of the answer,
        or just NAME what the answer is about? If it only names the topic,
        emit expr=null.

    (f) GENERIC TIME VOCABULARY USED NON-DEICTICALLY — SKIP.
        "What happens during the day in a beehive?" → no leaf
        "How do I plan my morning routine?" → no leaf
        "When does spring usually start?" → no leaf
        "past and future verb tenses" → no leaf
        "the future of AI" → no leaf

        DO emit for deictic uses: "this morning", "last spring", "spring 2024".

RELATION ENUM
=============
  "intersect" — the date phrase NAMES the time of interest (DEFAULT). Cues:
                "in", "during", "of", "from <date>", or no cue. Also for
                relative deictic phrases ("two weeks ago", "back in college")
                and resolved event-anchor + offset expressions.
  "after"     — strictly AFTER the resolved date. Cues: "after", "since",
                "post" — only when the user wants OPEN-ENDED search.
  "before"    — strictly BEFORE the resolved date. Cues: "before", "until",
                "prior to".
  "disjoint"  — matches OUTSIDE this date phrase. Cues: "not in", "outside",
                "excluding", "except".

When in doubt, use "intersect".

COMPOSITION RULE — relative phrase inside a window
====================================================
When a relative date phrase (a season, month, quarter without year) appears
WITHOUT an explicit year, resolve it against the year/period named by the
OTHER constraints in the SAME must-group, NOT against `Reference time`.

  "in 2024 not in summer"
    -> must(leaf("2024","intersect"), leaf("summer 2024","disjoint"))

  "in 2024 excluding the spring semester"
    -> must(leaf("2024","intersect"), leaf("spring 2024","disjoint"))

  "in Q1 2023 outside of February"
    -> must(leaf("Q1 2023","intersect"), leaf("February 2023","disjoint"))

  Already-qualified phrases need no composition:
  "What I did since 2022 outside of Q1 2023"
    -> must(leaf("2022","after"), leaf("Q1 2023","disjoint"))

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
{{"expr":{{"type":"leaf","phrase":"Q4 2023","relation":"intersect","children":[],"child":null}},"extremum":null}}

Query: "after 2020"
{{"expr":{{"type":"leaf","phrase":"2020","relation":"after","children":[],"child":null}},"extremum":null}}

Query: "What did I work on in March?"
{{"expr":{{"type":"leaf","phrase":"March","relation":"intersect","children":[],"child":null}},"extremum":null}}

Query: "in 2024 not in summer"
{{"expr":{{"type":"must","phrase":null,"relation":null,"children":[{{"type":"leaf","phrase":"2024","relation":"intersect","children":[],"child":null}},{{"type":"leaf","phrase":"summer 2024","relation":"disjoint","children":[],"child":null}}],"child":null}},"extremum":null}}

Query: "What did I do in Q3 2023 after the launch?"
{{"expr":{{"type":"must","phrase":null,"relation":null,"children":[{{"type":"leaf","phrase":"Q3 2023","relation":"intersect","children":[],"child":null}},{{"type":"leaf","phrase":"the launch","relation":"after","children":[],"child":null}}],"child":null}},"extremum":null}}

Query: "Most recent change since the redesign shipped"
{{"expr":{{"type":"leaf","phrase":"the redesign","relation":"after","children":[],"child":null}},"extremum":"latest"}}

Query: "in Q1 or Q4 of 2023"
{{"expr":{{"type":"should","phrase":null,"relation":null,"children":[{{"type":"leaf","phrase":"Q1 2023","relation":"intersect","children":[],"child":null}},{{"type":"leaf","phrase":"Q4 2023","relation":"intersect","children":[],"child":null}}],"child":null}},"extremum":null}}

Query: "Four days after Election Day 2020, what state did AP call?"
{{"expr":{{"type":"leaf","phrase":"November 7, 2020","relation":"intersect","children":[],"child":null}},"extremum":null}}

Query: "What movie was popular the year iPhone launched"
{{"expr":{{"type":"leaf","phrase":"2007","relation":"intersect","children":[],"child":null}},"extremum":null}}

Query: "latest budget review in Q2 2024"
{{"expr":{{"type":"leaf","phrase":"Q2 2024","relation":"intersect","children":[],"child":null}},"extremum":"latest"}}

Query: "what did I do recently"
{{"expr":null,"extremum":"latest"}}

Query: "Notes from the team retreat"
{{"expr":null,"extremum":null}}

Query: "When did the v3 launch happen?"
{{"expr":null,"extremum":null}}

Query: "Recent migration plan"
{{"expr":null,"extremum":"latest"}}
"""


_NODE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["type", "phrase", "relation", "children", "child"],
    "properties": {
        "type": {"type": "string", "enum": list(NODE_TYPES)},
        "phrase": {"type": ["string", "null"]},
        "relation": {
            "type": ["string", "null"],
            "enum": [*RELATION_VALUES, None],
        },
        "children": {
            "type": ["array", "null"],
            "items": {"$ref": "#/$defs/node"},
        },
        "child": {
            "anyOf": [
                {"$ref": "#/$defs/node"},
                {"type": "null"},
            ]
        },
    },
}

_PLAN_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["expr", "extremum"],
    "properties": {
        "expr": {
            "anyOf": [
                {"$ref": "#/$defs/node"},
                {"type": "null"},
            ],
            "description": "Recursive node tree; null = no temporal scope.",
        },
        "extremum": {
            "type": ["string", "null"],
            "enum": ["latest", "earliest", None],
        },
    },
    "$defs": {
        "node": _NODE_SCHEMA,
    },
}


@dataclass
class Leaf:
    phrase: str
    relation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "leaf",
            "phrase": self.phrase,
            "relation": self.relation,
            "children": [],
            "child": None,
        }


@dataclass
class Must:
    children: list["Node"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "must",
            "phrase": None,
            "relation": None,
            "children": [c.to_dict() for c in self.children],
            "child": None,
        }


@dataclass
class Should:
    children: list["Node"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "should",
            "phrase": None,
            "relation": None,
            "children": [c.to_dict() for c in self.children],
            "child": None,
        }


@dataclass
class Not:
    child: "Node"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "not",
            "phrase": None,
            "relation": None,
            "children": [],
            "child": self.child.to_dict(),
        }


Node = Union[Leaf, Must, Should, Not]


@dataclass
class TreeQueryPlan:
    expr: Node | None = None
    extremum: str | None = None
    raw: str | None = field(default=None, repr=False)
    parse_error: str | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "expr": self.expr.to_dict() if self.expr is not None else None,
            "extremum": self.extremum,
        }

    @classmethod
    def from_obj(cls, d: dict[str, Any], raw: str = "") -> "TreeQueryPlan":
        expr_raw = d.get("expr")
        try:
            expr = _parse_node(expr_raw) if expr_raw is not None else None
        except Exception as e:
            return cls(expr=None, extremum=_clean_extremum(d.get("extremum")),
                       raw=raw, parse_error=str(e))
        return cls(
            expr=expr,
            extremum=_clean_extremum(d.get("extremum")),
            raw=raw,
        )

    def iter_leaves(self):
        if self.expr is None:
            return
        yield from _iter_leaves(self.expr)

    @property
    def latest_intent(self) -> bool:
        return self.extremum == "latest"

    @property
    def earliest_intent(self) -> bool:
        return self.extremum == "earliest"


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


def _parse_node(d: Any) -> Node | None:
    if not isinstance(d, dict):
        return None
    t = _clean_str(d.get("type"))
    if t is None:
        return None
    t = t.lower()
    if t == "leaf":
        phrase = _clean_str(d.get("phrase"))
        relation = _clean_str(d.get("relation"))
        if not phrase or not relation:
            return None
        rl = relation.lower()
        if rl not in RELATION_VALUES:
            return None
        return Leaf(phrase=phrase, relation=rl)
    if t in ("must", "should"):
        kids_raw = d.get("children") or []
        kids: list[Node] = []
        for c in kids_raw:
            kn = _parse_node(c)
            if kn is not None:
                kids.append(kn)
        if not kids:
            return None
        if len(kids) == 1:
            return kids[0]
        return Must(children=kids) if t == "must" else Should(children=kids)
    if t == "not":
        cn = _parse_node(d.get("child"))
        if cn is None:
            return None
        return Not(child=cn)
    return None


def _iter_leaves(node: Node):
    if isinstance(node, Leaf):
        yield node
        return
    if isinstance(node, (Must, Should)):
        for c in node.children:
            yield from _iter_leaves(c)
        return
    if isinstance(node, Not):
        yield from _iter_leaves(node.child)


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


class TreePlannerMustShould:
    """LLM tree planner — must/should-labeled variant."""

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
        self._prompt_template = prompt_template or TREE_PLAN_PROMPT
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

    async def plan(self, query: str, ref_time: str) -> TreeQueryPlan:
        self._total += 1
        key = _cache_key(query, ref_time)
        if key in self._cache:
            self._cache_hits += 1
            try:
                return TreeQueryPlan.from_obj(
                    self._cache[key],
                    raw=json.dumps(self._cache[key]),
                )
            except Exception:
                pass

        prompt = self._prompt_template.format(query=query, ref_time=ref_time)
        format_config: ResponseFormatTextJSONSchemaConfigParam = {
            "type": "json_schema",
            "name": "query_plan_tree_mustshould",
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
                plan = TreeQueryPlan.from_obj(obj, raw=raw)
                self._cache[key] = obj
                self._save_cache()
                return plan
            except Exception as e:
                self._parse_failures += 1
                return TreeQueryPlan(parse_error=str(e), raw="")

    async def plan_many(self, items) -> dict[str, TreeQueryPlan]:
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


def evaluate_tree_match(
    plan: TreeQueryPlan,
    doc_ivs: list,
    leaf_anchor_resolver,
    notin_aggregate: bool = False,
    aggregator: str = "zadeh",
) -> float:
    """Same recursive shape as planner_tree.evaluate_tree_match.

    aggregator:
      "zadeh" -> must=min, should=max (strict fuzzy logic)
      "mean"  -> must=mean of leaf factors (partial credit / fusion),
                 should=max (selection, no bonus for extras)
    """
    from .core import (
        constraint_factor_for_doc,
        excluded_containment,
        excluded_containment_aggregate,
    )

    notin_fn = (
        excluded_containment_aggregate if notin_aggregate else excluded_containment
    )
    if aggregator not in ("zadeh", "mean"):
        raise ValueError(f"unknown aggregator: {aggregator!r}")

    def _eval(node: Node) -> float:
        if isinstance(node, Leaf):
            anchor_ivs = leaf_anchor_resolver(node)
            if not anchor_ivs:
                return 1.0
            if node.relation == "disjoint":
                cont = notin_fn(doc_ivs, anchor_ivs)
                return max(0.0, 1.0 - cont)
            return constraint_factor_for_doc(doc_ivs, anchor_ivs, node.relation)
        if isinstance(node, Must):
            if not node.children:
                return 1.0
            vs = [_eval(c) for c in node.children]
            if aggregator == "zadeh":
                return min(vs)
            return sum(vs) / len(vs)
        if isinstance(node, Should):
            if not node.children:
                return 0.0
            return max(_eval(c) for c in node.children)
        if isinstance(node, Not):
            return max(0.0, 1.0 - _eval(node.child))
        return 1.0

    if plan.expr is None:
        return 1.0
    return _eval(plan.expr)
