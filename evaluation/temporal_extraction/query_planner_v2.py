"""Simplified LLM-based temporal query planner.

Schema is dead simple: a list of Constraints, where each is a date phrase
plus a direction enum (in / after / before / not_in), and an optional
extremum selector (latest / earliest).

    {
      "constraints": [
        {"phrase": "Q4 2023", "direction": "in"},
        {"phrase": "summer",  "direction": "not_in"}
      ],
      "extremum": "latest" | "earliest" | null
    }

The LLM picks ONE direction per phrase; mutual exclusion among the four
directions is automatic (no `op + open_lower + open_upper` mismatch the
way the v2.x prompt allowed).

Examples (all complete plans):
  "in Q4 2023"               -> [{Q4 2023, in}]
  "after 2020"               -> [{2020, after}]
  "before June 2010"         -> [{June 2010, before}]
  "in 2024 not in summer"    -> [{2024, in}, {summer, not_in}]
  "outside Q3 2023"          -> [{Q3 2023, not_in}]
  "between 2020 and 2024"    -> [{2020 to 2024, in}]
  "latest update in Q2 2024" -> [{Q2 2024, in}], extremum=latest
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

if not os.environ.get("OPENAI_API_KEY"):
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    except Exception:
        pass


MODEL = "gpt-5-mini"
PER_CALL_TIMEOUT_S = 45.0
CONCURRENCY = 8

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache" / "planner_v2"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "llm_plan_cache.json"


PLAN_PROMPT = """For each DATE PHRASE in this query, classify its direction.

Query: {query}
Reference time: {ref_time}

A constraint has:
  - phrase: the date/period text from the query, copied verbatim. Only
    emit constraints for CONCRETE date phrases ("Q4 2023", "March 2024",
    "summer 2024", "last week", "mid-October"). Do NOT emit constraints
    for event/entity names ("the launch", "the migration", "George
    Washington") — those don't have a calendar resolution.
  - direction: pick exactly one of:
      "in"     — the date phrase NAMES the time of interest (DEFAULT)
                 (cues: "in", "during", "of", "from <date>", or no cue)
                 ALSO use "in" for relative phrases that NAME a time:
                 "two weeks ago", "a few days ago", "last quarter",
                 "back in college". "ago"/"back" describe times, not
                 boundaries.
      "after"  — matches strictly AFTER this date
                 (cues: "after", "since", "post")
      "before" — matches strictly BEFORE this date
                 (cues: "before", "until", "prior to")
      "not_in" — matches OUTSIDE this date phrase
                 (cues: "not in", "outside", "excluding", "except")
    When in doubt, use "in" — only choose "after"/"before"/"not_in"
    when the query has the explicit cue word for that direction.

Also extract:
  - extremum: set ONLY when the query asks the system to PICK the
    most-recent / oldest from MULTIPLE candidates the user knows
    exist. Set "latest" or "earliest", else null.

      "Most recent meeting in March 2024"     -> "latest"
      "What's my latest budget review"        -> "latest"
      "What was my earliest job"              -> "earliest"
      "the first thing I did this morning"    -> "earliest"

    DO NOT set extremum when "first/last" describes a SPECIFIC event:
      "When did Marcus host his first dinner party?"   -> null
      "When did Aiden have his first child?"           -> null
      "When did Sara move into her first apartment?"   -> null
      "Who was the first US president?"                -> null
    "his/her/my first X" almost always names a specific event.

Examples:

Query: "in Q4 2023"
{{"constraints":[{{"phrase":"Q4 2023","direction":"in"}}],"extremum":null}}

Query: "after 2020"
{{"constraints":[{{"phrase":"2020","direction":"after"}}],"extremum":null}}

Query: "before June 2010"
{{"constraints":[{{"phrase":"June 2010","direction":"before"}}],"extremum":null}}

Query: "in 2024 not in summer"
{{"constraints":[{{"phrase":"2024","direction":"in"}},{{"phrase":"summer","direction":"not_in"}}],"extremum":null}}

Query: "outside Q3 2023"
{{"constraints":[{{"phrase":"Q3 2023","direction":"not_in"}}],"extremum":null}}

Query: "latest budget review in Q2 2024"
{{"constraints":[{{"phrase":"Q2 2024","direction":"in"}}],"extremum":"latest"}}

Query: "what did I do recently"
{{"constraints":[],"extremum":"latest"}}
"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
DIRECTION_VALUES = ("in", "after", "before", "not_in")

# JSON schema enforced via Responses API `text.format`. Lets the prompt
# drop format-discipline prose ("Return ONLY JSON, no commentary, no
# fences") since the API guarantees a parseable object matching this
# shape. `additionalProperties: false` and "required" lists everywhere
# are needed for `strict: true`.
_PLAN_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["constraints", "extremum"],
    "properties": {
        "constraints": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["phrase", "direction"],
                "properties": {
                    "phrase": {"type": "string"},
                    "direction": {"type": "string", "enum": list(DIRECTION_VALUES)},
                },
            },
        },
        "extremum": {"type": ["string", "null"], "enum": ["latest", "earliest", None]},
    },
}


@dataclass
class Constraint:
    phrase: str
    direction: str  # "in" | "after" | "before" | "not_in"

    def to_dict(self) -> dict[str, Any]:
        return {"phrase": self.phrase, "direction": self.direction}


@dataclass
class QueryPlan:
    constraints: list[Constraint] = field(default_factory=list)
    extremum: str | None = None  # "latest" | "earliest" | None
    raw: str | None = field(default=None, repr=False)
    parse_error: str | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraints": [c.to_dict() for c in self.constraints],
            "extremum": self.extremum,
        }

    @classmethod
    def from_obj(cls, d: dict[str, Any], raw: str = "") -> QueryPlan:
        constraints = []
        for c in d.get("constraints") or []:
            cc = _clean_constraint(c)
            if cc is not None:
                constraints.append(cc)
        return cls(
            constraints=constraints,
            extremum=_clean_extremum(d.get("extremum")),
            raw=raw,
        )

    @property
    def includes(self) -> list[Constraint]:
        return [c for c in self.constraints if c.direction in ("in", "after", "before")]

    @property
    def excludes(self) -> list[Constraint]:
        return [c for c in self.constraints if c.direction == "not_in"]

    @property
    def has_open_constraint(self) -> bool:
        return any(c.direction in ("after", "before") for c in self.constraints)

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


def _clean_constraint(v: Any) -> Constraint | None:
    if not isinstance(v, dict):
        return None
    phrase = _clean_str(v.get("phrase"))
    direction = _clean_str(v.get("direction"))
    if not phrase or not direction:
        return None
    direction_l = direction.lower()
    if direction_l not in DIRECTION_VALUES:
        return None
    return Constraint(phrase=phrase, direction=direction_l)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
_cache: dict[str, str] = {}
_cache_loaded = False
_cache_lock = asyncio.Lock()


def _load_cache() -> dict[str, str]:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save_cache_sync(cache: dict[str, str]) -> None:
    tmp = CACHE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cache, indent=0))
    tmp.replace(CACHE_FILE)


def _cache_key(query: str, ref_time: str) -> str:
    body = f"{MODEL}|v3.2|{query}|{ref_time}"
    return hashlib.sha256(body.encode()).hexdigest()


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _extract_json_obj(s: str) -> dict[str, Any] | None:
    if not s:
        return None
    cleaned = _FENCE_RE.sub("", s).strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
class QueryPlanner:
    def __init__(self, model: str = MODEL) -> None:
        self.model = model
        self.client = AsyncOpenAI(timeout=PER_CALL_TIMEOUT_S, max_retries=1)
        self.sem = asyncio.Semaphore(CONCURRENCY)
        self.calls = 0
        self.cache_hits = 0
        self.parse_failures = 0
        global _cache, _cache_loaded
        if not _cache_loaded:
            _cache = _load_cache()
            _cache_loaded = True

    async def _llm(self, prompt: str) -> str:
        async with self.sem:
            try:
                resp = await asyncio.wait_for(
                    self.client.responses.create(
                        model=self.model,
                        input=prompt,
                        max_output_tokens=3072,
                        text={
                            "format": {
                                "type": "json_schema",
                                "name": "QueryPlan",
                                "strict": True,
                                "schema": _PLAN_JSON_SCHEMA,
                            }
                        },
                    ),
                    timeout=PER_CALL_TIMEOUT_S,
                )
            except Exception:
                return ""
            return resp.output_text or ""

    async def plan(self, query: str, ref_time: str) -> QueryPlan:
        prompt = PLAN_PROMPT.format(query=query, ref_time=ref_time)
        k = _cache_key(query, ref_time)
        if k in _cache:
            self.cache_hits += 1
            raw = _cache[k]
        else:
            self.calls += 1
            raw = await self._llm(prompt)
            if raw:
                async with _cache_lock:
                    _cache[k] = raw
                    _save_cache_sync(_cache)
        obj = _extract_json_obj(raw)
        if obj is None:
            self.parse_failures += 1
            return QueryPlan(raw=raw, parse_error="json_parse_failed")
        try:
            return QueryPlan.from_obj(obj, raw=raw)
        except Exception as e:
            self.parse_failures += 1
            return QueryPlan(raw=raw, parse_error=str(e))

    async def plan_many(
        self, items: list[tuple[str, str, str]]
    ) -> dict[str, QueryPlan]:
        async def _one(qid: str, qtext: str, rt: str) -> tuple[str, QueryPlan]:
            p = await self.plan(qtext, rt)
            return qid, p

        tasks = [_one(qid, qt, rt) for qid, qt, rt in items]
        results = await asyncio.gather(*tasks)
        return dict(results)

    def stats(self) -> dict[str, Any]:
        total = self.calls + self.cache_hits
        return {
            "model": self.model,
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "parse_failures": self.parse_failures,
            "total_queries": total,
            "cache_hit_rate": self.cache_hits / max(1, total),
        }


async def _smoke():
    test_queries = [
        ("My latest budget review in Q2 2024", "2025-06-15T00:00:00Z"),
        (
            "What client meetings did I have in 2024 not in summer?",
            "2025-06-15T00:00:00Z",
        ),
        ("What did I do after 2020 but not in 2023?", "2025-06-15T00:00:00Z"),
        ("What did I do in Q3 2023 after the launch?", "2025-06-15T00:00:00Z"),
        ("What feature did the team ship around mid-October?", "2025-01-15T00:00:00Z"),
        ("What did I do most recently?", "2025-06-15T00:00:00Z"),
        (
            "When did Marcus host his first dinner party back when I worked at Acme?",
            "2025-04-01T00:00:00Z",
        ),
    ]
    p = QueryPlanner()
    for q, rt in test_queries:
        plan = await p.plan(q, rt)
        print(f"\nQ: {q}")
        print(f"  plan: {json.dumps(plan.to_dict(), indent=2)}")
    print("\nstats:", p.stats())


if __name__ == "__main__":
    asyncio.run(_smoke())
