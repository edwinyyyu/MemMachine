"""Variant A: no-anaphora planner with RECURRING PATTERNS → empty rule.

Hypothesis: the pure-subtract no-anaphora variant regressed on lattice
(-0.100) because the LLM, lacking the anaphora safety valve, falls back
to enumerating recurring patterns ("What do I do on Thursdays?") as
13 concrete dates that don't align with the doc-side extractor's
single nearest-past-occurrence anchor.

Fix: tell the LLM that recurring / habitual queries should emit empty
targets — semantic search will find the matching doc by content
without temporal help.

This is a targeted addition to the EMPTY OUTPUT section. No other
changes to the no-anaphora prompt or schema.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
from pathlib import Path

from openai import AsyncOpenAI
from openai.types.responses import ResponseTextConfigParam
from openai.types.responses.response_format_text_json_schema_config_param import (
    ResponseFormatTextJSONSchemaConfigParam,
)

from temporal_retrieval_tr.planner import (
    MODEL, PER_CALL_TIMEOUT_S, CONCURRENCY, Plan,
    _json_to_targets,
)
from temporal_retrieval_tr.research._no_anaphora_planner import (
    PROMPT as BASE_PROMPT,
    _PLAN_JSON_SCHEMA_NO_ANAPHORA,
)

if not os.environ.get("OPENAI_API_KEY"):
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    except Exception:
        pass


PROMPT_VERSION = "v4-recurring-empty"

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache" / "planner_recurring_empty"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "llm_plan_cache.json"


# Replace the EMPTY OUTPUT section with a version that includes
# recurring-pattern handling. Examples deliberately use different
# day names / patterns than lattice queries to avoid over-fitting:
# lattice uses Thursdays/Tuesdays/afternoon activities/3pm habits;
# here we use Wednesdays/morning workouts/regular check-ins.
_OLD_EMPTY_SECTION = """EMPTY OUTPUT
============
If the query has NO temporal scope at all (e.g., "how do I plan my
morning?", "lessons from the launch", "how did the migration go?") emit
{{"targets": []}}.

"What did I do recently" / "show me what happened lately" → deictic;
resolve to a recent window (e.g. last 60-90 days from REF_TIME) as a
target with extremum="latest"."""

_NEW_EMPTY_SECTION = """EMPTY OUTPUT
============
Emit empty targets — {{"targets": []}} — when the query has NO concrete
temporal scope. Two cases:

1. No temporal flavor at all:
   "how do I plan my morning?", "lessons from the launch",
   "how did the migration go?"

2. Recurring or habitual patterns:
   "what do I do on Wednesdays?", "my morning workouts",
   "regular weekly check-ins", "afternoon habits"
   The matching doc is anchored to a single representative occurrence;
   enumerating every possible occurrence (every Wednesday, every
   workout) would force a temporal match against the wrong dates.
   Let semantic search find the doc by content.

"What did I do recently" / "show me what happened lately" → deictic;
resolve to a recent window (e.g. last 60-90 days from REF_TIME) as a
target with extremum="latest"."""

assert _OLD_EMPTY_SECTION in BASE_PROMPT, (
    "EMPTY OUTPUT section drifted; refresh _OLD_EMPTY_SECTION"
)
PROMPT = BASE_PROMPT.replace(_OLD_EMPTY_SECTION, _NEW_EMPTY_SECTION)


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


class RecurringEmptyPlanner:
    """No-anaphora planner + RECURRING PATTERNS → empty rule."""

    def __init__(self) -> None:
        self._client = AsyncOpenAI(timeout=PER_CALL_TIMEOUT_S)
        self._sem = asyncio.Semaphore(CONCURRENCY)
        self._calls = 0
        self._cache_hits = 0
        self._parse_failures = 0
        self._total = 0
        self._cache_file = CACHE_FILE
        self._cache = self._load_cache()

    def _load_cache(self) -> dict:
        if not self._cache_file.exists():
            return {}
        try:
            return json.loads(self._cache_file.read_text())
        except Exception:
            return {}

    def _save_cache(self) -> None:
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

    async def plan(self, query: str, ref_time: str) -> Plan:
        self._total += 1
        key = _cache_key(query, ref_time)
        if key in self._cache:
            self._cache_hits += 1
            obj = self._cache[key]
            return Plan(
                targets=_json_to_targets(obj.get("targets", [])),
                extremum=obj.get("extremum"),
                raw=json.dumps(obj),
            )

        prompt = PROMPT.format(query=query, ref_time=ref_time)
        format_config: ResponseFormatTextJSONSchemaConfigParam = {
            "type": "json_schema",
            "name": "plan",
            "strict": True,
            "schema": _PLAN_JSON_SCHEMA_NO_ANAPHORA,
        }
        text_config: ResponseTextConfigParam = {"format": format_config}
        async with self._sem:
            try:
                resp = await self._client.responses.create(
                    model=MODEL, input=prompt, text=text_config,
                )
                self._calls += 1
                raw = resp.output_text
                obj = json.loads(raw)
                targets = _json_to_targets(obj.get("targets", []))
                extremum = obj.get("extremum")
                if extremum not in ("latest", "earliest"):
                    extremum = None
                self._cache[key] = obj
                self._save_cache()
                return Plan(targets=targets, extremum=extremum, raw=raw)
            except Exception as e:
                self._parse_failures += 1
                return Plan(parse_error=str(e), raw="")

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
