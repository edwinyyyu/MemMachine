"""No-anaphora planner with explicit retrieval-orientation note.

Tests whether telling the LLM "this is past-event retrieval" fixes the
lattice/mixed_cue/recency regressions from pure-subtraction by aligning
query enumeration with the doc-side extractor's nearest-past-occurrence
behavior.
"""
from __future__ import annotations

from temporal_retrieval_tr.research._no_anaphora_planner import (
    NoAnaphoraPlanner, PROMPT as BASE_PROMPT,
    _PLAN_JSON_SCHEMA_NO_ANAPHORA, MODEL,
    Plan, _cache_key, _json_to_targets,
    PER_CALL_TIMEOUT_S, CONCURRENCY,
)

import asyncio, contextlib, hashlib, json, os
from pathlib import Path

from openai import AsyncOpenAI
from openai.types.responses import ResponseTextConfigParam
from openai.types.responses.response_format_text_json_schema_config_param import (
    ResponseFormatTextJSONSchemaConfigParam,
)

if not os.environ.get("OPENAI_API_KEY"):
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    except Exception:
        pass


PROMPT_VERSION = "v3-retrieval-oriented"

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache" / "planner_retrieval_oriented"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "llm_plan_cache.json"


RETRIEVAL_NOTE = """

RETRIEVAL ORIENTATION
=====================
You are translating queries for a SEARCH/RETRIEVAL system over a memory
of PAST events. When a query asks about recurring or ambiguous-direction
patterns ("what do I do on Thursdays?", "when do I exercise?"),
enumerate PAST occurrences (looking back from REF_TIME), not future
ones. Future-tense queries ("what is scheduled for next Thursday") of
course remain future-oriented.
"""

PROMPT = BASE_PROMPT.replace(
    "NOW PRODUCE THE PLAN FOR:",
    RETRIEVAL_NOTE + "\n\nNOW PRODUCE THE PLAN FOR:",
)


class RetrievalOrientedPlanner:
    """No-anaphora planner with retrieval-orientation note added."""

    def __init__(self) -> None:
        self._client = AsyncOpenAI(timeout=PER_CALL_TIMEOUT_S)
        self._sem = asyncio.Semaphore(CONCURRENCY)
        self._calls = 0
        self._cache_hits = 0
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

    def _key(self, query: str, ref_time: str) -> str:
        h = hashlib.sha256()
        h.update(MODEL.encode())
        h.update(b"|")
        h.update(PROMPT_VERSION.encode())
        h.update(b"|")
        h.update(query.encode())
        h.update(b"|")
        h.update(ref_time.encode())
        return h.hexdigest()

    async def plan(self, query: str, ref_time: str) -> Plan:
        self._total += 1
        key = self._key(query, ref_time)
        if key in self._cache:
            self._cache_hits += 1
            obj = self._cache[key]
            return Plan(targets=_json_to_targets(obj.get("targets", [])),
                        extremum=obj.get("extremum"),
                        raw=json.dumps(obj))

        prompt = PROMPT.format(query=query, ref_time=ref_time)
        fmt: ResponseFormatTextJSONSchemaConfigParam = {
            "type": "json_schema", "name": "plan", "strict": True,
            "schema": _PLAN_JSON_SCHEMA_NO_ANAPHORA,
        }
        text_config: ResponseTextConfigParam = {"format": fmt}
        async with self._sem:
            try:
                resp = await self._client.responses.create(
                    model=MODEL, input=prompt, text=text_config,
                )
                self._calls += 1
                obj = json.loads(resp.output_text)
                targets = _json_to_targets(obj.get("targets", []))
                extremum = obj.get("extremum")
                if extremum not in ("latest", "earliest"):
                    extremum = None
                self._cache[key] = obj
                self._save_cache()
                return Plan(targets=targets, extremum=extremum,
                            raw=resp.output_text)
            except Exception as e:
                return Plan(parse_error=str(e), raw="")
