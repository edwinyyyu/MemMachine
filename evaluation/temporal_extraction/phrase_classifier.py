"""LLM-based phrase classifier (v5.0).

Replaces the v4.5 regex helpers (looks_calendar, looks_anaphoric,
_PERSONAL_ERA_RE, _RECURRING_PERIOD_WORD_RE, strip_fabricated_year) with
a single LLM call that classifies each planner-emitted phrase into one
of five kinds, given the original query for context.

Robust to:
  - misspellings ("Marh" -> recurring_period)
  - multilingual periods ("marzo", "été", "春", "im Oktober")
  - non-Western recurring observances (Lunar New Year, Ramadan, Diwali)
  - era references ("during the pandemic", "in the dotcom era")
  - planner-fabricated year qualifiers ("May 2025" emitted for query
    "What did I publish in May?" -> recurring_period, NOT calendar_pin)

Cache key includes prompt version, model, query, phrase, ref_time,
direction, so each unique combination is classified once.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from dataclasses import dataclass
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
PER_CALL_TIMEOUT_S = 30.0
CONCURRENCY = 16
PROMPT_VERSION = (
    "v5.1"  # tightened anaphoric_event criteria — entity names go to generic_skip
)

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache" / "phrase_classifier"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "phrase_class_cache.json"


KINDS = (
    "calendar_pin",
    "recurring_period",
    "anaphoric_event",
    "personal_era",
    "generic_skip",
)


PROMPT = """Classify a temporal phrase emitted by a query planner.

User query: {query}
Reference time: {ref_time}
Phrase to classify: "{phrase}"
Direction: {direction}

Decide which KIND best describes how the retrieval system should
interpret this phrase:

- calendar_pin: phrase grounds to a SPECIFIC calendar interval (a
  particular year, a single deictic instance like "yesterday" or
  "two weeks ago"). The retrieval system can apply a hard mask
  matching documents whose timestamps fall in this interval.
  Examples: "March 2024", "Q4 2023", "yesterday", "two weeks ago",
  "last March", "summer 2024", "twenty-twenty-four", "FY2024".

- recurring_period: phrase denotes a year-unspecified period or
  recurring observance — the user could plausibly mean any of
  several instances. The system should NOT apply a mask; it should
  surface candidates from all plausible instances and let semantic
  rerank choose among them.
  Examples: "March", "summer", "Q1", "the holidays", "the winter
  holidays", "Lunar New Year", "Ramadan", "Diwali", "Marh"
  (misspelled March), "marzo" (Spanish March), "été" (French
  summer), "春" (Chinese spring), "im Oktober" (German "in
  October").

- anaphoric_event: phrase refers to a SPECIFIC corpus event the
  user has mentioned before, NOT a calendar period. The system
  should look up the event in the corpus via embedding similarity
  to find its date. STRICT criteria:
    - the phrase introduces a discrete event/milestone (a verb-y
      noun like launch, migration, redesign, offsite, keynote,
      merger, kickoff, freeze, year-end review)
    - the phrase typically starts with "the " (a definite article
      pointing back to something earlier mentioned), AND
    - the surrounding query has a direction cue ("after", "before",
      "since", "until") — the user is scoping time RELATIVE to that
      event
  Examples: "the launch" (after the launch), "the migration"
  (before the migration), "the offsite" (since the offsite),
  "the redesign" (after the redesign).
  COUNTER-EXAMPLES (NOT anaphoric_event):
    - bare entity names ("Theodor Körner", "Acme", "Daniele
      Dell'Innocenti") — these are not corpus events, just names.
      Use generic_skip.
    - phrases without "the " ("Sarah's promotion", "the team
      offsite at Acme") — be cautious; if the phrase doesn't
      clearly refer to a single previously-mentioned event,
      prefer generic_skip.
    - personal eras ("grad school") — use personal_era.

- personal_era: phrase denotes a USER-SPECIFIC stretch of life that
  can't be grounded against ref_time without more context. The
  system should NOT apply a mask; treat as semantic-only.
  Examples: "grad school", "back in college", "my parental leave",
  "during my fitness phase", "when I lived in Boston", "during the
  divorce", "the year I turned 30", "during the pandemic" (era ref
  spanning multiple years).

- generic_skip: phrase has no temporal scoping intent — the user is
  asking ABOUT the period topically, not constraining time.
  Examples: "the morning routine" (asking how to plan a routine,
  not a specific morning), "the future of AI" (topical), "spring
  usually starts when?" (definitional).

YEAR FABRICATION CHECK
======================
The planner sometimes inserts a year qualifier that's NOT in the
user's query. For example, the user asks "What did I publish in May?"
and the planner emits "May 2025". Treat such fabricated year
qualifiers as if they aren't there:

  User query: "What did I publish in May?"
  Phrase: "May 2025"
  -> kind = recurring_period (the year is fabricated; only "May"
     came from the user)

  User query: "What did I publish in May 2024?"
  Phrase: "May 2024"
  -> kind = calendar_pin (the year IS in the query)

  User query: "What were my Q1 retros?"
  Phrase: "Q1 2025"
  -> kind = recurring_period (year fabricated)

  User query: "What were my Q1 2024 retros?"
  Phrase: "Q1 2024"
  -> kind = calendar_pin (year in query)

When in doubt, prefer recurring_period over calendar_pin. False
recurring_period falls back to semantic+rerank — usually fine.
False calendar_pin masks out valid candidates — usually bad.

Output JSON: {{"kind": "<one of the kinds>", "rationale": "..."}}
"""


_CLASSIFY_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kind", "rationale"],
    "properties": {
        "kind": {"type": "string", "enum": list(KINDS)},
        "rationale": {"type": "string"},
    },
}


@dataclass
class PhraseClass:
    kind: str
    rationale: str = ""
    raw: str = ""


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


def _cache_key(query: str, ref_time: str, phrase: str, direction: str) -> str:
    body = f"{MODEL}|{PROMPT_VERSION}|{query}|{ref_time}|{phrase}|{direction}"
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


class PhraseClassifier:
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
                        max_output_tokens=2048,
                        text={
                            "format": {
                                "type": "json_schema",
                                "name": "PhraseClass",
                                "strict": True,
                                "schema": _CLASSIFY_JSON_SCHEMA,
                            }
                        },
                    ),
                    timeout=PER_CALL_TIMEOUT_S,
                )
            except Exception:
                return ""
            return resp.output_text or ""

    async def classify(
        self, query: str, ref_time: str, phrase: str, direction: str
    ) -> PhraseClass:
        prompt = PROMPT.format(
            query=query, ref_time=ref_time, phrase=phrase, direction=direction
        )
        k = _cache_key(query, ref_time, phrase, direction)
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
            # Safe default: no mask, fall through to rerank.
            return PhraseClass(
                kind="recurring_period", rationale="parse_failed", raw=raw
            )
        kind = obj.get("kind") or "recurring_period"
        if kind not in KINDS:
            self.parse_failures += 1
            kind = "recurring_period"
        return PhraseClass(kind=kind, rationale=str(obj.get("rationale", "")), raw=raw)

    async def classify_many(
        self, items: list[tuple[str, str, str, str, str]]
    ) -> dict[str, PhraseClass]:
        """items: list of (tag, query, ref_time, phrase, direction)."""

        async def _one(tag, q, rt, ph, dr):
            c = await self.classify(q, rt, ph, dr)
            return tag, c

        results = await asyncio.gather(*(_one(*it) for it in items))
        return dict(results)

    def stats(self) -> dict[str, Any]:
        total = self.calls + self.cache_hits
        return {
            "model": self.model,
            "prompt_version": PROMPT_VERSION,
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "parse_failures": self.parse_failures,
            "total_phrases": total,
            "cache_hit_rate": self.cache_hits / max(1, total),
        }


async def _smoke():
    cls = PhraseClassifier()
    cases = [
        # (query, ref_time, phrase, direction, expected)
        (
            "What did I publish in May?",
            "2025-06-15T00:00:00Z",
            "May 2025",
            "in",
            "recurring_period",
        ),
        (
            "What did I publish in May 2024?",
            "2025-06-15T00:00:00Z",
            "May 2024",
            "in",
            "calendar_pin",
        ),
        (
            "What did I work on in March?",
            "2025-06-15T00:00:00Z",
            "March",
            "in",
            "recurring_period",
        ),
        (
            "What did I work on in Marh?",
            "2025-06-15T00:00:00Z",
            "Marh",
            "in",
            "recurring_period",
        ),
        (
            "What did I do en marzo?",
            "2025-06-15T00:00:00Z",
            "marzo",
            "in",
            "recurring_period",
        ),
        (
            "What did I do en été?",
            "2025-06-15T00:00:00Z",
            "été",
            "in",
            "recurring_period",
        ),
        (
            "What did I cook for Lunar New Year?",
            "2025-06-15T00:00:00Z",
            "Lunar New Year",
            "in",
            "recurring_period",
        ),
        (
            "What did I cook during Ramadan?",
            "2025-06-15T00:00:00Z",
            "Ramadan",
            "in",
            "recurring_period",
        ),
        (
            "What courses did I take during the pandemic?",
            "2025-06-15T00:00:00Z",
            "the pandemic",
            "in",
            "personal_era",
        ),
        (
            "What did Maya say after the launch?",
            "2025-06-01T00:00:00Z",
            "the launch",
            "after",
            "anaphoric_event",
        ),
        (
            "What did I do during grad school?",
            "2025-04-01T00:00:00Z",
            "grad school",
            "in",
            "personal_era",
        ),
        (
            "What did I do yesterday?",
            "2025-06-15T00:00:00Z",
            "yesterday",
            "in",
            "calendar_pin",
        ),
        (
            "What happened two weeks ago?",
            "2025-06-15T00:00:00Z",
            "two weeks ago",
            "in",
            "calendar_pin",
        ),
    ]
    items = [(f"t{i}", q, rt, ph, dr) for i, (q, rt, ph, dr, _) in enumerate(cases)]
    res = await cls.classify_many(items)
    print(f"{'phrase':35s} {'expected':20s} {'got':20s} match")
    for (q, rt, ph, dr, expected), tag in zip(cases, [it[0] for it in items]):
        c = res[tag]
        match = "OK" if c.kind == expected else "FAIL"
        print(f"{ph[:35]:35s} {expected:20s} {c.kind:20s} {match}")
    print(json.dumps(cls.stats(), indent=2))


if __name__ == "__main__":
    asyncio.run(_smoke())
