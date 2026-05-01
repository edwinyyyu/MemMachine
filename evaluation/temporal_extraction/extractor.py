"""Two-pass LLM extractor using gpt-5-mini.

Pass 1: identify temporal spans.
Pass 2: normalize each span into a TimeExpression structured-output.

Uses asyncio.Semaphore(10), file-backed JSON cache keyed by prompt hash,
and prompt caching via a long stable system message (OpenAI infers cache
hits automatically for gpt-5-mini when the system prefix is repeated).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI
from resolver import ResolverError, post_process
from schema import (
    TimeExpression,
    time_expression_from_dict,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-5-mini"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
LLM_CACHE_FILE = CACHE_DIR / "llm_cache.json"


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
class LLMCache:
    def __init__(self, path: Path = LLM_CACHE_FILE) -> None:
        self.path = path
        self._cache: dict[str, str] = {}
        if path.exists():
            with path.open() as f:
                self._cache = json.load(f)
        self._new: dict[str, str] = {}

    @staticmethod
    def _key(model: str, prompt_key: str) -> str:
        return hashlib.sha256(f"{model}|{prompt_key}".encode()).hexdigest()

    def get(self, model: str, prompt_key: str) -> str | None:
        return self._cache.get(self._key(model, prompt_key))

    def put(self, model: str, prompt_key: str, response: str) -> None:
        k = self._key(model, prompt_key)
        self._cache[k] = response
        self._new[k] = response

    def save(self) -> None:
        if not self._new:
            return
        existing: dict[str, str] = {}
        if self.path.exists():
            with self.path.open() as f:
                existing = json.load(f)
        existing.update(self._new)
        tmp = self.path.with_suffix(".json.tmp")
        with tmp.open("w") as f:
            json.dump(existing, f)
        tmp.replace(self.path)
        self._new.clear()


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
PASS1_SYSTEM = """You are a meticulous temporal-reference extractor.

Your job: identify every temporal reference in a passage. A temporal
reference is any span that refers to a moment, span, duration, or
recurring pattern in time. It can be absolute ("March 5, 2026"),
relative ("yesterday", "2 weeks ago"), vague ("around 2010", "a decade
ago"), or recurring ("every Thursday at 3pm").

For each reference, output:
- surface: the exact substring from the passage, verbatim, with no
  edits to casing, spacing, or punctuation. Include "on", "in", "from",
  "during", etc. ONLY when they are part of the natural temporal phrase
  (e.g., include them for "from March 5 to March 12"; exclude them for
  "On March 15, 2026 I visited...").
- kind_guess: one of [instant, interval, duration, recurrence].
  - instant: a point-in-time (even if fuzzy): "yesterday", "2015".
  - interval: a start-to-end range: "from X to Y".
  - duration: an unanchored length: "for 3 weeks", "two hours long".
  - recurrence: a recurring pattern: "every Thursday".
- context_hint: a short (<=12 word) note of what it refers to.

Do NOT emit seasons ("summer") unless the year is specified or strongly
implied by context. Do NOT emit "once", "often", "always".

Output a single JSON object: {"refs": [...]}. If none, output {"refs": []}.
"""

PASS2_SYSTEM = """You resolve ONE temporal reference into absolute wall-clock form.

Reference time is given; use it to resolve all relative expressions.
Compute carefully - check weekday alignment, month length, year
rollovers.

Output JSON matching this schema exactly. Omit fields not relevant to
the kind by setting them to null.

{
  "kind": "instant" | "interval" | "duration" | "recurrence",
  "surface": string,
  "confidence": float in [0,1],
  "instant": { "earliest": ISO, "latest": ISO, "best": ISO|null, "granularity": string } | null,
  "interval": { "start": {...instant...}, "end": {...instant...} } | null,
  "duration": { "seconds": int } | null,
  "recurrence": {
    "rrule": string,
    "dtstart": {...instant...},
    "until": {...instant...} | null,
    "exdates": [{...instant...}]
  } | null
}

Granularity is one of: second, minute, hour, day, week, month, quarter,
year, decade, century.

Rules:
- Use UTC ISO 8601 with "Z" suffix for all datetimes.
- earliest is inclusive, latest is exclusive.
- For "about"/"around"/"roughly"/"a few"/"a couple", widen the interval
  by one granularity level; set best to the centered estimate.
- "yesterday" at ref R -> day bracket of R-1d, granularity=day.
- "N weeks ago" at ref R -> day bracket of R-7N days, granularity=day.
  Similarly "in N days", "N days ago", "N days from now".
- "last week" -> week bracket of the ISO week before R, granularity=week.
- "last year" -> year bracket of R.year-1, granularity=year.
- "in YYYY" -> year bracket of YYYY, granularity=year.
- "the 90s" -> decade bracket [1990, 2000), granularity=decade.
- "around YYYY" -> year bracket of YYYY plus +/- 2 years, granularity=year.
- For recurrences "every Thursday": rrule "FREQ=WEEKLY;BYDAY=TH".
  No trailing semicolon. No DTSTART inside the rrule string (put it in
  dtstart instead).
- "every day at 9am": rrule "FREQ=DAILY;BYHOUR=9;BYMINUTE=0".
- For recurrences with no explicit end, set until to null.
- Always fill the instant.earliest/latest/best triple. Best is the
  typical point estimate.
"""

PASS2_JSON_SCHEMA: dict[str, Any] = {
    "name": "time_expression",
    "strict": False,
    "schema": {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["instant", "interval", "duration", "recurrence"],
            },
            "surface": {"type": "string"},
            "confidence": {"type": "number"},
            "instant": {
                "type": ["object", "null"],
                "properties": {
                    "earliest": {"type": "string"},
                    "latest": {"type": "string"},
                    "best": {"type": ["string", "null"]},
                    "granularity": {"type": "string"},
                },
            },
            "interval": {
                "type": ["object", "null"],
                "properties": {
                    "start": {"type": "object"},
                    "end": {"type": "object"},
                },
            },
            "duration": {
                "type": ["object", "null"],
                "properties": {"seconds": {"type": "number"}},
            },
            "recurrence": {
                "type": ["object", "null"],
                "properties": {
                    "rrule": {"type": "string"},
                    "dtstart": {"type": "object"},
                    "until": {"type": ["object", "null"]},
                    "exdates": {
                        "type": "array",
                        "items": {"type": "object"},
                    },
                },
            },
        },
        "required": ["kind", "surface"],
    },
}


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------
class Extractor:
    def __init__(self, concurrency: int = 10) -> None:
        self.client = AsyncOpenAI()
        self.sem = asyncio.Semaphore(concurrency)
        self.cache = LLMCache()
        self.usage: dict[str, int] = {"input": 0, "output": 0}

    async def _call(
        self,
        model: str,
        system: str,
        user: str,
        *,
        json_schema: dict | None = None,
        json_object: bool = False,
        max_completion_tokens: int = 2000,
    ) -> str:
        # Cache key is model + system-prefix hash + user (system caching still
        # handled server-side by OpenAI; cache key here is just for replay).
        prompt_key = f"{hashlib.sha256(system.encode()).hexdigest()[:16]}|{user}"
        cached = self.cache.get(model, prompt_key)
        if cached is not None:
            return cached
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
        }
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": json_schema,
            }
        elif json_object:
            kwargs["response_format"] = {"type": "json_object"}
        async with self.sem:
            resp = await self.client.chat.completions.create(**kwargs)
        usage = resp.usage
        if usage:
            self.usage["input"] += getattr(usage, "prompt_tokens", 0) or 0
            self.usage["output"] += getattr(usage, "completion_tokens", 0) or 0
        content = resp.choices[0].message.content or ""
        self.cache.put(model, prompt_key, content)
        return content

    async def pass1(self, text: str, ref_time: datetime) -> list[dict[str, Any]]:
        wk = ref_time.strftime("%A")
        iso_ref = ref_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        user = (
            f"Reference time: {iso_ref} ({wk})\n"
            f"Passage:\n{text}\n\n"
            'Return {"refs": [...]} as JSON.'
        )
        raw = await self._call(MODEL, PASS1_SYSTEM, user, json_object=True)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return list(data.get("refs", []))

    async def pass2(
        self,
        surface: str,
        kind_guess: str,
        context_hint: str,
        surrounding: str,
        ref_time: datetime,
    ) -> dict[str, Any] | None:
        wk = ref_time.strftime("%A")
        iso_ref = ref_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        user = (
            f"Reference time: {iso_ref} ({wk})\n"
            f"Surrounding context: {surrounding}\n"
            f'Reference: "{surface}"\n'
            f"Kind hint: {kind_guess}\n"
            f"Context hint: {context_hint}\n\n"
            "Return JSON matching the schema."
        )
        raw = await self._call(
            MODEL,
            PASS2_SYSTEM,
            user,
            json_schema=PASS2_JSON_SCHEMA,
            max_completion_tokens=1500,
        )
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    async def extract(self, text: str, ref_time: datetime) -> list[TimeExpression]:
        refs = await self.pass1(text, ref_time)
        coros = []
        metadata = []
        for ref in refs:
            surface = ref.get("surface") or ""
            if not surface:
                continue
            kind_guess = ref.get("kind_guess", "instant")
            context_hint = ref.get("context_hint", "")
            metadata.append((surface, kind_guess))
            coros.append(self.pass2(surface, kind_guess, context_hint, text, ref_time))
        results = await asyncio.gather(*coros)
        out: list[TimeExpression] = []
        for (surface, _), pred in zip(metadata, results):
            if pred is None:
                continue
            # Attach reference_time
            pred["reference_time"] = ref_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            # Attach span offsets if we can find it
            try:
                te = time_expression_from_dict(pred)
            except Exception:
                continue
            # Span offsets in source text
            idx = text.find(surface)
            if idx >= 0:
                te.span_start = idx
                te.span_end = idx + len(surface)
            try:
                te, _warnings = post_process(te, auto_correct=True)
            except ResolverError:
                continue
            out.append(te)
        return out


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------
async def extract_many(
    items: list[tuple[str, str, datetime]],
) -> tuple[dict[str, list[TimeExpression]], dict[str, int]]:
    """items: (id, text, ref_time). Returns ({id: [TE]}, usage)."""
    ex = Extractor()
    results: dict[str, list[TimeExpression]] = {}

    async def one(iid: str, text: str, ref_time: datetime) -> None:
        results[iid] = await ex.extract(text, ref_time)

    await asyncio.gather(*(one(i, t, r) for i, t, r in items))
    ex.cache.save()
    return results, ex.usage
