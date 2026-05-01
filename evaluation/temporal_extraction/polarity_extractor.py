"""Polarity-aware temporal extractor.

Reuses the base ``extractor.Extractor`` Pass 1 (span identification) and
adds a polarity-aware Pass 2 that emits:

    { "time_expression": {...schema...}, "polarity": "affirmed"|...,
      "evidence": "..." }

Uses gpt-5-mini and a separate disk cache under ``cache/polarity/``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from extractor import PASS2_JSON_SCHEMA, Extractor
from openai import AsyncOpenAI
from polarity_schema import DEFAULT_POLARITY, POLARITY_VALUES
from resolver import ResolverError, post_process
from schema import TimeExpression, time_expression_from_dict

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-5-mini"
ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache" / "polarity"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LLM_CACHE_FILE = CACHE_DIR / "llm_cache.json"


# ---------------------------------------------------------------------------
# Cache (mirror of extractor.LLMCache but on a dedicated file)
# ---------------------------------------------------------------------------
class PolarityCache:
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
# Prompt
# ---------------------------------------------------------------------------
PASS2_POLARITY_SYSTEM = """You resolve ONE temporal reference into absolute
wall-clock form AND determine its polarity: did the event actually happen
at the stated time?

Reference time is given; use it to resolve all relative expressions.

Polarity labels (pick exactly one per reference):
- "affirmed"    : the surrounding clause asserts the event happened at this
  time. Example: "She attended the conference last March." -> affirmed.
- "negated"     : the clause explicitly states the event did NOT happen at
  this time via "didn't", "wasn't", "never", "no", "failed to", etc.
  Example: "She didn't attend the conference last March." -> negated.
- "hypothetical": the clause is conditional, counterfactual, aspirational,
  or future-unrealized ("if ... had ...", "would have", "plans to",
  "hoping to", "might go"). Example: "If she had attended, it would have
  been last March." -> hypothetical.
- "uncertain"   : the clause is hedged with "maybe", "probably", "I
  think", "possibly", "seems", "apparently". Example: "She maybe attended
  the conference last March." -> uncertain.

If no explicit polarity cue applies, default to "affirmed".

Also fill the resolved TimeExpression using the same schema as the
affirmative extractor (kind, surface, confidence, instant/interval/
duration/recurrence). Granularity is one of: second, minute, hour, day,
week, month, quarter, year, decade, century. Use UTC ISO 8601 with "Z".

Output JSON of the form:

{
  "time_expression": { ...schema... },
  "polarity": "affirmed" | "negated" | "hypothetical" | "uncertain",
  "evidence": "<short snippet showing the polarity cue, may be empty>"
}
"""


PASS2_POLARITY_JSON_SCHEMA: dict[str, Any] = {
    "name": "polarity_time_expression",
    "strict": False,
    "schema": {
        "type": "object",
        "properties": {
            "time_expression": PASS2_JSON_SCHEMA["schema"],
            "polarity": {
                "type": "string",
                "enum": list(POLARITY_VALUES),
            },
            "evidence": {"type": "string"},
        },
        "required": ["time_expression", "polarity"],
    },
}


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------
class PolarityExtractor:
    """Wraps base ``Extractor`` for Pass 1, supplies own Pass 2."""

    def __init__(self, concurrency: int = 10) -> None:
        self.base = Extractor(concurrency=concurrency)
        self.client = AsyncOpenAI()
        self.sem = asyncio.Semaphore(concurrency)
        self.cache = PolarityCache()
        self.usage: dict[str, int] = {"input": 0, "output": 0}

    async def _call(self, system: str, user: str, json_schema: dict) -> str:
        prompt_key = f"{hashlib.sha256(system.encode()).hexdigest()[:16]}|{user}"
        cached = self.cache.get(MODEL, prompt_key)
        if cached is not None:
            return cached
        kwargs: dict[str, Any] = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_completion_tokens": 1800,
            "response_format": {
                "type": "json_schema",
                "json_schema": json_schema,
            },
        }
        async with self.sem:
            resp = await self.client.chat.completions.create(**kwargs)
        usage = resp.usage
        if usage:
            self.usage["input"] += getattr(usage, "prompt_tokens", 0) or 0
            self.usage["output"] += getattr(usage, "completion_tokens", 0) or 0
        content = resp.choices[0].message.content or ""
        self.cache.put(MODEL, prompt_key, content)
        return content

    async def pass2_polarity(
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
            "Return JSON with {time_expression, polarity, evidence}."
        )
        raw = await self._call(PASS2_POLARITY_SYSTEM, user, PASS2_POLARITY_JSON_SCHEMA)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    async def extract(
        self, text: str, ref_time: datetime
    ) -> list[tuple[TimeExpression, str, str]]:
        """Returns list of (TimeExpression, polarity, evidence)."""
        refs = await self.base.pass1(text, ref_time)
        coros = []
        metadata: list[tuple[str, str]] = []
        for ref in refs:
            surface = ref.get("surface") or ""
            if not surface:
                continue
            kind_guess = ref.get("kind_guess", "instant")
            context_hint = ref.get("context_hint", "")
            metadata.append((surface, kind_guess))
            coros.append(
                self.pass2_polarity(surface, kind_guess, context_hint, text, ref_time)
            )
        results = await asyncio.gather(*coros)
        out: list[tuple[TimeExpression, str, str]] = []
        for (surface, _), pred in zip(metadata, results):
            if pred is None:
                continue
            te_dict = pred.get("time_expression")
            if not te_dict:
                continue
            polarity = pred.get("polarity", DEFAULT_POLARITY)
            if polarity not in POLARITY_VALUES:
                polarity = DEFAULT_POLARITY
            evidence = pred.get("evidence", "") or ""
            te_dict["reference_time"] = ref_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            try:
                te = time_expression_from_dict(te_dict)
            except Exception:
                continue
            idx = text.find(surface)
            if idx >= 0:
                te.span_start = idx
                te.span_end = idx + len(surface)
            try:
                te, _warnings = post_process(te, auto_correct=True)
            except ResolverError:
                continue
            out.append((te, polarity, evidence))
        return out

    def save(self) -> None:
        self.cache.save()
        self.base.cache.save()


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------
async def extract_many_polarity(
    items: list[tuple[str, str, datetime]],
) -> tuple[dict[str, list[tuple[TimeExpression, str, str]]], dict[str, int]]:
    ex = PolarityExtractor()
    results: dict[str, list[tuple[TimeExpression, str, str]]] = {}

    async def one(iid: str, text: str, ref_time: datetime) -> None:
        results[iid] = await ex.extract(text, ref_time)

    await asyncio.gather(*(one(i, t, r) for i, t, r in items))
    ex.save()
    # merge usage: include base's pass1 usage
    usage = {
        "input": ex.usage["input"] + ex.base.usage["input"],
        "output": ex.usage["output"] + ex.base.usage["output"],
    }
    return results, usage
