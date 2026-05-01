"""E4 — Named-era / implicit-time extraction.

Extended Pass 1 + Pass 2 prompts that detect and resolve:
- personal eras: "during college", "after COVID", "in my 20s",
  "before the kids were born"
- world-knowledge eras: "the Obama years", "the 90s", "post-WWII",
  "during the Cold War"

Uses gpt-5-mini + resolver.post_process for span validation. Outputs
TimeExpression objects compatible with the base IntervalStore.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

from advanced_common import LLMCaller
from resolver import ResolverError, post_process
from schema import TimeExpression, time_expression_from_dict

ERA_PASS1_SYSTEM = """You are a meticulous temporal-reference extractor
with explicit handling of NAMED ERAS and IMPLICIT WORLD-KNOWLEDGE TIMES.

Identify every temporal reference. These include:

A. Concrete times (absolute, relative, recurring) — same as before.
B. Personal eras — "during college", "after COVID", "in my 20s",
   "before the kids were born", "when I was a teenager", "since we moved".
C. World-knowledge eras — "the Obama years", "the 90s", "post-WWII",
   "during the Cold War", "the Great Recession", "pre-internet", "the
   Industrial Revolution".

For each reference, output:
- surface: exact verbatim substring. Include natural prefix words if
  they are part of the phrase ("during college"), but not sentence-
  setting prepositions ("on March 15").
- kind_guess: one of [instant, interval, duration, recurrence].
   - personal/world eras are almost always INTERVAL.
- era_type: "none" | "personal" | "world" — classify the reference.
- context_hint: short (<=12 word) note of what it refers to.

Output JSON: {"refs": [...]}. If none, output {"refs": []}.
"""

ERA_PASS2_SYSTEM = """You resolve ONE temporal reference to an absolute
UTC wall-clock window. You handle NAMED ERAS.

For world-knowledge eras, use the standard accepted dates:
- "the Obama years" -> 2009-01-20 to 2017-01-20
- "the Clinton era" -> 1993-01-20 to 2001-01-20
- "the 90s" / "the nineties" -> 1990-01-01 to 2000-01-01
- "the 80s" -> 1980-01-01 to 1990-01-01
- "the 2010s" -> 2010-01-01 to 2020-01-01
- "post-WWII" / "after WWII" -> 1945-09-02 to 1991-12-26 (generally the
  postwar era; use 1945-09-02 to 1960-01-01 for a tighter bracket if
  context implies immediate postwar)
- "the Cold War" -> 1947-03-12 to 1991-12-26
- "during COVID" / "the COVID era" -> 2020-03-11 to 2023-05-05
- "post-COVID" / "after COVID" -> 2023-05-05 to reference_time + 5y
  (fuzzy; mark granularity year)
- "the Great Recession" -> 2007-12-01 to 2009-06-01
- "pre-internet" -> 1900-01-01 to 1995-01-01
- "the Industrial Revolution" -> 1760-01-01 to 1840-01-01

For personal eras, estimate using general norms (or explicit context
clues in `surrounding`):
- "during college" -> assume age 18-22 if birth year is given or implied,
  else use a 4-year interval anchored to an explicit college time-cue.
  If nothing anchors it, use reference_time minus 10y to minus 6y as a
  rough fuzzy default.
- "in my 20s" -> if birth year known, birth+20 to birth+30; else
  reference_time minus 25y to minus 15y as a rough default.
- "before the kids were born" / "before kids" -> if a child birth year
  is mentioned in surrounding text, that's the upper bound; else
  reference_time minus 15y to minus 5y as a rough default.
- "when I was a teenager" -> age 13-19 if birth year known, else
  reference_time minus 35y to minus 25y fuzzy.

Output JSON matching this schema EXACTLY. Omit fields not relevant by
setting them to null.

{
  "kind": "instant" | "interval" | "duration" | "recurrence",
  "surface": string,
  "confidence": float in [0,1] (lower confidence=0.5 for rough personal
    era guesses with no anchor, 0.8 for well-known world eras, 1.0 for
    explicit dates),
  "instant": { "earliest": ISO, "latest": ISO, "best": ISO|null,
    "granularity": string } | null,
  "interval": { "start": {...instant...}, "end": {...instant...} } | null,
  "duration": { "seconds": int } | null,
  "recurrence": {...} | null
}

Granularity: one of [second, minute, hour, day, week, month, quarter,
year, decade, century]. For named eras at year-level ("the Obama years")
use "year" or "decade". For fuzzy personal eras use "year" or "decade".

Rules:
- UTC ISO 8601 with trailing "Z".
- earliest inclusive, latest exclusive.
- Always fill the instant.earliest/latest/best triple (for interval, fill
  both start and end).
"""

PASS1_SCHEMA = {
    "name": "era_refs",
    "strict": False,
    "schema": {
        "type": "object",
        "properties": {
            "refs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "surface": {"type": "string"},
                        "kind_guess": {"type": "string"},
                        "era_type": {"type": "string"},
                        "context_hint": {"type": "string"},
                    },
                    "required": ["surface"],
                },
            }
        },
        "required": ["refs"],
    },
}

PASS2_SCHEMA = {
    "name": "era_time_expression",
    "strict": False,
    "schema": {
        "type": "object",
        "properties": {
            "kind": {"type": "string"},
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
            "recurrence": {"type": ["object", "null"]},
        },
        "required": ["kind", "surface"],
    },
}


class EraExtractor:
    def __init__(self, llm: LLMCaller) -> None:
        self.llm = llm

    async def pass1(self, text: str, ref_time: datetime) -> list[dict[str, Any]]:
        iso_ref = ref_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        wk = ref_time.strftime("%A")
        user = (
            f"Reference time: {iso_ref} ({wk})\n"
            f"Passage:\n{text}\n\n"
            'Return {"refs": [...]}.'
        )
        raw = await self.llm.chat(
            ERA_PASS1_SYSTEM,
            user,
            json_schema=PASS1_SCHEMA,
            max_completion_tokens=2000,
            cache_tag="e4_pass1",
        )
        if not raw:
            return []
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return list(d.get("refs") or [])

    async def pass2(
        self,
        surface: str,
        kind_guess: str,
        context_hint: str,
        era_type: str,
        surrounding: str,
        ref_time: datetime,
    ) -> dict[str, Any] | None:
        iso_ref = ref_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        wk = ref_time.strftime("%A")
        user = (
            f"Reference time: {iso_ref} ({wk})\n"
            f"Surrounding context: {surrounding}\n"
            f'Reference: "{surface}"\n'
            f"Kind hint: {kind_guess}\n"
            f"Era type: {era_type}\n"
            f"Context hint: {context_hint}\n\n"
            "Return JSON matching the schema."
        )
        raw = await self.llm.chat(
            ERA_PASS2_SYSTEM,
            user,
            json_schema=PASS2_SCHEMA,
            max_completion_tokens=2500,
            cache_tag="e4_pass2",
        )
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    async def extract(self, text: str, ref_time: datetime) -> list[TimeExpression]:
        refs = await self.pass1(text, ref_time)
        tasks = []
        for r in refs:
            surface = (r.get("surface") or "").strip()
            if not surface:
                continue
            tasks.append(
                (
                    surface,
                    self.pass2(
                        surface,
                        r.get("kind_guess", "instant"),
                        r.get("context_hint", ""),
                        r.get("era_type", "none"),
                        text,
                        ref_time,
                    ),
                )
            )
        if not tasks:
            return []
        results = await asyncio.gather(*(t[1] for t in tasks))
        out: list[TimeExpression] = []
        for (surface, _), pred in zip(tasks, results):
            if not pred:
                continue
            pred["reference_time"] = ref_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            try:
                te = time_expression_from_dict(pred)
            except Exception:
                continue
            idx = text.find(surface)
            if idx >= 0:
                te.span_start = idx
                te.span_end = idx + len(surface)
            try:
                te, _ = post_process(te, auto_correct=True)
            except ResolverError:
                continue
            out.append(te)
        return out
