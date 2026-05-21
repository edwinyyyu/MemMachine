"""Unified-envelope temporal extractor (v2).

The v1 extractor emits four kinds (instant/interval/duration/recurrence)
into a discriminated union with nested FuzzyInstant/FuzzyInterval/
Recurrence sub-shapes and a `best` field per FuzzyInstant.

Downstream retrieval collapses every kind to a single
`Interval(earliest_us, latest_us, best_us, granularity)` at
`flatten_intervals`. Ablations (validated, see project_kind_reframe and
project_best_us_dispensable memories) showed:
  - recurrence expansion = dead weight; dtstart alone matches full
  - best_us = decorative; midpoint matches production on 6/7 benches
  - interval END = load-bearing; keep envelope width

This v2 extractor follows that finding: the LLM emits ONE shape — a
temporal envelope `(earliest, latest, granularity)` — and the
extractor wraps it as a TimeExpression(kind="instant",
instant=FuzzyInstant(...)) so the retriever needs no changes. The
downstream pipeline operates on Interval primitives identically.

What gets simpler:
- LLM no longer makes a kind decision before emitting fields
- JSON schema is flat (no discriminated union, no nested kinds)
- No `best` field (midpoint computed at flatten)
- No rrule / dtstart / until / exdates for recurrences (emit first
  occurrence as an instant; pattern info dropped — validated dead
  weight for retrieval recall)

Cache directory: `cache/temporal_retrieval_v2/` (separate from v1 so
both can coexist during A/B).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

from .extractor_common import _LLMCache, full_ref_context
from .schema import TimeEnvelope, parse_iso

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-5-mini"
PROMPT_VERSION = "v2_1"
CACHE_ROOT = (
    Path(__file__).resolve().parent / "cache" / f"temporal_retrieval_{PROMPT_VERSION}"
)
CACHE_ROOT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# PASS 1 — find candidate temporal surfaces in the passage
# ---------------------------------------------------------------------------
# Unchanged structurally from v1 but with kind_guess dropped (the LLM no
# longer makes a kind decision in pass 1; pass 2 emits the envelope
# directly). context_hint stays — it's useful disambiguation context for
# pass 2.
PASS1_SYSTEM_V2 = """You are a temporal-reference extractor.

Your job: identify EVERY span in a passage that names a specific point, span,
or recurring schedule of time that a reader could plot on a calendar given the
passage's reference time.

# Critical test before emitting anything

Does the passage USE this phrase to locate a specific occurrence on the
calendar — one that the reader could later recall, search for, or
reference by date? Or does the phrase describe a constraint, a
requirement, a format placeholder, or a rule that applies generally
across many possible occurrences?

- Specific occurrence ("yesterday we deployed", "shipped on March 15",
  "Q1 was rough", "during the pandemic", "every Thursday at 3pm I have
  therapy") -> EMIT.
- Constraint / rule / placeholder ("Policy: backups within the last
  hour", "every release requires a 30-minute window", "Subject format:
  [Date]") -> SKIP, even if temporal-shaped.

This is the deciding test for borderline cases. The retriever's job is
to surface timeless rule docs on non-temporal queries; anchoring a
policy at the reference time defeats that.

# What counts as a temporal reference (the criterion)

A span is a temporal reference if and only if, given the reference time and
any explicit anchoring in the passage, you could (in principle) state WHEN
it is or WHAT pattern it follows on a calendar — AND the critical test
above puts it on the "specific occurrence" side:

- Absolute dates: "March 5, 2026", "1986", "Q3 2025".
- Relative deictics: "yesterday", "2 weeks ago", "next Thursday", "last month".
- Approximations: "around 2010", "a few weeks ago", "some time ago",
  "recently", "lately".
- Eras with a calendar anchor: "the 90s", "back in college" (a
  personal-history span the passage may resolve), "during the pandemic".
- Recurring schedules tied to a real standing pattern in the writer's
  life: "every Thursday at 3pm" (a standing arrangement), "monthly"
  (a regular cadence). Emit the first / nearest known occurrence.
- Durations: "for 3 weeks", "two hours long" — emit ONLY if attached to
  a specific calendar anchor in the passage. "for 3 weeks starting
  June 1" -> emit. "a 30-minute monitoring window post-deploy" inside
  a policy -> SKIP (the anchor is generic, not specific).

# What does NOT count (skip)

The unifying principle: skip phrases that name time without pinning a
SPECIFIC occurrence, and skip everything inside policy / rule / format
contexts.

- Bare names of recurring annual events without a year-anchor: "summer",
  "Christmas", "Easter", "Ramadan", "graduation day", "the gala". They
  name a CLASS across many years; without a year you can't identify one.
  EXCEPTION: when the phrase IS the recurring schedule itself in a
  standing-arrangement context — "every summer we visit the lake" — emit it.
- Vague descriptors that label time without bounding it: adjective forms
  like "recent", "modern", "old", "new", "ancient". (Adverb forms —
  "recently", "lately" — DO emit; they anchor a window near ref time.)
- Bare frequency words: "often", "always", "sometimes", "once",
  "rarely", "occasionally". These describe how-often, not when.
- Bare approximators without a concrete reference: "about", "around",
  "roughly", "nearly" used alone.

# Policy / rule / template contexts — skip everything inside

When the surrounding sentence describes a generic policy, rule,
convention, requirement, or format, even temporal-shaped phrases inside
it are CONSTRAINTS or PLACEHOLDERS, not events. The reader will not
later ask "when was the 30-minute window" or "when was [Date]" — those
phrases don't refer to a specific moment.

Cue patterns that mark a policy / rule context — when present, all
temporal-shaped phrases in the same sentence are SKIPPED:
- Explicit policy header: "policy:", "convention:", "rule:",
  "guideline:", "standard:".
- Prescriptive modals as main predicate: "must X", "should X",
  "requires X", "is not permitted", "never X without Y", "always X
  before Y".
- Recurrence over an event-CLASS without naming a specific instance:
  "every release", "every deploy", "every PR", "every sprint", "each
  meeting" (where the noun is an event-class the org runs many of).

Cue patterns for templates / placeholders — skip the placeholder:
- Brackets around a generic word: "[Date]", "[Time]", "{date}",
  "<date>", "%timestamp%".
- Format strings spelled out: "YYYY-MM-DD", "HH:MM:SS", "MM/DD/YYYY".

If you skip on these cues, do not silently emit other temporal phrases
from the same sentence either; the whole sentence is policy/rule
content.

# Output

For each emitted reference:
- surface: the exact substring from the passage, verbatim. Prefer the
  LONGEST natural phrase carrying the temporal meaning — include
  determiners like "the" / "every" and qualifiers like "earlier",
  "later", "around", "about", "the first week of" when they are part of
  the phrase. Do NOT include a leading bare "on"/"in" preposition.
- context_hint: a short (<=12 words) note of what it refers to.

# Three minimal examples

Passage: "Earlier this month we repainted the kitchen, and the week before that we ordered the cabinets."
{"refs":[
  {"surface":"Earlier this month","context_hint":"earlier half of current calendar month"},
  {"surface":"the week before that","context_hint":"one week before 'earlier this month'"}
]}

Passage: "Back in the 90s, dial-up was the norm; by around 2008 most homes had broadband."
{"refs":[
  {"surface":"the 90s","context_hint":"decade 1990-1999"},
  {"surface":"around 2008","context_hint":"approximately the year 2008"}
]}

Passage: "Christmas was always cozy growing up, and last Christmas in particular was great."
{"refs":[
  {"surface":"last Christmas","context_hint":"Dec 25 of most recent past year"}
]}

Output a single JSON object: {"refs": [...]}. If none, output {"refs": []}.
"""


# ---------------------------------------------------------------------------
# PASS 2 — resolve a single surface into a temporal envelope
# ---------------------------------------------------------------------------
# Massive simplification vs v1:
# - Output is ONE shape: {earliest, latest, granularity, confidence}.
# - No `kind` field. No nested instant/interval/recurrence containers.
# - No `best` field.
# - Recurrences emit the first known occurrence (the "anchor instant")
#   as the envelope — the pattern is not encoded.
# - Durations: emit the locatable envelope when attached to an anchor;
#   skip if purely unanchored.
PASS2_SYSTEM_V2 = """You resolve ONE temporal reference into a wall-clock envelope.

Reference time is given; use it to resolve all relative expressions.
Compute carefully — check weekday alignment, month length, year rollovers.

# Output

Emit a single JSON object describing the temporal envelope this phrase
refers to:

{
  "surface": string,
  "earliest": ISO-8601 UTC datetime with "Z" suffix (inclusive left edge),
  "latest":   ISO-8601 UTC datetime with "Z" suffix (exclusive right edge),
  "granularity": one of [second, minute, hour, day, week, month, quarter, year, decade, century],
  "confidence": float in [0,1]
}

Every temporal phrase — instant or span, exact or fuzzy, recurring or
once-off — collapses to this envelope. There is no "kind" field; the
shape itself encodes whether the phrase pins a point or a span.

# How to think about earliest / latest

- A pinpoint reference (e.g. "March 15, 2024") becomes a single-day
  envelope: earliest = 2024-03-15T00:00:00Z, latest = 2024-03-16T00:00:00Z.
  The narrowness encodes the precision.
- A span ("Q1 2024") becomes earliest = 2024-01-01T00:00:00Z, latest =
  2024-04-01T00:00:00Z. The width encodes the span.
- A fuzzy reference ("around 2008") widens the envelope to express
  uncertainty: earliest = 2006-01-01T00:00:00Z, latest = 2011-01-01T00:00:00Z
  (year +/- 2). Set granularity to the precision unit ("year" here).
- A relative reference resolves against ref time. "yesterday" -> the
  day before ref; "last month" -> calendar month before ref; "the 90s"
  -> [1990-01-01, 2000-01-01).
- A recurring phrase ("every Thursday at 3pm"): emit the FIRST known
  occurrence's envelope. If the passage anchors the schedule earlier
  (e.g. "I've been seeing them every Thursday since March"), use that
  start. Otherwise pick the nearest past or upcoming Thursday from ref
  time. Recurrence pattern is not used downstream — only the anchor.
- A duration only counts if attached to an anchor ("for 3 weeks
  starting June 1"): emit [anchor, anchor+duration]. Purely unanchored
  durations ("two hours long") with no anchor should not have been
  emitted in pass 1 — return confidence=0.

# Rules

- Use UTC ISO 8601 with "Z" suffix.
- earliest is inclusive, latest is exclusive (a half-open interval).
- For "about" / "around" / "roughly" / "a few" / "a couple": widen by
  one granularity level. Granularity stays at the original precision.
- "the first week of <month> <year>": earliest = first-day-of-month at
  00:00, latest = 7 days later at 00:00, granularity = week.
- For confidence: 1.0 when the phrase pins unambiguously given the
  passage; lower (0.4-0.7) when underspecified ("some time ago"); 0
  when the surface is uninterpretable.

# Refuse to fabricate

If you cannot place the surface on the calendar without falling back
to the reference time as a fabricated anchor, set confidence=0.0 and
emit any placeholder envelope (ref_time as both edges is fine). The
downstream pipeline drops references with confidence below 0.5.

This applies when:
- The surface is a policy / rule constraint with no specific
  occurrence ("backup within the last hour" in a policy).
- The surface is a generic recurrence over an event class with no
  named instance ("every release", "every deploy").
- The surface is a template placeholder ([Date], {date}, YYYY-MM-DD).

In these cases, do not invent an envelope around ref_time — set
confidence=0.0 and let the pipeline drop it.
"""


PASS2_V2_JSON_SCHEMA: dict[str, Any] = {
    "name": "time_envelope",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "surface": {"type": "string"},
            "earliest": {"type": "string"},
            "latest": {"type": "string"},
            "granularity": {
                "type": "string",
                "enum": [
                    "second",
                    "minute",
                    "hour",
                    "day",
                    "week",
                    "month",
                    "quarter",
                    "year",
                    "decade",
                    "century",
                ],
            },
            "confidence": {"type": "number"},
        },
        "required": [
            "surface",
            "earliest",
            "latest",
            "granularity",
            "confidence",
        ],
        "additionalProperties": False,
    },
}


# ---------------------------------------------------------------------------
# Extractor class
# ---------------------------------------------------------------------------
class TemporalExtractorV2:
    """Unified-envelope two-pass extractor.

    Wire into TemporalRetriever via the `extractor` constructor param:

        TemporalRetriever(embed_fn=..., rerank_fn=...,
                          extractor=TemporalExtractorV2())

    Output is a list[TimeExpression] just like v1, but every entry has
    kind="instant" and the FuzzyInstant carries the full envelope. The
    downstream pipeline (filter, mask, recency, scoring) is unchanged.
    """

    def __init__(
        self,
        model: str = MODEL,
        client: AsyncOpenAI | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.model = model
        self.client = client or AsyncOpenAI()
        cd = Path(cache_dir) if cache_dir else CACHE_ROOT
        self.cache = _LLMCache(cd / "pass1_v2.json")
        self.shared_pass2_cache = _LLMCache(cd / "pass2_v2.json")

    async def _pass1(self, text: str, ref_time: datetime) -> list[dict]:
        ctx = full_ref_context(ref_time)
        user = f"{ctx}\n\nPassage:\n{text}"
        key = f"{PROMPT_VERSION}|pass1|{ctx}|||{text}"
        cached = self.cache.get(self.model, key)
        if cached is None:
            resp = await self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": PASS1_SYSTEM_V2},
                    {"role": "user", "content": user},
                ],
                text={"format": {"type": "json_object"}},
            )
            cached = resp.output_text
            self.cache.put(self.model, key, cached)
        try:
            data = json.loads(cached)
            refs = data.get("refs", [])
            if not isinstance(refs, list):
                return []
            return [r for r in refs if isinstance(r, dict) and "surface" in r]
        except (json.JSONDecodeError, AttributeError):
            return []

    async def _pass2(self, surface: str, hint: str, ref_time: datetime) -> dict | None:
        ctx = full_ref_context(ref_time)
        user = (
            f"{ctx}\n\n"
            f"Surface: {surface}\n"
            f"Context hint: {hint}\n\n"
            f"Resolve this surface into a temporal envelope."
        )
        key = f"{PROMPT_VERSION}|pass2|{ctx}|||{surface}|||{hint}"
        cached = self.shared_pass2_cache.get(self.model, key)
        if cached is None:
            resp = await self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": PASS2_SYSTEM_V2},
                    {"role": "user", "content": user},
                ],
                text={"format": {"type": "json_schema", **PASS2_V2_JSON_SCHEMA}},
            )
            cached = resp.output_text
            self.shared_pass2_cache.put(self.model, key, cached)
        try:
            return json.loads(cached)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _to_envelope(env: dict, ref_time: datetime) -> TimeEnvelope | None:
        # NOTE: v2 emits a `confidence` field in JSON but TimeEnvelope
        # no longer carries one (production extractor is v3.2 with
        # skip-don't-emit semantics; see extractor_v3_2_validation.json).
        # We ignore the LLM's confidence value here. v2 is retained for
        # historical comparison only.
        try:
            earliest = parse_iso(env["earliest"])
            latest = parse_iso(env["latest"])
            granularity = env.get("granularity", "day")
            surface = env.get("surface", "")
        except (KeyError, ValueError, TypeError):
            return None
        if latest <= earliest:
            return None
        return TimeEnvelope(
            surface=surface,
            earliest=earliest,
            latest=latest,
            granularity=granularity,
        )

    async def extract(self, text: str, ref_time: datetime) -> list[TimeEnvelope]:
        surfaces = await self._pass1(text, ref_time)
        if not surfaces:
            return []
        envs = await asyncio.gather(
            *(
                self._pass2(s["surface"], s.get("context_hint", ""), ref_time)
                for s in surfaces
            )
        )
        out: list[TimeEnvelope] = []
        for env in envs:
            if env is None:
                continue
            te = self._to_envelope(env, ref_time)
            if te is not None:
                out.append(te)
        return out

    def save_caches(self) -> None:
        """Persist all on-disk caches. Called by TemporalRetriever after indexing."""
        self.cache.save()
        self.shared_pass2_cache.save()
