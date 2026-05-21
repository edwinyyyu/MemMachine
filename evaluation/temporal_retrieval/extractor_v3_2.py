"""V3.2 extractor: skip-don't-emit for unresolvable phrases, no confidence field.

V3.1 emits unresolvable phrases with confidence=0 and a placeholder
envelope, relying on the downstream confidence_floor=0.5 to filter
them out. This is essentially "emit then filter".

V3.2 inverts the contract: the LLM is told "if you can't anchor the
phrase, do NOT emit it." There's no confidence field on the output
schema. Every emitted envelope is treated as fully usable. The
retriever's `confidence_floor` becomes a no-op (every TimeEnvelope
from this extractor has confidence=1.0 by construction).

Open question: does losing the conf=0.4-0.7 band (currently kept by
v3.1+floor=0.5) cost anything? V3.2's prompt asks the LLM to make a
sharper "anchor / skip" decision; underspecified phrases that v3.1
emitted with mid-confidence may either be (a) emitted with the same
envelope under v3.2 (LLM commits to a plausible interpretation) or
(b) skipped entirely. Bench result decides whether the simplification
is a wash, a win, or a loss.

Cache directory: cache/temporal_retrieval_v3_2/.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

from .extractor_common import _LLMCache, full_ref_context
from .schema import TimeEnvelope, parse_iso

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-5-mini"
PROMPT_VERSION = "v3_2"
CACHE_ROOT = (
    Path(__file__).resolve().parent / "cache" / f"temporal_retrieval_{PROMPT_VERSION}"
)
CACHE_ROOT.mkdir(parents=True, exist_ok=True)


SINGLE_PASS_SYSTEM_V3_2 = """You are a temporal-reference extractor.

Your job: identify EVERY span in a passage that names a specific point,
span, or recurring schedule of time, AND directly resolve each one
into a temporal envelope (a half-open interval on the calendar).

# Critical test before emitting anything

Does the passage USE this phrase to locate a specific occurrence on
the calendar — one that the reader could later recall, search for,
or reference by date? Or does the phrase describe a constraint, a
requirement, a format placeholder, or a rule that applies generally
across many possible occurrences?

- Specific occurrence ("yesterday we deployed", "shipped on March 15",
  "Q1 was rough", "during the pandemic", "every Thursday at 3pm I
  have therapy") -> EMIT.
- Constraint / rule / placeholder ("Policy: backups within the last
  hour", "every release requires a 30-minute window", "Subject
  format: [Date]") -> SKIP, even if temporal-shaped.

This is the deciding test for borderline cases. The retriever's job
is to surface timeless rule docs on non-temporal queries; anchoring
a policy at the reference time defeats that.

# What counts as a temporal reference

A span is a temporal reference if and only if, given the reference
time and any explicit anchoring in the passage, you could state WHEN
it is on a calendar — AND the critical test above puts it on the
"specific occurrence" side:

- Absolute dates: "March 5, 2026", "1986", "Q3 2025".
- Relative deictics: "yesterday", "2 weeks ago", "next Thursday".
- Approximations: "around 2010", "a few weeks ago", "recently".
- Eras with a calendar anchor: "the 90s", "back in college", "during
  the pandemic".
- Recurring schedules tied to a real standing pattern: "every
  Thursday at 3pm". Emit the first/nearest known occurrence.
- Durations: emit ONLY if attached to a specific calendar anchor.
  In particular, IMPACT-MAGNITUDE durations describe how long an
  effect lasted, not when it was on the calendar — skip them.
  Examples: "over-reported for 6 weeks", "froze for 12 minutes",
  "delayed 3 hours", "outage lasted 45 minutes". These are
  measurements of an event's effect, not retrievable time anchors;
  even when an anchor exists nearby (e.g. "Postmortem from May 12:
  ... for 6 weeks"), the duration itself isn't a separately
  retrievable reference — the anchor ("May 12") is.

# What does NOT count (skip)

- Bare names of recurring annual events without a year-anchor:
  "summer", "Christmas", "Easter", "graduation day". (EXCEPTION:
  when the phrase IS the recurring schedule itself in a
  standing-arrangement context — "every summer we visit the lake".)
- Vague descriptors: "recent", "modern", "old", "new", "ancient".
- Bare frequency words: "often", "always", "sometimes", "rarely".
- Bare approximators without concrete reference: "about", "around",
  "roughly" used alone.

# Policy / rule / template contexts — skip everything inside

When the surrounding sentence describes a generic policy, rule,
convention, requirement, or format, even temporal-shaped phrases
inside it are CONSTRAINTS or PLACEHOLDERS, not events. Cue patterns:

- Explicit policy header: "policy:", "convention:", "rule:",
  "guideline:", "standard:".
- Prescriptive modals as main predicate: "must X", "should X",
  "requires X", "never X without Y", "always X before Y".
- Recurrence over an event-CLASS without naming a specific instance:
  "every release", "every deploy", "every PR", "every sprint".
- Template placeholders: "[Date]", "{date}", "<date>", "YYYY-MM-DD".

If you skip on these cues, do not silently emit other temporal
phrases from the same sentence either; the whole sentence is
policy/rule content.

# How to think about earliest / latest

- A pinpoint reference (e.g. "March 15, 2024") -> single-day
  envelope: earliest = 2024-03-15T00:00:00Z, latest = 2024-03-16T00:00:00Z.
- A span ("Q1 2024") -> earliest = 2024-01-01T00:00:00Z, latest =
  2024-04-01T00:00:00Z.
- A fuzzy reference ("around 2008") -> widen by one unit: earliest
  = 2006-01-01T00:00:00Z, latest = 2011-01-01T00:00:00Z. Granularity
  stays at the original precision ("year" here).
- A relative reference resolves against ref time. "yesterday" ->
  day before ref. "last month" -> calendar month before ref. "the
  90s" -> [1990-01-01, 2000-01-01).
- A recurring phrase ("every Thursday at 3pm"): emit FIRST known
  occurrence. If the passage anchors the schedule earlier ("every
  Thursday since March"), use that start. Otherwise pick the
  nearest past/upcoming occurrence from ref time.
- A duration only counts if attached to an anchor ("for 3 weeks
  starting June 1") -> [anchor, anchor+duration].

# Rules

- Use UTC ISO 8601 with "Z" suffix.
- earliest is inclusive, latest is exclusive (half-open).
- For "about" / "around" / "roughly" / "a few" / "a couple", widen by
  one granularity level.
- Granularity is one of: second, minute, hour, day, week, month,
  quarter, year, decade, century.

# Skip — do not emit

If you cannot place a surface on the calendar without falling back
to ref time as a fabricated anchor, DO NOT emit it. Omit the
reference from your output entirely.

This applies to:
- Policy / rule constraints with no specific occurrence.
- Generic recurrences over an event class with no named instance.
- Template placeholders.
- Phrases that look temporal but lack a calendar anchor (e.g.,
  bare "the launch" or "grad school" without other context).

In those cases, do not invent an envelope around ref_time — just
skip the phrase. The downstream retriever falls back to semantic
cosine when no envelope is emitted.

# Output

A single JSON object: {"refs": [...]}. Each ref is:
{
  "surface": exact verbatim substring from the passage,
  "earliest": ISO-8601 UTC datetime with "Z",
  "latest":   ISO-8601 UTC datetime with "Z",
  "granularity": one of the granularity values above
}

If the passage has no temporal references that meet the bar, output
{"refs": []}.
"""


V3_2_JSON_SCHEMA: dict[str, Any] = {
    "name": "time_envelopes",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "refs": {
                "type": "array",
                "items": {
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
                    },
                    "required": [
                        "surface",
                        "earliest",
                        "latest",
                        "granularity",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["refs"],
        "additionalProperties": False,
    },
}


class TemporalExtractorV3_2:
    """Skip-don't-emit unified-envelope extractor (v3.2).

    Wire via TemporalRetriever's `extractor` constructor param.
    Every emitted TimeEnvelope has confidence=1.0 — the LLM's
    decision to emit replaces the confidence-floor filter.
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
        self.cache = _LLMCache(cd / "single_v3_2.json")
        self.shared_pass2_cache = self.cache

    async def _call(self, text: str, ref_time: datetime) -> list[dict]:
        ctx = full_ref_context(ref_time)
        user = f"{ctx}\n\nPassage:\n{text}"
        key = f"{PROMPT_VERSION}|single|{ctx}|||{text}"
        cached = self.cache.get(self.model, key)
        if cached is None:
            resp = await self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": SINGLE_PASS_SYSTEM_V3_2},
                    {"role": "user", "content": user},
                ],
                text={"format": {"type": "json_schema", **V3_2_JSON_SCHEMA}},
            )
            cached = resp.output_text
            self.cache.put(self.model, key, cached)
        try:
            data = json.loads(cached)
            refs = data.get("refs", [])
            if not isinstance(refs, list):
                return []
            return refs
        except (json.JSONDecodeError, AttributeError):
            return []

    @staticmethod
    def _to_envelope(env: dict, ref_time: datetime) -> TimeEnvelope | None:
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
        envs = await self._call(text, ref_time)
        out: list[TimeEnvelope] = []
        for env in envs:
            te = self._to_envelope(env, ref_time)
            if te is not None:
                out.append(te)
        return out

    def save_caches(self) -> None:
        self.cache.save()
