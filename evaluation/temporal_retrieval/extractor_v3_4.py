"""V3.4 extractor: adds `claim_type` (event | state) per reference.

Builds on V3.3 (which dropped `surface` and `granularity`). The new
field captures a semantic distinction the temporal layer can't recover
from intervals alone:

  - "event":  something HAPPENED at this point/period
  - "state":  something was TRUE THROUGHOUT this span (covers habits
              and continuing conditions — both reduce to query-fraction
              scoring, so habits fold into state)

Downstream scoring dispatches by claim_type:
  - event: doc-fraction `|q∩d|/|d|` rewards specificity inside query
  - state: query-fraction `|q∩d|/|q|` rewards coverage of query by
           the state span

Cache directory: cache/temporal_retrieval_v3_4/.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

from .extractor_common import _LLMCache, full_ref_context
from .schema import parse_iso

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-5-mini"
PROMPT_VERSION = "v3_4b"  # bump: simplified 3 types → 2 (event/state); cache invalidates
CACHE_ROOT = (
    Path(__file__).resolve().parent / "cache" / f"temporal_retrieval_{PROMPT_VERSION}"
)
CACHE_ROOT.mkdir(parents=True, exist_ok=True)


SINGLE_PASS_SYSTEM_V3_4 = """You are a temporal-reference extractor.

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
  = 2006-01-01T00:00:00Z, latest = 2011-01-01T00:00:00Z.
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
  one granularity level (day -> week, month -> year, etc.).

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
skip the phrase.

# Claim type — emit one per ref

Each emitted ref must include `claim_type`, classifying what the
passage says about that span:

- "event": something HAPPENED at this point/period — an action,
  occurrence, transition, or one-off happening. Examples:
    "I joined TechCorp in June 2012"            → event (joined)
    "Saw fireworks on July 4, 2025"             → event (saw)
    "Italy trip July 2018"                      → event (the trip)
    "Got married July 14, 2016"                 → event (got married)
    "Bought a house in 2019"                    → event (bought)
    "Yesterday I deployed the migration"        → event (deployed)
    "Released v5 on March 15, 2024"             → event (released)

- "state": something was TRUE THROUGHOUT this span — a continuing
  condition, ongoing activity, or recurring pattern that held across
  the duration. This includes habits ("every X") and conditions
  ("was at Initech") that pervade their span. Examples:
    "Worked at Initech from April 2022 through 2025"  → state
    "Throughout 2024 I was studying for the bar"      → state
    "Lived in California from 2018 onward"            → state
    "Lockdown was March to June 2020"                 → state
    "Owned the Honda from 2015 until 2021"            → state
    "Was on the strategy rotation all of Q1 2024"     → state
    "Every weekday I ate oatmeal in 2024"             → state
    "Yoga every Tuesday since fall 2019"              → state
    "Each December our family gathers"                → state
    "Met with my mentor monthly throughout 2023"      → state

# Deciding event vs state

When in doubt, ask: does the passage tell us what HAPPENED AT this
time (event) or what was TRUE FOR the duration (state)?

  - An event OCCUPIES its window (one action, then it's over).
  - A state PERVADES its window (the condition holds throughout, or
    recurs regularly across).

Stylistically:
  - Events use action verbs ("went", "shipped", "bought", "joined").
  - States use copula/stative verbs ("was", "lived", "worked",
    "owned"), duration framing ("throughout", "from X through Y",
    "since", "until"), or distributive markers ("every", "each").

A single passage can produce multiple refs of mixed types — emit each
with its own claim_type. Don't split a single state/habit phrase into
both an event and a state.

# Output

A single JSON object: {"refs": [...]}. Each ref is:
{
  "earliest":   ISO-8601 UTC datetime with "Z",
  "latest":     ISO-8601 UTC datetime with "Z",
  "claim_type": "event" | "state" | "habit"
}

If the passage has no temporal references that meet the bar, output
{"refs": []}.
"""


V3_4_JSON_SCHEMA: dict[str, Any] = {
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
                        "earliest": {"type": "string"},
                        "latest": {"type": "string"},
                        "claim_type": {
                            "type": "string",
                            "enum": ["event", "state"],
                        },
                    },
                    "required": ["earliest", "latest", "claim_type"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["refs"],
        "additionalProperties": False,
    },
}


class TemporalExtractorV3_4:
    """v3.4 extractor emits claim_type per ref alongside earliest/latest.

    Returns `list[dict]` from `extract()` — each entry has keys
    `earliest_us`, `latest_us`, `claim_type` — rather than the bare
    TimeEnvelope used by v3.3, so downstream scoring can dispatch on
    type.
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
        self.cache = _LLMCache(cd / "single_v3_4.json")
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
                    {"role": "system", "content": SINGLE_PASS_SYSTEM_V3_4},
                    {"role": "user", "content": user},
                ],
                text={"format": {"type": "json_schema", **V3_4_JSON_SCHEMA}},
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
    def _to_dict(env: dict, ref_time: datetime) -> dict | None:
        from .schema import to_us
        try:
            earliest = parse_iso(env["earliest"])
            latest = parse_iso(env["latest"])
            claim_type = env.get("claim_type", "event")
        except (KeyError, ValueError, TypeError):
            return None
        if latest <= earliest:
            return None
        if claim_type not in ("event", "state"):
            claim_type = "event"
        return {
            "earliest_us": to_us(earliest),
            "latest_us": to_us(latest),
            "claim_type": claim_type,
        }

    async def extract(self, text: str, ref_time: datetime) -> list[dict]:
        envs = await self._call(text, ref_time)
        out: list[dict] = []
        for env in envs:
            d = self._to_dict(env, ref_time)
            if d is not None:
                out.append(d)
        return out

    def save_caches(self) -> None:
        self.cache.save()
