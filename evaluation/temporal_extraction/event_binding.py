"""E1 — Event-time binding extractor.

For each document, issue a single gpt-5-mini structured-output call that
returns a list of (event_span, time_expr_surface) pairs. Then embed each
event_span with text-embedding-3-small. Store the earliest/latest/best
microseconds per binding for temporal overlap scoring.

event_span = concrete noun phrase describing what happened. If the time
expression has no associated concrete event, event_span is null, which
makes the binding act as a bare time (semantic match = 0).
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from advanced_common import LLMCaller

SYSTEM = """You extract (event, time) pairs from passages.

For each temporal reference you find in the passage, return:
- time_surface: the exact time-expression substring, verbatim.
- event_span: a concrete noun phrase describing what happened at that
  time. Must be 2-8 words, taken mostly from the passage, but you MAY
  rephrase lightly so it is self-contained (e.g., "visited the aquarium"
  rather than "I visited"). If no concrete event is associated with this
  time expression (e.g., "On March 15 the weather was nice"), set
  event_span to null.

Multiple time expressions with distinct events become distinct pairs. A
single event that spans an interval produces ONE pair.

Output a single JSON object: {"pairs": [{"time_surface": "...",
"event_span": "..." | null}, ...]}. If no temporal references, output
{"pairs": []}.
"""

PASS2_SYSTEM = """You resolve a single time expression into an absolute UTC bracket.

Given a reference time and a surface string, output JSON:
{"earliest": ISO, "latest": ISO, "best": ISO, "granularity": one of
[second,minute,hour,day,week,month,quarter,year,decade,century]}.

Use UTC ISO 8601 with trailing Z. earliest inclusive, latest exclusive.
For fuzzy expressions ("around 2019", "a decade ago"), widen by one
granularity level. best is the centered point estimate.
"""

PAIR_SCHEMA = {
    "name": "event_time_pairs",
    "strict": False,
    "schema": {
        "type": "object",
        "properties": {
            "pairs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "time_surface": {"type": "string"},
                        "event_span": {"type": ["string", "null"]},
                    },
                    "required": ["time_surface"],
                },
            }
        },
        "required": ["pairs"],
    },
}

RESOLVE_SCHEMA = {
    "name": "time_resolution",
    "strict": False,
    "schema": {
        "type": "object",
        "properties": {
            "earliest": {"type": "string"},
            "latest": {"type": "string"},
            "best": {"type": "string"},
            "granularity": {"type": "string"},
        },
        "required": ["earliest", "latest", "granularity"],
    },
}


async def extract_pairs(
    llm: LLMCaller, text: str, ref_time: datetime
) -> list[dict[str, Any]]:
    """Returns list of raw {'time_surface','event_span'} dicts."""
    iso_ref = ref_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    wk = ref_time.strftime("%A")
    user = (
        f"Reference time: {iso_ref} ({wk})\n"
        f"Passage:\n{text}\n\n"
        'Return {"pairs": [...]}.'
    )
    raw = await llm.chat(
        SYSTEM,
        user,
        json_schema=PAIR_SCHEMA,
        max_completion_tokens=1500,
        cache_tag="e1_pairs",
    )
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return list(d.get("pairs") or [])


async def resolve_time(
    llm: LLMCaller, surface: str, ref_time: datetime, surrounding: str
) -> dict[str, Any] | None:
    iso_ref = ref_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    wk = ref_time.strftime("%A")
    user = (
        f"Reference time: {iso_ref} ({wk})\n"
        f"Surrounding context: {surrounding}\n"
        f'Time expression: "{surface}"\n\n'
        "Return JSON with earliest/latest/best/granularity."
    )
    raw = await llm.chat(
        PASS2_SYSTEM,
        user,
        json_schema=RESOLVE_SCHEMA,
        max_completion_tokens=2000,
        cache_tag="e1_resolve",
    )
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if "earliest" not in d or "latest" not in d:
        return None
    return d
