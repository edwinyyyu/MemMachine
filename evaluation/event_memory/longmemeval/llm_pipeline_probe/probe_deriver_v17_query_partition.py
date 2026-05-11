"""v17 deriver probe — reframe around a single objective principle.

User feedback on v16:
  - Enumerated case-list ("product, place, person, project, process, event"
    + 5 granularity rules) is unintuitive and requires balancing priorities.
  - "Scope vs per-clause attribute" wording is complex.
  - Heavy-acronyms rule doesn't pull much weight as its own section.
  - Prefer shorter still.

v17 reframes around ONE principle: each derivative answers a DISTINCT plausible
future query. Two derivatives matching the same query are wasted; a missing
derivative for a plausible query is a coverage gap.

All earlier rules become natural consequences:
  - Granularity = how many distinct queries does this segment answer?
  - Self-containment = include what a plausible query would mention
    (this subsumes scope propagation, co-actors, AND acronym expansion)
  - Anti-fragmentation = single-topic narrative answers fewer queries, not
    one per sentence
  - No near-clones = two derivs answering the same query are wasted

Special cases that don't follow from the principle (bare entity, non-prose
surface) get a short tail section.
"""

from __future__ import annotations

import json
from typing import Any

from dotenv import load_dotenv

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


PROMPT_DERIVER = """\
A SEGMENT is stored verbatim in a retrieval system. Your DERIVATIVES are \
additional strings embedded for semantic search alongside it.

Goal: for any plausible natural-language query about this segment, at \
least one derivative (or the segment itself) should match via cosine \
similarity.

Each derivative answers a DISTINCT plausible query a future user might \
write. Two derivatives matching the same query are wasted; a missing \
derivative for a plausible query is a coverage gap. Generate the \
smallest set that achieves coverage. Emit at least one derivative per \
segment.

EACH DERIVATIVE:
- Reads like a natural full sentence a person could write — no \
fragments, no formatting artifacts.
- Uses the segment's wording verbatim for identifying terms: names, \
places, dates, amounts, distinctive concept words, technical terms.
- Includes everything a plausible query for this derivative would \
mention. If the query might say \"Anne\" or \"my wife\" or both — \
include both forms. If the query might use \"Cuban Missile Crisis\" but \
the segment says \"CMC\" — include both. If the query for \"where did \
Anne stay\" needs Anne, the lodging derivative repeats Anne.
- Never invents content the segment lacks.

SPECIAL CASES:
- A bare standalone term (a single noun like \"Tokyo\") → one \
derivative containing just that term.
- Non-prose surfaces (code, tables, encoded text, JSON, log lines, \
markup, lists of raw numbers): emit one description derivative — a \
sentence naming what the segment IS and what it is ABOUT — plus, \
optionally, one prose derivative per independently-searchable row of a \
multi-row table. Decode recognizable encodings (Caesar, base64) into \
prose, in a separate derivative. Never preserve pipes, code syntax, or \
brackets.

Output: a JSON object {{ \"derivatives\": [...] }} and nothing else.

SEGMENT:
{segment}
"""


DERIVATIVES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["derivatives"],
    "properties": {"derivatives": {"type": "array", "items": {"type": "string"}}},
}


async def derive(
    client, segment, *, prompt=PROMPT_DERIVER, model="gpt-5.4-nano", reasoning="medium"
):
    kwargs: dict[str, Any] = {
        "model": model,
        "input": prompt.format(segment=segment),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "derivatives",
                "schema": DERIVATIVES_SCHEMA,
                "strict": True,
            }
        },
    }
    if reasoning:
        kwargs["reasoning"] = {"effort": reasoning}
    resp = await client.responses.create(**kwargs)
    payload = json.loads(resp.output_text)
    return list(payload.get("derivatives", []))


if __name__ == "__main__":
    print(
        f"v17 length: {len(PROMPT_DERIVER)} chars / {len(PROMPT_DERIVER.split())} words"
    )
