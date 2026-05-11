"""v18 deriver probe — query-complexity calibrated granularity.

User principle: "the simpler the query, the more complex the derivative can
be; the more complex the query (synthesis, multi-hop, ambiguous task
completion), the simpler the derivative must be to avoid distractors."

The deriver doesn't know future queries directly, but it can infer from
the segment's CONTENT what kinds of queries are likely:

  - TOPICAL NARRATIVES (single-subject prose, profiles, descriptions, event
    chains) tend to be asked about via simple keyword lookup ("tell me
    about the Mustang restoration"). Bundle into 1-3 cohesive derivatives —
    atomizing scatters the topic across weak matches.

  - SPECIFIC DATA POINTS (dates, durations, amounts, prices, identifiers,
    quantitative attributes, version numbers) tend to be combined across
    queries in multi-hop or synthesis ("how many days between X and Y",
    "what's my plan vs my old one"). Emit each as a standalone short
    derivative so each fact is independently retrievable without
    distractors.

This subsumes v16's case-list while making the principle objective.

v17 failed (47.2%) because its principle-only framing ("answer a distinct
query") had no anti-atomization signal and let the model invent too many
distinct queries. v18 adds the calibration signal explicitly.

Also keep v16's strongest rules:
  - Identifying relations are atomic ("my wife Anne")
  - Self-containment includes scope + co-actors a query would mention
"""

from __future__ import annotations

import json
from typing import Any

from dotenv import load_dotenv

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


PROMPT_DERIVER = """\
A SEGMENT is stored verbatim in a retrieval system. Your DERIVATIVES are \
additional strings embedded for semantic search alongside it.

Goal: for any future natural-language query about this segment, at least \
one derivative (or the segment itself) should match via cosine similarity. \
Two derivatives that match the same query are wasted; a missing \
derivative for a plausible query is a coverage gap. Emit at least one \
derivative per segment.

CALIBRATE GRANULARITY by what kind of query is likely:

- If the segment is a TOPICAL NARRATIVE — a profile, description, event \
chain, or single-subject prose across many sentences — likely queries are \
simple keyword lookups (\"tell me about X\"). Emit 1-3 COHESIVE \
derivatives that bundle multiple sentences. Atomizing scatters the topic \
into weak matches.
- If the segment contains SPECIFIC DATA POINTS likely to be combined \
across queries (dates, durations, amounts, prices, identifiers, version \
numbers, quantitative attributes) — emit each as a STANDALONE short \
derivative. Multi-hop and synthesis queries need each fact retrievable \
without distractors from neighboring facts.
- A single focused statement (one chess move, one commit, one \
definition) → ONE near-original derivative.
- A bare standalone term (\"Tokyo\") → ONE derivative with just that term.

EACH DERIVATIVE:
- Reads like a natural full sentence (except bare-entity derivatives).
- Uses the segment's wording verbatim for identifying terms: names, \
places, dates, amounts, brands, distinctive concept words, technical \
terms.
- Keeps IDENTIFYING RELATIONS atomic: \"my wife Anne\", \"manager Sarah\", \
\"CFO Casey\" — never split the name from its role across derivatives.
- Includes everything a plausible query for this derivative would \
mention. If the lodging fact lives under a Tokyo trip with Anne, the \
lodging derivative repeats both \"Tokyo\" and \"Anne\". If a query might \
say \"Cuban Missile Crisis\" but the segment says \"CMC\" — include both.
- Never invents content the segment lacks.

NON-PROSE SURFACES (code, tables, encoded text, JSON, log lines, markup): \
emit one description derivative naming what the segment IS and what it \
is ABOUT; decode recognizable encodings (Caesar, base64) into a separate \
prose derivative; for multi-row tables, optionally emit one prose \
derivative per independently-searchable row. Never preserve pipes, code \
syntax, or brackets.

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
        f"v18 length: {len(PROMPT_DERIVER)} chars / {len(PROMPT_DERIVER.split())} words"
    )
