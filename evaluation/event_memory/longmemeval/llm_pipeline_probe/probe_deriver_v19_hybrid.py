"""v19 deriver probe — v18 calibration framing + v16 explicit backstops.

v17 (1815c, 47%) failed because principle-only is too permissive.
v18 (2458c, 64%) added user's calibration principle but lost v16's
explicit anti-atomization and acronym rules. A4 acronyms went 6/6 FAIL.

v19 keeps v18's user-motivated calibration framing (topical narrative
vs specific data points) as the PRIMARY signal, but reintroduces the
explicit clauses that turned out to be load-bearing:
  - "Co-mentioned attributes are NOT new topics" (anti-atomization)
  - Explicit acronym/translation rule
  - Explicit binding-atomic rule

Goal: match v16's 91.7% qualitative pass rate at shorter length.
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
one derivative (or the segment itself) should match via cosine \
similarity. Two derivatives that match the same query are wasted; a \
missing derivative for a plausible query is a coverage gap. Emit at \
least one derivative per segment.

CALIBRATE GRANULARITY by what kind of query is likely. The simpler the \
query, the more cohesive the derivative can be; the more complex the \
query (synthesis, multi-hop, comparison), the simpler each derivative \
must be to avoid distractors.

- If the segment is a TOPICAL NARRATIVE (a profile, description, event \
chain, or single-subject prose across many sentences) — likely queries \
are simple keyword lookups (\"tell me about X\"). Emit 1-3 COHESIVE \
derivatives that bundle multiple sentences. Co-mentioned things (a city \
it visited, a friend who appeared, a brand of a component) are \
ATTRIBUTES of the topic, NOT new topics — do not split them into \
separate derivatives.
- If the segment contains SPECIFIC DATA POINTS likely to be combined \
across queries (dates, durations, prices, amounts, identifiers, version \
numbers, comparable attributes) — emit each as a STANDALONE short \
derivative. Multi-hop and synthesis queries need each fact retrievable \
without distractors.
- A single focused statement (one chess move, one commit, one \
definition) → exactly ONE near-original derivative. Do not split its \
clauses; do not paraphrase.
- A bare standalone term (\"Tokyo\") → ONE derivative containing just \
that term.

EACH DERIVATIVE:
- Reads like a natural full sentence (except bare-entity).
- Uses the segment's wording verbatim for identifying terms: names, \
places, dates, amounts, brands, distinctive concept words.
- IDENTIFYING RELATIONS are atomic phrases. \"my wife Anne\", \"manager \
Sarah\", \"CFO Casey\" — name and role appear TOGETHER, never split \
across derivatives.
- Repeats any context a plausible query would mention. The lodging fact \
on a Tokyo trip with Anne: include both \"Tokyo\" and \"Anne\". The \
fact under a project: include the project name.
- For heavy acronyms (JFK, POTUS, CMC) or text in a script/language a \
query won't use: include both the original and the expanded/translated \
form in the same derivative.
- Never invents content the segment lacks.

NON-PROSE SURFACES (code, tables, encoded text, JSON, log lines, \
markup): emit one description derivative naming what the segment IS and \
what it is ABOUT; decode recognizable encodings (Caesar, base64) into a \
separate prose derivative; for multi-row tables, optionally emit one \
prose derivative per independently-searchable row. Never preserve \
pipes, code syntax, or brackets.

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
        f"v19 length: {len(PROMPT_DERIVER)} chars / {len(PROMPT_DERIVER.split())} words"
    )
