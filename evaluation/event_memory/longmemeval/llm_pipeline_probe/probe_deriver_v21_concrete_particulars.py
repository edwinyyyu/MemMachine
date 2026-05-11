"""v21 deriver — v20 with segmenter-v9-style "concrete particulars" framing.

User noted the segmenter v9 has a sharper list of nouns/things to keep:

  "Names, places, dates, numbers, identifiers, decisions, plans,
   preferences, relationships, emotional states tied to events, constraints,
   and distinctive phrasing all qualify. Drop generic abstractions or stock
   phrases that would fit many situations."

v20 had the narrower "names, places, dates, amounts, brands, distinctive
concept words" — missing identifiers, decisions, plans, preferences,
relationships, emotional states tied to events, and constraints.

These are all retrieval-relevant. A derivative that drops "decided" or
"prefers" or "feels conflicted about" loses the verb a query might use.

v21 adopts the v9-style "concrete particulars" principle + the richer list,
both in EACH DERIVATIVE (verbatim terms) and in the granularity rule (what
counts as a topic vs separate fact).
"""

from __future__ import annotations

import json
from typing import Any

from dotenv import load_dotenv

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


PROMPT_DERIVER = """\
A SEGMENT is stored verbatim in a retrieval system. Your DERIVATIVES are \
additional strings embedded for semantic search alongside it.

Goal: a future user searching for anything in this segment should find \
it via at least one derivative. Generate the FEWEST derivatives that \
cover every searchable thing — every extra slot wastes index space. \
Splitting a focused thought, paraphrasing one fact, listing items \
already inline, or atomizing a single-topic narrative are FAILURES, not \
coverage. Emit at least one derivative per segment.

How many derivatives? Count the distinct searchable things from a \
future user's perspective, NOT the number of sentences or named \
entities:

- A passage centered on ONE thing — one person, place, event, project, \
process, product, or relationship — even across many sentences, is ONE \
searchable thing → ONE derivative (sometimes 2-3 if too rich for one \
sentence). Co-mentioned particulars (a city it visited, a friend who \
appeared, a brand of a component) are ATTRIBUTES of the central thing, \
not new topics.
- Several genuinely independent particulars a future query could ask \
separately → one derivative per particular.
- A single focused statement (a chess move, a commit message, a \
definition) → exactly ONE near-original derivative. Do NOT split its \
clauses; do NOT paraphrase.
- A bare standalone term (\"Tokyo\") → ONE derivative with just that \
term.
- A list of items sharing one predicate (versions of a model, rows of a \
table) → ONE derivative naming the parent and listing items inline.

EACH DERIVATIVE:
- Is a full grammatical sentence (except bare-entity derivatives).
- Uses the segment's wording verbatim for CONCRETE PARTICULARS — \
anything specific to this segment a future query would mention: names, \
places, dates, numbers, identifiers, decisions, plans, preferences, \
relationships, emotional states tied to events, constraints, and \
distinctive phrasing. Drop generic abstractions or stock phrases that \
would fit many situations.
- Keeps identifying relations atomic: \"my wife Anne\", \"manager \
Sarah\", \"CFO Casey\" — name and role appear TOGETHER, never split.
- Is self-contained: repeat the shared SCOPE (the trip, project, \
period, person) in every sibling derivative whose fact lives under it, \
along with co-actors central to that scope. Per-clause mentions (a \
passerby at one event, a brand in only one clause) stay where they are.
- For heavy acronyms (JFK, POTUS, CMC) or text in a script/language a \
query won't use, include both the original and the expanded/translated \
form in the same derivative.
- Never invents content the segment lacks.

NON-PROSE SURFACES (code, tables, encoded text, JSON, log lines, \
markup): emit one DESCRIPTION derivative — a sentence naming what the \
segment IS and what it is ABOUT; decode recognizable encodings (Caesar, \
base64) into a separate prose derivative; for multi-row tables, \
optionally emit one prose derivative per independently-searchable row. \
Never preserve pipes, code syntax, or brackets.

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
        f"v21 length: {len(PROMPT_DERIVER)} chars / {len(PROMPT_DERIVER.split())} words"
    )
