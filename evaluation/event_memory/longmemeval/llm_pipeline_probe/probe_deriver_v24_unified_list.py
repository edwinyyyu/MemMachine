"""v24 deriver — single unified list of CONCRETE PARTICULARS.

User: "Is there a way to avoid writing two similar lists? Just have one
list with all the signal?"

v22 had two lists:
  - Case-list (what segment can be centered on): person, place, event,
    project, process, product, relationship
  - EACH DERIVATIVE verbatim list: names, places, dates, numbers,
    identifiers, decisions, plans, preferences, relationships, emotional
    states, constraints, distinctive phrasing

These serve different purposes (chunking vs preserving) but the
vocabulary is the same set of "things a query would mention". v24 defines
CONCRETE PARTICULARS once, explicitly states the dual purpose, then both
rules reference it without re-listing.

This is also a maintainability win — adding/removing a category requires
one edit instead of two.
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
Emit at least one derivative per segment.

CONCRETE PARTICULARS in a segment are the things a future query will \
specifically mention: names (people, places, brands, products, \
projects), specific events, dates, numbers, identifiers, decisions, \
plans, preferences, opinions, relationships, emotional states tied to \
events, constraints, and distinctive phrasing. They are what derivatives \
must preserve verbatim AND what counts as the central subject of a \
segment. Generic abstractions or stock phrases (that would fit many \
situations) are NOT particulars — drop them.

How many derivatives?
- A passage centered on ONE concrete particular — even across many \
sentences — is ONE searchable thing → ONE derivative (sometimes 2-3 if \
too rich for one sentence). Co-mentioned particulars are ATTRIBUTES of \
the central one, not new topics to split on.
- Several genuinely independent particulars a future query could ask \
separately → one derivative per particular.
- A single focused statement (a chess move, a commit, a definition) → \
exactly ONE near-original derivative. Do NOT split its clauses; do NOT \
paraphrase.
- A bare standalone particular (\"Tokyo\") → ONE derivative with just \
that term.
- A list of items sharing one predicate (versions of a model, rows of a \
table) → ONE derivative naming the parent and listing items inline.

EACH DERIVATIVE:
- Is a full grammatical sentence (except bare-particular derivatives).
- Preserves the segment's wording verbatim for the concrete particulars \
it covers.
- Keeps identifying relations atomic: \"my wife Anne\", \"manager \
Sarah\", \"CFO Casey\" — name and role appear TOGETHER, never split.
- Is self-contained: repeat the shared SCOPE (the trip, project, \
period, person) in every sibling derivative whose fact lives under it, \
along with co-actors central to that scope. Per-clause mentions (a \
passerby at one event, a brand in only one clause) stay where they are.
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
        f"v24 length: {len(PROMPT_DERIVER)} chars / {len(PROMPT_DERIVER.split())} words"
    )
