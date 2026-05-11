"""v20 deriver probe — v16 with minimal cosmetic cleanups.

After v17/v18/v19 all regressed when restructuring v16's content, switching
to a different approach: keep v16's structure and rules verbatim, just
polish the wording where the user flagged it.

Cosmetic changes vs v16:
  - Reorder "one product, place, person, project, process, event" to
    a more intuitive sequence
  - Tighten the EACH DERIVATIVE self-contained sentence
  - Condense the HEAVY ACRONYMS section into one inline sentence
  - Drop the "(occasionally 2-3 if the topic is rich)" hedge — the model
    will exercise judgment without it
  - Remove the "Per-item derivatives waste slots because all items share
    the same future query" explanation (the rule is enough)

Goal: same rules and qualitative behavior as v16, fewer chars.
v16: 3874 chars. Target: ~3000-3200 chars.
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

- A passage about ONE topic — one person, place, event, project, \
process, or product — even across many sentences, is ONE searchable \
thing → ONE derivative (sometimes 2-3 if too rich for one sentence). \
Co-mentioned names (a city it visited, a friend who appeared, a brand \
of a component) are ATTRIBUTES of the topic, not new topics to split on.
- Several genuinely independent facts a future query could ask \
separately → one derivative per fact.
- A single focused statement (a chess move, a commit message, a \
definition) → exactly ONE near-original derivative. Do NOT split its \
clauses; do NOT paraphrase.
- A bare standalone term (\"Tokyo\") → ONE derivative with just that \
term.
- A list of items sharing one predicate (versions of a model, rows of a \
table) → ONE derivative naming the parent and listing items inline.

EACH DERIVATIVE:
- Is a full grammatical sentence (except bare-entity derivatives).
- Uses the segment's wording verbatim for identifying terms: names, \
places, dates, amounts, brands, distinctive concept words.
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
        f"v20 length: {len(PROMPT_DERIVER)} chars / {len(PROMPT_DERIVER.split())} words"
    )
