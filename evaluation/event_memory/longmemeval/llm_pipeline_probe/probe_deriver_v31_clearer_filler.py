"""v31 deriver — v29 with clearer filler-paragraph structure.

User noted v29's filler paragraph ends with "Otherwise emit at least one
derivative per segment" — the "Otherwise" is dangling and unclear.

v31 restructures:
  - State the filler rule cleanly (filler → 0)
  - Sub-exception (content-bearing affirmative → emit for content)
  - Closing clause refers explicitly to "all other segments"

This is a wording fix only — same behavior intended as v29.
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
coverage.

PURE FILLER SEGMENTS — content-free reactions, affirmations, or chat \
plumbing only meaningful with the prior message — emit ZERO \
derivatives. Examples: \"yes\", \"no\", \"ok\", \"sounds good\", \"will \
do\", \"thanks\", \"lol\", \"great point\", \"omg yes!!\", \
\"acknowledged\". They carry no searchable content on their own. An \
affirmative that ALSO carries concrete content (\"ok, leaving Tuesday \
at 5\", \"yes, I'll bring the wine\") is NOT pure filler — emit \
derivatives for the content. All other segments get at least one \
derivative.

How many derivatives? Count the distinct searchable things from a \
future user's perspective, NOT the number of sentences or named \
entities:

- A passage centered on ONE thing — a person, place, event, project, \
process, product, or relationship — even across many sentences, is ONE \
searchable thing → ONE derivative. Co-mentioned particulars (a city it \
visited, a friend who appeared, a brand of a component) are ATTRIBUTES \
of the central thing, not new topics.
- Several genuinely independent particulars a future query could ask \
separately → one derivative per particular.
- A single focused statement (a function signature, a commit message, a \
definition) → exactly ONE near-original derivative. Do NOT split its \
clauses; do NOT paraphrase.
- A bare standalone term (\"Paris\") → ONE derivative with just that \
term.
- A list of items sharing one predicate (versions of a model, rows of a \
table) → ONE derivative naming the parent and listing items inline.

EACH DERIVATIVE:
- Is a full grammatical sentence (except bare-entity), using the \
segment's wording verbatim for CONCRETE PARTICULARS — names, places, \
dates, numbers, identifiers, decisions, plans, preferences, opinions, \
relationships, emotional states tied to events, constraints, and \
distinctive phrasing. Drop generic abstractions or stock phrases.
- Keeps identifying relations atomic: \"my mentor Alice\", \"team lead \
Bob\", \"CTO Charlie\" — name and role appear TOGETHER, never split.
- Is self-contained: repeat the shared scope (the trip, project, period, \
person, artifact) and any co-actors central to that scope in every \
sibling derivative that depends on them.
- Never invents content the segment lacks.

NON-PROSE SURFACES (code, tables, encoded text, JSON, log lines, \
markup): emit one DESCRIPTION derivative naming what the segment IS and \
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
        f"v31 length: {len(PROMPT_DERIVER)} chars / {len(PROMPT_DERIVER.split())} words"
    )
