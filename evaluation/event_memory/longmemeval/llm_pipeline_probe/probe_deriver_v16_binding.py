"""v16 deriver probe — v14c goal-first + explicit binding preservation.

User observation: atomization risks losing identifying relations. E.g.,
"my wife Anne" can split into "my wife" (one deriv) and "Anne" (another),
or scope-propagation can carry "trip" but drop "Anne" — so the lodging
derivative loses that Anne was on the trip.

v14c had: "Per-clause attributes (a brand or person mentioned in only
one clause) stay where they are." This is WRONG for "wife Anne": Anne is
part of the trip's SCOPE, not a per-clause attribute. The lodging fact
implicitly includes Anne (she was on the trip).

v16 refinement:
  (a) Entity-relation phrases ("my wife Anne", "manager Sarah", "CFO
      Casey") are ATOMIC — never split across derivatives.
  (b) Co-actors who belong to the SCOPE (the trip, project, period)
      travel with the scope, not just stay in one clause.
  (c) Truly per-clause mentions (Marcus appearing at one car-show event)
      stay where they are.
"""

from __future__ import annotations

import json
from typing import Any

from dotenv import load_dotenv

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


PROMPT_DERIVER = """\
A SEGMENT is stored verbatim in a retrieval system. Your DERIVATIVES are \
additional strings that get embedded for semantic search alongside it.

Goal: a future user searching for anything in this segment should find \
it via cosine similarity to at least one derivative. Generate the FEWEST \
derivatives that cover every searchable thing in the segment — every \
extra derivative slot wastes index space and dilutes ranking. Splitting \
a focused thought, paraphrasing the same fact, listing items already \
inline, or atomizing a single-topic narrative are FAILURES, not coverage. \
Emit at least one derivative per segment.

How many derivatives? Count the distinct searchable things from the \
user's perspective, NOT the sentences or named entities:

  - A passage about ONE topic — one product, place, person, project, \
process, event — even across many sentences, is ONE searchable thing. \
Output ONE derivative (occasionally 2-3 if the topic is rich and one \
sentence cannot hold it). Co-mentioned things (a city it visited, a \
friend who appeared, a brand of a component) are attributes of the \
topic, not new topics.
  - Several genuinely independent facts a future query could ask \
separately → one derivative per fact.
  - A single focused statement (one fact, one chess move, one \
definition, one commit message) → exactly ONE near-original derivative \
preserving the segment's wording. Do NOT split its clauses. Do NOT \
paraphrase.
  - A bare standalone entity (a single noun like \"Tokyo\") → exactly \
ONE derivative containing just that term.
  - A list of items sharing the same predicate (versions of one model, \
quarters of one year, rows of one set, members of one series) → ONE \
derivative naming the parent and listing the items inline. Per-item \
derivatives waste slots because all items share the same future query.

EACH DERIVATIVE:
  - Is a full grammatical sentence (except bare-entity derivatives).
  - Uses the segment's wording for identifying terms verbatim — names, \
places, dates, amounts, brands, distinctive concept words.
  - Preserves IDENTIFYING RELATIONS as atomic phrases. \"my wife Anne\", \
\"manager Sarah\", \"CFO Casey\", \"our dog Mochi\" — the name and its \
relation/role appear TOGETHER in every derivative that mentions the \
entity. Never emit one derivative with just \"Anne\" and another with \
just \"my wife\" — that splits the binding and a query for either form \
loses the other.
  - Is self-contained — readable as if it were the only memory in the \
index about its topic. Repeat the SHARED SCOPE (the trip, project, \
period, person, artifact the facts are ABOUT) in every sibling \
derivative whose fact lives under it. Co-actors who belong to that \
scope (e.g., the spouse on a trip, the team on a project) travel with \
the scope into every derivative; truly per-clause mentions (a passerby \
at one event, a brand named in only one clause) stay where they are.

NON-PROSE OR PARTLY-FORMATTED CONTENT (code, tables, encoded text, log \
lines, JSON, markup, lists of raw numbers): you MUST emit AT LEAST ONE \
DESCRIPTION derivative — a sentence saying what the segment IS (e.g., \
\"Python function returning max in a binary tree\", \"benchmark table \
comparing GPT, Claude, and Gemini on Task A and B\", \"Caesar cipher \
text\") and what it is ABOUT. In ADDITION:
  - Decode recognizable encodings (Caesar, base64) into prose, in a \
separate derivative.
  - For multi-row tables, emit per-row prose sentences when each row is \
independently searchable, in addition to the description.
  - Render values in prose. Never preserve pipes, code syntax, or \
brackets.

HEAVY ACRONYMS (JFK, POTUS, CMC) or text in a script/language a query \
won't use: ONE derivative includes both the original and the expanded/\
translated form. Skip for plain English with full names.

Never invent content the segment lacks.

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
        f"v16 length: {len(PROMPT_DERIVER)} chars / {len(PROMPT_DERIVER.split())} words"
    )
