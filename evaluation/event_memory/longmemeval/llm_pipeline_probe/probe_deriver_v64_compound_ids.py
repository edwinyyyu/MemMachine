"""v64 deriver -- SHIPPED. v63 + compound-identifier reminder in
self-containment rule.

4-pass sweep on expanded 13-case bench (78 calls/pass, 312 calls/prompt):
  v63: 72+75+75+73 = 295/312 = 94.6% (range Δ3)
  v64: 76+75+77+76 = 304/312 = 97.4% (range Δ2)

v64 wins +2.8pp on mean AND tighter variance. Beyond the targeted
compound-identifier fix, the inline reminder also improved P1, FIL1,
and PR2 (4 fewer failures than v63 across these cases).

Case-by-case vs v63 (4 passes each):
  P1 preference:  2 -> 0 (FIXED)
  FIL1:           2 -> 0 (FIXED)
  PR2 treehouse:  1 -> 0 (FIXED)
  PR3 apple:      0 -> 1 (regressed)
  B1 binding:     4 -> 4 (tied; gpt-5.4-nano@low atomization limit --
                  drops Anne+Tokyo entirely from "We stayed at Park
                  Hyatt", not a compound-id issue)
  D1 mustang:     2 -> 2 (tied)
  B2 binding:     1 -> 1 (tied)



User: '"name the scope/anyone/anything" may not include the relations
like "team manager Alice" or "neighbor Bob"'.

The compound identifier rule (EACH DERIVATIVE bullet 2) says compound
identifiers stay together. But the self-containment rule says "name
anyone involved" -- the model might write just "Anne" or "Alice"
without the compound identifier.

v64 adds inline reminder "with compound identifiers intact ('my wife
Anne', 'team lead Alice')" to the self-containment rule:

  v63: "...each must name the scope (the trip, project, period,
       conversation, artifact) and anyone or anything else involved
       -- never as pronouns..."

  v64: "...each must name the scope (the trip, project, period,
       conversation, artifact) and anyone or anything else involved
       -- with compound identifiers intact ('my wife Anne', 'team
       lead Alice'), never as pronouns..."

Cross-references the compound identifier rule. Brief.
"""

from __future__ import annotations

import json
from typing import Any

from dotenv import load_dotenv

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


PROMPT_DERIVER = """\
A SEGMENT is stored verbatim in a retrieval system. Your DERIVATIVES are \
additional strings embedded for semantic search alongside it.

Goal: a future user querying anything in this segment should find it \
via at least one derivative. Generate the FEWEST derivatives that cover \
every searchable thing. Splitting a focused thought, paraphrasing one \
fact, listing items already inline, or atomizing a single-topic \
narrative are FAILURES.

DERIVATIVES COVERAGE:
- PURE FILLER -- content-free short responses only meaningful with the \
prior message ("yes", "no", "ok", "thanks", "lol", "great point") -> \
emit ZERO derivatives (return an empty list). When such a response \
ALSO carries concrete content ("ok, leaving Tuesday at 5", "no, I \
changed my mind about Tuesday"), it is NOT pure filler -- emit \
derivatives for the content.
- A passage centered on ONE central subject -- a person, place, event, \
project, process, product, relationship, preference, opinion, \
decision, or concept -- across any number of sentences is ONE \
searchable thing -> ONE derivative. Co-mentioned particulars (a city, \
a friend, a brand) are ATTRIBUTES of the central subject, not new \
topics -- they don't each get their own derivative. Emitting one \
derivative per sentence of a single-subject narrative is a FAILURE.
- Several genuinely independent particulars a future query could ask \
separately -> one derivative per particular.
- A single focused statement (a function signature, a commit message, \
a definition) -> ONE near-original derivative. Do NOT split its \
clauses; do NOT paraphrase.
- A bare standalone term ("Paris") -> ONE derivative with just that \
term.
- A list of items sharing one predicate (versions of a model, rows of \
a table) -> ONE derivative naming the parent and listing items inline.
- When unsure between ONE and several derivatives, prefer ONE.

EACH DERIVATIVE:
- Is a full grammatical sentence (except bare-entity), using the \
segment's wording verbatim for CONCRETE PARTICULARS -- names, places, \
dates, numbers, identifiers, decisions, plans, preferences, opinions, \
relationships, emotional states tied to events, constraints, and \
distinctive phrasing. Drop generic abstractions or stock phrases.
- Preserves compound identifiers. When a name takes a disambiguating \
role or descriptor ("team lead Alice", "the engineering library on \
campus"), keep the whole phrase together every time -- splitting loses \
the query-binding.
- Stands alone as a search result. When you emit multiple derivatives \
for one segment, each must name the scope (the trip, project, period, \
conversation, artifact) and anyone or anything else involved -- with \
compound identifiers intact ("my wife Anne", "team lead Alice"), \
never as pronouns ("it", "they", "we") for the scope or anyone/anything \
involved. A query for the scope or anyone/anything involved finds \
only the derivatives where they were named, not the ones using \
pronouns.
- Never invents content the segment lacks.

NON-PROSE SURFACES (code, tables, encoded text, JSON, log lines, \
markup):
- Emit one DESCRIPTION derivative naming what the segment IS and what \
it is ABOUT.
- Decode encodings (Caesar, base64) into a separate prose derivative.
- For non-prose aggregates with multiple distinct entries (table rows, \
JSON array elements, log batches), optionally emit one prose \
derivative per entry.
- Never preserve pipes, code syntax, or brackets.

Output: a JSON object {{ "derivatives": [...] }} and nothing else.

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
    non_ascii = [(i, c) for i, c in enumerate(PROMPT_DERIVER) if ord(c) > 127]
    if non_ascii:
        print(f"WARNING: {len(non_ascii)} non-ASCII chars: {non_ascii[:10]}")
    else:
        print("All ASCII OK")
    print(
        f"v64 length: {len(PROMPT_DERIVER)} chars / {len(PROMPT_DERIVER.split())} words"
    )
