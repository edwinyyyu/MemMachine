"""v50 deriver -- v49 with subject-word variation and explicit
co-mentioned-particulars connection.

User feedback on v49:
  1. "subject" repeats 3x ("ONE central subject ... The central subject
     ... or any other coherent subject")
  2. Co-mentioned particulars rule needs explaining -- what does it
     have to do with per-sentence derivatives?

v50 changes the subject bullet:
  - "topic" / "subject" alternation: "ONE central topic ... The topic
     can be ... or any other coherent subject"
  - Restored explanatory particulars examples ("a city it visited",
     "a friend who appeared", "a brand of a component") -- they make
     the ATTRIBUTE pattern recognizable
  - Made the co-mentioned-to-failure connection explicit: "splitting
     them into per-particular or per-sentence derivatives is a FAILURE"

Other content unchanged.
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
- A passage centered on ONE central topic across any number of \
sentences is ONE searchable thing -> ONE derivative. The topic can be \
a person, place, thing, process, event, concept, preference, opinion, \
decision, or any other coherent subject. Co-mentioned particulars (a \
city it visited, a friend who appeared, a brand of a component) are \
ATTRIBUTES of that topic, not separate topics -- splitting them into \
per-particular or per-sentence derivatives is a FAILURE.
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
- Is a full grammatical sentence (except bare-entity), preserving the \
segment's wording verbatim for any concrete particular a future query \
might include -- names, places, dates, numbers, decisions, \
preferences, opinions, distinctive phrasing, or any other specific \
detail. Drop generic abstractions or stock phrases.
- Preserves compound identifiers. When a name takes a disambiguating \
role or descriptor ("team lead Alice", "the engineering library on \
campus"), keep the whole phrase together every time -- splitting loses \
the query-binding.
- Stands alone as a search result. When sibling derivatives share a \
scope (a trip, project, period, conversation, artifact) and the \
central co-actors of that scope, EVERY sibling must include both -- \
otherwise a query mentioning the scope or a co-actor returns only the \
introductory sibling, not the ones describing what actually happened.
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
        f"v50 length: {len(PROMPT_DERIVER)} chars / {len(PROMPT_DERIVER.split())} words"
    )
