"""v65 deriver -- SHIPPED. v64 + explicit completeness rule for ONE-derivative
outputs, placed in DERIVATIVES COVERAGE adjacent to the "ONE derivative"
trigger.

N=40 content-drop stress test on gpt-5-nano @ low:
  v64: 3/40 catastrophic drops + 7/40 partial = 25% problematic
  v65: 0/40 (statistically robust -- P(0/40 at v64's rate) ~4.5%)
  v66 (rule moved to EACH DERIVATIVE): 2/40 catastrophic -- regressed

Placement finding: rule must be ADJACENT to "ONE searchable thing ->
ONE derivative" trigger. When moved to EACH DERIVATIVE (semantically
correct location), 2/40 drops returned -- the trigger and rule become
decoupled, model commits to short output before reading the qualifier.

Higher reasoning (gpt-5-nano @ medium with v64) does NOT help -- 6/40
catastrophic drops vs 3/40 at low. The prompt fix is the unlock, not
reasoning level.



User-observed failure: gpt-5-nano @ low on D1 Mustang sometimes emits
ONE derivative that is just the first sentence of the segment ("We
finally finished restoring the 1967 Ford Mustang my grandfather left
me.") -- dropping all the specific details (engine, color, transmission,
etc.). The bench scorer checks count but not content preservation.

v65 adds an explicit completeness reminder to the ONE-central-subject
bullet:

  v64 had: "...is ONE searchable thing -> ONE derivative."
  v65 has: "...is ONE searchable thing -> ONE derivative that still
           covers EVERY specific detail (names, dates, numbers,
           distinctive phrases). ONE derivative is one searchable
           string, not a short summary."

Other content unchanged from v64.
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
searchable thing -> ONE derivative that still covers EVERY specific \
detail (names, dates, numbers, distinctive phrases). ONE derivative \
is one searchable string, not a short summary. Co-mentioned \
particulars (a city, a friend, a brand) are ATTRIBUTES of the central \
subject, not new topics -- they don't each get their own derivative. \
Emitting one derivative per sentence of a single-subject narrative is \
a FAILURE.
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
        f"v65 length: {len(PROMPT_DERIVER)} chars / {len(PROMPT_DERIVER.split())} words"
    )
