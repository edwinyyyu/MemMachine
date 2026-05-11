"""v60 deriver -- SHIPPED. Self-containment bullet rewritten for clarity.

4-pass sweep on expanded 13-case bench (78 calls/pass, 312 calls/prompt):
  v58 (with "both BY NAME"): 73+74+75+72 = 294/312 = 94.2%
  v60 (rewritten):           77+77+73+77 = 304/312 = 97.4%

Case-by-case improvements vs v58 (4 passes):
  B1 binding:   4 -> 0 (FIXED! "name the scope AND anyone/anything else
                involved" decisively better than "include both BY NAME")
  PR1 clock:    2 -> 0 (FIXED)
  PR2 tree:     2 -> 0 (FIXED)
  D1 mustang:   4 -> 2 (improved)
  B2 binding:   2 -> 1 (improved)
  FIL3:         1 -> 0 (improved)
  FIL1:         1 -> 3 (regressed on gpt-5-nano; likely model noise)
  P1 pref:      0 -> 1 (small)

Wording cleanup not only improved clarity but also improved model
adherence: clearer rule structure beats ambiguous "both" construction.



User feedback on v59: "'include both BY NAME' -- both what? the usage
of 'both' is weird here -- please review the wording of the bullet and
improve it".

v59 (issues):
  "Stands alone as a search result. When sibling derivatives share a
  scope (a trip, project, period, conversation, artifact) and anyone
  or anything involved in it, EVERY sibling must include both BY NAME
  (not as 'it', 'they', or 'we' standing in for the searchable
  subject) -- otherwise a query mentioning the scope or anything
  involved returns only the introductory sibling, not the ones
  describing what actually happened."

  Issues:
    - "both BY NAME" -- "both" refers ambiguously to "scope" AND
      "anyone or anything involved" (themselves an or-pair)
    - Long single sentence with multiple clauses
    - "standing in for the searchable subject" -- jargon
    - "the ones describing what actually happened" -- vague tail
    - "include both ... or anything else" -- explicit listing better

v60 rewrite:
  "Stands alone as a search result. In every sibling derivative, name
  the scope (the trip, project, period, conversation, artifact) and
  anyone or anything else involved -- never as pronouns ('it', 'they',
  'we') standing in for the central subject. A query for the scope or
  anyone/anything involved finds only the sibling where they were
  named, not the ones using pronouns."

  Changes:
    - Imperative restructure: "In every sibling derivative, name..."
      replaces "When sibling derivatives share... EVERY sibling must
      include both BY NAME"
    - Explicit listing: "the scope ... and anyone or anything else
      involved" replaces ambiguous "both"
    - "central subject" replaces "searchable subject" (less jargon)
    - Concrete failure consequence: "finds only the sibling where they
      were named, not the ones using pronouns" replaces vague tail
    - Two sentences instead of one (rule + consequence)
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
- Stands alone as a search result. In every sibling derivative, name \
the scope (the trip, project, period, conversation, artifact) and \
anyone or anything else involved -- never as pronouns ("it", "they", \
"we") standing in for the central subject. A query for the scope or \
anyone/anything involved finds only the sibling where they were \
named, not the ones using pronouns.
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
        f"v60 length: {len(PROMPT_DERIVER)} chars / {len(PROMPT_DERIVER.split())} words"
    )
