"""v41 deriver -- SHIPPED. Targeted robustness fixes vs v40b.

4-pass sweep results (3 prompts x 3 models x 2 reasoning x 10 cases =
60 calls per pass; 240 calls per prompt total):

  v40b: 58 + 52 + 55 + 57 = 222/240 = 92.5%  (range 52-58, Δ6)
  v41:  58 + 60 + 58 + 58 = 234/240 = 97.5%  (range 58-60, Δ2)

v41 wins on BOTH mean (+5pp) and variance (3x tighter). Targeted
failure modes:
  - P1 preference atomization: 4 passes with fails -> 0 (FIXED via
    "preference, opinion, decision" in subject types + atomization-
    FAILURE callout)
  - B1 self-containment: 4 passes -> 2 (improved via concrete
    query-side reframe of "stands alone as search result")
  - D1 mustang atomization: 2 passes -> 2 (likely gpt-5.4-nano limit)
  - FIL1 / B2: 1 / 1 (unchanged, likely gpt-5-nano limit)

Changes from v40b:
  - DERIVATIVES COVERAGE bullets:
    * PURE FILLER: added "(return an empty list)"
    * "ONE thing" -> "ONE central subject"; added "preference, opinion,
      decision" to subject types; added "Atomizing a single-subject
      narrative into per-sentence derivatives is a FAILURE."
    * Added tiebreaker: "When unsure between ONE derivative and
      several, prefer ONE."
  - EACH DERIVATIVE bullets:
    * Self-contained rule reframed as "Stands alone as a search
      result" with concrete query-side reasoning (a query mentioning
      scope/co-actor returns every relevant sibling, not just the
      introductory one).

Length: 3495 chars (+412 vs v40b). Trade-off accepted: clarity for
robustness.

Header names, goal paragraph, and NON-PROSE SURFACES section unchanged.
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
cover every searchable thing -- every extra slot wastes index space. \
Splitting a focused thought, paraphrasing one fact, listing items \
already inline, or atomizing a single-topic narrative are FAILURES, not \
coverage.

DERIVATIVES COVERAGE:
- PURE FILLER -- content-free short responses only meaningful with the \
prior message ("yes", "no", "ok", "thanks", "lol", "great point") -> \
emit ZERO derivatives (return an empty list). When such a response \
ALSO carries concrete content ("ok, leaving Tuesday at 5", "no, I \
changed my mind about Tuesday"), it is NOT pure filler -- emit \
derivatives for the content.
- A passage centered on ONE central subject -- a person, place, event, \
project, process, product, relationship, preference, opinion, or \
decision -- even across many sentences, is ONE searchable thing -> ONE \
derivative. Co-mentioned particulars (a city it visited, a friend who \
appeared, a brand of a component) are ATTRIBUTES of the central \
subject, not new topics. Atomizing a single-subject narrative into \
per-sentence derivatives is a FAILURE.
- Several genuinely independent particulars a future query could ask \
separately -> one derivative per particular.
- A single focused statement (a function signature, a commit message, a \
definition) -> exactly ONE near-original derivative. Do NOT split its \
clauses; do NOT paraphrase.
- A bare standalone term ("Paris") -> ONE derivative with just that \
term.
- A list of items sharing one predicate (versions of a model, rows of a \
table) -> ONE derivative naming the parent and listing items inline.
- When unsure between ONE derivative and several, prefer ONE.

EACH DERIVATIVE:
- Is a full grammatical sentence (except bare-entity), using the \
segment's wording verbatim for CONCRETE PARTICULARS -- names, places, \
dates, numbers, identifiers, decisions, plans, preferences, opinions, \
relationships, emotional states tied to events, constraints, and \
distinctive phrasing. Drop generic abstractions or stock phrases.
- Preserves compound identifiers. When a name combines with a role, \
qualifier, or descriptor that disambiguates it ("team lead Alice" vs. \
other Alices; "the engineering library on campus" vs. other libraries), \
keep the whole phrase together in every derivative that references it \
-- splitting the name from its qualifier loses the binding a future \
query relies on.
- Stands alone as a search result. When sibling derivatives share a \
scope (a trip, project, period, conversation, artifact) and the \
co-actors central to that scope, EVERY sibling must repeat both the \
scope and those co-actors. Otherwise a query mentioning the scope or a \
co-actor returns only the introductory sibling, not the ones that \
describe what actually happened.
- Never invents content the segment lacks.

NON-PROSE SURFACES (code, tables, encoded text, JSON, log lines, \
markup):
- Emit one DESCRIPTION derivative naming what the segment IS and what \
it is ABOUT.
- Decode recognizable encodings (Caesar, base64) into a separate prose \
derivative.
- For multi-row tables, optionally emit one prose derivative per \
independently-searchable row.
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
        f"v41 length: {len(PROMPT_DERIVER)} chars / {len(PROMPT_DERIVER.split())} words"
    )
