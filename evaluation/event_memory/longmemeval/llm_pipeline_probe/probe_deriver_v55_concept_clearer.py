"""v55 deriver -- SHIPPED. v45 with minimal targeted changes per user
feedback.

4-pass sweep results (3 prompts x 3 models x 2 reasoning x 10 cases =
60 calls per pass; 240 calls total):
  v45: 60+57+59+57 = 233/240 = 97.1% (range Δ3)
  v55: 57+58+57+59 = 231/240 = 96.3% (range Δ2)

v55 is within noise of v45 (-0.8pp) with slightly TIGHTER variance.
2 of 4 passes v55 won outright. The D1 mustang atomization on
gpt-5.4-nano regressed 2/4 -> 4/4 (model-limited weakness already known)
but FIL1 improved.



User asked for:
  1. Keep v45 as base
  2. Add general-but-not-too-specific category (e.g., "concept") to the
     subject list to cover theorems, definitions, ideas without naming
     them individually
  3. Address wording confusion: "what does sentences have to do with
     multiple co-mentioned particulars? What is per-sentence? Do you
     mean per-particular?" -- per-sentence and per-particular are
     different failure modes; wording needs to distinguish them

v55 changes from v45:
  (a) Subject list: "or decision" -> "decision, or concept" -- adds one
      general category that covers theorems, definitions, abstract ideas
  (b) Particulars rule now explicit about per-particular: "they don't
      each get their own derivative" -- makes the per-particular failure
      mode concrete
  (c) Per-sentence FAILURE rephrased without jargon: "Emitting one
      derivative per sentence of a single-subject narrative is a
      FAILURE." -- answers "what is per-sentence?" with the actual
      operation

Two failure modes (per-particular and per-sentence) now in separate
sentences, each named explicitly. No merging that caused v50's
muddled wording.

All other content verbatim from v45.
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
        f"v55 length: {len(PROMPT_DERIVER)} chars / {len(PROMPT_DERIVER.split())} words"
    )
