"""LLM segmenter, short-input specialization v47.

Diagnostic: when v33 sees a short message (1-3 sentences) in isolation,
it over-fragments at sentence boundaries; same v33 prompt with the
same message embedded in surrounding context correctly merges or drops
it. The over-fragmentation is not a rule-comprehension failure but a
short-input artifact: with no neighbors to compare against, the model
can't apply topic-shift or filler-drop rules.

v47 specializes for short inputs (typically 1-3 sentences). It
replaces topic-shift detection -- meaningless for a single message --
with a single binary classification: does the message carry memorable
content, or is it pure framing? If yes, emit it whole as one entry.
If no, emit zero entries.

The output structure is tied to the classification, not biased: a
short message that's substantive becomes one entry; a short message
that's pure framing becomes zero entries. Multiple-entry outputs are
not part of the contract because the topic-shift question doesn't
apply at this granularity.

Routing intended: caller dispatches short inputs (e.g., <= 200 chars
or <= 3 sentences) to this prompt, longer inputs to v33.
"""

from __future__ import annotations

import json

import openai
from dotenv import load_dotenv

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


SCHEMA_NATURAL = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "segments": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["segments"],
}


PROMPT_SHORT_V47 = """\
Decide whether this short conversational message carries memorable \
content.

Rules:
1. VERBATIM. If you emit the message, emit it as a contiguous verbatim \
quote. Never paraphrase, swap synonyms, or change wording -- \
"fabulous" stays "fabulous"; preserve whitespace, newlines, and \
special characters exactly. The only allowed trim is dropping a \
leading greeting that prefixes substantive content -- the substantive \
content stays.
2. MEMORABLE CONTENT TEST. The message carries memorable content if \
it contains anything specific that differentiates it from \
interchangeable conversation -- a named entity, place, date, number, \
identifier, decision, plan, opinion with backing, preference, \
relationship, emotional state tied to an event, constraint, or \
distinctive phrasing. The message is PURE FRAMING if it contains \
none of these -- a bare greeting, a reaction with no specifics ("Wow, \
that's amazing!"), a sign-off, an empty acknowledgment.
3. OUTPUT. If the message carries memorable content, emit it as ONE \
verbatim entry. If the message is pure framing, emit an empty list. \
A short message stays whole -- it is not split at sentence \
boundaries.

Output: a JSON object {{ "segments": [...] }} and nothing else.

MESSAGE:
{passage}"""


async def call(client, model, prompt, reasoning):
    kwargs: dict = {
        "model": model,
        "input": prompt,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "segments_response",
                "schema": SCHEMA_NATURAL,
                "strict": True,
            }
        },
    }
    if reasoning and model.startswith("gpt-5"):
        kwargs["reasoning"] = {"effort": reasoning}
    resp = await client.responses.create(**kwargs)
    raw = (resp.output_text or "").strip()
    parsed = json.loads(raw)
    return parsed.get("segments", [])


async def segment(
    client: openai.AsyncOpenAI,
    model: str,
    text: str,
    reasoning: str | None = "low",
) -> list[str]:
    """Run v47 on a short message. No windowing -- caller routes long
    passages elsewhere."""
    prompt = PROMPT_SHORT_V47.format(passage=text)
    return await call(client, model, prompt, reasoning)
