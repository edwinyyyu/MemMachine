"""Stable-fact gated topic-cue deriver v1.

Topic-cue v1 (probe_deriver_topiccue_v1.py) won g4 (+3.05pp c124) but
REGRESSED on g3 (-2.66pp c124, -7.50pp c2). The g3 regressions
concentrate on:

  - Temporal queries (date-anchored, ordinal-anchored, future-tense)
    — topic-cues drop the date, matching wrong-date segments.
  - Stable-fact queries on a high-baseline group — adding derivatives
    just crowds the same topical cluster; the gold segment's WHOLE
    derivative was already winning.

Diagnosis: in a conversation with many segments sharing a topic, a
topic-cue on EVERY segment creates a vector-space cluster where the
right gold doesn't win. Topic-cues only help when the segment carries
a STABLE generalizable fact that the segment's whole text alone
wouldn't surface via cosine.

This deriver gates topic-cue emission strictly. The LLM decides
whether the segment qualifies. The default is NO topic-cue.

Emit a topic-cue ONLY if all of:
  1. The segment is a STABLE FACT — a preference, attribute,
     possession, role, allegiance, hobby list, or member-of list.
     (One-time events, plans, ordinal events do NOT qualify.)
  2. The fact has a CATEGORY noun that a future generic query would
     name (sports, foods, books, hobbies, music, jobs, hometown,
     pets, etc.).
  3. Dropping specifics would not lose date/ordinal/future-tense
     anchoring (the segment isn't temporally indexed).

Specifically REJECT (emit no derivative) when the segment:
  - Has a specific date as the primary identifier of the fact.
  - Has an ordinal qualifier (first, second, third, latest, last).
  - Describes a one-time event regardless of date.
  - Is a future-tense plan ("will", "going to", "plans to").
  - Is a duration ("how long", "for N months/years").
  - Is a single-action verb statement ("X did Y on Z").

When emitted, the topic-cue:
  - Names the OWNER + CATEGORY in a 3-7 word noun phrase.
  - Does NOT include the answer value (the specific item).
  - Does NOT include dates, ordinals, or specific names of the
    queried-for entity.

Cost: at most 1 LLM call/segment. Most segments will receive an
empty topic-cue (the LLM returns ""), meaning only the whole-text
derivative anchors them.
"""

from __future__ import annotations

from typing import override
from uuid import uuid4

from pydantic import BaseModel

from memmachine_server.common.language_model import LanguageModel
from memmachine_server.episodic_memory.event_memory.data_types import (
    Context,
    Derivative,
    NullContext,
    ProducerContext,
    RewriteContext,
    Segment,
    SurroundingEventsContext,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.deriver.deriver import (
    Deriver,
)


PROMPT_STABLE_V1 = """\
You are deciding whether a memory segment deserves a TOPIC CUE -- an \
extra generic-shape phrase that gives the segment a second retrieval \
anchor for queries that name the topic without naming the specific \
answer.

DEFAULT: emit "" (empty string, NO topic cue).

EMIT a topic cue ONLY when ALL of these are true:
  (a) The segment expresses a STABLE FACT about an entity -- a \
preference, attribute, possession, role, allegiance, hobby, ongoing \
practice, or member list. NOT a one-time event, plan, or dated \
occurrence.
  (b) The fact has a CATEGORY NOUN that a future generic question \
would naturally use (sports, hobbies, foods, languages, pets, books, \
team, hometown, profession, club, allergy, etc.).
  (c) The segment does NOT carry a date, an ordinal qualifier \
(first, second, third, latest, last), a future-tense plan ("will", \
"going to", "plans to"), or a duration ("for X months"). If the \
segment is anchored by any of these, return "" -- the whole-text \
derivative already serves date-based and ordinal queries.

When emitted, the topic cue:
  - Is a 3-7 word noun phrase: OWNER + CATEGORY-NOUN.
  - DROPS the specific value of the category. ("Alice's favorite \
food" not "Alice's favorite food is paella".)
  - DROPS all dates, ordinals, future-tense markers.
  - Names the entity, not pronouns.
  - Captures the SINGLE most-likely query axis. Pick one.

EXAMPLES (Alice/Bob/Charlie/Dana, neutral domains):

SEGMENT: "Alice's pets are a cat named Mochi, a turtle named Slowpoke, \
and a fish named Bubbles, as of 2024-03-10."
TOPIC_CUE: "Alice's pet names"

SEGMENT: "Bob supports the Lakers basketball team and has been a fan \
since 2010."
TOPIC_CUE: "Bob's basketball team allegiance"

SEGMENT: "Charlie's hobbies include pottery, climbing, and writing \
short fiction."
TOPIC_CUE: "Charlie's hobbies"

SEGMENT: "Dana speaks fluent Portuguese and conversational Mandarin."
TOPIC_CUE: "Dana's spoken languages"

SEGMENT: "Alice went on a road trip to Lisbon with her sister in July \
2024."
TOPIC_CUE: ""    # dated one-time event -- whole derivative is enough

SEGMENT: "Bob won his second amateur chess tournament on 2025-02-14."
TOPIC_CUE: ""    # ordinal + dated event

SEGMENT: "Charlie plans to start a podcast about beekeeping next \
month."
TOPIC_CUE: ""    # future-tense plan

SEGMENT: "Dana spent six months in Lyon learning French during 2023."
TOPIC_CUE: ""    # duration + dated

SEGMENT: "Alice baked a chocolate-cardamom cake for Bob's birthday \
on 2024-09-12."
TOPIC_CUE: ""    # one-time dated event

SEGMENT: "Bob's favorite breakfast is sourdough toast with honey and \
sea salt."
TOPIC_CUE: "Bob's favorite breakfast"

SEGMENT: "Charlie is allergic to shellfish and tree nuts."
TOPIC_CUE: "Charlie's food allergies"

SEGMENT: "Dana said hi and asked how things are going."
TOPIC_CUE: ""    # filler / no specific content

Output: a JSON object {{ "topic_cue": "..." }}.

SEGMENT: {segment_text}"""


class _StableResponse(BaseModel):
    topic_cue: str


def _format_with_context(context: Context, text: str) -> str:
    match context:
        case ProducerContext(producer=producer):
            return f"{producer}: {text}"
        case SurroundingEventsContext(producer=producer):
            return f"{producer}: {text}"
        case NullContext():
            return text
        case RewriteContext(text_to_embed=text_to_embed):
            return text_to_embed
        case _:
            raise NotImplementedError(
                f"Unsupported context type: {type(context).__name__}"
            )


class GenericDeriver(Deriver):
    """Strict-gated topic-cue deriver. Emits whole + at most one topic cue."""

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_STABLE_V1,
        max_attempts: int = 3,
    ) -> None:
        self._language_model = language_model
        self._prompt_template = prompt_template
        self._max_attempts = max_attempts

    async def _generate_cue(self, segment_text: str) -> str:
        prompt = self._prompt_template.format(segment_text=segment_text)
        response = await self._language_model.generate_parsed_response(
            output_format=_StableResponse,
            user_prompt=prompt,
            max_attempts=self._max_attempts,
        )
        if response is None:
            return ""
        return (response.topic_cue or "").strip()

    @override
    async def derive(self, segment: Segment) -> list[Derivative]:
        match segment.block:
            case TextBlock(text=text):
                pass
            case _:
                raise NotImplementedError(
                    f"Unsupported block type: {type(segment.block).__name__}"
                )

        whole_text = _format_with_context(segment.context, text)
        derivatives: list[Derivative] = [
            Derivative(
                uuid=uuid4(),
                segment_uuid=segment.uuid,
                timestamp=segment.timestamp,
                context=segment.context,
                block=TextBlock(text=whole_text),
                properties=segment.properties,
            )
        ]

        cue = await self._generate_cue(text)
        if cue:
            derivatives.append(
                Derivative(
                    uuid=uuid4(),
                    segment_uuid=segment.uuid,
                    timestamp=segment.timestamp,
                    context=NullContext(),
                    block=TextBlock(text=cue),
                    properties=segment.properties,
                )
            )

        return derivatives
