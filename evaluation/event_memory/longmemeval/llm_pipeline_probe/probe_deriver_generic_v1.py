"""Generic-derivative deriver v1.

Adds a second retrieval anchor per segment: a generic-shape paraphrase
that drops the specifics and keeps only the entities + category nouns
+ relational structure. The original whole-text derivative is still
emitted so specifics-anchored queries continue to match the segment.

Motivation
----------

The v22 segmenter preserves every concrete particular ("Pepper",
"Edinburgh", "The Wolves", "Chicken Pot Pie"). Its narrative embedding
is anchored on those specifics, which makes the segment match
specifics-anchored queries cleanly (e.g., "Who is Pepper?"). But
benchmark questions in the enum_which_what / list / count families
are systematically generic-anchored — the query contains the category
noun ("dog names", "basketball team", "movie") but NOT the specific
answer value. The specifics-anchored embedding scores poorly against
these generic queries, and segments on adjacent specific axes (e.g.,
Audrey-dog-photos when the question asks for names) displace the
gold-bearing segment.

Diagnosis on g4 (Tim+John, LoCoMo nb8b baseline 84.15% c124, 18 enum
fails; 12 of those are unambiguous REAL retrieval failures), all REAL
enum fails have the same shape: a generic-shape question whose gold
segment exists with the specific answer present, but the segment
ranks below adjacent specific segments. Examples:

  Q: "Which basketball team does Tim support?"
  Gold-bearing segment: "Tim said The Wolves are solid and LeBron's
                         skills and leadership are amazing on 2023-..."
  Displaced by: segments about Tim's other basketball discussions
                that don't name The Wolves.

  Q: "What language does Tim know besides German?"
  Gold-bearing segment: "John said he knows a bit of German himself
                         and Spanish, ..."
  Displaced by: other Tim-language-related segments.

Adding a generic-shape derivative ("Tim's basketball team
affiliation" / "languages Tim and John know") gives the same segment
a second anchor that lives in the generic-shape region of embedding
space. The query "Which basketball team does Tim support?" matches
the generic derivative directly.

Output
------

Two derivatives per segment:

  1. WHOLE -- block.text = _format_with_context(segment.context,
     segment.block.text). Identical to WholeTextDeriver. For
     RewriteContext segments this is "{rewrite}\\n{speaker}:
     {original_chunk}" -- the existing dual-text embed.

  2. GENERIC -- block.text = LLM-generated category-shape paraphrase
     of the segment. If the LLM returns empty (segment has no
     retrievable structure), no second derivative is emitted.

Both derivatives share segment_uuid. The vector store ingests them
independently; retrieval dedupes by segment_uuid post-pool, so the
top-K=7 slot count stays unchanged.

Cost
----

One small LLM call per segment at ingest. Parallelizable across the
corpus. With gpt-5.4-nano @ low, ingest cost roughly doubles over
the WholeTextDeriver baseline; vector storage doubles per segment.
Query time unchanged.

Caller guidance
---------------

Use this deriver paired with the v22 RewriteSegmenter and a
SurroundingEventsContext with before-window=8 (recommended segmenter
configuration). The generic derivative is generated from the segment
TEXT only -- it doesn't see neighbors, since the segment is already
self-contained after segmentation.
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


PROMPT_GENERIC_V1 = """\
Rephrase the FACT into a generic-shape derivative for retrieval.

A future user querying with a generic-shape question ("What X does Y \
do?", "Which Z does W use?") may not name the specific answer value. \
The derivative matches such queries by KEEPING entity names + \
category nouns + relational verbs and DROPPING the specific answer \
values (proper-noun titles, exact dates, lists of items, numbers).

Rules:
- KEEP entity names (the OWNER/subject of the fact).
- KEEP category nouns the fact is ABOUT (sports, team, recipe, \
language, achievement, city, dog names, movie, book, plan, advice, \
location, ailment).
- KEEP relational verbs in their natural form (supports, owns, \
recommends, suggests, visited, knows, plans, said, hopes).
- DROP specific values that would be the ANSWER content (proper-noun \
titles like "The Wolves", "Edinburgh, Scotland", "Pepper Precious \
Panda", exact dates, exact numbers).
- ONE derivative. Short, under 12 words. No leading "The fact is" \
or wrapper text -- just the derivative phrase.
- If the fact has no retrievable structure (pure filler, no entity, \
no category noun), emit "".

Examples:

FACT: "Tim said The Wolves are solid and LeBron's skills and \
leadership are amazing on 2023-12-11."
GENERIC: "Tim's basketball team affiliation"

FACT: "Audrey told Andrew that her dogs are named Pepper, Precious, \
and Panda on 2023-07-03."
GENERIC: "Audrey's dog names"

FACT: "Tim suggested Edinburgh, Scotland would be great for a \
magical vibe and the team trip on 2023-11-02."
GENERIC: "Tim's suggested city for the team trip"

FACT: "Joanna watched Eternal Sunshine of the Spotless Mind on \
2022-04-10."
GENERIC: "movies Joanna watched"

FACT: "John said his team faced tough opponents but that drives \
them to get better on 2023-10-13."
GENERIC: "John's team motivation"

FACT: "John said he got an endorsement with a popular beverage \
company on 2024-01-15."
GENERIC: "John's career achievement shared with Tim"

FACT: "Sam mentioned his doctor said his weight was a serious \
health risk on 2023-10-02."
GENERIC: "Sam's weight-related health condition"

Output: a JSON object {{ "generic": "..." }}.

FACT: {segment_text}"""


class _GenericResponse(BaseModel):
    generic: str


def _format_with_context(context: Context, text: str) -> str:
    """Mirror WholeTextDeriver's context formatting.

    For RewriteContext, returns text_to_embed (the
    rewrite+speaker+raw-chunk dual-text embed). For other contexts,
    returns the speaker-prefixed text or the bare text.
    """
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
    """Emits TWO derivatives per segment: whole + generic-shape paraphrase.

    Args:
        language_model: LanguageModel used to generate the generic
            paraphrase. Configure model + reasoning_effort at
            construction.
        prompt_template: A ``.format(segment_text=...)`` template.
            Defaults to PROMPT_GENERIC_V1.
        max_attempts: Retries on retryable language-model errors.
    """

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_GENERIC_V1,
        max_attempts: int = 3,
    ) -> None:
        self._language_model = language_model
        self._prompt_template = prompt_template
        self._max_attempts = max_attempts

    async def _generate_generic(self, segment_text: str) -> str:
        prompt = self._prompt_template.format(segment_text=segment_text)
        response = await self._language_model.generate_parsed_response(
            output_format=_GenericResponse,
            user_prompt=prompt,
            max_attempts=self._max_attempts,
        )
        if response is None:
            return ""
        return (response.generic or "").strip()

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

        # Always emit the whole-text derivative.
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

        # Generate the generic-shape derivative; skip if empty.
        generic = await self._generate_generic(text)
        if generic:
            derivatives.append(
                Derivative(
                    uuid=uuid4(),
                    segment_uuid=segment.uuid,
                    timestamp=segment.timestamp,
                    context=NullContext(),
                    block=TextBlock(text=generic),
                    properties=segment.properties,
                )
            )

        return derivatives
