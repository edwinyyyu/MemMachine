"""Atomic deriver v1 — multiple atomic queryable cues per segment.

Hypothesis: v22 nano segmenter produces coherent multi-particular
statements (good for display) but single-paraphrase derivatives miss
retrieval shots when query phrasing doesn't match the dominant aspect
of the statement. Multiple atomic derivatives per segment give the
vector store multiple retrieval handles for the same display segment.

Retrieval dedups by segment_uuid — best-scoring derivative wins, so
adding more derivatives never displaces other segments. Only adds
shots.

Cost: roughly Nx more derivative storage and embedding cost (where N
is avg atomic count). LLM deriver call count unchanged.
"""

from __future__ import annotations

from typing import override
from uuid import uuid4

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
from memmachine_server.episodic_memory.event_memory.deriver.deriver import Deriver
from pydantic import BaseModel

PROMPT_DERIVER_ATOMIC_V1 = """\
A SEGMENT is a single memory statement stored verbatim in a retrieval \
system. Your job is to produce DERIVATIVES -- additional strings \
embedded as separate vectors alongside the segment. Retrieval dedupes \
by segment, so multiple derivatives compete to be the best match for a \
query, never displacing other segments.

Goal: produce ONE derivative per QUERYABLE ASPECT of the segment, so \
diverse query phrasings each find a matching derivative.

QUERYABLE ASPECTS commonly include:
- WHO did/experienced something (subject + action)
- WHAT happened or was decided (event or claim)
- WHEN it happened (time anchor)
- WHERE it took place (location)
- WHY it happened (motivation, reason)
- HOW it was done (manner, method)
- WHAT KIND of thing it is (category, type, attribute)
- OUTCOME / REACTION (result, emotional response)
- ATTACHED MEDIA description, if any

EACH DERIVATIVE:
- Is a complete grammatical sentence, self-contained: the SUBJECT \
(person, place, or thing the segment is about) and the SPECIFIC \
ASPECT must both appear by name. Pronouns and demonstratives are \
FAILURES.
- Reuses the segment's verbatim concrete particulars -- names, dates, \
numbers, identifiers, distinctive phrases.
- Is anchored to the same date the segment is anchored to, if the \
segment carries a date.
- Targets ONE queryable aspect. Two aspects sharing a sentence are \
fine (e.g. "Alice played tennis on 2024-05-12") but a derivative \
summarizing the WHOLE segment is wasteful -- prefer aspect-targeted \
phrasings.

COVERAGE RULES:
- Every concrete particular (each named entity, date, number, decision, \
opinion, emotional state) must appear in at least one derivative.
- Produce 2-6 derivatives for a typical segment. Single-fact segments \
(a bare name, a one-sentence assertion) may have 1-2; rich segments \
(multi-attribute events with motivation and outcome) may have 4-6.
- Do not duplicate the same aspect with synonyms. Each derivative \
should be queryable by a distinct angle.
- A passage with no specific content -> empty list.

EACH DERIVATIVE STANDS ALONE:
- A query for any one entity or particular should find the derivative \
that mentions it by name -- not via pronouns to a scope established \
earlier in the segment.
- Never invent content not in the segment.

Output: a JSON object {{ "derivatives": [...] }} and nothing else.

SEGMENT:
{segment}"""


class _DeriverResponse(BaseModel):
    derivatives: list[str]


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


class AtomicDeriver(Deriver):
    """Produces multiple atomic queryable derivatives per segment."""

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_DERIVER_ATOMIC_V1,
        emit_at_least_one: bool = True,
        max_attempts: int = 3,
    ) -> None:
        self._language_model = language_model
        self._prompt_template = prompt_template
        self._emit_at_least_one = emit_at_least_one
        self._max_attempts = max_attempts

    async def _derive_texts(self, segment_text: str) -> list[str]:
        prompt = self._prompt_template.format(segment=segment_text)
        response = await self._language_model.generate_parsed_response(
            output_format=_DeriverResponse,
            user_prompt=prompt,
            max_attempts=self._max_attempts,
        )
        if response is None:
            return []
        return [d.strip() for d in response.derivatives if d and d.strip()]

    @override
    async def derive(self, segment: Segment) -> list[Derivative]:
        match segment.block:
            case TextBlock(text=text):
                pass
            case _:
                raise NotImplementedError(
                    f"Unsupported block type: {type(segment.block).__name__}"
                )

        derivative_texts = await self._derive_texts(text)
        if not derivative_texts and self._emit_at_least_one:
            derivative_texts = [text]

        return [
            Derivative(
                uuid=uuid4(),
                segment_uuid=segment.uuid,
                timestamp=segment.timestamp,
                context=segment.context,
                block=TextBlock(text=_format_with_context(segment.context, d)),
                properties=segment.properties,
            )
            for d in derivative_texts
        ]
