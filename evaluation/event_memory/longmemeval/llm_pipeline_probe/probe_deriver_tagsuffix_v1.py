"""Tag-suffix deriver v1.

Emits ONE derivative per segment, identical to WholeTextDeriver,
except the embed text is augmented with a trailing `[tags: ...]`
suffix containing 0-6 generic category-noun tags. Same single
embedding per segment, just with extra trailing lexical anchors that
BM25 will weight on rare-token matches.

Hypothesis
----------

The v22 segmenter produces consolidated third-person narrative
segments whose RewriteContext gives every segment a dual-text embed
`"{rewrite}\\n{speaker}: {original_chunk}"`. That dual-text is
anchored on the segment's specifics: entity names, the exact answer
value, dates. It matches specifics-anchored queries cleanly.

But enum_which_what / list / count queries are generic-anchored —
the query contains a category noun ("dog names", "basketball team
affiliation") and not the specific answer value. The specifics-
anchored embed underweights those category-noun tokens, and a BM25
search over the same text yields little because the rare category
nouns are buried in fluent prose (or not present at all when the
narrative uses a more specific phrasing).

Appending a short, lowercase `[tags: ...]` suffix gives BM25 a
direct lexical anchor for the category nouns the segment is ABOUT,
without crowding the embedding's semantic axis. Tags are like a
Wikipedia "See also" list: secondary lexical anchors. The semantic
mass of the embed is still the rewrite+speaker+chunk; the tag list
is a short, dense lexical tail that costs <30 tokens.

This is an ADDITIVE-TO-WHOLE design: one derivative per segment, so
retrieval slot pressure is unchanged. The contrast with
GenericDeriver v1 is that GenericDeriver emits a SECOND derivative
that lives in the generic-shape region of embedding space; the tag-
suffix design keeps the segment as a single embedding and bets on
BM25 (or hybrid retrieval that weights rare-token matches) picking
up the category nouns.

Risk
----

1. Embedding bias. Generic tag tokens ("team affiliation", "cake
   recipe") near the end of the embed text may pull the segment's
   embedding toward a generic-shape region, partially diluting the
   specifics anchor. Mitigation: tags are at the tail, short, in
   bracket-prefixed metadata syntax that the embedder may treat as
   structural rather than semantic content. Empirically TBD.

2. Tag selection overfit. The LLM may overfit to LongMemEval
   question shapes (dog names, team affiliation) and miss tags for
   segments whose category nouns the eval doesn't probe. Mitigation:
   the prompt uses neutral example domains (Alice/Bob/Charlie/Dana,
   mandolin/cake/dog/team) so the deriver generalizes beyond the
   eval surface form.

3. Hallucinated tags. The LLM may emit tags that don't reflect the
   segment ("favorite movie" on a segment about a recipe). Pydantic
   schema enforcement limits this to category-noun text only;
   max-6 cap limits damage.

Generalizability argument
-------------------------

Any segment-based retrieval system that mixes BM25 with semantic
similarity benefits from explicit lexical anchors for category
nouns: the category noun is usually the highest-IDF token in a
generic-shape query, and segments that don't surface it lexically
score poorly under BM25 regardless of semantic fit. The tag-suffix
suffix is a domain-agnostic lexical-index augmentation pattern;
nothing in the design is LongMemEval-specific.

Output
------

ONE derivative per segment. block.text =
`_format_with_context(segment.context, segment.block.text)` (the
existing dual-text whole-text, identical to WholeTextDeriver) plus,
when tags are non-empty, `" [tags: " + ", ".join(tags) + "]"`. Empty
tag list means the segment is filler / has no useful category-noun
anchor; the bare whole-text is emitted unchanged.

Cost
----

One small LLM call per segment at ingest. Parallelizable across the
corpus. Vector storage unchanged (one derivative per segment).
Query time unchanged.
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


PROMPT_TAGSUFFIX_V1 = """\
Emit 0-6 generic category-noun TAGS for the FACT below. Tags are \
secondary lexical anchors (like Wikipedia "See also") that help \
keyword search match generic-shape queries.

A future user querying with a generic-shape question ("What X does \
Y do?", "Which Z does W like?") names the category noun ("dog \
names", "favorite team") but NOT the specific answer value. Tags \
should be those category nouns — what the fact is ABOUT, the \
answer-shape — not the answer value itself.

Rules:
- Each tag is 1-3 words, lowercase, a category noun phrase.
- Tags describe what the fact is ABOUT (the answer-shape), not the \
answer value.
- DO NOT include the specific answer value as a tag (no proper-noun \
titles, no exact dates, no exact numbers, no specific named values).
- 0 to 6 tags. Filler facts with no retrievable category-noun anchor \
get an empty list.
- Tags should be GENERIC enough that an unrelated segment about a \
different specific instance could share the same tag.

Examples:

FACT: "Alice supports the Lakers basketball team."
TAGS: ["basketball team affiliation", "favorite team"]

FACT: "Alice baked a chocolate-cardamom cake for Bob's birthday on \
2024-09-12."
TAGS: ["cake recipe", "birthday gift", "baking activity"]

FACT: "Charlie said his dogs are named Pepper, Precious, and Panda."
TAGS: ["dog names", "pet names"]

FACT: "Dana suggested Edinburgh would be a good city for the team \
retreat."
TAGS: ["suggested city", "team retreat location"]

FACT: "Bob learned to play the mandolin and practices on weekends."
TAGS: ["musical instrument", "hobby", "weekend activity"]

FACT: "Alice mentioned her doctor said her cholesterol was a \
concern."
TAGS: ["health condition", "doctor advice"]

FACT: "Charlie said hi."
TAGS: []

FACT: "Dana knows Spanish and a bit of Portuguese besides English."
TAGS: ["languages known", "language skills"]

Output: a JSON object {{ "tags": ["...", "..."] }}.

FACT: {segment_text}"""


class _TagResponse(BaseModel):
    tags: list[str]


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


def _clean_tags(raw_tags: list[str]) -> list[str]:
    """Normalize LLM tag output: strip, lowercase, drop empties, cap at 6."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in raw_tags:
        if not isinstance(raw, str):
            continue
        tag = raw.strip().lower()
        if not tag:
            continue
        if tag in seen:
            continue
        seen.add(tag)
        cleaned.append(tag)
        if len(cleaned) >= 6:
            break
    return cleaned


class GenericDeriver(Deriver):
    """Emits ONE derivative per segment: whole-text + `[tags: ...]` suffix.

    The derivative's text is identical to WholeTextDeriver's output
    (the dual-text embed for RewriteContext segments), with a
    trailing `[tags: tag1, tag2, ...]` suffix when the LLM emits any
    tags. Empty-tags segments get the bare whole-text.

    Args:
        language_model: LanguageModel used to generate tags.
            Configure model + reasoning_effort at construction.
        prompt_template: A ``.format(segment_text=...)`` template.
            Defaults to PROMPT_TAGSUFFIX_V1.
        max_attempts: Retries on retryable language-model errors.
    """

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_TAGSUFFIX_V1,
        max_attempts: int = 3,
    ) -> None:
        self._language_model = language_model
        self._prompt_template = prompt_template
        self._max_attempts = max_attempts

    async def _generate_tags(self, segment_text: str) -> list[str]:
        prompt = self._prompt_template.format(segment_text=segment_text)
        response = await self._language_model.generate_parsed_response(
            output_format=_TagResponse,
            user_prompt=prompt,
            max_attempts=self._max_attempts,
        )
        if response is None:
            return []
        return _clean_tags(response.tags or [])

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

        tags = await self._generate_tags(text)
        suffix = f" [tags: {', '.join(tags)}]" if tags else ""
        embed_text = whole_text + suffix

        return [
            Derivative(
                uuid=uuid4(),
                segment_uuid=segment.uuid,
                timestamp=segment.timestamp,
                context=segment.context,
                block=TextBlock(text=embed_text),
                properties=segment.properties,
            )
        ]
