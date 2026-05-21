"""Subject-line deriver v1.

Adds a second retrieval anchor per segment: a very SHORT (3-5 word)
subject-line derivative, shaped like a Wikipedia article title or an
email subject line. The original whole-text derivative is still
emitted so specifics-anchored queries continue to match the segment.

Hypothesis
----------

A subject-line derivative -- forced to 3-5 words containing the
entity name plus a 1-3 word descriptor -- provides a focused
retrieval anchor that is:

1. Query-shaped. Real search queries are typically short noun
   phrases ("Alice's languages", "Bob birthday cake"). A 3-5 word
   derivative lives in the same embedding region as those queries,
   whereas a 12-word paraphrase ("languages that Alice speaks
   fluently as of last year") drifts toward statement-shape and
   dilutes the match.

2. Paraphrase-noise free. The hard length cap forces the LLM to
   encode ONLY the most-distinctive identifier of the segment. It
   cannot pad with filler verbs, hedges, or generic category
   nouns that don't disambiguate. The signal-to-noise ratio of the
   embedding goes up because there is almost no room for noise.

3. Complementary to the whole-text embedding. The whole-text
   derivative anchors on specifics (proper nouns, dates,
   quantities). The subject-line derivative anchors on the
   topic-label of the segment. Together they cover both
   specifics-anchored queries ("What did Alice say about the
   Lakers?") and topic-anchored queries ("Alice basketball
   loyalty").

Risks
-----

- Under-disambiguation: 3-5 words may be too few to separate
  segments that share an owner+topic (e.g., two segments about
  "Alice's languages" with different details). Mitigation: the
  whole-text derivative remains the primary anchor; the subject
  line only adds a SECOND anchor, never replaces.

- Collapse: multiple segments may produce identical subject lines.
  This is acceptable -- retrieval dedupes by segment_uuid
  post-pool, so two segments tied on subject line are reranked by
  their distinct whole-text embeddings.

- Over-pruning: very short derivatives may drop a load-bearing
  cue (an unusual proper noun) that would have matched a query.
  Mitigation: the whole-text derivative still carries that cue;
  the subject line is purely additive.

- Filler segments: pure phatic segments have no useful subject
  line. The deriver emits "" in that case and only the whole-text
  derivative is stored.

Generalizability argument
-------------------------

Subject-line shape is domain-agnostic: any segment about a person,
project, place, or entity has a natural 3-5 word topic-label
("Bob's wedding plans", "Charlie pottery hobby", "Dana
Portuguese fluency"). The prompt examples use neutral domains
(Alice/Bob/Charlie/Dana with mandolin/cake/dog/team) so the prompt
does not leak the eval distribution. The deriver is paired with
the v22 segmenter, whose segments are already self-contained
third-person narratives -- the segment text alone suffices to
produce a subject line; no neighbor context is needed.

Output
------

One or two derivatives per segment:

  1. WHOLE -- block.text = _format_with_context(segment.context,
     segment.block.text). Identical to WholeTextDeriver.

  2. SUBJECT_LINE -- block.text = LLM-generated 3-5 word
     subject-line phrase. EMITTED ONLY when non-empty. Stored
     under NullContext so the embed is FOCUSED on just those 3-5
     words (no producer prefix, no rewrite context).

Both derivatives share segment_uuid. The vector store ingests
them independently; retrieval dedupes by segment_uuid post-pool.

Cost
----

One small LLM call per segment at ingest. Parallelizable across
the corpus. With gpt-5.4-nano @ low, ingest cost roughly doubles
over the WholeTextDeriver baseline; vector storage doubles per
segment for non-filler segments, no change for filler.

Caller guidance
---------------

Use this deriver paired with the v22 RewriteSegmenter and a
SurroundingEventsContext with before-window=8 (recommended
segmenter configuration). The subject line is generated from the
segment TEXT only -- the segment is already self-contained after
segmentation.
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


PROMPT_SUBJECTLINE_V1 = """\
Write a SUBJECT LINE for the FACT -- a 3-5 word topic label, \
shaped like a Wikipedia article title or an email subject line.

The subject line is a focused retrieval anchor. A future user \
searching for this fact will type a short query like the subject \
line itself ("Alice's languages", "Bob's birthday cake"). The \
subject line must live in that query-shaped region of embedding \
space.

Rules:
- 3 to 5 words. No longer. No shorter unless impossible.
- MUST include the entity name (the OWNER/subject of the fact) \
plus a 1-3 word descriptor of the topic.
- Use a possessive ("Alice's hobbies") OR an entity + noun phrase \
("Alice basketball loyalty") -- whichever reads more naturally.
- DROP specific answer values (proper-noun titles, exact dates, \
exact numbers, item lists). The subject line is a TOPIC LABEL, \
not a fact restatement.
- DROP filler verbs (said, mentioned, told, shared) and hedges \
(probably, maybe).
- No leading "Subject:" or "Topic:" wrapper. Just the phrase.
- If the fact has no retrievable structure (pure phatic filler, \
no entity, no topic), emit "".

Examples:

FACT: "Alice supports the Lakers basketball team."
SUBJECT LINE: "Alice basketball loyalty"

FACT: "Charlie's hobbies include pottery and climbing."
SUBJECT LINE: "Charlie's hobbies"

FACT: "Bob baked a chocolate-cardamom cake for Alice's birthday \
on 2024-09-12."
SUBJECT LINE: "Bob's birthday baking"

FACT: "Dana speaks Portuguese and conversational Mandarin."
SUBJECT LINE: "Dana's languages"

FACT: "Alice adopted a golden retriever named Mochi on \
2023-05-04."
SUBJECT LINE: "Alice's dog adoption"

FACT: "Charlie plays mandolin in a weekend folk band."
SUBJECT LINE: "Charlie's mandolin band"

FACT: "Bob plans to move to Lisbon next spring for a software \
job."
SUBJECT LINE: "Bob's Lisbon move"

FACT: "Dana recommended a book about urban beekeeping to Alice."
SUBJECT LINE: "Dana's book recommendation"

FACT: "Yeah, that sounds great."
SUBJECT LINE: ""

Output: a JSON object {{ "subject_line": "..." }}.

FACT: {segment_text}"""


class _SubjectLineResponse(BaseModel):
    subject_line: str


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
    """Emits 1 or 2 derivatives per segment: whole + (optional) subject line.

    Args:
        language_model: LanguageModel used to generate the subject
            line. Configure model + reasoning_effort at
            construction.
        prompt_template: A ``.format(segment_text=...)`` template.
            Defaults to PROMPT_SUBJECTLINE_V1.
        max_attempts: Retries on retryable language-model errors.
    """

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_SUBJECTLINE_V1,
        max_attempts: int = 3,
    ) -> None:
        self._language_model = language_model
        self._prompt_template = prompt_template
        self._max_attempts = max_attempts

    async def _generate_subject_line(self, segment_text: str) -> str:
        prompt = self._prompt_template.format(segment_text=segment_text)
        response = await self._language_model.generate_parsed_response(
            output_format=_SubjectLineResponse,
            user_prompt=prompt,
            max_attempts=self._max_attempts,
        )
        if response is None:
            return ""
        return (response.subject_line or "").strip()

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

        # Generate the subject-line derivative; skip if empty so
        # filler segments stay single-derivative.
        subject_line = await self._generate_subject_line(text)
        if subject_line:
            derivatives.append(
                Derivative(
                    uuid=uuid4(),
                    segment_uuid=segment.uuid,
                    timestamp=segment.timestamp,
                    context=NullContext(),
                    block=TextBlock(text=subject_line),
                    properties=segment.properties,
                )
            )

        return derivatives
