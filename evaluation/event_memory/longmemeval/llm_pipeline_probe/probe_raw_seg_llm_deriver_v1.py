"""Raw-segmenter + LLM-deriver architecture (v1).

Architectural premise (validated by v22-rawev experiment)
---------------------------------------------------------

v22-rawev showed that swapping v22's rewritten segments for raw original
messages at answer time (while keeping v22's retrieval) ties v22-baseline
on accuracy. The segmenter's win is entirely RETRIEVAL — the rewritten
text adds zero answer-side benefit at this budget.

This decouples segment (faithful representation shown to the answerer)
from derivative (retrieval-targeted, hallucination-allowed). The deriver
is now a free-for-all optimizing for one thing: matching queries to
segments. It can:

- emit ZERO derivatives → drops segment from retrieval entirely (perfect
  for greetings, acknowledgments, "lol", contentless replies)
- emit MULTIPLE derivatives → multiple retrieval angles per segment
  (good for content-rich messages mentioning multiple distinct facts)
- emit aggressive paraphrases / topical tags / hallucinated bridges
  (without corrupting what the answerer sees)

Architecture
------------

Segmenter (`RawChunkSegmenter`):
  - Splits each event's text into chunks (single-chunk for LoCoMo)
  - Sets context = RawSegmentEventContext(producer, before, current_event_text)
  - block.text = raw chunk content (UNMODIFIED from source)

Deriver (`LLMRewriteDeriver`):
  - Reads segment.context (producer + before-neighbors + full event text)
  - Reads segment.block.text (the raw chunk)
  - Generates 0..N v22-style 3p rewrites as derivative texts
  - Returns list[Derivative]; empty list = segment invisible to retrieval

Display + BM25 (framework, unchanged):
  - Display: `[<timestamp>] {producer}: {raw_chunk}` (faithful)
  - BM25: same as display (literal message text + speaker + time)

Embedding (framework, unchanged):
  - Vector embedding of each derivative text (via WholeTextDeriver-style
    pipeline; our LLMRewriteDeriver replaces WholeTextDeriver entirely)
"""

from __future__ import annotations

from typing import override
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from memmachine_server.common.language_model import LanguageModel
from memmachine_server.episodic_memory.event_memory.data_types import (
    Derivative,
    Event,
    ProducerContext,
    RawSegmentEventContext,
    Segment,
    SurroundingEvent,
    SurroundingEventsContext,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.deriver.deriver import (
    Deriver,
)
from memmachine_server.episodic_memory.event_memory.segmenter.segmenter import (
    Segmenter,
)
from pydantic import BaseModel


PROMPT_DERIVE_V1 = """\
You are generating SEARCH KEYS for a message that has already been \
stored verbatim. Your output is what semantic search will match against \
when a future user asks a question. The user will then see the ORIGINAL \
message text, not your output, so your job is solely to maximize the \
chance that future queries about any specific content in the message \
match this segment.

OUTPUT a JSON list of standalone third-person key statements about \
{speaker}.

KEEP what is specific to this message; DROP what is interchangeable \
across similar messages. Specific content includes: names of people, \
places, organizations, brands, named objects, named activities; \
dates, times, durations, quantities; identifiers, titles, quoted \
phrases, proper nouns; decisions, plans, preferences, opinions, \
relationships, roles, emotional states tied to events; described \
events (something that happened or will happen); attached-media \
descriptions. Interchangeable content has none of these -- bare \
greetings, sign-offs, acknowledgments, reactions, and questions \
whose phrasing introduces nothing from the specific list above.

When the message contains NO specific content (a bare greeting, a \
"thanks", a "lol", a generic question with no specific subject), \
output an EMPTY list. The segment will be excluded from retrieval, \
which is the correct outcome for content-free messages.

EACH KEY STATEMENT:
- Corresponds to one EVENT or one queryable particular in the message. \
A multi-sentence elaboration of the same event is ONE key. Distinct \
events each get their own key. Multiple queryable particulars in the \
same message can each get their own key if they would be searched for \
independently.
- Contains every CONCRETE PARTICULAR for its event/particular -- \
subject, action, time, place, attendees, motivation, outcome, attached \
media. Verbatim preservation of distinctive phrasing, quoted phrases, \
and attached-media descriptions.
- Refers to {speaker} by name in third person. First-person ``I``, \
``my``, ``me`` in the raw message resolve to {speaker}; second-person \
``you`` resolves to the addressee's name when known from context; \
demonstratives resolve to concrete referents.
- Reports the content the message conveys, not the speech-act of \
conveying it. ``Bob said that ...``, ``Bob told Alice that ...``, \
``Bob mentioned that ...`` framing is dropped unless the speech-act \
itself is the event (a promise, an apology, an explicit announcement, \
a question whose phrasing is the searchable particular).

DATE AND TIME HANDLING.

ALLOWED in output:
1. Natural relative phrases the speaker actually used: ``yesterday``, \
``last week``, ``next month``, ``three years ago``, ``recently``, \
``now``. KEEP verbatim.
2. Explicit absolute dates the speaker stated: ``June 14, 2025``, \
``September 2022``, ``in 2010``. KEEP verbatim.

DO NOT translate relative phrases into absolute dates. ``next month`` \
stays ``next month``, never ``April 2023``.

DO NOT add inline date anchors the speaker did not state -- no \
prepended ``[date]``, no ``On YYYY-MM-DD, ...`` prefix, no \
``... (YYYY-MM-DD)`` suffix.

NEIGHBORING TURNS appear before the message strictly to help resolve \
addressees, demonstratives, anaphora, and unresolved relative \
references. Content drawn from neighbors is NEVER emitted -- only \
content drawn from this message.

Output: a JSON object {{ "keys": [...] }}. The list is empty when the \
message has no specific content worth searching for.

{neighbors_block}MESSAGE FROM {speaker} on {date}:
{passage}"""


class _DeriveResponse(BaseModel):
    keys: list[str]


def _format_neighbors(before: list[SurroundingEvent]) -> str:
    lines: list[str] = []
    if before:
        lines.append("PRIOR TURNS (resolution context only, do not emit):")
        for ev in before:
            lines.append(f"- {ev.producer}: {ev.text}")
        lines.append("")
    return "\n".join(lines) + ("\n" if lines else "")


class RawChunkSegmenter(Segmenter):
    """Splits events into raw chunks; attaches RawSegmentEventContext.

    For LoCoMo (all messages <500 chars) every event becomes one segment
    whose block.text IS the raw message. The full event text is duplicated
    into context.current_event_text for the deriver's convenience.
    """

    def __init__(
        self,
        *,
        chunk_size: int = 1500,
    ) -> None:
        self._chunk_size = chunk_size
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=0,
            separators=[
                "\n\n\n",
                "\n\n",
                "\n",
                ". ",
                "? ",
                "! ",
                "; ",
                ": ",
                ", ",
                " ",
                "",
            ],
            keep_separator="end",
        )

    @override
    async def segment(self, event: Event) -> list[Segment]:
        match event.context:
            case SurroundingEventsContext(
                producer=producer, before=before
            ):
                speaker = producer
                before_events: list[SurroundingEvent] = list(before)
            case ProducerContext(producer=producer):
                speaker = producer
                before_events = []
            case _:
                speaker = "the speaker"
                before_events = []

        segments: list[Segment] = []
        for block_index, block in enumerate(event.blocks):
            match block:
                case TextBlock(text=text):
                    full_event_text = text
                    chunks = (
                        self._splitter.split_text(text)
                        if len(text) > self._chunk_size
                        else [text]
                    )
                    offset = 0
                    for chunk in chunks:
                        chunk_stripped = chunk.strip()
                        if not chunk_stripped:
                            continue
                        segments.append(
                            Segment(
                                uuid=uuid4(),
                                event_uuid=event.uuid,
                                index=block_index,
                                offset=offset,
                                timestamp=event.timestamp,
                                block=TextBlock(text=chunk_stripped),
                                context=RawSegmentEventContext(
                                    producer=speaker,
                                    before=before_events,
                                    current_event_text=full_event_text,
                                ),
                                properties=event.properties,
                            )
                        )
                        offset += 1
                case _:
                    raise NotImplementedError(
                        f"Unsupported block type: {type(block).__name__}"
                    )
        return segments


class LLMRewriteDeriver(Deriver):
    """LLM-based deriver that emits 0..N v22-style 3p key statements per segment.

    Reads RawSegmentEventContext for producer + before-neighbors + full
    event text. Generates retrieval-optimized derivative texts; empty
    list drops the segment from retrieval.

    Derivative texts are formatted via _format_with_context, which for
    RawSegmentEventContext prepends ``{producer}: `` -- same as v22's
    RewriteContext + text_to_embed but with the producer prefix now
    explicit at the embedder boundary.
    """

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_DERIVE_V1,
        max_attempts: int = 3,
    ) -> None:
        self._language_model = language_model
        self._prompt_template = prompt_template
        self._max_attempts = max_attempts

    @override
    async def derive(self, segment: Segment) -> list[Derivative]:
        match segment.context:
            case RawSegmentEventContext(
                producer=producer,
                before=before,
                current_event_text=current_event_text,
            ):
                speaker = producer
                before_events = before
                event_text = current_event_text
            case _:
                # Non-experimental contexts: fall back to passing the raw
                # block text through with no LLM derivation. Keeps the
                # deriver safe to mix with non-raw segmenters.
                if isinstance(segment.block, TextBlock):
                    return [
                        Derivative(
                            uuid=uuid4(),
                            segment_uuid=segment.uuid,
                            timestamp=segment.timestamp,
                            context=segment.context,
                            block=TextBlock(text=segment.block.text),
                            properties=segment.properties,
                        )
                    ]
                return []

        date_str = segment.timestamp.strftime("%Y-%m-%d")
        neighbors_block = _format_neighbors(before_events)
        prompt = self._prompt_template.format(
            speaker=speaker,
            date=date_str,
            passage=event_text,
            neighbors_block=neighbors_block,
        )
        response = await self._language_model.generate_parsed_response(
            output_format=_DeriveResponse,
            user_prompt=prompt,
            max_attempts=self._max_attempts,
        )
        if response is None:
            return []
        keys = [k.strip() for k in response.keys if k and k.strip()]
        return [
            Derivative(
                uuid=uuid4(),
                segment_uuid=segment.uuid,
                timestamp=segment.timestamp,
                context=segment.context,
                block=TextBlock(text=key),
                properties=segment.properties,
            )
            for key in keys
        ]
