"""Small-chunk raw segmenter + per-chunk contextualizing deriver (v1).

Goal
----
Test the hypothesis: splitting raw conversational turns into SMALLER raw
chunks lets more segments (higher K) fit a fixed answer-token budget,
buying retrieval diversity -- WITHOUT losing contextualization, because
the deriver embeds a fully reference-resolved key per chunk.

This is the "raw-text segments + v22-like deriver" target architecture,
applied at SUB-TURN granularity.

Architecture
------------
``WindowChunkSegmenter`` (deterministic, no LLM):
  - Splits each event's text via RecursiveCharacterTextSplitter at a
    small ``chunk_size`` (sentence/clause separators preferred).
  - block.text = the raw chunk, VERBATIM (no strip) so adjacent chunks
    of one turn reconstruct the turn when concatenated.
  - context = RawSegmentEventContext(producer, before, current_event_text)
    -- carries the FULL turn + prior turns so the deriver can resolve
    references the isolated chunk cannot.

``ChunkContextDeriver`` (LLM, one call per chunk):
  - Derives ONE dual-text derivative for THIS chunk: a 3rd-person,
    fully reference-resolved rewrite of only the chunk's content,
    concatenated with the raw chunk.
  - The full turn + prior turns are resolution context only -- content
    from outside the chunk is never emitted (else sibling chunks of one
    turn would get overlapping derivatives and retrieve degenerately).
  - A content-free chunk (greeting/sign-off/bare ack) -> empty rewrite
    -> zero derivatives -> chunk invisible to retrieval (saves budget).

Display + answer path (framework, unchanged):
  - Each chunk renders ``[<ts>] {producer}: {raw_chunk}``.
  - ``string_from_segment_context`` concatenates consecutive segments
    that share event_uuid + index under ONE header -- so when sibling
    chunks of a turn are retrieved together they stitch back into the
    contiguous turn for free (no overlap, no stitch metadata needed).
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

# Sentence/clause-preferring separators: the splitter cuts at the
# earliest separator in this list that yields chunks under chunk_size,
# so a multi-sentence turn breaks at sentence boundaries before it ever
# breaks mid-sentence.
_SEPARATORS = [
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
]


class WindowChunkSegmenter(Segmenter):
    """Deterministic raw-chunk segmenter with a tunable small chunk size.

    Args:
        chunk_size: Max code-point length per chunk. For LoCoMo (median
            turn ~109 chars) a value near/below the median splits a
            meaningful fraction of turns into sentence-level chunks.
            A large value (>= longest turn) degenerates to one chunk
            per turn (the whole-turn raw baseline).
    """

    def __init__(self, *, chunk_size: int = 110) -> None:
        self._chunk_size = chunk_size
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=0,
            separators=_SEPARATORS,
            keep_separator="end",
        )

    @override
    async def segment(self, event: Event) -> list[Segment]:
        match event.context:
            case SurroundingEventsContext(producer=producer, before=before):
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
                        # Keep chunks verbatim (no strip): with
                        # keep_separator="end" sibling chunks of one
                        # turn concatenate back to the source turn.
                        if not chunk.strip():
                            continue
                        segments.append(
                            Segment(
                                uuid=uuid4(),
                                event_uuid=event.uuid,
                                index=block_index,
                                offset=offset,
                                timestamp=event.timestamp,
                                block=TextBlock(text=chunk),
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


PROMPT_CHUNK_DERIVE_V1 = """\
You are generating ONE retrieval key for a CHUNK -- a span of a longer \
message. The chunk's raw text is stored verbatim and concatenated with \
your key as the embedding input. Your key must let a future question \
about anything in THE CHUNK match this chunk.

OUTPUT a JSON object {{ "rewrite": "..." }}.

The rewrite is a single string -- one or more sentences -- in the \
third person about {speaker}, paraphrasing the chunk's content from a \
different angle than its raw wording.

DERIVE ONLY THE CHUNK. The FULL MESSAGE and PRIOR TURNS are given \
solely so you can resolve references the isolated chunk cannot -- \
pronouns, demonstratives, who ``you`` is, what ``it``/``that``/``the \
trip`` points to, and relative-time anchors. Content that lives in \
the rest of the message or in prior turns is NEVER put into the \
rewrite -- only what THIS CHUNK itself states. Pulling in content \
from outside the chunk is a FAILURE.

KEEP every CONCRETE PARTICULAR the chunk states, verbatim -- names of \
people, places, organizations, brands; dates, times, durations, \
quantities; identifiers, titles, quoted phrases; decisions, plans, \
preferences, opinions, relationships, emotional states tied to \
events; described events; attached-media descriptions. Resolve \
first-person ``I``/``my``/``me`` to {speaker}; resolve second-person \
``you`` to the addressee's name when known; resolve demonstratives to \
their concrete referents.

REPORT content as facts, not as speech acts. Drop ``{speaker} said \
that ...`` / ``{speaker} told X that ...`` wrappers unless the speech \
act itself is the event (a promise, an apology, an explicit \
announcement).

DATE AND TIME: keep natural relative phrases verbatim (``yesterday``, \
``next month``, ``three years ago``); keep explicit absolute dates \
the speaker stated. DO NOT translate a relative phrase into an \
absolute date. DO NOT add an inline ``[date]`` or ``On YYYY-MM-DD`` \
anchor the speaker did not state.

DROP A CONTENT-FREE CHUNK. When the chunk is ENTIRELY a phatic opener \
or closer, a bare acknowledgment, or a reaction with no specific \
content of its own (``Hi``, ``Hey there``, ``Take care``, ``Sounds \
great``, ``Haha yeah``, standalone ``Thanks``), output an empty \
string ``""`` -- the chunk becomes invisible to retrieval, which is \
correct. A short chunk that DOES carry a specific particular \
(``leaving at 5 on Tuesday``, ``picked the blue one``) is NOT \
content-free -- write a rewrite for it.

{neighbors_block}FULL MESSAGE FROM {speaker} on {date}:
{passage}

THE CHUNK (derive only this):
{chunk}"""


class _RewriteResponse(BaseModel):
    rewrite: str


def _format_neighbors(before: list[SurroundingEvent]) -> str:
    lines: list[str] = []
    if before:
        lines.append("PRIOR TURNS (resolution context only, do not emit):")
        for ev in before:
            lines.append(f"- {ev.producer}: {ev.text}")
        lines.append("")
    return "\n".join(lines) + ("\n" if lines else "")


class ChunkContextDeriver(Deriver):
    """Per-chunk dual-text deriver.

    Emits ONE derivative per chunk: ``{3p_rewrite}\\n{speaker}: {chunk}``.
    A content-free chunk emits zero derivatives. Reads
    RawSegmentEventContext for producer + prior turns + the full turn.
    """

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_CHUNK_DERIVE_V1,
        max_attempts: int = 3,
    ) -> None:
        self._language_model = language_model
        self._prompt_template = prompt_template
        self._max_attempts = max_attempts

    @override
    async def derive(self, segment: Segment) -> list[Derivative]:
        if not isinstance(segment.block, TextBlock):
            return []

        if not isinstance(segment.context, RawSegmentEventContext):
            # Fallback for non-experimental contexts: pass-through.
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

        producer = segment.context.producer
        before = segment.context.before
        event_text = segment.context.current_event_text
        chunk = segment.block.text
        date_str = segment.timestamp.strftime("%Y-%m-%d")
        neighbors_block = _format_neighbors(before)

        prompt = self._prompt_template.format(
            speaker=producer,
            date=date_str,
            passage=event_text,
            chunk=chunk,
            neighbors_block=neighbors_block,
        )
        response = await self._language_model.generate_parsed_response(
            output_format=_RewriteResponse,
            user_prompt=prompt,
            max_attempts=self._max_attempts,
        )
        if response is None:
            return []
        rewrite = response.rewrite.strip()
        if not rewrite:
            # Content-free chunk -> no derivative -> invisible to retrieval.
            return []

        dual_text = f"{rewrite}\n{producer}: {chunk.strip()}"
        return [
            Derivative(
                uuid=uuid4(),
                segment_uuid=segment.uuid,
                timestamp=segment.timestamp,
                context=segment.context,
                block=TextBlock(text=dual_text),
                properties=segment.properties,
            )
        ]
