"""Raw small-chunk segmenter with decoupled retrieval text (v1).

Target architecture: raw-text segments + a v22-like deriver, with the
retrieval representation fully decoupled from the display text.

Per chunk the segmenter emits a ``Segment`` whose:
  - ``block.text`` is the RAW conversation chunk -- verbatim, shown to
    the answering model (zero rewrite information loss).
  - ``context`` is a ``DecoupledRetrievalContext`` carrying a clean,
    reference-resolved 3rd-person rewrite of the chunk in BOTH
    ``text_to_embed`` (semantic retrieval) and ``text_to_score_bm25``
    (lexical retrieval).

Why decouple. When BM25 scores the raw turn, speaker-name vocatives in
greeting/filler turns ("Hey Caroline!") false-match name-bearing
queries -- a confirmed failure mode of raw-text segments. Routing BM25
(and the embedder) onto the clean rewrite removes vocatives/filler from
both retrieval channels while the answerer still reads raw text.

Splitting turns into sub-turn chunks (``chunk_size`` near the LoCoMo
median turn length) lets more segments fit a fixed answer-token budget
-- retrieval diversity -- and isolates a turn's greeting clause into
its own chunk that the rewrite step then drops.

The deriver paired with this segmenter is the trivial ``WholeTextDeriver``:
it reads ``DecoupledRetrievalContext.text_to_embed`` and passes it
through as the embedded derivative.
"""

from __future__ import annotations

import asyncio
from typing import override
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from memmachine_server.common.language_model import LanguageModel
from memmachine_server.episodic_memory.event_memory.data_types import (
    DecoupledRetrievalContext,
    Event,
    ProducerContext,
    Segment,
    SurroundingEvent,
    SurroundingEventsContext,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.segmenter.segmenter import (
    Segmenter,
)
from pydantic import BaseModel

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


PROMPT_CHUNK_REWRITE_V1 = """\
You are generating the RETRIEVAL TEXT for a CHUNK -- a span of a longer \
conversation message. The chunk's raw text is what a future reader \
sees; your retrieval text is what semantic search and lexical search \
score. Your job: make a future question about anything in THE CHUNK \
match this chunk.

OUTPUT a JSON object {{ "rewrite": "..." }}.

The rewrite is a single string -- one or more sentences -- in the \
third person about {speaker}, stating the chunk's content plainly.

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
string ``""`` -- the chunk drops out of retrieval, which is correct. \
A short chunk that DOES carry a specific particular (``leaving at 5 \
on Tuesday``, ``picked the blue one``) is NOT content-free -- write a \
rewrite for it.

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


class RawChunkRewriteSegmenter(Segmenter):
    """Splits turns into small raw chunks; emits ``DecoupledRetrievalContext``.

    Args:
        language_model: rewrites each chunk into a clean retrieval text.
        chunk_size: max code-point length per raw chunk. Near the LoCoMo
            median turn length (~109) splits a meaningful fraction of
            turns at sentence boundaries.
        max_attempts: language-model retry budget.

    A chunk whose rewrite is empty (content-free) emits no segment.
    """

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        chunk_size: int = 110,
        prompt_template: str = PROMPT_CHUNK_REWRITE_V1,
        max_attempts: int = 3,
    ) -> None:
        self._language_model = language_model
        self._chunk_size = chunk_size
        self._prompt_template = prompt_template
        self._max_attempts = max_attempts
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=0,
            separators=_SEPARATORS,
            keep_separator="end",
        )

    async def _rewrite_chunk(
        self,
        *,
        chunk: str,
        full_event_text: str,
        speaker: str,
        date: str,
        neighbors_block: str,
    ) -> str:
        prompt = self._prompt_template.format(
            speaker=speaker,
            date=date,
            passage=full_event_text,
            chunk=chunk,
            neighbors_block=neighbors_block,
        )
        response = await self._language_model.generate_parsed_response(
            output_format=_RewriteResponse,
            user_prompt=prompt,
            max_attempts=self._max_attempts,
        )
        if response is None:
            return ""
        return response.rewrite.strip()

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
        date_str = event.timestamp.strftime("%Y-%m-%d")
        neighbors_block = _format_neighbors(before_events)

        segments: list[Segment] = []
        for block_index, block in enumerate(event.blocks):
            match block:
                case TextBlock(text=text):
                    chunks = (
                        self._splitter.split_text(text)
                        if len(text) > self._chunk_size
                        else [text]
                    )
                    chunks = [c for c in chunks if c.strip()]
                    rewrites = await asyncio.gather(
                        *(
                            self._rewrite_chunk(
                                chunk=chunk,
                                full_event_text=text,
                                speaker=speaker,
                                date=date_str,
                                neighbors_block=neighbors_block,
                            )
                            for chunk in chunks
                        )
                    )
                    offset = 0
                    for chunk, rewrite in zip(chunks, rewrites, strict=True):
                        if not rewrite:
                            # Content-free chunk: emit no segment.
                            continue
                        segments.append(
                            Segment(
                                uuid=uuid4(),
                                event_uuid=event.uuid,
                                index=block_index,
                                offset=offset,
                                timestamp=event.timestamp,
                                block=TextBlock(text=chunk),
                                context=DecoupledRetrievalContext(
                                    producer=speaker,
                                    text_to_embed=rewrite,
                                    text_to_score_bm25=rewrite,
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
