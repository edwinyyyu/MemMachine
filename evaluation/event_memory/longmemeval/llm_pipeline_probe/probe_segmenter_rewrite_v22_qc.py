"""LLM rewriting segmenter v22 + quality-check pass.

Two-pass architecture:
  Pass 1: v22 segmenter produces initial statements
  Pass 2: review LLM takes (message, statements) and emits corrected
          statements -- adds missing particulars, resolves remaining
          relative dates, merges same-event splits, names unresolved
          pronouns.

Cost: 2x LLM calls per message (vs single-pass v22). Per-user note this
is a "last resort for production" experiment to gauge headroom.
"""

from __future__ import annotations

from typing import override
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from memmachine_server.common.language_model import LanguageModel
from memmachine_server.episodic_memory.event_memory.data_types import (
    Event,
    ProducerContext,
    RewriteContext,
    Segment,
    SurroundingEventsContext,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.segmenter.segmenter import (
    Segmenter,
)
from probe_segmenter_rewrite_v22 import (
    PROMPT_REWRITE_V22,
    _format_neighbors,
    _RewriteResponse,
)

PROMPT_REVIEW_V1 = """\
The MESSAGE below was rewritten into a list of standalone third-person \
memory statements (STATEMENTS). Your job is to REVIEW and CORRECT the \
list so the FINAL output satisfies every requirement below. Output a \
JSON list of CORRECTED statements -- the same items if no change is \
needed.

Each FINAL statement must:
- Preserve every CONCRETE PARTICULAR present in the MESSAGE -- names, \
dates, numbers, identifiers, decisions, plans, preferences, opinions, \
relationships, emotional states tied to events, attached-media \
descriptions, distinctive phrasing. If any particular from the MESSAGE \
is not yet covered by any STATEMENT, ADD a statement that contains it.
- Describe one EVENT in the message. If two STATEMENTS describe the \
same event (same date, same subject, same action), MERGE them into \
one that contains every particular from both, then drop the duplicate.
- Use the speaker's name, the addressee's name, and concrete entity \
names. Pronouns and demonstratives without antecedent (he, she, \
they, it, this, that) RESOLVE to the entity they reference.
- Contain no relative time phrases ("last week", "yesterday", "the \
weekend", bare month names) after resolution. Replace each with the \
absolute date or interval anchored at {date}.
- Report the content the MESSAGE conveys, not the speech-act \
("X said that ...", "X told Y that ...", "X mentioned that ..."), \
unless the speech-act itself is the event (apology, promise, explicit \
announcement). FOLD speech-act-wrapped fragments into a single \
content-bearing statement.
- Drop interchangeable filler -- bare greetings, sign-offs, \
acknowledgments, reactions whose phrasing introduces no specific \
content.

Output: a JSON object {{ "memories": [...] }} and nothing else.

{neighbors_block}MESSAGE FROM {speaker} on {date}:
{passage}

INITIAL STATEMENTS:
{statements}"""


class RewriteSegmenter(Segmenter):
    """v22 + quality-check second pass."""

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_REWRITE_V22,
        review_template: str = PROMPT_REVIEW_V1,
        chunk_size: int = 1500,
        max_attempts: int = 3,
    ) -> None:
        self._language_model = language_model
        self._prompt_template = prompt_template
        self._review_template = review_template
        self._chunk_size = chunk_size
        self._max_attempts = max_attempts
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

    async def _rewrite_chunk(
        self, chunk: str, speaker: str, date: str, neighbors_block: str
    ) -> list[str]:
        prompt = self._prompt_template.format(
            speaker=speaker,
            date=date,
            passage=chunk,
            neighbors_block=neighbors_block,
        )
        response = await self._language_model.generate_parsed_response(
            output_format=_RewriteResponse,
            user_prompt=prompt,
            max_attempts=self._max_attempts,
        )
        if response is None:
            return []
        return [m.strip() for m in response.memories if m and m.strip()]

    async def _review_chunk(
        self,
        chunk: str,
        speaker: str,
        date: str,
        neighbors_block: str,
        initial_statements: list[str],
    ) -> list[str]:
        if not initial_statements:
            return []
        rendered = "\n".join(f"- {s}" for s in initial_statements)
        prompt = self._review_template.format(
            speaker=speaker,
            date=date,
            passage=chunk,
            neighbors_block=neighbors_block,
            statements=rendered,
        )
        response = await self._language_model.generate_parsed_response(
            output_format=_RewriteResponse,
            user_prompt=prompt,
            max_attempts=self._max_attempts,
        )
        if response is None:
            return initial_statements
        corrected = [m.strip() for m in response.memories if m and m.strip()]
        return corrected or initial_statements

    @staticmethod
    def _build_embed_text(rewrite: str, original_chunk: str, speaker: str) -> str:
        return f"{rewrite}\n{speaker}: {original_chunk}"

    @override
    async def segment(self, event: Event) -> list[Segment]:
        match event.context:
            case SurroundingEventsContext(
                producer=producer, before=before, after=after
            ):
                speaker = producer
                neighbors_block = _format_neighbors(before, after, producer)
            case ProducerContext(producer=producer):
                speaker = producer
                neighbors_block = ""
            case _:
                speaker = "the speaker"
                neighbors_block = ""
        date_str = event.timestamp.strftime("%Y-%m-%d")

        segments: list[Segment] = []
        for block_index, block in enumerate(event.blocks):
            match block:
                case TextBlock(text=text):
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
                        initial = await self._rewrite_chunk(
                            chunk_stripped, speaker, date_str, neighbors_block
                        )
                        memories = await self._review_chunk(
                            chunk_stripped,
                            speaker,
                            date_str,
                            neighbors_block,
                            initial,
                        )
                        for memory in memories:
                            embed_text = RewriteSegmenter._build_embed_text(
                                memory, chunk_stripped, speaker
                            )
                            segments.append(
                                Segment(
                                    uuid=uuid4(),
                                    event_uuid=event.uuid,
                                    index=block_index,
                                    offset=offset,
                                    timestamp=event.timestamp,
                                    block=TextBlock(text=memory),
                                    context=RewriteContext(text_to_embed=embed_text),
                                    properties=event.properties,
                                )
                            )
                            offset += 1
                case _:
                    raise NotImplementedError(
                        f"Unsupported block type: {type(block).__name__}"
                    )
        return segments
