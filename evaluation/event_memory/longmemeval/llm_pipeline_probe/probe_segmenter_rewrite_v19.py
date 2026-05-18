"""LLM rewriting segmenter v19 — tighter principle wording, no examples.

v18 used the abstract "truth-bearing assertion" criterion for keep/
drop. At low reasoning that test fired too liberally — bare \
greetings and speech-act framing were kept. v19 replaces the
abstract test with an enumerated SEARCHABLE-CONTENT category list:
a statement is emitted iff the source segment contains at least one
listed category.

Other tightening:
- Rule 7 event definition pinned to one occurrence in time/space.
- Rule 9 (now folded into rule 10) names the categories explicitly.
- Rule wording compressed by ~30% vs v18.

Still no inline I/O examples. Still no segment-count bias.

Architecture identical to v7: RewriteContext dual-surface embed,
deterministic 1500-char split.
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
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.segmenter.segmenter import (
    Segmenter,
)
from pydantic import BaseModel

PROMPT_REWRITE_V19 = """\
Rewrite a single conversational message into a JSON list of \
self-contained third-person memory statements. Each statement is \
stored verbatim and later retrieved by semantic search.

CONTEXT:
- The message was spoken by {speaker} on {date}.
- You only see this one message.

A statement is SEARCHABLE CONTENT if it contains at least one of:
- a named person, place, organization, brand, named object, or \
named activity
- a specific date, time, duration, or quantity
- a stated decision, plan, intention, preference, opinion, hobby, \
relationship detail, role, or attribute
- a described event (something that happened or will happen)
- a description of attached media (photos, videos, files) including \
what is depicted

A message segment that has none of the above produces no statement. \
Greetings, sign-offs, bare acknowledgments, bare reactions, and \
questions whose phrasing introduces no content of the categories \
above produce no statement.

PROPERTIES (each statement must satisfy ALL):

1. THIRD PERSON. The speaker is referred to by name. First-person \
self-references resolve to the speaker's name. Second-person \
references resolve to the addressee's descriptor when the \
addressee's name is unknown. Demonstrative and ambiguous pronouns \
resolve to their concrete referents.

2. ENTITY NAMED. Every queryable entity in the statement appears \
by name at least once. Within one statement, after an entity is \
named, natural pronouns and possessives are acceptable.

3. SPECIFICS VERBATIM. Every concrete particular drawn from the \
message — names, dates, numbers, quantities, titles, brands, \
places, named objects, named activities, quoted phrases, proper \
nouns, attached-media descriptions — appears in the statement in \
the form the message uses. Vague substitutions for specifics are \
failures.

4. POLARITY AND EMOTION PRESERVED. Direction, polarity, temporal \
direction, and emotional or evaluative content from the message \
are preserved.

5. RELATIVE TIME RESOLVED. Each relative time reference becomes an \
absolute date or interval anchored at {date}: past-tense and \
backward markers resolve backward; future-tense and forward markers \
resolve forward; bare month/day references resolve to the nearest \
occurrence consistent with the surrounding tense; bare durations \
resolve to a span ending at {date}; bare shortcuts like "the \
weekend" or "the holidays" resolve to the nearest past occurrence \
when no tense marker is present. After resolution, the original \
relative phrase does not appear in the statement.

6. DATE ANCHORED. Each statement carries the date of the event it \
describes — {date} for events that took place during the message, \
or the resolved date for events that took place elsewhere. The \
date appears once.

7. ONE EVENT. A statement describes exactly one event. An event is \
one occurrence, one decision, one plan, one observation, one state, \
or one preference. Distinct events — different times, different \
occasions, different actions — each get their own statement, even \
when they share participants or topics.

8. COMPLETE FOR ITS EVENT. Every particular the message gives about \
that one event — subject, action, time, place, attendees, \
motivation, outcome, attached media — appears in the statement.

9. SOURCE ANCHORED. Every detail traces to words present in the \
message. Inferred attributes (a title from context, a gender from \
a name, an age, an ethnicity, an unstated role) are excluded. \
Explicitly stated attributes are preserved.

10. EVENT CONTENT, NOT SPEECH-ACT FRAMING. A statement reports the \
content conveyed, not the act of conveying. Speech-act framing \
("X said that", "X told Y that", "X mentioned that") is included \
only when the act itself is the event (an apology, a promise, a \
commitment, an explicit announcement).

11. UNIQUE. No two statements carry the same content.

Output: a JSON object {{ "memories": [...] }} where memories is the \
list of statements. The list is empty when the message contains no \
searchable content.

MESSAGE FROM {speaker} on {date}:
{passage}"""


class _RewriteResponse(BaseModel):
    memories: list[str]


class RewriteSegmenter(Segmenter):
    """v19 — tighter principle wording, enumerated searchable-content categories."""

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_REWRITE_V19,
        chunk_size: int = 1500,
        max_attempts: int = 3,
    ) -> None:
        self._language_model = language_model
        self._prompt_template = prompt_template
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

    async def _rewrite_chunk(self, chunk: str, speaker: str, date: str) -> list[str]:
        prompt = self._prompt_template.format(speaker=speaker, date=date, passage=chunk)
        response = await self._language_model.generate_parsed_response(
            output_format=_RewriteResponse,
            user_prompt=prompt,
            max_attempts=self._max_attempts,
        )
        if response is None:
            return []
        return [m.strip() for m in response.memories if m and m.strip()]

    @staticmethod
    def _build_embed_text(rewrite: str, original_chunk: str, speaker: str) -> str:
        return f"{rewrite}\n{speaker}: {original_chunk}"

    @override
    async def segment(self, event: Event) -> list[Segment]:
        speaker = (
            event.context.producer
            if isinstance(event.context, ProducerContext)
            else "the speaker"
        )
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
                        memories = await self._rewrite_chunk(
                            chunk_stripped, speaker, date_str
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
