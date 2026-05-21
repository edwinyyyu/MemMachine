"""v22-first-person-clean-v1 -- pure 1p rewrite: keep first-person voice, drop garbage.

Hypothesis
----------

Prior 1p variants (fp, fp-cot, fp-min, fp-dual-v1/v2, etc.) added
speaker name back via parenthetical or self-reference, which
defeated the point of 1p. The user's clarification: the rewrite
should produce ONLY first-person content. NO speaker name. NO date.
The ProducerContext attached to the segment provides the speaker,
and the segment timestamp provides the date.

This is the simplest possible 1p form. The framework prepends
``[<timestamp>] {speaker}: `` to every segment at retrieval time, so
the reader sees:

  [Friday, January 21, 2022, 12:00 PM] Bob: I won my first video
  game tournament last week.

No information is lost: speaker is in the header prefix, timestamp
is in the header prefix, content is the speaker's own voice. The
segmenter's job is just to clean the raw message -- drop filler
(greetings, sign-offs, acknowledgments, generic questions) and keep
the specific content as the speaker phrased it.

Architecture
------------

block.text = first-person clean content ("I ...", "my ...", with
            "you" resolved to addressee's name when known)
context = ProducerContext(producer=speaker)

ProducerContext is correct here because the framework supplies the
``{speaker}: `` prefix at format time, completing the natural
attribution. No dual-text embed -- block.text IS the embed input
under ProducerContext default formatting.

What the segmenter does NOT do
------------------------------

- Does NOT add speaker name. ``Bob won his first tournament...``
  is a FAILURE (3p reframing).
- Does NOT add date. ``I won on 2022-01-21`` is a FAILURE if
  speaker did not state the date.
- Does NOT add ``(Bob)`` parenthetical or any speaker self-id.
- Does NOT use third-person pronouns for the speaker (he/she/they
  referring to {speaker} are FAILURES).

What the segmenter DOES do
--------------------------

- Drops bare filler (greetings, sign-offs, acknowledgments)
- Resolves ``you`` to addressee's name when known
- Resolves demonstratives (``this``, ``that``, ``it``) to their
  concrete referents when needed
- Splits multi-event passages into one statement per event
- Preserves natural relative phrases verbatim
- Preserves explicit absolute dates the speaker stated
"""

from __future__ import annotations

from typing import override
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from memmachine_server.common.language_model import LanguageModel
from memmachine_server.episodic_memory.event_memory.data_types import (
    Event,
    ProducerContext,
    Segment,
    SurroundingEventsContext,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.segmenter.segmenter import (
    Segmenter,
)
from pydantic import BaseModel


PROMPT_REWRITE_V22_FIRST_PERSON_CLEAN_V1 = """\
Rewrite the MESSAGE into a JSON list of standalone FIRST-PERSON \
memory statements as {speaker} would say them. Each statement is \
stored verbatim and later retrieved by semantic search. A future \
user querying any specific content in the message should find at \
least one statement that contains that content.

The reader will see each statement prefixed with the message's \
timestamp and {speaker}'s name (e.g. ``[Friday, January 21, 2022 \
12:00 PM] Bob: ...``). DO NOT add the speaker's name or the date \
to the statement text itself -- the system handles those.

KEEP what is specific to this message; DROP what is interchangeable \
across similar messages. Specific content includes: names of people, \
places, organizations, brands, named objects, named activities; \
dates, times, durations, quantities; identifiers, titles, quoted \
phrases, proper nouns; decisions, plans, preferences, opinions, \
relationships, roles, emotional states tied to events; described \
events (something that happened or will happen); attached-media \
descriptions. Interchangeable content has none of these -- bare \
greetings, sign-offs, acknowledgments, reactions, and questions \
whose phrasing introduces nothing from the specific list above. \
Dropping specific content is a FAILURE; emitting a statement for \
interchangeable content is a FAILURE.

FIRST-PERSON VOICE.

Every statement is in {speaker}'s own voice using FIRST-PERSON \
pronouns. ``I``, ``me``, ``my``, ``mine``, ``myself``. NEVER refer \
to {speaker} by name. NEVER use third-person pronouns (he/she/they, \
him/her/them, his/her/their) for {speaker}. The speaker's identity \
comes from the prepended header, not from the statement text.

Examples (assume MESSAGE FROM Bob):
- Raw: "I won my first video game tournament last week."
  Statement: ``I won my first video game tournament last week.``
- Raw: "Bob said his mother passed away three years ago." (hypothetical)
  Statement: ``My mother passed away three years ago.``
- FAILURE: ``Bob won his first video game tournament last week.``
- FAILURE: ``(Bob): I won my first video game tournament last week.``
- FAILURE: ``I (Bob) won my first video game tournament last week.``

Other parties use their NAMES (resolve ``you`` to the addressee's \
name when known from context; resolve demonstratives to their \
concrete referents):
- Raw: "You should check out Alice's new bakery."
  Statement (addressee is Carol): ``Carol should check out Alice's \
new bakery.``

EACH STATEMENT:
- Corresponds to one EVENT in the message, not to one sentence. A \
multi-sentence elaboration of the same event is ONE statement. \
Distinct events each get their own statement.
- Contains every concrete particular the message gives about its \
event -- action, time, place, attendees, motivation, outcome, \
attached media.
- Preserves every CONCRETE PARTICULAR from the message verbatim -- \
numbers, identifiers, decisions, plans, preferences, opinions, \
relationships, emotional states tied to events, distinctive \
phrasing, quoted phrases, attached-media descriptions. Generic \
abstractions or stock paraphrases for specifics are FAILURES.
- Preserves polarity, direction, and emotional tone. ``Used to`` \
implies no longer; ``didn't get to bed until 2 AM`` implies a late \
end, not a late start.
- Traces every detail back to words in the message. Inferred \
attributes (a title from context, an age, an ethnicity, an \
unstated role) are FAILURES; explicitly stated attributes are \
preserved.

DATE AND TIME HANDLING.

ALLOWED in output:
1. Natural relative phrases the speaker actually used: \
``yesterday``, ``today``, ``tonight``, ``last week``, ``next \
month``, ``three years ago``, ``a few days ago``, ``the weekend``, \
``the holidays``, ``recently``, ``now``, ``just``. KEEP verbatim.
2. Explicit absolute dates the speaker actually stated: ``June 14, \
2025``, ``September 2022``, ``March 3rd``, ``in 2010``. KEEP \
verbatim.

DO NOT translate a relative phrase into an absolute date. ``next \
month`` MUST stay ``next month``, never ``April 2023``.

DO NOT add any date or time anchor the speaker did not state. Each \
of these is a FAILURE:
- bracket prefixes ``[Friday, January 21, 2022] ...`` or \
``[2022-01-21] ...``
- ``On YYYY-MM-DD, ...`` sentence prefix
- trailing ``... on YYYY-MM-DD`` or ``... (2022-01-21)``

NEIGHBORING TURNS appear before and after the message strictly to \
help resolve second-person addressees, demonstratives, anaphora, \
and unresolved referents. Content drawn from the neighbors is NEVER \
emitted -- only content drawn from the message itself.

Output: a JSON object {{ "memories": [...] }}. The list is empty \
when the message contains no specific content.

{neighbors_block}MESSAGE FROM {speaker} on {date}:
{passage}"""


class _RewriteResponse(BaseModel):
    memories: list[str]


def _format_neighbors(before: list, after: list, current_speaker: str) -> str:
    lines = []
    if before:
        lines.append("PRIOR TURNS (context only, do not emit):")
        for ev in before:
            lines.append(f"- {ev.producer}: {ev.text}")
        lines.append("")
    if after:
        lines.append("LATER TURNS (context only, do not emit):")
        for ev in after:
            lines.append(f"- {ev.producer}: {ev.text}")
        lines.append("")
    return "\n".join(lines) + ("\n" if lines else "")


class RewriteSegmenter(Segmenter):
    """Pure first-person rewrite: no speaker name, no date in segment text."""

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_REWRITE_V22_FIRST_PERSON_CLEAN_V1,
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
                        memories = await self._rewrite_chunk(
                            chunk_stripped, speaker, date_str, neighbors_block
                        )
                        for memory in memories:
                            segments.append(
                                Segment(
                                    uuid=uuid4(),
                                    event_uuid=event.uuid,
                                    index=block_index,
                                    offset=offset,
                                    timestamp=event.timestamp,
                                    block=TextBlock(text=memory),
                                    context=ProducerContext(producer=speaker),
                                    properties=event.properties,
                                )
                            )
                            offset += 1
                case _:
                    raise NotImplementedError(
                        f"Unsupported block type: {type(block).__name__}"
                    )
        return segments
