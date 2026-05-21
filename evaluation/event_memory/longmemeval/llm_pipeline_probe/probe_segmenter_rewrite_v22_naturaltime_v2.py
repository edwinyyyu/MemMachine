"""v22-naturaltime-v2 -- v22 with NO date resolution AND no reader-format leakage.

What v1 got wrong
-----------------

v1's NATURAL TIMES rule said "the reader sees the message's send time
prepended to each statement (e.g. ``[Friday, December 8, 2023 at 7:42 PM]``)
and resolves relative phrases against THAT anchor". The model treated
that as a format SPECIFICATION it should emit -- 33.3% of v1 segments
START with a fabricated ``[Friday, January 21, 2022]`` prefix duplicating
the framework header.

It also retained 21.2% inline ``YYYY-MM-DD`` despite an explicit FAILURE
rule, suggesting the rule wasn't strict enough about ALL inline-date
forms (speech-act "said ... on YYYY-MM-DD" suffix kept slipping through).

v2 fixes
--------

1. DROP all reference to what the reader sees -- never tell the model
   about the framework header format.
2. State the date-emission rule POSITIVELY: "Do not add a date or time
   prefix, suffix, or parenthetical to any statement."
3. Allow ONLY two date forms in output: (a) natural relative phrases the
   speaker used (``yesterday``, ``next month``), (b) explicit absolute
   dates the speaker stated (``June 14, 2025``, ``September 2022``,
   ``in 2010``, ``March 3rd``).
4. Enumerate ALL forbidden date forms with examples.

Architecture
------------

block.text = clean 3p rewrite with relative phrasing preserved
context = RewriteContext(text_to_embed = "{rewrite}\\n{speaker}: {raw_chunk}")

Identical to v22 baseline EXCEPT the time-handling rules.
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
from pydantic import BaseModel


PROMPT_REWRITE_V22_NATURALTIME_V2 = """\
Rewrite the MESSAGE into a JSON list of standalone third-person \
memory statements. Each statement is stored verbatim and later \
retrieved by semantic search. A future user querying any specific \
content in the message should find at least one statement that \
contains that content.

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

EACH STATEMENT:
- Corresponds to one EVENT in the message, not to one sentence. An \
event is a single occurrence, decision, plan, observation, state, \
or preference at one point or span in time. A multi-sentence \
elaboration of the same event (subject in one sentence, reason in \
the next, outcome after that) is ONE statement that contains all of \
those sentences' particulars. Emitting one statement per sentence \
is a FAILURE; emitting one statement per concrete particular is a \
FAILURE; merging two distinct events into one statement is a \
FAILURE. Distinct events (different times, different occasions, \
different actions) each get their own statement, even when they \
share participants or topics.
- Contains every concrete particular the message gives about its \
event -- subject, action, time, place, attendees, motivation, \
outcome, attached media.
- Refers to the speaker by name. First-person self-references \
resolve to the speaker's name; second-person references resolve to \
the addressee's name when one is known; demonstrative and ambiguous \
pronouns resolve to their concrete referents. The first occurrence \
of any queryable entity is named; subsequent references within the \
same statement may use natural pronouns and possessives.
- Preserves every CONCRETE PARTICULAR from the message verbatim -- \
names, numbers, identifiers, decisions, plans, preferences, \
opinions, relationships, emotional states tied to events, \
distinctive phrasing, attached-media descriptions. Generic \
abstractions or stock paraphrases for specifics are FAILURES.
- Preserves polarity, direction, and emotional tone. "Used to" \
implies no longer; "didn't get to bed until 2 AM" implies a late \
end, not a late start.
- Traces every detail back to words in the message. Inferred \
attributes (a title from context, a gender from a name, an age, an \
ethnicity, an unstated role) are FAILURES; explicitly stated \
attributes are preserved.
- Reports the content the message conveys, not the speech-act of \
conveying it. "X said that ...", "X told Y that ...", "X mentioned \
that ..." framing is included ONLY when the speech-act itself is \
the event (an apology, a promise, an explicit announcement).

DATE AND TIME HANDLING.

ALLOWED date and time forms in statements:
1. Natural relative phrases the speaker actually used: ``yesterday``, \
``today``, ``tonight``, ``last week``, ``next month``, ``three \
years ago``, ``a few days ago``, ``the weekend``, ``the holidays``, \
``recently``, ``now``, ``just``. KEEP these verbatim.
2. Explicit absolute dates the speaker actually stated: ``June 14, \
2025``, ``September 2022``, ``March 3rd``, ``in 2010``. KEEP these \
verbatim.

DO NOT translate a relative phrase into an absolute date. ``next \
month`` MUST stay as ``next month``, not become ``April 2023`` or \
``2023-04-23``. ``three years ago`` MUST stay as ``three years \
ago``, not become ``2020`` or ``2020-01-23``. Forcing a span phrase \
(``next month``, ``the weekend``, ``recently``) into a single day \
is a FAILURE.

DO NOT add any date or time anchor that the speaker did not state. \
Every form listed below is a FAILURE:
- prepended bracket headers like ``[Friday, January 21, 2022] ...`` \
or ``[2022-01-21] ...`` or ``[2022-01-21 14:30] ...``
- prepended ``On YYYY-MM-DD, ...`` sentence prefix
- trailing ``... on YYYY-MM-DD`` or ``... on January 21, 2022`` \
suffix anchoring the statement to the message date
- trailing ``... (2022-01-21)`` or ``... (Date: ...)`` or \
``... (Event date: ...)`` parenthetical
- speech-act + date such as ``X said ... on 2022-01-21``, \
``X told Y ... on 2022-01-21``, ``X asked ... on 2022-01-21``

When the message refers to a current event with no explicit date and \
no relative phrase, just describe the event in natural prose WITHOUT \
any date anchor. The system tracks when each statement was sent.

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
    """v22 with natural relative phrasing and no inline-date anchor (v2)."""

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_REWRITE_V22_NATURALTIME_V2,
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
                        memories = await self._rewrite_chunk(
                            chunk_stripped, speaker, date_str, neighbors_block
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
