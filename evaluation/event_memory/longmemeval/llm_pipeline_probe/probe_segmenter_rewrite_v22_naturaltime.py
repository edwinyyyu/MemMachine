"""v22-naturaltime -- v22 baseline with NO relative-date resolution.

Hypothesis
----------

v22 (and prior) instruct the model to resolve EVERY relative time reference
into an absolute YYYY-MM-DD anchored at the message date. This:

1. Fabricates false precision: ``next month`` becomes ``2023-04-23`` (an
   arbitrary day), when the speaker meant a SPAN. ``the weekend``,
   ``recently``, ``a few years ago`` get pinned to single days that the
   speaker never asserted.
2. Adds no retrieval value: ``YYYY-MM-DD`` strings don't help embeddings
   (date tokens carry no semantic weight) and BM25 only matches when the
   query happens to phrase the date the same way (rare).
3. Costs ~7-15 tokens per segment carrying an inline anchor.

Empirical: dropping all inline message-dates (nomsgdate) lost -57 on fb
K=7. But that variant ALSO kept the absolute-date-resolution rule for
different-day events, just moved them inline. The current variant tests
the opposite: keep natural relative phrasing intact AND drop all inline
date anchors. The QA reader has the message timestamp prepended to every
segment (``[Friday, December 8, 2023 at 7:42 PM]``) and does the date
math at answer time using natural phrasing.

Architecture
------------

block.text = clean 3p rewrite with NATURAL relative phrasing preserved
context = RewriteContext(text_to_embed = "{rewrite}\\n{speaker}: {raw_chunk}")

Identical to v22 baseline EXCEPT:
- Removed: "Resolves every relative time reference into an absolute date..."
- Removed: "Is anchored to the date of the event it describes -- {date}..."
- Added: NATURAL TIMES rule — preserve relative phrases verbatim; the
  reader sees the message timestamp prepended and resolves against that.
- Added: explicit absolute dates the speaker stated ("June 14, 2025",
  "September 2022", "in 2010") stay verbatim.

What this fixes vs v22
----------------------

- ``next month`` no longer becomes a specific YYYY-MM-DD day (preserves
  span semantics).
- Speech-act-plus-inline-date ambiguity disappears (no inline dates).
- ~83% of v22 segments carry an inline date; this variant should drop
  that to near 0% (only when the SPEAKER stated an absolute date).
- Tokens-per-segment should drop ~5-10%.

Anticipated outcome
-------------------

If hypothesis holds: parity or lift on cat2 (temporal) because the model
no longer fabricates wrong specific dates from vague phrasing; lift on
cat3/cat4 from token efficiency; small lift overall.

If hypothesis fails: cat2 regression because some genuinely useful
resolved dates were doing real retrieval work (e.g., when query asks
"in April 2023" and segment now says "next month" instead of
"2023-04-23"). In that case, retreat to a hybrid where ONLY clearly-
absolute resolutions ("three years ago" -> a specific year) are
preserved.
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


PROMPT_REWRITE_V22_NATURALTIME = """\
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
names, dates, numbers, identifiers, decisions, plans, preferences, \
opinions, relationships, emotional states tied to events, \
distinctive phrasing, attached-media descriptions. Generic \
abstractions or stock paraphrases for specifics are FAILURES.
- Preserves polarity, direction, and emotional tone. "Used to" \
implies no longer; "didn't get to bed until 2 AM" implies a late \
end, not a late start.
- Preserves NATURAL RELATIVE TIMES verbatim. Phrases like \
``yesterday``, ``today``, ``tonight``, ``last week``, ``next month``, \
``three years ago``, ``the weekend``, ``the holidays``, ``recently``, \
``now``, ``just``, ``a few days ago`` are kept AS-IS. Do NOT translate \
them into ``YYYY-MM-DD``, ``January 2023``, or any other fabricated \
absolute form. The reader sees the message's send time prepended to \
each statement (e.g. ``[Friday, December 8, 2023 at 7:42 PM]``) and \
resolves relative phrases against THAT anchor. Translating a span \
phrase (``next month``, ``the weekend``) into a single day is a \
FAILURE; translating any relative phrase into ``YYYY-MM-DD`` is a \
FAILURE.
- Preserves EXPLICIT ABSOLUTE dates that the speaker stated verbatim \
(``June 14, 2025``, ``September 2022``, ``in 2010``, ``March 3rd``). \
Do NOT add any inline ``on YYYY-MM-DD`` anchor that the speaker did \
not state. ``On YYYY-MM-DD, ...`` sentence prefixes, ``(YYYY-MM-DD)`` \
parentheticals, ``[YYYY-MM-DD]`` brackets, and trailing ``on \
YYYY-MM-DD`` suffixes are FAILURES.
- Traces every detail back to words in the message. Inferred \
attributes (a title from context, a gender from a name, an age, an \
ethnicity, an unstated role) are FAILURES; explicitly stated \
attributes are preserved.
- Reports the content the message conveys, not the speech-act of \
conveying it. "X said that ...", "X told Y that ...", "X mentioned \
that ..." framing is included ONLY when the speech-act itself is \
the event (an apology, a promise, an explicit announcement).

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
    """v22 baseline with natural relative phrasing and no inline-date anchor."""

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_REWRITE_V22_NATURALTIME,
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
