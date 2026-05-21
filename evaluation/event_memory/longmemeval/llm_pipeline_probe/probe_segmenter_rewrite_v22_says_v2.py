"""v22-says v2: says framing + v22 baseline's absolute-date resolution.

Hypothesis under test
---------------------
v22-says v1 reframed each statement as ``{speaker} says ...`` AND
preserved relative time phrases verbatim (``yesterday``, ``last week``)
on the theory that the framework's chronological prefix +
``[<long-date>]`` annotation at retrieval time gives the answerer
enough anchor to compute the absolute date itself.

The empirical result on g3 (K=7, gpt-5-mini answer+judge, mem0-bench,
BM25 additive 0.5, no reranker) was a -4.26pp regression on c124
(v22-says v1 81.91% vs v22 baseline 86.17%). That is the EXACT same
drop as v22-fp v1, which also kept relative phrases. The shared
ingredient -- preserving relative phrases -- looks like the culprit:
the embedding only sees the segment text, NOT the framework prefix.
Without an absolute date in the segment, temporal queries lose their
cosine anchor and the segment falls out of the top-K.

v2 holds the says framing constant and reverts the date-resolution
clause to v22 baseline's wording (resolve every relative phrase to an
``on YYYY-MM-DD`` absolute inline). If v2 matches or beats v22
baseline, says framing is at worst neutral and at best a small win
once the date-anchor regression is removed. If v2 still trails v22
baseline, says framing itself is the regressor (the ``X says ...``
wrapper consumes 3-4 tokens per segment that carry no query signal
and may dilute the cosine match against shorter v22 statements).

Two-change recipe
-----------------
1. Date clause REVERTS to v22 baseline (absolute date inline, drop
   the message date when the resolved date equals ``{date}``).
2. Says framing KEPT from v22-says v1 (``{speaker} says ...``,
   first-person -> speaker name, second-person -> addressee name,
   ``says`` is the only speech-act verb used as a wrapper -- other
   speech-act framings ("X promised", "X apologized") still appear
   only when the speech-act itself is the event).

Neutral examples (Alice/Bob/Charlie/Dana + neutral domains -- book
club, climbing gym, mandolin, puppy training) are used so the prompt
does not telegraph the LoCoMo character set.
"""

from __future__ import annotations

from typing import override
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

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


PROMPT_REWRITE_V22_SAYS_V2 = """\
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
- Frames the speaker's content as ``{speaker} says ...`` (or \
``{speaker} says that ...``) in third-person. The speaker is the \
SUBJECT of the speech-act verb. First-person self-references in the \
raw message resolve to the speaker's name + appropriate third-person \
pronouns inside the ``says`` clause; second-person references \
(``you``, ``your``) resolve to the addressee's name when one is \
known; demonstrative and ambiguous pronouns resolve to their concrete \
referents. The first occurrence of any queryable entity is named; \
subsequent references within the same statement may use natural \
pronouns and possessives. Example (neutral): a message ``I finally \
finished The Buried Giant last night`` from Alice on 2026-04-09 \
becomes ``Alice says she finished reading The Buried Giant on \
2026-04-08.``
- Preserves every CONCRETE PARTICULAR from the message verbatim -- \
names, dates, numbers, identifiers, decisions, plans, preferences, \
opinions, relationships, emotional states tied to events, \
distinctive phrasing, attached-media descriptions. Generic \
abstractions or stock paraphrases for specifics are FAILURES.
- Preserves polarity, direction, and emotional tone. "Used to" \
implies no longer; "didn't get to bed until 2 AM" implies a late \
end, not a late start.
- Resolves every relative time reference into an absolute date or \
interval anchored at {date}: past-tense and backward markers go \
backward; future-tense and forward markers go forward; bare \
month/day references resolve to the nearest occurrence consistent \
with the surrounding tense; bare durations resolve to a span ending \
at {date}; bare shortcuts ("the weekend", "the holidays") resolve \
to the nearest past occurrence when no tense marker is present. \
The original relative phrase MUST NOT appear after resolution. \
Resolved dates appear inline in canonical ``on YYYY-MM-DD`` form \
(or ``from YYYY-MM-DD to YYYY-MM-DD`` for spans). Example (neutral): \
``Bob says he and Charlie hit the climbing gym yesterday`` from a \
message on 2026-04-09 becomes ``Bob says he and Charlie went to the \
climbing gym on 2026-04-08.``
- Is anchored to the date of the event it describes -- {date} for \
events that happened during the message, or the resolved date for \
events that happened on a different date. One date per statement. \
When the resolved event date equals {date}, omit the inline date \
phrase (the framework prefix already carries it); when the resolved \
event date differs from {date}, write the inline ``on YYYY-MM-DD`` \
phrase explicitly.
- Traces every detail back to words in the message. Inferred \
attributes (a title from context, a gender from a name, an age, an \
ethnicity, an unstated role) are FAILURES; explicitly stated \
attributes are preserved.
- Reports the content the message conveys via the ``{speaker} says \
...`` frame. OTHER speech-act framings -- "X promised that ...", \
"X apologized for ...", "X announced that ..." -- are included ONLY \
when that specific speech-act itself is the event (an apology, a \
promise, an explicit announcement). Routine mentions, recounts, and \
opinions take the plain ``says`` wrapper. Example (neutral): ``Dana \
says her mandolin lesson is moving to Thursdays.`` (routine); \
``Charlie apologized to Alice for missing the book club meeting.`` \
(apology IS the event).

NEIGHBORING TURNS appear before and after the message strictly to \
help resolve second-person addressees, demonstratives, anaphora, \
and unresolved relative references. Content drawn from the \
neighbors is NEVER emitted -- only content drawn from the message \
itself.

Output: a JSON object {{ "memories": [...] }}. The list is empty \
when the message contains no specific content.

{neighbors_block}MESSAGE FROM {speaker} on {date}:
{passage}"""


class _RewriteResponse(BaseModel):
    """Structured response from the rewriting language model."""

    memories: list[str]


def _format_neighbors(before: list, after: list) -> str:
    """Render the surrounding-events block fed to the LLM.

    Empty string when both lists are empty -- the prompt template
    tolerates an empty block. The block is labeled "PRIOR TURNS" /
    "LATER TURNS" so the LLM treats them as resolution context, not as
    content to emit.
    """
    lines: list[str] = []
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
    """v22-says v2: says framing + v22 baseline absolute-date resolution.

    Identical to v22-says v1 except the date-handling clause is
    reverted to v22 baseline's absolute-date resolution, restoring the
    cosine anchor on temporal queries while keeping the says frame.
    """

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_REWRITE_V22_SAYS_V2,
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
        self,
        chunk: str,
        speaker: str,
        date: str,
        neighbors_block: str,
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
        """Embed-channel text: rewrite + raw speaker-prefixed chunk.

        The dual-text embed lets retrieval match either the
        third-person narrative (good for query phrasings that mirror
        the rewrite) or the original conversational surface (good for
        query phrasings that mirror the raw turn).
        """
        return f"{rewrite}\n{speaker}: {original_chunk}"

    @override
    async def segment(self, event: Event) -> list[Segment]:
        match event.context:
            case SurroundingEventsContext(
                producer=producer, before=before, after=after
            ):
                speaker = producer
                neighbors_block = _format_neighbors(before, after)
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
                                    context=RewriteContext(
                                        text_to_embed=embed_text
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
