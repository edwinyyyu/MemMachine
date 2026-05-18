"""LLM rewriting segmenter v26 — v22 + multi-paraphrase per event.

Hypothesis: BM25+embedding retrieval at small K is hit-or-miss for
paraphrastic variation. If each event gets 2-3 distinct phrasings
of the same content, retrieval has more shots at landing each
instance in top-K, and same-class events cluster better via shared
lexical/semantic overlap.

This is the v22 prompt with the EACH STATEMENT block reframed to
EACH EVENT GETS 2-3 PARAPHRASED STATEMENTS. Every paraphrase
preserves the same concrete particulars but varies lexical
phrasing (synonyms, restructured sentence, alternative subject
framing).
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

PROMPT_REWRITE_V26 = """\
Rewrite the MESSAGE into a JSON list of standalone third-person \
memory statements. Each statement is stored verbatim and later \
retrieved by semantic search and BM25. A future user querying any \
specific content in the message should find at least one statement \
that contains that content.

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

UNIT OF GRANULARITY:
- One EVENT in the message is the unit, not one sentence. An event \
is a single occurrence, decision, plan, observation, state, or \
preference at one point or span in time. A multi-sentence \
elaboration of the same event (subject in one sentence, reason in \
the next, outcome after that) is ONE event covering all of those \
sentences' particulars. Treating one sentence as one event is a \
FAILURE; treating one concrete particular as one event is a \
FAILURE; merging two distinct events into one is a FAILURE. \
Distinct events (different times, different occasions, different \
actions) each get their own treatment, even when they share \
participants or topics.

FOR EACH EVENT, EMIT 2-3 PARAPHRASED STATEMENTS:
- Every paraphrase of the same event preserves the SAME concrete \
particulars verbatim -- names, dates, numbers, identifiers, \
quoted phrases, proper nouns, attached-media descriptions. The \
paraphrases differ only in lexical phrasing: synonyms for \
non-specific verbs and connective words, reordered clauses, \
alternative subject framing (active vs. passive, "X did Y" vs. \
"Y was done by X" vs. "When X did Y, ..."), or different \
syntactic shape. Emitting fewer than 2 paraphrases for an event \
is a FAILURE; emitting more than 3 paraphrases for an event is a \
FAILURE; paraphrases that diverge on any concrete particular are \
a FAILURE; paraphrases that are near-identical (only swapping a \
single word) are a FAILURE.
- Example: an event where Alice biked 12 miles to the lake on \
Saturday could be expressed as (1) "Alice biked 12 miles to the \
lake on Saturday." (2) "On Saturday, Alice rode her bike for 12 \
miles to reach the lake." (3) "Alice's Saturday outing was a \
12-mile bike ride to the lake." All three preserve Alice / 12 \
miles / lake / Saturday but vary structure and synonyms.
- Group the paraphrases for one event together in the output list, \
in order. The order of events follows the order they appear in \
the message.

EVERY STATEMENT (each paraphrase):
- Contains every concrete particular the message gives about its \
event -- subject, action, time, place, attendees, motivation, \
outcome, attached media.
- Refers to the speaker by name. First-person self-references \
resolve to the speaker's name; second-person references resolve to \
the addressee's name when one is known; demonstrative and ambiguous \
pronouns resolve to their concrete referents. The first occurrence \
of any queryable entity is named; subsequent references within the \
same statement may use natural pronouns and possessives.
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
The original relative phrase MUST NOT appear after resolution.
- Is anchored to the date of the event it describes -- {date} for \
events that happened during the message, or the resolved date for \
events that happened on a different date. One date per statement.
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
and unresolved relative references. Content drawn from the \
neighbors is NEVER emitted -- only content drawn from the message \
itself.

Output: a JSON object {{ "memories": [...] }}. The list is empty \
when the message contains no specific content. Otherwise it \
contains 2-3 paraphrases per event, with all paraphrases for one \
event grouped together in order.

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
    """v26 — v22 + multi-paraphrase (2-3 phrasings per event)."""

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_REWRITE_V26,
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
