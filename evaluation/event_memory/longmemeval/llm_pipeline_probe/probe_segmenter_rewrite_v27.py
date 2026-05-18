"""LLM rewriting segmenter v27 — v22 + Q-style framing for retrieval alignment.

Hypothesis: retrieval works by embedding similarity (and BM25 keyword
overlap) between user questions and stored segments. v22 segments are
declarative ("Joanna watched Eternal Sunshine on 2022-04-10 with Nate").
User queries are interrogative ("When did Joanna watch X?", "What
movies has Joanna seen?"). The declarative-vs-interrogative phrasing
mismatch reduces cosine similarity AND BM25 overlap on the question's
wh-words.

v27 keeps every v22 constraint about WHAT to extract but changes HOW
each statement is phrased. Every memory statement now begins with a
short topic-and-question phrase ("Question this fact answers" form),
then the answer, then the remaining particulars. The topic phrase uses
the same wh-shape the future query is likely to take (when / where /
who / what / why / how).

Generalizable principle: retrieval improves when segment phrasing
mirrors the question shape it would answer. Each declarative event
becomes a (topic question) -> (answer with all particulars) pair.

Example transformation:
- v22: "Joanna watched Eternal Sunshine of the Spotless Mind on
  2022-04-10 with Nate."
- v27: "When Joanna watched Eternal Sunshine of the Spotless Mind:
  2022-04-10, with Nate."
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

PROMPT_REWRITE_V27 = """\
Rewrite the MESSAGE into a JSON list of standalone third-person \
memory statements. Each statement is stored verbatim and later \
retrieved by semantic search using a future user's natural-language \
question. A future user querying any specific content in the message \
should find at least one statement that contains that content.

PHRASE EACH STATEMENT AS A QUESTION-ANSWER PAIR. Lead with a short \
topic phrase that mirrors the shape of the question this fact would \
answer; then give the answer along with every concrete particular. \
The topic phrase uses the wh-shape most natural to the fact:
- WHEN ... : for dated occurrences, durations, sequences in time
- WHERE ... : for locations and venues
- WHO ... : for people, attendees, relationships
- WHAT ... : for objects, titles, decisions, plans, preferences, \
opinions, attached media, named activities
- WHY ... : for motivations, reasons, causes
- HOW ... : for methods, manners, processes
Use the wh-word that matches the fact's most queryable aspect; if \
multiple aspects are queryable, write a separate statement for each. \
Examples of the topic-then-answer phrasing:
- "When Alice ran the Berlin Marathon: 2023-09-24, finishing in 3h41m."
- "What Bob's favorite restaurant is: Mariella's in North Beach, for \
their cacio e pepe."
- "Who attended Charlie's dorm potluck on 2024-11-09: Charlie, Dana, \
and Eli, with Dana bringing dumplings."
- "Why Dana switched majors in 2023: she found chemistry labs \
exhausting and preferred her statistics elective."
Statements never use the literal characters "Q:", "A:", or a \
question mark; the topic phrase is a declarative noun phrase that \
NAMES the question. The answer follows after a colon and contains \
the full particulars.

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
- Begins with a wh-topic phrase ("When X did Y", "What X's favorite Z \
is", "Who attended ...", "Why X decided ...", "Where X went ...", \
"How X learned ...") followed by a colon and the answer. The topic \
phrase NAMES the question; the answer carries every concrete \
particular the message gives -- subject, action, time, place, \
attendees, motivation, outcome, attached media.
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
    """v27 — v22 + Q-style topic-then-answer phrasing."""

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_REWRITE_V27,
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
