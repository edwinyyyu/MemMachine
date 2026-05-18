"""LLM rewriting segmenter v33 — v32 + answer framing inheritance.

v32 (= v22 + anti-speech-act rule) regressed -5.3pp on g3 c124
(82.98% vs v22's 88.30%, mem0-bench K=7 OLD). Empirical post-mortem
on g3: v32 produces +20% more segments at -20% mean length, with
asked-pattern dropping 8.6%->2.1% and said-pattern 50.1%->36.1%.
The LLM did follow the speech-act rule but compensated by over-
fragmenting answer messages — chopping each into smaller pieces
that no longer carry the question's specifier framing.

Mechanism: in a Q->A turn pair, the question carries the specificity
("favorite", "when", "which X"). v22 emits "X asked Y about
favorite Z" as a topical anchor; v32 drops it, and the answer's
segment "Y likes Z" no longer encodes "favorite". Retrieval on
"what's Y's favorite Z" then misses.

v33 fix: keep v32's speech-act drop rule for question-only messages,
and ADD a rule that answer-message processing must absorb the
specifier terms from the prior question (available as a before-
neighbor). The answer's rewrite encodes what kind of question it
answers.

Prerequisite: ingest with --neighbor-direction before
--neighbor-window N (N >= 1). Without before-neighbors the answer
processor can't see the question to inherit from.
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

PROMPT_REWRITE_V33 = """\
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

QUESTIONS ASKED INSIDE THE MESSAGE ARE NOT FACTS. When this MESSAGE \
asks the other speaker a question ("What's your favorite game?", \
"How was your weekend?", "Did you finish the script?"), that \
question is a speech-act -- it carries no stored fact and MUST be \
dropped. Emitting any statement of the form "X asked Y about Z", \
"X asked Y what/whether/if ...", "Y's question to X was ...", "X \
wanted to know about Y" is a FAILURE. Only the ANSWER to such a \
question becomes a stored fact, and only if the answer carries \
specific content. If this MESSAGE asks a question that itself \
contains a specific plan, opinion, or decision ("Should we still \
meet at the Blue Door at 7pm?"), keep ONLY the embedded specific \
content as a fact ("The planned meeting is at the Blue Door at \
7pm on {date}."), not the act of asking.

ANSWER FRAMING INHERITANCE. When this MESSAGE answers a question \
that appears in a PRIOR TURN, the segment(s) you emit MUST absorb \
the SPECIFIER terms from that prior question. The prior question \
is the source of what kind of fact this answer is; the answer \
must encode that framing or it will be unretrievable. \
- If the prior question asked about a FAVORITE, the answer \
  segment uses the word "favorite". Example: prior "What's your \
  favorite movie genre?" + this "Fantasy, I guess." -> emit \
  "{speaker}'s favorite movie genre is fantasy on {date}." -- not \
  "{speaker} mentioned fantasy" or "{speaker} likes fantasy". \
- If the prior question asked WHEN, the answer segment carries \
  the explicit date or time the answer provides. \
- If the prior question asked WHICH X or WHAT X, the answer \
  segment names X explicitly. Example: prior "Which book did you \
  read last?" + this "The one about dragons." -> emit \
  "{speaker}'s last-read book is the one about dragons on \
  {date}.", carrying "last-read book" from the question. \
- If the prior question asked HOW MANY, the answer segment \
  carries the count as a number. \
- If the prior question asked WHERE, the answer segment names \
  the place. \
Skipping inheritance is a FAILURE: an answer segment that does \
not encode its question's frame is unretrievable when a future \
user queries with that frame.

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
- Traces every detail back to words in the message OR to specifier \
terms inherited from a prior-turn question per the ANSWER FRAMING \
INHERITANCE rule. Inferred attributes (a title from context, a \
gender from a name, an age, an ethnicity, an unstated role) are \
FAILURES; explicitly stated attributes are preserved.
- Reports the content the message conveys, not the speech-act of \
conveying it. "X said that ...", "X told Y that ...", "X mentioned \
that ..." framing is included ONLY when the speech-act itself is \
the event (an apology, a promise, an explicit announcement).

NEIGHBORING TURNS appear before the message strictly to help \
resolve second-person addressees, demonstratives, anaphora, and \
unresolved relative references, AND to supply specifier framing \
when the prior turn is a question this MESSAGE answers. Content \
drawn from the neighbors is NEVER emitted as its own fact -- only \
content drawn from the message itself, possibly reframed with \
inherited specifier terms.

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
    """v33 — v32 + answer framing inheritance from prior-turn question."""

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_REWRITE_V33,
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
