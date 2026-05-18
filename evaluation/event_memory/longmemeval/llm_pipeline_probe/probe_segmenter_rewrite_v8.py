"""LLM rewriting segmenter v8 — aggregation-first density.

v7 produced fragmented same-subject statements (e.g. 5 separate
sentences about "Caroline's friends" on one date). Mem0 V3 hits 91.3%
at top-10 because each memory packs more facts. v8 strengthens the
aggregation rule with explicit guidance: same-subject facts in one
message become ONE rich statement, not many fragments.

Architecture: identical to v7 (RewriteContext dual-surface embed +
deterministic 1500-char split + LLM rewrite). Only rule 7 changed:
old "ONE TOPIC PER STATEMENT" → new "MERGE SAME-SUBJECT FACTS" with
concrete examples and an anti-pattern.
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

PROMPT_REWRITE_V8 = """\
Rewrite a single conversational message into a JSON list of \
self-contained third-person memory statements. The memory system \
stores these statements and later retrieves them by semantic search; \
each statement must be findable and meaningful on its own.

CONTEXT:
- The message was spoken by {speaker} on {date}.
- You only see this one message. Do not reference what came before \
or after.

RULES:
1. THIRD PERSON. Use the speaker's name in place of first-person \
references ("I" -> "{speaker}"). Use a descriptive name in place of \
second-person references when the addressee is not known to you \
("you" -> "the other party"). Replace ambiguous pronouns with their \
concrete referents.

2. SELF-CONTAINED. A statement must be understandable in isolation, \
without the surrounding conversation. State who is involved, what \
happened or what is being claimed, and any time, location, or \
quantity that grounds the statement.

3. PRESERVE EVERY SPECIFIC. Names, dates, numbers, quantities, titles, \
brands, places, named objects, named activities, quoted phrases, and \
proper nouns must appear in the statement exactly as written in the \
message. Replacing a specific with a vague category is a failure: \
"Ferrari 488 GTB" stays "Ferrari 488 GTB" (not "a car"); \
"promoted to assistant manager" stays "assistant manager" (not \
"manager"); "416 pages" stays "416 pages" (not "about 400 pages"); \
"watched 'Eternal Sunshine of the Spotless Mind'" keeps the full \
title.

4. PRESERVE MEANING. Mirror the exact direction, polarity, and \
emotional content of the original. "Used to enjoy X" means no longer \
enjoys X. "Didn't get to bed until 2 AM" means LATE BEDTIME, not late \
wakeup. "Can't stop X-ing" means doing X frequently. Emotional states \
("scared but reassured", "happy and thankful"), motivations \
("inspired by her own journey"), and subjective descriptions \
("therapeutic", "nerve-wracking") are part of the meaning and stay.

5. TIME RESOLUTION. Convert every relative time reference into an \
absolute date or interval, using {date} as the anchor.
   - Past-tense markers ("yesterday", "last week", "ago", verbs in \
past tense, "used to") resolve BACKWARD from {date}.
   - Future-tense markers ("tomorrow", "next week", "upcoming", \
"will", "expecting", "this coming", verbs in future tense) resolve \
FORWARD from {date}.
   - Ambiguous bare month/day references ("in March", "on the 5th") \
resolve to the nearest occurrence consistent with the surrounding \
tense -- backward for past tense, forward for future tense.
   - Absolute dates and explicit durations stay verbatim -- "18 days" \
stays "18 days".
   - After resolution, the original relative phrase MUST NOT appear \
in the statement. Replace "yesterday" with the resolved date; do not \
keep "yesterday" alongside the date.

6. DATE-ANCHOR EACH STATEMENT. Every statement should be anchored to \
a specific date drawn from the message context. When the statement \
describes an event, action, observation, or claim made on the \
observation date {date}, include "{date}" in the statement (e.g., \
"On {date}, {speaker} said ..." or "{speaker} attended X on \
{date}"). When the statement is about an event that happened on a \
different date (already in the message or resolved per rule 5), use \
that date instead. The date may appear once per statement and should \
sit naturally inside the prose, not as a redundant prefix.

7. MERGE SAME-SUBJECT FACTS. When the message contains multiple facts \
about the SAME central subject -- the same person, relationship, \
event, decision, project, place, hobby, or feeling -- combine them \
into ONE rich statement, not many fragments. The central subject's \
attributes, supporting details, motivations, emotional context, and \
temporal grounding all belong together. Examples of correct merging:
   - 5 sentences reflecting on "my friends" -> ONE statement listing \
their names/roles, the 4-year duration, the breakup support, and the \
gratitude in one prose block.
   - 3 sentences about "the camping trip" -> ONE statement covering \
where, when, who, and what happened.
   - A description + quoted reaction + emotional context about the \
SAME event -> ONE statement.
   Anti-pattern: emitting "{speaker} said her friends are important. \
{speaker} said she has known them 4 years. {speaker} said they helped \
during a breakup." as three statements scatters the information and \
hurts retrieval. Emit ONE statement combining all three details.
   Only split when the message covers GENUINELY UNRELATED subjects \
(switching from "my job" to "my dog", or from "yesterday's party" to \
"next month's trip"). Within a single subject, prefer ONE rich \
statement. A typical message yields 1-3 statements, not 5-10.

8. NO FABRICATION. Every detail in a statement must trace back to \
words in the message. Do not infer attributes (gender, age, \
ethnicity, role) from names or context. Do not import information \
from outside the message.

9. NO META-COMMENTARY. Extract the CONTENT of what was said, not the \
fact that it was said. Bare questions ("{speaker} asked Mel who \
supports her", "{speaker} asked the other party about X") produce no \
statement unless the question itself carries an incidental fact. If \
the message contains incidental personal facts inside a question or \
request, extract those facts.

10. DROP PURE FILLER. Greetings ("Hi"), sign-offs ("Bye", "Take \
care"), bare reassurances ("I'm here for you"), and bare \
acknowledgments ("Sounds good", "Thanks", "Got it") that carry no \
specifics produce no statement. A message whose every part is \
interchangeable filler -- no name, date, number, decision, plan, \
preference, opinion, event, hobby, relationship detail, or attached \
media description -- emits an empty list.

11. NO DUPLICATES. Do not emit two statements that carry the same \
information in different words. Each fact appears exactly once.

Output: a JSON object {{ "memories": [...] }} where memories is a \
list of strings. Empty list if the message has no extractable \
specifics.

MESSAGE FROM {speaker} on {date}:
{passage}"""


class _RewriteResponse(BaseModel):
    memories: list[str]


class RewriteSegmenter(Segmenter):
    """v8 — aggregation-first dense rewrites with RewriteContext dual embed."""

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_REWRITE_V8,
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
