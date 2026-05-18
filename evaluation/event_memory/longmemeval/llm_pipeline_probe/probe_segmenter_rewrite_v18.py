"""LLM rewriting segmenter v18 — principle-only, no examples, no count bias.

Strips out inline examples, segment-count biases, and overloaded
rules from v17. Each rule names one objective property the output
must have, defined in terms portable across models. The principles
are evaluated at the statement level, not as procedures the LLM
follows.

Key contrasts to v17:
- Rule 7 split: event scoping (no cross-event content) and detail
  elaboration (same-event details stay together) are distinct
  principles, each in its own slot.
- Rule 9 reworded around ASSERTIONS not SPEECH-ACTS — objective
  test ("does the message contain a truth-bearing assertion") in
  place of subjective ("is this a bare question").
- All inline examples removed.
- All "by default", "prefer", "most messages produce" biases
  removed.
- Completeness lives in rule 3 (preserve every specific), not as
  a clause attached to a count trigger.

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

PROMPT_REWRITE_V18 = """\
Rewrite a single conversational message into a JSON list of \
self-contained third-person memory statements. The memory system \
stores these statements and retrieves them by semantic search. \
Each statement is judged on the principles below.

CONTEXT:
- The message was spoken by {speaker} on {date}.
- You only see this one message. Do not reference what came before \
or after.

PRINCIPLES (each statement must satisfy ALL):

1. THIRD-PERSON GROUNDED. The speaker's name replaces all \
first-person references. Second-person references resolve to a \
descriptive name when the addressee is not known. Demonstrative \
and ambiguous pronouns resolve to their concrete referents the \
first time they appear in the statement.

2. ENTITY NAMED. Every queryable entity (a person, place, \
organization, named object, or named activity that future queries \
could ask about) appears by its name at least once in the \
statement. After it is named once, natural pronouns and \
possessives within the same statement are acceptable.

3. SPECIFICS VERBATIM. Names, dates, numbers, quantities, titles, \
brands, places, named objects, named activities, quoted phrases, \
proper nouns, attached-media descriptions, and any other concrete \
particular a future query could reference verbatim appear in the \
statement in the exact form the message uses. Vague substitutions \
for specifics are failures.

4. POLARITY AND EMOTION INTACT. Direction (positive vs negative), \
polarity ("used to" implies no longer; "didn't get to" implies \
not), temporal direction ("until 2 AM" implies late end, not late \
start), and emotional or evaluative content carried by the message \
are preserved in the statement.

5. RELATIVE TIME RESOLVED. Every relative time reference becomes \
an absolute date or interval anchored at {date}. Past-tense and \
backward markers resolve backward; future-tense and forward \
markers resolve forward; bare month/day references resolve to the \
nearest occurrence consistent with the surrounding tense; bare \
durations resolve to a span ending at {date} unless context \
specifies a different anchor; bare conversational shortcuts ("the \
weekend", "the holidays") resolve to the nearest past occurrence \
when no tense marker is present. The original relative phrase must \
not remain in the statement after resolution.

6. DATE-ANCHORED. Each statement explicitly carries the date of \
the event it describes. When the event is the message itself \
(something said, observed, or claimed on the observation date), \
the date is {date}. When the event happened on a different date \
(already in the message or resolved per principle 5), that date \
is used. The date appears once per statement.

7. SINGLE-EVENT SCOPED. A statement describes exactly one \
event -- one occurrence, action, decision, plan, observation, or \
state. Content drawn from two distinct events in the message \
(different times, different occasions, or different actions even \
if sharing a participant) belongs to two distinct statements.

8. SINGLE-EVENT COMPLETE. All concrete particulars in the message \
that describe the same single event -- subject, action, time, \
place, attendees, motivation, outcome, sub-details, attached \
media -- belong to the same statement. Splitting one event's \
particulars across multiple statements is a failure.

9. CONTENT NOT SOURCE. A statement's content is the truth-bearing \
assertion conveyed by the message, not the act of conveying it. \
A message segment that conveys no assertion (a question with no \
embedded fact; an acknowledgment; a reaction; a greeting) yields \
no statement. A message segment that conveys an assertion through \
a question or request ("Have you read X?" asserts the existence of \
X as a referent) yields a statement for that assertion. The \
speech-act framing itself is content only when the message marks \
the act as the event (an apology, a promise, a commitment).

10. SOURCE-ANCHORED. Every detail in the statement traces to \
words present in the message. Inferred attributes (a title from \
context, a gender from a name, an age, an ethnicity, an unstated \
role) are excluded. Explicitly stated attributes are preserved.

11. UNIQUE. Two statements never carry the same concrete content \
in different words.

OUTPUT-LEVEL PROPERTY:
The list contains one statement per distinct event in the message. \
A message with no truth-bearing assertions yields the empty list.

Output format: a JSON object {{ "memories": [...] }} where memories \
is a list of strings.

MESSAGE FROM {speaker} on {date}:
{passage}"""


class _RewriteResponse(BaseModel):
    memories: list[str]


class RewriteSegmenter(Segmenter):
    """v18 — principle-only, no examples, no count bias."""

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_REWRITE_V18,
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
