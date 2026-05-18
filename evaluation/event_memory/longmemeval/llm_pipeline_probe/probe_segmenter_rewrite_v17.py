"""LLM rewriting segmenter v17 — v14 + v16's direct framing + tighter anti-pronoun.

v14's strict anti-pronoun rule ("never use pronouns for the queryable
subject") forces "Nate told Joanna that Nate's support helps Joanna",
which has 2 Nate / 2 Joanna repetitions per sentence — bloats tokens
and dilutes the embedding signal (token repetition averages down the
semantic centroid).

v17 keeps the anti-pronoun guarantee for *multi-statement outputs*
(rule 2 — every queryable entity must be named somewhere in each
statement) but allows ONE pronoun chain within a statement after the
entity is named once. Combined with v16's direct framing (drop "X
told Y that ..." speech-act wrappers when the fact itself IS the
content), segments become denser.

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

PROMPT_REWRITE_V17 = """\
Rewrite a single conversational message into a JSON list of \
concise self-contained third-person memory statements. The memory \
system stores these statements and retrieves them by semantic \
search; each statement must be findable, dense, and meaningful on \
its own.

CONTEXT:
- The message was spoken by {speaker} on {date}.
- You only see this one message. Do not reference what came before \
or after.

RULES:
1. THIRD PERSON, DIRECT FRAMING. Use the speaker's name as the \
acting subject ("I went to Bali" -> "{speaker} went to Bali", NOT \
"{speaker} said she went to Bali"). Drop "X said that" / "X told Y \
that" / "X mentioned that" wrappers when the fact itself is the \
content; state the fact directly. Keep speech-act framing ONLY when \
the act of telling/asking IS the fact (e.g., "{speaker} apologized \
for missing the meeting"). Use a descriptive name for unknown \
addressees ("you" -> "the other party"). Replace ambiguous \
pronouns with their concrete referents the first time they appear.

2. NAME EACH ENTITY AT LEAST ONCE PER STATEMENT, then natural \
references are OK. Every queryable entity (person, place, \
organization, named object) must appear by name at least once in \
the statement. After the entity is named, natural pronouns and \
possessives ("he", "her", "their", "it") are fine within the same \
statement. Avoid repeating the same name 3+ times when one mention \
plus pronouns reads more naturally. Example: instead of "Nate told \
Joanna that Nate's support means a lot to Joanna and that Nate is \
here for Joanna" write "Nate told Joanna his support means a lot \
to her and he is here for her" -- each entity named once, then \
pronouns. A statement using only pronouns (no entity name \
anywhere) won't be found by a query for the named entity and is \
still a failure.

3. PRESERVE EVERY SPECIFIC. Names, dates, numbers, quantities, \
titles, brands, places, named objects, named activities, quoted \
phrases, proper nouns, descriptions of attached media (photos, \
videos, files including what they depict), or any other concrete \
particular a future query might reference verbatim must appear in \
the statement exactly as written in the message. Replacing a \
specific with a vague category is a failure: "Ferrari 488 GTB" \
stays "Ferrari 488 GTB" (not "a car"); "promoted to assistant \
manager" stays "assistant manager" (not "manager"); "416 pages" \
stays "416 pages" (not "about 400 pages"); "watched 'Eternal \
Sunshine of the Spotless Mind'" keeps the full title.

4. PRESERVE MEANING. Mirror the exact direction, polarity, and \
emotional content of the original. "Used to enjoy X" means no \
longer enjoys X. "Didn't get to bed until 2 AM" means LATE \
BEDTIME, not late wakeup. Emotional states ("scared but \
reassured", "happy and thankful"), motivations, and subjective \
descriptions are part of the meaning and stay.

5. TIME RESOLUTION. Convert every relative time reference into an \
absolute date or interval, using {date} as the anchor.
   - Past markers ("yesterday", "last week", "ago", past tense, \
"used to") resolve BACKWARD from {date}.
   - Future markers ("tomorrow", "next week", "upcoming", "will", \
"expecting", "this coming") resolve FORWARD from {date}.
   - Ambiguous bare month/day references resolve to the nearest \
occurrence consistent with surrounding tense.
   - Bare durations ("for 5 years now") resolve to a span ending \
at {date} unless context indicates otherwise.
   - Conversational shortcuts ("the weekend", "the holidays") \
resolve to the nearest past occurrence when no tense marker is \
present.
   - Absolute dates and explicit durations stay verbatim.
   - The original relative phrase MUST NOT appear after resolution.

6. DATE-ANCHOR EACH STATEMENT. Every statement should be anchored \
to a specific date. When the statement describes something that \
happened on the observation date {date}, include "{date}" naturally \
inside the prose. When the event happened on a different date \
(in the message or resolved per rule 5), use that date instead. \
One date per statement.

7. ONE EVENT PER STATEMENT, MERGE ELABORATIONS, COVER EVERY \
SPECIFIC. Pack every detail of a single event -- subject, action, \
time, place, attendees, motivation, outcome, sub-details, attached \
media -- into ONE rich statement covering every concrete specific \
(every name, date, number, quoted phrase, proper noun, attached \
media description). ONE statement is one searchable prose string, \
not a short summary that drops particulars. "{speaker} got promoted \
to assistant manager at Shopify on 2023-05-08 after two years" \
stays as ONE statement preserving role, employer, date, and \
duration. SEPARATE EVENTS that share a subject must be SEPARATE \
statements: "Calvin met Frank Ocean at a festival in August 2022 \
and they're recording together in August 2023" -> TWO statements \
(2022 meeting + 2023 recording). Same subject does NOT mean same \
event.

8. NO FABRICATION. Every detail must trace back to words in the \
message. INFER means adding information not stated (assuming a \
title from context, guessing gender, inferring age, role) -- \
never do this. PRESERVE means keeping information explicitly \
stated. Stated roles, titles, and attributes are preserved; \
unstated ones are never inferred.

9. SKIP PURE SPEECH-ACTS. Bare questions, congratulations, \
acknowledgments, and reactions that carry NO embedded fact about \
who/what/when/where produce NO statement. Examples that produce \
NO statement: "{speaker} asked the other party what kind of \
hobbies they have", "{speaker} congratulated the other party on \
winning", "{speaker} asked what game it was", "{speaker} \
expressed sympathy", "{speaker} said sounds fun". A speech-act \
IS worth keeping ONLY when it embeds a specific fact ("{speaker} \
has a library and asked what books are in it" -> extract \
"{speaker} has a library").

10. KEEP IF ANY PART HAS SPECIFICS. Scan the entire message. If \
any part contains a specific (a name, place, date, number, \
decision, plan, preference, opinion with backing, named activity, \
attached media description), emit at least one statement capturing \
those specifics. A leading reaction or filler ("Wow!", "That's \
amazing!") does NOT negate specifics that appear later. Emit an \
empty list ONLY when the entire message has no specifics anywhere \
-- bare greeting/sign-off/reaction.

11. NO DUPLICATES. Each fact appears in exactly one statement.

Output: a JSON object {{ "memories": [...] }} where memories is a \
list of concise prose strings. Most messages produce 1-3 \
statements. Empty list only when no specifics exist anywhere.

MESSAGE FROM {speaker} on {date}:
{passage}"""


class _RewriteResponse(BaseModel):
    memories: list[str]


class RewriteSegmenter(Segmenter):
    """v17 — v14 strengths + v16 direct framing + relaxed anti-pronoun."""

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_REWRITE_V17,
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
