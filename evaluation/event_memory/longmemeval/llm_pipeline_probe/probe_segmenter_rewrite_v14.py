"""LLM rewriting segmenter v14 — v13 + deriver-writer feedback fixes.

Two feedback sources merged:

v13 base (prompt-engineering feedback):
  - rule 10 flipped to existential keep-test (any specifics → keep)
  - rule 9 question-with-referent carve-out clarified
  - rule 3 includes attached media in preserve list
  - rule 7 same-event-not-same-subject made explicit
  - rule 5 handles bare durations and conversational shortcuts
  - rule 8 fabrication-vs-preserve boundary clarified

v14 additions (deriver-writer feedback):
  - rule 7 completeness clause adjacent to count trigger (lesson from
    deriver v64→v65: "one X" triggers content drop unless the rule
    explicitly says the one must cover every specific).
  - rule 3 escape hatch ("or any other concrete particular a future
    query might reference verbatim") — open list, not closed.
  - rule 2 anti-pronoun rule: multi-statement outputs must name
    entities explicitly, never use pronouns for the queryable
    subject. Generalizes deriver v60 binding lesson.
  - rule 9 imperative reword: drop the double-negative.
  - terminology unification: rule 7 uses "EVENT" consistently (not
    topic / subject / narrative interchangeably).

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

PROMPT_REWRITE_V14 = """\
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

2. SELF-CONTAINED, NO PRONOUN STAND-INS. A statement must be \
understandable in isolation, without the surrounding conversation. \
State who is involved, what happened or what is being claimed, and \
any time, location, or quantity that grounds the statement. When \
the output has multiple statements, every statement must name \
people, places, and things explicitly -- never as pronouns ("it", \
"they", "we", "she", "he", "this", "that") for the queryable \
subject. A statement using only pronouns won't be found by a query \
for the named entity.

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
Sunshine of the Spotless Mind'" keeps the full title; an attached \
photo of "a band performing on stage with a sign that says all are \
welcome" keeps that description verbatim.

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
   - Bare durations without an explicit anchor ("for 5 years now", \
"3 months in") resolve to a span ending at {date} unless context \
identifies a different anchor.
   - Conversational shortcuts ("the weekend", "the holidays") \
resolve to the nearest occurrence on the past side when no tense \
marker is present; speakers more often refer back than forward.
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
that date instead. The date may appear once per statement and \
should sit naturally inside the prose, not as a redundant prefix.

7. ONE EVENT PER STATEMENT, MERGE ELABORATIONS, COVER EVERY \
SPECIFIC. An EVENT is a single occurrence, decision, action, or \
narrative beat with one core happening (a trip, a promotion, a \
purchase, a conversation thread on one topic, a planned outing). \
Details that elaborate the SAME event -- subject, action, time, \
place, attendees, motivation, outcome, sub-details, attached media \
-- all go in ONE rich statement covering every concrete specific \
from those details (every name, date, number, quoted phrase, proper \
noun, attached media description). ONE statement is one searchable \
prose string, not a short summary that drops particulars. \
"{speaker} got promoted to assistant manager at Shopify last week \
after two years" stays as ONE statement preserving "assistant \
manager", "Shopify", "last week" (resolved), and "two years". A \
camping trip's location, attendees, and activities stay as ONE \
statement preserving each name, place, and activity. But SEPARATE \
EVENTS that share a subject must be SEPARATE statements: "Calvin \
met Frank Ocean at a festival in August 2022 and they're recording \
together in August 2023" becomes TWO statements (2022 meeting + \
2023 recording). "{speaker} went to Bali last year and is planning \
to return next month" becomes TWO statements (past trip + future \
plan). Same subject does NOT mean same event. Splitting one event \
into multiple statements is a failure; merging two distinct events \
into one statement is also a failure.

8. NO FABRICATION. Every detail in a statement must trace back to \
words in the message. INFER means adding information that is not \
stated (assuming a "Dr." title from context, guessing gender from a \
name, inferring age, ethnicity, or unstated role) -- never do this. \
PRESERVE (rule 3) means keeping information that IS explicitly \
stated ("{speaker} got promoted to assistant manager" preserves \
"assistant manager"; "we hiked Mount Batur" preserves "Mount \
Batur"). Stated roles, titles, and attributes are preserved; \
unstated ones are never inferred.

9. EXTRACT FACTS, NOT SPEECH-ACTS. Extract the CONTENT of what was \
said, not the framing that it was said. A bare question with no \
embedded fact ("How was your day?", "What's new?") emits no \
statement. A question whose phrasing references a SPECIFIC ENTITY \
owned, done, or possessed by the addressee (their library, their \
guitar, their daughter Sarah, their Bali trip) extracts the \
referent as a fact: "What kind of books are in your library?" \
becomes "On {date}, {speaker} knows the other party has a library \
and asked what kind of books are in it." A question with an \
incidental personal fact ("Have you tried the new bakery on 4th \
Street?") extracts the fact.

10. KEEP IF ANY PART HAS SPECIFICS. Scan the entire message. If any \
part contains a specific (a name, place, date, number, decision, \
plan, preference, opinion with backing, named activity, attached \
media description, or a question referencing a specific entity), \
emit at least one statement capturing those specifics. A leading \
reaction or filler ("Wow!", "That's amazing!", "Oh my god") does \
NOT negate specifics that appear later in the same message. Emit \
an empty list ONLY when the entire message has no specifics \
anywhere -- a bare greeting ("Hi"), a bare sign-off ("Bye"), or a \
pure reaction with no content ("Sounds good", "Thanks", "Got it"). \
When in doubt, keep.

11. NO DUPLICATES. Do not emit two statements that carry the same \
information in different words. Each fact appears exactly once.

Output: a JSON object {{ "memories": [...] }} where memories is a \
list of strings. Empty list only when no specifics exist anywhere \
in the message.

MESSAGE FROM {speaker} on {date}:
{passage}"""


class _RewriteResponse(BaseModel):
    memories: list[str]


class RewriteSegmenter(Segmenter):
    """v14 — v13 + completeness clause in rule 7 + anti-pronoun in rule 2."""

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_REWRITE_V14,
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
