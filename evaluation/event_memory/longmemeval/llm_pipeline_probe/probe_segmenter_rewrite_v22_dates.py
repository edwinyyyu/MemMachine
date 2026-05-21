"""v22-dates RewriteSegmenter — v22 + explicit message-date vs
event-mentioned-date discipline.

Motivation: v22's rewrite leaks the message ``{date}`` into segment
text in 4+ inconsistent forms across the LoCoMo fb corpus (7127
segments audited):

  - ``on YYYY-MM-DD`` suffix                          26.4%
  - ``On YYYY-MM-DD,`` sentence prefix                20.5%
  - ``on YYYY-MM-DD`` mid-sentence                     2.5%
  - ``(Date: YYYY-MM-DD)`` / ``(Event date: ...)``   <1%
  - ``next week as of YYYY-MM-DD`` failed resolution

Worse, segments confuse MESSAGE DATE with EVENT-MENTIONED DATE. A
cat2 regression demonstrates the failure mode: query "When did Nate
get his first two turtles?" gold = 2019. The message was sent on
2022-01-23 and said "I've had my turtles for three years". v22
produced segments anchored on 2022-01-23 (message date) instead of
resolving "3 years ago" to ~2019 (event date). Retrieval then
surfaced 2022-01-23-anchored segments, and the answerer reported the
wrong date.

Data-model: Event = single message. Segments inherit the message
timestamp. The framework's ``EventMemory.string_from_segment_context``
prepends each segment with ``[<formatted long-date>] `` AT QA TIME
ONLY -- this is presentation; the embedding does not see it. The
embedding text remains ``{rewrite}\\n{speaker}: {raw_chunk}``.

Design: v23 makes the message date a SYSTEM-OWNED prefix that the
prompt must NOT echo. Any date appearing in the rewrite refers to an
event MENTIONED in the message -- never the message itself. v23
enforces ONE canonical inline form (``on YYYY-MM-DD`` in natural
prose) and bans all other variants (sentence-prefix, parenthetical,
bracket, ``as of``). v23 also resolves relative phrases through the
message-date anchor, then DROPS the date when the resolved date
equals the message date (the framework prefix already carries it).

All other v22 rules (KEEP/DROP, one-event-per-statement, preserve
particulars verbatim, polarity, speech-act-only-when-event,
neighbors-as-context-only) are preserved verbatim.

Risk: when a message contains BOTH a coincides-with-now event and a
mentioned-other-date event, the model may either over-zealously drop
both dates (losing the mentioned-event anchor) or echo the message
date in both (defeating the canonical form). The examples below
cover this case explicitly; empirical validation pending.
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

PROMPT_REWRITE_V22_DATES = """\
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

MESSAGE DATE vs EVENT-MENTIONED DATE:
- The MESSAGE was sent on {date}. The retrieval system automatically \
prepends the message timestamp to each statement when surfacing it \
for question answering. DO NOT include the message date {date} in \
your statement text in any form -- not as a suffix, not as a \
sentence prefix, not parenthesized, not bracketed.
- Any date that appears in your statement text refers to an EVENT \
MENTIONED in the message, NOT the message itself. The event being \
described may coincide with the message-sending moment, or it may \
refer to a different past or future moment that the speaker is \
recounting or anticipating.

RELATIVE-TIME RESOLUTION:
- Resolve every relative time reference ("yesterday", "last week", \
"this summer", "three years ago", "next month", "the weekend") into \
an ABSOLUTE date anchored at the message date {date}. Past-tense and \
backward markers go backward; future-tense and forward markers go \
forward; bare month/day references resolve to the nearest occurrence \
consistent with the surrounding tense; bare durations ending "now" \
resolve to a span ending at the message date; bare shortcuts ("the \
weekend", "the holidays") resolve to the nearest past occurrence \
when no tense marker is present.
- If the resolved date EQUALS the message date (the mentioned event \
coincides with the message-sending moment -- e.g., "I'm watching \
the game right now"), DROP the date from the statement. The \
framework's automatic prefix carries it.
- If the resolved date DIFFERS from the message date, include it \
inline as ``on YYYY-MM-DD`` in natural prose -- for example, \
"Alice's tournament was on 2026-03-14".
- A bare relative phrase ("last week", "next month", "yesterday", \
"three years ago", "the weekend") appearing in the output AFTER \
resolution is a FAILURE.

CANONICAL DATE FORMAT:
- Use exactly ONE inline date form: ``on YYYY-MM-DD`` woven into \
natural prose.
- Forbidden forms (each is a FAILURE): a sentence-prefix ``On \
YYYY-MM-DD, ...``; a parenthetical ``(Date: YYYY-MM-DD)`` or ``(Event \
date: YYYY-MM-DD)``; a square-bracket ``[YYYY-MM-DD]``; an ``as of \
YYYY-MM-DD`` qualifier; any suffix attaching the message date to a \
statement.
- One event date per statement. A statement that describes events \
occurring on multiple different dates must be split into separate \
statements, each carrying its own event date.

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
- Traces every detail back to words in the message. Inferred \
attributes (a title from context, a gender from a name, an age, an \
ethnicity, an unstated role) are FAILURES; explicitly stated \
attributes are preserved.
- Reports the content the message conveys, not the speech-act of \
conveying it. "X said that ...", "X told Y that ...", "X mentioned \
that ..." framing is included ONLY when the speech-act itself is \
the event (an apology, a promise, an explicit announcement). For \
mere conversational reporting, drop the speech-act wrapper.

NEIGHBORING TURNS appear before and after the message strictly to \
help resolve second-person addressees, demonstratives, anaphora, \
and unresolved relative references. Content drawn from the \
neighbors is NEVER emitted -- only content drawn from the message \
itself.

EXAMPLES (each example shows how the message date should and should \
not appear in the output):

Example 1 -- message-time event; the framework's prefix carries the \
date, so DROP the date from the rewrite.
MESSAGE FROM {speaker} on {date}:
I'm finally trying that ramen place on Castro Street right now -- \
the tonkotsu is incredible.
->
{{ "memories": ["{speaker} is eating tonkotsu ramen at the ramen \
place on Castro Street and finds it incredible."] }}

Example 2 -- mentioned event with a relative reference ("three \
years ago") that resolves to a DIFFERENT date than the message \
date. Include the resolved date inline.
MESSAGE FROM Alice on 2026-04-10:
I adopted my two cockatiels three years ago, right before I moved \
to Portland.
->
{{ "memories": ["Alice adopted her two cockatiels on 2023-04-10, \
right before she moved to Portland."] }}

Example 3 -- mentioned event with an EXPLICIT absolute date that \
differs from the message date. Keep the explicit date in canonical \
``on YYYY-MM-DD`` form.
MESSAGE FROM Bob on 2026-05-02:
Charlie's wedding on June 14, 2025 was the best party I went to \
last year.
->
{{ "memories": ["Bob considered Charlie's wedding on 2025-06-14 the \
best party he attended in 2025."] }}

Example 4 -- speech-act-as-event: a promise made on the message \
date. The speech-act IS the event, so keep the wrapper; the event \
coincides with the message moment, so drop the date.
MESSAGE FROM Dana on 2026-05-18:
I promise I'll stop missing our Thursday mandolin practice -- I'll \
be there every week from now on.
->
{{ "memories": ["Dana promised to stop missing the Thursday \
mandolin practice and to attend every week going forward."] }}

Example 5 -- conversational reporting that should DROP the \
speech-act wrapper. The event is the trip itself, not the act of \
mentioning it.
MESSAGE FROM Alice on 2026-05-18:
I mentioned to Bob that I went hiking in Big Sur last weekend with \
my book club.
->
{{ "memories": ["Alice went hiking in Big Sur with her book club on \
2026-05-16."] }}

Example 6 -- message with NO mentioned event (a bare question with \
no specific content). Emit an empty list.
MESSAGE FROM {speaker} on {date}:
How are you doing today?
->
{{ "memories": [] }}

Example 7 -- multi-event message that splits into multiple \
statements, each carrying its own event date.
MESSAGE FROM Charlie on 2026-05-18:
I climbed Half Dome two months ago and I'm flying to Tokyo next \
Friday for a conference on AI alignment.
->
{{ "memories": ["Charlie climbed Half Dome on 2026-03-18.", \
"Charlie is flying to Tokyo on 2026-05-22 for a conference on AI \
alignment."] }}

Example 8 -- message that mentions a FUTURE event with a relative \
reference resolved forward from the message date.
MESSAGE FROM Dana on 2026-05-18:
My puppy Mochi is getting his second round of vaccinations next \
Wednesday at the Mission vet clinic.
->
{{ "memories": ["Dana's puppy Mochi is getting his second round of \
vaccinations on 2026-05-20 at the Mission vet clinic."] }}

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
    """Deterministic split + LLM rewrite into third-person statements,
    indexed against a dual-text embed channel (rewrite + raw chunk).

    Args:
        language_model: The LanguageModel used to rewrite each chunk.
            Configure the model and reasoning effort at construction
            of the LanguageModel itself.
        prompt_template: A ``.format(speaker=..., date=..., passage=...,
            neighbors_block=...)`` template producing the full prompt.
            Defaults to the v22-dates prompt with explicit message-date
            vs event-mentioned-date discipline.
        chunk_size: Maximum characters per deterministic chunk before
            calling the LLM. Defaults to 1500, which keeps typical
            single-turn messages as one chunk while still chunking
            outlier-long inputs deterministically rather than relying
            on the LLM to split.
        max_attempts: Retries on retryable language-model errors.

    Caller guidance: populate ``event.context`` with a
    ``SurroundingEventsContext`` whose ``before`` field contains the
    most recent prior turns of the conversation (recommended depth: 8
    turns). Leave ``after`` empty for streaming-friendly ingestion --
    after-neighbors never help on the LoCoMo benchmark and force the
    pipeline to buffer for future context.
    """

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_REWRITE_V22_DATES,
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
