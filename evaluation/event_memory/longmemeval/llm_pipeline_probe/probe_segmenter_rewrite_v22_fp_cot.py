"""v22-fp-cot RewriteSegmenter -- first-person rewrite with EXPLICIT
chain-of-thought structured output for relative-date resolution.

Motivation
----------

v22-fp empirically fails on two correlated failure modes when run at
``gpt-5.4-nano`` low/medium:

  * msg_date_leak (~5-6%): the message date appears inline in the
    output even though the framework's segment-header carries it.
  * bare_relative (~14-20%): relative phrases like ``last week``,
    ``yesterday``, ``next month`` survive into the output instead of
    being resolved to an absolute date and stripped.

Bumping ``reasoning_effort`` did not move the needle. The diagnosis:
the model is trying to do too many things at once -- detect relative
phrases, anchor them to the message date, decide whether the resolved
date equals the message date, decide whether to keep or drop it, AND
emit natural prose -- all in a single generation pass. Substeps get
silently dropped.

Hypothesis
----------

If we ask the model to first MATERIALIZE its detected relative phrases
and their resolutions as structured intermediate output, and only then
emit the final 1p statements, the per-substep accountability should
reduce both leak and bare-relative failures. The CoT trace is captured
in the API call's structured output (``relative_resolutions``); only
``memories`` is consumed downstream.

Architecture
------------

Same as v22-fp: each emitted segment carries
``ProducerContext(producer=speaker)``; the segment's stored block text
stays bare first-person prose.

Caller guidance
---------------

Populate ``event.context`` with ``SurroundingEventsContext`` (depth ~8
turns in ``before``, ``after`` empty) so the rewrite can resolve
second-person addressees and demonstratives.
"""

from __future__ import annotations

from typing import override
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from memmachine_server.common.language_model import LanguageModel
from memmachine_server.episodic_memory.event_memory.data_types import (
    Event,
    ProducerContext,
    Segment,
    SurroundingEventsContext,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.segmenter.segmenter import (
    Segmenter,
)
from pydantic import BaseModel


PROMPT_REWRITE_V22_FP_COT = """\
Rewrite the MESSAGE into a JSON object with two fields: \
``relative_resolutions`` (a chain-of-thought trace of every relative \
time phrase you detect and the absolute date it resolves to) and \
``memories`` (the final list of standalone first-person memory \
statements from {speaker}'s point of view). The ``memories`` strings \
are stored verbatim and later retrieved by semantic search. A future \
user querying any specific content in the message should find at \
least one statement that contains that content.

KEEP what is specific to this message; DROP what is interchangeable \
across similar messages. Specific content includes: names of people, \
places, organizations, brands, named objects, named activities; dates, \
times, durations, quantities; identifiers, titles, quoted phrases, \
proper nouns; decisions, plans, preferences, opinions, relationships, \
roles, emotional states tied to events; described events (something \
that happened or will happen); attached-media descriptions. \
Interchangeable content has none of these -- bare greetings, sign-offs, \
acknowledgments, reactions, and questions whose phrasing introduces \
nothing from the specific list above. Dropping specific content is a \
FAILURE; emitting a statement for interchangeable content is a FAILURE.

PERSON RULES (first-person from {speaker}'s perspective):
- Preserve {speaker}'s own ``I`` / ``me`` / ``my`` / ``mine`` / \
``myself`` from the raw message verbatim. Do NOT replace them with the \
speaker's name.
- Names other than {speaker} stay in third-person: addressees, \
third-party participants, places, brands, organizations, named \
objects. Resolve a second-person reference (``you``, ``your``) to the \
addressee's NAME when the addressee is known from the conversation \
context; otherwise leave the second-person form intact.
- Resolve demonstrative or ambiguous pronouns (``this``, ``that``, \
``it``, ``they``) to their concrete referents using neighboring turns \
when needed.
- First-person plural (``we``, ``our``) refers to {speaker} together \
with one or more named others; preserve the plural and name the \
others when identifiable.

MESSAGE DATE vs EVENT-MENTIONED DATE:
- The MESSAGE was sent on {date}. The retrieval system automatically \
prepends the message timestamp to each statement when surfacing it for \
question answering. DO NOT include the message date {date} in your \
statement text in any form -- not as a suffix, not as a sentence \
prefix, not parenthesized, not bracketed.
- Any date that appears in your statement text refers to an EVENT \
MENTIONED in the message, NOT the message itself.

CHAIN-OF-THOUGHT PROCEDURE (follow these three steps in order):

STEP 1 -- DETECT AND RESOLVE relative time phrases.
Scan the MESSAGE for every relative time reference. Relative phrases \
include but are not limited to: ``today``, ``tonight``, ``right now``, \
``just``, ``earlier today``, ``yesterday``, ``last night``, \
``tomorrow``, ``last week``, ``next week``, ``this weekend``, ``last \
weekend``, ``next month``, ``last month``, ``last year``, ``this \
year``, ``this summer``, ``this winter``, ``three years ago``, ``two \
months ago``, ``next Friday``, ``the weekend``, ``the holidays``, \
``recently``, and any bare month/day reference whose year is implicit. \
For EACH such phrase, populate one entry in ``relative_resolutions`` \
with:
  * ``phrase``: the phrase exactly as it appears in the message.
  * ``resolved_date``: the absolute ISO date ``YYYY-MM-DD`` it \
resolves to, anchored at the message date {date}. Past-tense and \
backward markers go backward; future-tense and forward markers go \
forward; bare month/day references resolve to the nearest occurrence \
consistent with the surrounding tense; bare durations ending ``now`` \
resolve to a span ending at the message date; bare shortcuts \
(``the weekend``, ``the holidays``) resolve to the nearest past \
occurrence when no tense marker is present.
  * ``equals_message_date``: ``true`` if ``resolved_date`` equals \
{date}, otherwise ``false``. Set ``true`` for ``today``, ``tonight``, \
``right now``, ``just``, ``earlier today``, and for any case where \
the verb is present-tense or simple-past describing the current day \
with no temporal marker (treat that as an implicit ``today``).

If the message contains NO relative time phrase, ``relative_resolutions`` \
is an empty list.

STEP 2 -- DECIDE date treatment for each resolved phrase.
For each entry in ``relative_resolutions``:
  * If ``equals_message_date`` is ``true``: when you write the final \
statement that covers this phrase, REMOVE the relative phrase entirely \
and include NO date in that statement. The framework's automatic \
prefix carries the message date.
  * If ``equals_message_date`` is ``false``: when you write the final \
statement that covers this phrase, REMOVE the relative phrase and \
weave the absolute date inline as ``on YYYY-MM-DD`` using the \
``resolved_date`` value.
Explicit absolute dates already in the message (``June 14, 2025``, \
``March 3``) are NOT relative phrases and do NOT appear in \
``relative_resolutions``; convert them directly to ``on YYYY-MM-DD`` \
form when writing statements.

STEP 3 -- EMIT the final 1p statements in ``memories``.
Apply the date decisions from Step 2. After applying them, no \
relative phrase from ``relative_resolutions`` may appear anywhere in \
the ``memories`` strings.

A bare relative phrase (``last week``, ``next month``, ``yesterday``, \
``three years ago``, ``the weekend``, ``last year``, ``today``, \
``tonight``, ``recently``) appearing in any ``memories`` string is a \
FAILURE. A statement containing both a relative phrase AND its \
resolved absolute date is a FAILURE (the relative phrase must be \
stripped).
  FAIL:  ``I took a solo trip last year on 2022-05-04.``
  PASS:  ``I took a solo trip on 2022-05-04.``
  FAIL:  ``My laptop crashed last week and I lost all my work.``
  PASS:  ``My laptop crashed on 2022-09-07 and I lost all my work.``
  PASS (resolved date equals message date, both phrase AND date \
dropped):  ``I checked out an art show with a friend and found it \
inspiring.``

CANONICAL DATE FORMAT:
- Use exactly ONE inline date form in ``memories``: ``on YYYY-MM-DD`` \
woven into natural prose.
- Forbidden forms (each is a FAILURE): a sentence-prefix \
``On YYYY-MM-DD, ...``; a parenthetical ``(Date: YYYY-MM-DD)`` or \
``(Event date: YYYY-MM-DD)``; a square-bracket ``[YYYY-MM-DD]``; an \
``as of YYYY-MM-DD`` qualifier; any suffix attaching the message date \
to a statement.
- One event date per statement. A statement that describes events \
occurring on multiple different dates must be split into separate \
statements, each carrying its own event date.

EACH STATEMENT in ``memories``:
- Corresponds to one EVENT in the message, not to one sentence. An \
event is a single occurrence, decision, plan, observation, state, or \
preference at one point or span in time. A multi-sentence elaboration \
of the same event (subject in one sentence, reason in the next, \
outcome after that) is ONE statement that contains all of those \
sentences' particulars. Emitting one statement per sentence is a \
FAILURE; emitting one statement per concrete particular is a FAILURE; \
merging two distinct events into one statement is a FAILURE. Distinct \
events (different times, different occasions, different actions) each \
get their own statement, even when they share participants or topics.
- Contains every concrete particular the message gives about its \
event -- subject, action, time, place, attendees, motivation, outcome, \
attached media.
- Preserves every CONCRETE PARTICULAR from the message verbatim -- \
names, dates other than the dropped message date, numbers, \
identifiers, decisions, plans, preferences, opinions, relationships, \
emotional states tied to events, distinctive phrasing, attached-media \
descriptions. Generic abstractions or stock paraphrases for specifics \
are FAILURES.
- Preserves polarity, direction, and emotional tone. ``Used to`` \
implies no longer; ``didn't get to bed until 2 AM`` implies a late \
end, not a late start.
- Traces every detail back to words in the message. Inferred \
attributes (a title from context, a gender from a name, an age, an \
ethnicity, an unstated role) are FAILURES; explicitly stated \
attributes are preserved.
- Reports the content the message conveys, not the speech-act of \
conveying it. ``I said that ...``, ``I told Y that ...``, ``I \
mentioned that ...`` framing is included ONLY when the speech-act \
itself is the event (an apology, a promise, an explicit announcement). \
For mere conversational reporting, drop the speech-act wrapper.

NEIGHBORING TURNS appear before and after the message strictly to \
help resolve second-person addressees, demonstratives, anaphora, and \
unresolved relative references. Content drawn from the neighbors is \
NEVER emitted -- only content drawn from the message itself.

EXAMPLES (neutral names Alice/Bob/Charlie/Dana, neutral domains; the \
SPEAKER in each example owns the first-person voice). Each example \
shows both the ``relative_resolutions`` trace and the final \
``memories``.

Example 1 -- message-time event; framework prefix carries the date.
MESSAGE FROM Alice on 2026-05-18:
I'm finally trying that ramen place on Castro Street right now -- the \
tonkotsu is incredible.
->
{{
  "relative_resolutions": [
    {{"phrase": "right now", "resolved_date": "2026-05-18", \
"equals_message_date": true}}
  ],
  "memories": ["I am eating tonkotsu ramen at the ramen place on \
Castro Street and find it incredible."]
}}

Example 2 -- mentioned event with a relative reference resolved to a \
different date.
MESSAGE FROM Alice on 2026-04-10:
I adopted my two cockatiels three years ago, right before I moved to \
Portland.
->
{{
  "relative_resolutions": [
    {{"phrase": "three years ago", "resolved_date": "2023-04-10", \
"equals_message_date": false}}
  ],
  "memories": ["I adopted my two cockatiels on 2023-04-10, right \
before I moved to Portland."]
}}

Example 3 -- mentioned event with an EXPLICIT absolute date plus a \
relative phrase that resolves to a span (``last year``).
MESSAGE FROM Bob on 2026-05-02:
Charlie's wedding on June 14, 2025 was the best party I went to last \
year.
->
{{
  "relative_resolutions": [
    {{"phrase": "last year", "resolved_date": "2025-01-01", \
"equals_message_date": false}}
  ],
  "memories": ["I consider Charlie's wedding on 2025-06-14 the best \
party I attended in 2025."]
}}

Example 4 -- speech-act-as-event: a promise made on the message date. \
No relative phrases.
MESSAGE FROM Dana on 2026-05-18:
I promise I'll stop missing our Thursday mandolin practice -- I'll be \
there every week from now on.
->
{{
  "relative_resolutions": [],
  "memories": ["I promise to stop missing the Thursday mandolin \
practice and to attend every week going forward."]
}}

Example 5 -- conversational reporting that should DROP the speech-act \
wrapper; ``last weekend`` resolves to a non-message date.
MESSAGE FROM Alice on 2026-05-18:
I mentioned to Bob that I went hiking in Big Sur last weekend with my \
book club.
->
{{
  "relative_resolutions": [
    {{"phrase": "last weekend", "resolved_date": "2026-05-16", \
"equals_message_date": false}}
  ],
  "memories": ["I went hiking in Big Sur with my book club on \
2026-05-16."]
}}

Example 6 -- bare question with no specific content and no relative \
phrases.
MESSAGE FROM Alice on 2026-05-18:
How are you doing today?
->
{{
  "relative_resolutions": [
    {{"phrase": "today", "resolved_date": "2026-05-18", \
"equals_message_date": true}}
  ],
  "memories": []
}}

Example 7 -- multi-event message that splits into multiple statements; \
two distinct relative phrases resolving to different dates.
MESSAGE FROM Charlie on 2026-05-18:
I climbed Half Dome two months ago and I'm flying to Tokyo next Friday \
for a conference on AI alignment.
->
{{
  "relative_resolutions": [
    {{"phrase": "two months ago", "resolved_date": "2026-03-18", \
"equals_message_date": false}},
    {{"phrase": "next Friday", "resolved_date": "2026-05-22", \
"equals_message_date": false}}
  ],
  "memories": ["I climbed Half Dome on 2026-03-18.", "I am flying to \
Tokyo on 2026-05-22 for a conference on AI alignment."]
}}

Example 8 -- second-person addressee resolution; ``tomorrow`` resolves \
to a non-message date.
PRIOR TURNS (context only, do not emit):
- Bob: Hey Alice, what are you up to this weekend?

MESSAGE FROM Alice on 2026-05-18:
Just baking your favorite cake for our potluck tomorrow.
->
{{
  "relative_resolutions": [
    {{"phrase": "tomorrow", "resolved_date": "2026-05-19", \
"equals_message_date": false}}
  ],
  "memories": ["I am baking Bob's favorite cake for our potluck on \
2026-05-19."]
}}

Output: a JSON object \
{{ "relative_resolutions": [...], "memories": [...] }}. \
``memories`` is empty when the message contains no specific content. \
``relative_resolutions`` is empty when the message contains no \
relative time phrases.

{neighbors_block}MESSAGE FROM {speaker} on {date}:
{passage}"""


class _RelativeResolution(BaseModel):
    """One detected relative time phrase and its absolute resolution."""

    phrase: str
    resolved_date: str
    equals_message_date: bool


class _RewriteResponse(BaseModel):
    """Structured response from the rewriting language model.

    ``relative_resolutions`` is the LLM's chain-of-thought trace: every
    relative time phrase it detected in the message, the absolute date
    it resolved each to, and whether that absolute date equals the
    message date. This scaffolding is NOT consumed downstream; only
    ``memories`` (the final 1p statements) is.
    """

    relative_resolutions: list[_RelativeResolution]
    memories: list[str]


def _format_neighbors(before: list, after: list) -> str:
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
    """First-person rewrite segmenter with explicit CoT date resolution.

    Each emitted segment carries ``ProducerContext(producer=speaker)``;
    paired with ``WholeTextDeriver`` the derivative becomes
    ``"{speaker}: {1p_statement}"`` (speaker visible to the embedder
    and BM25 via the derivative path). The segment's stored block text
    remains bare 1p (no speaker prefix) so the framework's same-event
    merge under one ``[<timestamp>]`` header does not duplicate the
    speaker name.
    """

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_REWRITE_V22_FP_COT,
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
                            segments.append(
                                Segment(
                                    uuid=uuid4(),
                                    event_uuid=event.uuid,
                                    index=block_index,
                                    offset=offset,
                                    timestamp=event.timestamp,
                                    block=TextBlock(text=memory),
                                    context=ProducerContext(producer=speaker),
                                    properties=event.properties,
                                )
                            )
                            offset += 1
                case _:
                    raise NotImplementedError(
                        f"Unsupported block type: {type(block).__name__}"
                    )
        return segments
