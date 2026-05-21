"""v22-fp-rulesfirst RewriteSegmenter -- v22-fp prompt restructured so
the two highest-violation rules (drop-message-date and strip-bare-
relative-after-resolution) lead the prompt in a dedicated section.

Motivation
----------

Auditing v22-fp on 24 LoCoMo turns (gpt-5.4-nano @ low and @ medium)
showed two residual failure modes:

  - ``msg_date_leak``: an inline date equal to the message date the
    framework already prepends (~5-6% of statements).
  - ``bare_relative``: a relative phrase like "last week", "yesterday",
    "next month" left in the output instead of being resolved and
    stripped (~14-20% of statements).

Person-form discipline and forbidden-date-format compliance are
clean (0/35). Raising reasoning effort did not move the needle, so
the bottleneck is prompt structure, not reasoning depth.

The v22-fp prompt buries these two rules ~halfway through ~300 lines.
v22-fp-rulesfirst hoists exactly those two rules into a leading
"FIRST: DATE DISCIPLINE" section with one PASS/FAIL pair each. All
other v22-fp content (KEEP/DROP, person rules, one-event-per-statement,
polarity, preserve-particulars, neighbors-as-context-only, speech-act-
only-when-event, canonical-form and forbidden-form list, neutral
examples) is retained verbatim in substance and lightly re-ordered so
the leading section flows naturally into the body.

Class shape, neighbors formatting, and template fields are unchanged
from v22-fp.
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


PROMPT_REWRITE_V22_FP_RULESFIRST = """\
Rewrite the MESSAGE into a JSON list of standalone first-person memory \
statements from {speaker}'s point of view. Each statement is stored \
verbatim and later retrieved by semantic search. A future user querying \
any specific content in the message should find at least one statement \
that contains that content.

FIRST: DATE DISCIPLINE (read this before anything else; these are the \
two rules most often violated and they override the natural-prose \
instinct to keep conversational time phrases).

RULE 1 -- DROP the message date when the event coincides with it. \
The MESSAGE was sent on {date}. The retrieval system automatically \
prepends the message timestamp to every statement at question-answering \
time. Therefore, when a mentioned event resolves to the SAME day as \
{date} (triggered by ``today``, ``tonight``, ``right now``, ``just``, \
``earlier today``, or simply no temporal marker at all when the verb \
describes the current day), the output statement must contain NO date \
of any form -- no ``on {date}``, no ``today``, no ``tonight``.
  FAIL (message sent on 2026-05-18; event is today):
    ``I am baking a cake for the potluck on 2026-05-18.``
  PASS (message sent on 2026-05-18; event is today; framework prefix \
carries the date):
    ``I am baking a cake for the potluck.``

RULE 2 -- STRIP the bare relative phrase after resolving it. After \
you resolve ``last week`` / ``yesterday`` / ``next month`` / ``three \
years ago`` / ``the weekend`` / ``last night`` / ``recently`` to an \
absolute date, REMOVE the relative phrase from the sentence. The \
relative phrase and the resolved absolute date must NEVER coexist in \
the same statement.
  FAIL (resolved date kept AND relative phrase left in):
    ``I adopted my two cockatiels three years ago on 2023-04-10.``
  PASS (relative phrase stripped, absolute date kept):
    ``I adopted my two cockatiels on 2023-04-10.``

Hold these two rules in mind for every statement you emit. Now apply \
them on top of the rest of the rewrite policy:

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

RELATIVE-TIME RESOLUTION (mechanics behind RULE 1 and RULE 2):
- Resolve every relative time reference (``yesterday``, ``last week``, \
``this summer``, ``three years ago``, ``next month``, ``the weekend``, \
``last night``, ``tonight``, ``today``, ``recently``) into an ABSOLUTE \
date anchored at the message date {date}. Past-tense and backward \
markers go backward; future-tense and forward markers go forward; bare \
month/day references resolve to the nearest occurrence consistent with \
the surrounding tense; bare durations ending ``now`` resolve to a span \
ending at the message date; bare shortcuts (``the weekend``, ``the \
holidays``) resolve to the nearest past occurrence when no tense \
marker is present.
- If the resolved date EQUALS the message date {date}, apply RULE 1: \
the output statement contains NO date.
- If the resolved date DIFFERS from the message date, include it \
inline as ``on YYYY-MM-DD`` AND apply RULE 2: remove the relative \
phrase that produced it.
- Any date that appears in your statement text refers to an EVENT \
MENTIONED in the message, NOT the message itself.

CANONICAL DATE FORMAT:
- Use exactly ONE inline date form: ``on YYYY-MM-DD`` woven into \
natural prose -- for example, ``My tournament was on 2026-03-14``.
- Forbidden forms (each is a FAILURE): a sentence-prefix \
``On YYYY-MM-DD, ...``; a parenthetical ``(Date: YYYY-MM-DD)`` or \
``(Event date: YYYY-MM-DD)``; a square-bracket ``[YYYY-MM-DD]``; an \
``as of YYYY-MM-DD`` qualifier; any suffix attaching the message date \
to a statement.
- One event date per statement. A statement that describes events \
occurring on multiple different dates must be split into separate \
statements, each carrying its own event date.

EACH STATEMENT:
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
SPEAKER in each example owns the first-person voice):

Example 1 -- message-time event; framework prefix carries the date \
(RULE 1 in action).
MESSAGE FROM Alice on 2026-05-18:
I'm finally trying that ramen place on Castro Street right now -- the \
tonkotsu is incredible.
->
{{ "memories": ["I am eating tonkotsu ramen at the ramen place on \
Castro Street and find it incredible."] }}

Example 2 -- mentioned event with a relative reference resolved to a \
different date (RULE 2 in action: ``three years ago`` is stripped).
MESSAGE FROM Alice on 2026-04-10:
I adopted my two cockatiels three years ago, right before I moved to \
Portland.
->
{{ "memories": ["I adopted my two cockatiels on 2023-04-10, right \
before I moved to Portland."] }}

Example 3 -- mentioned event with an EXPLICIT absolute date.
MESSAGE FROM Bob on 2026-05-02:
Charlie's wedding on June 14, 2025 was the best party I went to last \
year.
->
{{ "memories": ["I consider Charlie's wedding on 2025-06-14 the best \
party I attended in 2025."] }}

Example 4 -- speech-act-as-event: a promise made on the message date \
(RULE 1: the promise itself is on the message date, so no inline date).
MESSAGE FROM Dana on 2026-05-18:
I promise I'll stop missing our Thursday mandolin practice -- I'll be \
there every week from now on.
->
{{ "memories": ["I promise to stop missing the Thursday mandolin \
practice and to attend every week going forward."] }}

Example 5 -- conversational reporting that should DROP the speech-act \
wrapper AND strip the relative phrase (RULE 2: ``last weekend`` is \
stripped after resolution).
MESSAGE FROM Alice on 2026-05-18:
I mentioned to Bob that I went hiking in Big Sur last weekend with my \
book club.
->
{{ "memories": ["I went hiking in Big Sur with my book club on \
2026-05-16."] }}

Example 6 -- bare question with no specific content. Emit an empty \
list.
MESSAGE FROM Alice on 2026-05-18:
How are you doing today?
->
{{ "memories": [] }}

Example 7 -- multi-event message that splits into multiple statements \
(both RULE 2 cases: ``two months ago`` and ``next Friday`` are stripped \
after resolution).
MESSAGE FROM Charlie on 2026-05-18:
I climbed Half Dome two months ago and I'm flying to Tokyo next Friday \
for a conference on AI alignment.
->
{{ "memories": ["I climbed Half Dome on 2026-03-18.", "I am flying to \
Tokyo on 2026-05-22 for a conference on AI alignment."] }}

Example 8 -- second-person addressee resolution; ``tomorrow`` resolves \
to a date that differs from the message date, so RULE 2 strips it and \
the absolute date stays.
PRIOR TURNS (context only, do not emit):
- Bob: Hey Alice, what are you up to this weekend?

MESSAGE FROM Alice on 2026-05-18:
Just baking your favorite cake for our potluck tomorrow.
->
{{ "memories": ["I am baking Bob's favorite cake for our potluck on \
2026-05-19."] }}

Output: a JSON object {{ "memories": [...] }}. The list is empty when \
the message contains no specific content.

{neighbors_block}MESSAGE FROM {speaker} on {date}:
{passage}"""


class _RewriteResponse(BaseModel):
    """Structured response from the rewriting language model."""

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
    """First-person rewrite segmenter.

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
        prompt_template: str = PROMPT_REWRITE_V22_FP_RULESFIRST,
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
