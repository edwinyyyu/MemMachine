"""v22-fp-min-v3 RewriteSegmenter — gated LIST FAITHFULNESS rule.

Hypothesis
----------

v2's global LIST FAITHFULNESS rule fixed g4 cat1 multi-hop (80.65% parity
with baseline) but the cat3 inferential advantage shrank from v1's
+36pp to v2's +18pp. Sampled cat3 wins on v1 traced to the natural,
conversational first-person prose voice ("I'm allergic to fur") matching
inferential queries better than v22's third-person ("Alice is allergic
to fur"). v2's list rule, while only firing on real enumerations
semantically, is stated globally in the prompt and primes the model
toward list-emission patterns even on non-enumerable narrative /
preference / opinion content -- shifting style toward enumeration at
the cost of natural prose.

v3 fix
------

Re-frame the LIST FAITHFULNESS rule as CONDITIONAL with an explicit
"when AND ONLY WHEN" gate. Pair it with a counter-example: a
preference / opinion message that stays as natural 1p prose and is
NOT shaped as a list. Keep the firing-case example. Everything else
(KEEP/DROP, one-event-per-statement, speech-act, date discipline, 1p
voice, ProducerContext, neighbors block) is unchanged from v2.

The intent is: the rule is dormant on narrative / preference / opinion
turns (preserving v1's cat3 advantage) and active on real enumerations
(preserving v2's cat1 fix).
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


PROMPT_REWRITE_V22_FP_MIN_V3 = """\
Rewrite the MESSAGE into a JSON list of standalone first-person memory \
statements from {speaker}'s point of view. Each statement is stored \
verbatim and retrieved later by semantic search.

KEEP specific content (names, places, dates, numbers, decisions, plans, \
preferences, opinions, described events, attached media). DROP \
interchangeable content (bare greetings, sign-offs, acknowledgments, \
reactions, generic questions).

ONE statement per EVENT, not per sentence and not per particular. A \
multi-sentence elaboration of the same occurrence is ONE statement \
that contains all of its particulars. Distinct events (different \
times, different occasions, different actions) each get their own \
statement. Report content, not the speech-act of conveying it -- \
``I said that ...`` / ``I told X that ...`` wrappers are dropped \
unless the speech-act itself is the event (a promise, an apology, an \
announcement).

LIST FAITHFULNESS (CONDITIONAL). This rule fires WHEN AND ONLY WHEN \
the raw message contains an explicit enumeration of three or more \
comparable items (a list of names, books, places, foods, brands, \
hobbies, languages, dates, etc.). When it fires, the statement MUST \
preserve EVERY listed item verbatim; summaries that collapse the list \
into ``various``, ``several``, ``a few``, ``many``, ``some``, or a \
partial sample are FAILURES. For messages WITHOUT such an enumeration \
-- narratives, single preferences, opinions, individual events, \
reflections, plans -- this rule is DORMANT: write natural \
conversational first-person prose and do NOT impose list structure or \
bullet-style phrasing. Two comma-separated phrases describing the SAME \
event are not an enumeration; only three-or-more comparable items \
count.

PERSON: keep {speaker}'s own ``I`` / ``me`` / ``my`` / ``myself`` \
verbatim. Everyone else stays third-person -- resolve ``you`` / \
``your`` to the addressee's NAME when known from context, and \
resolve demonstratives (``this``, ``that``, ``it``, ``they``) to \
their concrete referents.

DATES. The MESSAGE was sent on {date}. The framework prepends the \
message timestamp automatically when surfacing the statement, so the \
statement text MUST NOT contain {date} in any form. Resolve every \
relative time reference (``yesterday``, ``last week``, ``three years \
ago``, ``next Friday``, ``the weekend``, ``today``, ``tonight``, \
``recently``, ``now``, ``just``) to an absolute date anchored at \
{date}.
  - If the resolved date EQUALS {date}, the statement contains NO \
date and NO relative phrase.
  - If the resolved date DIFFERS from {date}, weave it into prose as \
``on YYYY-MM-DD`` and DELETE the original relative phrase from the \
sentence. The relative phrase appearing alongside the resolved date \
is a FAILURE.
One event date per statement; split multi-date messages into multiple \
statements.

Forbidden date forms (each is a FAILURE): sentence-prefix \
``On YYYY-MM-DD, ...``; parenthetical ``(Date: ...)`` or \
``(Event date: ...)``; square-bracket ``[YYYY-MM-DD]``; \
``as of YYYY-MM-DD``.

NEIGHBORING TURNS (when shown) are resolution context only -- never \
emit their content.

EXAMPLES (neutral names; the SPEAKER owns the first-person voice).

Example 1 -- message-time event; no date in output.
MESSAGE FROM Alice on 2026-05-18:
I'm finally trying that ramen place on Castro Street right now -- the \
tonkotsu is incredible.
->
{{ "memories": ["I am eating tonkotsu ramen at the ramen place on \
Castro Street and find it incredible."] }}

Example 2 -- relative reference resolves to a DIFFERENT date; the \
relative phrase is removed.
MESSAGE FROM Bob on 2026-04-10:
I adopted my two cockatiels three years ago, right before I moved to \
Portland.
->
{{ "memories": ["I adopted my two cockatiels on 2023-04-10, right \
before I moved to Portland."] }}

Example 3 -- explicit absolute date in the message.
MESSAGE FROM Charlie on 2026-05-02:
Dana's wedding on June 14, 2025 was the best party I went to last \
year.
->
{{ "memories": ["I consider Dana's wedding on 2025-06-14 the best \
party I attended in 2025."] }}

Example 4 -- speech-act IS the event.
MESSAGE FROM Dana on 2026-05-18:
I promise I'll stop missing our Thursday mandolin practice -- I'll be \
there every week from now on.
->
{{ "memories": ["I promise to stop missing the Thursday mandolin \
practice and to attend every week going forward."] }}

Example 5 -- bare filler; emit an empty list.
MESSAGE FROM Alice on 2026-05-18:
How are you doing today?
->
{{ "memories": [] }}

Example 6 -- LIST FAITHFULNESS FIRES: the message is an explicit \
enumeration of comparable items; every item is preserved verbatim.
MESSAGE FROM Bob on 2026-05-18:
For book club next month I'm picking between Piranesi, Tomorrow and \
Tomorrow and Tomorrow, The Overstory, Sea of Tranquility, and Klara \
and the Sun.
->
{{ "memories": ["I am picking my book club selection for 2026-06 \
between Piranesi, Tomorrow and Tomorrow and Tomorrow, The Overstory, \
Sea of Tranquility, and Klara and the Sun."] }}

Example 7 -- LIST FAITHFULNESS DORMANT: a preference / opinion turn \
stays as natural first-person prose; NO list structure is imposed.
MESSAGE FROM Charlie on 2026-05-18:
Honestly the new espresso machine has been a letdown -- the milk \
steamer is loud and the shots pull unevenly so my mornings feel \
rushed.
->
{{ "memories": ["I am disappointed with my new espresso machine \
because the milk steamer is loud and the shots pull unevenly, which \
makes my mornings feel rushed."] }}

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
    """First-person rewrite segmenter (minimalist-v3 gated-list variant).

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
        prompt_template: str = PROMPT_REWRITE_V22_FP_MIN_V3,
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
