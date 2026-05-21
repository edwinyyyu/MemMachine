"""v22-min3p-v2 RewriteSegmenter — minimalist 3p rewrite + LIST FAITHFULNESS.

Hypothesis
----------

v22-min3p already wins fb cat1 by +2.13pp (83.69% vs 81.56% baseline) and
ties / slightly beats baseline on c124 / c1234. The list-faithfulness
rule (unconditional form) lifted cat1 by +6.46pp on g4 in the fp-min v2
ablation. Stacking it onto min3p targets the same enumeration-collapse
failure mode that still bounds min3p cat1 from above.

  - cat1 (single-hop) lookups frequently ask "which X has Y mentioned?"
    or "how many X has Y done?" -- both fail when the segmenter
    compresses enumerations into ``various`` / ``several`` / a partial
    sample.
  - The min3p prompt has no rule forbidding that compression; v22-fp-min
    v2's empirical lift on cat1 came from the rule, not from the 1p
    voice (3p min won cat1 on fb without the rule).

v2 fix
------

Take v22-min3p verbatim. Insert ONE unconditional LIST FAITHFULNESS
paragraph immediately after the ONE-statement-per-EVENT paragraph (so
the model sees the list rule before reaching person / date discipline).
Adapt phrasing from 3p ``{speaker} said that ...`` to be analogous to
the fp-min-v2 paragraph. Add two 3p list-faithful examples (book club
selection; climbing partners + gyms) with neutral names. Everything
else (KEEP/DROP, 3p voice rule, date discipline, forbidden formats,
neighbors block, ProducerContext) is byte-identical to v22-min3p.

Anticipated failure mode
------------------------

The list rule fires on legitimate three-or-more enumerations but may
also fire on non-list comma-separated descriptors that the model now
refuses to compress, producing slightly longer statements without
recall benefit. This is the same risk fp-min-v2 ran (and won on cat1
despite). If v2 ties baseline on fb, the next angle is to push the
``three-or-more comparable items`` threshold language into a stronger
form (``every named brand / title / person / place / food the message
lists must appear in the statement``) since the failure mode is
extraction-recall on listed nouns specifically.
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


PROMPT_REWRITE_V22_MIN3P_V2 = """\
Rewrite the MESSAGE into a JSON list of standalone third-person memory \
statements about {speaker}. Each statement is stored verbatim and \
retrieved later by semantic search.

KEEP specific content (names, places, dates, numbers, decisions, plans, \
preferences, opinions, described events, attached media). DROP \
interchangeable content (bare greetings, sign-offs, acknowledgments, \
reactions, generic questions).

ONE statement per EVENT, not per sentence and not per particular. A \
multi-sentence elaboration of the same occurrence is ONE statement \
that contains all of its particulars. Distinct events (different \
times, different occasions, different actions) each get their own \
statement. Report content, not the speech-act of conveying it -- \
``{speaker} said that ...`` / ``{speaker} told X that ...`` wrappers \
are dropped unless the speech-act itself is the event (a promise, an \
apology, an announcement).

LIST FAITHFULNESS. When the raw message contains an enumeration (a \
list of three or more comparable items: names, books, places, foods, \
brands, hobbies, languages, dates, etc.), the statement MUST preserve \
EVERY listed item verbatim. Summaries that collapse the list into \
``various``, ``several``, ``a few``, ``many``, ``some``, or a partial \
sample are FAILURES. If the list itself is the queryable content, the \
statement is a single enumeration -- not multiple statements. Two \
comma-separated phrases that describe the SAME event are not an \
enumeration; only three-or-more comparable items count.

PERSON: refer to the speaker by name ({speaker}). Resolve \
first-person self-references (``I`` / ``me`` / ``my`` / ``mine`` / \
``myself``) in the raw to {speaker}'s name on first occurrence; \
subsequent in-statement references use {speaker}'s pronouns \
(he/she/they + his/her/their + him/her/them). Everyone else also \
stays third-person -- resolve ``you`` / ``your`` to the addressee's \
NAME when known from context, and resolve demonstratives (``this``, \
``that``, ``it``, ``they``) to their concrete referents.

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

EXAMPLES (neutral names; the SPEAKER is referred to by name).

Example 1 -- message-time event; no date in output.
MESSAGE FROM Alice on 2026-05-18:
I'm finally trying that ramen place on Castro Street right now -- the \
tonkotsu is incredible.
->
{{ "memories": ["Alice is eating tonkotsu ramen at the ramen place on \
Castro Street and finds it incredible."] }}

Example 2 -- relative reference resolves to a DIFFERENT date; the \
relative phrase is removed.
MESSAGE FROM Bob on 2026-04-10:
I adopted my two cockatiels three years ago, right before I moved to \
Portland.
->
{{ "memories": ["Bob adopted his two cockatiels on 2023-04-10, right \
before he moved to Portland."] }}

Example 3 -- explicit absolute date in the message.
MESSAGE FROM Charlie on 2026-05-02:
Dana's wedding on June 14, 2025 was the best party I went to last \
year.
->
{{ "memories": ["Charlie considers Dana's wedding on 2025-06-14 the \
best party he attended in 2025."] }}

Example 4 -- speech-act IS the event.
MESSAGE FROM Dana on 2026-05-18:
I promise I'll stop missing our Thursday mandolin practice -- I'll be \
there every week from now on.
->
{{ "memories": ["Dana promises to stop missing the Thursday mandolin \
practice and to attend every week going forward."] }}

Example 5 -- bare filler; emit an empty list.
MESSAGE FROM Alice on 2026-05-18:
How are you doing today?
->
{{ "memories": [] }}

Example 6 -- LIST FAITHFULNESS: a single enumeration of comparable \
items must keep every item verbatim.
MESSAGE FROM Bob on 2026-05-18:
For book club next month I'm picking between Piranesi, Tomorrow and \
Tomorrow and Tomorrow, The Overstory, Sea of Tranquility, and Klara \
and the Sun.
->
{{ "memories": ["Bob is picking his book club selection for 2026-06 \
between Piranesi, Tomorrow and Tomorrow and Tomorrow, The Overstory, \
Sea of Tranquility, and Klara and the Sun."] }}

Example 7 -- LIST FAITHFULNESS across two distinct lists: each list \
becomes its own statement; items inside each list are preserved \
verbatim.
MESSAGE FROM Charlie on 2026-05-18:
My climbing partners this season are Avery, Jordan, Riley, Casey, and \
Sam. The gyms we rotate through are Movement Sunnyvale, Dogpatch \
Boulders, Planet Granite, and Mission Cliffs.
->
{{ "memories": ["Charlie's climbing partners this season are Avery, \
Jordan, Riley, Casey, and Sam.", "The climbing gyms Charlie rotates \
through this season are Movement Sunnyvale, Dogpatch Boulders, Planet \
Granite, and Mission Cliffs."] }}

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
    """Third-person rewrite segmenter (minimalist-v2 list-faithful variant).

    Each emitted segment carries ``ProducerContext(producer=speaker)``;
    paired with ``WholeTextDeriver`` the derivative becomes
    ``"{speaker}: {3p_statement}"`` (speaker visible to the embedder
    and BM25 via the derivative path). The segment's stored block text
    remains bare 3p (no extra speaker prefix) so the framework's
    same-event merge under one ``[<timestamp>]`` header does not
    duplicate the speaker name.
    """

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_REWRITE_V22_MIN3P_V2,
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
