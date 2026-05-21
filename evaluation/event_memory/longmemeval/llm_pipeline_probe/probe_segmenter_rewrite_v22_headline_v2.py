"""v22-headline-v2 RewriteSegmenter — headline style with date-proximity
and list-faithfulness rules.

Hypothesis
----------

v22-headline v1 regressed cat2 (temporal) by -12.50pp on mem0-bench at
K=7. The likely cause: em-dash-separated detail bundles split the
``on YYYY-MM-DD`` date from the action verb, weakening the semantic
binding between event and date for temporal queries. Cat1 (multi-hop)
also regressed -5.41pp, suggesting enumerations were collapsed too
aggressively.

v2 makes TWO targeted changes on top of headline v1:

  1. DATE-PROXIMITY rule. The ``on YYYY-MM-DD`` phrase MUST sit
     immediately after the verb (or after the verb's object). No
     dash, comma, or sub-clause is permitted between the verb and the
     date. The trailing detail bundle, if any, comes AFTER the date.
  2. LIST FAITHFULNESS rule. Enumerations (3+ comparable items) must
     be preserved verbatim. ``various`` / ``several`` / partial
     samples are FAILURES.

Everything else (3p subject, dual-text RewriteContext, terse grammar,
KEEP/DROP rule, speech-act drop, neighbor-context block) is carried
over unchanged from headline v1.
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


PROMPT_REWRITE_V22_HEADLINE_V2 = """\
Rewrite the MESSAGE into a JSON list of HEADLINE-style memory entries \
about {speaker}. Each headline is stored verbatim and retrieved later \
by semantic search.

HEADLINE GRAMMAR. Each headline is a terse fact tuple, NOT a natural \
sentence. Drop articles ("a", "an", "the"), copulas ("is", "was", \
"were", "be") where meaning is unambiguous, and discourse glue \
("that", "which", "and then", "so"). Keep the verb. Use em-dashes \
("--") to separate the main clause from a trailing detail bundle, \
and commas to separate items inside that bundle. One event per \
headline.

  Shape: ``Subject VERB object [on YYYY-MM-DD] [at PLACE] -- detail, \
detail``

KEEP every specific particular from the message: names, places, \
organizations, brands, named activities; dates, times, durations, \
quantities; identifiers, quoted phrases; decisions, plans, \
preferences, opinions, relationships, emotional reactions tied to \
events; attached-media descriptions. DROP interchangeable content \
(bare greetings, sign-offs, acknowledgments, generic reactions, \
phatic questions). Dropping a specific particular is a FAILURE. \
Emitting a headline for purely interchangeable content is a FAILURE.

ONE event per headline. A multi-sentence elaboration of the same \
occurrence (subject, reason, outcome) becomes ONE headline whose \
detail bundle carries all of its particulars. Distinct events \
(different times, different occasions, different actions) each get \
their own headline. Report content, not the speech-act of conveying \
it -- ``X said that ...`` / ``X told Y that ...`` wrappers are \
dropped unless the speech-act itself IS the event (a promise, an \
apology, an announcement).

SUBJECT. The subject of each headline is {speaker}'s NAME (third \
person). Resolve {speaker}'s first-person references ("I", "me", \
"my", "myself") to {speaker}'s name. Resolve "you" / "your" to the \
addressee's name when known. Resolve demonstratives ("this", \
"that", "it", "they") to their concrete referents.

DATES. The MESSAGE was sent on {date}. The framework prepends the \
message timestamp automatically when surfacing the headline, so the \
headline MUST NOT contain {date} in any form. Resolve every relative \
time reference ("yesterday", "last week", "three years ago", "next \
Friday", "the weekend", "today", "tonight", "recently", "now", \
"just") to an absolute date anchored at {date}.
  - If the resolved date EQUALS {date}, the headline contains NO \
date and NO relative phrase.
  - If the resolved date DIFFERS from {date}, weave it into the \
headline as ``on YYYY-MM-DD`` and DELETE the original relative \
phrase. The relative phrase appearing alongside the resolved date \
is a FAILURE.
One event date per headline; split multi-date messages into multiple \
headlines.

DATE PROXIMITY. The inline date phrase ``on YYYY-MM-DD`` MUST appear \
immediately after the verb that names the event, or immediately \
after the object of that verb. Do NOT separate the date from the \
action verb with a dash, a comma, or an intervening sub-clause. The \
trailing detail bundle (em-dash + comma-separated items) comes \
AFTER the date, never before it.
  CORRECT: ``Alice biked with Bob on 2026-05-11 -- freeing, \
beautiful``
  INCORRECT: ``Alice biked with Bob -- on 2026-05-11 -- found it \
freeing``
  INCORRECT: ``Alice, on 2026-05-11, biked with Bob``

Forbidden date forms (each is a FAILURE): sentence-prefix \
``On YYYY-MM-DD, ...``; parenthetical ``(Date: ...)``; \
square-bracket ``[YYYY-MM-DD]``; ``as of YYYY-MM-DD``; date inside \
the trailing detail bundle after an em-dash.

LIST FAITHFULNESS. When the raw message contains an enumeration of \
3 or more comparable items (names, places, foods, books, tools, \
steps, options, etc.), the headline MUST preserve EVERY item \
verbatim. Summarizing with ``various``, ``several``, ``a few``, \
``multiple``, ``and others``, or by sampling only some items is a \
FAILURE. If the list is too long to fit one headline, keep ALL items \
in a single headline anyway -- list completeness outranks brevity.

PRESERVE polarity, direction, and emotional tone. "Used to" implies \
no longer; "didn't get to bed until 2 AM" implies a late end, not a \
late start. Preserve concrete particulars verbatim where possible \
(distinctive nouns, quoted phrases, numbers, named entities).

NEIGHBORING TURNS (when shown) are resolution context only -- never \
emit their content.

EXAMPLES (neutral names; subject is the SPEAKER's name).

Example 1 -- message-time event; no date in headline.
MESSAGE FROM Alice on 2026-05-18:
I'm finally trying that ramen place on Castro Street right now -- \
the tonkotsu is incredible.
->
{{ "memories": ["Alice eating tonkotsu ramen at ramen place on \
Castro Street -- incredible"] }}

Example 2 -- relative date; date sits immediately after the verb's \
object.
MESSAGE FROM Bob on 2026-04-10:
I adopted my two cockatiels three years ago, right before I moved \
to Portland.
->
{{ "memories": ["Bob adopted two cockatiels on 2023-04-10 -- right \
before moving to Portland"] }}

Example 3 -- explicit absolute date; date attaches to the verb, \
detail bundle follows.
MESSAGE FROM Charlie on 2026-05-02:
Dana's wedding on June 14, 2025 was the best party I went to last \
year.
->
{{ "memories": ["Charlie attended Dana's wedding on 2025-06-14 -- \
best party of 2025"] }}

Example 4 -- LIST FAITHFULNESS: every item preserved verbatim.
MESSAGE FROM Dana on 2026-05-18:
For the camping trip next weekend I'm packing my tent, sleeping bag, \
camp stove, headlamp, and water filter.
->
{{ "memories": ["Dana packing for camping trip on 2026-05-23 -- \
tent, sleeping bag, camp stove, headlamp, water filter"] }}

Example 5 -- DATE PROXIMITY: date sits right after the verb, not \
inside the detail bundle.
MESSAGE FROM Alice on 2026-05-18:
I had a blast biking nearby with my neighbor last week -- so \
freeing and beautiful.
->
{{ "memories": ["Alice biked nearby with neighbor on 2026-05-11 -- \
freeing, beautiful"] }}

Example 6 -- bare filler; emit an empty list.
MESSAGE FROM Alice on 2026-05-18:
How are you doing today?
->
{{ "memories": [] }}

Output: a JSON object {{ "memories": [...] }}. The list is empty \
when the message contains no specific content.

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
    """Headline-style rewrite segmenter with date-proximity + list rules.

    Each emitted segment stores the headline as its block text and
    carries a ``RewriteContext`` whose ``text_to_embed`` combines the
    headline with the raw speaker-prefixed chunk -- the same dual-text
    embed channel as v22 baseline, so retrieval still matches against
    the original conversational surface even when a query phrasing
    misses the terse headline form.
    """

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_REWRITE_V22_HEADLINE_V2,
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
        """Embed-channel text: headline + raw speaker-prefixed chunk.

        Dual-text embed preserves v22's retrieval behavior: queries
        phrased like the headline match the terse form, queries
        phrased like the original turn still match the raw chunk.
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
