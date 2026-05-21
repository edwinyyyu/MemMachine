"""v22-headline RewriteSegmenter — ultra-terse headline-style segments.

Hypothesis
----------

v22 baseline and v22-fp-min both emit natural-language sentences (3p
prose and 1p prose respectively). At K=7 with a ≤340-token budget that
allows ~48 tokens per segment after framework headers, but typical v22
sentences hit 25-35 tokens each — usable headroom is being spent on
articles, copulas, and discourse glue ("had a great time at", "found
it to be", "which was", "and we ended up").

This variant compresses each event into a headline-style fact tuple:

    Subject ACTION object [on YYYY-MM-DD] [at place] — detail, detail

Articles, copulas, and connectives are dropped where the meaning is
unambiguous; particulars are preserved verbatim. The goal is to pack
~50% more semantic content into the same K=7 token envelope, leaving
room for details that v22 baseline drops to stay terse (e.g. a third
distinctive particular when an event has many).

Design choices
--------------

  - 3p output: subject is {speaker}'s name (or addressee name / named
    referent for cross-speaker statements). No "I"/"my" forms.
  - Headline grammar: noun + verb (often base or simple-past with the
    auxiliary dropped where grammatical), comma- or em-dash-separated
    detail clauses. Em-dash separates the verb-phrase from a trailing
    detail bundle; commas separate detail items inside the bundle.
  - Dates: only `on YYYY-MM-DD` inline. The relative phrase is
    deleted after resolution. If the resolved date equals {date} the
    headline contains NO date.
  - One event per headline. Same KEEP/DROP rule as v22: keep specific
    content, drop interchangeable content.
  - Speech-act drop unless the speech-act IS the event.
  - RewriteContext(text_to_embed = f"{headline}\\n{speaker}: {chunk}")
    so retrieval still has the raw conversational surface alongside
    the terse headline (preserves v22's dual-text channel).

Estimated cost
--------------

Average headline length: 12-20 tokens (vs v22 prose 25-35t). At K=7
this leaves ~200-260 tokens of the 340 budget free for additional
particulars within each headline, or for the same particular count at
half the token cost.
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


PROMPT_REWRITE_V22_HEADLINE = """\
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

Forbidden date forms (each is a FAILURE): sentence-prefix \
``On YYYY-MM-DD, ...``; parenthetical ``(Date: ...)``; \
square-bracket ``[YYYY-MM-DD]``; ``as of YYYY-MM-DD``.

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

Example 2 -- relative reference resolves to a DIFFERENT date; the \
relative phrase is removed.
MESSAGE FROM Bob on 2026-04-10:
I adopted my two cockatiels three years ago, right before I moved \
to Portland.
->
{{ "memories": ["Bob adopted two cockatiels on 2023-04-10 -- right \
before moving to Portland"] }}

Example 3 -- explicit absolute date in the message.
MESSAGE FROM Charlie on 2026-05-02:
Dana's wedding on June 14, 2025 was the best party I went to last \
year.
->
{{ "memories": ["Charlie attended Dana's wedding on 2025-06-14 -- \
best party of 2025"] }}

Example 4 -- speech-act IS the event.
MESSAGE FROM Dana on 2026-05-18:
I promise I'll stop missing our Thursday mandolin practice -- I'll \
be there every week from now on.
->
{{ "memories": ["Dana promised to stop missing Thursday mandolin \
practice -- attend every week going forward"] }}

Example 5 -- multi-particular event becomes ONE headline with detail \
bundle.
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
    """Headline-style rewrite segmenter (terse 3p fact tuples).

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
        prompt_template: str = PROMPT_REWRITE_V22_HEADLINE,
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
