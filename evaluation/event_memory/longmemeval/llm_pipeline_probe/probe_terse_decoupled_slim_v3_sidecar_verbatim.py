"""SIDECAR: verbatim memory/terse + date resolutions injected into bm25/embed only.

The user's insight (Q1): only the answerer needs truth-accuracy in
visible text. Since [event_date] is rendered alongside the segment when
shown to the answerer, the answerer can read "James adopted Ned last
week" with header [2022-04-08] and compute "first week of April 2022".
The retrieval channels (BM25 + embed) are the ones that need absolute
dates as tokens -- so the LLM emits a separate date_resolutions list
that we append only to bm25_text + embed_text, never to block.text.

Result: stop forcing the LLM to mechanically substitute phrases inside
the memory field (which it does inconsistently in REVIEW v1-v4). The
visible text stays source-faithful; retrieval gets the absolute-date
tokens via the sidecar.

Schema:
  - memory: third-person rewrite; KEEP relative phrases verbatim
  - terse: compressed; same handling (keep relative phrases verbatim)
  - date_resolutions: list of absolute date strings for any relative
    phrases the speaker used (natural format, matched precision)
  - queries: 1-3 questions
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import override
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from memmachine_server.common.language_model import LanguageModel
from memmachine_server.episodic_memory.event_memory.data_types import (
    DecoupledRetrievalContext,
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


PROMPT_SIDECAR = """\
You convert one chat MESSAGE into memory items for later retrieval.

Each item is ONE TOPIC the speaker raises -- a single occasion, plan, \
fact, opinion, or preference -- carrying EVERY detail the message gives \
about that topic. Keeping a topic whole is the priority: several \
sentences about one topic are ONE item, and a run of particulars about \
one thing is ONE item. Never split by sentence or by particular. Open a \
separate item only when the speaker genuinely turns to a different \
topic; most messages raise just one. Emit NOTHING for interchangeable \
filler -- greetings, sign-offs, thanks, acknowledgements, reassurance, \
pleasantries, generic reflections, generic questions.

A PARTICULAR is any detail that makes a topic specific rather than \
generic: names, places, dates, numbers, identifiers, decisions, plans, \
preferences and opinions (with their direction), quoted wording, \
attached-media details. Every particular in the message must reach the \
output -- in an item's statement or its queries. Losing a particular \
is the main failure to avoid.

Each item has four fields:

(A) "memory" -- a third-person statement of the topic, about \
{speaker}. State the content itself, not the act of communicating it \
(drop "{speaker} said that ..." wrappers, unless the communicative act \
IS the point, e.g. a promise or apology). Use {speaker}'s name; \
resolve "I"/"my" to {speaker}, "you" to the person addressed, and \
this/that/there to what they refer to. When the message agrees with, \
confirms, commits to, answers, or continues something raised in a \
prior turn, the rewrite must name the specific subject and purpose \
-- the particular activity, object, place, person, or topic -- \
carried over from the prior turns. Generic placeholders like "the \
other person", "something", "come over", "go together", "do that", \
"will do it" without the specific subject are insufficient: the \
rewrite must be specific enough that a reader who has not seen the \
prior turns can identify what is being agreed to, committed to, or \
answered. Keep every particular.

Dates in the statement: the message's own date ({date}) is attached \
automatically when this memory is surfaced, so the reader of (A) always \
sees the event date alongside the statement. KEEP every relative time \
reference exactly as the speaker wrote it -- "yesterday", "last week", \
"three years ago", "next Friday", "the weekend", "today", "recently", \
"now", "just". Do NOT resolve relative phrases to absolute dates in \
(A); do NOT rewrite stated dates into a different format. The event \
date supplies the anchor; the prose stays in the speaker's natural \
phrasing. Never write {date} itself in (A).

(B) "terse" -- field (A) rewritten in the fewest words that stay \
unambiguous: drop articles, filler and hedges; keep every particular \
and the same date handling (keep relative phrases verbatim). Write \
tight readable prose -- a full sentence, not a headline or telegraphic \
fragment. This is the ONLY text retrieval shows to a reader, so it \
must stand on its own; the event date {date} is appended automatically.

(C) "verbatim" -- the portion of the speaker's message that corresponds \
to this item, kept VERBATIM except for these deictic substitutions:

  - "I"/"my"/"me"/"mine"/"myself" referring to {speaker}: KEEP as is. \
    The display prefixes "{speaker}: " automatically; do not write \
    "{speaker}" inside verbatim.
  - "you"/"your"/"yourself" referring to a specific addressee: REPLACE \
    with that person's NAME (e.g. "you" -> "Calvin") when the \
    addressee is named in neighboring turns.
  - "he"/"she"/"they"/"him"/"her"/"them"/"his"/"hers"/"their" referring \
    to a specific named entity from neighboring turns (and NOT \
    referring to {speaker}): REPLACE with that name.
  - When such a pronoun refers to {speaker}, keep as "I"/"my"/"me" \
    rather than awkwardly substituting "{speaker}". For example, if \
    the message contains "{speaker} said she went home" and the \
    referent of "she" is {speaker}, write "I said I went home", not \
    "{speaker} said {speaker} went home".
  - "this"/"that"/"these"/"those"/"there"/"here" referring to a \
    specific object, place, or event from neighboring turns: REPLACE \
    with the specific referent.
  - Temporal references ("yesterday", "last week", "tomorrow", \
    "today", "this morning", "next Friday", "the weekend", "now", \
    "just", "recently", "three years ago"): KEEP VERBATIM. The event \
    date is rendered alongside; dateinstr resolves these at answer \
    time.

Everything else from the message stays verbatim: wording, phrasing, \
punctuation, capitalization, emoji, attached-media descriptions. Do \
not paraphrase, compress, expand, or reorder. Do not add commentary \
or interpretive framing. If a pronoun's referent is ambiguous from \
neighboring turns, leave it as the original pronoun rather than \
guess.

(D) "date_resolutions" -- a list of absolute date strings for every \
relative time reference that appears in (A) or (B) or (C), anchored \
at {date}. This list is consumed by retrieval only (not shown to the \
reader), so it just needs to contain the resolved dates as tokens. \
Use natural format ("April 8, 2022", "March 2024", "2024") at the \
precision the speaker stated. One entry per relative phrase in source \
order; deduplicate exact repeats. Empty list if the message has no \
relative phrases.

(E) "queries" -- 1 to 3 short questions a user might later ask that \
this item answers. Vary their angle (who/what/when/where/why/how). \
Phrase them as a person would ask; never include the answer.

Treat any NEIGHBORING TURNS shown as context for resolving references \
only; never emit items for them.

Return JSON: {{"items": [{{"memory": "...", "terse": "...", \
"verbatim": "...", "date_resolutions": [...], "queries": ["...", "..."]}}, \
...]}}. Return an empty list if the message has no topic worth \
remembering.

{neighbors_block}MESSAGE FROM {speaker} on {date}:
{passage}"""


_MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})(?:-(\d{2}))?\b")


def _date_aliases_verbose(event_date: datetime, memory_text: str) -> str:
    """Event date + any ISO date inside memory -> natural-form aliases.

    Same shape as bo-natural's _date_aliases. Since SIDECAR keeps memory
    verbatim, the regex will usually find no ISO dates in memory --
    leaving just the event-date alias here. Resolutions enter via the
    separate date_resolutions list (see _format_resolutions).
    """
    dates: set[tuple[int, int, int]] = {
        (event_date.year, event_date.month, event_date.day)
    }
    for match in _ISO_RE.finditer(memory_text):
        year = int(match.group(1))
        month = int(match.group(2))
        day = int(match.group(3)) if match.group(3) else 0
        if 1 <= month <= 12:
            dates.add((year, month, day))
    parts: list[str] = []
    seen: set[str] = set()
    for year, month, day in sorted(dates):
        month_name = _MONTHS[month - 1]
        for alias in (
            f"{month_name} {year}",
            f"{month_name} {day}, {year}" if day else None,
        ):
            if alias and alias not in seen:
                seen.add(alias)
                parts.append(alias)
    return "; ".join(parts)


def _format_resolutions(resolutions: list[str]) -> str:
    """LLM-emitted resolution strings, deduped, joined for the sidecar."""
    seen: set[str] = set()
    cleaned: list[str] = []
    for raw in resolutions:
        if not raw:
            continue
        norm = raw.strip()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        cleaned.append(norm)
    return "; ".join(cleaned)


class _MemoryItem(BaseModel):
    memory: str
    terse: str
    verbatim: str
    date_resolutions: list[str]
    queries: list[str]


class _RewriteResponse(BaseModel):
    items: list[_MemoryItem]


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


class SidecarSegmenter(Segmenter):
    """Verbatim memory/terse + date resolutions injected into bm25/embed.

    `dates_in_embed` toggles whether the resolved-dates sidecar is
    appended to text_to_embed. Default True matches the v1 SIDECAR
    behavior; setting False mirrors bo-natural production (dates in
    bm25 only) so the embed neighborhood isn't polluted with date
    tokens that don't help non-temporal queries.
    """

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_SIDECAR,
        chunk_size: int = 1500,
        max_attempts: int = 3,
        dates_in_embed: bool = True,
    ) -> None:
        self._language_model = language_model
        self._prompt_template = prompt_template
        self._chunk_size = chunk_size
        self._max_attempts = max_attempts
        self._dates_in_embed = dates_in_embed
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
        self, chunk: str, speaker: str, date: str, neighbors_block: str
    ) -> list[_MemoryItem]:
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
        return [item for item in response.items if item.memory and item.memory.strip()]

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
                        items = await self._rewrite_chunk(
                            chunk_stripped, speaker, date_str, neighbors_block
                        )
                        for item in items:
                            memory = item.memory.strip()
                            terse = item.terse.strip() or memory
                            # verbatim = verbatim with personal/spatial deictic
                            # substitution (temporal deictics kept). Speaker
                            # prefix added below for display attribution.
                            verbatim = item.verbatim.strip() or terse
                            display_text = f"{speaker}: {verbatim}"
                            # Build the combined date sidecar: event-date
                            # alias (always) + any LLM resolutions for the
                            # relative phrases preserved verbatim in memory.
                            event_aliases = _date_aliases_verbose(
                                event.timestamp, memory
                            )
                            resolutions = _format_resolutions(item.date_resolutions)
                            date_parts = [p for p in (event_aliases, resolutions) if p]
                            dates_line = "; ".join(date_parts)
                            bm25_text = memory
                            if dates_line:
                                bm25_text = f"{bm25_text}\nDates: {dates_line}"
                            q = " ".join(
                                q.strip() for q in item.queries if q and q.strip()
                            )
                            embed_parts = [memory]
                            if q:
                                embed_parts.append(f"Queries: {q}")
                            embed_parts.append(f"{speaker}: {chunk_stripped}")
                            if dates_line and self._dates_in_embed:
                                embed_parts.append(f"Dates: {dates_line}")
                            embed_text = "\n".join(embed_parts)
                            segments.append(
                                Segment(
                                    uuid=uuid4(),
                                    event_uuid=event.uuid,
                                    index=block_index,
                                    offset=offset,
                                    timestamp=event.timestamp,
                                    block=TextBlock(text=display_text),
                                    context=DecoupledRetrievalContext(
                                        producer=speaker,
                                        text_to_embed=embed_text,
                                        text_to_score_bm25=bm25_text,
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
