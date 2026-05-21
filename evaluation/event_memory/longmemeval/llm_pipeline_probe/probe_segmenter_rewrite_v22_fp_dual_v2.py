"""v22-fp-dual-v2 RewriteSegmenter -- fp-min-v2 prompt + dual-text embed only.

Hypothesis
----------

fp-dual-v1 lost -81 questions on fb (c124 1206/1444 = 83.52%). Two
confounds in v1:
  (1) Used full v22-style prompt with all FAILURE clauses (longer
      prompt; the fp.py full-v22-1p variant also lost g4 -6).
  (2) Added SPEAKER SELF-IDENTIFICATION rule ("I (Caroline)") to
      anchor BM25, which may have nudged the model to emit MORE,
      SHORTER, more-atomic statements: fp-dual-v1 = 8540 segments
      (+20% vs baseline 7127), avg 21t/seg (vs baseline 33t).

Over-fragmentation reduces per-segment context and hurts retrieval
precision. To isolate the architectural change from the prompt
change, v2:

- Reuses fp-min-v2's MINIMAL prompt verbatim (no self-id rule, no
  v22-style FAILURE clauses beyond what's already in fp-min-v2).
- Changes ONLY the segmenter code: context becomes RewriteContext
  with dual-text embed, and block.text is manually prefixed with
  "{speaker}: " so BM25 retains the speaker name (since
  RewriteContext has no producer header).

Architecture
------------

  block.text = f"{speaker}: {1p_text}"
    -> same string fp-min-v2 produces via ProducerContext's render.
       BM25 input via string_from_segment_context (RewriteContext
       case = [ts] + block.text) is therefore identical to fp-min-v2.

  context = RewriteContext(text_to_embed=f"{1p_text}\\n{speaker}: {raw_chunk}")
    -> v22 baseline's embed pattern (dual-text with raw_chunk).
       Embedder and BM25-via-derivative get the dual-text.

This isolates the EMBED change from fp-min-v2's existing setup.
Hypothesis: fp-min-v2 lost -18 from missing raw_chunk in embed; v2
restores raw_chunk → retrieval should recover much of the loss.

Anticipated failure modes
-------------------------

1. Block-text manual prefix may interact badly with same-event merge
   in string_from_segment_context (RewriteContext path strips
   producer from header but block.text now duplicates the prefix
   when multiple segments share an event). Mitigation: same-event
   merge concatenates blocks under one timestamp header, so
   duplicate "{speaker}: " prefixes within one rendered block are
   acceptable (BM25 just sees redundant tokens, weighted down).
2. The 1p body still tends to be shorter than 3p (~24t vs ~33t),
   so per-segment retrieval surface is smaller. The dual-text embed
   adds the raw_chunk, which compensates on the EMBEDDING channel
   but not the BM25 channel.
3. Display: the rendered string is `[<ts>] {speaker}: {1p_text}` --
   identical to fp-min-v2's display. The QA model has seen this
   format before; no readability concern.

Generalizability: the manual-prefix-in-block-text pattern is a
deliberate code-level decoupling tactic; not LongMemEval-specific.

Everything else (KEEP/DROP, ONE statement per EVENT, LIST FAITHFULNESS,
PERSON, DATES, NEIGHBORING TURNS) copied verbatim from fp-min-v2.
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


# Reuse fp-min-v2's prompt verbatim (no SELF-ID, no full-v22 FAILURE rules).
PROMPT_REWRITE_V22_FP_DUAL_V2 = """\
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

LIST FAITHFULNESS. When the raw message contains an enumeration (a \
list of three or more comparable items: names, books, places, foods, \
brands, hobbies, languages, dates, etc.), the statement MUST preserve \
EVERY listed item verbatim. Summaries that collapse the list into \
``various``, ``several``, ``a few``, ``many``, ``some``, or a partial \
sample are FAILURES. If the list itself is the queryable content, the \
statement is a single enumeration -- not multiple statements. Two \
comma-separated phrases that describe the SAME event are not an \
enumeration; only three-or-more comparable items count.

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

Example 6 -- LIST FAITHFULNESS: a single enumeration of comparable \
items must keep every item verbatim.
MESSAGE FROM Bob on 2026-05-18:
For book club next month I'm picking between Piranesi, Tomorrow and \
Tomorrow and Tomorrow, The Overstory, Sea of Tranquility, and Klara \
and the Sun.
->
{{ "memories": ["I am picking my book club selection for 2026-06 \
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
{{ "memories": ["My climbing partners this season are Avery, Jordan, \
Riley, Casey, and Sam.", "The climbing gyms I rotate through this \
season are Movement Sunnyvale, Dogpatch Boulders, Planet Granite, and \
Mission Cliffs."] }}

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
    """fp-min-v2 prompt with v22 dual-text embed architecture.

    block.text = "{speaker}: {1p_text}" (manual prefix for BM25)
    context = RewriteContext(text_to_embed=f"{1p_text}\\n{speaker}: {raw_chunk}")
    """

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_REWRITE_V22_FP_DUAL_V2,
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
                            # Manual speaker prefix in block.text so BM25
                            # (which uses string_from_segment_context for
                            # RewriteContext = "[ts] {block.text}") sees the
                            # speaker name.
                            block_text = f"{speaker}: {memory}"
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
                                    block=TextBlock(text=block_text),
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
