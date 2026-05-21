"""v22-min3p-dual RewriteSegmenter -- min3p prompt + v22 dual-text embed.

Hypothesis
----------

min3p (3p minimal prompt + ProducerContext single-text embed) TIES
v22 baseline at +1 question on fb. Only non-losing variant tested.

v22 baseline uses RewriteContext with dual-text embed:
    text_to_embed = f"{3p_rewrite}\\n{speaker}: {raw_chunk}"

min3p uses ProducerContext single-text embed:
    embed = f"{speaker}: {3p_rewrite}"

The dual-text embed has been the most fb-validated working architecture
(89.13%). min3p's minimal prompt produces shorter segments (1.14 vs
1.51 segs/event, ~24t vs ~33t per segment) — fewer tokens overall.

Hypothesis: adding dual-text embed to min3p's minimal prompt may push
min3p from TIE (+1) to WIN. The 3p body + dual-text combination has
been shown to work for v22 baseline; the simpler prompt may produce
cleaner segments that the dual-text embed enriches well.

Failure-mode check: fp-dual-v2 result showed dual-text HURT the 1p
body (-78 fb). But that's a 1p-specific issue (1p body + raw_chunk
have conflicting perspectives). 3p body + raw_chunk should be
consistent (both about {speaker}).

Architecture
------------

block.text = 3p_rewrite (clean, names inline; no manual prefix needed
since 3p rewrite already references speaker by name).

context = RewriteContext(text_to_embed=f"{3p_rewrite}\\n{speaker}: {raw_chunk}")

BM25 input via string_from_segment_context = `[ts] {3p_rewrite}` —
subject names appear inline in the 3p rewrite, so BM25 has the speaker
anchor naturally.

Display rendering = `[ts] {3p_rewrite}` — clean, identical to v22
baseline display.

Pareto goal: same accuracy as baseline (or higher) at fewer tokens
(min3p's segments are shorter on average).
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


# Verbatim copy of min3p's prompt.
PROMPT_REWRITE_V22_MIN3P_DUAL = """\
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
    """min3p prompt with v22 dual-text embed (RewriteContext).

    block.text = clean 3p rewrite (subject name inline)
    context = RewriteContext(text_to_embed=f"{3p_rewrite}\\n{speaker}: {raw_chunk}")
    """

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_REWRITE_V22_MIN3P_DUAL,
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
