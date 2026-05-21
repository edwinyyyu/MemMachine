"""v22-natural-says-v1 -- 3p reportage prose with natural speech verbs.

Hypothesis
----------

v22 baseline reports content WITHOUT speech-act framing in most cases
(per the rule "Reports the content the message conveys, not the
speech-act of conveying it"). But for queries like "what did X say
about Y" or "what did X ask", the speech-act IS the relevant event.

Earlier "says-v1/v2/v3" probes used colon-prefix format
(``Alice says: ...``) which reads awkwardly to the QA model. User
clarification: write natural reportage prose where the speech verb is
an ordinary verb in the sentence:

  "Caroline asks Melanie what her favorite part of the road trip was."
  "Bob says he won a basketball game yesterday."
  "Dana promised to attend every week going forward."

Each statement uses a natural speech verb (says, asks, tells, agrees,
denies, mentions, replies, suggests, promises, wonders, complains,
explains) as the main verb. The content the speaker conveyed follows
naturally in the same sentence.

No inline date is added (uses naturaltime-v2 discipline). No colon
prefix. No "X said ON YYYY-MM-DD". The framework prepends the
timestamp; the segmenter handles speaker reference via the natural
sentence subject.

Architecture
------------

block.text = natural 3p reportage prose
context = RewriteContext(text_to_embed = "{rewrite}\\n{speaker}: {raw_chunk}")

RewriteContext (not ProducerContext) because the segment text itself
names the speaker as sentence subject -- adding the framework's
``Bob: `` header prefix would double the speaker reference.
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


PROMPT_REWRITE_V22_NATURAL_SAYS_V1 = """\
Rewrite the MESSAGE into a JSON list of standalone third-person \
memory statements about {speaker}. Each statement is stored verbatim \
and later retrieved by semantic search.

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

NATURAL REPORTAGE FORMAT. Write each statement as natural English \
prose with a speech verb as the main verb. The verb describes how \
{speaker} conveyed the content. Examples of valid forms:

  Bob says he won a basketball game yesterday.
  Caroline asks Melanie what her favorite part of the road trip was.
  Dana promises to attend every Thursday practice going forward.
  Alice tells Bob that her mother passed away three years ago.
  Charlie wonders whether the new venue will be ready by next month.
  Eve mentions she has been working on her screenplay lately.
  Frank explains that the noise is from construction across the street.
  Gina agrees that the recipe needs more salt.

Use ordinary verbs: says, asks, tells, mentions, agrees, disagrees, \
denies, suggests, promises, wonders, replies, explains, complains, \
notes, observes, admits, recalls. Pick whichever verb most naturally \
fits the speech act.

DO NOT use colon-prefix format. ``Bob says: he won a basketball \
game`` is a FAILURE. ``Bob: he won a basketball game`` is a \
FAILURE. The speech verb is just a regular verb in regular prose.

EACH STATEMENT:
- Corresponds to one EVENT in the message, not to one sentence. An \
event is a single occurrence, decision, plan, observation, state, \
or preference at one point or span in time. A multi-sentence \
elaboration of the same event is ONE statement that contains all \
of those sentences' particulars. Distinct events each get their own \
statement.
- Contains every concrete particular the message gives about the \
event -- subject, action, time, place, attendees, motivation, \
outcome, attached media.
- Names the speaker ({speaker}) as the sentence subject. \
First-person self-references in the raw message resolve to \
{speaker}'s name on first occurrence; subsequent in-statement \
references use pronouns (he/she/they, his/her/their, \
him/her/them). Second-person ``you`` resolves to the addressee's \
name when known.
- Preserves every CONCRETE PARTICULAR verbatim -- names, numbers, \
quoted phrases, distinctive wording, attached-media descriptions. \
Generic paraphrases for specifics are FAILURES.
- Preserves polarity, direction, and emotional tone.

DATE AND TIME HANDLING.

ALLOWED in output:
1. Natural relative phrases the speaker actually used: \
``yesterday``, ``today``, ``tonight``, ``last week``, ``next \
month``, ``three years ago``, ``a few days ago``, ``the weekend``, \
``the holidays``, ``recently``, ``now``, ``just``. KEEP verbatim.
2. Explicit absolute dates the speaker actually stated: ``June 14, \
2025``, ``September 2022``, ``March 3rd``, ``in 2010``. KEEP \
verbatim.

DO NOT translate a relative phrase into an absolute date. ``next \
month`` MUST stay ``next month``, never ``April 2023``. ``three \
years ago`` stays as is.

DO NOT add any date or time anchor the speaker did not state. Each \
of these is a FAILURE:
- bracket prefixes ``[Friday, January 21, 2022] ...`` or \
``[2022-01-21] ...``
- ``On YYYY-MM-DD, ...`` sentence prefix
- trailing ``... on YYYY-MM-DD`` or ``... on January 21, 2022`` \
suffix
- ``... (2022-01-21)`` parenthetical date tail
- ``X said ... on 2022-01-21`` speech-act + date suffix

NEIGHBORING TURNS appear before and after the message strictly to \
help resolve second-person addressees, demonstratives, anaphora, \
and unresolved referents. Content drawn from the neighbors is NEVER \
emitted -- only content drawn from the message itself.

Output: a JSON object {{ "memories": [...] }}. The list is empty \
when the message contains no specific content.

{neighbors_block}MESSAGE FROM {speaker} on {date}:
{passage}"""


class _RewriteResponse(BaseModel):
    memories: list[str]


def _format_neighbors(before: list, after: list, current_speaker: str) -> str:
    lines = []
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
    """3p reportage prose with natural speech verbs as sentence main verbs."""

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_REWRITE_V22_NATURAL_SAYS_V1,
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
        self, chunk: str, speaker: str, date: str, neighbors_block: str
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
                neighbors_block = _format_neighbors(before, after, producer)
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
