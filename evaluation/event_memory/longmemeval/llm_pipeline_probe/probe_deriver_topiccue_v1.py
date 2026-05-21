"""Topic-cue deriver v1.

Adds one or two retrieval anchors per segment shaped as a TOPIC CUE:
a noun phrase that names what the segment is ABOUT in terms a future
curious outsider would use to ask about it. The whole-text derivative
is still emitted so specifics-anchored queries continue to match.

Hypothesis
----------

The v22 segmenter produces consolidated third-person narrative
segments anchored on the concrete particulars the speaker mentioned
("a stuffed animal dog named Tilly that is always with her while
she writes"). That embedding lives near the SPECIFIC ENTITIES the
speaker named (stuffed-animal, dog, Tilly) -- not near the AREA OF
CURIOSITY a future questioner would target ("what does Alice do
while writing", "Alice's writing habits", "Alice's writing
companions").

The v1 generic-shape derivative ("Alice's writing-companion") helps
when the question already names the category noun, but it is still
phrased from the SPEAKER's frame -- the noun ("writing-companion")
is the literal category of the disclosed fact. Many queries don't
hit the disclosed category: they hit the ACTIVITY / RELATION / ROLE
the speaker happened to be inside when the fact was disclosed.

A topic-cue lifts the framing one level: it names the segment's
topic from the OUTSIDER's perspective -- the area of life / activity
/ relationship a future questioner would ask about, not the specific
category noun of the disclosed fact. For the Tilly segment, the
topic-cue is "Alice's writing routine" or "Alice's writing
companions" -- the AREA where the fact lives, not the fact's
category noun.

This gives the segment a second anchor that sits where queries like
"what does Alice do while writing" naturally land. The query embeds
near the topic-cue, the topic-cue dedupes back to the segment by
segment_uuid, and the gold-bearing segment surfaces in top-K.

Distinction from v1's category-noun phrasing
--------------------------------------------

  Segment: "Alice drinks oat-milk lattes every morning before work."
  v1 generic-shape   : "Alice's coffee preference"
  topic-cue          : "Alice's morning routine" or "Alice's drink
                        choices"
  Query that misses v1 but hits topic-cue:
        "What does Alice do in the morning?"

  Segment: "Bob keeps a stuffed dragon on his desk while he edits."
  v1 generic-shape   : "Bob's desk decoration"
  topic-cue          : "Bob's editing routine" or "what Bob keeps
                        nearby while working"
  Query that misses v1 but hits topic-cue:
        "What does Bob do while editing?"

  Segment: "Charlie repainted the bedroom blue and then took a nap."
  v1 generic-shape   : "Charlie's home painting activity"
  topic-cue          : "Charlie's weekend home projects" or
                        "Charlie's afternoon activities"
  Query that misses v1 but hits topic-cue:
        "What did Charlie do this weekend?"

The topic-cue answers the question "what AREA OF CURIOSITY would a
future questioner target to find this segment?" -- not "what
CATEGORY does this fact belong to?".

Output
------

One or two derivatives per segment, in addition to the whole-text
derivative:

  1. WHOLE -- block.text = _format_with_context(segment.context,
     segment.block.text). Identical to WholeTextDeriver. For
     RewriteContext segments this is the rewrite+speaker+raw-chunk
     dual-text embed.

  2. TOPIC-CUE(s) -- 1-2 noun phrases naming the segment's topic
     from a curious outsider's perspective. Emit 2 ONLY when the
     segment genuinely covers two areas of curiosity (e.g., an
     activity AND a relationship, a routine AND a possession). If
     the LLM returns empty (segment has no retrievable topic), no
     topic-cue derivative is emitted.

All derivatives share segment_uuid. The vector store ingests them
independently; retrieval dedupes by segment_uuid post-pool, so the
top-K=7 slot count stays unchanged.

Anticipated failure modes
-------------------------

- OVER-BROAD TOPICS. "Alice's daily life" or "Bob's habits" match
  almost any query about Alice/Bob and pull unrelated segments into
  the top-K. The prompt counters this by constraining the topic-cue
  to be 4-12 words and to name a SPECIFIC area (activity / routine
  / relationship / role / project), not a global life-area.

- OVER-NARROW TOPICS. "Alice's oat-milk latte routine" is too close
  to the segment's specifics and doesn't generalize beyond
  paraphrases. The prompt asks for the AREA OF CURIOSITY, not the
  literal fact's category noun.

- PARAPHRASE MISMATCH. The future query might use "morning habits"
  while the topic-cue says "morning routine". Embedding similarity
  should bridge this, but it isn't guaranteed -- emitting two
  topic-cues (when the segment genuinely supports it) hedges
  vocabulary risk without explicitly enumerating paraphrases.

- WRONG FRAME. The model may slip back into speaker-frame ("Alice's
  Tilly possession") instead of outsider-frame ("Alice's writing
  companions"). The prompt's rule set names the outsider-frame
  explicitly and gives multiple worked examples across domains.

- OVER-EMISSION OF TWO CUES. If the model emits 2 cues for segments
  that only have one area of curiosity, the second cue is noise that
  competes for top-K slots (within-segment, the dedup is fine; but
  across the corpus, weak second-cue topics waste embedding space
  and may share vocabulary with off-target segments). The prompt is
  strict: 2 ONLY when there are two genuinely distinct areas.

- TOPIC-CUE MATCHES MULTIPLE SEGMENTS' TOPIC-CUES. Even with good
  topic-cues, two segments about "Alice's writing routine" (one
  about Tilly, one about her morning tea) will both anchor near
  the same query. This is by design -- the top-K should contain
  BOTH; the question is whether the gold-bearing one is in the K.
  If recall improves but precision drops, this is the trade-off and
  is acceptable for displacement failure-modes.

Why these rules generalize across domains
-----------------------------------------

The rule set is principle-only: "name the AREA OF CURIOSITY a future
questioner would target". It does not bake in domain-specific
vocabulary (no sports/pets/recipes). The worked examples span four
unrelated domains (morning beverages, editing-desk objects, home
projects, music practice) using neutral names (Alice/Bob/Charlie/
Dana) and generic activities -- no overlap with benchmark entities.
The prompt asks the LLM the diagnostic question "what would a
curious outsider ask about?" and lets the model project that across
domains, rather than enumerating area-types it must recognize.

Cost
----

One small LLM call per segment at ingest. Parallelizable. With
gpt-5.4-nano @ low, ingest cost roughly doubles over WholeTextDeriver
baseline; vector storage grows by 1-2 vectors per segment. Query
time unchanged.

Caller guidance
---------------

Pair with the v22 RewriteSegmenter and a SurroundingEventsContext
with before-window=8. The topic-cue derivative is generated from
the segment TEXT only -- it does not see neighbors. The segment is
already self-contained after segmentation, and the topic-cue should
reflect the segment's own area of curiosity, not the broader
conversation thread.
"""

from __future__ import annotations

from typing import override
from uuid import uuid4

from pydantic import BaseModel
from pydantic import Field

from memmachine_server.common.language_model import LanguageModel
from memmachine_server.episodic_memory.event_memory.data_types import (
    Context,
    Derivative,
    NullContext,
    ProducerContext,
    RewriteContext,
    Segment,
    SurroundingEventsContext,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.deriver.deriver import (
    Deriver,
)


PROMPT_TOPICCUE_V1 = """\
Name the SEGMENT's topic as a curious outsider would describe it.

A future user querying memory often does NOT name the specific fact \
or its category noun. They name the AREA OF CURIOSITY -- the \
activity, routine, relationship, role, project, or life-context the \
segment sits inside. Your job is to emit 1-2 noun phrases that name \
that area from the OUTSIDER's perspective, not the speaker's.

Diagnostic question to ask before writing each cue:
  "What AREA OF CURIOSITY would a future questioner target to find \
this segment?"

NOT: "what CATEGORY does the disclosed fact belong to?" -- that is \
the speaker's frame and tends to mirror the segment's specifics.

Rules:
- Each topic-cue is a NOUN PHRASE (4-12 words). Never a question, \
never a full sentence.
- Each topic-cue NAMES THE AREA: an activity, routine, relationship, \
role, project, habit, ritual, or life-context that hosts the fact. \
Not the fact's category noun.
- KEEP entity names (the owner/subject of the area).
- Use OUTSIDER vocabulary: words a future questioner would use \
without already knowing the specifics. Avoid the segment's literal \
specifics (proper-noun titles, exact dates, exact numbers).
- Emit 1 topic-cue by default. Emit 2 ONLY when the segment \
genuinely covers two distinct areas of curiosity (e.g., a routine \
AND a relationship; a project AND a habit). If in doubt, emit 1.
- If the segment has no retrievable area (pure filler, no entity, \
no activity/routine/relationship/role), emit an empty list.

Examples:

SEGMENT: "Alice drinks oat-milk lattes every morning before work \
and listens to a news podcast while she walks to the office."
TOPIC-CUES: ["Alice's morning routine", "Alice's commute habits"]

SEGMENT: "Bob keeps a small stuffed dragon on his desk and says it \
helps him concentrate while he edits manuscripts."
TOPIC-CUES: ["Bob's editing routine"]

SEGMENT: "Charlie repainted the bedroom blue over the weekend and \
took a long nap on the freshly-made bed afterwards."
TOPIC-CUES: ["Charlie's weekend home projects"]

SEGMENT: "Dana practices mandolin for forty minutes after dinner \
and keeps a notebook of finger exercises she wants to master."
TOPIC-CUES: ["Dana's mandolin practice", "Dana's evening routine"]

SEGMENT: "Riley meets her sister Pat for coffee every Sunday and \
they trade book recommendations afterwards at the used bookshop."
TOPIC-CUES: ["Riley's Sunday ritual with Pat", "Riley's book \
recommendations"]

SEGMENT: "Casey mentioned that her physical therapist said \
strengthening her left ankle is the priority before she can run \
again."
TOPIC-CUES: ["Casey's rehab plan"]

SEGMENT: "Jordan said the puppy goes everywhere with him and \
usually sits on the rug by his desk while he programs."
TOPIC-CUES: ["Jordan's programming routine", "Jordan's time with \
his puppy"]

SEGMENT: "Avery confirmed the team trip is on for next month and \
that the group will share a rental cabin near the lake."
TOPIC-CUES: ["Avery's upcoming team trip"]

Output: a JSON object {{ "topic_cues": ["...", "..."] }}. One or \
two entries; never zero unless the segment has no retrievable area.

SEGMENT: {segment_text}"""


class _TopicCueResponse(BaseModel):
    topic_cues: list[str] = Field(default_factory=list)


def _format_with_context(context: Context, text: str) -> str:
    """Mirror WholeTextDeriver's context formatting.

    For RewriteContext, returns text_to_embed (the
    rewrite+speaker+raw-chunk dual-text embed). For other contexts,
    returns the speaker-prefixed text or the bare text.
    """
    match context:
        case ProducerContext(producer=producer):
            return f"{producer}: {text}"
        case SurroundingEventsContext(producer=producer):
            return f"{producer}: {text}"
        case NullContext():
            return text
        case RewriteContext(text_to_embed=text_to_embed):
            return text_to_embed
        case _:
            raise NotImplementedError(
                f"Unsupported context type: {type(context).__name__}"
            )


class GenericDeriver(Deriver):
    """Emits whole-text + 1-2 topic-cue derivatives per segment.

    A "topic-cue" is a 4-12-word noun phrase that names the area of
    curiosity a future questioner would target to find the segment
    (activity / routine / relationship / role / project), framed from
    the OUTSIDER's perspective rather than the speaker's. The topic-
    cue is intended to give the segment a retrieval anchor in the
    region of embedding space where curiosity-shape queries land,
    complementing the specifics-anchored whole-text derivative.

    Args:
        language_model: LanguageModel used to generate the topic-cue
            phrase(s). Configure model + reasoning_effort at
            construction.
        prompt_template: A ``.format(segment_text=...)`` template.
            Defaults to PROMPT_TOPICCUE_V1.
        max_attempts: Retries on retryable language-model errors.
    """

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_TOPICCUE_V1,
        max_attempts: int = 3,
    ) -> None:
        self._language_model = language_model
        self._prompt_template = prompt_template
        self._max_attempts = max_attempts

    async def _generate_topic_cues(self, segment_text: str) -> list[str]:
        prompt = self._prompt_template.format(segment_text=segment_text)
        response = await self._language_model.generate_parsed_response(
            output_format=_TopicCueResponse,
            user_prompt=prompt,
            max_attempts=self._max_attempts,
        )
        if response is None:
            return []
        cues: list[str] = []
        for cue in response.topic_cues or []:
            stripped = (cue or "").strip()
            if stripped:
                cues.append(stripped)
        # Hard cap at 2; the prompt asks for 1-2 but be defensive.
        return cues[:2]

    @override
    async def derive(self, segment: Segment) -> list[Derivative]:
        match segment.block:
            case TextBlock(text=text):
                pass
            case _:
                raise NotImplementedError(
                    f"Unsupported block type: {type(segment.block).__name__}"
                )

        whole_text = _format_with_context(segment.context, text)

        # Always emit the whole-text derivative.
        derivatives: list[Derivative] = [
            Derivative(
                uuid=uuid4(),
                segment_uuid=segment.uuid,
                timestamp=segment.timestamp,
                context=segment.context,
                block=TextBlock(text=whole_text),
                properties=segment.properties,
            )
        ]

        # Generate topic-cue derivative(s); skip if none.
        topic_cues = await self._generate_topic_cues(text)
        for cue in topic_cues:
            derivatives.append(
                Derivative(
                    uuid=uuid4(),
                    segment_uuid=segment.uuid,
                    timestamp=segment.timestamp,
                    context=NullContext(),
                    block=TextBlock(text=cue),
                    properties=segment.properties,
                )
            )

        return derivatives
