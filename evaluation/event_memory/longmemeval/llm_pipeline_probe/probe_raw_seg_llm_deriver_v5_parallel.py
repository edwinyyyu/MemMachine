"""Raw-segmenter + parallel derivatives (v5).

What this iterates from v4
--------------------------

v4 emits a SINGLE 3p rewrite (no dates) per non-filler segment. Per
user guidance ("can embed the raw text + llm generated content in
parallel even"), v5 emits TWO parallel derivatives per non-filler
segment:

  1. The raw chunk verbatim (the speaker's literal words)
  2. The clean 3p rewrite from v4 (no dates, names resolved)

Both are indexed independently in the same vector index. Search may
match a query against either angle, giving the segment two retrieval
chances per query.

Rationale
---------

Embeddings of raw conversational text and 3p paraphrases occupy
different regions of the embedding space. A query phrased as a
question often clusters with literal answer text. A query phrased as
fact-retrieval often clusters with 3p declarative form.

Dedup
-----

The framework's search dedups by event_uuid AFTER retrieval (when
--answer-with-raw-events is set), so two derivatives matching for the
same segment produce ONE raw-event line in the answer context. Cost
is paid only in vector storage and retrieval ranking, not in answer
tokens.

K interpretation
----------------

max_num_segments parameter is enforced at the segment level (post-
retrieval). With 2 derivatives per segment, top-K vector results may
contain duplicates that resolve back to fewer unique segments. Setting
K higher than v1/v3/v4's K may be needed to maintain unique-segment
parity.

Hypothesis
----------

If raw + rewrite parallel beats rewrite-alone by a clear margin,
parallel-angle retrieval is the structural lever. If raw-alone beats
rewrite-alone, the rewrite was adding noise. If parallel ties
rewrite-alone, the two angles are redundant.

Architecture
------------

Segmenter: RawChunkSegmenter (same as v1/v3/v4)
Deriver: emits [raw_chunk_verbatim, no_date_3p_rewrite] for non-filler
        segments; [] for pure filler.
"""

from __future__ import annotations

from typing import override
from uuid import uuid4

from memmachine_server.common.language_model import LanguageModel
from memmachine_server.episodic_memory.event_memory.data_types import (
    Derivative,
    RawSegmentEventContext,
    Segment,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.deriver.deriver import (
    Deriver,
)
from probe_raw_seg_llm_deriver_v4_nodate import (
    PROMPT_DERIVE_V4_NODATE,
    _format_neighbors,
)
from pydantic import BaseModel


class _DeriveResponse(BaseModel):
    rewrite: str


class ParallelLLMRewriteDeriver(Deriver):
    """v5 deriver: emits parallel [raw_chunk, 3p_rewrite_no_dates] per
    non-filler segment, [] for pure filler.

    Tests whether raw-text and abstracted-rewrite derivatives produce
    complementary retrieval coverage. Both derive from the same segment
    so post-retrieval dedup by event_uuid prevents double-counting in
    the answer context.
    """

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        prompt_template: str = PROMPT_DERIVE_V4_NODATE,
        max_attempts: int = 3,
    ) -> None:
        self._language_model = language_model
        self._prompt_template = prompt_template
        self._max_attempts = max_attempts

    @override
    async def derive(self, segment: Segment) -> list[Derivative]:
        if not isinstance(segment.block, TextBlock):
            return []

        if not isinstance(segment.context, RawSegmentEventContext):
            return [
                Derivative(
                    uuid=uuid4(),
                    segment_uuid=segment.uuid,
                    timestamp=segment.timestamp,
                    context=segment.context,
                    block=TextBlock(text=segment.block.text),
                    properties=segment.properties,
                )
            ]

        producer = segment.context.producer
        before = segment.context.before
        event_text = segment.context.current_event_text
        neighbors_block = _format_neighbors(before)
        raw_chunk = segment.block.text

        prompt = self._prompt_template.format(
            speaker=producer,
            passage=event_text,
            neighbors_block=neighbors_block,
        )
        response = await self._language_model.generate_parsed_response(
            output_format=_DeriveResponse,
            user_prompt=prompt,
            max_attempts=self._max_attempts,
        )
        if response is None:
            return []
        rewrite = response.rewrite.strip()
        if not rewrite:
            # Pure filler -> no derivatives -> segment invisible.
            return []

        derivative_texts = [raw_chunk, rewrite]
        return [
            Derivative(
                uuid=uuid4(),
                segment_uuid=segment.uuid,
                timestamp=segment.timestamp,
                context=segment.context,
                block=TextBlock(text=text),
                properties=segment.properties,
            )
            for text in derivative_texts
        ]
