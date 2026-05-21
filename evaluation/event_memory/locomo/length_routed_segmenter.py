"""Length-routed LLM segmenter for LoCoMo validation.

Dispatches per-Event:
  - Short input (<= threshold chars): use the v47s short-specialized prompt,
    which returns 0 or 1 segment containing the whole message.
  - Long input (> threshold chars): use the v33 production prompt with its
    topic-shift apparatus.

Both paths flow through LLMTextSegmenter so the source-anchored stitching
and windowing logic are preserved. The dispatch is just a thin wrapper
that picks which prompt template to send.

Threshold default 200 chars covers ~76% of LoCoMo messages -- the bulk
where over-fragmentation lives. Longer messages (~24%) carry through to
v33 because they may have genuine internal topic shifts.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Probe path -- v47s prompt lives in the longmemeval probe directory.
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent / "longmemeval" / "llm_pipeline_probe"),
)

from memmachine_server.common.language_model import LanguageModel
from memmachine_server.episodic_memory.event_memory.data_types import (
    Event,
    Segment,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.segmenter.llm_text_segmenter import (
    PROMPT_F_NATURAL_V33,
    LLMTextSegmenter,
)
from memmachine_server.episodic_memory.event_memory.segmenter.segmenter import (
    Segmenter,
)
from probe_segmenter_short_v47 import PROMPT_SHORT_V47


class LengthRoutedSegmenter(Segmenter):
    """Per-event length-routed LLM segmenter.

    Args:
        language_model: shared LanguageModel for both short and long calls.
        threshold_chars: route inputs <= threshold_chars to the short prompt,
            larger inputs to the long (v33) prompt.
    """

    def __init__(
        self,
        *,
        language_model: LanguageModel,
        threshold_chars: int = 200,
    ) -> None:
        self._threshold_chars = threshold_chars
        self._short = LLMTextSegmenter(
            language_model=language_model,
            prompt_template=PROMPT_SHORT_V47,
        )
        self._long = LLMTextSegmenter(
            language_model=language_model,
            prompt_template=PROMPT_F_NATURAL_V33,
        )

    async def segment(self, event: Event) -> list[Segment]:
        for block in event.blocks:
            if isinstance(block, TextBlock):
                if len(block.text) <= self._threshold_chars:
                    return await self._short.segment(event)
                return await self._long.segment(event)
        return await self._long.segment(event)
