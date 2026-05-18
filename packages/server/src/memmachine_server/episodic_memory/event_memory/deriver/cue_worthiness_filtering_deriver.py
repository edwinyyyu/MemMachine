"""Deriver that filters out segments unfit as retrieval cues.

A `CueWorthinessFilteringDeriver` wraps any inner `Deriver`. Before the
inner deriver runs, the segment's raw text is shown to a `LanguageModel`
with a principle-only prompt; if the model's response begins with
"REJECT", the segment yields zero derivatives. Otherwise the inner
deriver's output is passed through unchanged.

Asymmetric contract: the model MUST never reject a segment a human would
still want to remember. False keeps add a small amount of vector-store
noise; false rejects lose the memory permanently. The default prompt
encodes this asymmetry ("when in doubt, KEEP"), and any LanguageModel
implementation can be swapped in via dependency injection.
"""

from __future__ import annotations

import logging
from typing import override

from memmachine_server.common.language_model import LanguageModel
from memmachine_server.episodic_memory.event_memory.data_types import (
    Derivative,
    Segment,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.deriver.deriver import Deriver

logger = logging.getLogger(__name__)


# Principle-only prompt. Conveys the rule generically: "even one named
# entity, place, person, brand, concept, preference, specific question,
# or unique phrasing is enough to keep" — distinctive single-word
# concepts ("diabetic", "Coca-Cola Company") flow through, generic
# section labels ("**Resources:**", "continue") get rejected.
DEFAULT_CUE_PROMPT = """\
You are gating which pieces of text get embedded into a long-term retrieval \
index. Decide if the TEXT below would still be useful, on its own, as a \
retrieval cue — judge it standalone, without any surrounding context.

Reject ONLY if the text is purely conversational or scaffolding plumbing \
with no content of its own: bare acknowledgements, greetings, thanks, \
generic "continue"/"more"/"do another" requests, lone formatting fragments \
(headers, bullets, code fences), or single characters/numbers without referent.

Keep everything else. When in doubt, keep. \
Even one named entity, place, person, brand, concept, preference, \
specific question, or unique phrasing is enough to keep.

Reply with exactly one token: KEEP or REJECT.

TEXT:
{text}"""


class CueWorthinessFilteringDeriver(Deriver):
    """Drops segments classified as plumbing; otherwise delegates to inner.

    The classifier sees the segment's raw `TextBlock` text — not the
    formatted-with-context string the inner deriver would emit — because
    the raw text is what was tuned against and what a human would judge
    standalone.

    `language_model` is any `LanguageModel`. Configure the model and any
    reasoning effort at construction of the LanguageModel itself; this
    deriver only calls `generate_response(user_prompt=...)` and inspects
    whether the answer starts with REJECT or KEEP.

    Args:
        inner: The deriver whose output is gated.
        language_model: The cue-worthiness classifier.
        prompt_template: Prompt with a `{text}` placeholder. Defaults to
            the validated principle-only prompt.
        fail_open: When True (default), language-model errors fall back
            to KEEP so transient failures never lose memories.
    """

    def __init__(
        self,
        *,
        inner: Deriver,
        language_model: LanguageModel,
        prompt_template: str = DEFAULT_CUE_PROMPT,
        fail_open: bool = True,
    ) -> None:
        """Initialize the filter; see class docstring for arguments."""
        self._inner = inner
        self._language_model = language_model
        self._prompt_template = prompt_template
        self._fail_open = fail_open

    @override
    async def derive(self, segment: Segment) -> list[Derivative]:
        match segment.block:
            case TextBlock(text=text):
                pass
            case _:
                raise NotImplementedError(
                    f"Unsupported block type: {type(segment.block).__name__}"
                )

        prompt = self._prompt_template.format(text=text)
        try:
            response, _ = await self._language_model.generate_response(
                user_prompt=prompt
            )
        except Exception as e:
            if self._fail_open:
                logger.warning("cue classifier failed, defaulting to KEEP: %s", e)
                return await self._inner.derive(segment)
            raise

        verdict = (response or "").strip().upper()
        if verdict.startswith("REJECT"):
            return []
        # When in doubt, KEEP (asymmetric).
        return await self._inner.derive(segment)
