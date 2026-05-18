"""Tests for CueWorthinessFilteringDeriver."""

from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid4

import pytest
from pydantic import BaseModel

from memmachine_server.common.language_model import LanguageModel
from memmachine_server.episodic_memory.event_memory.data_types import (
    Block,
    NullContext,
    Segment,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.deriver.cue_worthiness_filtering_deriver import (
    CueWorthinessFilteringDeriver,
)
from memmachine_server.episodic_memory.event_memory.deriver.text_deriver import (
    WholeTextDeriver,
)

pytestmark = pytest.mark.asyncio

_TS = datetime(2026, 1, 15, 10, 30, tzinfo=UTC)


def _make_segment(*, text: str) -> Segment:
    return Segment(
        uuid=uuid4(),
        event_uuid=uuid4(),
        index=0,
        offset=0,
        timestamp=_TS,
        context=NullContext(),
        block=TextBlock(text=text),
        properties={},
    )


def _make_segment_with_unsupported_block() -> Segment:
    class _OtherBlock(BaseModel):
        block_type: str = "other"

    return Segment.model_construct(
        uuid=uuid4(),
        event_uuid=uuid4(),
        index=0,
        offset=0,
        timestamp=_TS,
        context=NullContext(),
        block=cast(Block, _OtherBlock()),
        properties={},
    )


class _FakeLanguageModel(LanguageModel):
    """Minimal LanguageModel stub that returns a canned verdict per prompt.

    Only `generate_response` is exercised by the deriver; the other ABC
    methods are stubbed to satisfy the abstract base.
    """

    def __init__(self, response: str | dict[str, str] | Exception) -> None:
        # response: a fixed string, a dict mapping the unique cue text to a
        # response, or an Exception to raise on every call.
        self._response = response
        self.user_prompts: list[str] = []

    async def generate_response(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, str] | None = None,
        max_attempts: int = 1,
    ) -> tuple[str, Any]:
        assert user_prompt is not None
        self.user_prompts.append(user_prompt)
        if isinstance(self._response, Exception):
            raise self._response
        if isinstance(self._response, dict):
            for key, value in self._response.items():
                if key in user_prompt:
                    return value, None
            raise AssertionError(
                f"No matching key in fake response dict for prompt: {user_prompt!r}"
            )
        return self._response, None

    async def generate_parsed_response(self, *args, **kwargs):
        raise NotImplementedError

    async def generate_response_with_token_usage(self, *args, **kwargs):
        raise NotImplementedError


class TestCueWorthinessFilteringDeriver:
    async def test_keep_passes_through_inner_derivatives(self):
        seg = _make_segment(text="I'm in Chicago.")
        lm = _FakeLanguageModel("KEEP")
        deriver = CueWorthinessFilteringDeriver(
            inner=WholeTextDeriver(), language_model=lm
        )

        result = await deriver.derive(seg)

        assert len(result) == 1
        assert result[0].block == TextBlock(text="I'm in Chicago.")
        assert len(lm.user_prompts) == 1
        # Prompt should embed the segment's raw text.
        assert "I'm in Chicago." in lm.user_prompts[0]

    async def test_reject_returns_empty_and_skips_inner(self):
        seg = _make_segment(text="Acknowledged.")
        lm = _FakeLanguageModel("REJECT")

        # Inner deriver that would explode if called — proves it's skipped.
        class _ExplodingInner(WholeTextDeriver):
            async def derive(self, segment):  # type: ignore[override]
                pytest.fail("Inner deriver should not run when classifier rejects")

        deriver = CueWorthinessFilteringDeriver(
            inner=_ExplodingInner(), language_model=lm
        )

        result = await deriver.derive(seg)

        assert result == []

    async def test_classifier_sees_raw_text_not_formatted(self):
        seg = Segment(
            uuid=uuid4(),
            event_uuid=uuid4(),
            index=0,
            offset=0,
            timestamp=_TS,
            context=NullContext(),
            block=TextBlock(text="Coca-Cola Company"),
            properties={},
        )
        lm = _FakeLanguageModel("keep")  # case-insensitive
        deriver = CueWorthinessFilteringDeriver(
            inner=WholeTextDeriver(), language_model=lm
        )

        result = await deriver.derive(seg)

        assert len(result) == 1
        assert result[0].block == TextBlock(text="Coca-Cola Company")
        # The prompt embeds the raw text, NOT a "User: ..." formatted form.
        assert "Coca-Cola Company" in lm.user_prompts[0]

    async def test_unsupported_block_raises_before_calling_lm(self):
        seg = _make_segment_with_unsupported_block()
        lm = _FakeLanguageModel("KEEP")
        deriver = CueWorthinessFilteringDeriver(
            inner=WholeTextDeriver(), language_model=lm
        )

        with pytest.raises(NotImplementedError, match="Unsupported block type"):
            await deriver.derive(seg)
        assert lm.user_prompts == []

    async def test_unrecognized_response_defaults_to_keep(self):
        """The asymmetric rule: anything not starting with REJECT is KEEP."""
        seg = _make_segment(text="something")
        lm = _FakeLanguageModel("UNCERTAIN")
        deriver = CueWorthinessFilteringDeriver(
            inner=WholeTextDeriver(), language_model=lm
        )

        result = await deriver.derive(seg)

        assert len(result) == 1
        assert result[0].block == TextBlock(text="something")

    async def test_lm_failure_fails_open_to_keep(self):
        seg = _make_segment(text="something")
        lm = _FakeLanguageModel(RuntimeError("API down"))
        deriver = CueWorthinessFilteringDeriver(
            inner=WholeTextDeriver(), language_model=lm, fail_open=True
        )

        result = await deriver.derive(seg)

        assert len(result) == 1
        assert result[0].block == TextBlock(text="something")

    async def test_lm_failure_propagates_when_fail_closed(self):
        seg = _make_segment(text="something")
        lm = _FakeLanguageModel(RuntimeError("API down"))
        deriver = CueWorthinessFilteringDeriver(
            inner=WholeTextDeriver(), language_model=lm, fail_open=False
        )

        with pytest.raises(RuntimeError, match="API down"):
            await deriver.derive(seg)
