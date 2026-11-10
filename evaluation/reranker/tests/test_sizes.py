"""Size normalization: the fit segments as TextSegmenter does, and config.CUT holds the fit."""

import asyncio
import uuid
from datetime import UTC, datetime

import pytest

from rbench import config, sizes
from rbench.paths import PREPARED


def test_the_fit_splits_as_text_segmenter_segments():
    """sizes.fit runs TextSegmenter's splitter per text block: the same pieces as segment()."""
    from memmachine_server.episodic_memory.event_memory.data_types import Event, NullContext, TextBlock
    from memmachine_server.episodic_memory.event_memory.segmenter.text_segmenter import TextSegmenter

    text = ("A first sentence of a turn. " * 40 + "\n\n" + "Then a list: one, two, three; " * 30).strip()
    event = Event(uuid=uuid.uuid4(), timestamp=datetime(2026, 1, 1, tzinfo=UTC), context=NullContext(),
                  blocks=[TextBlock(text=text)])
    segmenter = TextSegmenter(120)
    pieces = [s.block.text for s in asyncio.run(segmenter.segment(event))]
    assert pieces == segmenter._text_splitter.split_text(text) and len(pieces) > 5


@pytest.mark.skipif(not (PREPARED / "longmemeval_s").exists(), reason="needs LongMemEval-S prepared")
def test_config_records_the_fit():
    f = sizes.fit("longmemeval_s")
    assert config.CUT[("longmemeval_s", "seg500")] == f.length
    assert abs(f.cut_chars - f.target_chars) < 1.0 and f.length < 500  # TextSegmenter's segments run under its limit
    assert config.cut_for("longmemeval_s", "seg500") == f.chunking and config.cut_for("nq", "seg500") == "cut500"
