"""Chunkers must tile text exactly: no gaps, no overlap, joined back they equal the text."""

import numpy as np

from rbench.chunking import _segment, fixed_spans
from rbench.segmenter import split


def test_fixed_windows_tile_and_respect_size():
    starts = np.arange(0, 100, 4)  # 25 tokens of 4 characters
    spans = fixed_spans(starts, 100, 10)
    assert spans == [(0, 40), (40, 80), (80, 100)]


def test_fixed_short_document_is_one_chunk():
    assert fixed_spans(np.arange(0, 20, 4), 20, 10) == [(0, 20)]


def test_recursive_segments_join_back_and_fit():
    text = "First paragraph here.\n\nSecond one, a bit longer. It has two sentences.\n\n  Indented third.\n" * 5
    pieces = split(text, 60)
    assert "".join(pieces) == text
    assert all(len(p) <= 60 for p in pieces)
    spans = _segment((text, 60))
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
