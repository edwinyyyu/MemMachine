"""Every dataset as EventMemory partitions: gold spans survive segmentation, offsets locate
segments in their event's text, and documents carry no date or producer."""

import json

import numpy as np
import pytest

from rbench import partitions as pt
from rbench.datasets import INACTIVE
from rbench.paths import PREPARED
from rbench.samples import sample
from rbench.schema import queries

DOCUMENT_DATASETS = [d for d in INACTIVE if (PREPARED / d / "queries.jsonl").exists()]


def test_offsets_locate_stripped_segments():
    text = "  Heading\n\n" + "A sentence that runs on. " * 40 + "\n\n   last words  "
    pieces = [text[s:e].strip() for s, e in ((0, 11), (11, 500), (500, len(text)))]
    found = pt.offsets(text, pieces)
    assert [text[s:e] for s, e in found] == pieces
    with pytest.raises(ValueError):
        pt.offsets(text, ["not in the text"])


@pytest.mark.skipif(not DOCUMENT_DATASETS, reason="needs prepared document datasets")
@pytest.mark.parametrize("chunking", ["seg500", "cut500"])
def test_document_gold_maps_to_segments(chunking):
    """No gold span is lost or empty, and each segment is its event's text at its offsets."""
    for dataset in DOCUMENT_DATASETS:
        rows = queries(dataset)
        for k, members in pt.groups(dataset, sample(dataset, "tiny")).items():
            p = pt.part(dataset, rows[members[0]])
            seg, segments, _ = pt.offline(dataset, k, chunking)
            assert [seg.text(p, i) for i in range(len(seg))] == [s.block.text for s in segments]
            for qi in members:
                g = pt.gold(pt.gold_spans(dataset, rows[qi], p), p, seg)
                assert all(span and sum(n for _, n in span) > 0 for alt in g.alternatives for span in alt), (dataset, k)


@pytest.mark.skipif(not DOCUMENT_DATASETS, reason="needs prepared document datasets")
def test_documents_render_and_embed_without_dates_or_producers():
    from memmachine_server.episodic_memory.event_memory.event_memory import EventMemory

    dataset = DOCUMENT_DATASETS[0]
    q = queries(dataset)[sample(dataset, "tiny")[0]]
    p = pt.part(dataset, q)
    seg, segments, texts = pt.offline(dataset, p.key, "seg500")
    f = pt.formats(dataset)
    for i in np.linspace(0, len(seg) - 1, 5).astype(int):
        plain = json.dumps(seg.text(p, i), ensure_ascii=False)
        assert texts[i] == plain  # the embedding text is the segment's JSON text alone
        assert EventMemory.string_from_segment_context([segments[i]], format_options=f.rerank) == plain
        assert EventMemory.string_from_segment_contexts([[segments[i]]], format_options=f.read) == plain
    assert int(seg.run.min()) == int(seg.run.max())  # no header: every run costs only quotes and newline
