"""EventMemory's window rule and union reading of overlapping windows."""

import numpy as np

from rbench import expansion
from rbench.chunking import ChunkTable


def test_split_matches_eventmemory():
    assert expansion.split(3) == (1, 2) and expansion.split(15) == (5, 10) and expansion.split(0) == (0, 0)


def test_windows_stay_in_document_and_reading_skips_repeats(monkeypatch):
    # two documents: chunks 0-4 in doc 0, chunks 5-6 in doc 1
    table = ChunkTable(doc_index=np.array([0, 0, 0, 0, 0, 1, 1]), start=np.array([0, 10, 20, 30, 40, 0, 10]),
                       end=np.array([10, 20, 30, 40, 50, 10, 20]), tokens=np.full(7, 5), qtokens=np.full(7, 5))

    class FakeCorpus:
        doc_ids = ("a", "b")
        texts = ("x" * 50, "y" * 20)

    monkeypatch.setattr(expansion, "corpus", lambda _: FakeCorpus)
    w = expansion.Windows("d", table, 3)
    assert w.ids(2).tolist() == [1, 2, 3, 4]
    assert w.ids(0).tolist() == [0, 1, 2]  # clipped at the document's start
    assert w.ids(5).tolist() == [5, 6]  # never crosses into another document
    assert w.read_order(np.array([2, 3, 5])).tolist() == [1, 2, 3, 4, 5, 6]
    assert w.tokens(2) == 20
