"""Context expansion: a retrieved seed chunk is read (or reranked) with its neighbors.

Expansion decouples three units that plain chunking ties together: the unit the embedder
sees (the seed chunk), the unit the reranker sees (the seed, or its window) and the unit
the reader gets (the window). Comparing seeds of 64 tokens expanded to ~256-token windows
with plain 256-token chunks separates "what is embedded" from "what is read".

Windows follow EventMemory's expand_context: for e, the window is e // 3 chunks before the
seed and e - e // 3 after, within the seed's document.

Reading an ordered list of seeds takes each seed's window in document order and skips
chunks already read, so reader cost is union tokens: overlapping windows are not paid twice.
(EventMemory renders each hit's window separately, repeating overlaps; that rendering cost
is higher than the union and is not what is compared here.)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from rbench.chunking import ChunkTable
from rbench.schema import corpus


def split(e: int) -> tuple[int, int]:
    """(before, after) neighbor counts for EventMemory's expand_context e."""
    return e // 3, e - e // 3


@dataclass(frozen=True)
class Windows:
    dataset: str
    table: ChunkTable
    e: int

    @cached_property
    def _position(self) -> tuple[list[np.ndarray], np.ndarray]:
        by_doc = self.table.by_doc(len(corpus(self.dataset).doc_ids))
        position = np.empty(len(self.table), dtype=np.int64)
        for rows in by_doc:
            position[rows] = np.arange(len(rows))
        return by_doc, position

    def ids(self, seed: int) -> np.ndarray:
        """The seed's window: chunk ids in document order, the seed included."""
        by_doc, position = self._position
        rows = by_doc[self.table.doc_index[seed]]
        before, after = split(self.e)
        i = position[seed]
        return rows[max(0, i - before) : i + after + 1]

    def text(self, seed: int) -> str:
        w = self.ids(seed)
        doc = corpus(self.dataset).texts[self.table.doc_index[seed]]
        return doc[self.table.start[w[0]] : self.table.end[w[-1]]].strip()

    def tokens(self, seed: int) -> int:
        return int(self.table.tokens[self.ids(seed)].sum())

    def read_order(self, seeds: np.ndarray) -> np.ndarray:
        """Chunks in reading order when each seed arrives with its window; repeats skipped."""
        seen: set[int] = set()
        out = []
        for s in seeds:
            for c in self.ids(int(s)):
                if int(c) not in seen:
                    seen.add(int(c))
                    out.append(int(c))
        return np.asarray(out, dtype=np.int64)
