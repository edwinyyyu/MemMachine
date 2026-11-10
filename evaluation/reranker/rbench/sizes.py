"""Size normalization: the simple chunker's length fitted to TextSegmenter's measured size.

The user's request (2026-11-09; their original: "to normalize chunk sizes, recursive chunking
may require some fitting"): seg<N> stays fixed (EventMemory's TextSegmenter, production's
splitter) and cut<M>'s M is fitted so its measured mean segment size matches seg<N>'s. CPU only,
computed before anything is embedded (`python -m rbench fit-cut`); config.CUT records the result.

What is matched: the statistic the reports print as each chunking's measured size
(report.data.segment_size): per question, the mean characters per segment of the partition it
searches, averaged over questions. Characters because both chunkers cut characters: at equal
mean size the same text is split into about as many segments, so the index holds about as many
vectors and a retrieved seed carries about as much text to the reranker and the reader (tokens
follow; the reports print both). The fit runs over every question of the dataset (sample
"all"), so one length serves every sample of it.

TextSegmenter's limit is a maximum: it splits at natural boundaries, so its segments average
less than N, while the simple chunker cuts at exactly M and only a turn's last piece is short.
M therefore comes out below N.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rbench import partitions as pt
from rbench.samples import sample
from rbench.schema import queries


@dataclass(frozen=True)
class Fit:
    dataset: str
    target: str  # the chunking matched (seg<N>)
    length: int  # M of the fitted cut<M>
    target_chars: float  # the target's measured mean (per question, averaged over questions)
    cut_chars: float  # the fitted cut's
    questions: int

    @property
    def chunking(self) -> str:
        return f"cut{self.length}"


def fit(dataset: str, target: str = "seg500", scope: str | None = None) -> Fit:
    """The integer M whose cut<M> mean segment size comes nearest target's (ties: the larger)."""
    from memmachine_server.episodic_memory.event_memory.segmenter.text_segmenter import TextSegmenter

    if not target.startswith("seg"):
        raise ValueError(f"the fit matches TextSegmenter's size: expected seg<N>, not {target!r}")
    n = int(target[3:])
    rows = queries(dataset)
    picked = sample(dataset, "all")
    parts: dict[str, list[str]] = {}
    order: list[str] = []
    for qi in picked:
        k = pt.key(dataset, rows[qi], scope)
        if k not in parts:
            parts[k] = [b.text for e in pt.part(dataset, rows[qi], scope).events for b in e.blocks]
        order.append(k)
    splitter = TextSegmenter(n)._text_splitter  # what TextSegmenter.segment runs per text block
    pieces: dict[str, tuple[int, int]] = {}  # distinct text -> (segments, characters)
    for texts in parts.values():
        for t in texts:
            if t not in pieces:
                chunks = splitter.split_text(t)
                pieces[t] = (len(chunks), sum(len(c) for c in chunks))
    seg_mean = {k: sum(pieces[t][1] for t in ts) / max(1, sum(pieces[t][0] for t in ts)) for k, ts in parts.items()}
    lengths = {k: np.asarray([len(t) for t in ts], dtype=np.int64) for k, ts in parts.items()}
    target_chars = float(np.mean([seg_mean[k] for k in order]))

    def cut_mean(m: int) -> float:
        per = {k: float(v.sum()) / max(1, int(np.ceil(v / m).sum())) for k, v in lengths.items()}
        return float(np.mean([per[k] for k in order]))

    lo, hi = 1, n  # cut_mean rises with m (each turn's piece count falls); the target lies below n
    while lo < hi:  # the largest m whose mean is at most the target
        mid = (lo + hi + 1) // 2
        if cut_mean(mid) <= target_chars:
            lo = mid
        else:
            hi = mid - 1
    m = min((lo, min(lo + 1, n)), key=lambda x: (abs(cut_mean(x) - target_chars), -x))
    return Fit(dataset, target, m, target_chars, cut_mean(m), len(order))
