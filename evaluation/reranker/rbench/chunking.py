"""Chunkings: fixed token windows and EventMemory's recursive segmenter, matched in size.

Names:
  fix<N>  "simple cutting": windows of N embeddinggemma tokens. A chunk runs from its first
          token's start to the next chunk's first token's start.
  rec<N>  "recursive": EventMemory's TextSegmenter (segmenter.py) with its character limit
          fitted per dataset so that its mean chunk equals fix<N>'s realized mean. The mean is
          below N where documents, or their last windows, are shorter than N.
  seg<C>  the TextSegmenter at a fixed limit of C characters; seg500 is EventMemory's default.

Why fit the recursive limit: a character limit gives different token sizes on different
text (code, legal prose, tables), so comparing chunkers at one nominal limit would compare
sizes as much as boundaries. Matching the mean leaves boundaries as the difference.

Every chunking tiles its documents exactly (no gaps, no overlap). That keeps the reading
cost of a chunk well defined and lets any gold span be mapped to the chunks covering it.
Chunk texts are stripped of edge whitespace before embedding and reranking, as
EventMemory's TextDeriver does.
"""

from __future__ import annotations

import json
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import cache

import numpy as np

from rbench import text as tok
from rbench.paths import CHUNKS
from rbench.schema import corpus
from rbench.segmenter import split

SIZES = (64, 128, 256, 512, 1024, 2048)
FIT_SAMPLE_DOCS = 400  # documents sampled per binary-search step; the final mean is measured on all
FIT_TOLERANCE = 0.02  # flagged in fit.json when the full corpus misses the target by more
WORKERS = 8


@dataclass(frozen=True)
class ChunkTable:
    doc_index: np.ndarray
    start: np.ndarray
    end: np.ndarray
    tokens: np.ndarray  # embeddinggemma tokens starting in [start, end)
    qtokens: np.ndarray  # Qwen3 tokens starting in [start, end)

    def __len__(self) -> int:
        return len(self.start)

    def texts(self, dataset: str) -> list[str]:
        docs = corpus(dataset).texts
        return [docs[d][s:e].strip() for d, s, e in zip(self.doc_index, self.start, self.end, strict=True)]

    def by_doc(self, n_docs: int) -> list[np.ndarray]:
        """Chunk ids of each document, in document order."""
        order = np.argsort(self.doc_index, kind="stable")
        bounds = np.searchsorted(self.doc_index[order], np.arange(n_docs + 1))
        return [order[bounds[i] : bounds[i + 1]] for i in range(n_docs)]


def load(dataset: str, chunking: str) -> ChunkTable:
    data = np.load(CHUNKS / dataset / f"{chunking}.npz")
    return ChunkTable(**{k: data[k] for k in ("doc_index", "start", "end", "tokens", "qtokens")})


def fit_record(dataset: str) -> dict:
    path = CHUNKS / dataset / "fit.json"
    return json.loads(path.read_text()) if path.exists() else {}


# ------------------------------------------------------------------ splitting


@cache
def _starts(dataset: str, repo: str) -> tuple[np.ndarray, ...]:
    return tuple(tok.token_starts(list(corpus(dataset).texts), repo))


def _count(starts: np.ndarray, start: int, end: int) -> int:
    return int(np.searchsorted(starts, end, "left") - np.searchsorted(starts, start, "left"))


def fixed_spans(starts: np.ndarray, length: int, size: int) -> list[tuple[int, int]]:
    if len(starts) <= size:
        return [(0, length)]
    cuts = [0] + [int(starts[k]) for k in range(size, len(starts), size)]
    return list(zip(cuts, cuts[1:] + [length], strict=True))


def _segment(args: tuple[str, int]) -> list[tuple[int, int]]:
    text, limit = args
    spans, at = [], 0
    for piece in split(text, limit):
        spans.append((at, at + len(piece)))
        at += len(piece)
    return spans or [(0, len(text))]


def recursive_spans(texts: list[str], limit: int, pool: ProcessPoolExecutor) -> list[list[tuple[int, int]]]:
    return list(pool.map(_segment, [(t, limit) for t in texts], chunksize=16))


def _mean_tokens(spans: list[list[tuple[int, int]]], starts: list[np.ndarray]) -> float:
    return float(np.mean([_count(o, s, e) for rows, o in zip(spans, starts, strict=True) for s, e in rows]))


def fit_limit(dataset: str, target: float, pool: ProcessPoolExecutor) -> int:
    """The character limit whose mean chunk is closest to `target` tokens, by binary search."""
    texts, starts = corpus(dataset).texts, _starts(dataset, tok.GEMMA)
    picked = sorted(random.Random(f"fit:{dataset}").sample(range(len(texts)), min(FIT_SAMPLE_DOCS, len(texts))))
    sample_texts, sample_starts = [texts[i] for i in picked], [starts[i] for i in picked]
    lo, hi, best = int(target), int(target * 16), (float("inf"), int(target))
    while lo <= hi:
        mid = (lo + hi) // 2
        got = _mean_tokens(recursive_spans(sample_texts, mid, pool), sample_starts)
        best = min(best, (abs(got - target), mid))
        lo, hi = (mid + 1, hi) if got < target else (lo, mid - 1)
    return best[1]


def _table(dataset: str, spans: list[list[tuple[int, int]]]) -> ChunkTable:
    g, q = _starts(dataset, tok.GEMMA), _starts(dataset, tok.QWEN)
    rows = [(d, s, e, _count(g[d], s, e), _count(q[d], s, e)) for d, doc in enumerate(spans) for s, e in doc]
    cols = [np.asarray(c, dtype=np.int64) for c in zip(*rows, strict=True)]
    return ChunkTable(*cols)


def check_tiling(dataset: str, t: ChunkTable) -> None:
    texts = corpus(dataset).texts
    for d, rows in enumerate(t.by_doc(len(texts))):
        assert t.start[rows[0]] == 0 and t.end[rows[-1]] == len(texts[d]), f"doc {d} not covered"
        assert np.all(t.start[rows[1:]] == t.end[rows[:-1]]), f"doc {d} has a gap or overlap"


def build(dataset: str, chunking: str) -> ChunkTable:
    """Chunk a dataset (cached in data/chunks/<dataset>/<chunking>.npz)."""
    path = CHUNKS / dataset / f"{chunking}.npz"
    if path.exists():
        return load(dataset, chunking)
    texts = list(corpus(dataset).texts)
    fit = fit_record(dataset)
    kind, size = chunking[:3], int(chunking[3:])
    with ProcessPoolExecutor(WORKERS) as pool:
        if kind == "fix":
            spans = [fixed_spans(o, len(t), size) for o, t in zip(_starts(dataset, tok.GEMMA), texts, strict=True)]
            info: dict = {"size": size}
        elif kind == "rec":
            target = build(dataset, f"fix{size}").tokens.mean()
            limit = fit_limit(dataset, float(target), pool)
            spans, info = recursive_spans(texts, limit, pool), {"char_limit": limit, "target": float(target)}
        elif kind == "seg":
            spans, info = recursive_spans(texts, size, pool), {"char_limit": size, "target": None}
        else:
            raise ValueError(f"unknown chunking {chunking}")
    table = _table(dataset, spans)
    check_tiling(dataset, table)
    info.update(chunks=len(table), mean_tokens=float(table.tokens.mean()), median_tokens=float(np.median(table.tokens)),
                max_tokens=int(table.tokens.max()), mean_qtokens=float(table.qtokens.mean()))
    if info.get("target") and abs(info["mean_tokens"] / info["target"] - 1) > FIT_TOLERANCE:
        info["warning"] = "the full corpus misses the target mean by more than the tolerance"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, doc_index=table.doc_index, start=table.start, end=table.end, tokens=table.tokens, qtokens=table.qtokens)
    fit = fit_record(dataset)
    fit[chunking] = info
    (CHUNKS / dataset / "fit.json").write_text(json.dumps(fit, indent=1) + "\n")
    return table
