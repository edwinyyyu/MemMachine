"""Quality measures of a reading order. This docstring is the benchmark's metric glossary.

No retrieved item is ever counted as a "hit". Every measure is computed per query from gold
characters and spans, then averaged over queries with equal weight (a macro-average); nothing
is pooled across queries.

Reading. Whole chunks are read in ranked order. "Within B tokens" means the longest prefix of
the order whose embeddinggemma tokens sum to at most B (reading stops at the first chunk that
does not fit; overlapping expansion windows are counted once). A prefix's cost is the most any
step up to it cost: EventMemory's rendered context can shrink by a header when a segment joins
two windows, but a reader that stopped at the first chunk that did not fit never gets there.
"Within k" means the first k chunks.

Three readings of a gold span, each defined once and used the same way for every dataset:
  complete   every chunk overlapping the span's extent (trimmed of edge whitespace) has been
             read: the reader has the entire span. The default meaning of "read".
  touched    at least one chunk overlapping the span has been read: a potential hit. This is
             how LongMemEval and most retrieval benchmarks count a retrieved unit. It flatters
             small chunks on long gold units (a chunk may touch a long turn without holding
             the sentence that matters), so it is an optimistic bound.
  fraction   the share of the span's characters inside chunks read (recall below): the
             reading between the two, for gold units much longer than the evidence.

Per query, with alternatives = the acceptable answers (any one suffices) and each answer a
set of spans that are all needed:
  first-gold coverage@B   1 if some span of some answer is read within B tokens, else 0
  every-gold coverage@B   1 if every span of some one answer is read within B tokens, else 0
  first / every           the reader tokens at which those first become 1 (the cost); coverage
                          at any budget B is cost <= B, so B is a continuous knob
  first_touch / every_touch   the same with "touched" instead of "complete": some span of some
                          answer touched / every span of one answer touched
  recall@B                max over answers of (that answer's gold characters inside chunks read
                          within B) / (that answer's gold characters). Partial spans count partially.
  precision@B             (characters read within B that lie in any answer's spans, overlaps once)
                          / (characters read within B); 0 when nothing fits
  F1@B                    2PR / (P + R) of that query's recall and precision (0 when both are 0);
                          the reported F1 is the mean of per-query F1, not F1 of mean P and R
  the @k versions         the same with "within the first k chunks"
  first_k, every_k (and the _touch_k ones)   chunks read when those events happen
  rank                    position of the first chunk that overlaps any gold character. The only
                          "touching" measure; reported in the headroom table, not as quality.
Reported coverage, recall, precision and F1 are means of these per-query values over the
test queries. Anything not reached within the ranking head is inf (coverage 0).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rbench.chunking import ChunkTable
from rbench.schema import corpus, queries

BUDGETS = (250, 500, 1000, 2000, 4000, 8000, 16000)  # table columns (500-4,000 is the realistic range)
KS = (1, 2, 3, 5, 10, 20, 30, 50)
# Recall and precision are stored on regular grids for charts: every 250 reader tokens and
# every k up to 50. Every BUDGETS and KS value is included.
CURVE_BUDGETS = tuple(range(250, 16000 + 1, 250))
CURVE_KS = tuple(range(1, 51))
assert set(BUDGETS) <= set(CURVE_BUDGETS) and set(KS) <= set(CURVE_KS)
INF = math.inf
FIELDS = (["first", "every", "rank", "first_k", "every_k", "first_touch", "every_touch", "first_touch_k", "every_touch_k"]
          + [f"recall_t{b}" for b in CURVE_BUDGETS] + [f"precision_t{b}" for b in CURVE_BUDGETS]
          + [f"recall_k{k}" for k in CURVE_KS] + [f"precision_k{k}" for k in CURVE_KS])


@dataclass(frozen=True)
class Gold:
    """A query's gold in chunk terms.

    alternatives: alternatives -> spans -> ((chunk id, gold characters of that span in it), ...)
    chunk_union:  gold characters per chunk over all spans of all alternatives, overlaps once
    cheapest:     chunk ids of the alternative needing the fewest reader tokens (for the
                  perfect-reranker ordering)
    """

    alternatives: tuple[tuple[tuple[tuple[int, int], ...], ...], ...]
    chunk_union: dict[int, int] | None = None
    cheapest: tuple[int, ...] = ()

    @property
    def union(self) -> dict[int, int]:
        if self.chunk_union is not None:
            return self.chunk_union
        out: dict[int, int] = {}
        for alternative in self.alternatives:
            for span in alternative:
                for c, n in span:
                    out[c] = max(out.get(c, 0), n)
        return out


def trim(text: str, start: int, end: int) -> tuple[int, int]:
    """A span shrunk to its non-whitespace extent (unchanged if it is all whitespace)."""
    s, e = start, end
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return (s, e) if s < e else (start, end)


def _merged_length(intervals: list[tuple[int, int]]) -> int:
    total, last = 0, -1
    for s, e in sorted(intervals):
        if e <= last:
            continue
        total += e - max(s, last)
        last = e
    return total


def gold(dataset: str, table: ChunkTable) -> list[Gold]:
    docs = corpus(dataset)
    by_doc = table.by_doc(len(docs.doc_ids))
    out = []
    for q in queries(dataset):
        alternatives, pieces = [], {}
        for alternative in q["gold"]:
            spans = []
            for doc_id, s, e in alternative:
                s, e = trim(docs.texts[docs.index[doc_id]], s, e)
                rows = by_doc[docs.index[doc_id]]
                hit = rows[(table.start[rows] < e) & (table.end[rows] > s)]
                span = []
                for c in hit:
                    a, b = max(s, int(table.start[c])), min(e, int(table.end[c]))
                    span.append((int(c), b - a))
                    pieces.setdefault(int(c), []).append((a, b))
                spans.append(tuple(span))
            alternatives.append(tuple(spans))
        cost = [sum(int(table.tokens[c]) for c in {c for span in alt for c, _ in span}) for alt in alternatives]
        best = alternatives[int(np.argmin(cost))]
        out.append(Gold(tuple(alternatives), {c: _merged_length(v) for c, v in pieces.items()},
                        tuple(sorted({c for span in best for c, _ in span}))))
    return out


def evaluate(order: np.ndarray, cum: np.ndarray, chars: np.ndarray, g: Gold) -> list[float]:
    """FIELDS for reading chunks in `order` (chunk ids), where cum[i] is the reader cost after
    the first i + 1 of them: cumulative chunk tokens for plain RAG, or the tokens of
    EventMemory's rendered context (reading.py). A prefix costs the most any step up to it
    cost (the rendered cost can dip when windows merge), so costs never decrease."""
    cum = np.maximum.accumulate(np.asarray(cum, dtype=np.float64))
    union = g.union
    position = {int(c): i for i, c in enumerate(order)}

    def last(span) -> float:  # position at which a span is fully read
        ps = [position.get(c) for c, _ in span]
        return INF if not ps or None in ps else max(ps)

    def first_of(span) -> float:  # position at which any part of a span is first read (touched)
        ps = [position[c] for c, _ in span if c in position]
        return min(ps) if ps else INF

    lasts = [[last(span) for span in alternative] for alternative in g.alternatives]
    firsts = [[first_of(span) for span in alternative] for alternative in g.alternatives]
    first_pos = min((p for alt in lasts for p in alt), default=INF)
    every_pos = min((max(alt) for alt in lasts if alt), default=INF)
    first_touch = min((p for alt in firsts for p in alt), default=INF)
    every_touch = min((max(alt) for alt in firsts if alt), default=INF)
    cost = lambda p: float(cum[int(p)]) if p < INF else INF  # noqa: E731
    rank = next((i + 1 for i, c in enumerate(order) if int(c) in union), INF)
    read_chars = np.cumsum(chars[order])
    gold_chars = np.cumsum([union.get(int(c), 0) for c in order])
    # Each answer's gold characters read after each prefix; a chunk counts once, at its first
    # position, and once per span of the answer it overlaps.
    first_seen = np.zeros(len(order), dtype=bool)
    first_seen[np.unique(np.asarray(order, dtype=np.int64), return_index=True)[1]] = True
    answer_read = []
    for alt in g.alternatives:
        total = sum(n for span in alt for _, n in span)
        if total:
            per_chunk: dict[int, int] = {}
            for span in alt:
                for c, n in span:
                    per_chunk[c] = per_chunk.get(c, 0) + n
            got = np.asarray([per_chunk.get(int(c), 0) for c in order], dtype=np.float64) * first_seen
            answer_read.append(np.cumsum(got) / total)
    best_read = np.max(answer_read, axis=0) if answer_read and len(order) else np.zeros(len(order))

    def recall_precision(read: int) -> tuple[float, float]:
        if read == 0:
            return 0.0, 0.0
        return float(best_read[read - 1]), float(gold_chars[read - 1] / max(1, read_chars[read - 1]))

    by_budget = [recall_precision(int(np.searchsorted(cum, b, "right"))) for b in CURVE_BUDGETS]
    by_k = [recall_precision(min(k, len(order))) for k in CURVE_KS]
    return [cost(first_pos), cost(every_pos), rank, first_pos + 1, every_pos + 1,
            cost(first_touch), cost(every_touch), first_touch + 1, every_touch + 1,
            *(r for r, _ in by_budget), *(p for _, p in by_budget), *(r for r, _ in by_k), *(p for _, p in by_k)]
