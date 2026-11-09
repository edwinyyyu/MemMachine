"""Reader cost of EventMemory's query result: tokens of the string the LLM receives.

As a query result's segment limit grows by one, EventMemory admits one more segment
(eventmemory.admission_order) and renders the admitted set (string_from_segment_contexts):
contexts restricted to admitted segments are joined into components when they share one,
components are ordered chronologically and separated by a blank line, and each run of one
event block's consecutive segments gets one '[timestamp] producer: ' header and its text as
one JSON string. Overlapping windows are therefore counted once.

Re-rendering and tokenizing at every step would be quadratic, so the cost is kept
incrementally under an additive token model:

  cost = sum over admitted segments of inner(s)          (tokens of its JSON-escaped text)
       + sum over runs of run_cost(first segment)         (header + quotes + newline)
       - NEWLINE x components + SEPARATOR x (components - 1)

tests/test_reading.py checks the incremental result against recomputing the model from
scratch at every step (exact), and the model against real tokenization of the rendered
string (within a fraction of a percent on LongMemEval text).
"""

from __future__ import annotations

import bisect
import heapq
from collections.abc import Sequence
from functools import cache

import numpy as np

from rbench.text import GEMMA, count


@cache
def constants() -> tuple[int, int, int]:
    quotes, newline, separator = (int(n) for n in count(['""', "\n", "\n\n"], GEMMA))
    return quotes, newline, separator


class Reader:
    """Cumulative reader cost as segments are admitted one at a time.

    Segment ids must be positions in the partition's chronological order. block[s] is an id
    shared by segments of one event block, inner[s] its text's tokens, run[s] the cost of a
    run starting at s (its header, the two quotes and a newline).

    Contexts are the query result's: string_from_query_result restricts every one of them to
    the admitted segments before joining components, so a context links admitted segments
    even if the segment limit stopped before it. Every context is in the query result from
    the start (lazy=False: a reranked pool, whose vector search limit is its depth), or only
    once admission reaches it (lazy=True: the first stage, whose vector search limit is the
    seeds it reads; pass each admitted segment's context to admit)."""

    def __init__(self, contexts: Sequence[Sequence[int]], block: np.ndarray, inner: np.ndarray, run: np.ndarray,
                 lazy: bool = False):
        self.contexts = [[int(s) for s in c] for c in contexts]
        self.block, self.inner, self.run = block, inner, run
        self.parent = list(range(len(contexts)))
        self.members: dict[int, list[int]] = {}  # root context -> sorted admitted segments
        self.run_cost: dict[int, int] = {}
        self.contexts_of: dict[int, list[int]] = {}  # segment -> opened contexts holding it
        self.admitted: set[int] = set()
        self.opened = 0
        self.total_inner = self.total_runs = self.components = 0
        _, self.newline, self.separator = constants()
        if not lazy:
            self.open(len(self.contexts) - 1)

    def _find(self, c: int) -> int:
        root = c
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[c] != root:
            self.parent[c], c = root, self.parent[c]
        return root

    def _runs(self, members: list[int]) -> int:
        return sum(int(self.run[m]) for i, m in enumerate(members) if i == 0 or self.block[m] != self.block[members[i - 1]])

    def _union(self, contexts: Sequence[int]) -> int:
        """Join the components of these contexts; return the root that holds them."""
        roots = sorted({self._find(c) for c in contexts}, key=lambda r: -len(self.members.get(r, [])))
        keep, others = roots[0], roots[1:]
        joining = [r for r in others if self.members.get(r)]
        if joining:  # components merge: rebuild the merged list
            members = self.members.get(keep, [])
            for r in joining:
                members = list(heapq.merge(members, self.members.pop(r)))
                self.total_runs -= self.run_cost.pop(r)
            self.total_runs -= self.run_cost.get(keep, 0)
            self.members[keep] = members
            self.run_cost[keep] = self._runs(members)
            self.total_runs += self.run_cost[keep]
            self.components -= len(joining)  # keep holds the most segments, so it was not empty either
        for r in others:
            self.parent[r] = keep
        return keep

    def open(self, last: int) -> None:
        """Put contexts up to index `last` in the query result; each links the components of
        the admitted segments it holds."""
        while self.opened <= min(last, len(self.contexts) - 1):
            c = self.opened
            self.opened += 1
            for s in self.contexts[c]:
                self.contexts_of.setdefault(s, []).append(c)
            linked = {c2 for s in self.contexts[c] if s in self.admitted for c2 in self.contexts_of[s]}
            if len(linked) > 1:
                self._union(sorted(linked))

    def admit(self, s: int, context: int | None = None) -> int:
        """Admit segment s (from `context`, when contexts open lazily); return the rendered
        cost of everything admitted so far."""
        if context is not None:
            self.open(context)
        root = self._union(self.contexts_of[s])
        members = self.members.setdefault(root, [])
        if not members:
            self.components += 1
        i = bisect.bisect_left(members, s)
        pred = members[i - 1] if i > 0 else None
        succ = members[i] if i < len(members) else None
        delta = int(self.run[s]) if pred is None or self.block[pred] != self.block[s] else 0
        if succ is not None:
            was_start = pred is None or self.block[pred] != self.block[succ]
            is_start = self.block[s] != self.block[succ]
            delta += int(self.run[succ]) * (int(is_start) - int(was_start))
        members.insert(i, s)
        self.run_cost[root] = self.run_cost.get(root, 0) + delta
        self.total_runs += delta
        self.admitted.add(s)
        self.total_inner += int(self.inner[s])
        return self.total_inner + self.total_runs - self.newline * self.components + self.separator * max(0, self.components - 1)


def reading(contexts: Sequence[Sequence[int]], order: Sequence[int], block: np.ndarray, inner: np.ndarray,
            run: np.ndarray, max_cost: float = float("inf"), context_of: Sequence[int] | None = None) -> np.ndarray:
    """Cumulative rendered cost after each segment of `order` (an admission order) is admitted,
    stopping after the first segment whose cost exceeds max_cost. With context_of (the context
    each admitted segment came from), contexts enter the query result only as admission
    reaches them (Reader lazy)."""
    reader, out = Reader(contexts, block, inner, run, lazy=context_of is not None), []
    for k, s in enumerate(order):
        out.append(reader.admit(int(s), None if context_of is None else int(context_of[k])))
        if out[-1] > max_cost:
            break
    return np.asarray(out, dtype=np.float64)
