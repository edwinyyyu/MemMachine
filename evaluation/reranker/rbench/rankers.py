"""Rankers: how a query's first-stage head is reordered before reading.

  first_stage   the embedding order, unchanged (the no-reranker baseline; its vector search
                limit is whatever the reader budget reads). With a depth L ("first_stage@n20"),
                the first stage with vector search limit L: only its top L seeds are read, so it
                compares with reranking at the same vector search limit.
  rerank        the head's first `depth` chunks sorted by reranker score, and nothing after
                them: EventMemory reranks every seed it retrieves, so the depth is the vector
                search limit and reading ends with the reranked chunks
  oracle        as rerank, gold first: the chunks of the cheapest complete answer, then
                other gold chunks, then the rest of the pool, in vector order. With seeds
                alone it reads only the evidence first, so for "every gold read in full" no
                reranker over the same pool reads it with fewer tokens: a bound there, and
                only there. It is no bound for first gold (a reranker can read the cheapest
                evidence turn first), for touched readings, for recall and precision, or with
                windows (a reranker can cover the evidence with fewer, non-overlapping windows).

With context expansion (expand = e > 0: EventMemory's expand_context) the head's items are
seeds and each is read with its window:
  first_stage+x   seeds in first-stage order, expanded
  rerank+x        rerank the seeds themselves, then expand the order ("rerank, then expand")
  xrerank+x       rerank each seed's whole window, then read the windows ("expand, then
                  rerank"; what EventMemory's query does when expand_context > 0). Its depth
                  counts seeds, like every ranker's, so both scoring positions rerank the
                  same seeds (the same vector search limit).
  oracle+x        seeds whose window holds gold first (cheapest answer's seeds first)
  bound, bound+x  not a ranking: the cheapest reading of the same pool any ranking could achieve,
                  per reading (bounds.py; exact with seeds alone, a valid floor with windows)

A depth is a count of seeds ("n20"): EventMemory's vector_search_limit, how systems are
configured. Equal reranker input is compared through each configuration's measured reranker
tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# vector search limits replayed (those up to the deepest pool, and the pool itself: depths())
COUNT_DEPTHS = (5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500, 10000)


@dataclass(frozen=True)
class Depth:
    """A vector search limit: how many of the first stage's seeds a ranker reorders."""

    value: int

    @property
    def label(self) -> str:
        return f"n{self.value}"

    def chunks(self, head: np.ndarray) -> int:
        """How many head seeds this depth reranks, capped by the head."""
        return min(self.value, len(head))


DEPTHS = tuple(Depth(v) for v in COUNT_DEPTHS)


def depths(deepest: int) -> tuple[Depth, ...]:
    """The vector search limits replayed for pools up to `deepest` seeds: COUNT_DEPTHS up to it,
    and the pool itself (a measured depth need not be on the grid)."""
    return tuple(Depth(v) for v in sorted({v for v in COUNT_DEPTHS if v <= deepest} | {deepest}))


@dataclass(frozen=True)
class Ranker:
    kind: str  # first_stage | rerank | xrerank | oracle | bound
    model: str | None = None
    depth: Depth | None = None
    expand: int = 0

    @property
    def id(self) -> str:
        tail = f"+x{self.expand}" if self.expand else ""
        if self.kind == "first_stage":
            return "first_stage" + (f"@{self.depth.label}" if self.depth else "") + tail
        head = {"rerank": f"rerank:{self.model}", "xrerank": f"xrerank:{self.model}", "oracle": "oracle",
                "bound": "bound"}[self.kind]
        return f"{head}@{self.depth.label}{tail}"

    def order(self, n: int, sims: np.ndarray, logits: np.ndarray | None, is_gold: np.ndarray) -> np.ndarray:
        """Head positions in reading order: all of them for the first stage, else a
        permutation of the first n (the pool this ranker reorders; nothing after it is read).

        For the oracle, is_gold is a priority: 2 for the cheapest answer's chunks, 1 for other
        gold, 0 for the rest (booleans work too)."""
        if self.kind == "first_stage":
            return np.arange(len(sims) if self.depth is None else n)
        if self.kind == "oracle":
            key = -np.asarray(is_gold[:n], dtype=float)
        elif self.kind in ("rerank", "xrerank"):
            key = -logits[:n]
        else:
            raise ValueError(self.kind)
        return np.argsort(key, kind="stable")


def for_models(models: list[str], expansions: tuple[int, ...] = (), depths: tuple[Depth, ...] = DEPTHS,
               limited_first_stage: bool = False) -> list[Ranker]:
    """Every ranker replayed: the baseline, then per depth the oracle and each model's rerank;
    then for each expansion the baseline, oracle, rerank-then-expand and
    expand-then-rerank. limited_first_stage adds the first stage at each depth as its vector
    search limit."""
    out = [Ranker("first_stage")]
    for d in depths:
        out.append(Ranker("oracle", depth=d))
        out.append(Ranker("bound", depth=d))
        out += [Ranker("first_stage", depth=d)] if limited_first_stage else []
        for m in models:
            out.append(Ranker("rerank", m, d))
    for e in expansions:
        out.append(Ranker("first_stage", expand=e))
        for d in depths:
            out.append(Ranker("oracle", depth=d, expand=e))
            out.append(Ranker("bound", depth=d, expand=e))
            out += [Ranker("first_stage", depth=d, expand=e)] if limited_first_stage else []
            for m in models:
                out += [Ranker("rerank", m, d, expand=e), Ranker("xrerank", m, d, expand=e)]
    return out
