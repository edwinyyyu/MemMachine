"""Rankers: how a query's first-stage head is reordered before reading.

  first_stage   the embedding order, unchanged (the no-reranker baseline; its vector search
                limit is whatever the reader budget reads). With a depth L ("first_stage@n20"),
                the first stage with vector search limit L: only its top L seeds are read, so it
                compares with reranking at the same vector search limit.
  rerank        the head's first `depth` chunks sorted by reranker score, and nothing after
                them: EventMemory reranks every seed it retrieves, so the depth is the vector
                search limit and reading ends with the reranked chunks
  fuse          as rerank, by w·z(reranker) + (1-w)·z(similarity), z-scored over the
                reranked chunks: keeps some first-stage signal
  rrf           as rerank, by reciprocal-rank fusion of the two orders (k = 60)
  oracle        as rerank, gold first: the chunks of the cheapest complete answer, then
                other gold chunks, then the rest of the pool. A reference for how much
                reordering the pool can buy, not a strict bound: it is optimal for "every
                gold" when that answer is in the pool, and near it for "first gold".

With context expansion (expand = e > 0, see expansion.py) the head's items are seeds and
each is read with its window:
  first_stage+x   seeds in first-stage order, expanded
  rerank+x        rerank the seeds themselves, then expand the order ("rerank, then expand")
  xrerank+x       rerank each seed's whole window, then read the windows ("expand, then
                  rerank"; what EventMemory's query does when expand_context > 0). In the
                  history simulation its token depths count seed tokens, like every ranker's,
                  so both scoring positions rerank the same seeds (the same vector search
                  limit); the flat simulation counts window tokens, since windows are scored.
  oracle+x        seeds whose window holds gold first (cheapest answer's seeds first)

A depth is a count of chunks ("n20") or of first-stage tokens ("t4096": the head chunks
within 4,096 tokens, at least one). Count depths are how systems are configured (EventMemory's
vector_search_limit is a count of seeds), and the history simulation uses only them (HISTORY_DEPTHS);
equal reranker input is compared through each configuration's measured reranker tokens. Token
depths remain for the flat datasets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

COUNT_DEPTHS = (5, 10, 20, 30, 50, 75, 100, 150, 200, 300)
TOKEN_DEPTHS = (1024, 2048, 4096, 8192, 16384)
FUSION_WEIGHTS = (0.5, 0.8)
RRF_K = 60


@dataclass(frozen=True)
class Depth:
    kind: str  # "n" or "t"
    value: int

    @property
    def label(self) -> str:
        return f"{self.kind}{self.value}"

    def chunks(self, head_tokens: np.ndarray) -> int:
        """How many head chunks this depth reranks, capped by the head."""
        if self.kind == "n":
            return min(self.value, len(head_tokens))
        return max(1, int(np.searchsorted(np.cumsum(head_tokens), self.value, "right")))


DEPTHS = tuple(Depth("n", v) for v in COUNT_DEPTHS) + tuple(Depth("t", v) for v in TOKEN_DEPTHS)
HISTORY_DEPTHS = tuple(Depth("n", v) for v in COUNT_DEPTHS)


@dataclass(frozen=True)
class Ranker:
    kind: str  # first_stage | rerank | fuse | rrf | oracle
    model: str | None = None
    depth: Depth | None = None
    weight: float | None = None
    expand: int = 0

    @property
    def id(self) -> str:
        tail = f"+x{self.expand}" if self.expand else ""
        if self.kind == "first_stage":
            return "first_stage" + (f"@{self.depth.label}" if self.depth else "") + tail
        head = {"rerank": f"rerank:{self.model}", "xrerank": f"xrerank:{self.model}", "fuse": f"fuse{self.weight}:{self.model}",
                "rrf": f"rrf:{self.model}", "oracle": "oracle"}[self.kind]
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
        elif self.kind == "fuse":
            key = -(self.weight * zscore(logits[:n]) + (1 - self.weight) * zscore(sims[:n].astype(np.float64)))
        elif self.kind == "rrf":
            rank = np.empty(n)
            rank[np.argsort(-logits[:n], kind="stable")] = np.arange(n)
            key = -(1 / (RRF_K + 1 + rank) + 1 / (RRF_K + 1 + np.arange(n)))
        else:
            raise ValueError(self.kind)
        return np.argsort(key, kind="stable")


def zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def for_models(models: list[str], expansions: tuple[int, ...] = (), depths: tuple[Depth, ...] = DEPTHS,
               limited_first_stage: bool = False) -> list[Ranker]:
    """Every ranker replayed: the baseline, then per depth the oracle and each model's rerank,
    fuse and rrf; then for each expansion the baseline, oracle, rerank-then-expand and
    expand-then-rerank. limited_first_stage adds the first stage at each depth as its vector
    search limit."""
    out = [Ranker("first_stage")]
    for d in depths:
        out.append(Ranker("oracle", depth=d))
        out += [Ranker("first_stage", depth=d)] if limited_first_stage else []
        for m in models:
            out += [Ranker("rerank", m, d), *(Ranker("fuse", m, d, w) for w in FUSION_WEIGHTS), Ranker("rrf", m, d)]
    for e in expansions:
        out.append(Ranker("first_stage", expand=e))
        for d in depths:
            out.append(Ranker("oracle", depth=d, expand=e))
            out += [Ranker("first_stage", depth=d, expand=e)] if limited_first_stage else []
            for m in models:
                out += [Ranker("rerank", m, d, expand=e), Ranker("xrerank", m, d, expand=e)]
    return out
