"""Best settings: which value of each factor works best, one factor at a time.

The user's request (2026-11-05 and 2026-11-09): vary overfetch, top-k, chunk size, the chunker
and what the reader and the reranker see, and report which setting works best for each. The
simulation already replays all of them; this module chooses.

One factor at a time, every other held at the core setting (frontier.setting: EventMemory's
default segment length, expand_context 0, the reranker scoring the seeds, at its deepest
pool). Per model, reading and reader budget B, for each value of a factor: the reranked
coverage at B and the first stage's under the same value. The best value is the one with the
highest reranked coverage; the cheapest equivalent is the cheapest value (FACTORS' order:
fewer seeds, fewer segments, fewer reranker tokens) whose coverage reaches the best's 95%
lower bound. The bound comes from bootstrap resamples of the questions' clusters, each
re-choosing the best value (choices are made and measured on the same questions, with no dev
split, so the best leans optimistic; the resampling widens the interval for it).

Factors (values that were not simulated are skipped):
  L             vector search limit (seeds retrieved and reranked), up to the pool
  k             seeds kept after reranking, at the deepest pool, within the budget too (event
                readings: they need the seed count at which each event happens)
  m             segments read (EventMemory's max_num_segments), at the deepest pool, within the
                budget too (events)
  segment       the segment length (seg<N>), each at its own deepest pool
  chunker       EventMemory's TextSegmenter (seg<N>) or the simple chunker (cut<N>), same N
  reader sees   the seeds alone, or their windows at each expand_context (rerank, then expand)
  reranker sees the seeds (rerank, then expand) or the windows (expand, then rerank), at
                expand_context 3, each at its own pool
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rbench.report import frontier
from rbench.report import readings as rd
from rbench.report.data import Results
from rbench.report.stats import RESAMPLES

KEPT = (1, 2, 3, 5, 10, 20, 30, 50)
SEGMENTS = (5, 10, 20, 50, 100, 200)


@dataclass(frozen=True)
class Value:
    """One value of a factor: its label, each side's per-question value at the budget, and
    the reranker's mean tokens (its cost)."""

    label: str
    reranked: np.ndarray
    first: np.ndarray | None
    tokens: float
    config: tuple[str, str] | None = None  # (chunking, ranker) of the reranked side


@dataclass(frozen=True)
class Choice:
    factor: str
    values: tuple[Value, ...]
    best: int
    cheapest: int
    lower: float  # the best's 95% lower bound (re-chosen per resample)

    def coverage(self, i: int) -> float:
        return float(self.values[i].reranked.mean())


def _hits(rows, reading: str, budget: float) -> np.ndarray:
    return rd.hit(rd.costs(rows, reading), budget)


def _count_hits(rows, reading: str, column: str, limit: int, budget: float) -> np.ndarray | None:
    """Per question: 1 if the event happens within `limit` seeds (column "<reading>_ctx") or
    segments ("<reading>_k") and within the budget (reading stops at whichever ends first);
    events only."""
    name = f"{reading}_{column}"
    if rd.is_share(reading) or name not in rows:
        return None
    return ((rows[name].to_numpy() <= limit) & (rows[reading].to_numpy() <= budget)).astype(float)


def factors(r: Results, model: str, reading: str, budget: float) -> dict[str, list[Value]]:
    """Every factor's values at one budget, for one model (empty where nothing was simulated)."""
    core, _ = frontier.setting(r)
    out: dict[str, list[Value]] = {}

    def value(label, c, rerank_name, first_name) -> Value | None:
        rows, base = r.rows(c, rerank_name), r.rows(c, first_name)
        if rows.empty:
            return None
        return Value(label, _hits(rows, reading, budget), None if base.empty else _hits(base, reading, budget),
                     float(rows.rr_tokens.mean()), (c, rerank_name))

    out["L"] = [v for d in r.depths_at(core, model, "rerank")
                if (v := value(f"L {d[1:]}", core, f"rerank:{model}@{d}", f"first_stage@{d}"))]
    pool = r.pool_depth(core, model, "rerank")
    if pool:
        rows, base = r.rows(core, f"rerank:{model}@{pool}"), r.rows(core, f"first_stage@{pool}")
        for factor, column, grid in (("k", "ctx", KEPT), ("m", "k", SEGMENTS)):
            values = []
            for n in grid:
                hits = _count_hits(rows, reading, column, n, budget)
                if hits is not None and (factor == "m" or n <= int(pool[1:])):  # k cannot exceed the pool
                    values.append(Value(f"{factor} {n}", hits, _count_hits(base, reading, column, n, budget),
                                        float(rows.rr_tokens.mean()), (core, f"rerank:{model}@{pool}")))
            out[factor] = values
    out["segment"] = [v for c in r.chunkings if c.startswith("seg")
                      if (d := r.pool_depth(c, model, "rerank"))
                      if (v := value(c, c, f"rerank:{model}@{d}", f"first_stage@{d}"))]
    n = core[3:]
    out["chunker"] = [v for c in (f"seg{n}", f"cut{n}") if c in r.chunkings
                      if (d := r.pool_depth(c, model, "rerank"))
                      if (v := value(c, c, f"rerank:{model}@{d}", f"first_stage@{d}"))]
    if pool:
        out["reader sees"] = [v for e in (0, *r.expansions)
                              if (v := value("seeds" if not e else f"windows x{e}", core,
                                             f"rerank:{model}@{pool}" + (f"+x{e}" if e else ""),
                                             f"first_stage@{pool}" + (f"+x{e}" if e else "")))]
    windows = r.pool_depth(core, model, "xrerank")
    if pool and windows and 3 in r.expansions:
        out["reranker sees"] = [v for v in (
            value("seeds, then x3", core, f"rerank:{model}@{pool}+x3", f"first_stage@{pool}+x3"),
            value("windows x3", core, f"xrerank:{model}@{windows}+x3", f"first_stage@{windows}+x3")) if v]
    return {f: v for f, v in out.items() if v}


def choose(r: Results, factor: str, values: list[Value]) -> Choice:
    """The best value and the cheapest within its 95% lower bound (re-chosen per resample)."""
    hits = np.stack([v.reranked for v in values])  # [value, question]
    best = int(np.argmax(hits.mean(1)))
    clusters = frontier._clusters(r)
    labels, inverse = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inverse == k) for k in range(len(labels))]
    rng = np.random.default_rng(0)
    draws = np.empty(RESAMPLES)
    for i in range(RESAMPLES):
        index = np.concatenate([members[k] for k in rng.integers(0, len(labels), len(labels))])
        draws[i] = hits[:, index].mean(1).max()  # each resample re-chooses its best value
    lower = float(np.percentile(draws, 2.5))
    cheapest = next(i for i in range(len(values)) if hits[i].mean() >= lower)  # values come cheapest first
    if factor in ("segment", "chunker", "reader sees", "reranker sees"):  # cheapest: fewest reranker tokens
        ok = [i for i in range(len(values)) if hits[i].mean() >= lower]
        cheapest = min(ok, key=lambda i: values[i].tokens)
    return Choice(factor, tuple(values), best, cheapest, lower)


def overall(r: Results, model: str, reading: str, budget: float) -> tuple[tuple[str, str], float, tuple[str, str], float]:
    """The best simulated combination at a budget, reranked and first stage alone: ((chunking,
    ranker), coverage) for each side."""
    d = r.select(expand=None)
    best_r, best_f = (None, -1.0), (None, -1.0)
    for (c, name), rows in d.groupby(["chunking", "ranker"]):
        kind = rows.kind.iloc[0]
        if kind in ("rerank", "xrerank") and rows.model.iloc[0] == model:
            v = float(_hits(rows, reading, budget).mean())
            best_r = max(best_r, ((c, name), v), key=lambda x: x[1])
        elif kind == "first_stage":
            v = float(_hits(rows, reading, budget).mean())
            best_f = max(best_f, ((c, name), v), key=lambda x: x[1])
    return best_r[0], best_r[1], best_f[0], best_f[1]


def limit_choice(r: Results, model: str, chunking: str, reading: str, budget: float, expand: int = 0) -> Choice | None:
    """The vector search limit factor at one chunking and expansion (rerank_cost.png marks it)."""
    tail = f"+x{expand}" if expand else ""
    values = []
    for d in r.depths_at(chunking, model, "rerank"):
        rows, base = r.rows(chunking, f"rerank:{model}@{d}{tail}"), r.rows(chunking, f"first_stage@{d}{tail}")
        if not rows.empty:
            values.append(Value(f"L {d[1:]}", _hits(rows, reading, budget),
                                None if base.empty else _hits(base, reading, budget), float(rows.rr_tokens.mean()),
                                (chunking, f"rerank:{model}@{d}{tail}")))
    return choose(r, "L", values) if values else None
