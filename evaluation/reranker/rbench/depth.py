"""How deep the vector search must go: the pool depth, measured by budget fill.

The pool rule (the user's, 2026-11-09): the reranker's pool must be deep enough that reading
it does not run out before the largest reader budget evaluated (the chart axis's end,
config.EVAL.token_top), so curves keep rising across the axis instead of flattening because
the pool is used up; cost decides that axis end. The fill depth at budget B is the number of
contexts (seeds, or their windows) whose rendered reading cost reaches B: the pool's cost does
not depend on the order it is read in, so it is measured in first-stage order with the real
Reader. A pool fills B for a query when its fill depth is at most the pool; the pool is the
fill depth at POOL_COVERAGE of the sample's queries, per segment length, chunker and scoring
position (seeds, or windows: these fill a budget several times faster).

Also measured and reported, not used as the pool rule: the coverage depth.

For each query, EventMemory.query (no reranker, expand_context 0) ranks every segment of the
partition it searches. The depth a query needs is the rank of its deepest needed seed:
  every   all evidence among the top D seeds: the deepest segment of the cheapest-to-reach
          answer (for LongMemEval, the deepest segment of any gold turn: all are needed)
  first   some evidence in full: the shallowest answer span's deepest segment
D at a coverage target is the smallest depth at which that share of queries has its
evidence among the seeds. Seeds only: with windows (expand_context > 0) a window can bring in
evidence that is not a seed, so a pool of D seeds covers at least as much with windows.

The search is free once partitions are embedded. The benchmark reranks the whole pool (the
vector search limit is the pool, rerankers.py), and every limit up to D replays for free, so
where returns diminish is read off the replay, not chosen before it.

Output: results/depth/<embedder>/<dataset>__<chunking>__<sample>.json with per-query depths,
the coverage quantiles and D at each target in TARGETS, and the fill depths per expand_context
(FILL_EXPANSIONS) and budget (FILL_BUDGETS and config.EVAL.token_top).
"""

from __future__ import annotations

import asyncio
import json

import numpy as np

from rbench import config
from rbench import partitions as pt
from rbench.datasets import info
from rbench.memory import Memory
from rbench.paths import RESULTS
from rbench.reading import Reader
from rbench.samples import sample
from rbench.schema import queries

TARGETS = (0.90, 0.95, 0.99)
POOL_COVERAGE = 0.95  # the pool fills the axis end for at least this share of queries
FILL_BUDGETS = (4000, 8000, 12000, 16000)  # candidate axis ends priced before a paid run
FILL_EXPANSIONS = (0, 3)  # scoring positions: the seeds, and the windows grid.py scores
DEPTH = RESULTS / "depth"


def budgets() -> tuple[int, ...]:
    return tuple(sorted(set(FILL_BUDGETS) | {config.EVAL.token_top}))


def fill_costs(contexts: list[list[int]], seg: pt.Segments, top: float) -> list[int]:
    """Rendered cost of reading the first L contexts in full, for L = 1, 2, ..., until it
    reaches `top` (or the contexts run out)."""
    reader, out = Reader(contexts, seg.event, seg.inner, seg.run, lazy=True), []
    for k, context in enumerate(contexts):
        reader.open(k)
        for s in context:
            if s not in reader.admitted:
                reader.admit(s, k)
        out.append(reader.cost)
        if out[-1] >= top:
            break
    return out


def fill_depth(costs: list[int], budget: float) -> int:
    """Contexts read before the cost reaches `budget` (all of them when it never does)."""
    return next((k + 1 for k, c in enumerate(costs) if c >= budget), len(costs))


def path(embedder: str, dataset: str, chunking: str, sample_name: str):
    return DEPTH / embedder / f"{dataset}__{chunking}__{sample_name}.json"


def needed(order: list[int], g) -> tuple[float, float]:
    """(every, first): the depth (1-based seed rank) at which all of some answer's spans, or
    some span in full, are among the seeds; inf when not reachable."""
    rank = {s: i + 1 for i, s in enumerate(order)}

    def deepest(span) -> float:
        return max((rank.get(c, np.inf) for c, _ in span), default=np.inf)

    every = min((max((deepest(s) for s in alt), default=np.inf) for alt in g.alternatives), default=np.inf)
    first = min((deepest(s) for alt in g.alternatives for s in alt), default=np.inf)
    return float(every), float(first)


async def _measure(embedder: str, dataset: str, chunking: str, sample_name: str, scope: str, variant: str,
                   expansions: tuple[int, ...]) -> dict:
    rows, fmt = queries(dataset), pt.formats(dataset)
    out = {}
    for k, members in pt.groups(dataset, sample(dataset, sample_name), scope).items():
        p = pt.part(dataset, rows[members[0]], scope)
        async with Memory(embedder, dataset, chunking) as m:
            seg = await pt.table(m, dataset, p)
            partition = await (await m.event_memory()).open_or_create_partition(k)
            index = {u: i for i, u in enumerate(seg.uuid)}
            for qi in members:
                q = rows[qi]
                query = q["variants"][variant]
                result = await partition.query(query, vector_search_limit=len(seg), expand_context=0,
                                               format_options=fmt.rerank)
                order = [index[str(c.seed_segment_uuid)] for c in result.scored_segment_contexts]
                every, first = needed(order, pt.gold(pt.gold_spans(dataset, q, p), p, seg))
                top = max(budgets())
                seeds = fill_costs([[s] for s in order], seg, top)
                fill = {"0": {str(b): fill_depth(seeds, b) for b in budgets()}}
                for e in [e for e in expansions if e]:  # windows fill a budget with no more contexts than seeds
                    windowed = await partition.query(query, vector_search_limit=len(seeds), expand_context=e,
                                                     format_options=fmt.rerank)
                    contexts = [[index[str(x.uuid)] for x in c.segments] for c in windowed.scored_segment_contexts]
                    costs = fill_costs(contexts, seg, top)
                    fill[str(e)] = {str(b): fill_depth(costs, b) for b in budgets()}
                out[q["qid"]] = {"every": every, "first": first, "segments": len(seg), "fill": fill}
    return out


def measure(embedder: str, dataset: str, chunking: str, sample_name: str, scope: str | None = None,
            variant: str | None = None, expansions: tuple[int, ...] = FILL_EXPANSIONS) -> dict:
    """Measure and save the depths a sample's queries need; returns the summary."""
    scope, variant = pt.scope_of(dataset, scope), variant or info(dataset).default_variant
    expansions = tuple(sorted({0, *expansions}))
    per = asyncio.run(_measure(embedder, dataset, chunking, sample_name, scope, variant, expansions))
    summary = {"embedder": embedder, "dataset": dataset, "chunking": chunking, "sample": sample_name,
               "queries": len(per), "segments_median": float(np.median([v["segments"] for v in per.values()]))}
    for reading in ("every", "first"):
        d = np.sort([v[reading] for v in per.values()])
        summary[reading] = {
            "quantiles": {f"p{int(100 * t)}": _quantile(d, t) for t in (0.5, *TARGETS)} | {"max": float(d.max())},
            "D": {f"{t:.2f}": _quantile(d, t) for t in TARGETS},
            "unreachable": int(np.isinf(d).sum()),
        }
    summary["fill"] = {}  # expand -> budget -> {pool: fill depth at POOL_COVERAGE, median, max}
    for e in expansions:
        summary["fill"][str(e)] = {}
        for b in budgets():
            d = np.sort([v["fill"][str(e)][str(b)] for v in per.values()])
            summary["fill"][str(e)][str(b)] = {"pool": _quantile(d, POOL_COVERAGE), "median": float(np.median(d)),
                                               "max": float(d.max())}
    target = path(embedder, dataset, chunking, sample_name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"summary": summary, "per_query": per}, indent=1) + "\n")
    return summary


def _quantile(sorted_depths: np.ndarray, target: float) -> float:
    """The smallest depth at which at least `target` of the queries are covered (an order
    statistic, not interpolated)."""
    n = len(sorted_depths)
    return float(sorted_depths[min(n - 1, int(np.ceil(target * n)) - 1)])


def load(embedder: str, dataset: str, chunking: str, sample_name: str) -> dict:
    p = path(embedder, dataset, chunking, sample_name)
    if not p.exists():
        raise SystemExit(f"no measured depth for {dataset} {chunking} {sample_name} ({embedder}): run "
                         f"`python -m rbench depth {dataset} {chunking} {sample_name}` first")
    return json.loads(p.read_text())["summary"]


def _known(embedder: str, dataset: str, chunking: str, sample_name: str, expand: int) -> bool:
    if not path(embedder, dataset, chunking, sample_name).exists():
        return False
    fill = load(embedder, dataset, chunking, sample_name).get("fill", {})
    return str(expand) in fill and str(config.EVAL.token_top) in fill[str(expand)]


def _fill(embedder: str, dataset: str, chunking: str, sample_name: str, expand: int) -> int:
    return int(load(embedder, dataset, chunking, sample_name)["fill"][str(expand)][str(config.EVAL.token_top)]["pool"])


def pool(run, chunking: str, embedder: str, measure_if_missing: bool = True) -> int | None:
    """A run's pool (vector search limit) at a chunking: its own pool_seeds, else the fill depth
    of its scoring position (seeds, or windows at run.expand) at config.EVAL.token_top, at
    POOL_COVERAGE of the sample's queries (measured now if the partitions are encoded and
    measure_if_missing; None when it cannot be known yet)."""
    if run.pool_seeds:
        return run.pool_seeds
    if not _known(embedder, run.dataset, chunking, run.sample, run.expand):
        if not measure_if_missing:
            return None
        measure(embedder, run.dataset, chunking, run.sample, run.scope, run.variant, (*FILL_EXPANSIONS, run.expand))
    return _fill(embedder, run.dataset, chunking, run.sample, run.expand)


def estimated_pool(run, chunking: str, embedder: str) -> tuple[int, str] | None:
    """For pricing before a run's sample is embedded: the fill depth measured on the largest
    smaller sample of the same dataset and chunking that is (any embedder), and which one it
    is. None when there is none."""
    from rbench.samples import ORDER

    for sample_name in reversed(ORDER[: ORDER.index(run.sample)]):
        for e in (embedder, *sorted(p.name for p in DEPTH.glob("*") if p.name != embedder)):
            if _known(e, run.dataset, chunking, sample_name, run.expand):
                n = load(e, run.dataset, chunking, sample_name)["queries"]
                return _fill(e, run.dataset, chunking, sample_name, run.expand), f"{sample_name} ({n} queries, {e})"
    return None
