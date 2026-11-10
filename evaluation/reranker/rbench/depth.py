"""How deep the vector search must go: the pool depth D, measured by coverage.

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
the quantiles, and D at each target in TARGETS.
"""

from __future__ import annotations

import asyncio
import json

import numpy as np

from rbench import partitions as pt
from rbench.datasets import info
from rbench.memory import Memory
from rbench.paths import RESULTS
from rbench.samples import sample
from rbench.schema import queries

TARGETS = (0.90, 0.95, 0.99)
DEPTH = RESULTS / "depth"


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


async def _measure(embedder: str, dataset: str, chunking: str, sample_name: str, scope: str, variant: str) -> dict:
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
                result = await partition.query(q["variants"][variant], vector_search_limit=len(seg), expand_context=0,
                                               format_options=fmt.rerank)
                order = [index[str(c.seed_segment_uuid)] for c in result.scored_segment_contexts]
                every, first = needed(order, pt.gold(pt.gold_spans(dataset, q, p), p, seg))
                out[q["qid"]] = {"every": every, "first": first, "segments": len(seg)}
    return out


def measure(embedder: str, dataset: str, chunking: str, sample_name: str, scope: str | None = None,
            variant: str | None = None) -> dict:
    """Measure and save the depths a sample's queries need; returns the summary."""
    scope, variant = pt.scope_of(dataset, scope), variant or info(dataset).default_variant
    per = asyncio.run(_measure(embedder, dataset, chunking, sample_name, scope, variant))
    summary = {"embedder": embedder, "dataset": dataset, "chunking": chunking, "sample": sample_name,
               "queries": len(per), "segments_median": float(np.median([v["segments"] for v in per.values()]))}
    for reading in ("every", "first"):
        d = np.sort([v[reading] for v in per.values()])
        summary[reading] = {
            "quantiles": {f"p{int(100 * t)}": _quantile(d, t) for t in (0.5, *TARGETS)} | {"max": float(d.max())},
            "D": {f"{t:.2f}": _quantile(d, t) for t in TARGETS},
            "unreachable": int(np.isinf(d).sum()),
        }
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


POOL_COVERAGE = 0.95  # the pool holds all of a question's evidence for at least this share of questions


def pool(run, chunking: str, embedder: str, measure_if_missing: bool = True) -> int | None:
    """A run's pool (vector search limit) at a chunking: its own pool_seeds, else D at
    POOL_COVERAGE of "every" measured on the run's sample (measured now if the partitions are
    encoded and measure_if_missing; None when it cannot be known yet)."""
    if run.pool_seeds:
        return run.pool_seeds
    if not path(embedder, run.dataset, chunking, run.sample).exists():
        if not measure_if_missing:
            return None
        measure(embedder, run.dataset, chunking, run.sample, run.scope, run.variant)
    return int(load(embedder, run.dataset, chunking, run.sample)["every"]["D"][f"{POOL_COVERAGE:.2f}"])
