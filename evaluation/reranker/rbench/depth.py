"""How far a pool reaches: the pools are fixed, and this measures what they hold.

The pools are fixed (config.EVAL.pool_seeds 200, pool_windows 100: the user's decision,
2026-11-10, "just do L100 and L200", replacing a measured pool that filled the axis for 95% of
questions, a margin that was not the user's). What is measured and reported beside them: the
fill depth at budget B, the number of contexts (seeds, or their windows) whose rendered
reading cost reaches B (the pool's cost does not depend on the order it is read in, so it is
measured in first-stage order with the real Reader; a pool fills B for a query when its fill
depth is at most the pool), so a report can say for how many questions reading the pool
reaches the axis end (config.EVAL.token_top); the fill depth at POOL_COVERAGE of the
questions; the coverage depth; and each pool's evidence coverage (the share of queries whose
evidence is all in the pool: the ceiling no reading of that pool can pass, whatever the
budget). At shorter segments a pool holds less text, so reading it can end before the axis
end and the curves level off earlier: accepted (the user, 2026-11-09: "limit to 16k tokens or
less"); a stage can set its own pool (grid.Run.pool_seeds).

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
from rbench.dbs import replace_atomically
from rbench.memory import QUERY_SEEDS, Memory
from rbench.paths import RESULTS
from rbench.reading import Reader
from rbench.samples import sample
from rbench.schema import queries
from rbench.text import READER

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


def windows_hold(contexts: list[list[int]], g) -> float:
    """The number of contexts (windows) whose union first holds every segment of some answer
    (inf if these contexts never do)."""
    answers = [{c for span in alt for c, _ in span} for alt in g.alternatives]
    union: set[int] = set()
    for k, context in enumerate(contexts):
        union.update(context)
        if any(a and a <= union for a in answers):
            return float(k + 1)
    return float("inf")


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


async def measure_partition(m: Memory, dataset: str, k: str, p, members: list[int], variant: str,
                            expansions: tuple[int, ...], progress=None) -> dict:
    """Each of a partition's queries' depths (keyed by qid), from its open Memory: what
    _measure records per query. Engine calls run in order, one at a time."""
    rows, fmt = queries(dataset), pt.formats(dataset)
    out = {}
    seg = await pt.table(m, dataset, p)
    partition = await (await m.event_memory()).open_or_create_partition(k)
    index = {u: i for i, u in enumerate(seg.uuid)}
    for qi in members:
        q = rows[qi]
        query = q["variants"][variant]
        # every segment ranked, up to the SQLite variable limit (memory.QUERY_SEEDS): deeper evidence counts as unreached
        result = await partition.query(query, vector_search_limit=min(len(seg), QUERY_SEEDS), expand_context=0,
                                       format_options=fmt.rerank)
        order = [index[str(c.seed_segment_uuid)] for c in result.scored_segment_contexts]
        g = pt.gold(pt.gold_spans(dataset, q, p), p, seg)
        every, first = needed(order, g)
        top = max(budgets())
        seeds = fill_costs([[s] for s in order], seg, top)
        fill = {"0": {str(b): fill_depth(seeds, b) for b in budgets()}}
        holds = {"0": every}  # the pool depth at which all of some answer's evidence is in it
        for e in [e for e in expansions if e]:  # windows fill a budget with no more contexts than seeds
            windowed = await partition.query(query, vector_search_limit=len(seeds), expand_context=e,
                                             format_options=fmt.rerank)
            contexts = [[index[str(x.uuid)] for x in c.segments] for c in windowed.scored_segment_contexts]
            costs = fill_costs(contexts, seg, top)
            fill[str(e)] = {str(b): fill_depth(costs, b) for b in budgets()}
            holds[str(e)] = windows_hold(contexts, g)
        out[q["qid"]] = {"every": every, "first": first, "segments": len(seg), "fill": fill, "holds": holds}
        if progress is not None:
            progress.step()
    return out


async def _measure(embedder: str, dataset: str, chunking: str, sample_name: str, scope: str, variant: str,
                   expansions: tuple[int, ...]) -> dict:
    from rbench.progress import Progress

    rows = queries(dataset)
    out = {}
    keyed = pt.groups(dataset, sample(dataset, sample_name), scope)
    progress = Progress(f"{dataset} {chunking} depth", sum(len(v) for v in keyed.values()), "queries")
    for k, members in keyed.items():
        p = pt.part(dataset, rows[members[0]], scope)
        async with Memory(embedder, dataset, chunking) as m:
            out.update(await measure_partition(m, dataset, k, p, members, variant, expansions, progress))
    return out


def measure(embedder: str, dataset: str, chunking: str, sample_name: str, scope: str | None = None,
            variant: str | None = None, expansions: tuple[int, ...] = FILL_EXPANSIONS) -> dict:
    """Measure and save the depths a sample's queries need; returns the summary."""
    scope, variant = pt.scope_of(dataset, scope), variant or info(dataset).default_variant
    expansions = tuple(sorted({0, *expansions}))
    per = asyncio.run(_measure(embedder, dataset, chunking, sample_name, scope, variant, expansions))
    return save(embedder, dataset, chunking, sample_name, per, expansions)


def save(embedder: str, dataset: str, chunking: str, sample_name: str, per: dict, expansions: tuple[int, ...]) -> dict:
    """Summarize and save per-query depths (measure's, or those an overlapped ingest measured
    partition by partition: pipeline.prepare_and_score); returns the summary."""
    expansions = tuple(sorted({0, *expansions}))
    summary = {"embedder": embedder, "dataset": dataset, "chunking": chunking, "sample": sample_name, "reader": READER,
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
            pool = _quantile(d, POOL_COVERAGE)
            # the ceiling: the share of queries whose evidence is all in the pool (no reading of
            # this pool can complete the rest, whatever the budget)
            holds = np.asarray([v["holds"][str(e)] for v in per.values()])
            summary["fill"][str(e)][str(b)] = {"pool": pool, "median": float(np.median(d)), "max": float(d.max()),
                                               "evidence_in_pool": float((holds <= pool).mean())}
    target = path(embedder, dataset, chunking, sample_name)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps({"summary": summary, "per_query": per}, indent=1) + "\n"
    replace_atomically(target, lambda temporary: temporary.write_text(text))
    return summary


def _quantile(sorted_depths: np.ndarray, target: float) -> float:
    """The smallest depth at which at least `target` of the queries are covered (an order
    statistic, not interpolated)."""
    n = len(sorted_depths)
    return float(sorted_depths[min(n - 1, int(np.ceil(target * n)) - 1)])


def _read(embedder: str, dataset: str, chunking: str, sample_name: str) -> dict | None:
    """The saved depths, None when missing, unreadable or counted in another tokenizer's reader
    tokens (measured again: fill depths are in reader tokens)."""
    p = path(embedder, dataset, chunking, sample_name)
    try:
        saved = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return saved if saved.get("summary", {}).get("reader") == READER else None


def load(embedder: str, dataset: str, chunking: str, sample_name: str) -> dict:
    saved = _read(embedder, dataset, chunking, sample_name)
    if saved is None:
        raise SystemExit(f"no measured depth for {dataset} {chunking} {sample_name} ({embedder}): run "
                         f"`python -m rbench depth {dataset} {chunking} {sample_name}` first")
    return saved["summary"]


def _known(embedder: str, dataset: str, chunking: str, sample_name: str, expand: int) -> bool:
    saved = _read(embedder, dataset, chunking, sample_name)
    if saved is None:
        return False
    fill = saved["summary"].get("fill", {})
    return str(expand) in fill and str(config.EVAL.token_top) in fill[str(expand)]


def measured_fill(embedder: str, dataset: str, chunking: str, sample_name: str, expand: int) -> int:
    """The fill depth at POOL_COVERAGE of the questions (how deep a pool would have to go to
    fill the axis for that share): reported, not the pool."""
    return int(load(embedder, dataset, chunking, sample_name)["fill"][str(expand)][str(config.EVAL.token_top)]["pool"])


def pool(run, chunking: str | None = None, embedder: str | None = None, measure_if_missing: bool = True) -> int:
    """A run's pool (vector search limit): its own pool_seeds, else the configured one of its
    scoring position (config.EVAL.pool_seeds for the seeds, pool_windows for windows at
    run.expand). Fixed in advance, so known before anything is embedded."""
    if run.pool_seeds:
        return run.pool_seeds
    return config.EVAL.pool_windows if run.expand else config.EVAL.pool_seeds


def pool_shares(embedder: str, dataset: str, chunking: str, sample_name: str, expand: int,
                pool_: int) -> tuple[float, float] | None:
    """For a pool of pool_ contexts: the share of the sample's questions for which reading it
    reaches the axis end (config.EVAL.token_top), and the share whose evidence it holds in full
    (the ceiling no reading of it can pass); None before the depths are measured."""
    if not _known(embedder, dataset, chunking, sample_name, expand):
        return None
    per = _read(embedder, dataset, chunking, sample_name)["per_query"].values()
    top = str(config.EVAL.token_top)
    fills = np.mean([v["fill"][str(expand)][top] <= pool_ for v in per])
    holds = np.mean([v["holds"][str(expand)] <= pool_ for v in per])
    return float(fills), float(holds)


def shares_for(run, chunking: str, embedder: str) -> tuple[float, float] | None:
    """pool_shares for a run's pool, from its own sample or the largest smaller one measured
    (any embedder, its own first)."""
    from rbench.samples import ORDER

    for sample_name in reversed(ORDER[: ORDER.index(run.sample) + 1]):
        for e in (embedder, *sorted(p.name for p in DEPTH.glob("*") if p.name != embedder)):
            got = pool_shares(e, run.dataset, chunking, sample_name, run.expand, pool(run))
            if got is not None:
                return got
    return None
