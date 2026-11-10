"""Running a grid stage: prepare inputs, rerank (under a total spend cap), simulate, report.

Every step skips what is already cached, so a stage can be rerun after an interruption or
extended later, and only new work is done.

Every dataset runs through MemMachine's EventMemory (partitions.py, simulation.py).
Simulation always uses every reranker (rerankers.RERANKERS) and every expansion that has
scores, whichever stage scored them: a configuration is kept only when every query of the
sample was scored deep enough. So running one stage never hides another stage's results.
"""

from __future__ import annotations

from collections import defaultdict

from rbench import config, depth, simulation
from rbench import partitions as pt
from rbench.embedding import DEFAULT as EMBEDDER
from rbench.grid import Run
from rbench.report import figures, readings, tables
from rbench.report.data import load
from rbench.rerankers import RERANKERS, pair_ids, score
from rbench.samples import sample

_PREPARED: set[tuple] = set()  # (embedder, dataset, scope, chunking, sample) done in this process


def prepare(run: Run, embedder: str = EMBEDDER) -> None:
    """Encode the partitions the run's queries search (each embedded once; the stores are kept),
    then, for a run without a set pool, measure the depth its pool needs (depth.py, free)."""
    for c in run.chunkings:
        if (embedder, run.dataset, run.scope, c, run.sample) in _PREPARED:
            continue  # the same stage's other scoring position
        _PREPARED.add((embedder, run.dataset, run.scope, c, run.sample))
        pt.ingest(embedder, run.dataset, c, sample(run.dataset, run.sample), run.scope)
        print(f"{run.dataset} {c} {run.sample}: pool {depth.pool(run, c, embedder)} seeds", flush=True)


def rerank(run: Run, model: str, max_usd: float | None, embedder: str = EMBEDDER) -> None:
    spec = RERANKERS[model]
    for c in run.chunkings:
        work = simulation.scoring_work(embedder, run.dataset, c, run.sample, model, run.expand,
                                       depth.pool(run, c, embedder), run.scope, run.variant)
        pairs = sum(len(items) for _, items in work)
        tokens = sum(len(r) for q, items in work for r in pair_ids(q, [d for _, d in items], spec.template))
        label = f"{run.dataset}/{run.scope}/{run.variant}/{c}{f'+x{run.expand}' if run.expand else ''}"
        print(f"{model} {label}: {pairs} pairs to score, {tokens / 1e6:.2f}M tokens, "
              f"${tokens / 1e6 * (spec.usd_per_mtok or 0.0):.2f}", flush=True)
        if pairs:
            score(model, work, label, max_usd)


def simulate_and_report(runs: list[Run], embedder: str = EMBEDDER) -> list[str]:
    """Simulate every (dataset, scope, variant, sample, chunking) the runs touch, with all
    rerankers and the replayed expansions plus any whose windows a run scored, then write each
    (dataset, scope, variant, sample) report."""
    jobs: dict[tuple, set[int]] = defaultdict(lambda: set(config.EVAL.expansions))
    pools: dict[tuple, set[int]] = defaultdict(set)  # the runs' pools; the first stage searches as deep as the deepest
    for run in runs:
        for c in run.chunkings:
            key = (run.dataset, run.scope, run.variant, run.sample, c)
            jobs[key] |= {run.expand} - {0}
            pools[key].add(depth.pool(run, c, embedder))
            pools[key] |= depth.measured_pools(embedder, run.dataset, c, run.sample)  # other stages' pools too
    for key, expansions in sorted(jobs.items()):
        dataset, scope, variant, sample_name, c = key
        simulation.run(embedder, dataset, c, sample_name, list(RERANKERS), tuple(sorted(expansions)), scope, variant,
                       max(pools[key]), tuple(sorted(pools[key])))
    written = []
    for dataset, scope, variant, sample_name in sorted({k[:4] for k in jobs}):
        results = load(embedder, dataset, scope, variant, sample_name)
        tables.clear(results)
        written += [tables.write(results, e) for e in readings.available(results.df)] + [figures.write(results)]
    if written:
        written.append(tables.index())
    return written
