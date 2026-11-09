"""Running a grid stage: prepare inputs, rerank (under a total spend cap), simulate, report.

Every step skips what is already cached, so a stage can be rerun after an interruption or
extended later, and only new work is done.

Simulation always uses every reranker (rerankers.RERANKERS) and every expansion that has
scores, whichever stage scored them: a configuration is kept only when every query of
the sample was scored deep enough (simulate.py). So running one stage never hides another
stage's results.
"""

from __future__ import annotations

from collections import defaultdict

from rbench import chunking, retrieval, simulate, simulate_histories, units
from rbench import histories as hs
from rbench.datasets import info
from rbench.embedding import DEFAULT as EMBEDDER
from rbench.grid import Run
from rbench.report import figures, tables
from rbench.report.data import load
from rbench.rerankers import INSTRUCTION, RERANKERS, ScoreCache, pair_ids, score


def _size(chunking_name: str) -> int:
    return int(chunking_name.removeprefix("seg"))


_PREPARED: set[tuple] = set()  # (embedder, dataset, chunking, sample) done in this process


def prepare(run: Run, embedder: str = EMBEDDER) -> None:
    if info(run.dataset).partitioned:  # each history encoded into its EventMemory partition once (kept)
        from rbench.samples import sample
        from rbench.schema import queries

        rows = queries(run.dataset)
        for c in run.chunkings:
            if (embedder, run.dataset, c, run.sample) in _PREPARED:
                continue  # the same stage's other scoring position
            _PREPARED.add((embedder, run.dataset, c, run.sample))
            hs.ingest(embedder, run.dataset, _size(c), [rows[i]["qid"] for i in sample(run.dataset, run.sample)])
        return
    for c in run.chunkings:
        chunking.build(run.dataset, c)
        retrieval.build(embedder, run.dataset, run.scope, c, run.variant)


def missing(run: Run, model: str, chunking_name: str, embedder: str = EMBEDDER):
    """The run's pairs not yet in the model's cache, as rerankers.Work."""
    cache = ScoreCache(model)
    out = []
    for query, items in units.pool_items(run, chunking_name, embedder):
        have = cache.get([k for k, _ in items])
        todo = [(k, d) for k, d in items if k not in have]
        if todo:
            out.append((query, todo))
    return out


def rerank(run: Run, model: str, max_usd: float | None, embedder: str = EMBEDDER) -> None:
    spec = RERANKERS[model]
    itext = INSTRUCTION
    for c in run.chunkings:
        if info(run.dataset).partitioned:
            work = simulate_histories.scoring_work(embedder, run.dataset, _size(c), run.sample, model, run.expand,
                                                   run.pool_seeds)
        else:
            work = missing(run, model, c, embedder)
        pairs = sum(len(items) for _, items in work)
        tokens = sum(len(r) for q, items in work for r in pair_ids(q, [d for _, d in items], itext))
        label = f"{run.dataset}/{run.scope}/{run.variant}/{c}{f'+x{run.expand}' if run.expand else ''}"
        print(f"{model} {label}: {pairs} pairs to score, {tokens / 1e6:.2f}M tokens, "
              f"${tokens / 1e6 * (spec.usd_per_mtok or 0.0):.2f}", flush=True)
        if pairs:
            score(model, work, itext, label, max_usd)


def simulate_and_report(runs: list[Run], embedder: str = EMBEDDER) -> list[str]:
    """Simulate every (dataset, scope, variant, sample, chunking) the runs
    touch, with all rerankers, then write each (dataset, scope, variant, sample) report."""
    groups: dict[tuple, set] = defaultdict(set)
    reported = set()
    history_jobs: dict[tuple, set[int]] = defaultdict(lambda: set(simulate_histories.EXPANSIONS))
    for run in runs:
        if info(run.dataset).partitioned:  # both scoring positions replay in one simulation
            for c in run.chunkings:  # the replayed expansions, plus any whose windows this run scores
                history_jobs[(run.dataset, _size(c), run.sample)] |= {run.expand} - {0}
            reported.add((run.dataset, "history", run.variant, run.sample))
            continue
        for c in run.chunkings:
            groups[(run.dataset, run.scope, run.variant, run.sample)].add(c)
    for (dataset, size, sample_name), expansions in sorted(history_jobs.items()):
        simulate_histories.run(embedder, dataset, size, sample_name, list(RERANKERS), tuple(sorted(expansions)))
    for (dataset, scope, variant, sample_name), chunkings in sorted(groups.items()):
        for c in sorted(chunkings):
            simulate.run(embedder, dataset, scope, variant, sample_name, c, list(RERANKERS))
        reported.add((dataset, scope, variant, sample_name))
    written = []
    for dataset, scope, variant, sample_name in sorted(reported):
        results = load(embedder, dataset, scope, variant, sample_name)
        written += [tables.write(results, e) for e in tables.EVENTS] + [figures.write(results)]
    return written
