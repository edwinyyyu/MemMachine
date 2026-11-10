"""Running a grid stage: prepare inputs, rerank, simulate, report.

Every step skips what is already cached, so a stage can be rerun after an interruption or
extended later, and only new work is done.

Every dataset runs through MemMachine's EventMemory (partitions.py, simulation.py).
Simulation always uses every reranker (rerankers.RERANKERS) and every expansion that has
scores, whichever stage scored them: a configuration is kept only when every query of the
sample was scored deep enough. So running one stage never hides another stage's results.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import defaultdict

from rbench import config, depth, simulation
from rbench import partitions as pt
from rbench.embedding import DEFAULT as EMBEDDER
from rbench.grid import Run
from rbench.report import figures, readings, tables
from rbench.report.data import load
from rbench.rerankers import RERANKERS, pair_ids, score
from rbench.samples import sample
from rbench.progress import elapsed
from rbench.schema import queries

_PREPARED: set[tuple] = set()  # (embedder, dataset, scope, chunking, sample) done in this process


def prepare(run: Run, embedder: str = EMBEDDER, chunkings: tuple[str, ...] | None = None) -> None:
    """Encode the partitions the run's queries search (each embedded once; the stores are kept),
    then measure how far reading its pool reaches and how much evidence it holds (depth.py,
    free: reported beside the pool, which is fixed).
    chunkings: only these of the run's (default all)."""
    for c in chunkings or run.chunkings:
        if (embedder, run.dataset, run.scope, c, run.sample) in _PREPARED:
            continue  # the same stage's other scoring position
        _PREPARED.add((embedder, run.dataset, run.scope, c, run.sample))
        pt.ingest(embedder, run.dataset, c, sample(run.dataset, run.sample), run.scope)
        if not depth._known(embedder, run.dataset, c, run.sample, run.expand):
            depth.measure(embedder, run.dataset, c, run.sample, run.scope, run.variant,
                          (*depth.FILL_EXPANSIONS, run.expand))
        print(f"{run.dataset} {c} {run.sample}: pool {depth.pool(run)} seeds", flush=True)


def rerank(run: Run, model: str, embedder: str = EMBEDDER, chunkings: tuple[str, ...] | None = None) -> None:
    """Score a run's pairs at one model, for its chunkings (or only these, already encoded)."""
    spec = RERANKERS[model]
    for c in chunkings or run.chunkings:
        work = simulation.scoring_work(embedder, run.dataset, c, run.sample, model, run.expand,
                                       depth.pool(run, c, embedder), run.scope, run.variant)
        pairs = sum(len(items) for _, items in work)
        tokens = sum(len(r) for q, items in work for r in pair_ids(q, [d for _, d in items], spec.template))
        label = f"{run.dataset}/{run.scope}/{run.variant}/{c}{f'+x{run.expand}' if run.expand else ''}"
        print(f"{model} {label}: {pairs} pairs to score, {tokens / 1e6:.2f}M tokens, "
              f"${tokens / 1e6 * (spec.usd_per_mtok or 0.0):.2f}", flush=True)
        if pairs:
            score(model, work, label)




class _Scorer(threading.Thread):
    """One hosted reranker scoring in the background while later partitions embed. Each call
    takes everything queued (rerankers.score); a failure (out of credits, an hour of errors)
    is kept and raised in the main thread (check, close), which then stops ingesting."""

    def __init__(self, model: str, note: str) -> None:
        super().__init__(daemon=True, name=f"score-{model}")
        self.model, self.note = model, note
        self.queue: queue.Queue = queue.Queue()
        self.error: BaseException | None = None
        self.pairs = 0
        self.cancel = threading.Event()  # set: send nothing more (requests in flight finish and are cached)

    def submit(self, work) -> None:
        if work:
            self.queue.put(work)

    def run(self) -> None:
        last, closing = time.monotonic(), False
        while not closing:
            work, item = [], self.queue.get()
            while True:
                if item is None:
                    closing = True
                else:
                    work += item
                try:
                    item = self.queue.get_nowait()
                except queue.Empty:
                    break
            if work and self.error is None and not self.cancel.is_set():
                try:
                    score(self.model, work, self.note, progress=False, cancel=self.cancel)
                    self.pairs += sum(len(items) for _, items in work)
                except BaseException as err:  # SystemExit too: raised again by check() in the main thread
                    self.error = err
            if time.monotonic() - last > 30 or closing:
                last = time.monotonic()
                print(f"  {self.model} {self.note}: {self.pairs} pairs scored while embedding", flush=True)

    def check(self) -> None:
        if self.error is not None:
            raise self.error

    def close(self) -> None:
        self.queue.put(None)
        self.join()
        self.check()

    def abort(self) -> None:
        """Stop: queued work is dropped and nothing more is sent; the requests in flight finish
        and are cached (a paid request is never left unrecorded)."""
        self.cancel.set()
        self.queue.put(None)
        self.join()


def _overlapped(embedder: str, dataset: str, scope: str, variant: str, sample_name: str, c: str, members: list[Run],
                hosted: list[str], caps: dict[int, int]) -> list[str]:
    """prepare_and_score's overlapped path for one (dataset, scope, variant, sample, chunking)."""
    from rbench.memory import CachedReranker

    out: list[str] = []
    label = f"{dataset}/{scope}/{variant}/{c}"
    scorers = {m: _Scorer(m, f"{label} early") for m in hosted}
    for s in scorers.values():
        s.start()
    expansions = tuple(sorted({0, *depth.FILL_EXPANSIONS, *(r.expand for r in members)}))
    per: dict = {}
    early: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)  # (model, expand) -> (rank, tokens)
    rows, fmt = queries(dataset), pt.formats(dataset)
    wanted = {(m, r.expand) for r in members for m in r.models if m in hosted}
    submitted: dict[str, set[str]] = {m: set() for m in hosted}  # keys queued but maybe not cached yet: sent once

    async def after(m, k, idx, p) -> None:
        for s in scorers.values():
            s.check()  # a scorer that stopped stops the ingest too
        measured = await depth.measure_partition(m, dataset, k, p, idx, variant, expansions)
        per.update(measured)
        for model in hosted:
            template = RERANKERS[model].template
            for e, cap in caps.items():
                if (model, e) not in wanted:  # only the (model, expansion) pairs a run asked for
                    continue
                reranker = CachedReranker(model, mode="record")
                partition = await (await m.event_memory(reranker)).open_or_create_partition(k)
                for qi in idx:
                    q = rows[qi]
                    await partition.query(q["variants"][variant], vector_search_limit=cap,
                                          expand_context=e, format_options=fmt.rerank)
                work = []
                for query, items in reranker.work.items():
                    # once each, in order (e.g. a window that is its seed, or two identical segments)
                    items = [(key, d) for key, d in items if key not in submitted[model] and not submitted[model].add(key)]
                    if items:
                        work.append((query, items))
                for query, items in work:
                    rank = {key: i for i, (key, _) in enumerate(reranker.asked[query])}
                    lengths = [len(x) for x in pair_ids(query, [d for _, d in items], template)]
                    early[(model, e)] += [(rank[key], n) for (key, _), n in zip(items, lengths, strict=True)]
                scorers[model].submit(work)

    failure: BaseException | None = None
    try:
        pt.ingest(embedder, dataset, c, sample(dataset, sample_name), scope, after=after)
    except BaseException as err:  # stop the scorers cleanly before raising (Ctrl-C too)
        failure = err
    for s in scorers.values():
        try:
            s.abort() if failure is not None else s.close()  # after a failure: only what is in flight finishes
            s.check()
        except BaseException as err:
            failure = failure or err
    if failure is not None:
        raise failure
    depth.save(embedder, dataset, c, sample_name, per, expansions)
    for r in members:
        _PREPARED.add((embedder, dataset, scope, c, sample_name))
        print(f"{dataset} {c} {sample_name}: pool {depth.pool(r, c, embedder)} seeds"
              + (f" (windows x{r.expand})" if r.expand else ""), flush=True)
    for r in members:  # the rest, up to the measured pools (pairs scored early are cached)
        for m in r.models:
            rerank(r, m, embedder, (c,))
    for (model, e), pairs in sorted(early.items()):
        usd = (RERANKERS[model].usd_per_mtok or 0.0) / 1e6
        out.append(f"{model} {label} {'windows x' + str(e) if e else 'seeds'}: {len(pairs)} pairs "
                   f"(${sum(n for _, n in pairs) * usd:.2f}) scored while embedding, each query to its pool of {caps[e]}")
        print(out[-1], flush=True)
    return out


def prepare_and_score(runs: list[Run], embedder: str = EMBEDDER) -> list[str]:
    """prepare, then rerank every run's models, overlapped where it saves time at no cost.

    Where a hosted reranker has work and partitions still need embedding, the reranker scores
    each partition's pairs while the next ones embed: in the ingest loop, once a partition is
    in, the same thread measures its queries' depths and runs EventMemory's query in record
    mode (every engine call stays sequential), and hands the pairs (plain data) to one
    background scorer per model (_Scorer). The pools are fixed (config.EVAL.pool_seeds,
    pool_windows), so each query is scored to its whole pool while the next partitions embed
    and nothing is scored that a sequential run would not; after ingest the depths are saved
    (reported beside the pools) and the usual rerank finds only what is left. What was scored
    while embedding is counted and priced in the lines returned (also printed). Local
    rerankers run after ingest as before (they would compete with the local embedder for the
    GPU)."""
    from rbench.memory import store_root

    lines: list[str] = []
    jobs: dict[tuple, list[Run]] = defaultdict(list)
    for run in runs:
        for c in run.chunkings:
            jobs[(run.dataset, run.scope, run.variant, run.sample, c)].append(run)
    for (dataset, scope, variant, sample_name, c), members in jobs.items():
        hosted = sorted({m for r in members for m in r.models if RERANKERS[m].backend != "local"})
        keyed = pt.groups(dataset, sample(dataset, sample_name), scope)
        ready = all((store_root(embedder, dataset, c) / "done" / k).exists() for k in keyed)
        caps = {r.expand: depth.pool(r) for r in members}  # the pools are fixed: each query scored to its own
        if not hosted or ready:
            for r in members:
                prepare(r, embedder, (c,))
            for r in members:
                for m in r.models:
                    rerank(r, m, embedder, (c,))
            continue
        lines += _overlapped(embedder, dataset, scope, variant, sample_name, c, members, hosted, caps)
    return lines


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
            pools[key] |= {config.EVAL.pool_seeds, config.EVAL.pool_windows}  # both, whichever stage scored them
    for key, expansions in sorted(jobs.items()):
        dataset, scope, variant, sample_name, c = key
        started = time.monotonic()
        simulation.run(embedder, dataset, c, sample_name, list(RERANKERS), tuple(sorted(expansions)), scope, variant,
                       max(pools[key]), tuple(sorted(pools[key])))
        print(f"  simulation {dataset} {c} {sample_name}: {elapsed(started)}", flush=True)
    return written_reports(embedder, sorted({k[:4] for k in jobs}))


def written_reports(embedder: str, keys) -> list[str]:
    """Reports and charts for each (dataset, scope, variant, sample) from its saved results,
    with each step's time (`--only draw` runs this alone)."""
    written = []
    for dataset, scope, variant, sample_name in keys:
        results = load(embedder, dataset, scope, variant, sample_name)
        tables.clear(results)
        started = time.monotonic()
        written += [tables.write(results, e) for e in readings.available(results.df)]
        print(f"  tables {dataset} {sample_name}: {elapsed(started)}", flush=True)
        started = time.monotonic()
        written.append(figures.write(results))
        print(f"  charts {dataset} {sample_name}: {elapsed(started)}", flush=True)
    if written:
        written.append(tables.index())
    return written
