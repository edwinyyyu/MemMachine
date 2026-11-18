"""The core simulation's measurements under MemMachine's runtime pool: rerank 4 x k, keep k.

The core simulation (simulation.py) replays EventMemory as configured directly: a vector search
limit L, every one of the L seeds reranked and read in reranked order until the reader budget.
MemMachine's LongTermMemory instead asks EventMemory for 4 x k seeds for k results
(runtime.OVERFETCH), and keeps the first k after reranking. This module replays that policy
through the core simulation's own machinery (first_stage_cached, simulation._replay, the reading
and metric code), into a results set of its own (sample "all-runtime", beside the core's and
never mixed into it), so every core measurement can be computed for it unchanged:

  first_stage@n<k>[+x<e>]          the first k seeds of the vector search (as in the core)
  rerank:<m>@n<k>[+x<e>]           rerank the first 4 x k seeds, keep the first k   (the runtime)
  rerank:<m>~200@n<k>[+x<e>]       rerank the first 200 seeds, keep the first k     (a flat 200)

so in this set `@n<k>` is the limit k, the results returned (MemMachine's `limit`), on both
sides; the reranker's cost is its measured pair tokens over the pool it scores (4 x k or 200
seeds), and the reader reads the k results in order until its budget. With expand_context e
each kept seed is read with its window (rerank the seeds, then expand, as in the core; windows
were not scored). frontier.equal_spend(results, "<m>", ...) and ("<m>~200", ...) then give the
same-money comparison at limit k for each pool, and readings/stats the rest.

  python -m rbench runtime-core [--models q4-mem,q8-mem]
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rbench import partitions as pt
from rbench.rankers import Depth, Ranker
from rbench.runtime import KS, OVERFETCH, POOL

SAMPLE = "all-runtime"  # the results set's name (simulation.path's sample folder)
SOURCE = "all"  # the questions: the core's
FLAT = "~200"  # the flat-pool rankers' model suffix: rerank:<m>~200@n<k>


@dataclass(frozen=True)
class Keep(Ranker):
    """A reranker over the first `depth` seeds that keeps the first `keep` of its order; its id is
    `alias` (the results set's naming: the limit k, not the pool)."""

    keep: int = 0
    alias: str = ""

    @property
    def id(self) -> str:
        return self.alias

    def order(self, n: int, sims: np.ndarray, logits: np.ndarray | None, is_gold: np.ndarray) -> np.ndarray:
        return super().order(n, sims, logits, is_gold)[: self.keep]


def rankers_for(models: list[str], expansions: tuple[int, ...]) -> list[Ranker]:
    """The first stage unlimited, and at each limit k and expand_context: the first stage's first
    k, and per model the runtime pool (4 x k) and the flat 200, each keeping k."""
    out: list[Ranker] = [Ranker("first_stage")]
    for e in (0, *expansions):
        tail = f"+x{e}" if e else ""
        if e:
            out.append(Ranker("first_stage", expand=e))
        for k in KS:
            out.append(Ranker("first_stage", depth=Depth(k), expand=e))
            for m in models:
                out.append(Keep("rerank", m, Depth(OVERFETCH * k), e, keep=k, alias=f"rerank:{m}@n{k}{tail}"))
                out.append(Keep("rerank", m, Depth(POOL), e, keep=k, alias=f"rerank:{m}{FLAT}@n{k}{tail}"))
    return out


async def _run(embedder: str, dataset: str, chunking: str, models: list[str], expansions: tuple[int, ...],
               limit: int) -> pd.DataFrame:
    from rbench.memory import Memory
    from rbench.progress import Progress
    from rbench.rerankers import RERANKERS, ScoreCache, prompt_key
    from rbench.samples import sample
    from rbench.schema import queries
    from rbench.simulation import COLUMNS, _replay, cluster_of, extra, first_stage_cached

    rows, fmt = queries(dataset), pt.formats(dataset)
    scope, variant = "history", "question"
    prompts = {m: prompt_key(RERANKERS[m].template) for m in models}
    caches = {m: ScoreCache(m) for m in models}
    rankers = rankers_for(models, expansions)
    records: dict[str, list] = {r.id: [] for r in rankers}
    dropped: dict[str, str] = {}
    order: list[int] = []
    keyed = pt.groups(dataset, sample(dataset, SOURCE), scope)
    progress = Progress(f"{dataset} {chunking} runtime simulation", sum(len(v) for v in keyed.values()), "queries")
    for key, members in keyed.items():
        p = pt.part(dataset, rows[members[0]], scope)
        async with Memory(embedder, dataset, chunking) as m:
            seg = await pt.table(m, dataset, p)
            partition = await (await m.event_memory()).open_or_create_partition(key)
            sizes = [float(seg.chars.mean()), float(seg.tokens.mean())]
            for qi in members:
                q = rows[qi]
                query = q["variants"][variant]
                head, sims, units, texts = await first_stage_cached(m, key, partition, seg, query, expansions,
                                                                    fmt.rerank, limit)
                g = pt.gold(pt.gold_spans(dataset, q, p), p, seg)
                _replay(q, query, seg, head, sims, units, texts, g, seg.chars, sizes, models, prompts, caches, rankers,
                        records, dropped, pt.query_date(dataset, q))
                order.append(qi)
                progress.step()
    if dropped:
        raise RuntimeError(f"configurations without scores this deep: {dropped}")
    frames = []
    for r in rankers:
        pool = None if r.model is None else ("200" if FLAT in r.id else "4k")
        depth = f"n{r.keep}" if isinstance(r, Keep) else (r.depth.label if r.depth else None)
        meta = {"qid": [rows[i]["qid"] for i in order], "cluster": [cluster_of(dataset, rows[i]) for i in order],
                "chunking": chunking, "ranker": r.id, "kind": r.kind,
                "model": r.model, "depth": depth, "pool": pool,  # the flat pool's rows: by ranker id (~200)
                "expand": r.expand, "question_type": [rows[i].get("features", {}).get("question_type") for i in order]}
        frames.append(pd.concat([pd.DataFrame(meta), pd.DataFrame(records[r.id], columns=[*COLUMNS, *extra()])], axis=1))
    out = pd.concat(frames, ignore_index=True)
    curves = [c for c in out.columns if c.startswith(("recall_", "precision_"))]
    out[curves] = out[curves].astype("float32")
    return out


def run(embedder: str, dataset: str = "longmemeval_m", chunking: str = "seg500",
        models: tuple[str, ...] = ("q4-mem", "q8-mem")) -> str:
    """Replay the runtime pool for the core's questions and write the results set (simulation.path,
    sample SAMPLE). The core's first stage is reused from its cache: the same expansions and
    limit as the core run (config.EVAL.expansions; simulation.FIRST_STAGE_LIMIT, 400)."""
    from rbench import config
    from rbench.dbs import replace_atomically
    from rbench.simulation import FIRST_STAGE_LIMIT, path

    expansions = tuple(sorted(config.EVAL.expansions))
    out = asyncio.run(_run(embedder, dataset, chunking, list(models), expansions, max(FIRST_STAGE_LIMIT, POOL)))
    target = path(embedder, dataset, "history", "question", SAMPLE, chunking)
    target.parent.mkdir(parents=True, exist_ok=True)
    replace_atomically(target, lambda temporary: out.to_parquet(temporary, index=False))
    return str(target)
