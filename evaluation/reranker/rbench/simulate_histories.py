"""Replay reading for partitioned (chat-history) datasets: one tidy row per (query, ranker).

Per query, inside its own history's EventMemory partition (memory.py, histories.py):
  1. The first stage: EventMemory.query without a reranker, at a vector search limit deep
     enough to fill the largest reader budget (FIRST_STAGE_LIMIT), at each expand_context.
     Its seeds in similarity order, and each seed's context (the seed, or its window) exactly
     as EventMemory returns it. Reranker inputs are those contexts rendered as EventMemory's
     query renders them for its reranker.
  2. A ranker orders the seeds (rankers.py). Depths are counts of seeds, EventMemory's
     vector_search_limit, for every ranker, so reranking seeds and reranking their windows are
     compared at the same overfetch. A reranked configuration reads only its reranked seeds
     (EventMemory reranks every seed it retrieves); the first stage reads as many as the budget
     takes, or, with a depth L ("first_stage@nL"), only its top L.
  3. Each seed becomes a context: the seed alone, or its EventMemory window (expand e).
     Rerank-then-expand scores seeds; expand-then-rerank (xrerank) scores windows.
  4. Segments are admitted as EventMemory admits them under a growing segment limit
     (reading.admission), and the reader cost after each one is the token count
     of the context string EventMemory would render (reading.py): headers with timestamps,
     JSON text, components; overlaps counted once.
  5. metrics.evaluate gives the coverage events (complete and touched), recall and precision
     against the gold turns.

Expansions: rerank-then-expand and the expanded first stage need no new scores, so they are
replayed at every value in config.EVAL.expansions; expand-then-rerank needs the windows scored and appears
only where a run scored them (grid.py scores expand 3).

Output: results/sim/<embedder>/<dataset>/history/question/<sample>/seg<size>.parquet,
the same columns as simulate.py plus:
  admitted                 segments admitted before reading stopped (past MAX_COST, or at the
                           end of the reranked pool)
  <event>_ctx              contexts (seeds) read when the event happens: coverage after
                           reranking at k kept seeds is <event>_ctx <= k
  segs_t<B>, ctx_t<B>      segments and contexts read within reader budget B: EventMemory's
                           max_num_segments, and the first stage's vector_search_limit, at B
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import asyncio

from rbench import histories as hs
from rbench import metrics, rankers
from rbench.memory import RENDER, RERANK_FORMAT, CachedReranker, Memory
from rbench.paths import SIM
from rbench.reading import admission, reading
from rbench.rerankers import RERANKERS, ScoreCache, pair_overhead, prompt_key
from rbench.samples import sample
from rbench.schema import queries
from rbench.simulate import COLUMNS, scored_prefix

MAX_COST = 16000  # reading stops past the largest reader budget measured
FIRST_STAGE_LIMIT = 400  # the first stage's vector search limit: enough seeds to fill MAX_COST
EVENTS = ("first", "every", "first_touch", "every_touch")
EXTRA = (["admitted"] + [f"{e}_ctx" for e in EVENTS]
         + [f"segs_t{b}" for b in metrics.BUDGETS] + [f"ctx_t{b}" for b in metrics.BUDGETS])


def path(embedder: str, dataset: str, sample_name: str, size: int) -> object:
    return SIM / embedder / dataset / "history" / "question" / sample_name / f"seg{size}.parquet"


def gold_for(dataset: str, qid: str, seg: hs.Segments) -> metrics.Gold:
    turns = hs.gold_segments(dataset, qid, seg)
    chars = {i: len(seg.texts[i]) for t in turns for i in t}
    spans = tuple(tuple((i, chars[i]) for i in t) for t in turns)
    return metrics.Gold((spans,), chars, tuple(sorted(chars)))


def keys_for(query: str, prompt: str, texts: list[str]) -> list[str]:
    """Cache keys of rendered units (they are already rendered: no further render step)."""
    return [ScoreCache.key(prompt, RENDER, query, t) for t in texts]


def read(contexts: list[list[int]], seeds: list[int], seg: hs.Segments, lazy: bool = False
         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(admitted segment ids, cumulative rendered cost, the context each came from), stopping
    past MAX_COST. lazy: contexts enter the query result only as admission reaches them (the
    first stage without a vector search limit, whose limit is the seeds it reads)."""
    pairs = admission(contexts, seeds)
    order, ctx = [s for s, _ in pairs], [k for _, k in pairs]
    cum = reading(contexts, order, seg.event, seg.inner, seg.run, MAX_COST, ctx if lazy else None)
    return np.asarray(order[: len(cum)], dtype=np.int64), cum, np.asarray(ctx[: len(cum)], dtype=np.int64)


def counts(values: list[float], cum: np.ndarray, ctx: np.ndarray) -> list[float]:
    """The count normalizations: contexts (seeds) read when each event happens, and the
    segments and contexts read within each reader budget (EventMemory's max_num_segments and,
    for the first stage, its vector_search_limit at that budget)."""
    f = dict(zip(metrics.FIELDS, values, strict=True))
    seen = np.maximum.accumulate(ctx) + 1 if len(ctx) else ctx  # contexts reached after each segment
    at = [float(seen[int(f[f"{e}_k"]) - 1]) if np.isfinite(f[f"{e}_k"]) else np.inf for e in EVENTS]
    reach = np.maximum.accumulate(cum) if len(cum) else cum
    segs = [int(np.searchsorted(reach, b, "right")) for b in metrics.BUDGETS]
    return at + segs + [float(seen[n - 1]) if n else 0.0 for n in segs]


def run(embedder: str, dataset: str, size: int, sample_name: str, models: list[str],
        expansions: tuple[int, ...] | None = None) -> pd.DataFrame:
    """expansions: expand_context values to replay (default: config.EVAL.expansions)."""
    from rbench import config

    expansions = config.EVAL.expansions if expansions is None else expansions
    return asyncio.run(_run(embedder, dataset, size, sample_name, models, expansions))


async def first_stage(embedder: str, dataset: str, size: int, qid: str, query: str, expansions: tuple[int, ...]):
    """(segments, seed ids in similarity order, their similarities, {expand: each seed's
    context as segment ids}, {expand: each context rendered for the reranker}) from
    EventMemory.query without a reranker."""
    from memmachine_server.episodic_memory.event_memory.event_memory import EventMemory

    async with Memory(embedder, dataset, size) as m:
        seg = await hs.table(m, dataset, qid)
        partition = await (await m.event_memory()).open_or_create_partition(qid)
        results = {e: await partition.query(query, vector_search_limit=FIRST_STAGE_LIMIT, expand_context=e)
                   for e in (0, *expansions)}
    index = {u: i for i, u in enumerate(seg.uuid)}
    first = results[0].scored_segment_contexts
    seeds = [str(c.seed_segment_uuid) for c in first]
    units, texts = {}, {}
    for e, result in results.items():
        by_seed = {str(c.seed_segment_uuid): c.segments for c in result.scored_segment_contexts}
        assert set(by_seed) == set(seeds), "every expansion retrieves the same seeds"
        units[e] = [[index[str(x.uuid)] for x in by_seed[u]] for u in seeds]
        texts[e] = [EventMemory.string_from_segment_context(by_seed[u], format_options=RERANK_FORMAT) for u in seeds]
    return seg, [index[u] for u in seeds], np.asarray([c.score for c in first]), units, texts


async def _run(embedder: str, dataset: str, size: int, sample_name: str, models: list[str],
               expansions: tuple[int, ...]) -> pd.DataFrame:
    rows = queries(dataset)
    prompts = {m: prompt_key(RERANKERS[m].template) for m in models}
    caches = {m: ScoreCache(m) for m in models}
    all_rankers = rankers.for_models(models, expansions, rankers.HISTORY_DEPTHS, limited_first_stage=True)
    records: dict[str, list] = {r.id: [] for r in all_rankers}
    dropped: dict[str, str] = {}
    picked = sample(dataset, sample_name)
    for qi in picked:
        q = rows[qi]
        query = q["variants"]["question"]
        seg, head, sims, units, texts = await first_stage(embedder, dataset, size, q["qid"], query, expansions)
        g = gold_for(dataset, q["qid"], seg)
        overheads = {m: pair_overhead(query, RERANKERS[m].template) for m in models}
        chars = np.asarray([len(t) for t in seg.texts])
        scores = {(m, e): scored_prefix(caches[m], m, keys_for(query, prompts[m], texts[e])) for m in models for e in units}
        seed_tokens = seg.tokens[head]
        priority = {e: np.asarray([2 if any(i in g.union for i in u) else 0 for u in units[e]]) for e in units}
        for r in all_rankers:
            if r.id not in records:
                continue
            n = r.depth.chunks(seed_tokens) if r.depth else 0
            logits = None
            cost = 0.0
            if r.model is not None:
                logits, pair_tokens = scores[(r.model, r.expand if r.kind == "xrerank" else 0)]
                if n > len(logits):  # some query was not scored this deep: drop the configuration
                    del records[r.id]
                    dropped[r.id] = f"{q['qid']} has {len(logits)} scored, {n} needed"
                    continue
                cost = float(pair_tokens[:n].sum())
            perm = r.order(n, sims, logits, priority[r.expand])
            seeds = [head[i] for i in perm]
            contexts = [units[r.expand][i] for i in perm]
            admitted, cum, ctx = read(contexts, seeds, seg, lazy=r.kind == "first_stage" and r.depth is None)
            values = metrics.evaluate(admitted, cum, chars, g)
            prefix = cost - max(0, n - 1) * overheads[r.model] if r.model is not None else 0.0
            records[r.id].append([n if r.model else 0, cost, prefix, n if r.kind == "oracle" else 0, *values,
                                  len(admitted), *counts(values, cum, ctx)])
    frames = []
    for r in all_rankers:
        if r.id not in records:
            continue
        meta = {"qid": [rows[i]["qid"] for i in picked], "split": [rows[i]["split"] for i in picked],
                "cluster": [rows[i]["qid"] for i in picked], "chunking": f"seg{size}", "ranker": r.id, "kind": r.kind,
                "model": r.model, "depth": r.depth.label if r.depth else None, "expand": r.expand,
                "question_type": [rows[i]["features"]["question_type"] for i in picked]}
        frames.append(pd.concat([pd.DataFrame(meta), pd.DataFrame(records[r.id], columns=[*COLUMNS, *EXTRA])], axis=1))
    report_dropped(dataset, size, dropped, set(records))
    out = pd.concat(frames, ignore_index=True)
    curves = [c for c in out.columns if c.startswith(("recall_", "precision_"))]
    out[curves] = out[curves].astype("float32")  # ~230 curve columns per row; float32 halves the files
    target = path(embedder, dataset, sample_name, size)
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(target, index=False)
    return out


def report_dropped(dataset: str, size: int, dropped: dict[str, str], kept: set[str]) -> None:
    """Say which configurations of the scored models were left out because some question was
    not scored deep enough (models with no scores at all, and windows never scored at an
    expansion, drop out silently: that is by design)."""
    for m in sorted({k.split(":")[1].split("@")[0] for k in kept if ":" in k}):
        for kind, what in (("rerank", "seeds"), ("xrerank", "windows")):
            def expand(k: str) -> str:
                return k.split("+x")[1] if "+x" in k else "0"

            scored = {expand(k) for k in kept if k.startswith(f"{kind}:{m}@")}
            relevant = {k: v for k, v in dropped.items() if k.startswith(f"{kind}:{m}@") and expand(k) in scored}
            lost = sorted({k.split("@")[1].split("+")[0] for k in relevant}, key=lambda d: int(d[1:]))
            if lost:
                example = next(v for k, v in relevant.items() if k.split("@")[1].split("+")[0] == lost[0])
                print(f"{dataset} seg{size}: {m} reranking {what} is left out at vector search limits {', '.join(lost)}: "
                      f"some question was not scored that deep (e.g. {example})", flush=True)


def scoring_work(embedder: str, dataset: str, size: int, sample_name: str, model: str, expand: int,
                 pool_seeds: int):
    """Pairs to score for a run, as rerankers.Work: what EventMemory.query sends its reranker
    for each question at vector search limit pool_seeds (the deepest the run replays), found by
    querying with the reranker in record mode; unique by key, not yet in the model's cache."""
    return asyncio.run(_scoring_work(embedder, dataset, size, sample_name, model, expand, pool_seeds))


async def _scoring_work(embedder: str, dataset: str, size: int, sample_name: str, model: str, expand: int,
                        pool_seeds: int):
    rows = queries(dataset)
    reranker = CachedReranker(model, mode="record")
    for qi in sample(dataset, sample_name):
        q = rows[qi]
        async with Memory(embedder, dataset, size) as m:
            partition = await (await m.event_memory(reranker)).open_or_create_partition(q["qid"])
            await partition.query(q["variants"]["question"], vector_search_limit=pool_seeds, expand_context=expand)
    work, seen = [], set()
    for query, items in reranker.work.items():
        items = [(k, t) for k, t in items if k not in seen]  # queued for another question already
        seen.update(k for k, _ in items)
        if items:
            work.append((query, items))
    return work
