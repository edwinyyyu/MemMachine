"""Every limit k, exactly: per question, the k results of each pool, read in full, for every k.

The runtime results set (runtime_sim) replays a few limits. This table holds every one: for each
question, expand_context and k, the k results of
  first   the first stage's first k seeds (k up to 400, its whole search)
  4k      the first 4 x k seeds reranked, the best k kept (k up to 50: 4 x 50 = 200, the pool scored)
  200     the first 200 seeds reranked, the best k kept (k up to 200)
read in full through the core's reading and metric code (simulation.read, metrics.evaluate and
metrics.span_costs): their reader tokens (T), the reranker's pair tokens for that pool (R), the
reader cost at which each event happens within them (inf if it does not), the share of evidence
turns read in full or touched, and the evidence characters' recall and precision.

A cross-encoder's score is per pair, so every pool's ranking is a sort of the cached scores of
the first 200 seeds: each k costs one sort, and the first stage and the flat 200 are prefixes of
one reading each. Budget first (runtime_report): at a reader budget B a question reads the
largest k whose k results fit in B, from a pool of 4 x that k; at the same money, the largest k
whose reranker pairs and reader tokens the money pays for.

  python -m rbench runtime-core      (runs it before the report when the table is missing)
"""

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd

from rbench import partitions as pt
from rbench.runtime import OVERFETCH, POOL

SAMPLE = "all-runtime-fit"
SOURCE = "all"
KMAX = {"first": 400, "4k": POOL // OVERFETCH, "200": POOL}
EVENTS = ("first", "every", "first_touch", "every_touch")
COLUMNS = ["T", "R", *EVENTS, "share_full", "share_touch", "recall", "precision"]


def _answer_curves(order: np.ndarray, chars: np.ndarray, g) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """metrics.evaluate's recall and precision ingredients after each admitted segment: the best
    answer's share of evidence characters read, the evidence characters read, and all characters
    read (the same computation, kept as arrays)."""
    union = g.union
    read_chars = np.cumsum(chars[order]) if len(order) else np.zeros(0)
    gold_chars = np.cumsum([union.get(int(c), 0) for c in order]) if len(order) else np.zeros(0)
    first_seen = np.zeros(len(order), dtype=bool)
    if len(order):
        first_seen[np.unique(np.asarray(order, dtype=np.int64), return_index=True)[1]] = True
    answer_read = []
    for alt in g.alternatives:
        total = sum(n for span in alt for _, n in span)
        if total:
            per_chunk: dict[int, int] = {}
            for span in alt:
                for c, n in span:
                    per_chunk[c] = per_chunk.get(c, 0) + n
            got = np.asarray([per_chunk.get(int(c), 0) for c in order], dtype=np.float64) * first_seen
            answer_read.append(np.cumsum(got) / total)
    best = np.max(answer_read, axis=0) if answer_read and len(order) else np.zeros(len(order))
    return best, gold_chars, read_chars


def _read(contexts, seeds, seg, lazy: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """simulation.read's admission and reading, with the rendered cost at the end of each context:
    (admitted segments, cumulative cost, the context of each, the cost once the first i + 1
    contexts are read). lazy: contexts join the result as their turn comes and each is opened when
    it ends, so the cost after context i is exactly that of a result of the first i + 1 contexts
    (a context can link segments read before it: with windows, a later one can change the cost);
    not lazy: every context is in the result from the start (a pool returned whole)."""
    from rbench.reading import Reader, admission

    pairs = admission(contexts, seeds)
    reader = Reader(contexts, seg.event, seg.inner, seg.run, lazy=lazy)
    order, ctx, cum, ends = [], [], [], np.zeros(len(contexts))
    j = 0
    for i in range(len(contexts)):
        while j < len(pairs) and pairs[j][1] == i:
            s = pairs[j][0]
            cum.append(reader.admit(int(s), i if lazy else None))
            order.append(int(s))
            ctx.append(i)
            j += 1
        if lazy:
            reader.open(i)
        ends[i] = reader.cost
    return (np.asarray(order, dtype=np.int64), np.asarray(cum, dtype=np.float64), np.asarray(ctx, dtype=np.int64),
            ends)


def prefixes(positions, head, units, seg, g, upto: int, lazy: bool = True) -> np.ndarray:
    """[k, COLUMNS less R]: the first k results of one ordering (head positions), read in full,
    for k = 1..upto (the core's reading: windows' shared segments admitted once). T is exact at
    every k (_read). Event costs are the reading's own: exact without windows; with windows a
    prefix's costs read the later contexts as not yet in the result. Whether an event happens
    within the k results, the shares, recall and precision do not depend on costs."""
    from rbench import metrics

    seeds = [head[i] for i in positions][:upto]
    contexts = [units[i] for i in positions][:upto]
    admitted, cum, ctx, ends = _read(contexts, seeds, seg, lazy)
    cum = np.maximum.accumulate(cum) if len(cum) else cum
    f = dict(zip(metrics.FIELDS, metrics.evaluate(admitted, cum, seg.chars, g), strict=True))
    seen = ctx + 1  # results reached after each admitted segment (contexts are admitted in order)
    _, _, full_k, touched_k = metrics.span_costs(admitted, cum, g, seen)
    full_k, touched_k = np.asarray(full_k, dtype=float), np.asarray(touched_k, dtype=float)
    best, gold_chars, read_chars = _answer_curves(admitted, seg.chars, g)
    ks = np.arange(1, upto + 1)
    n = np.searchsorted(seen, ks, "right")  # admitted segments within the first k results
    last = np.maximum(n - 1, 0)
    out = np.full((upto, len(COLUMNS) - 1), np.nan)
    out[:, 0] = ends[:upto]
    for j, e in enumerate(EVENTS, 1):
        pos = f[f"{e}_k"]  # segments read when it happens (inf: never)
        at = seen[int(pos) - 1] if np.isfinite(pos) else np.inf  # results reached by then
        out[:, j] = np.where(ks >= at, f[e], np.inf)
    out[:, 5] = [float(np.mean(full_k <= k)) if len(full_k) else 0.0 for k in ks]
    out[:, 6] = [float(np.mean(touched_k <= k)) if len(touched_k) else 0.0 for k in ks]
    has = (n > 0) & (len(admitted) > 0)
    out[:, 7] = np.where(has, best[last] if len(best) else 0.0, 0.0)
    out[:, 8] = np.where(has, gold_chars[last] / np.maximum(1, read_chars[last]) if len(read_chars) else 0.0, 0.0)
    return out


def _block(meta: dict, values: np.ndarray, rr: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(values, columns=[c for c in COLUMNS if c != "R"])
    frame.insert(1, "R", rr)
    for name, v in reversed(meta.items()):
        frame.insert(0, name, v)
    return frame


async def _run(embedder: str, dataset: str, chunking: str, models: list[str], expansions: tuple[int, ...],
               limit: int) -> pd.DataFrame:
    from rbench.memory import Memory
    from rbench.progress import Progress
    from rbench.rerankers import RERANKERS, ScoreCache, prompt_key, query_for
    from rbench.samples import sample
    from rbench.schema import queries
    from rbench.simulation import cluster_of, first_stage_cached, keys_for_hashes, scored_prefix

    rows, fmt = queries(dataset), pt.formats(dataset)
    prompts = {m: prompt_key(RERANKERS[m].template) for m in models}
    caches = {m: ScoreCache(m) for m in models}
    frames = []
    keyed = pt.groups(dataset, sample(dataset, SOURCE), "history")
    progress = Progress(f"{dataset} {chunking} every k", sum(len(v) for v in keyed.values()), "queries")
    for key, members in keyed.items():
        p = pt.part(dataset, rows[members[0]], "history")
        async with Memory(embedder, dataset, chunking) as m:
            seg = await pt.table(m, dataset, p)
            partition = await (await m.event_memory()).open_or_create_partition(key)
            for qi in members:
                q = rows[qi]
                query = q["variants"]["question"]
                head, _, units, texts = await first_stage_cached(m, key, partition, seg, query, expansions, fmt.rerank,
                                                                 limit)
                g = pt.gold(pt.gold_spans(dataset, q, p), p, seg)
                date = pt.query_date(dataset, q)
                base = {"qid": q["qid"], "cluster": cluster_of(dataset, q),
                        "question_type": q.get("features", {}).get("question_type")}
                scores = {mm: scored_prefix(caches[mm], mm, keys_for_hashes(query_for(mm, query, date), prompts[mm],
                                                                            texts[0])) for mm in models}
                for e in (0, *expansions):
                    n = min(KMAX["first"], len(head))
                    frames.append(_block({**base, "policy": "first", "model": None, "expand": e, "k": np.arange(1, n + 1)},
                                         prefixes(list(range(n)), head, units[e], seg, g, n), np.zeros(n)))
                    for mm, (logits, pair_tokens) in scores.items():
                        if len(logits) < POOL:
                            raise RuntimeError(f"{q['qid']}: {mm} has {len(logits)} scores, {POOL} needed")
                        flat = np.argsort(-logits[:POOL], kind="stable")
                        frames.append(_block({**base, "policy": "200", "model": mm, "expand": e,
                                              "k": np.arange(1, POOL + 1)},
                                             prefixes(flat, head, units[e], seg, g, POOL),
                                             np.full(POOL, float(pair_tokens[:POOL].sum()))))
                        values = np.full((KMAX["4k"], len(COLUMNS) - 1), np.nan)
                        rr = np.zeros(KMAX["4k"])
                        for k in range(1, KMAX["4k"] + 1):
                            order = np.argsort(-logits[: OVERFETCH * k], kind="stable")[:k]
                            values[k - 1] = prefixes(order, head, units[e], seg, g, k, lazy=False)[-1]  # returned whole
                            rr[k - 1] = float(pair_tokens[: OVERFETCH * k].sum())
                        frames.append(_block({**base, "policy": "4k", "model": mm, "expand": e,
                                              "k": np.arange(1, KMAX["4k"] + 1)}, values, rr))
                progress.step()
    out = pd.concat(frames, ignore_index=True)
    for c in COLUMNS:
        out[c] = out[c].astype("float32")
    return out


def path(embedder: str, dataset: str = "longmemeval_m", chunking: str = "seg500"):
    from rbench.simulation import path as sim_path

    return sim_path(embedder, dataset, "history", "question", SAMPLE, chunking)


def run(embedder: str, dataset: str = "longmemeval_m", chunking: str = "seg500",
        models: tuple[str, ...] = ("q4-mem", "q8-mem")) -> str:
    """Build the table from the core's cached first stage and scores, and write it."""
    from rbench import config
    from rbench.dbs import replace_atomically
    from rbench.simulation import FIRST_STAGE_LIMIT

    expansions = tuple(sorted(config.EVAL.expansions))
    out = asyncio.run(_run(embedder, dataset, chunking, list(models), expansions, max(FIRST_STAGE_LIMIT, POOL)))
    target = path(embedder, dataset, chunking)
    target.parent.mkdir(parents=True, exist_ok=True)
    replace_atomically(target, lambda temporary: out.to_parquet(temporary, index=False))
    return str(target)
