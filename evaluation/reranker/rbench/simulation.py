"""Replay reading for every dataset: one tidy row per (query, ranker).

Per query, inside the EventMemory partition it searches (memory.py, partitions.py):
  1. The first stage: EventMemory.query without a reranker, at each expand_context, as deep as
     the deepest pool scored (the run's vector search limit, depth.py; at least
     FIRST_STAGE_LIMIT).
     Its seeds in similarity order, and each seed's context (the seed, or its window) exactly
     as EventMemory returns it. Reranker inputs are those contexts rendered as EventMemory's
     query renders them for its reranker (partitions.formats).
  2. A ranker orders the seeds (rankers.py). Depths are counts of seeds, EventMemory's
     vector_search_limit, for every ranker, so reranking seeds and reranking their windows are
     compared at the same overfetch. A reranked configuration reads only its reranked seeds
     (EventMemory reranks every seed it retrieves); the first stage reads as many as the budget
     takes, or, with a depth L ("first_stage@nL"), only its top L.
  3. Each seed becomes a context: the seed alone, or its EventMemory window (expand e).
     Rerank-then-expand scores seeds; expand-then-rerank (xrerank) scores windows.
  4. Segments are admitted as EventMemory admits them under a growing segment limit
     (reading.admission), and the reader cost after each one is the token count of the
     context string EventMemory would render (reading.py): headers (with timestamps for chat
     histories, none for documents), JSON text, components; overlaps counted once.
  5. metrics.evaluate gives the coverage events (complete and touched), recall and precision
     against the gold spans (partitions.gold: LongMemEval's gold turns, the other datasets'
     character spans).

Expansions: rerank-then-expand and the expanded first stage need no new scores, so they are
replayed at every value in config.EVAL.expansions; expand-then-rerank needs the windows scored
and appears only where a run scored them (grid.Run.expand).

Output: results/sim/<embedder>/<dataset>/<scope>/<variant>/<sample>/<chunking>.parquet
  qid, cluster, chunking                   the query; `cluster` is what intervals resample
  ranker, kind, model, depth, expand       the configuration (rankers.Ranker)
  question_type                            the dataset's question type, where it has one
  pairs, rr_tokens, rr_prefix_tokens       reranker cost: pairs scored, Qwen3 tokens as an API
                                           bills them (every pair whole), and as a server with a
                                           prefix cache computes them (prompt and query once)
  oracle_pairs                             the depth the perfect reranker reorders
  metrics.FIELDS                           coverage costs (reader tokens), ranks, recall, precision
  admitted                                 segments admitted: reading runs to the end of the
                                           pool (the first stage's: its whole vector search), so
                                           no curve is cut short of its plateau
  seg_chars, seg_tokens                    the partition's mean segment size, measured
  unit_tokens                              the mean segment-text tokens one retrieved seed's
                                           context holds at the row's expand_context (the seed
                                           alone, or its window), over every seed retrieved
  bound_exact                              bound rows: whether the bound is exact (bounds.py)
  share_full_k, share_touch_k              per evidence turn: the seeds read when it is read in full /
                                           touched (recall at k seeds kept: report/topk.py)
  share_full, share_touch                  per evidence turn (span) of the answer read in full first:
                                           the reader cost at which it is read in full / touched
                                           (lists; report/readings.py's share readings)
  <event>_ctx                              contexts (seeds) read when the event happens:
                                           coverage after reranking at k kept seeds is <= k
  segs_t<B>, ctx_t<B>                      segments and contexts read within reader budget B
                                           (EventMemory's max_num_segments, and the first
                                           stage's vector_search_limit, at B), for every budget
                                           in config.EVAL.budgets and config.EVAL.columns
"""

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd

from rbench import bounds, config, metrics, rankers
from rbench import partitions as pt
from rbench.datasets import info
from rbench.dbs import replace_atomically
from rbench.memory import RENDER, CachedReranker, Memory
from rbench.paths import SIM
from rbench.reading import admission, reading
from rbench.rerankers import RERANKERS, ScoreCache, pair_overhead, prompt_key, to_logit
from rbench.samples import sample
from rbench.schema import queries

from memmachine_server.episodic_memory.event_memory.event_memory import EventMemory  # noqa: I001

FIRST_STAGE_LIMIT = 400  # the first stage's vector search limit when no deeper pool was scored
COLUMNS = ["pairs", "rr_tokens", "rr_prefix_tokens", "oracle_pairs", *metrics.FIELDS]
EVENTS = ("first", "every", "first_touch", "every_touch")


# Bump when the simulation's logic or columns change: results written by an earlier version are
# simulated again (run() otherwise reuses results whose inputs are unchanged).
VERSION = 1


def fingerprint(embedder: str, dataset: str, chunking: str, sample_name: str, models: list[str],
                expansions: tuple[int, ...], scope: str, variant: str, limit: int, pools: tuple[int, ...]) -> str:
    """What a simulation's results depend on: VERSION, its arguments, the stored budgets and
    columns, each partition's encoding (its done stamp: new segments, new uuids) and each
    model's score cache (its last score and call: new scores can deepen what is replayed)."""
    import hashlib
    import json

    from rbench.memory import store_root

    h = hashlib.sha1(json.dumps([VERSION, embedder, dataset, chunking, sample_name, sorted(models), sorted(expansions),
                                 scope, variant, limit, sorted(pools), stored_budgets(), extra(),
                                 config.EVAL.token_top]).encode())
    root = store_root(embedder, dataset, chunking)
    for k in sorted(pt.groups(dataset, sample(dataset, sample_name), scope)):
        done = root / "done" / k
        h.update((done.read_text() if done.exists() else "-").encode())
    for m in sorted(models):
        db = ScoreCache(m).db
        h.update(repr((db.execute("SELECT MAX(rowid) FROM scores").fetchone(),
                       db.execute("SELECT MAX(rowid) FROM calls").fetchone())).encode())
    return h.hexdigest()


def stored_budgets() -> tuple[int, ...]:
    """Reader budgets the count columns are stored at: every one the report uses."""
    return tuple(sorted(set(config.EVAL.budgets) | set(config.EVAL.columns)))


def extra() -> list[str]:
    budgets = stored_budgets()
    return (["admitted", "seg_chars", "seg_tokens", "unit_tokens", "share_full", "share_touch", "share_full_k",
             "share_touch_k", "bound_exact"]
            + [f"{e}_ctx" for e in EVENTS]
            + [f"segs_t{b}" for b in budgets] + [f"ctx_t{b}" for b in budgets])


def path(embedder: str, dataset: str, scope: str, variant: str, sample_name: str, chunking: str):
    return SIM / embedder / dataset / scope / variant / sample_name / f"{chunking}.parquet"


def cluster_of(dataset: str, row: dict) -> str:
    """What bootstrap intervals resample for a query (the dataset's cluster field)."""
    field = info(dataset).cluster
    return str(row[field] if field in row else row["features"][field])


def scored_prefix(cache: ScoreCache, model: str, keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Logits and pair tokens for the longest scored prefix of a list of cache keys."""
    got = cache.get(keys)
    n = 0
    while n < len(keys) and keys[n] in got:
        n += 1
    return (to_logit(model, np.asarray([got[k][0] for k in keys[:n]], dtype=np.float64)),
            np.asarray([got[k][1] for k in keys[:n]], dtype=np.float64))


def keys_for(query: str, prompt: str, texts: list[str]) -> list[str]:
    """Cache keys of rendered units (they are already rendered: no further render step)."""
    return [ScoreCache.key(prompt, RENDER, query, t) for t in texts]


def read(contexts: list[list[int]], seeds: list[int], seg: pt.Segments, lazy: bool = False
         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(admitted segment ids, cumulative rendered cost, the context each came from), reading
    every context. lazy: contexts enter the query result only as admission reaches them (the
    first stage without a vector search limit, whose limit is the seeds it reads)."""
    pairs = admission(contexts, seeds)
    order, ctx = [s for s, _ in pairs], [k for _, k in pairs]
    cum = reading(contexts, order, seg.event, seg.inner, seg.run, context_of=ctx if lazy else None)
    return np.asarray(order[: len(cum)], dtype=np.int64), cum, np.asarray(ctx[: len(cum)], dtype=np.int64)


def counts(values: list[float], cum: np.ndarray, ctx: np.ndarray) -> list[float]:
    """The count normalizations: contexts (seeds) read when each event happens, and the
    segments and contexts read within each stored reader budget."""
    f = dict(zip(metrics.FIELDS, values, strict=True))
    seen = np.maximum.accumulate(ctx) + 1 if len(ctx) else ctx  # contexts reached after each segment
    at = [float(seen[int(f[f"{e}_k"]) - 1]) if np.isfinite(f[f"{e}_k"]) else np.inf for e in EVENTS]
    reach = np.maximum.accumulate(cum) if len(cum) else cum
    segs = [int(np.searchsorted(reach, b, "right")) for b in stored_budgets()]
    return at + segs + [float(seen[n - 1]) if n else 0.0 for n in segs]


async def first_stage(partition, seg: pt.Segments, query: str, expansions: tuple[int, ...], rerank_format,
                      limit: int = FIRST_STAGE_LIMIT):
    """(seed ids in similarity order, their similarities, {expand: each seed's context as
    segment ids}, {expand: each context rendered for the reranker}) from EventMemory.query
    without a reranker."""
    results = {e: await partition.query(query, vector_search_limit=limit, expand_context=e,
                                        format_options=rerank_format) for e in (0, *expansions)}
    index = {u: i for i, u in enumerate(seg.uuid)}
    first = results[0].scored_segment_contexts
    seeds = [str(c.seed_segment_uuid) for c in first]
    units, texts = {}, {}
    for e, result in results.items():
        by_seed = {str(c.seed_segment_uuid): c.segments for c in result.scored_segment_contexts}
        assert set(by_seed) == set(seeds), "every expansion retrieves the same seeds"
        units[e] = [[index[str(x.uuid)] for x in by_seed[u]] for u in seeds]
        texts[e] = [EventMemory.string_from_segment_context(by_seed[u], format_options=rerank_format) for u in seeds]
    return [index[u] for u in seeds], np.asarray([c.score for c in first]), units, texts


def keys_for_hashes(query: str, prompt: str, hashes: list[str]) -> list[str]:
    """keys_for from the rendered units' hashes (rerankers.ScoreCache.key's form)."""
    from rbench.rerankers import _h

    q = _h(query)[:16]
    return [f"{prompt}|{RENDER}|{q}|{h}" for h in hashes]


async def first_stage_cached(m: Memory, k: str, partition, seg: pt.Segments, query: str, expansions: tuple[int, ...],
                             rerank_format, limit: int):
    """first_stage with each rendered context kept as its hash (all the replay needs: the score
    cache's key), cached beside the stores per query: keyed by the partition's encoding (its done
    stamp), the query, the expansions, the limit and the format. A metric, reading or ranker
    change then replays without querying EventMemory again (the queries and rendering are about
    80% of a simulation's time). Written atomically; an unreadable file is recomputed."""
    import hashlib
    import json
    import os

    from rbench.rerankers import _h

    expansions = tuple(sorted({0, *expansions}))
    name = hashlib.sha1(json.dumps([m.done(k).read_text(), query, expansions, limit, repr(rerank_format)]).encode())
    path = m.root / "first_stage" / k / f"{name.hexdigest()[:24]}.npz"
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as d:
                units = {e: np.split(d[f"units_{e}"], np.cumsum(d[f"lengths_{e}"])[:-1]) for e in expansions}
                return (d["head"].tolist(), d["sims"], {e: [u.tolist() for u in v] for e, v in units.items()},
                        {e: d[f"hashes_{e}"].astype(str).tolist() for e in expansions})
        except Exception:  # noqa: BLE001 - recomputed below
            pass
    head, sims, units, texts = await first_stage(partition, seg, query, expansions[1:], rerank_format, limit)
    hashes = {e: [_h(t)[:16] for t in texts[e]] for e in expansions}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(temporary, head=np.asarray(head, dtype=np.int64), sims=np.asarray(sims, dtype=np.float64),
                        **{f"units_{e}": np.asarray([s for u in units[e] for s in u], dtype=np.int64) for e in expansions},
                        **{f"lengths_{e}": np.asarray([len(u) for u in units[e]], dtype=np.int64) for e in expansions},
                        **{f"hashes_{e}": np.asarray(hashes[e], dtype="S16") for e in expansions})
    os.replace(temporary, path)
    return head, sims, units, hashes


def run(embedder: str, dataset: str, chunking: str, sample_name: str, models: list[str],
        expansions: tuple[int, ...] | None = None, scope: str | None = None, variant: str | None = None,
        limit: int | None = None, pools: tuple[int, ...] = ()) -> pd.DataFrame:
    """expansions: expand_context values to replay (default: config.EVAL.expansions); scope and
    variant default to the dataset's own; limit: the first stage's vector search limit (the
    deepest pool scored; default FIRST_STAGE_LIMIT); pools: the runs' pool depths, replayed as
    limits beside the grid (a measured depth is rarely on it)."""
    expansions = config.EVAL.expansions if expansions is None else expansions
    scope, variant = pt.scope_of(dataset, scope), variant or info(dataset).default_variant
    limit = max(FIRST_STAGE_LIMIT, limit or 0)
    target = path(embedder, dataset, scope, variant, sample_name, chunking)
    stamp = target.with_suffix(".fingerprint")
    key = fingerprint(embedder, dataset, chunking, sample_name, models, tuple(expansions), scope, variant, limit,
                      tuple(pools))
    if target.exists() and stamp.exists() and stamp.read_text() == key:
        print(f"{dataset} {chunking} {sample_name}: simulation current (its inputs are unchanged), reused", flush=True)
        return pd.read_parquet(target)
    out = asyncio.run(_run(embedder, dataset, chunking, sample_name, models, expansions, scope, variant, limit,
                           tuple(pools)))
    stamp.write_text(key)  # after the results: a run cut short leaves no stamp, so it is simulated again
    return out


async def _run(embedder: str, dataset: str, chunking: str, sample_name: str, models: list[str],
               expansions: tuple[int, ...], scope: str, variant: str, limit: int, pools: tuple[int, ...]) -> pd.DataFrame:
    rows = queries(dataset)
    fmt = pt.formats(dataset)
    prompts = {m: prompt_key(RERANKERS[m].template) for m in models}
    caches = {m: ScoreCache(m) for m in models}
    all_rankers = rankers.for_models(models, expansions, rankers.depths(limit, pools), limited_first_stage=True)
    records: dict[str, list] = {r.id: [] for r in all_rankers}
    dropped: dict[str, str] = {}
    order: list[int] = []
    from rbench.progress import Progress

    keyed = pt.groups(dataset, sample(dataset, sample_name), scope)
    progress = Progress(f"{dataset} {chunking} simulation", sum(len(v) for v in keyed.values()), "queries")
    for k, members in keyed.items():
        p = pt.part(dataset, rows[members[0]], scope)
        async with Memory(embedder, dataset, chunking) as m:
            seg = await pt.table(m, dataset, p)
            partition = await (await m.event_memory()).open_or_create_partition(k)
            chars = seg.chars
            sizes = [float(chars.mean()), float(seg.tokens.mean())]
            for qi in members:
                q = rows[qi]
                query = q["variants"][variant]
                head, sims, units, texts = await first_stage_cached(m, k, partition, seg, query, expansions, fmt.rerank,
                                                                    limit)
                g = pt.gold(pt.gold_spans(dataset, q, p), p, seg)
                _replay(q, query, seg, head, sims, units, texts, g, chars, sizes, models, prompts, caches,
                        all_rankers, records, dropped)
                order.append(qi)
                progress.step()
    frames = []
    for r in all_rankers:
        if r.id not in records:
            continue
        meta = {"qid": [rows[i]["qid"] for i in order], "cluster": [cluster_of(dataset, rows[i]) for i in order], "chunking": chunking, "ranker": r.id,
                "kind": r.kind, "model": r.model, "depth": r.depth.label if r.depth else None, "expand": r.expand,
                "question_type": [rows[i].get("features", {}).get("question_type") for i in order]}
        frames.append(pd.concat([pd.DataFrame(meta), pd.DataFrame(records[r.id], columns=[*COLUMNS, *extra()])], axis=1))
    report_dropped(dataset, chunking, dropped, set(records))
    out = pd.concat(frames, ignore_index=True)
    curves = [c for c in out.columns if c.startswith(("recall_", "precision_"))]
    out[curves] = out[curves].astype("float32")  # ~230 curve columns per row; float32 halves the files
    target = path(embedder, dataset, scope, variant, sample_name, chunking)
    target.parent.mkdir(parents=True, exist_ok=True)
    replace_atomically(target, lambda temporary: out.to_parquet(temporary, index=False))
    return out


def _replay(q, query, seg, head, sims, units, texts, g, chars, sizes, models, prompts, caches, all_rankers, records,
            dropped) -> None:
    """Every ranker's reading of one query, appended to records (dropping a configuration
    whose scores do not reach its depth for this query)."""
    overheads = {m: pair_overhead(query, RERANKERS[m].template) for m in models}
    scores = {(m, e): scored_prefix(caches[m], m, keys_for_hashes(query, prompts[m], texts[e])) for m in models
              for e in units}  # texts: the rendered contexts' hashes (first_stage_cached)
    unit_tokens = {e: float(np.mean([seg.tokens[u].sum() for u in units[e]])) if units[e] else 0.0 for e in units}
    priority = {e: np.asarray([2 if any(i in g.union for i in u) else 0 for u in units[e]]) for e in units}
    for r in all_rankers:
        if r.id not in records:
            continue
        n = r.depth.chunks(seg.tokens[head]) if r.depth else 0
        if r.kind == "bound":  # the cheapest reading of the top n seeds' pool any ranking could achieve
            pool = {s for u in units[r.expand][:n] for s in u}
            b = bounds.bound(g, pool, seg.inner, seg.run)
            values = [b.get(f, np.nan) for f in metrics.FIELDS]
            records[r.id].append([0, 0.0, 0.0, 0, *values, np.nan, *sizes, unit_tokens[r.expand], b["share_full"],
                                  b["share_touch"], [], [], b["exact"],
                                  *([np.nan] * (len(EVENTS) + 2 * len(stored_budgets())))])
            continue
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
        seen = np.maximum.accumulate(ctx) + 1 if len(ctx) else ctx  # seeds reached after each admitted segment
        full, touched, full_k, touched_k = metrics.span_costs(admitted, cum, g, seen)
        records[r.id].append([n if r.model else 0, cost, prefix, n if r.kind == "oracle" else 0, *values,
                              len(admitted), *sizes, unit_tokens[r.expand], full, touched, full_k, touched_k, np.nan,
                              *counts(values, cum, ctx)])


def report_dropped(dataset: str, chunking: str, dropped: dict[str, str], kept: set[str]) -> None:
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
                print(f"{dataset} {chunking}: {m} reranking {what} is left out at vector search limits {', '.join(lost)}: "
                      f"some question was not scored that deep (e.g. {example})", flush=True)


def scoring_work(embedder: str, dataset: str, chunking: str, sample_name: str, model: str, expand: int,
                 pool_seeds: int, scope: str | None = None, variant: str | None = None):
    """Pairs to score for a run, as rerankers.Work: what EventMemory.query sends its reranker
    for each query at vector search limit pool_seeds (the deepest the run replays), found by
    querying with the reranker in record mode; unique by key, not yet in the model's cache."""
    reranker = asyncio.run(_record(embedder, dataset, chunking, sample_name, model, expand, pool_seeds,
                                   pt.scope_of(dataset, scope), variant or info(dataset).default_variant))
    work, seen = [], set()
    for query, items in reranker.work.items():
        items = [(key, t) for key, t in items if key not in seen and not seen.add(key)]  # once each, in order
        if items:
            work.append((query, items))
    return work


def pool_pairs(embedder: str, dataset: str, chunking: str, sample_name: str, model: str, expand: int, pool_seeds: int,
               scope: str | None = None, variant: str | None = None) -> dict[str, list[tuple[str, str]]]:
    """Every pair EventMemory.query sends its reranker per query (scored or not), in order:
    query text -> [(cache key, rendered context)]."""
    return asyncio.run(_record(embedder, dataset, chunking, sample_name, model, expand, pool_seeds,
                               pt.scope_of(dataset, scope), variant or info(dataset).default_variant)).asked


async def _record(embedder: str, dataset: str, chunking: str, sample_name: str, model: str, expand: int,
                  pool_seeds: int, scope: str, variant: str) -> CachedReranker:
    rows = queries(dataset)
    fmt = pt.formats(dataset)
    reranker = CachedReranker(model, mode="record")
    for k, members in pt.groups(dataset, sample(dataset, sample_name), scope).items():
        async with Memory(embedder, dataset, chunking) as m:
            partition = await (await m.event_memory(reranker)).open_or_create_partition(k)
            for qi in members:
                await partition.query(rows[qi]["variants"][variant], vector_search_limit=pool_seeds,
                                      expand_context=expand, format_options=fmt.rerank)
    return reranker
