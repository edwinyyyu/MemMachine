"""Replay reading for every ranker and write one tidy row per (query, ranker).

Inputs are cached: pools (retrieval.py), chunk tables, and reranker scores
(rerankers.ScoreCache). A (model, depth) is recorded only if every query of the sample was
scored at least that deep, so a configuration never mixes complete and partial queries.
Every reranker in rerankers.RERANKERS is tried; those without scores simply drop out.

With expansions (expansion.py), each head item is a seed read with its window.
Rerank-then-expand uses the seeds' own scores; expand-then-rerank uses scores of the
windows' text, which a separate scoring run produces (grid.Run.expand).

Output: results/sim/<embedder>/<dataset>/<scope>/<variant>/<sample>/<chunking>__<instruction>__<render>.parquet
  qid, split, cluster, chunking            the query; `cluster` is what intervals resample
  ranker, kind, model, depth, weight, expand   the configuration (rankers.Ranker)
  pairs, rr_tokens, rr_prefix_tokens       reranker cost: pairs scored, Qwen3 tokens as an API
                                           bills them (every pair whole), and as a server with a
                                           prefix cache computes them (prompt and query once)
  oracle_pairs                             the depth the gold-first reference reorders
  metrics.FIELDS                           coverage costs (union reader tokens), ranks, recall,
                                           precision
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rbench import chunking, metrics, rankers, retrieval, units
from rbench.datasets import info
from rbench.expansion import Windows
from rbench.paths import SIM
from rbench.rerankers import ScoreCache, instruction_text, pair_overhead, prompt_key, to_logit
from rbench.samples import sample
from rbench.schema import corpus, queries

EXPANSIONS = (3, 15)  # EventMemory expand_context values: windows of 4 and 16 chunks (expansion.split)
MAX_WINDOW_TOKENS = 2048  # expansions whose windows would exceed this (nominally) are not replayed
COLUMNS = ["pairs", "rr_tokens", "rr_prefix_tokens", "oracle_pairs", *metrics.FIELDS]


def expansions_for(chunking_name: str) -> tuple[int, ...]:
    """Expansions worth replaying for a chunking: windows up to MAX_WINDOW_TOKENS nominal tokens."""
    size = int("".join(ch for ch in chunking_name if ch.isdigit()))
    if chunking_name.startswith("seg"):  # a character limit; about 4 characters per token
        size //= 4
    return tuple(e for e in EXPANSIONS if (1 + e) * size <= MAX_WINDOW_TOKENS)


def path(embedder, dataset, scope, variant, sample_name, chunking_name, instruction, render_mode):
    return SIM / embedder / dataset / scope / variant / sample_name / f"{chunking_name}__{instruction}__{render_mode}.parquet"


def cluster_of(dataset: str, row: dict) -> str:
    field = info(dataset).cluster
    return str(row.get(field) if field == "doc_id" else row["features"][field])


def check_inputs(dataset: str, table: chunking.ChunkTable, pools: retrieval.Pools) -> None:
    """Refuse cached inputs that no longer match the prepared data."""
    docs, rows = corpus(dataset), queries(dataset)
    if pools.qids.tolist() != [q["qid"] for q in rows]:
        raise RuntimeError(f"{dataset}: cached pools were built for other queries; delete them and rebuild")
    if table.doc_index.max() >= len(docs.doc_ids) or any(
            table.end[rows_][-1] != len(docs.texts[d]) for d, rows_ in enumerate(table.by_doc(len(docs.doc_ids)))):
        raise RuntimeError(f"{dataset}: cached chunk table does not tile the current documents; delete and rebuild")


def gold_priority(chunk_sets: list[set[int]], union: dict[int, int], cheapest: set[int]) -> np.ndarray:
    """The gold-first reference's sort key per head item: 2 if it holds a chunk of the cheapest
    complete answer, 1 if it holds other gold, else 0."""
    return np.asarray([2 if cs & cheapest else 1 if any(c in union for c in cs) else 0 for cs in chunk_sets])


def scored_prefix(cache: ScoreCache, model: str, keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Logits and pair tokens for the longest scored prefix of a list of cache keys."""
    got = cache.get(keys)
    n = 0
    while n < len(keys) and keys[n] in got:
        n += 1
    return (to_logit(model, np.asarray([got[k][0] for k in keys[:n]], dtype=np.float64)),
            np.asarray([got[k][1] for k in keys[:n]], dtype=np.float64))


def run(embedder: str, dataset: str, scope: str, variant: str, sample_name: str, chunking_name: str,
        models: list[str], instruction: str = "default", render_mode: str = "plain",
        expansions: tuple[int, ...] | None = None) -> pd.DataFrame:
    table = chunking.load(dataset, chunking_name)
    chunk_texts = table.texts(dataset)
    chars = table.end - table.start
    pools = retrieval.load(embedder, dataset, scope, chunking_name, variant)
    check_inputs(dataset, table, pools)
    golds = metrics.gold(dataset, table)
    expansions = expansions_for(chunking_name) if expansions is None else expansions
    windows = {e: Windows(dataset, table, e) for e in expansions}
    rows, picked = queries(dataset), sample(dataset, sample_name)
    caches = {m: ScoreCache(m) for m in models}
    itext = instruction_text(instruction, dataset)
    prompt = prompt_key(itext)
    all_rankers = rankers.for_models(models, expansions)
    records: dict[str, list] = {r.id: [] for r in all_rankers}
    for qi in picked:
        head, sims = pools.head(qi)
        g = golds[qi]
        union, cheapest = g.union, set(g.cheapest)
        query = rows[qi]["variants"][variant]
        overhead = pair_overhead(query, itext) if models else 0

        def scores(unit_texts: list[str], query: str = query) -> dict[str, tuple[np.ndarray, np.ndarray]]:
            keys = units.keys(prompt, render_mode, query, unit_texts)
            return {m: scored_prefix(caches[m], m, keys) for m in models}

        seed_scores = scores(units.texts(table, chunk_texts, head, None))
        per_e = {0: (gold_priority([{int(c)} for c in head], union, cheapest), table.tokens[head], None)}
        for e, w in windows.items():
            ids = [set(map(int, w.ids(int(c)))) for c in head]
            per_e[e] = (gold_priority(ids, union, cheapest), units.tokens(table, head, w),
                        scores(units.texts(table, chunk_texts, head, w)))
        for r in all_rankers:
            if r.id not in records:
                continue
            priority, window_tokens, window_scores = per_e[r.expand]
            n = r.depth.chunks(window_tokens if r.kind == "xrerank" else table.tokens[head]) if r.depth else 0
            logits, cost = None, 0.0
            if r.model is not None:
                logits, pair_tokens = (window_scores if r.kind == "xrerank" else seed_scores)[r.model]
                if n > len(logits):
                    del records[r.id]  # some query was not scored this deep: drop the configuration
                    continue
                cost = float(pair_tokens[:n].sum())
            seeds = head[r.order(n, sims, logits, priority)]
            order = seeds if r.expand == 0 else windows[r.expand].read_order(seeds)
            values = metrics.evaluate(order, np.cumsum(table.tokens[order]), chars, g)
            prefix_cost = cost - max(0, n - 1) * overhead if r.model is not None else 0.0
            records[r.id].append([n if r.model else 0, cost, prefix_cost, n if r.kind == "oracle" else 0, *values])
    frames = []
    for r in all_rankers:
        if r.id not in records:
            continue
        meta = {"qid": [rows[i]["qid"] for i in picked], "split": [rows[i]["split"] for i in picked],
                "cluster": [cluster_of(dataset, rows[i]) for i in picked], "chunking": chunking_name, "ranker": r.id,
                "kind": r.kind, "model": r.model, "depth": r.depth.label if r.depth else None, "weight": r.weight,
                "expand": r.expand}
        frames.append(pd.concat([pd.DataFrame(meta), pd.DataFrame(records[r.id], columns=COLUMNS)], axis=1))
    out = pd.concat(frames, ignore_index=True)
    target = path(embedder, dataset, scope, variant, sample_name, chunking_name, instruction, render_mode)
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(target, index=False)
    return out
