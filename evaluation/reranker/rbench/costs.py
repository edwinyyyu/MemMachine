"""Costs: what a grid stage will spend.

Sources of every rate (kept here so reports and estimates agree):
- DeepInfra list prices, checked 2026-11-05 on deepinfra.com: Qwen3-Reranker 0.6B $0.010,
  4B $0.025, 8B $0.050 per million input tokens (rerankers.RERANKERS). DeepInfra bills
  every pair whole: the fixed prompt, the instruction and the query are paid per pair.
- This Mac (M3 Pro, 18 GB), measured 2026-11-05: Qwen3-Reranker-0.6B at about 2,500 pair
  tokens/s (PyTorch MPS fp16, GPU to itself: rerankers.Reranker.local_tok_per_s);
  embeddinggemma-300m at about 10,500 tokens/s through llama-server.
- DeepInfra's embeddinggemma-300m: $0.002 per million tokens (checked 2026-11-05).
"""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np

from rbench import config, depth, simulation
from rbench import partitions as pt
from rbench.embedding import DEFAULT as DEFAULT_EMBEDDER
from rbench.embedding import EMBEDDERS
from rbench.memory import store_root
from rbench.rerankers import RERANKERS, ScoreCache, pair_ids
from rbench.samples import ORDER
from rbench.samples import sample as sample_of
from rbench.schema import queries

from memmachine_server.episodic_memory.event_memory.event_memory import EventMemory  # noqa: I001

MAC_EMBED_TOK_PER_S = 10500


# Disk per stored segment besides its vector (float16 x dimensions in the index), measured
# 2026-11-09 on LongMemEval-S proto at seg500 (163,303 segments, 768 dimensions): index 1,544 B
# (1,536 of vector), segments.sqlite 1,135 B with texts averaging 299 characters (taken as 835 B
# plus the text), vectors.sqlite 163 B (the vectors live in the index), the simulation's segment
# table 92 B (text-free format). An approximation from one store: segments.sqlite's fixed part
# is not separately measured at other lengths.
INDEX_EXTRA, SEGMENT_ROW, VECTOR_ROW, TABLE_ROW = 8, 835, 163, 92


def disk_bytes(segments: float, chars: float, dims: int) -> float:
    """Estimated disk of `segments` stored segments averaging `chars` characters."""
    return segments * (2 * dims + INDEX_EXTRA + SEGMENT_ROW + chars + VECTOR_ROW + TABLE_ROW)


IN_FLIGHT = 16  # the adaptive limit's start (limits.START): it grows while DeepInfra allows, so hours are an upper bound


def _hours(model: str, tokens: float) -> float | str:
    """Wall time to score `tokens` on DeepInfra, projected from the model's measured requests
    (its call log: median tokens per second per request, times the starting limit of requests in
    flight; the adaptive limit grows past it while DeepInfra allows, so this is an upper bound); a
    note when nothing has been timed yet (preflight smoke times the first)."""
    if RERANKERS[model].backend != "deepinfra":
        return "local"
    calls = ScoreCache(model).db.execute("SELECT tokens, seconds FROM calls WHERE seconds > 0").fetchall()
    if not calls:
        return "not measured yet (run smoke)"
    rate = float(np.median([t / s for t, s in calls]))
    return round(tokens / (rate * IN_FLIGHT) / 3600, 2)


def estimate(runs, embedder: str = DEFAULT_EMBEDDER, probes: int = 8) -> dict:
    """Reranker and embedding tokens (and dollars) a list of grid.Run needs.

    Runs that score the same pairs are counted once: they are grouped by (dataset, scope,
    variant, chunking, expand, model), and a group scores the union of its queries to its
    deepest pool (samples nest, so pilot queries are part of the core, and the core's pools
    are a prefix of the depth stage's). Where the group's partitions are encoded, its pairs are
    counted exactly (EventMemory queried with the reranker in record mode), minus those already
    cached. Before they are encoded, a few queries are probed: pair tokens are measured on
    pool_seeds randomly drawn seeds of MemMachine's segments (windows of neighbouring segments,
    rendered as EventMemory renders them for its reranker) and scaled to the queries. That is
    rough: retrieved seeds run longer than random ones at long segment lengths, and the cache is
    ignored. Embedding tokens are charged once for each partition not yet encoded."""

    def encoded(dataset: str, c: str, k: str) -> bool:
        return (store_root(embedder, dataset, c) / "done" / k).exists()

    groups: dict[tuple, list] = defaultdict(list)
    for run in runs:
        for c in run.chunkings:
            for m in run.models:
                groups[(run.dataset, run.scope, run.variant, c, run.expand, m)].append(run)
    per_model: dict[str, float] = {}
    basis: dict[str, set[str]] = defaultdict(set)
    unknown: list[str] = []  # groups whose pool depth is not measured yet
    by_group: dict[str, dict] = {}  # model, dataset, chunking, sample, seeds or windows -> pool, tokens, dollars
    for (dataset, scope, variant, c, expand, model), members in groups.items():
        sample_name = max((r.sample for r in members), key=ORDER.index)  # the largest holds the others
        template = RERANKERS[model].template
        rows, picked = queries(dataset), sample_of(dataset, sample_name)
        keys = pt.groups(dataset, picked, scope)
        ready = all(encoded(dataset, c, k) for k in keys)
        label = f"{dataset} {c} {sample_name}" + (f" windows x{expand}" if expand else " seeds")
        pool_seeds = max(depth.pool(r) for r in members)  # fixed in advance (config.EVAL.pool_seeds / pool_windows)
        if ready:
            work = simulation.scoring_work(embedder, dataset, c, sample_name, model, expand, pool_seeds, scope, variant)
            tokens = float(sum(len(x) for q, items in work for x in pair_ids(q, [d for _, d in items], template)))
            basis[model].add("exact (partitions encoded; cached pairs excluded)")
        else:
            rerank = pt.formats(dataset).rerank
            probe = picked[:: max(1, len(picked) // probes)][:probes]
            pair_tok = []
            for qi in probe:
                q = rows[qi]
                _, segments, _ = pt.offline(dataset, pt.key(dataset, q, scope), c, scope)
                seeds = random.Random(q["qid"]).sample(range(len(segments)), min(pool_seeds, len(segments)))
                before, after = expand // 3, expand - expand // 3  # EventMemory's split of expand_context
                texts = [EventMemory.string_from_segment_context(segments[max(0, s - before) : s + after + 1],
                                                                 format_options=rerank) for s in seeds]
                pair_tok.append(sum(len(x) for x in pair_ids(q["variants"][variant], texts, template)))
            tokens = float(np.mean(pair_tok)) * len(picked)
            basis[model].add("probe estimate (before encoding: random seeds, while retrieved ones run longer at long "
                             "segment lengths; ignores cached pairs)")
        per_model[model] = per_model.get(model, 0.0) + tokens
        shares = depth.shares_for(members[0], c, embedder)
        usd = tokens / 1e6 * (RERANKERS[model].usd_per_mtok or 0.0)
        by_group[f"{model} {label}"] = {"pool": pool_seeds,
                                        "reaches the axis end for (share of questions)":
                                            None if shares is None else round(shares[0], 3),
                                        "holds all evidence for": None if shares is None else round(shares[1], 3),
                                        "tokens_M": round(tokens / 1e6, 2), "usd": round(usd, 2)}


    to_embed: set[tuple[str, str, str, str]] = set()  # (dataset, scope, chunking, partition) not encoded yet
    for run in runs:
        for c in run.chunkings:
            to_embed |= {(run.dataset, run.scope, c, k) for k in pt.groups(run.dataset, sample_of(run.dataset, run.sample),
                                                                             run.scope) if not encoded(run.dataset, c, k)}
    per_partition: dict[tuple[str, str, str], float] = {}
    disk: dict[str, dict] = {}
    for dataset, scope, c in sorted({x[:3] for x in to_embed}):
        ks = sorted(k for d, s, n, k in to_embed if (d, s, n) == (dataset, scope, c))
        probe = ks[:: max(1, len(ks) // probes)][:probes]
        per_partition[(dataset, scope, c)] = float(np.mean([pt.embedding_tokens(dataset, k, c, scope, embedder)
                                                            for k in probe]))
        stored = [pt.offline(dataset, k, c, scope)[1] for k in probe]  # MemMachine Segments (cached by offline)
        segments = float(np.mean([len(s) for s in stored])) * len(ks)
        chars = float(np.mean([len(x.block.text) for s in stored for x in s]))
        disk[f"{dataset} {c}"] = {"partitions": len(ks), "segments_M": round(segments / 1e6, 2),
                                  "GB": round(disk_bytes(segments, chars, EMBEDDERS[embedder].dimensions) / 1e9, 1)}
    embed_tok = sum(per_partition[x[:3]] for x in to_embed)  # each partition without vectors is embedded once
    summary = {m: {"billed_M": t / 1e6, "deepinfra_usd": t / 1e6 * (RERANKERS[m].usd_per_mtok or 0.0),
                   "local_hours": t / RERANKERS[m].local_tok_per_s / 3600 if RERANKERS[m].local_tok_per_s else None,
                   "deepinfra_hours": _hours(m, t), "basis": sorted(basis[m])} for m, t in per_model.items()}
    summary["embedding"] = {"embedder": embedder, "tokens_M": embed_tok / 1e6,
                            "usd": embed_tok / 1e6 * EMBEDDERS[embedder].usd_per_mtok,
                            "local_hours": embed_tok / MAC_EMBED_TOK_PER_S / 3600, "estimate": True}
    if disk:
        import shutil

        from rbench.paths import DATA

        summary["disk for the stores not yet encoded (estimate)"] = disk | {
            "total GB": round(sum(v["GB"] for v in disk.values()), 1),
            "free GB now": round(shutil.disk_usage(DATA).free / 1e9, 1)}
    if by_group:
        summary["by scoring job (pool in seeds)"] = by_group
        summary["pools"] = (f"fixed: {config.EVAL.pool_seeds} seeds, {config.EVAL.pool_windows} windows (the user's "
                            "decision, 2026-11-10); the shares beside each are measured on its sample or the largest "
                            "smaller one. At shorter segments the same pool holds less text (at 250 characters about half "
                            "of 500's), so reading it ends before the axis end and curves level off earlier; a stage can "
                            "set its own pool (grid.Run.pool_seeds)")
    if unknown:
        summary["not priced (pool depth not measured: encode the partitions first)"] = unknown
    return {"summary": summary}
