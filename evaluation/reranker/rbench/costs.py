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


# Disk per stored segment, in the benchmark's own vector store (vectorstore.py). Measured
# 2026-11-10 on 5 LongMemEval-S histories at seg500 (8,096 segments): segments.sqlite 1,157 B
# with texts averaging about 300 characters (taken as 835 B plus the text, as measured
# 2026-11-09 on 163,303 segments), records.sqlite 192 B, the simulation's segment table 92 B
# (text-free format). Per distinct derived text, one row of the permanent copy
# (EmbeddingCache): the float32 vector plus 625 B (key, hash, their two indexes, page slack),
# measured on 3,000 rows at 1,024, 2,048 and 4,096 dimensions (4,720, 8,819 and 17,009 B); a
# row whose vector fits in a 4 KB page takes the page (4,220 B at 768). segments.sqlite's
# fixed part is not separately measured at other lengths.
SEGMENT_ROW, RECORD_ROW, TABLE_ROW = 835, 192, 92
CONTENT_ROW, CONTENT_PAGE_ROW = 625, 4220


def disk_bytes(segments: float, chars: float, dims: int, distinct: float | None = None) -> float:
    """Estimated disk of `segments` stored segments averaging `chars` characters: the segment
    store, the vector store's records and the segment table per segment; one row of the kept
    permanent copy (float32 vector, key and hash) per distinct derived text (default: one per
    segment), at the stored width `dims`."""
    distinct = segments if distinct is None else distinct
    return (segments * (SEGMENT_ROW + chars + RECORD_ROW + TABLE_ROW)
            + distinct * max(4 * dims + CONTENT_ROW, CONTENT_PAGE_ROW))


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


def _partition_counts(dataset: str, chunking: str, scope: str, k: str, q: dict, embedder: str) -> dict[str, np.ndarray]:
    """One partition's derived texts, from the production segmenter and deriver: per text its
    hash (the first 8 bytes of its sha1, as the permanent copy keys it), characters and billed
    tokens (the embedder's tokenizer, special tokens included), and its segments' characters.
    Cached per partition in data/counts/<dataset>/<chunking>/<key>.npz, stamped with what changes
    them (the scope, the tokenizer, the prepared data's date), so a resumed run counts only what
    it has not counted before."""
    import asyncio
    import hashlib
    import json

    from memmachine_server.episodic_memory.event_memory.deriver.text_deriver import WholeTextDeriver

    from rbench.dbs import replace_atomically
    from rbench.memory import segmenter
    from rbench.paths import DATA, PREPARED
    from rbench.text import count

    spec = EMBEDDERS[embedder]
    prepared = PREPARED / dataset / "stats.json"
    stamp = json.dumps([scope, spec.tokenizer, spec.special_tokens, prepared.stat().st_mtime_ns if prepared.exists() else 0])
    path = DATA / "counts" / dataset / chunking / f"{k}.npz"
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as data:
                if str(data["stamp"]) == stamp:
                    return {c: data[c] for c in ("hash", "chars", "tokens", "segment_chars")}
        except Exception:  # noqa: BLE001 - unreadable: counted again
            pass
    fmt = pt.formats(dataset).embed

    async def derive() -> tuple[list[str], int]:
        s, d = segmenter(chunking), WholeTextDeriver()
        texts, chars = [], 0
        for e in pt.part(dataset, q, scope).events:
            for seg in await s.segment(e, format_options=fmt):
                chars += len(seg.block.text)
                texts += [x.block.text for x in await d.derive(seg, format_options=fmt)]
        return texts, chars

    texts, chars = asyncio.run(derive())
    out = {"hash": np.asarray([int.from_bytes(hashlib.sha1(t.encode()).digest()[:8], "big") for t in texts], dtype=np.uint64),
           "chars": np.asarray([len(t) for t in texts], dtype=np.int64),
           "tokens": (count(texts, spec.tokenizer) + spec.special_tokens).astype(np.int64),
           "segment_chars": np.asarray(chars, dtype=np.int64)}
    path.parent.mkdir(parents=True, exist_ok=True)
    replace_atomically(path, lambda temporary: np.savez(temporary, stamp=np.str_(stamp), **out))
    return out


def _in_copy(embedder: str) -> np.ndarray:
    """The text hashes (as _partition_counts makes them) already in the embedder's permanent
    copy: embedded and paid for, so not counted again."""
    from rbench.dbs import connect
    from rbench.memory import EmbeddingCache

    path = EmbeddingCache.path_for(EMBEDDERS[embedder])
    if not path.exists():
        return np.zeros(0, dtype=np.uint64)
    db = connect(path)
    try:
        keys = [int(k[:16], 16) for (k,) in db.execute("SELECT key FROM vectors WHERE key NOT LIKE 'v:%'")]
    finally:
        db.close()
    return np.asarray(keys, dtype=np.uint64)


def exact_counts(dataset: str, chunking: str, keys: list[str], scope: str, embedder: str) -> dict:
    """Exact counts for partitions not yet encoded (CPU only, free; the user, 2026-11-10: count,
    do not extrapolate): segments (each one row in the segment store and one record in the
    vector store), and the distinct derived texts not yet in the permanent copy (each one row of
    it, which dedups by text: what is embedded and billed), their characters and billed tokens.
    Per-partition counts are cached (_partition_counts); the copy is read each time."""
    from rbench.progress import Progress

    rows = queries(dataset)
    by_key = {pt.key(dataset, r, scope): i for i, r in enumerate(rows)}
    progress = Progress(f"{dataset} {chunking} exact counts", len(keys), "partitions")
    parts = []
    for k in keys:
        parts.append(_partition_counts(dataset, chunking, scope, k, rows[by_key[k]], embedder))
        progress.step()
    hashes = np.concatenate([p["hash"] for p in parts]) if parts else np.zeros(0, dtype=np.uint64)
    chars = np.concatenate([p["chars"] for p in parts]) if parts else np.zeros(0, dtype=np.int64)
    tokens = np.concatenate([p["tokens"] for p in parts]) if parts else np.zeros(0, dtype=np.int64)
    distinct, first = np.unique(hashes, return_index=True)
    copied = np.isin(distinct, _in_copy(embedder))
    new = first[~copied]
    return {"segments": int(len(hashes)), "segment_chars": int(sum(int(p["segment_chars"]) for p in parts)),
            "distinct_texts": int(len(new)), "distinct_chars": int(chars[new].sum()), "tokens": int(tokens[new].sum()),
            "already_in_copy": int(copied.sum()), "partitions": len(keys)}


def disk_need(runs, embedder: str = DEFAULT_EMBEDDER, probes: int = 8) -> float:
    """GB the stores not yet encoded will take (estimate's disk rows, alone)."""
    def encoded(dataset: str, c: str, k: str) -> bool:
        return (store_root(embedder, dataset, c) / "done" / k).exists()

    todo: set[tuple[str, str, str, str]] = set()
    for run in runs:
        for c in run.chunkings:
            todo |= {(run.dataset, run.scope, c, k) for k in pt.groups(run.dataset, sample_of(run.dataset, run.sample),
                                                                         run.scope) if not encoded(run.dataset, c, k)}
    total = 0.0
    for dataset, scope, c in sorted({x[:3] for x in todo}):
        ks = sorted(k for d, s, n, k in todo if (d, s, n) == (dataset, scope, c))
        n = exact_counts(dataset, c, ks, scope, embedder)
        total += disk_bytes(n["segments"], n["segment_chars"] / max(1, n["segments"]), EMBEDDERS[embedder].dimensions,
                            n["distinct_texts"]) / 1e9
    return total


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
        n = exact_counts(dataset, c, ks, scope, embedder)  # counted, not extrapolated (CPU, cached)
        per_partition[(dataset, scope, c)] = float(n["tokens"])
        disk[f"{dataset} {c}"] = {"partitions": len(ks), "segments (exact)": n["segments"],
                                  "distinct texts not yet in the permanent copy (exact)": n["distinct_texts"],
                                  "distinct texts already in it": n["already_in_copy"],
                                  "GB": round(disk_bytes(n["segments"], n["segment_chars"] / max(1, n["segments"]),
                                                         EMBEDDERS[embedder].dimensions, n["distinct_texts"]) / 1e9, 1)}
    embed_tok = sum(per_partition.values())  # each text of a partition without vectors, embedded once
    summary = {m: {"billed_M": t / 1e6, "deepinfra_usd": t / 1e6 * (RERANKERS[m].usd_per_mtok or 0.0),
                   "local_hours": t / RERANKERS[m].local_tok_per_s / 3600 if RERANKERS[m].local_tok_per_s else None,
                   "deepinfra_hours": _hours(m, t), "basis": sorted(basis[m])} for m, t in per_model.items()}
    summary["embedding"] = {"embedder": embedder, "tokens_M": embed_tok / 1e6,
                            "usd": embed_tok / 1e6 * EMBEDDERS[embedder].usd_per_mtok,
                            "local_hours": embed_tok / MAC_EMBED_TOK_PER_S / 3600,
                            "basis": "exact: every distinct derived text not yet in the permanent copy, counted with the embedder's tokenizer"}
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
