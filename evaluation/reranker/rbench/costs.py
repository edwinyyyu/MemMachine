"""Costs: what a grid stage will spend, and what each configuration costs per query.

Sources of every rate (kept here so reports and estimates agree):
- DeepInfra list prices, checked 2026-11-05 on deepinfra.com: Qwen3-Reranker 0.6B $0.010,
  4B $0.025, 8B $0.050 per million input tokens (rerankers.RERANKERS). DeepInfra bills
  every pair whole: the fixed prompt, the instruction and the query are paid per pair.
- This Mac (M3 Pro, 18 GB), measured 2026-11-05: Qwen3-Reranker-0.6B at about 2,500 pair
  tokens/s (PyTorch MPS fp16, GPU to itself); embeddinggemma-300m at about 10,500 tokens/s
  through llama-server.
- DeepInfra's embeddinggemma-300m: $0.002 per million tokens (checked 2026-11-05).
- A served model with a prefix cache (vLLM's enable_prefix_caching, as the model card's
  vLLM example sets) computes the shared prompt and query once per query: the
  prefix-cached token count, reported beside the billed one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rbench import chunking, retrieval
from rbench.embedding import DEFAULT as DEFAULT_EMBEDDER
from rbench.paths import CHUNKS, VECTORS
from rbench.rerankers import INSTRUCTION, RERANKERS, _fixed_ids, pair_ids, pair_overhead
from rbench.samples import sample
from rbench.schema import corpus, queries

MAC_RERANK_TOK_PER_S = {"q06-local": 2500, "q06": 2500}
MAC_EMBED_TOK_PER_S = 10500


@dataclass
class Tally:
    """Reranker tokens a set of runs needs: as billed per pair, and as computed with a prefix cache."""

    queries: int = 0
    pairs: int = 0
    billed: float = 0.0
    prefix_cached: float = 0.0

    def add(self, other: "Tally") -> None:
        self.queries += other.queries
        self.pairs += other.pairs
        self.billed += other.billed
        self.prefix_cached += other.prefix_cached


def run_tokens(run, chunking_name: str, embedder: str = DEFAULT_EMBEDDER) -> Tally:
    """Tokens to score one chunking of a run. Exact (units.pool_items, the same definition
    scoring uses) when the run's pools are built; otherwise every chunk is assumed to be the
    chunking's mean size and every query's pool full."""
    from rbench import units

    itext = INSTRUCTION
    suffix = len(_fixed_ids()[1])
    tally = Tally()
    if retrieval.path(embedder, run.dataset, run.scope, chunking_name, run.variant).exists():
        for query, items in units.pool_items(run, chunking_name, embedder):
            lengths = [len(r) for r in pair_ids(query, [d for _, d in items], itext)]
            overhead = pair_overhead(query, itext)
            tally.queries += 1
            tally.pairs += len(lengths)
            tally.billed += sum(lengths)
            tally.prefix_cached += sum(lengths) - max(0, len(lengths) - 1) * overhead
        return tally
    table = chunking.load(run.dataset, chunking_name)
    rows, docs = queries(run.dataset), corpus(run.dataset)
    unit, qunit = float(table.tokens.mean()) * (1 + run.expand), float(table.qtokens.mean()) * (1 + run.expand)
    per_doc = np.bincount(table.doc_index, minlength=len(docs.doc_ids))
    for i in sample(run.dataset, run.sample):
        if run.scope == "corpus":
            cands = len(table)
        else:
            names = [rows[i]["doc_id"]] if run.scope == "doc" else rows[i]["scope_docs"]
            cands = int(sum(per_doc[docs.index[n]] for n in names))
        n = min(cands, max(run.min_depth, math.ceil(run.pool_tokens / unit)))
        overhead = pair_overhead(rows[i]["variants"][run.variant], itext)
        tally.queries += 1
        tally.pairs += n
        tally.billed += n * (overhead + suffix + qunit)
        tally.prefix_cached += overhead + n * (suffix + qunit)
    return tally


def embed_tokens(dataset: str, chunking_name: str, embedder: str = DEFAULT_EMBEDDER) -> int:
    """Tokens still to embed for a chunking (0 if its vectors exist)."""
    if (VECTORS / embedder / dataset / f"{chunking_name}.npy").exists():
        return 0
    if not (CHUNKS / dataset / f"{chunking_name}.npz").exists():
        return int(sum(len(t) for t in corpus(dataset).texts) / 4.2)  # about 4.2 characters per token
    table = chunking.load(dataset, chunking_name)
    return int(table.tokens.sum() + 9 * len(table))  # 9: the document prompt and special tokens


def estimate(runs, embedder: str = DEFAULT_EMBEDDER) -> dict:
    """Totals per model for a list of grid.Run, counting each distinct scoring job once.

    Two runs that score the same (model, dataset, scope, variant, chunking) share the cache, so only the deepest pool on the largest sample is counted."""
    from rbench.samples import ORDER

    jobs: dict[tuple, tuple] = {}
    embed: dict[tuple, int] = {}
    for run in runs:
        for c in run.chunkings:
            embed[(run.dataset, c)] = embed_tokens(run.dataset, c, embedder)
            for m in run.models:
                k = (m, run.dataset, run.scope, run.variant, c, run.expand)
                old = jobs.get(k)
                new = (run.pool_tokens, run.min_depth, run.sample)
                if old is None or (new[0], new[1], ORDER.index(new[2])) > (old[0], old[1], ORDER.index(old[2])):
                    jobs[k] = new
    per_model: dict[str, Tally] = {}
    per_job = []
    from rbench.grid import Run

    for (m, d, s, v, c, e), (pool_tokens, min_depth, sample_name) in jobs.items():
        job = Run(d, s, v, (c,), (m,), sample_name, pool_tokens, min_depth, expand=e)
        t = run_tokens(job, c, embedder)
        per_model.setdefault(m, Tally()).add(t)
        per_job.append({"model": m, "dataset": d, "scope": s, "variant": v, "chunking": c, "expand": e, "sample": sample_name,
                        "pool_tokens": pool_tokens, "queries": t.queries, "pairs": t.pairs, "billed_M": t.billed / 1e6,
                        "prefix_cached_M": t.prefix_cached / 1e6})
    summary = {}
    for m, t in per_model.items():
        price = RERANKERS[m].usd_per_mtok
        summary[m] = {
            "billed_M": t.billed / 1e6,
            "prefix_cached_M": t.prefix_cached / 1e6,
            "deepinfra_usd": t.billed / 1e6 * price if price else None,
            "mac_hours": t.billed / MAC_RERANK_TOK_PER_S[m] / 3600 if m in MAC_RERANK_TOK_PER_S else None,
        }
    etok = sum(embed.values())
    from rbench.embedding import EMBEDDERS

    summary["embedding"] = {"embedder": embedder, "tokens_M": etok / 1e6,
                            "usd": etok / 1e6 * EMBEDDERS[embedder].usd_per_mtok,
                            "mac_hours_if_local": etok / MAC_EMBED_TOK_PER_S / 3600}
    return {"summary": summary, "jobs": per_job}


def estimate_histories(runs, embedder: str = DEFAULT_EMBEDDER, histories: int = 8) -> dict:
    """ESTIMATE for history (partitioned) datasets.

    Runs that score the same pairs are counted once: they are grouped by (dataset, segment
    length, expand, model), and a group scores the union of its questions to its
    deepest pool (samples nest, so pilot questions are part of the core, and the core's pools
    are a prefix of the depth stage's). Where the group's pools exist, its pairs are counted
    exactly (EventMemory queried with the reranker in record mode), minus those already cached.
    Before the histories are encoded, a few are probed: pair tokens are measured on pool_seeds
    randomly drawn seeds of MemMachine's segments (windows of neighbouring segments, rendered as
    EventMemory renders them for its reranker) and scaled to the questions. That is rough:
    retrieved seeds run longer than random ones at long segment lengths, and the cache is
    ignored. Embedding tokens are charged once for each sampled history not yet encoded."""
    import random
    from collections import defaultdict

    from rbench import histories as hs
    from rbench import simulate_histories
    from rbench.embedding import EMBEDDERS
    from rbench.memory import MEMORY, RERANK_FORMAT

    from memmachine_server.episodic_memory.event_memory.event_memory import EventMemory  # noqa: I001

    def encoded(dataset: str, size: int, qid: str) -> bool:
        return (MEMORY / embedder / dataset / f"seg{size}" / "done" / qid).exists()
    from rbench.samples import ORDER
    from rbench.samples import sample as sample_of

    groups: dict[tuple, list] = defaultdict(list)
    for run in runs:
        for c in run.chunkings:
            for m in run.models:
                groups[(run.dataset, int(c.removeprefix("seg")), run.expand, m)].append(run)
    per_model: dict[str, float] = {}
    basis: dict[str, set[str]] = defaultdict(set)
    for (dataset, size, expand, model), members in groups.items():
        sample_name = max((r.sample for r in members), key=ORDER.index)  # the largest holds the others
        pool_seeds = max(r.pool_seeds for r in members)
        itext = INSTRUCTION
        rows = queries(dataset)
        if all(encoded(dataset, size, rows[i]["qid"]) for i in sample_of(dataset, sample_name)):
            work = simulate_histories.scoring_work(embedder, dataset, size, sample_name, model, expand, pool_seeds)
            tokens = float(sum(len(x) for q, items in work for x in pair_ids(q, [d for _, d in items], itext)))
            basis[model].add("exact (histories encoded; cached pairs excluded)")
        else:
            rows = queries(dataset)
            picked = sample_of(dataset, sample_name)
            probe = [rows[i] for i in picked[:: max(1, len(picked) // histories)][:histories]]
            pair_tok = []
            for q in probe:
                _, segments, _ = hs.offline(dataset, q["qid"], size)
                seeds = random.Random(q["qid"]).sample(range(len(segments)), min(pool_seeds, len(segments)))
                before, after = expand // 3, expand - expand // 3
                texts = [EventMemory.string_from_segment_context(segments[max(0, s - before) : s + after + 1],
                                                                 format_options=RERANK_FORMAT) for s in seeds]
                pair_tok.append(sum(len(x) for x in pair_ids(q["variants"]["question"], texts, itext)))
            tokens = float(np.mean(pair_tok)) * len(picked)
            basis[model].add("probe estimate (before encoding: random seeds, while retrieved ones run longer at long "
                             "segment lengths; ignores cached pairs)")
        per_model[model] = per_model.get(model, 0.0) + tokens

    embed_tok = 0.0
    to_embed: set[tuple[str, int, str]] = set()  # (dataset, size, qid) without vectors yet
    for run in runs:
        rows = queries(run.dataset)
        for c in run.chunkings:
            size = int(c.removeprefix("seg"))
            to_embed |= {(run.dataset, size, rows[i]["qid"]) for i in sample_of(run.dataset, run.sample)
                         if not encoded(run.dataset, size, rows[i]["qid"])}
    per_history: dict[tuple[str, int], float] = {}
    for dataset, size in {(d, n) for d, n, _ in to_embed}:
        rows = queries(dataset)
        qids = sorted(q for d, n, q in to_embed if (d, n) == (dataset, size))
        probe = qids[:: max(1, len(qids) // histories)][:histories]
        per_history[(dataset, size)] = float(np.mean([hs.embedding_tokens(dataset, q, size) for q in probe]))
    for dataset, size, _ in to_embed:  # each sampled history without vectors is embedded once
        embed_tok += per_history[(dataset, size)]
    summary = {m: {"billed_M": t / 1e6, "deepinfra_usd": t / 1e6 * (RERANKERS[m].usd_per_mtok or 0.0),
                   "basis": sorted(basis[m])} for m, t in per_model.items()}
    summary["embedding"] = {"embedder": embedder, "tokens_M": embed_tok / 1e6,
                            "usd": embed_tok / 1e6 * EMBEDDERS[embedder].usd_per_mtok, "estimate": True}
    return {"summary": summary}
