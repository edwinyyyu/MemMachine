"""Shared bench helpers: load_bench, metrics, make_cosine_rerank_fn, BENCH_NAMES.

These are the building blocks for any A/B harness over the bench suite.
The cosine reranker is the cheap default — expensive LLM calls
(extractor, planner) are shared across runs via on-disk cache.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from temporal_retrieval.research._common import DATA_DIR, EVAL_ROOT


# The full bench list used by the validation harness.
BENCH_NAMES = [
    "adversarial", "allen", "ambiguous_year", "ambiguous_year_adv",
    "axis", "causal_relative", "composition", "cotemporal",
    "dense_cluster", "disc", "edge_conjunctive_temporal", "edge_era_refs",
    "edge_multi_te_doc", "edge_relative_time", "engagement_disjoint",
    "era", "goldilocks", "goldilocks_v2", "hard_bench", "hard_dense_cluster",
    "latest_recent", "lattice", "mixed_cue", "negation_temporal",
    "notin_multi_interval", "open_ended_date", "polarity", "precedents",
    "realq", "realq_deictic", "realq_v2", "sensitivity_curated",
    "speculative_anchors", "temporal_essential", "timeless_policies",
    "utterance",
    "v7_compound_hard", "v7_doc_directional",
    "same_topic_recency",
    "same_topic_recency_hard",
    "recency_stress_deep",
    "recency_vs_rerank",
    "state_vs_event",
    "state_vs_event_v2",
]


def load_bench(bench: str):
    """Load (docs, queries, gold) for a bench, or (None, None, None) if missing."""
    try:
        with open(DATA_DIR / f"{bench}_docs.jsonl") as f:
            docs = [json.loads(line) for line in f]
        with open(DATA_DIR / f"{bench}_queries.jsonl") as f:
            queries = [json.loads(line) for line in f]
        with open(DATA_DIR / f"{bench}_gold.jsonl") as f:
            gold_rows = [json.loads(line) for line in f]
    except FileNotFoundError:
        return None, None, None
    gold = {g["query_id"]: set(g["relevant_doc_ids"]) for g in gold_rows}
    return docs, queries, gold


def make_cosine_rerank_fn(embed_fn):
    """Cosine-similarity reranker — fast, no cross-encoder."""
    async def rerank(query: str, doc_texts: list[str]) -> list[float]:
        if not doc_texts:
            return []
        qe = (await embed_fn([query]))[0]
        des = await embed_fn(doc_texts)
        qn = float(np.linalg.norm(qe)) or 1e-9
        out = []
        for de in des:
            dn = float(np.linalg.norm(de)) or 1e-9
            out.append(float(np.dot(qe, de) / (qn * dn)))
        return out
    return rerank


# Disk-cached embed wrapper — cuts cost of parameter sweeps to near zero
# once the cache is warm. text-embedding-3-small is deterministic so cache
# by SHA-256(text) is safe.
_EMBED_CACHE_DIR = EVAL_ROOT / "temporal_retrieval_tr" / "cache" / "embed"


def make_cached_embed_fn(inner_embed_fn, cache_dir: Path | None = None):
    """Wrap an embed function with a disk-backed text→vector cache.

    Reads on construction; writes the in-memory cache to disk at the
    end of each .save() call (and on shutdown via atexit).
    """
    cdir = Path(cache_dir or _EMBED_CACHE_DIR)
    cdir.mkdir(parents=True, exist_ok=True)
    cache_file = cdir / "embed_cache.pkl"
    cache: dict[str, np.ndarray] = {}
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                cache = pickle.load(f)
        except Exception:
            cache = {}

    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def save() -> None:
        tmp = cache_file.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(cache, f)
        tmp.replace(cache_file)

    async def cached(texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        keys = [_key(t) for t in texts]
        out: list[np.ndarray | None] = [cache.get(k) for k in keys]
        miss_idx = [i for i, v in enumerate(out) if v is None]
        if miss_idx:
            miss_texts = [texts[i] for i in miss_idx]
            new_vecs = await inner_embed_fn(miss_texts)
            for i, v in zip(miss_idx, new_vecs, strict=False):
                out[i] = v
                cache[keys[i]] = v
        return out  # type: ignore[return-value]

    cached.save = save  # type: ignore[attr-defined]
    cached.cache_size = lambda: len(cache)  # type: ignore[attr-defined]

    import atexit
    atexit.register(save)
    return cached


def metrics(rankings: dict, gold: dict, k_r5: int = 5) -> dict:
    """Compute R@1, R@5, R@10, and all_R@5 over a bench's rankings."""
    n_r1 = n_r5 = n_r10 = n_eval = 0
    all_r5 = []
    for qid, ranking in rankings.items():
        gs = gold.get(qid, set())
        if not gs:
            continue
        n_eval += 1
        first = next((i + 1 for i, d in enumerate(ranking) if d in gs), None)
        if first is not None:
            if first <= 1:
                n_r1 += 1
            if first <= k_r5:
                n_r5 += 1
            if first <= 10:
                n_r10 += 1
        top = set(ranking[:k_r5])
        all_r5.append(len(top & gs) / len(gs))
    return {
        "n": n_eval,
        "R@1": n_r1 / max(1, n_eval),
        "R@5": n_r5 / max(1, n_eval),
        "R@10": n_r10 / max(1, n_eval),
        "all_R@5": sum(all_r5) / max(1, len(all_r5)),
    }
