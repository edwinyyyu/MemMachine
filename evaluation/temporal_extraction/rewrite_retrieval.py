"""F11 — Retrieval via query-rewrite + fusion.

Given a list of (original + K rewrite variants), extract time expressions
on each variant using the base Extractor, run referent-only retrieval on
a shared IntervalStore, and fuse candidate lists.

Three modes:
- "original_only": baseline. No rewriting. Extract + retrieve just the
  original query.
- "rrf": Reciprocal-Rank Fusion across variants. Σ 1/(60 + rank_i).
- "max": max of score across variants.

The extractor cache (``extractor.LLMCache``) handles the per-variant
extraction cost: identical variant strings never re-hit the API.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from eval import flatten_query_intervals  # reuse existing helper
from extractor import Extractor, LLMCache
from schema import TimeExpression
from scorer import Interval, score_pair
from store import IntervalStore

# Hard per-variant extraction timeout (two passes happen inside ex.extract).
VARIANT_EXTRACT_TIMEOUT_SEC = 30.0

ROOT = Path(__file__).resolve().parent
REWRITE_EXTRACT_CACHE = ROOT / "cache" / "rewrite" / "variant_llm_cache.json"


def _score_referents_sum_per_doc(
    store: IntervalStore, q_intervals: list[Interval]
) -> dict[str, float]:
    """Mirror of eval.temporal_retrieve — sum over q-intervals of best-per-doc."""
    out: dict[str, float] = defaultdict(float)
    for qi in q_intervals:
        rows = store.query_overlap(qi.earliest_us, qi.latest_us)
        best_per_doc: dict[str, float] = {}
        for _expr_id, doc_id, e_us, l_us, b_us, gran in rows:
            s = Interval(
                earliest_us=e_us, latest_us=l_us, best_us=b_us, granularity=gran
            )
            sc = score_pair(qi, s)
            if sc > best_per_doc.get(doc_id, 0.0):
                best_per_doc[doc_id] = sc
        for d, sc in best_per_doc.items():
            out[d] += sc
    return dict(out)


def rank_from_scores(scores: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda x: -x[1])


def rrf_fuse(
    ranked_lists: list[list[tuple[str, float]]],
    all_doc_ids: Iterable[str],
    k: int = 60,
) -> dict[str, float]:
    """Reciprocal-rank fusion. Only docs with score>0 in a list contribute
    for that list. Docs absent from a list get 0 contribution from it.
    """
    fused: dict[str, float] = defaultdict(float)
    for rl in ranked_lists:
        # Only rank docs with score > 0 for this variant. If rl is all
        # zeros (no matches), skip.
        pos = [d for d, s in rl if s > 0]
        for rank, d in enumerate(pos, start=1):
            fused[d] += 1.0 / (k + rank)
    # Ensure every candidate doc appears (score 0) so downstream can read
    # stable orderings.
    for d in all_doc_ids:
        if d not in fused:
            fused[d] = 0.0
    return dict(fused)


def max_fuse(
    ranked_lists: list[list[tuple[str, float]]],
) -> dict[str, float]:
    """Per-doc max score across variant lists."""
    fused: dict[str, float] = defaultdict(float)
    for rl in ranked_lists:
        for d, s in rl:
            if s > fused[d]:
                fused[d] = s
    return dict(fused)


async def retrieve_variant(
    ex: Extractor,
    store: IntervalStore,
    variant_text: str,
    ref_time: datetime,
) -> tuple[dict[str, float], list[TimeExpression]]:
    """Extract a single variant and run referent-only retrieval."""
    try:
        tes = await asyncio.wait_for(
            ex.extract(variant_text, ref_time),
            timeout=VARIANT_EXTRACT_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        print(f"    extract TIMEOUT for variant {variant_text!r}")
        tes = []
    except Exception as e:
        print(f"    extract failed for variant {variant_text!r}: {e}")
        tes = []
    q_ivs: list[Interval] = []
    for te in tes:
        q_ivs.extend(flatten_query_intervals(te))
    scores = _score_referents_sum_per_doc(store, q_ivs)
    return scores, tes


async def retrieve_with_rewrites(
    ex: Extractor,
    store: IntervalStore,
    original_text: str,
    variants: list[str],
    ref_time: datetime,
    all_doc_ids: Iterable[str],
    fuse_mode: str = "rrf",
) -> tuple[
    dict[str, float], dict[str, dict[str, float]], dict[str, list[TimeExpression]]
]:
    """Run retrieval over [original, *variants] and fuse.

    Returns (fused_scores, per_variant_scores, per_variant_tes).
    ``per_variant_scores`` is keyed by variant-text for debug/analysis.
    """
    # Ensure original is first; dedupe against variants.
    texts: list[str] = [original_text]
    seen = {original_text.strip().lower()}
    for v in variants:
        if v.strip().lower() in seen:
            continue
        seen.add(v.strip().lower())
        texts.append(v)

    # Parallelise variant extraction — the extractor's internal semaphore
    # (concurrency=10) bounds the global API rate.
    results = await asyncio.gather(
        *(retrieve_variant(ex, store, t, ref_time) for t in texts)
    )
    per_scores: dict[str, dict[str, float]] = {}
    per_tes: dict[str, list[TimeExpression]] = {}
    ranked_lists: list[list[tuple[str, float]]] = []
    for t, (s, tes) in zip(texts, results):
        per_scores[t] = s
        per_tes[t] = tes
        ranked_lists.append(rank_from_scores(s))

    if fuse_mode == "rrf":
        fused = rrf_fuse(ranked_lists, all_doc_ids)
    elif fuse_mode == "max":
        fused = max_fuse(ranked_lists)
    elif fuse_mode == "original_only":
        fused = dict(per_scores[original_text])
    else:
        raise ValueError(f"unknown fuse_mode: {fuse_mode}")
    return fused, per_scores, per_tes


def rank_docs(scores: dict[str, float], all_doc_ids: Iterable[str]) -> list[str]:
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    ranked_ids = [d for d, _s in ranked]
    seen = set(ranked_ids)
    # Append any docs with no score so rankings cover the full pool.
    for d in all_doc_ids:
        if d not in seen:
            ranked_ids.append(d)
    return ranked_ids


def build_variant_extractor(
    cache_file: Path = REWRITE_EXTRACT_CACHE, concurrency: int = 20
) -> Extractor:
    """Extractor whose LLM cache is isolated from the shared base cache so
    that the base query-extraction cache is not mutated by variant calls.
    Concurrency is bumped beyond the default 10 since many variants will
    run in parallel under rewrite_eval.
    """
    ex = Extractor(concurrency=concurrency)
    # Preserve the base corpus cache as a READ fallback by copying its
    # contents into the variant cache file the first time: we point the
    # extractor at variant_cache but also warm it with base cache entries.
    base_cache_path = ROOT / "cache" / "llm_cache.json"
    if base_cache_path.exists() and not cache_file.exists():
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(base_cache_path.read_text())
    ex.cache = LLMCache(path=cache_file)
    return ex
