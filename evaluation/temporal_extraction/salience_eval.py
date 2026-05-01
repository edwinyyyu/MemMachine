"""End-to-end salience evaluation.

Compares three retrieval variants on 4 benchmarks:
  - SEMANTIC-ONLY  (text-embedding-3-small cosine)
  - V7 baseline    (T+S, score_blend with cv_ref=0.20, no salience)
  - V7+salience    (T+S, score_blend_with_salience, cv_ref=0.20)

Benchmarks:
  - hard bench       (600 docs / 75 queries; per-tier breakdown)
  - mixed-cue stress (200 docs / 40 queries; per-cue-type breakdown)
  - real_benchmark_small (TempReason small: 139 docs / 60 queries)
  - dense cluster    (100 docs / 30 queries)

For T-channel (interval+axis+tag) we reuse the existing v2 extractor
caches under cache/<bench>_v2_{docs,queries}/. For benchmarks that don't
have one, we extract on the fly with v2 + reasoning_effort=minimal patch.

Salience is extracted via gpt-5-mini reasoning_effort=minimal and cached
under cache/salience/llm_cache.json.

Outputs: results/per_doc_salience.md  +  results/salience_eval.json
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ---- patch: gpt-5-mini reasoning_effort=minimal, mirrors hard_benchmark_pipeline ----
import extractor_common  # noqa: E402

_orig_call = extractor_common.BaseImprovedExtractor._call


async def _patched_call(self, *args, **kwargs):
    original_create = self.client.chat.completions.create

    async def patched_create(**call_kwargs):
        model = call_kwargs.get("model", "")
        if isinstance(model, str) and model.startswith("gpt-5"):
            call_kwargs["reasoning_effort"] = "minimal"
        return await original_create(**call_kwargs)

    self.client.chat.completions.create = patched_create
    try:
        return await _orig_call(self, *args, **kwargs)
    finally:
        self.client.chat.completions.create = original_create


extractor_common.BaseImprovedExtractor._call = _patched_call


from axis_distributions import (
    AXES,
    AxisDistribution,
    axes_for_expression,
    merge_axis_dists,
)
from baselines import embed_all  # noqa: E402
from expander import expand  # noqa: E402
from extractor_v2 import ExtractorV2  # noqa: E402
from multi_axis_scorer import axis_score as axis_score_fn  # noqa: E402
from multi_axis_scorer import tag_score
from multi_axis_tags import tags_for_axes  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402
from rag_fusion import score_blend  # noqa: E402
from rag_fusion_salience import (
    score_blend_with_salience,
    score_blend_with_salience_post,
)
from salience_extractor import SalienceExtractor  # noqa: E402
from schema import GRANULARITY_ORDER, TimeExpression, parse_iso, to_us  # noqa: E402
from scorer import Interval, score_jaccard_composite  # noqa: E402

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PER_CALL_TIMEOUT_S = 60.0
CONCURRENCY = 6
SALIENCE_CONCURRENCY = 12

PRICE_IN_PER_M = 0.25
PRICE_OUT_PER_M = 2.00


# ---------------------------------------------------------------------------
# Helpers (copied from hard_benchmark_pipeline — keep self-contained)
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def flatten_intervals(te: TimeExpression) -> list[Interval]:
    out: list[Interval] = []
    if te.kind == "instant" and te.instant:
        out.append(
            Interval(
                earliest_us=to_us(te.instant.earliest),
                latest_us=to_us(te.instant.latest),
                best_us=to_us(te.instant.best) if te.instant.best else None,
                granularity=te.instant.granularity,
            )
        )
    elif te.kind == "interval" and te.interval:
        g = (
            te.interval.start.granularity
            if GRANULARITY_ORDER.get(te.interval.start.granularity, 0)
            >= GRANULARITY_ORDER.get(te.interval.end.granularity, 0)
            else te.interval.end.granularity
        )
        best = te.interval.start.best or te.interval.start.earliest
        out.append(
            Interval(
                earliest_us=to_us(te.interval.start.earliest),
                latest_us=to_us(te.interval.end.latest),
                best_us=to_us(best),
                granularity=g,
            )
        )
    elif te.kind == "recurrence" and te.recurrence:
        now = datetime.now(tz=timezone.utc)
        anchor = te.recurrence.dtstart.best or te.recurrence.dtstart.earliest
        start = min(now - timedelta(days=365 * 5), anchor - timedelta(days=365))
        end = now + timedelta(days=365 * 2)
        if te.recurrence.until is not None:
            end = min(end, te.recurrence.until.latest or te.recurrence.until.earliest)
        try:
            for inst in expand(te.recurrence, start, end):
                out.append(
                    Interval(
                        earliest_us=to_us(inst.earliest),
                        latest_us=to_us(inst.latest),
                        best_us=to_us(inst.best) if inst.best else None,
                        granularity=inst.granularity,
                    )
                )
        except Exception:
            pass
    return out


def interval_pair_best(q_ivs, d_ivs):
    if not q_ivs or not d_ivs:
        return 0.0
    total = 0.0
    for qi in q_ivs:
        best = 0.0
        for si in d_ivs:
            s = score_jaccard_composite(qi, si)
            if s > best:
                best = s
        total += best
    return total


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def recall_at_k(ranked, relevant, k):
    if not relevant:
        return float("nan")
    return len(set(ranked[:k]) & relevant) / len(relevant)


def mrr(ranked, relevant):
    if not relevant:
        return float("nan")
    for i, d in enumerate(ranked, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked, relevant, k):
    if not relevant:
        return float("nan")
    dcg = 0.0
    for i, d in enumerate(ranked[:k], start=1):
        if d in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def nanmean(xs):
    vs = [v for v in xs if not math.isnan(v)]
    return sum(vs) / len(vs) if vs else float("nan")


def eval_rankings(ranked_per_q, gold, qids):
    r1, r3, r5, mr, nd = [], [], [], [], []
    for qid in qids:
        rel = gold.get(qid, set())
        if not rel:
            continue
        ranked = ranked_per_q.get(qid, [])
        r1.append(recall_at_k(ranked, rel, 1))
        r3.append(recall_at_k(ranked, rel, 3))
        r5.append(recall_at_k(ranked, rel, 5))
        mr.append(mrr(ranked, rel))
        nd.append(ndcg_at_k(ranked, rel, 10))
    return {
        "n": len(r5),
        "recall@1": nanmean(r1),
        "recall@3": nanmean(r3),
        "recall@5": nanmean(r5),
        "mrr": nanmean(mr),
        "ndcg@10": nanmean(nd),
    }


def query_rank_of_gold(ranked, relevant) -> int | None:
    for i, d in enumerate(ranked, start=1):
        if d in relevant:
            return i
    return None


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
async def run_v2_extract(items, label: str, cache_subdir: str):
    ex = ExtractorV2(concurrency=CONCURRENCY, cache_subdir=cache_subdir)
    ex.client = AsyncOpenAI(timeout=PER_CALL_TIMEOUT_S, max_retries=1)
    results: dict[str, list[TimeExpression]] = {}
    n_timeout = 0
    n_error = 0
    completed = [0]
    total = len(items)

    async def one(iid, text, ref):
        nonlocal n_timeout, n_error
        try:
            tes = await asyncio.wait_for(
                ex.extract(text, ref), timeout=PER_CALL_TIMEOUT_S * 3
            )
        except asyncio.TimeoutError:
            n_timeout += 1
            tes = []
        except Exception:
            n_error += 1
            tes = []
        completed[0] += 1
        if completed[0] % 100 == 0:
            print(f"  [{label}] {completed[0]}/{total}", flush=True)
        return iid, tes

    pairs = await asyncio.gather(*(one(*it) for it in items))
    for iid, tes in pairs:
        results[iid] = tes
    ex.cache.save()
    try:
        ex.shared_pass2_cache.save()
    except Exception:
        pass
    return results


def build_memory(extracted):
    out: dict[str, dict[str, Any]] = {}
    for did, tes in extracted.items():
        intervals: list[Interval] = []
        axes_per: list[dict[str, AxisDistribution]] = []
        multi_tags: set[str] = set()
        for te in tes:
            intervals.extend(flatten_intervals(te))
            ax = axes_for_expression(te)
            axes_per.append(ax)
            multi_tags |= tags_for_axes(ax)
        axes_merged = merge_axis_dists(axes_per)
        out[did] = {
            "intervals": intervals,
            "axes_merged": axes_merged,
            "multi_tags": multi_tags,
        }
    return out


def rank_t(q_mem, doc_mem, alpha=0.5, beta=0.35, gamma=0.15):
    qa = q_mem.get("axes_merged") or {}
    q_tags = q_mem.get("multi_tags") or set()
    q_ivs = q_mem.get("intervals") or []
    raw_iv: dict[str, float] = {}
    for did, bundle in doc_mem.items():
        raw_iv[did] = interval_pair_best(q_ivs, bundle["intervals"])
    max_iv = max(raw_iv.values()) if raw_iv else 0.0
    scores: dict[str, float] = {}
    for did, bundle in doc_mem.items():
        iv_norm = raw_iv[did] / max_iv if max_iv > 0 else 0.0
        a_sc = axis_score_fn(qa, bundle["axes_merged"])
        t_sc = tag_score(q_tags, bundle["multi_tags"])
        scores[did] = alpha * iv_norm + beta * a_sc + gamma * t_sc
    return scores


def rank_semantic(qid, q_embs, doc_embs):
    qv = q_embs[qid]
    qn = np.linalg.norm(qv) or 1e-9
    out: dict[str, float] = {}
    for d, v in doc_embs.items():
        vn = np.linalg.norm(v) or 1e-9
        out[d] = float(np.dot(qv, v) / (qn * vn))
    return out


def rank_v7(t, s, weights=None, cv_ref=0.20):
    if weights is None:
        weights = {"T": 0.5, "S": 0.5}
    fused = score_blend(
        {"T": t, "S": s}, weights, top_k_per=40, dispersion_cv_ref=cv_ref
    )
    ranked = [d for d, _ in fused]
    seen = set(ranked)
    tail = [
        d
        for d, _ in sorted(s.items(), key=lambda x: x[1], reverse=True)
        if d not in seen
    ]
    return ranked + tail


def rank_v7_salience(t, s, doc_salience, weights=None, cv_ref=0.20):
    if weights is None:
        weights = {"T": 0.5, "S": 0.5}
    fused = score_blend_with_salience(
        {"T": t, "S": s},
        weights,
        doc_salience,
        channel_to_key={"T": "T", "S": "S"},
        salience_floor=0.05,
        salience_temperature=1.0,
        top_k_per=40,
        dispersion_cv_ref=cv_ref,
    )
    ranked = [d for d, _ in fused]
    seen = set(ranked)
    tail = [
        d
        for d, _ in sorted(s.items(), key=lambda x: x[1], reverse=True)
        if d not in seen
    ]
    return ranked + tail


def rank_v7_salience_post(t, s, doc_salience, weights=None, cv_ref=0.20):
    if weights is None:
        weights = {"T": 0.5, "S": 0.5}
    fused = score_blend_with_salience_post(
        {"T": t, "S": s},
        weights,
        doc_salience,
        channel_to_key={"T": "T", "S": "S"},
        salience_floor=0.05,
        salience_temperature=1.0,
        top_k_per=40,
        dispersion_cv_ref=cv_ref,
    )
    ranked = [d for d, _ in fused]
    seen = set(ranked)
    tail = [
        d
        for d, _ in sorted(s.items(), key=lambda x: x[1], reverse=True)
        if d not in seen
    ]
    return ranked + tail


# ---------------------------------------------------------------------------
# Run a benchmark
# ---------------------------------------------------------------------------
async def run_benchmark(
    name: str,
    docs: list[dict],
    queries: list[dict],
    gold: dict[str, set[str]],
    cache_doc: str,
    cache_q: str,
    *,
    salience_extractor: SalienceExtractor,
    extra_subset_key: str | None = None,
):
    print(f"\n=== {name}: {len(docs)} docs, {len(queries)} queries ===", flush=True)
    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs]
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries]

    print("[T] extracting docs...", flush=True)
    doc_ext = await run_v2_extract(doc_items, f"{name}-docs", cache_doc)
    print("[T] extracting queries...", flush=True)
    q_ext = await run_v2_extract(q_items, f"{name}-queries", cache_q)

    print("[T] building memory...", flush=True)
    doc_mem = build_memory(doc_ext)
    q_mem = build_memory(q_ext)
    for d in docs:
        doc_mem.setdefault(
            d["doc_id"],
            {
                "intervals": [],
                "axes_merged": {
                    a: AxisDistribution(axis=a, values={}, informative=False)
                    for a in AXES
                },
                "multi_tags": set(),
            },
        )

    print("[S] embedding...", flush=True)
    doc_texts = [d["text"] for d in docs]
    q_texts = [q["text"] for q in queries]
    doc_embs_arr = await embed_all(doc_texts)
    q_embs_arr = await embed_all(q_texts)
    doc_embs = {d["doc_id"]: doc_embs_arr[i] for i, d in enumerate(docs)}
    q_embs = {q["query_id"]: q_embs_arr[i] for i, q in enumerate(queries)}

    print("[Sal] extracting per-doc salience...", flush=True)
    doc_salience = await salience_extractor.extract_many(
        [(d["doc_id"], d["text"]) for d in docs], progress_every=100
    )
    salience_extractor.cache.save()

    # --- Per-query scores ---
    per_q_t: dict[str, dict[str, float]] = {}
    per_q_s: dict[str, dict[str, float]] = {}
    for q in queries:
        qid = q["query_id"]
        per_q_t[qid] = rank_t(
            q_mem.get(qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()}),
            doc_mem,
        )
        per_q_s[qid] = rank_semantic(qid, q_embs, doc_embs)

    # --- Variants ---
    variants: dict[str, dict[str, list[str]]] = {}
    sem_v: dict[str, list[str]] = {}
    for q in queries:
        qid = q["query_id"]
        sem_v[qid] = [
            d for d, _ in sorted(per_q_s[qid].items(), key=lambda x: x[1], reverse=True)
        ]
    variants["SEMANTIC"] = sem_v

    v7_v: dict[str, list[str]] = {}
    for q in queries:
        qid = q["query_id"]
        v7_v[qid] = rank_v7(per_q_t[qid], per_q_s[qid], cv_ref=0.20)
    variants["V7"] = v7_v

    v7_sal_v: dict[str, list[str]] = {}
    v7_sal_post_v: dict[str, list[str]] = {}
    for q in queries:
        qid = q["query_id"]
        v7_sal_v[qid] = rank_v7_salience(
            per_q_t[qid], per_q_s[qid], doc_salience, cv_ref=0.20
        )
        v7_sal_post_v[qid] = rank_v7_salience_post(
            per_q_t[qid], per_q_s[qid], doc_salience, cv_ref=0.20
        )
    variants["V7+salience"] = v7_sal_v
    variants["V7+salience-post"] = v7_sal_post_v

    # --- Subsets ---
    all_qids = {q["query_id"] for q in queries}
    subsets = {"all": all_qids}
    if extra_subset_key:
        groups: dict[str, set[str]] = defaultdict(set)
        for q in queries:
            sub = q.get(extra_subset_key)
            if sub:
                groups[sub].add(q["query_id"])
        for k, v in groups.items():
            subsets[k] = v

    per_variant: dict[str, dict[str, dict[str, float]]] = {}
    for var, ranked_per_q in variants.items():
        per_variant[var] = {}
        for sub_name, qids in subsets.items():
            per_variant[var][sub_name] = eval_rankings(ranked_per_q, gold, qids)

    # --- Salience distribution stats ---
    sal_stats = {
        "avg": {"S": 0.0, "T": 0.0, "L": 0.0, "E": 0.0},
        "n": len(doc_salience),
    }
    if doc_salience:
        for k in ("S", "T", "L", "E"):
            sal_stats["avg"][k] = sum(v[k] for v in doc_salience.values()) / len(
                doc_salience
            )

    # Per-cue salience averages if cue_type present
    sal_by_cue: dict[str, dict[str, float]] = {}
    if any("cue_type" in d for d in docs):
        for cue in ("date", "content", "recurrence", "era"):
            cue_docs = [d["doc_id"] for d in docs if d.get("cue_type") == cue]
            if not cue_docs:
                continue
            avg = dict.fromkeys(("S", "T", "L", "E"), 0.0)
            for did in cue_docs:
                v = doc_salience.get(did, {})
                for k in avg:
                    avg[k] += v.get(k, 0.0)
            for k in avg:
                avg[k] /= max(1, len(cue_docs))
            sal_by_cue[cue] = avg

    return {
        "name": name,
        "n_docs": len(docs),
        "n_queries": len(queries),
        "per_variant": per_variant,
        "subsets": {k: len(v) for k, v in subsets.items()},
        "salience_avg": sal_stats,
        "salience_by_cue": sal_by_cue,
        "doc_salience": doc_salience,
        "variants_ranked": variants,
        "extracted_n": {
            "doc_avg_tes": sum(len(v) for v in doc_ext.values()) / max(1, len(doc_ext)),
            "q_avg_tes": sum(len(v) for v in q_ext.values()) / max(1, len(q_ext)),
        },
    }


# ---------------------------------------------------------------------------
# Salience sanity check
# ---------------------------------------------------------------------------
async def sanity_check_salience(extractor: SalienceExtractor):
    samples = [
        "On March 15, 2024, I had dinner with Sarah at Gusto.",
        "I love hiking in the mountains.",
        "Every Thursday I do tennis lessons at 6pm.",
        "Back in the 90s we used to spend summers in Maine.",
        "Yesterday I picked up the package from the post office.",
        "Vincent Ostrom works for Indiana University Bloomington from Jan, 1964 to Jan, 1990.",
        "What if I had been born in 1980? How different things would be.",
    ]
    out = []
    for s in samples:
        v = await extractor.extract_one(s)
        out.append({"text": s, "salience": v})
    extractor.cache.save()
    return out


# ---------------------------------------------------------------------------
# Failure analysis on hard medium-tier
# ---------------------------------------------------------------------------
def analyze_hard_medium(hard_result, gold, queries):
    """For 5 medium queries V7 lost (rank > 1) — did salience fix any?"""
    medium_qs = [q for q in queries if q.get("subset") == "medium"]
    out = []
    for q in medium_qs:
        qid = q["query_id"]
        rel = gold.get(qid, set())
        if not rel:
            continue
        v7_rank = query_rank_of_gold(hard_result["variants_ranked"]["V7"][qid], rel)
        v7_sal_rank = query_rank_of_gold(
            hard_result["variants_ranked"]["V7+salience"][qid], rel
        )
        if v7_rank is not None and v7_rank > 1:
            out.append(
                {
                    "qid": qid,
                    "query": q["text"],
                    "rank_v7": v7_rank,
                    "rank_v7_salience": v7_sal_rank,
                    "delta": (v7_rank - v7_sal_rank)
                    if v7_sal_rank is not None
                    else None,
                }
            )
    out.sort(key=lambda x: x["rank_v7"])
    return out[:5]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    t0 = time.time()
    sal_ex = SalienceExtractor(concurrency=SALIENCE_CONCURRENCY)

    # ---- Sanity check ----
    print("=== Salience sanity check ===", flush=True)
    sanity = await sanity_check_salience(sal_ex)
    for r in sanity:
        v = r["salience"]
        print(
            f"  {r['text'][:60]:<60} S={v['S']:.2f} T={v['T']:.2f} L={v['L']:.2f} E={v['E']:.2f}",
            flush=True,
        )

    all_results: dict[str, Any] = {"sanity": sanity, "benchmarks": {}}

    # ---- 1. Mixed-cue (smallest, run first) ----
    print("\n>>> Mixed-cue benchmark <<<", flush=True)
    docs = load_jsonl(DATA_DIR / "mixed_cue_docs.jsonl")
    queries = load_jsonl(DATA_DIR / "mixed_cue_queries.jsonl")
    gold_raw = load_jsonl(DATA_DIR / "mixed_cue_gold.jsonl")
    gold = {g["query_id"]: set(g["relevant_doc_ids"]) for g in gold_raw}
    res = await run_benchmark(
        "mixed_cue",
        docs,
        queries,
        gold,
        cache_doc="mixed_cue_v2_docs",
        cache_q="mixed_cue_v2_queries",
        salience_extractor=sal_ex,
        extra_subset_key="cue_type",
    )
    all_results["benchmarks"]["mixed_cue"] = res

    if time.time() - t0 > 25 * 60:
        print("WALL CAP — bailing before further benchmarks", flush=True)
        await _save_partial(all_results, t0)
        return

    # ---- 2. Dense cluster ----
    print("\n>>> Dense cluster benchmark <<<", flush=True)
    docs = load_jsonl(DATA_DIR / "dense_cluster_docs.jsonl")
    queries = load_jsonl(DATA_DIR / "dense_cluster_queries.jsonl")
    gold_raw = load_jsonl(DATA_DIR / "dense_cluster_gold.jsonl")
    gold = {g["query_id"]: set(g["relevant_doc_ids"]) for g in gold_raw}
    res = await run_benchmark(
        "dense_cluster",
        docs,
        queries,
        gold,
        cache_doc="dense_cluster_v2_docs",
        cache_q="dense_cluster_v2_queries",
        salience_extractor=sal_ex,
    )
    all_results["benchmarks"]["dense_cluster"] = res

    if time.time() - t0 > 25 * 60:
        print("WALL CAP — bailing", flush=True)
        await _save_partial(all_results, t0)
        return

    # ---- 3. real_benchmark_small (TempReason small) ----
    print("\n>>> real_benchmark_small (TempReason) <<<", flush=True)
    docs = load_jsonl(DATA_DIR / "real_benchmark_small_docs.jsonl")
    queries = load_jsonl(DATA_DIR / "real_benchmark_small_queries.jsonl")
    gold_raw = load_jsonl(DATA_DIR / "real_benchmark_small_gold.jsonl")
    gold = {g["query_id"]: set(g["relevant_doc_ids"]) for g in gold_raw}
    res = await run_benchmark(
        "tempreason_small",
        docs,
        queries,
        gold,
        cache_doc="real_benchmark_small_v2_docs",
        cache_q="real_benchmark_small_v2_queries",
        salience_extractor=sal_ex,
    )
    all_results["benchmarks"]["tempreason_small"] = res

    if time.time() - t0 > 25 * 60:
        print("WALL CAP — bailing", flush=True)
        await _save_partial(all_results, t0)
        return

    # ---- 4. Hard bench (largest, run last) ----
    print("\n>>> Hard benchmark <<<", flush=True)
    docs = load_jsonl(DATA_DIR / "hard_bench_docs.jsonl")
    queries = load_jsonl(DATA_DIR / "hard_bench_queries.jsonl")
    gold_raw = load_jsonl(DATA_DIR / "hard_bench_gold.jsonl")
    gold = {g["query_id"]: set(g["relevant_doc_ids"]) for g in gold_raw}
    res = await run_benchmark(
        "hard_bench",
        docs,
        queries,
        gold,
        cache_doc="hard_bench_v2_docs",
        cache_q="hard_bench_v2_queries",
        salience_extractor=sal_ex,
        extra_subset_key="subset",
    )
    # Failure analysis
    medium_failures = analyze_hard_medium(res, gold, queries)
    res["medium_failure_analysis"] = medium_failures
    all_results["benchmarks"]["hard_bench"] = res

    await _save_partial(all_results, t0)


async def _save_partial(all_results, t0):
    wall = time.time() - t0

    # Print summary table
    print("\n\n=== SUMMARY (R@1 / MRR per benchmark) ===", flush=True)
    print(
        f"{'Benchmark':<22} {'Variant':<14} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'MRR':>6} {'NDCG':>6}",
        flush=True,
    )
    for bname, br in all_results["benchmarks"].items():
        for var in ("SEMANTIC", "V7", "V7+salience"):
            m = br["per_variant"].get(var, {}).get("all", {})
            if not m:
                continue
            print(
                f"{bname:<22} {var:<14} {m.get('recall@1', 0):>6.3f} "
                f"{m.get('recall@3', 0):>6.3f} {m.get('recall@5', 0):>6.3f} "
                f"{m.get('mrr', 0):>6.3f} {m.get('ndcg@10', 0):>6.3f}",
                flush=True,
            )

    # Save JSON (strip non-serializable)
    def _strip(obj):
        if isinstance(obj, dict):
            return {
                k: _strip(v) for k, v in obj.items() if k not in ("variants_ranked",)
            }
        if isinstance(obj, list):
            return [_strip(v) for v in obj]
        if isinstance(obj, set):
            return sorted(obj)
        if isinstance(obj, float) and math.isnan(obj):
            return None
        return obj

    out = _strip(all_results)
    out["wall_seconds"] = wall
    json_path = RESULTS_DIR / "salience_eval.json"
    json_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote {json_path}", flush=True)

    # Write markdown report
    write_report(all_results, wall)


def write_report(all_results, wall_s):
    lines = []
    lines.append("# Per-Doc Cue Salience: Eval Report\n")
    lines.append(f"_Total wall: {wall_s:.0f}s_\n")
    lines.append("")
    lines.append("## 1. Salience extractor sanity check\n")
    lines.append("Sample inputs and resulting salience vectors (S/T/L/E):\n")
    lines.append("| Text | S | T | L | E |")
    lines.append("|---|---|---|---|---|")
    for r in all_results.get("sanity", []):
        v = r["salience"]
        text = r["text"].replace("|", "\\|")
        lines.append(
            f"| {text} | {v['S']:.2f} | {v['T']:.2f} | {v['L']:.2f} | {v['E']:.2f} |"
        )
    lines.append("")

    # Salience distribution
    lines.append("## 2. Salience distribution per benchmark\n")
    lines.append("Average per-channel salience over docs:\n")
    lines.append("| Benchmark | n | S | T | L | E |")
    lines.append("|---|---|---|---|---|---|")
    for bname, br in all_results["benchmarks"].items():
        avg = br["salience_avg"]["avg"]
        n = br["salience_avg"]["n"]
        lines.append(
            f"| {bname} | {n} | {avg['S']:.2f} | {avg['T']:.2f} | "
            f"{avg['L']:.2f} | {avg['E']:.2f} |"
        )
    lines.append("")

    # Mixed-cue per cue-type breakdown
    if "mixed_cue" in all_results["benchmarks"]:
        mc = all_results["benchmarks"]["mixed_cue"]
        lines.append("## 3. Mixed-cue: per-cue-type salience averages\n")
        lines.append("| Cue type | S | T | L | E |")
        lines.append("|---|---|---|---|---|")
        for cue, avg in mc["salience_by_cue"].items():
            lines.append(
                f"| {cue} | {avg['S']:.2f} | {avg['T']:.2f} | "
                f"{avg['L']:.2f} | {avg['E']:.2f} |"
            )
        lines.append("")

    # Per-benchmark, per-variant metrics
    lines.append("## 4. Per-benchmark variant metrics\n")
    for bname, br in all_results["benchmarks"].items():
        lines.append(f"### {bname}\n")
        lines.append("| Variant | subset | n | R@1 | R@3 | R@5 | MRR | NDCG@10 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for var in ("SEMANTIC", "V7", "V7+salience"):
            for sub in br["per_variant"].get(var, {}):
                m = br["per_variant"][var][sub]
                lines.append(
                    f"| {var} | {sub} | {m['n']} | "
                    f"{m['recall@1']:.3f} | {m['recall@3']:.3f} | "
                    f"{m['recall@5']:.3f} | {m['mrr']:.3f} | "
                    f"{m['ndcg@10']:.3f} |"
                )
        lines.append("")

    # Hard-medium failure analysis
    if (
        "hard_bench" in all_results["benchmarks"]
        and "medium_failure_analysis" in all_results["benchmarks"]["hard_bench"]
    ):
        lines.append("## 5. Hard-medium failure analysis\n")
        lines.append(
            "V7 medium-tier queries that V7 ranked > 1 and whether salience helped:\n"
        )
        lines.append("| qid | query | rank_V7 | rank_V7+sal | delta |")
        lines.append("|---|---|---|---|---|")
        for fa in all_results["benchmarks"]["hard_bench"]["medium_failure_analysis"]:
            q = fa["query"].replace("|", "\\|")
            d = fa["delta"]
            d_s = f"{d:+d}" if d is not None else "n/a"
            sal_r = (
                fa["rank_v7_salience"] if fa["rank_v7_salience"] is not None else "n/a"
            )
            lines.append(f"| {fa['qid']} | {q} | {fa['rank_v7']} | {sal_r} | {d_s} |")
        lines.append("")

    # Verdict
    lines.append("## 6. Verdict\n")
    lines.append("See report body — comparison V7 vs V7+salience per benchmark.\n")
    lines.append("")
    lines.append("## 7. Cost\n")
    lines.append(
        "Salience extraction: gpt-5-mini reasoning_effort=minimal. "
        "~1k docs total at ~$0.0015/doc = ~$1.50 one-time at ingest. "
        "Retrieval cost unchanged (pure local arithmetic).\n"
    )

    out_path = RESULTS_DIR / "per_doc_salience.md"
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
