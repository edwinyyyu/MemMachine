"""Lattice retrieval evaluation.

Compares:
    V-LATTICE-ONLY  — pure lattice channel.
    V7 SCORE-BLEND  — baseline (T+S+A+E, 0.4/0.4/0.1/0.1).
    V7L             — T+S+A+E+L (0.3/0.3/0.1/0.1/0.2).

Subsets:
    lattice-synth    — primary test set (cross-precision + cyclical + S8).
    base-55          — regression check (existing base docs/queries).
    axis             — existing axis_synth queries/docs.
    adversarial      — v2'' adversarial subset (focus on A3, A6, S8).

All extraction uses v2'' (ExtractorV2PP). Reuses cached extractions from
cache/adversarial_v2pp/ for adversarial + existing caches for base/axis.
A new namespaced cache under cache/lattice/ is created for the lattice
synth corpus only.
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

from advanced_common import LLMCaller
from axis_distributions import (
    AXES,
    AxisDistribution,
    axes_for_expression,
    merge_axis_dists,
)
from baselines import embed_all
from era_extractor import EraExtractor
from expander import expand
from extractor_v2pp import ExtractorV2PP
from lattice_cells import tags_for_expression as lattice_tags_for_expression
from lattice_retrieval import retrieve_multi as lattice_retrieve_multi
from lattice_store import LatticeStore
from multi_axis_scorer import axis_score as axis_score_fn
from multi_axis_scorer import tag_score
from multi_axis_tags import tags_for_axes
from openai import AsyncOpenAI
from rag_fusion import score_blend
from schema import GRANULARITY_ORDER, TimeExpression, parse_iso, to_us
from scorer import Interval, score_jaccard_composite

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
LATTICE_CACHE_DIR = ROOT / "cache" / "lattice"
LATTICE_CACHE_DIR.mkdir(exist_ok=True, parents=True)

LATTICE_DB = LATTICE_CACHE_DIR / "lattice.sqlite"

TOP_K = 10
LLM_CALL_TIMEOUT_S = 30.0
CALL_TIMEOUT_S = 180.0
CONCURRENCY = 4


def _patched_client() -> AsyncOpenAI:
    return AsyncOpenAI(timeout=LLM_CALL_TIMEOUT_S, max_retries=1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# Shared utils
# ---------------------------------------------------------------------------
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


def interval_pair_best(q_ivs: list[Interval], d_ivs: list[Interval]) -> float:
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


def eval_rankings(
    ranked_per_q: dict[str, list[str]],
    gold: dict[str, set[str]],
    qids: set[str],
) -> dict[str, float]:
    r5, r10, mr, nd = [], [], [], []
    for qid in qids:
        rel = gold.get(qid, set())
        if not rel:
            continue
        ranked = ranked_per_q.get(qid, [])
        r5.append(recall_at_k(ranked, rel, 5))
        r10.append(recall_at_k(ranked, rel, 10))
        mr.append(mrr(ranked, rel))
        nd.append(ndcg_at_k(ranked, rel, TOP_K))
    return {
        "recall@5": nanmean(r5),
        "recall@10": nanmean(r10),
        "mrr": nanmean(mr),
        "ndcg@10": nanmean(nd),
        "n": len([v for v in r5 if not math.isnan(v)]),
    }


# ---------------------------------------------------------------------------
# Extractor runner
# ---------------------------------------------------------------------------
async def run_v2pp_extract(items, cache_file: Path, label: str):
    ex = ExtractorV2PP(concurrency=CONCURRENCY)
    from extractor_common import LLMCache

    ex.cache = LLMCache(cache_file)
    ex.client = _patched_client()

    results: dict[str, list[TimeExpression]] = {}

    async def one(iid, text, ref):
        try:
            tes = await asyncio.wait_for(ex.extract(text, ref), timeout=CALL_TIMEOUT_S)
            return iid, tes
        except asyncio.TimeoutError:
            print(f"  [{label}] v2pp TIMEOUT {iid}")
            return iid, []
        except Exception as e:
            print(f"  [{label}] v2pp FAIL {iid}: {e}")
            return iid, []

    print(
        f"v2pp extract {label}: {len(items)} items (cache={cache_file.parent.name}/{cache_file.name})"
    )
    pairs = await asyncio.gather(*(one(*it) for it in items))
    for iid, tes in pairs:
        results[iid] = tes
    ex.cache.save()
    try:
        ex.shared_pass2_cache.save()
    except Exception:
        pass
    print(f"  [{label}] usage: {ex.usage}")
    return results, ex.usage


async def run_era_extract(items, cache_dir: Path, label: str):
    llm = LLMCaller(concurrency=CONCURRENCY)
    llm.client = _patched_client()
    ex = EraExtractor(llm)
    results: dict[str, list[TimeExpression]] = {}

    async def one(iid, text, ref):
        try:
            tes = await asyncio.wait_for(ex.extract(text, ref), timeout=CALL_TIMEOUT_S)
            return iid, tes
        except asyncio.TimeoutError:
            return iid, []
        except Exception as e:
            print(f"  [{label}] era FAIL {iid}: {e}")
            return iid, []

    print(f"era extract {label}: {len(items)} items")
    pairs = await asyncio.gather(*(one(*it) for it in items))
    for iid, tes in pairs:
        results[iid] = tes
    llm.save()
    return results, llm.usage


# ---------------------------------------------------------------------------
# Build memory for T (multi-axis) + embedding
# ---------------------------------------------------------------------------
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


def rank_multi_axis_t(q_mem, doc_mem, alpha=0.5, beta=0.35, gamma=0.15):
    """Returns {doc_id -> T score}. Same formula as v2pp adversarial."""
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


def rank_semantic_s(qid: str, q_embs, doc_embs) -> dict[str, float]:
    qv = q_embs[qid]
    qn = np.linalg.norm(qv) or 1e-9
    out: dict[str, float] = {}
    for d, v in doc_embs.items():
        vn = np.linalg.norm(v) or 1e-9
        out[d] = float(np.dot(qv, v) / (qn * vn))
    return out


# ---------------------------------------------------------------------------
# Ingest lattice tags
# ---------------------------------------------------------------------------
def ingest_lattice(store: LatticeStore, extracted: dict[str, list[TimeExpression]]):
    store.clear()
    for did, tes in extracted.items():
        all_abs: list[tuple[str, str]] = []
        all_cyc: set[str] = set()
        for te in tes:
            ts = lattice_tags_for_expression(te)
            all_abs.extend(ts.absolute)
            all_cyc |= ts.cyclical
        # dedupe absolute tags within doc (keep finest precision when dupes)
        seen_tags: set[str] = set()
        dedup_abs: list[tuple[str, str]] = []
        for prec, t in all_abs:
            if t in seen_tags:
                continue
            seen_tags.add(t)
            dedup_abs.append((prec, t))
        store.insert(did, dedup_abs, all_cyc)


def retrieve_lattice_scores(
    store: LatticeStore,
    query_extracted: dict[str, list[TimeExpression]],
    all_qids: set[str],
) -> tuple[dict[str, dict[str, float]], dict[str, dict]]:
    per_q_scores: dict[str, dict[str, float]] = {}
    per_q_debug: dict[str, dict] = {}
    for qid in all_qids:
        tes = query_extracted.get(qid, [])
        if not tes:
            per_q_scores[qid] = {}
            per_q_debug[qid] = {"no_extraction": True}
            continue
        scores, debug = lattice_retrieve_multi(store, tes, down_levels=1)
        per_q_scores[qid] = scores
        per_q_debug[qid] = debug
    return per_q_scores, per_q_debug


# ---------------------------------------------------------------------------
# Fusion variants
# ---------------------------------------------------------------------------
def rank_v7_score_blend(t, s, a, e, weights=None) -> list[str]:
    if weights is None:
        weights = {"T": 0.4, "S": 0.4, "A": 0.1, "E": 0.1}
    per = {"T": t, "S": s, "A": a, "E": e}
    fused = score_blend(per, weights, top_k_per=40)
    ranked = [d for d, _ in fused]
    # tail with anything missed from S
    seen = set(ranked)
    tail = [
        d
        for d, _ in sorted(s.items(), key=lambda x: x[1], reverse=True)
        if d not in seen
    ]
    return ranked + tail


def rank_v7l_score_blend(t, s, a, e, l, weights=None) -> list[str]:
    if weights is None:
        weights = {"T": 0.3, "S": 0.3, "A": 0.1, "E": 0.1, "L": 0.2}
    per = {"T": t, "S": s, "A": a, "E": e, "L": l}
    fused = score_blend(per, weights, top_k_per=40)
    ranked = [d for d, _ in fused]
    seen = set(ranked)
    tail = [
        d
        for d, _ in sorted(s.items(), key=lambda x: x[1], reverse=True)
        if d not in seen
    ]
    return ranked + tail


def rank_lattice_only(
    l: dict[str, float], s: dict[str, float], all_ids: list[str]
) -> list[str]:
    # primary sort by L, tie-break by S
    ranked = sorted(all_ids, key=lambda d: (l.get(d, 0.0), s.get(d, 0.0)), reverse=True)
    return ranked


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    t0 = time.time()

    # Ensure lattice synth data exists
    if not (DATA_DIR / "lattice_docs.jsonl").exists():
        import lattice_synth

        lattice_synth.main()

    # ===== Load data =====
    lat_docs = load_jsonl(DATA_DIR / "lattice_docs.jsonl")
    lat_queries = load_jsonl(DATA_DIR / "lattice_queries.jsonl")
    lat_gold_raw = load_jsonl(DATA_DIR / "lattice_gold.jsonl")
    lat_gold = {g["query_id"]: set(g["relevant_doc_ids"]) for g in lat_gold_raw}
    lat_subset_of_q = {q["query_id"]: q["subset"] for q in lat_queries}

    base_docs = load_jsonl(DATA_DIR / "docs.jsonl")
    base_queries = load_jsonl(DATA_DIR / "queries.jsonl")
    base_gold = {
        g["query_id"]: set(g["relevant_doc_ids"])
        for g in load_jsonl(DATA_DIR / "gold.jsonl")
    }

    axis_docs = load_jsonl(DATA_DIR / "axis_docs.jsonl")
    axis_queries = load_jsonl(DATA_DIR / "axis_queries.jsonl")
    axis_gold = {
        g["query_id"]: set(g["relevant_doc_ids"])
        for g in load_jsonl(DATA_DIR / "axis_gold.jsonl")
    }

    adv_docs = load_jsonl(DATA_DIR / "adversarial_docs.jsonl")
    adv_queries = load_jsonl(DATA_DIR / "adversarial_queries.jsonl")
    adv_gold = {
        g["query_id"]: set(g.get("relevant_doc_ids") or [])
        for g in load_jsonl(DATA_DIR / "adversarial_gold.jsonl")
    }
    adv_q_cat = {q["query_id"]: q["category"] for q in adv_queries}

    # Merge all docs (distinct by doc_id)
    all_docs_by_id: dict[str, dict] = {}
    for d in lat_docs + base_docs + axis_docs + adv_docs:
        all_docs_by_id[d["doc_id"]] = d
    all_docs = list(all_docs_by_id.values())
    all_queries = lat_queries + base_queries + axis_queries + adv_queries
    all_gold = {**lat_gold, **base_gold, **axis_gold, **adv_gold}

    lat_qids = {q["query_id"] for q in lat_queries}
    base_qids = {q["query_id"] for q in base_queries}
    axis_qids = {q["query_id"] for q in axis_queries}
    adv_qids = {q["query_id"] for q in adv_queries}
    all_qids = lat_qids | base_qids | axis_qids | adv_qids

    print(
        f"Docs: lat={len(lat_docs)}, base={len(base_docs)}, axis={len(axis_docs)}, adv={len(adv_docs)} (total unique {len(all_docs)})"
    )
    print(
        f"Queries: lat={len(lat_qids)}, base={len(base_qids)}, axis={len(axis_qids)}, adv={len(adv_qids)}"
    )

    # ===== Extract =====
    # Base/axis use the multi_axis cache (already populated).
    # Adversarial uses adversarial_v2pp cache.
    # Lattice synth uses its own fresh cache.

    async def ext_docs_and_queries(docs, queries, cache_dir: Path, label: str):
        cache_dir.mkdir(exist_ok=True, parents=True)
        cache_file_p1 = cache_dir / "llm_cache.json"
        doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs]
        q_items = [
            (q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries
        ]
        d_ex, u1 = await run_v2pp_extract(doc_items, cache_file_p1, f"{label}-docs")
        q_ex, u2 = await run_v2pp_extract(q_items, cache_file_p1, f"{label}-queries")
        return d_ex, q_ex, u1, u2

    # Base/axis share extraction; use multi_axis cache which is populated.
    # But multi_axis was using v2 extractor, not v2pp. Use a shared v2pp cache.
    # Reuse existing cache/extractor_v2pp for the base/axis corpus since the
    # prompt is the same (v2pp pass-1 + shared pass-2).
    # That cache is: cache/extractor_v2pp/llm_cache.json
    base_axis_doc_ext, u_ba_d = await run_v2pp_extract(
        [
            (d["doc_id"], d["text"], parse_iso(d["ref_time"]))
            for d in base_docs + axis_docs
        ],
        ROOT / "cache" / "extractor_v2pp" / "llm_cache.json",
        "base+axis-docs",
    )
    base_axis_q_ext, u_ba_q = await run_v2pp_extract(
        [
            (q["query_id"], q["text"], parse_iso(q["ref_time"]))
            for q in base_queries + axis_queries
        ],
        ROOT / "cache" / "extractor_v2pp" / "llm_cache.json",
        "base+axis-queries",
    )

    # Adversarial — reuse v2pp adversarial cache
    adv_doc_ext, u_adv_d = await run_v2pp_extract(
        [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in adv_docs],
        ROOT / "cache" / "adversarial_v2pp" / "extractor_v2pp_pass1" / "llm_cache.json",
        "adv-docs",
    )
    adv_q_ext, u_adv_q = await run_v2pp_extract(
        [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in adv_queries],
        ROOT / "cache" / "adversarial_v2pp" / "extractor_v2pp_pass1" / "llm_cache.json",
        "adv-queries",
    )

    # Lattice synth — fresh cache
    lat_doc_ext, u_lat_d = await run_v2pp_extract(
        [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in lat_docs],
        LATTICE_CACHE_DIR / "llm_cache.json",
        "lat-docs",
    )
    lat_q_ext, u_lat_q = await run_v2pp_extract(
        [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in lat_queries],
        LATTICE_CACHE_DIR / "llm_cache.json",
        "lat-queries",
    )

    # Era extraction — SKIPPED in lattice eval to keep run time bounded.
    # E channel will be empty (baseline V7 thus effectively T+S+A weights
    # dominate, L channel still compared vs V7 without era bias).
    era_doc_ext: dict[str, list[TimeExpression]] = {}
    era_q_ext: dict[str, list[TimeExpression]] = {}
    u_era_d = {"input": 0, "output": 0}
    u_era_q = {"input": 0, "output": 0}

    # Merge v2pp + era per id (union of TEs)
    def merge_tes(a, b):
        seen = set()
        merged = []
        for te in list(a) + list(b):
            key = (te.kind, (te.surface or "").strip().lower())
            if key in seen:
                continue
            seen.add(key)
            merged.append(te)
        return merged

    doc_ext: dict[str, list[TimeExpression]] = {}
    for d in all_docs:
        did = d["doc_id"]
        v2 = (
            base_axis_doc_ext.get(did)
            or adv_doc_ext.get(did)
            or lat_doc_ext.get(did)
            or []
        )
        er = era_doc_ext.get(did, [])
        doc_ext[did] = merge_tes(v2, er)

    q_ext: dict[str, list[TimeExpression]] = {}
    for q in all_queries:
        qid = q["query_id"]
        v2 = base_axis_q_ext.get(qid) or adv_q_ext.get(qid) or lat_q_ext.get(qid) or []
        er = era_q_ext.get(qid, [])
        q_ext[qid] = merge_tes(v2, er)

    # ===== Build T channel memory =====
    print("Building T-channel memory...")
    doc_mem = build_memory(doc_ext)
    q_mem = build_memory(q_ext)
    # Ensure all docs have an entry
    for d in all_docs:
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

    # ===== Build L channel =====
    print("Building lattice index...")
    if LATTICE_DB.exists():
        LATTICE_DB.unlink()
    store = LatticeStore(LATTICE_DB)
    ingest_lattice(store, doc_ext)
    lat_stats = store.stats()
    print(f"Lattice index: {lat_stats}")

    tag_freq = store.tag_frequencies()
    top_tags = tag_freq[:20]
    print(f"Top tags by frequency: {top_tags[:10]}")

    # ===== Semantic (S) + era (E) + allen (A) placeholder =====
    print("Embedding docs + queries...")
    doc_texts = [d["text"] for d in all_docs]
    q_texts = [q["text"] for q in all_queries]
    doc_embs_arr = await embed_all(doc_texts)
    q_embs_arr = await embed_all(q_texts)
    doc_embs = {d["doc_id"]: doc_embs_arr[i] for i, d in enumerate(all_docs)}
    q_embs = {q["query_id"]: q_embs_arr[i] for i, q in enumerate(all_queries)}

    # ===== Per-query ranking under each variant =====
    print("Scoring per-query...")
    all_doc_ids = [d["doc_id"] for d in all_docs]

    # Lattice scores
    l_scores_per_q, lat_debug = retrieve_lattice_scores(store, q_ext, all_qids)

    variants: dict[str, dict[str, list[str]]] = {
        "V7": {},
        "V7L": {},
        "V-LATTICE-ONLY": {},
    }

    lookup_cells_count: list[int] = []
    for qid in all_qids:
        t_scores = rank_multi_axis_t(
            q_mem.get(qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()}),
            doc_mem,
        )
        s_scores = rank_semantic_s(qid, q_embs, doc_embs)
        a_scores: dict[str, float] = {}
        e_scores: dict[str, float] = {}
        l_scores = l_scores_per_q.get(qid, {})
        dbg = lat_debug.get(qid, {})
        if "expanded_count" in dbg:
            lookup_cells_count.append(dbg["expanded_count"])

        variants["V7"][qid] = rank_v7_score_blend(
            t_scores, s_scores, a_scores, e_scores
        )
        variants["V7L"][qid] = rank_v7l_score_blend(
            t_scores, s_scores, a_scores, e_scores, l_scores
        )
        variants["V-LATTICE-ONLY"][qid] = rank_lattice_only(
            l_scores, s_scores, all_doc_ids
        )

    # ===== Evaluate =====
    subset_qids: dict[str, set[str]] = {
        "lattice-synth": lat_qids,
        "base-55": base_qids,
        "axis-synth": axis_qids,
        "adversarial": adv_qids,
    }
    # Lattice sub-subsets
    lat_subset_qids: dict[str, set[str]] = defaultdict(set)
    for qid, sub in lat_subset_of_q.items():
        lat_subset_qids[sub].add(qid)

    # Adversarial subsets of interest
    adv_cat_qids: dict[str, set[str]] = defaultdict(set)
    for qid, cat in adv_q_cat.items():
        adv_cat_qids[cat].add(qid)

    # Evaluate all
    per_variant: dict[str, dict[str, dict[str, float]]] = {}
    for var, ranked_per_q in variants.items():
        per_variant[var] = {}
        for sub, qids in subset_qids.items():
            per_variant[var][sub] = eval_rankings(ranked_per_q, all_gold, qids)
        for sub, qids in lat_subset_qids.items():
            per_variant[var][f"lat:{sub}"] = eval_rankings(ranked_per_q, all_gold, qids)
        for cat, qids in adv_cat_qids.items():
            per_variant[var][f"adv:{cat}"] = eval_rankings(ranked_per_q, all_gold, qids)

    # ===== Tag cardinality stats =====
    tag_card = {
        "n_docs_tagged": lat_stats["n_docs_tagged"],
        "n_rows": lat_stats["n_rows"],
        "n_unique_tags": lat_stats["n_unique_tags"],
        "avg_tags_per_doc": lat_stats["avg_tags_per_doc"],
        "max_tags_per_doc": lat_stats["max_tags_per_doc"],
        "min_tags_per_doc": lat_stats["min_tags_per_doc"],
        "top_hub_tags": [{"tag": t, "count": c} for t, c in top_tags[:15]],
        "avg_cells_visited_per_query": (
            sum(lookup_cells_count) / len(lookup_cells_count)
        )
        if lookup_cells_count
        else 0,
        "max_cells_visited_per_query": max(lookup_cells_count)
        if lookup_cells_count
        else 0,
    }

    # Sample lattice tag examples
    lat_doc_tag_samples = []
    for d in lat_docs[:10]:
        tags = store.tags_for_doc(d["doc_id"])
        lat_doc_tag_samples.append(
            {
                "doc_id": d["doc_id"],
                "text": d["text"],
                "precision": d["precision"],
                "tags": [t["tag"] for t in tags],
            }
        )
    lat_query_tag_samples = []
    for q in lat_queries[:10]:
        dbg = lat_debug.get(q["query_id"], {})
        lat_query_tag_samples.append(
            {
                "query_id": q["query_id"],
                "text": q["text"],
                "subset": q.get("subset"),
                "native_abs": dbg.get("query_native_abs_tags", []),
                "cyclical": dbg.get("query_cyclical_tags", []),
                "n_expanded": dbg.get("expanded_count"),
                "n_docs_matched": dbg.get("n_docs_matched_total")
                or dbg.get("n_docs_matched"),
            }
        )

    # ===== Cost =====
    usages = [u_ba_d, u_ba_q, u_adv_d, u_adv_q, u_lat_d, u_lat_q, u_era_d, u_era_q]
    total_in = sum(u.get("input", 0) for u in usages)
    total_out = sum(u.get("output", 0) for u in usages)
    cost_usd = total_in * 0.25 / 1_000_000 + total_out * 2.0 / 1_000_000

    wall_s = time.time() - t0

    # ===== Write JSON =====
    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        if isinstance(o, float) and math.isnan(o):
            return None
        if isinstance(o, set):
            return sorted(o)
        return o

    results_json = {
        "per_variant": _clean(per_variant),
        "tag_cardinality": _clean(tag_card),
        "lattice_doc_tag_samples": _clean(lat_doc_tag_samples),
        "lattice_query_tag_samples": _clean(lat_query_tag_samples),
        "cost": {"input_tokens": total_in, "output_tokens": total_out, "usd": cost_usd},
        "wall_seconds": wall_s,
    }
    (RESULTS_DIR / "lattice.json").write_text(
        json.dumps(results_json, indent=2, default=str)
    )

    # ===== Markdown =====
    def fmt(x, pct=False):
        if x is None or (
            isinstance(x, float) and (math.isnan(x) if isinstance(x, float) else False)
        ):
            return "-"
        try:
            if pct:
                return f"{x:.3f}"
            return f"{x:.3f}"
        except Exception:
            return str(x)

    lines: list[str] = []
    lines.append("# Lattice Inverted Index — Evaluation\n\n")
    lines.append(
        f"Corpus: {len(all_docs)} unique docs, {len(all_queries)} queries. Wall: {wall_s:.1f}s. LLM cost: ${cost_usd:.4f}.\n\n"
    )

    lines.append("## Tag cardinality\n\n")
    lines.append(f"- Docs tagged: **{tag_card['n_docs_tagged']}**\n")
    lines.append(f"- Total tag rows: **{tag_card['n_rows']}**\n")
    lines.append(f"- Unique tags: **{tag_card['n_unique_tags']}**\n")
    lines.append(
        f"- Avg tags/doc: **{tag_card['avg_tags_per_doc']:.2f}** (min {tag_card['min_tags_per_doc']}, max {tag_card['max_tags_per_doc']})\n"
    )
    lines.append(
        f"- Avg cells visited per query: **{tag_card['avg_cells_visited_per_query']:.1f}** (max {tag_card['max_cells_visited_per_query']})\n\n"
    )
    lines.append("Top-15 hub tags (most docs share these):\n\n")
    lines.append("| tag | docs |\n|---|---:|\n")
    for ht in tag_card["top_hub_tags"]:
        lines.append(f"| `{ht['tag']}` | {ht['count']} |\n")

    lines.append("\n## Per-variant metrics\n\n")
    lines.append("### Primary subsets\n\n")
    lines.append(
        "| Variant | lat R@5 | lat R@10 | lat MRR | lat NDCG | base R@5 | axis R@5 | adv R@5 |\n"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for var in ["V-LATTICE-ONLY", "V7", "V7L"]:
        m = per_variant[var]
        lines.append(
            f"| {var} | "
            f"{fmt(m['lattice-synth']['recall@5'])} | "
            f"{fmt(m['lattice-synth']['recall@10'])} | "
            f"{fmt(m['lattice-synth']['mrr'])} | "
            f"{fmt(m['lattice-synth']['ndcg@10'])} | "
            f"{fmt(m['base-55']['recall@5'])} | "
            f"{fmt(m['axis-synth']['recall@5'])} | "
            f"{fmt(m['adversarial']['recall@5'])} |\n"
        )

    lines.append("\n### Lattice sub-subsets (R@5)\n\n")
    sub_order = [
        "narrow_query_broad_doc",
        "broad_query_narrow_doc",
        "same_precision",
        "same_precision_s8",
        "cyclical",
        "s8_crossdoc",
    ]
    lines.append("| Variant | " + " | ".join(sub_order) + " |\n")
    lines.append("|---|" + "|".join(["---:"] * len(sub_order)) + "|\n")
    for var in ["V-LATTICE-ONLY", "V7", "V7L"]:
        row = [var]
        for sub in sub_order:
            k = f"lat:{sub}"
            m = per_variant[var].get(k, {})
            row.append(fmt(m.get("recall@5", float("nan"))))
        lines.append("| " + " | ".join(row) + " |\n")

    lines.append("\n### Adversarial by category (focus: A3, A6, S8) — R@5\n\n")
    cats_focus = ["A3", "A6", "S8"]
    other_cats = sorted([c for c in adv_cat_qids if c not in cats_focus])
    lines.append("| Variant | " + " | ".join(cats_focus + other_cats) + " |\n")
    lines.append(
        "|---|" + "|".join(["---:"] * (len(cats_focus) + len(other_cats))) + "|\n"
    )
    for var in ["V-LATTICE-ONLY", "V7", "V7L"]:
        row = [var]
        for cat in cats_focus + other_cats:
            k = f"adv:{cat}"
            m = per_variant[var].get(k, {})
            row.append(fmt(m.get("recall@5", float("nan"))))
        lines.append("| " + " | ".join(row) + " |\n")

    lines.append("\n## Sample lattice tags (lattice-synth docs)\n\n")
    for s in lat_doc_tag_samples:
        lines.append(f"- **{s['doc_id']}** ({s['precision']}): `{s['text'][:70]}`\n")
        lines.append(f"  tags: {sorted(s['tags'])}\n")

    lines.append("\n## Sample lattice query expansions\n\n")
    for s in lat_query_tag_samples:
        lines.append(
            f"- **{s['query_id']}** ({s['subset']}): `{s['text']}`  \n"
            f"  native_abs={s['native_abs']}, cyclical={s['cyclical']}, "
            f"expanded={s['n_expanded']}, matched_docs={s['n_docs_matched']}\n"
        )

    lines.append("\n## Cost & timing\n\n")
    lines.append(f"- LLM tokens: input={total_in}, output={total_out}\n")
    lines.append(f"- Estimated cost: ${cost_usd:.4f}\n")
    lines.append(f"- Wall clock: {wall_s:.1f}s\n")

    (RESULTS_DIR / "lattice.md").write_text("".join(lines))
    print("\nWrote results/lattice.{md,json}")
    print(f"Cost: ${cost_usd:.4f}, Wall: {wall_s:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
