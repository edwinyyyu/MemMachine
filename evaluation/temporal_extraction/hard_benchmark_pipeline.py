"""Full temporal-pipeline eval on hard_bench (synthetic stress benchmark).

Variants:
- SEMANTIC-ONLY            (text-embedding-3-small cosine)
- T-only                   (multi-axis: intervals + axes + tags)
- V7  (T+S, weights 0.5/0.5)        @ cv_ref ∈ {0.10, 0.20, 0.30, 0.50}
- V7L (T+S+L, weights 0.4/0.4/0.2)  @ cv_ref ∈ {0.10, 0.20, 0.30, 0.50}

For brevity, the multi-cv_ref sweep is done after a single extraction pass:
the score arrays are reused.

Hard per-LLM-call timeout: 60s. Wall cap: 35 minutes.

Also patches the v2 extractor for gpt-5-mini reasoning_effort=minimal,
matching the tempreason_fast_run pattern.

Usage:  uv run python hard_benchmark_pipeline.py
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ---- patch: gpt-5-mini reasoning_effort=minimal, mirrors tempreason_fast_run ----
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

# ---- imports after patch ----
from axis_distributions import (
    AXES,
    AxisDistribution,
    axes_for_expression,
    merge_axis_dists,
)
from baselines import embed_all  # noqa: E402
from expander import expand  # noqa: E402
from extractor_v2 import ExtractorV2  # noqa: E402
from lattice_cells import (
    tags_for_expression as lattice_tags_for_expression,
)
from lattice_retrieval import retrieve_multi as lattice_retrieve_multi  # noqa: E402
from lattice_store import LatticeStore  # noqa: E402
from multi_axis_scorer import axis_score as axis_score_fn  # noqa: E402
from multi_axis_scorer import tag_score
from multi_axis_tags import tags_for_axes  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402
from rag_fusion import score_blend  # noqa: E402
from schema import GRANULARITY_ORDER, TimeExpression, parse_iso, to_us  # noqa: E402
from scorer import Interval, score_jaccard_composite  # noqa: E402

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_DIR = ROOT / "cache" / "hard_bench_v2"
CACHE_DIR.mkdir(exist_ok=True, parents=True)
LATTICE_DB = CACHE_DIR / "lattice.sqlite"

TOP_K = 10
PER_CALL_TIMEOUT_S = 60.0
# Reduced concurrency to avoid rate-limit-induced timeouts during long
# extraction runs (observed: clean for first ~400 calls then 30%+ timeouts).
CONCURRENCY = 6
HARD_TIMEOUT_FRAC = 0.50  # tolerate higher timeout rate; recoverable via cache
WALL_CAP_S = 35 * 60

PRICE_IN_PER_M = 0.25
PRICE_OUT_PER_M = 2.00


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# Interval flatten — same as tempreason_pipeline_eval
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
    r1, r3, r5, r10, mr, nd = [], [], [], [], [], []
    for qid in qids:
        rel = gold.get(qid, set())
        if not rel:
            continue
        ranked = ranked_per_q.get(qid, [])
        r1.append(recall_at_k(ranked, rel, 1))
        r3.append(recall_at_k(ranked, rel, 3))
        r5.append(recall_at_k(ranked, rel, 5))
        r10.append(recall_at_k(ranked, rel, 10))
        mr.append(mrr(ranked, rel))
        nd.append(ndcg_at_k(ranked, rel, TOP_K))
    return {
        "n": len(r5),
        "recall@1": nanmean(r1),
        "recall@3": nanmean(r3),
        "recall@5": nanmean(r5),
        "recall@10": nanmean(r10),
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
        except Exception as e:
            n_error += 1
            print(f"  [{label}] FAIL {iid}: {e}", flush=True)
            tes = []
        completed[0] += 1
        if completed[0] % 50 == 0:
            print(
                f"  [{label}] {completed[0]}/{total} (timeout={n_timeout})", flush=True
            )
        return iid, tes

    print(f"v2 {label}: {total} items", flush=True)
    pairs = await asyncio.gather(*(one(*it) for it in items))
    for iid, tes in pairs:
        results[iid] = tes
    ex.cache.save()
    try:
        ex.shared_pass2_cache.save()
    except Exception:
        pass

    cost = (
        ex.usage["input"] * PRICE_IN_PER_M / 1_000_000
        + ex.usage["output"] * PRICE_OUT_PER_M / 1_000_000
    )
    print(
        f"  [{label}] usage in={ex.usage['input']}, out={ex.usage['output']}, cost=${cost:.4f}",
        flush=True,
    )
    print(f"  [{label}] timeouts={n_timeout}, errors={n_error}", flush=True)
    return results, ex.usage, n_timeout, n_error


# ---------------------------------------------------------------------------
# Memory build
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


def rank_semantic_s(qid, q_embs, doc_embs):
    qv = q_embs[qid]
    qn = np.linalg.norm(qv) or 1e-9
    out: dict[str, float] = {}
    for d, v in doc_embs.items():
        vn = np.linalg.norm(v) or 1e-9
        out[d] = float(np.dot(qv, v) / (qn * vn))
    return out


def ingest_lattice(store: LatticeStore, extracted):
    store.clear()
    for did, tes in extracted.items():
        all_abs: list[tuple[str, str]] = []
        all_cyc: set[str] = set()
        for te in tes:
            ts = lattice_tags_for_expression(te)
            all_abs.extend(ts.absolute)
            all_cyc |= ts.cyclical
        seen = set()
        dedup = []
        for prec, t in all_abs:
            if t in seen:
                continue
            seen.add(t)
            dedup.append((prec, t))
        store.insert(did, dedup, all_cyc)


def retrieve_lattice_scores(store, query_extracted, all_qids):
    per_q_scores: dict[str, dict[str, float]] = {}
    for qid in all_qids:
        tes = query_extracted.get(qid, [])
        if not tes:
            per_q_scores[qid] = {}
            continue
        scores, _ = lattice_retrieve_multi(store, tes, down_levels=1)
        per_q_scores[qid] = scores
    return per_q_scores


def rank_v7l(t, s, l, weights=None, cv_ref=0.20):
    if weights is None:
        weights = {"T": 0.4, "S": 0.4, "L": 0.2}
    fused = score_blend(
        {"T": t, "S": s, "L": l}, weights, top_k_per=40, dispersion_cv_ref=cv_ref
    )
    ranked = [d for d, _ in fused]
    seen = set(ranked)
    tail = [
        d
        for d, _ in sorted(s.items(), key=lambda x: x[1], reverse=True)
        if d not in seen
    ]
    return ranked + tail


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


def rank_t_only(t, all_doc_ids, s_scores):
    return sorted(
        all_doc_ids, key=lambda d: (t.get(d, 0.0), s_scores.get(d, 0.0)), reverse=True
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    t0 = time.time()
    wall_start = time.time()

    docs = load_jsonl(DATA_DIR / "hard_bench_docs.jsonl")
    queries = load_jsonl(DATA_DIR / "hard_bench_queries.jsonl")
    gold_raw = load_jsonl(DATA_DIR / "hard_bench_gold.jsonl")
    gold = {g["query_id"]: set(g["relevant_doc_ids"]) for g in gold_raw}
    subset_of_q = {q["query_id"]: q["subset"] for q in queries}

    print(f"Hard bench: {len(docs)} docs, {len(queries)} queries", flush=True)

    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs]
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries]

    print("=== Extracting docs (v2) ===", flush=True)
    doc_ext, doc_usage, doc_to, doc_err = await run_v2_extract(
        doc_items, "docs", "hard_bench_v2_docs"
    )
    if doc_to / max(1, len(doc_items)) > HARD_TIMEOUT_FRAC:
        print(f"FAIL-FAST: doc timeouts {doc_to}/{len(doc_items)}", flush=True)
        return

    print("=== Extracting queries (v2) ===", flush=True)
    q_ext, q_usage, q_to, q_err = await run_v2_extract(
        q_items, "queries", "hard_bench_v2_queries"
    )

    if time.time() - wall_start > WALL_CAP_S:
        print("WALL CAP HIT after extraction — bailing", flush=True)
        return

    total_in = doc_usage["input"] + q_usage["input"]
    total_out = doc_usage["output"] + q_usage["output"]
    cost_extract = total_in * PRICE_IN_PER_M / 1e6 + total_out * PRICE_OUT_PER_M / 1e6
    print(f"v2 extraction cost: ${cost_extract:.4f}", flush=True)

    # --- Build memories ---
    print("Building T-channel memory...", flush=True)
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

    # --- Build lattice ---
    print("Building lattice index...", flush=True)
    if LATTICE_DB.exists():
        LATTICE_DB.unlink()
    store = LatticeStore(LATTICE_DB)
    ingest_lattice(store, doc_ext)
    lat_stats = store.stats()
    print(f"Lattice: {lat_stats}", flush=True)

    # --- Embed ---
    print("Embedding (text-embedding-3-small)...", flush=True)
    doc_texts = [d["text"] for d in docs]
    q_texts = [q["text"] for q in queries]
    doc_embs_arr = await embed_all(doc_texts)
    q_embs_arr = await embed_all(q_texts)
    doc_embs = {d["doc_id"]: doc_embs_arr[i] for i, d in enumerate(docs)}
    q_embs = {q["query_id"]: q_embs_arr[i] for i, q in enumerate(queries)}

    # --- Compute per-query scores once ---
    all_qids = {q["query_id"] for q in queries}
    all_doc_ids = [d["doc_id"] for d in docs]
    print("Computing per-query T, S, L scores...", flush=True)
    l_scores_per_q = retrieve_lattice_scores(store, q_ext, all_qids)
    per_q_t_scores: dict[str, dict[str, float]] = {}
    per_q_s_scores: dict[str, dict[str, float]] = {}
    for q in queries:
        qid = q["query_id"]
        per_q_t_scores[qid] = rank_multi_axis_t(
            q_mem.get(qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()}),
            doc_mem,
        )
        per_q_s_scores[qid] = rank_semantic_s(qid, q_embs, doc_embs)

    # --- Build variants ---
    variants: dict[str, dict[str, list[str]]] = {}

    # SEMANTIC-ONLY
    sem_v: dict[str, list[str]] = {}
    for q in queries:
        qid = q["query_id"]
        sem_ranked = sorted(
            per_q_s_scores[qid].items(), key=lambda x: x[1], reverse=True
        )
        sem_v[qid] = [d for d, _ in sem_ranked]
    variants["SEMANTIC-ONLY"] = sem_v

    # T-only
    t_v: dict[str, list[str]] = {}
    for q in queries:
        qid = q["query_id"]
        t_v[qid] = rank_t_only(per_q_t_scores[qid], all_doc_ids, per_q_s_scores[qid])
    variants["T-only"] = t_v

    # V7 default (cv_ref=0.20)
    v7_default: dict[str, list[str]] = {}
    for q in queries:
        qid = q["query_id"]
        v7_default[qid] = rank_v7(per_q_t_scores[qid], per_q_s_scores[qid], cv_ref=0.20)
    variants["V7 (T+S, cv=0.20)"] = v7_default

    # V7 cv_ref sweep
    cv_refs = [0.10, 0.20, 0.30, 0.50]
    for cv in cv_refs:
        if cv == 0.20:
            continue  # already added
        var_name = f"V7 (T+S, cv={cv:.2f})"
        rd: dict[str, list[str]] = {}
        for q in queries:
            qid = q["query_id"]
            rd[qid] = rank_v7(per_q_t_scores[qid], per_q_s_scores[qid], cv_ref=cv)
        variants[var_name] = rd

    # V7L (T+S+L, default cv=0.20)
    v7l_default: dict[str, list[str]] = {}
    for q in queries:
        qid = q["query_id"]
        v7l_default[qid] = rank_v7l(
            per_q_t_scores[qid],
            per_q_s_scores[qid],
            l_scores_per_q.get(qid, {}),
            cv_ref=0.20,
        )
    variants["V7L (T+S+L, cv=0.20)"] = v7l_default

    for cv in cv_refs:
        if cv == 0.20:
            continue
        var_name = f"V7L (T+S+L, cv={cv:.2f})"
        rd: dict[str, list[str]] = {}
        for q in queries:
            qid = q["query_id"]
            rd[qid] = rank_v7l(
                per_q_t_scores[qid],
                per_q_s_scores[qid],
                l_scores_per_q.get(qid, {}),
                cv_ref=cv,
            )
        variants[var_name] = rd

    # --- Eval: per-tier ---
    easy_qids = {qid for qid, sub in subset_of_q.items() if sub == "easy"}
    medium_qids = {qid for qid, sub in subset_of_q.items() if sub == "medium"}
    hard_qids = {qid for qid, sub in subset_of_q.items() if sub == "hard"}
    subsets = {
        "all": all_qids,
        "easy": easy_qids,
        "medium": medium_qids,
        "hard": hard_qids,
    }

    per_variant: dict[str, dict[str, dict[str, float]]] = {}
    for var, ranked_per_q in variants.items():
        per_variant[var] = {}
        for sub_name, qids in subsets.items():
            per_variant[var][sub_name] = eval_rankings(ranked_per_q, gold, qids)

    # --- Failure analysis ---
    rank_compare = []
    for q in queries:
        qid = q["query_id"]
        rel = gold.get(qid, set())
        if not rel:
            continue
        sem_rank = query_rank_of_gold(variants["SEMANTIC-ONLY"][qid], rel)
        t_rank = query_rank_of_gold(variants["T-only"].get(qid, []), rel)
        v7_rank = query_rank_of_gold(variants["V7 (T+S, cv=0.20)"][qid], rel)
        v7l_rank = query_rank_of_gold(variants["V7L (T+S+L, cv=0.20)"][qid], rel)
        gold_doc_id = next(iter(rel))
        gold_text = next((d["text"] for d in docs if d["doc_id"] == gold_doc_id), "<?>")
        sem_top3 = variants["SEMANTIC-ONLY"][qid][:3]
        sem_top3_texts = [
            next((d["text"] for d in docs if d["doc_id"] == did), "<?>")
            for did in sem_top3
        ]
        v7_top3 = variants["V7 (T+S, cv=0.20)"][qid][:3]
        v7_top3_texts = [
            next((d["text"] for d in docs if d["doc_id"] == did), "<?>")
            for did in v7_top3
        ]
        rank_compare.append(
            {
                "qid": qid,
                "subset": q["subset"],
                "query": q["text"],
                "ref_time": q["ref_time"],
                "gold_doc_id": gold_doc_id,
                "gold_text": gold_text,
                "rank_sem": sem_rank,
                "rank_t": t_rank,
                "rank_v7": v7_rank,
                "rank_v7l": v7l_rank,
                "sem_top3": sem_top3,
                "sem_top3_texts": sem_top3_texts,
                "v7_top3": v7_top3,
                "v7_top3_texts": v7_top3_texts,
                "n_q_extractions": len(q_ext.get(qid, [])),
                "n_gold_extractions": len(doc_ext.get(gold_doc_id, [])),
            }
        )

    # Wins: sem rank > 1 (and finite or None) but V7 rank == 1
    wins = [
        r
        for r in rank_compare
        if r["rank_v7"] == 1 and (r["rank_sem"] is None or r["rank_sem"] > 1)
    ]
    losses = [
        r
        for r in rank_compare
        if r["rank_sem"] == 1 and (r["rank_v7"] is None or r["rank_v7"] > 1)
    ]
    persistent_misses = [
        r
        for r in rank_compare
        if (r["rank_sem"] is None or r["rank_sem"] > 5)
        and (r["rank_v7"] is None or r["rank_v7"] > 5)
    ]

    cost_total = cost_extract
    wall_s = time.time() - t0

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

    out_json = {
        "benchmark": {
            "name": "hard_bench (synthetic stress)",
            "n_docs": len(docs),
            "n_queries": len(queries),
            "n_easy": len(easy_qids),
            "n_medium": len(medium_qids),
            "n_hard": len(hard_qids),
        },
        "extraction": {
            "doc_timeouts": doc_to,
            "doc_errors": doc_err,
            "query_timeouts": q_to,
            "query_errors": q_err,
            "doc_extractions_per_doc_mean": (
                sum(len(v) for v in doc_ext.values()) / max(1, len(doc_ext))
            ),
            "q_extractions_per_q_mean": (
                sum(len(v) for v in q_ext.values()) / max(1, len(q_ext))
            ),
        },
        "lattice_stats": _clean(lat_stats),
        "per_variant": _clean(per_variant),
        "failure_analysis": {
            "n_total": len(rank_compare),
            "n_wins_v7_over_sem": len(wins),
            "n_losses_v7_to_sem": len(losses),
            "n_persistent_misses": len(persistent_misses),
            "wins": _clean(wins[:10]),
            "losses": _clean(losses[:10]),
            "persistent_misses": _clean(persistent_misses[:5]),
        },
        "cost": {
            "extraction_v2_usd": cost_extract,
            "total_usd": cost_total,
        },
        "wall_seconds": wall_s,
    }

    out_path = RESULTS_DIR / "hard_bench_pipeline.json"
    out_path.write_text(json.dumps(out_json, indent=2, default=str))
    print(f"\nWrote {out_path}", flush=True)

    # Print summary
    print("\n=== Summary (all queries) ===")
    print(
        f"{'Variant':<32} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'R@10':>6} {'MRR':>6} {'NDCG':>6}"
    )
    for var in variants:
        m = per_variant[var]["all"]
        print(
            f"{var:<32} {m['recall@1']:>6.3f} {m['recall@3']:>6.3f} "
            f"{m['recall@5']:>6.3f} {m['recall@10']:>6.3f} "
            f"{m['mrr']:>6.3f} {m['ndcg@10']:>6.3f}"
        )

    print("\n=== Summary (hard tier only) ===")
    print(
        f"{'Variant':<32} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'R@10':>6} {'MRR':>6} {'NDCG':>6}"
    )
    for var in variants:
        m = per_variant[var]["hard"]
        if m["n"] == 0:
            continue
        print(
            f"{var:<32} {m['recall@1']:>6.3f} {m['recall@3']:>6.3f} "
            f"{m['recall@5']:>6.3f} {m['recall@10']:>6.3f} "
            f"{m['mrr']:>6.3f} {m['ndcg@10']:>6.3f}"
        )

    print(f"\nCost: ${cost_total:.4f}, wall: {wall_s:.1f}s")
    print(f"Wins (V7 > Sem): {len(wins)}, Losses (V7 < Sem): {len(losses)}")


if __name__ == "__main__":
    asyncio.run(main())
