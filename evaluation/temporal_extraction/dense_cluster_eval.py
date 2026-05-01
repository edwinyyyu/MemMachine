"""Dense time-cluster eval — stress test where T should fail to disambiguate.

Variants:
- SEMANTIC-ONLY
- T-ONLY
- V7 default (T=0.5, S=0.5)
- V7 TempReason-tuned (T=0.3, S=0.7)
- V7 auto-tuned (sweep T=0.0..0.7, S=1.0..0.3 in 0.1 steps; pick best by R@1)

Diagnostic: T-score variance within each query's top-10 candidate pool.

Monkeypatches gpt-5-mini with reasoning_effort=minimal (copying tempreason_fast_run.py).
Hard per-call timeout: 60s.

Usage: uv run python dense_cluster_eval.py
"""

from __future__ import annotations

import asyncio
import json
import math
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Monkey-patch (copied from tempreason_fast_run.py): force reasoning_effort=minimal
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
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
from schema import GRANULARITY_ORDER, TimeExpression, parse_iso, to_us  # noqa: E402
from scorer import Interval, score_jaccard_composite  # noqa: E402

# Try v2pp first (production extractor); fall back to v2.
try:
    from extractor_v2pp import ExtractorV2pp as Extractor  # type: ignore[attr-defined]

    EXTRACTOR_NAME = "v2pp"
except Exception:
    try:
        from extractor_v2p import (
            ExtractorV2p as Extractor,  # type: ignore[attr-defined]
        )

        EXTRACTOR_NAME = "v2p"
    except Exception:
        Extractor = ExtractorV2  # type: ignore[assignment]
        EXTRACTOR_NAME = "v2"

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_BASE = ROOT / "cache" / "dense_cluster"
CACHE_BASE.mkdir(exist_ok=True, parents=True)

TOP_K = 10
PER_CALL_TIMEOUT_S = 60.0
CONCURRENCY = 8

PRICE_IN_PER_M = 0.25
PRICE_OUT_PER_M = 2.00


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# ---------- Interval flatten (same as tempreason) ----------
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


# ---------- Metrics ----------
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


def nanmean(xs):
    vs = [v for v in xs if not math.isnan(v)]
    return sum(vs) / len(vs) if vs else float("nan")


def eval_rankings(ranked_per_q, gold, qids):
    r1, r3, r5, mr = [], [], [], []
    for qid in qids:
        rel = gold.get(qid, set())
        if not rel:
            continue
        ranked = ranked_per_q.get(qid, [])
        r1.append(recall_at_k(ranked, rel, 1))
        r3.append(recall_at_k(ranked, rel, 3))
        r5.append(recall_at_k(ranked, rel, 5))
        mr.append(mrr(ranked, rel))
    return {
        "n": len(r1),
        "recall@1": nanmean(r1),
        "recall@3": nanmean(r3),
        "recall@5": nanmean(r5),
        "mrr": nanmean(mr),
    }


# ---------- Extraction ----------
async def run_extract(items, label: str, cache_subdir: str):
    ex = Extractor(concurrency=CONCURRENCY, cache_subdir=cache_subdir)
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
                ex.extract(text, ref), timeout=PER_CALL_TIMEOUT_S * 1.5
            )
        except asyncio.TimeoutError:
            n_timeout += 1
            tes = []
        except Exception as e:
            n_error += 1
            print(f"  [{label}] FAIL {iid}: {e}", flush=True)
            tes = []
        completed[0] += 1
        if completed[0] % 20 == 0:
            print(
                f"  [{label}] {completed[0]}/{total} (timeout={n_timeout})", flush=True
            )
        return iid, tes

    print(f"{EXTRACTOR_NAME} {label}: {total} items", flush=True)
    pairs = await asyncio.gather(*(one(*it) for it in items))
    for iid, tes in pairs:
        results[iid] = tes
    try:
        ex.cache.save()
    except Exception:
        pass
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


def rank_v7(t, s, weights):
    fused = score_blend({"T": t, "S": s}, weights, top_k_per=40)
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


# ---------- Main ----------
async def main() -> None:
    t0 = time.time()
    WALL_CAP_S = 18 * 60

    docs = load_jsonl(DATA_DIR / "dense_cluster_docs.jsonl")
    queries = load_jsonl(DATA_DIR / "dense_cluster_queries.jsonl")
    gold_raw = load_jsonl(DATA_DIR / "dense_cluster_gold.jsonl")
    gold = {g["query_id"]: set(g["relevant_doc_ids"]) for g in gold_raw}

    print(f"Dense cluster: {len(docs)} docs, {len(queries)} queries", flush=True)

    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs]
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries]

    print(f"=== Extracting docs ({EXTRACTOR_NAME}) ===", flush=True)
    doc_ext, doc_usage, doc_to, doc_err = await run_extract(
        doc_items, "docs", "dense_cluster/docs"
    )

    print(f"=== Extracting queries ({EXTRACTOR_NAME}) ===", flush=True)
    q_ext, q_usage, q_to, q_err = await run_extract(
        q_items, "queries", "dense_cluster/queries"
    )

    total_in = doc_usage["input"] + q_usage["input"]
    total_out = doc_usage["output"] + q_usage["output"]
    cost_extract = total_in * PRICE_IN_PER_M / 1e6 + total_out * PRICE_OUT_PER_M / 1e6
    print(f"Extraction cost: ${cost_extract:.4f}", flush=True)

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

    print("Embedding (text-embedding-3-small)...", flush=True)
    doc_texts = [d["text"] for d in docs]
    q_texts = [q["text"] for q in queries]
    doc_embs_arr = await embed_all(doc_texts)
    q_embs_arr = await embed_all(q_texts)
    doc_embs = {d["doc_id"]: doc_embs_arr[i] for i, d in enumerate(docs)}
    q_embs = {q["query_id"]: q_embs_arr[i] for i, q in enumerate(queries)}

    # --- Score per-query ---
    print("Scoring per-query...", flush=True)
    all_qids = {q["query_id"] for q in queries}
    all_doc_ids = [d["doc_id"] for d in docs]

    per_q_t: dict[str, dict[str, float]] = {}
    per_q_s: dict[str, dict[str, float]] = {}
    for q in queries:
        qid = q["query_id"]
        per_q_t[qid] = rank_multi_axis_t(
            q_mem.get(qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()}),
            doc_mem,
        )
        per_q_s[qid] = rank_semantic_s(qid, q_embs, doc_embs)

    # --- Variants ---
    variants: dict[str, dict[str, list[str]]] = {}

    # SEMANTIC-ONLY
    sem_var: dict[str, list[str]] = {}
    for q in queries:
        qid = q["query_id"]
        sem_var[qid] = [
            d for d, _ in sorted(per_q_s[qid].items(), key=lambda x: x[1], reverse=True)
        ]
    variants["SEMANTIC-ONLY"] = sem_var

    # T-ONLY
    t_var: dict[str, list[str]] = {}
    for q in queries:
        qid = q["query_id"]
        t_var[qid] = rank_t_only(per_q_t[qid], all_doc_ids, per_q_s[qid])
    variants["T-ONLY"] = t_var

    # V7 default (0.5, 0.5)
    v7_default: dict[str, list[str]] = {}
    for q in queries:
        qid = q["query_id"]
        v7_default[qid] = rank_v7(per_q_t[qid], per_q_s[qid], {"T": 0.5, "S": 0.5})
    variants["V7 (T=0.5, S=0.5)"] = v7_default

    # V7 TempReason-tuned (0.3, 0.7)
    v7_tr: dict[str, list[str]] = {}
    for q in queries:
        qid = q["query_id"]
        v7_tr[qid] = rank_v7(per_q_t[qid], per_q_s[qid], {"T": 0.3, "S": 0.7})
    variants["V7 (T=0.3, S=0.7) [TempReason]"] = v7_tr

    # Auto-tune sweep
    print("Auto-tune sweep...", flush=True)
    sweep_results: list[dict] = []
    for i in range(8):  # 0.0..0.7
        wt = round(i * 0.1, 1)
        ws = round(1.0 - wt, 1)
        ranked_per_q: dict[str, list[str]] = {}
        for q in queries:
            qid = q["query_id"]
            ranked_per_q[qid] = rank_v7(per_q_t[qid], per_q_s[qid], {"T": wt, "S": ws})
        m = eval_rankings(ranked_per_q, gold, all_qids)
        sweep_results.append(
            {
                "T": wt,
                "S": ws,
                "recall@1": m["recall@1"],
                "recall@3": m["recall@3"],
                "recall@5": m["recall@5"],
                "mrr": m["mrr"],
                "ranked_per_q": ranked_per_q,
            }
        )

    best = max(sweep_results, key=lambda r: (r["recall@1"], r["mrr"]))
    variants[f"V7 AUTO (T={best['T']}, S={best['S']})"] = best["ranked_per_q"]

    # --- Eval ---
    per_variant: dict[str, dict[str, float]] = {}
    for var, ranked_per_q in variants.items():
        per_variant[var] = eval_rankings(ranked_per_q, gold, all_qids)

    # --- Diagnostic: T-score variance within top-10 candidates ---
    # Use the union of (V7-default top-10) and (S top-10) as the candidate pool.
    diag_per_q: list[dict] = []
    for q in queries:
        qid = q["query_id"]
        # pool = top-10 by S, top-10 by V7-default, top-10 by T-only
        pool = set(sem_var[qid][:10]) | set(v7_default[qid][:10]) | set(t_var[qid][:10])
        t_scores_in_pool = [per_q_t[qid].get(d, 0.0) for d in pool]
        s_scores_in_pool = [per_q_s[qid].get(d, 0.0) for d in pool]
        diag_per_q.append(
            {
                "qid": qid,
                "pool_size": len(pool),
                "t_mean": statistics.mean(t_scores_in_pool)
                if t_scores_in_pool
                else 0.0,
                "t_var": statistics.pvariance(t_scores_in_pool)
                if len(t_scores_in_pool) > 1
                else 0.0,
                "t_std": statistics.pstdev(t_scores_in_pool)
                if len(t_scores_in_pool) > 1
                else 0.0,
                "t_max": max(t_scores_in_pool) if t_scores_in_pool else 0.0,
                "t_min": min(t_scores_in_pool) if t_scores_in_pool else 0.0,
                "s_mean": statistics.mean(s_scores_in_pool)
                if s_scores_in_pool
                else 0.0,
                "s_std": statistics.pstdev(s_scores_in_pool)
                if len(s_scores_in_pool) > 1
                else 0.0,
            }
        )

    diag_summary = {
        "n_queries": len(diag_per_q),
        "mean_t_std_in_pool": statistics.mean(d["t_std"] for d in diag_per_q),
        "mean_t_var_in_pool": statistics.mean(d["t_var"] for d in diag_per_q),
        "mean_s_std_in_pool": statistics.mean(d["s_std"] for d in diag_per_q),
        "mean_t_max": statistics.mean(d["t_max"] for d in diag_per_q),
        "mean_t_min": statistics.mean(d["t_min"] for d in diag_per_q),
        "mean_t_mean": statistics.mean(d["t_mean"] for d in diag_per_q),
        "n_with_zero_t_var": sum(1 for d in diag_per_q if d["t_var"] < 1e-9),
    }

    # --- Failure analysis ---
    failures: list[dict] = []
    sem_ranks: list[int] = []
    v7_default_ranks: list[int] = []
    v7_tr_ranks: list[int] = []
    auto_ranks: list[int] = []
    t_only_ranks: list[int] = []

    def rank_of(ranked, rel):
        for i, d in enumerate(ranked, start=1):
            if d in rel:
                return i
        return None

    for q in queries:
        qid = q["query_id"]
        rel = gold.get(qid, set())
        sem_r = rank_of(sem_var[qid], rel)
        t_r = rank_of(t_var[qid], rel)
        v7d_r = rank_of(v7_default[qid], rel)
        v7tr_r = rank_of(v7_tr[qid], rel)
        auto_r = rank_of(best["ranked_per_q"][qid], rel)
        sem_ranks.append(sem_r if sem_r is not None else 999)
        t_only_ranks.append(t_r if t_r is not None else 999)
        v7_default_ranks.append(v7d_r if v7d_r is not None else 999)
        v7_tr_ranks.append(v7tr_r if v7tr_r is not None else 999)
        auto_ranks.append(auto_r if auto_r is not None else 999)
        if sem_r == 1 and (v7d_r and v7d_r > 1):
            gold_doc_id = next(iter(rel))
            gold_text = next(
                (d["text"] for d in docs if d["doc_id"] == gold_doc_id), "<?>"
            )
            failures.append(
                {
                    "qid": qid,
                    "query": q["text"],
                    "gold_text": gold_text,
                    "sem_rank": sem_r,
                    "v7_default_rank": v7d_r,
                    "v7_tr_rank": v7tr_r,
                    "auto_rank": auto_r,
                    "t_only_rank": t_r,
                }
            )

    cost_total = cost_extract
    wall_s = time.time() - t0

    # --- Write JSON ---
    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items() if k != "ranked_per_q"}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        if isinstance(o, float) and math.isnan(o):
            return None
        if isinstance(o, set):
            return sorted(o)
        return o

    out_json = {
        "benchmark": {
            "name": "dense_cluster (April 2024)",
            "extractor": EXTRACTOR_NAME,
            "n_docs": len(docs),
            "n_queries": len(queries),
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
        "per_variant": _clean(per_variant),
        "auto_tune_sweep": _clean(sweep_results),
        "auto_tune_best": {
            "T": best["T"],
            "S": best["S"],
            "recall@1": best["recall@1"],
            "recall@3": best["recall@3"],
            "mrr": best["mrr"],
        },
        "diagnostic_t_variance": diag_summary,
        "diagnostic_per_q_first10": _clean(diag_per_q[:10]),
        "failure_examples_sem_to_v7": _clean(failures[:10]),
        "rank_distributions": {
            "sem_rank_at_1_count": sum(1 for r in sem_ranks if r == 1),
            "v7_default_rank_at_1_count": sum(1 for r in v7_default_ranks if r == 1),
            "v7_tr_rank_at_1_count": sum(1 for r in v7_tr_ranks if r == 1),
            "auto_rank_at_1_count": sum(1 for r in auto_ranks if r == 1),
            "t_only_rank_at_1_count": sum(1 for r in t_only_ranks if r == 1),
        },
        "cost": {
            "extraction_usd": cost_extract,
            "total_usd": cost_total,
        },
        "wall_seconds": wall_s,
    }

    out_path = RESULTS_DIR / "dense_cluster.json"
    out_path.write_text(json.dumps(out_json, indent=2, default=str))
    print(f"Wrote {out_path}", flush=True)

    # ---- Markdown report ----
    md_lines: list[str] = []
    md_lines.append("# Dense Time-Cluster Stress Test")
    md_lines.append("")
    md_lines.append(f"- Extractor: **{EXTRACTOR_NAME}** (reasoning_effort=minimal)")
    md_lines.append(
        f"- Corpus: **{len(docs)} docs**, all dated April 2024 (days {1}..{30})"
    )
    md_lines.append(f"- Queries: **{len(queries)}**, each with exactly 1 gold doc")
    md_lines.append("- Embedding model: text-embedding-3-small")
    md_lines.append(f"- Cost: **${cost_total:.4f}**, wall: {wall_s:.1f}s")
    md_lines.append("")
    md_lines.append("## Per-variant retrieval metrics")
    md_lines.append("")
    md_lines.append("| Variant | R@1 | R@3 | R@5 | MRR |")
    md_lines.append("|---|---:|---:|---:|---:|")
    for var, m in per_variant.items():
        md_lines.append(
            f"| {var} | {m['recall@1']:.3f} | {m['recall@3']:.3f} | {m['recall@5']:.3f} | {m['mrr']:.3f} |"
        )
    md_lines.append("")
    md_lines.append("## Auto-tune weight sweep (V7 T+S)")
    md_lines.append("")
    md_lines.append("| T | S | R@1 | R@3 | R@5 | MRR |")
    md_lines.append("|---:|---:|---:|---:|---:|---:|")
    for r in sweep_results:
        md_lines.append(
            f"| {r['T']} | {r['S']} | {r['recall@1']:.3f} | {r['recall@3']:.3f} | {r['recall@5']:.3f} | {r['mrr']:.3f} |"
        )
    md_lines.append("")
    md_lines.append(
        f"**Best**: T={best['T']}, S={best['S']} → R@1={best['recall@1']:.3f}, MRR={best['mrr']:.3f}"
    )
    md_lines.append("")
    md_lines.append("## T-score variance diagnostic")
    md_lines.append("")
    md_lines.append(
        "Within each query's top-10 candidate pool (union of S/T-only/V7 top-10):"
    )
    md_lines.append("")
    md_lines.append(
        f"- Mean T std-dev in pool: **{diag_summary['mean_t_std_in_pool']:.4f}**"
    )
    md_lines.append(f"- Mean T mean (in pool): {diag_summary['mean_t_mean']:.4f}")
    md_lines.append(f"- Mean T max (in pool): {diag_summary['mean_t_max']:.4f}")
    md_lines.append(f"- Mean T min (in pool): {diag_summary['mean_t_min']:.4f}")
    md_lines.append(
        f"- Mean S std-dev in pool (reference): {diag_summary['mean_s_std_in_pool']:.4f}"
    )
    md_lines.append(
        f"- Queries with ZERO T variance in pool: {diag_summary['n_with_zero_t_var']}/{diag_summary['n_queries']}"
    )
    md_lines.append("")
    md_lines.append("## Conclusion (auto-generated)")
    md_lines.append("")
    sem_r1 = per_variant["SEMANTIC-ONLY"]["recall@1"]
    v7d_r1 = per_variant["V7 (T=0.5, S=0.5)"]["recall@1"]
    auto_r1 = best["recall@1"]
    if best["T"] <= 0.2:
        md_lines.append(
            f"- Auto-tune chose T={best['T']} (low T weight). V7 with default 0.5/0.5 lost {sem_r1 - v7d_r1:+.3f} R@1 vs SEMANTIC-ONLY."
        )
        md_lines.append(
            "- Verdict: **In dense time-cluster regime, T should be down-weighted or dropped.** Default V7 weights are not robust to this regime."
        )
    elif best["T"] == 0.0:
        md_lines.append(
            "- Auto-tune chose T=0.0 — i.e. T should be DROPPED entirely in this regime."
        )
    else:
        md_lines.append(
            f"- Auto-tune chose T={best['T']} — T retains some discriminative value even within the cluster."
        )
    md_lines.append("")
    md_lines.append("## Failure cases (SEMANTIC R@1 → V7 default lost rank)")
    md_lines.append("")
    if failures:
        md_lines.append("| qid | query | gold_text | sem | v7_def | v7_tr | auto |")
        md_lines.append("|---|---|---|---:|---:|---:|---:|")
        for f in failures[:10]:
            md_lines.append(
                f"| {f['qid']} | {f['query']} | {f['gold_text'][:60]} | {f['sem_rank']} | {f['v7_default_rank']} | {f['v7_tr_rank']} | {f['auto_rank']} |"
            )
    else:
        md_lines.append("(none — V7 default did not demote any sem-rank-1 below)")
    md_lines.append("")

    md_path = RESULTS_DIR / "dense_cluster.md"
    md_path.write_text("\n".join(md_lines))
    print(f"Wrote {md_path}", flush=True)

    # Print summary
    print("\n=== Summary ===")
    print(f"{'Variant':<46} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'MRR':>6}")
    for var, m in per_variant.items():
        print(
            f"{var:<46} {m['recall@1']:>6.3f} {m['recall@3']:>6.3f} {m['recall@5']:>6.3f} {m['mrr']:>6.3f}"
        )
    print(f"\nAuto-tune best: T={best['T']}, S={best['S']}")
    print(f"Mean T std-dev in candidate pool: {diag_summary['mean_t_std_in_pool']:.4f}")
    print(f"Cost: ${cost_total:.4f}, wall: {wall_s:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
