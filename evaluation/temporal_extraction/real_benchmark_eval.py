"""Run the v2pp + multi-axis + lattice + V7L SCORE-BLEND pipeline on a
real-distribution temporal retrieval benchmark (TempReason-derived).

Compares:
- SEMANTIC-ONLY  (text-embedding-3-small cosine) — baseline
- V7-noL         (T + S, weights 0.4/0.4) — pipeline minus lattice
- V7L            (T + S + L, weights 0.3/0.3/0.2) — ship-best
  (Allen and Era channels are zero / skipped, mirroring lattice_eval.)

Usage: uv run python real_benchmark_eval.py
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

ROOT = Path("/Users/eyu/edwinyyyu/mmcc/extra_memory/evaluation/temporal_extraction")
sys.path.insert(0, str(ROOT))

from axis_distributions import (
    AXES,
    AxisDistribution,
    axes_for_expression,
    merge_axis_dists,
)
from baselines import embed_all
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
CACHE_DIR = ROOT / "cache" / "real_benchmark"
CACHE_DIR.mkdir(exist_ok=True, parents=True)
LATTICE_DB = CACHE_DIR / "lattice.sqlite"

TOP_K = 10
LLM_CALL_TIMEOUT_S = 60.0  # per-API-call timeout (passed to OpenAI client)
CONCURRENCY = 12  # higher concurrency drains the queue faster
COST_CAP_USD = 5.0
HARD_STOP_USD = 4.0  # stop extraction if cumulative > this


def _patched_client() -> AsyncOpenAI:
    return AsyncOpenAI(timeout=LLM_CALL_TIMEOUT_S, max_retries=1)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# Interval flattening
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
        "n": len(r5),
    }


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
async def run_v2pp_extract(
    items, cache_file: Path, label: str, cost_so_far_usd: float = 0.0
):
    ex = ExtractorV2PP(concurrency=CONCURRENCY)
    from extractor_common import LLMCache

    ex.cache = LLMCache(cache_file)
    ex.client = _patched_client()

    results: dict[str, list[TimeExpression]] = {}
    completed = [0]
    total = len(items)

    async def one(iid, text, ref):
        try:
            tes = await ex.extract(text, ref)
        except Exception as e:
            print(f"  [{label}] FAIL {iid}: {e}", flush=True)
            tes = []
        completed[0] += 1
        if completed[0] % 50 == 0:
            print(f"  [{label}] {completed[0]}/{total}", flush=True)
        return iid, tes

    print(f"v2pp {label}: {total} items", flush=True)
    pairs = await asyncio.gather(*(one(*it) for it in items))
    for iid, tes in pairs:
        results[iid] = tes
    ex.cache.save()
    try:
        ex.shared_pass2_cache.save()
    except Exception:
        pass

    cost = ex.usage["input"] * 0.25 / 1_000_000 + ex.usage["output"] * 2.0 / 1_000_000
    print(
        f"  [{label}] usage in={ex.usage['input']}, out={ex.usage['output']}, cost=${cost:.4f}"
    )
    return results, ex.usage


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


def rank_v7l(t, s, l, weights=None):
    if weights is None:
        weights = {"T": 0.3, "S": 0.3, "L": 0.2}
    fused = score_blend({"T": t, "S": s, "L": l}, weights, top_k_per=40)
    ranked = [d for d, _ in fused]
    seen = set(ranked)
    tail = [
        d
        for d, _ in sorted(s.items(), key=lambda x: x[1], reverse=True)
        if d not in seen
    ]
    return ranked + tail


def rank_v7_noL(t, s, weights=None):
    if weights is None:
        weights = {"T": 0.4, "S": 0.4}
    fused = score_blend({"T": t, "S": s}, weights, top_k_per=40)
    ranked = [d for d, _ in fused]
    seen = set(ranked)
    tail = [
        d
        for d, _ in sorted(s.items(), key=lambda x: x[1], reverse=True)
        if d not in seen
    ]
    return ranked + tail


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    t0 = time.time()

    # Load
    docs = load_jsonl(DATA_DIR / "real_benchmark_docs.jsonl")
    queries = load_jsonl(DATA_DIR / "real_benchmark_queries.jsonl")
    gold_raw = load_jsonl(DATA_DIR / "real_benchmark_gold.jsonl")
    gold = {g["query_id"]: set(g["relevant_doc_ids"]) for g in gold_raw}
    subset_of_q = {q["query_id"]: q["subset"] for q in queries}

    print(f"Real benchmark: {len(docs)} docs, {len(queries)} queries")

    # Extraction — v2pp on docs and queries
    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs]
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries]

    cache_file = CACHE_DIR / "v2pp_cache.json"
    doc_ext, doc_usage = await run_v2pp_extract(doc_items, cache_file, "real-docs")
    cost_after_docs = doc_usage["input"] * 0.25 / 1e6 + doc_usage["output"] * 2.0 / 1e6
    print(f"Cost after docs: ${cost_after_docs:.4f}")
    if cost_after_docs > HARD_STOP_USD:
        print(
            f"COST CAP HIT (${cost_after_docs:.4f} > ${HARD_STOP_USD}); aborting before queries."
        )
        return

    q_ext, q_usage = await run_v2pp_extract(q_items, cache_file, "real-queries")
    total_in = doc_usage["input"] + q_usage["input"]
    total_out = doc_usage["output"] + q_usage["output"]
    cost_total = total_in * 0.25 / 1e6 + total_out * 2.0 / 1e6
    print(f"Total LLM cost: ${cost_total:.4f}")

    # Build memories
    print("Building T-channel memory...")
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

    # Build lattice
    print("Building lattice index...")
    if LATTICE_DB.exists():
        LATTICE_DB.unlink()
    store = LatticeStore(LATTICE_DB)
    ingest_lattice(store, doc_ext)
    lat_stats = store.stats()
    print(f"Lattice: {lat_stats}")

    # Embeddings
    print("Embedding (text-embedding-3-small)...")
    doc_texts = [d["text"] for d in docs]
    q_texts = [q["text"] for q in queries]
    doc_embs_arr = await embed_all(doc_texts)
    q_embs_arr = await embed_all(q_texts)
    doc_embs = {d["doc_id"]: doc_embs_arr[i] for i, d in enumerate(docs)}
    q_embs = {q["query_id"]: q_embs_arr[i] for i, q in enumerate(queries)}

    # Retrieve
    print("Scoring per-query...")
    all_qids = {q["query_id"] for q in queries}
    l_scores_per_q = retrieve_lattice_scores(store, q_ext, all_qids)

    variants: dict[str, dict[str, list[str]]] = {
        "SEMANTIC-ONLY": {},
        "V7 (T+S, no L)": {},
        "V7L (T+S+L, ship-best)": {},
    }

    for q in queries:
        qid = q["query_id"]
        t_scores = rank_multi_axis_t(
            q_mem.get(qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()}),
            doc_mem,
        )
        s_scores = rank_semantic_s(qid, q_embs, doc_embs)
        l_scores = l_scores_per_q.get(qid, {})

        # Semantic-only ranking
        sem_ranked = sorted(s_scores.items(), key=lambda x: x[1], reverse=True)
        variants["SEMANTIC-ONLY"][qid] = [d for d, _ in sem_ranked]
        variants["V7 (T+S, no L)"][qid] = rank_v7_noL(t_scores, s_scores)
        variants["V7L (T+S+L, ship-best)"][qid] = rank_v7l(t_scores, s_scores, l_scores)

    # Eval
    L2_qids = {qid for qid, sub in subset_of_q.items() if sub == "L2"}
    L3_qids = {qid for qid, sub in subset_of_q.items() if sub == "L3"}
    subsets = {"all": all_qids, "L2": L2_qids, "L3": L3_qids}

    per_variant: dict[str, dict[str, dict[str, float]]] = {}
    for var, ranked_per_q in variants.items():
        per_variant[var] = {}
        for sub_name, qids in subsets.items():
            per_variant[var][sub_name] = eval_rankings(ranked_per_q, gold, qids)

    # Failure analysis: pick failures from each system
    failures: dict[str, list[dict]] = {}
    for var, ranked_per_q in variants.items():
        var_fails = []
        for q in queries:
            qid = q["query_id"]
            rel = gold.get(qid, set())
            ranked = ranked_per_q.get(qid, [])
            if not (set(ranked[:5]) & rel):
                gold_doc_text = next(
                    (d["text"] for d in docs if d["doc_id"] in rel), "<missing>"
                )
                top5_texts = [
                    next((d["text"] for d in docs if d["doc_id"] == did), "<?>")
                    for did in ranked[:5]
                ]
                var_fails.append(
                    {
                        "qid": qid,
                        "subset": q["subset"],
                        "query": q["text"],
                        "ref_time": q["ref_time"],
                        "gold_doc_id": list(rel)[0] if rel else None,
                        "gold_doc_text": gold_doc_text,
                        "top5_doc_ids": ranked[:5],
                        "top5_doc_texts": top5_texts,
                        "n_q_extractions": len(q_ext.get(qid, [])),
                        "n_gold_extractions": (
                            len(doc_ext.get(list(rel)[0], [])) if rel else 0
                        ),
                    }
                )
        failures[var] = var_fails

    wall_s = time.time() - t0

    # Write JSON
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
        "benchmark": {
            "name": "TempReason-derived (test L2 + L3)",
            "n_docs": len(docs),
            "n_queries": len(queries),
            "n_L2": len(L2_qids),
            "n_L3": len(L3_qids),
            "source": "tonytan48/TempReason test_l2.json + test_l3.json",
        },
        "per_variant": _clean(per_variant),
        "failures_per_variant": {
            var: {"n": len(f), "samples": _clean(f[:8])} for var, f in failures.items()
        },
        "lattice_stats": _clean(lat_stats),
        "cost": {
            "input_tokens": total_in,
            "output_tokens": total_out,
            "usd": cost_total,
        },
        "wall_seconds": wall_s,
    }
    out_json = RESULTS_DIR / "real_benchmark.json"
    out_json.write_text(json.dumps(results_json, indent=2, default=str))
    print(f"\nWrote {out_json}")

    # Print summary
    print("\n=== Summary ===")
    print(
        f"{'Variant':<30} {'subset':<6} {'R@5':>7} {'R@10':>7} {'MRR':>7} {'NDCG':>7}"
    )
    for var in ["SEMANTIC-ONLY", "V7 (T+S, no L)", "V7L (T+S+L, ship-best)"]:
        for sub in ["all", "L2", "L3"]:
            m = per_variant[var][sub]
            print(
                f"{var:<30} {sub:<6} {m['recall@5']:>7.3f} {m['recall@10']:>7.3f} "
                f"{m['mrr']:>7.3f} {m['ndcg@10']:>7.3f}"
            )
    print(f"\nCost: ${cost_total:.4f}, wall: {wall_s:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
