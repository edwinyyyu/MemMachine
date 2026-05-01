"""Per-query weight-adaptation upper bound.

Sweeps w_T over a grid per query. For each query, finds the BEST w_T (the one
that ranks the gold doc highest, ties broken by lowest w_T deviation from 0.4).
Reports:
  - V7L (fixed 0.4) baseline metrics
  - Oracle (per-query optimal w_T) metrics
  - Distribution of optimal w_T values per benchmark

If the oracle gap over baseline is small, per-query weight adaptation is a dead
end regardless of oracle quality. If large, the LLM's failure was about
information access (corpus stats) rather than architecture.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


from lattice_cells import tags_for_expression as lattice_tags_for_expression
from lattice_retrieval import retrieve_multi as lattice_retrieve_multi
from lattice_store import LatticeStore
from rag_fusion import score_blend
from salience_eval import (  # type: ignore
    AXES,
    DATA_DIR,
    AxisDistribution,
    build_memory,
    embed_all,
    parse_iso,
    rank_semantic,
    run_v2_extract,
)
from salience_eval import (
    rank_t as rank_multi_axis_t,
)

WEIGHT_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
W_L = 0.2


def ingest_lattice(store: LatticeStore, extracted) -> None:
    for doc_id, tes in extracted.items():
        for te in tes:
            ts = lattice_tags_for_expression(te)
            store.insert(doc_id, ts.absolute, ts.cyclical)


def lattice_scores_for_query(store, q_extracted, query_ids):
    out = {}
    for qid in query_ids:
        tes = q_extracted.get(qid, [])
        scores, _ = lattice_retrieve_multi(store, tes, down_levels=1)
        out[qid] = scores
    return out


def rank_blend(t, s, l, w_T: float) -> list[str]:
    w_S = max(0.0, 1.0 - w_T - W_L)
    chans = {"T": t, "S": s, "L": l}
    weights = {"T": w_T, "S": w_S, "L": W_L}
    fused = score_blend(chans, weights, top_k_per=40, dispersion_cv_ref=0.20)
    ranked = [d for d, _ in fused]
    seen = set(ranked)
    tail = [
        d
        for d, _ in sorted(s.items(), key=lambda x: x[1], reverse=True)
        if d not in seen
    ]
    return ranked + tail


def hit_rank(ranking: list[str], gold: set[str], k: int = 10) -> int | None:
    for i, d in enumerate(ranking[:k]):
        if d in gold:
            return i + 1
    return None


def best_w_T_for_query(t, s, l, gold: set[str]) -> tuple[float, int | None]:
    """Pick the w_T from grid that puts gold highest. Tie-break: closest to 0.4."""
    best = (None, 11, 1.0)  # (w_T, hit_rank, dist_from_0.4)
    for w_T in WEIGHT_GRID:
        r = rank_blend(t, s, l, w_T)
        h = hit_rank(r, gold, k=10)
        if h is None:
            h = 11
        dist = abs(w_T - 0.4)
        cur = (w_T, h, dist)
        if (h, dist) < (best[1], best[2]):
            best = cur
    return best[0], (best[1] if best[1] <= 10 else None)


def metrics(rankings, gold, qids):
    r1 = r3 = r5 = r10 = 0
    mrr_sum = 0.0
    ndcg_sum = 0.0
    n = 0
    for qid in qids:
        rel = set(gold.get(qid, []))
        if not rel:
            continue
        r = rankings.get(qid, [])
        hit = None
        for i, d in enumerate(r[:10]):
            if d in rel:
                hit = i + 1
                break
        if hit:
            if hit <= 1:
                r1 += 1
            if hit <= 3:
                r3 += 1
            if hit <= 5:
                r5 += 1
            if hit <= 10:
                r10 += 1
            mrr_sum += 1.0 / hit
            dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(r[:10]) if d in rel)
            ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), 10)))
            ndcg_sum += dcg / ideal if ideal else 0.0
        n += 1
    return {
        "n": n,
        "r@1": r1 / n if n else 0,
        "r@3": r3 / n if n else 0,
        "r@5": r5 / n if n else 0,
        "r@10": r10 / n if n else 0,
        "mrr": mrr_sum / n if n else 0,
        "ndcg@10": ndcg_sum / n if n else 0,
    }


async def run_bench(name, docs_path, queries_path, gold_path, cache_doc, cache_q):
    docs = [json.loads(l) for l in open(DATA_DIR / docs_path)]
    queries = [json.loads(l) for l in open(DATA_DIR / queries_path)]
    gold_rows = [json.loads(l) for l in open(DATA_DIR / gold_path)]
    gold = {g["query_id"]: set(g["relevant_doc_ids"]) for g in gold_rows}

    print(f"\n=== {name}: {len(docs)} docs, {len(queries)} queries ===")

    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs]
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries]

    doc_ext = await run_v2_extract(doc_items, f"{name}-docs", cache_doc)
    q_ext = await run_v2_extract(q_items, f"{name}-queries", cache_q)

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

    doc_texts = [d["text"] for d in docs]
    q_texts = [q["text"] for q in queries]
    doc_embs_arr = await embed_all(doc_texts)
    q_embs_arr = await embed_all(q_texts)
    doc_embs = {d["doc_id"]: doc_embs_arr[i] for i, d in enumerate(docs)}
    q_embs = {q["query_id"]: q_embs_arr[i] for i, q in enumerate(queries)}

    per_q_t, per_q_s = {}, {}
    for q in queries:
        qid = q["query_id"]
        per_q_t[qid] = rank_multi_axis_t(
            q_mem.get(qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()}),
            doc_mem,
        )
        per_q_s[qid] = rank_semantic(qid, q_embs, doc_embs)

    lat_db = ROOT / "cache" / "iter_weight" / f"lat_{name}.sqlite"
    lat_db.parent.mkdir(parents=True, exist_ok=True)
    if lat_db.exists():
        lat_db.unlink()
    lat = LatticeStore(str(lat_db))
    ingest_lattice(lat, doc_ext)
    per_q_l = lattice_scores_for_query(lat, q_ext, [q["query_id"] for q in queries])

    qids = [q["query_id"] for q in queries]

    # Baseline: fixed w_T=0.4
    baseline = {
        qid: rank_blend(per_q_t[qid], per_q_s[qid], per_q_l[qid], w_T=0.4)
        for qid in qids
    }

    # Oracle: per-query best w_T from grid
    oracle = {}
    optimal_w_T = {}
    for qid in qids:
        rel = gold.get(qid, set())
        if not rel:
            optimal_w_T[qid] = 0.4
            oracle[qid] = baseline[qid]
            continue
        w_T, _ = best_w_T_for_query(per_q_t[qid], per_q_s[qid], per_q_l[qid], rel)
        optimal_w_T[qid] = w_T
        oracle[qid] = rank_blend(per_q_t[qid], per_q_s[qid], per_q_l[qid], w_T=w_T)

    # Also compute per-w_T grid metrics (corpus-level sweep)
    grid_metrics = {}
    for w_T in WEIGHT_GRID:
        gridded = {
            qid: rank_blend(per_q_t[qid], per_q_s[qid], per_q_l[qid], w_T=w_T)
            for qid in qids
        }
        grid_metrics[w_T] = metrics(
            gridded, {k: list(v) for k, v in gold.items()}, qids
        )

    base_m = metrics(baseline, {k: list(v) for k, v in gold.items()}, qids)
    oracle_m = metrics(oracle, {k: list(v) for k, v in gold.items()}, qids)

    # Best fixed w_T (best single weight across the corpus)
    best_fixed = max(
        grid_metrics.items(), key=lambda kv: (kv[1]["r@5"], kv[1]["r@1"], kv[1]["mrr"])
    )

    print(f"{'Weight':>8} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'MRR':>6}")
    for w_T, m in grid_metrics.items():
        marker = "  <- best fixed" if w_T == best_fixed[0] else ""
        marker += "  (baseline)" if w_T == 0.4 else ""
        print(
            f"{w_T:>8.2f} {m['r@1']:>6.3f} {m['r@3']:>6.3f} {m['r@5']:>6.3f} {m['mrr']:>6.3f}{marker}"
        )
    print(
        f"{'ORACLE':>8} {oracle_m['r@1']:>6.3f} {oracle_m['r@3']:>6.3f} {oracle_m['r@5']:>6.3f} {oracle_m['mrr']:>6.3f}"
    )

    # Distribution of optimal w_T
    ctr = Counter(optimal_w_T.values())
    print(f"  optimal w_T distribution: {dict(sorted(ctr.items()))}")
    print(
        f"  oracle vs baseline: ΔR@1={oracle_m['r@1'] - base_m['r@1']:+.3f}  "
        f"ΔR@5={oracle_m['r@5'] - base_m['r@5']:+.3f}  ΔMRR={oracle_m['mrr'] - base_m['mrr']:+.3f}"
    )
    print(
        f"  best fixed w_T={best_fixed[0]:.2f}: R@1={best_fixed[1]['r@1']:.3f} R@5={best_fixed[1]['r@5']:.3f}"
    )

    return {
        "baseline": base_m,
        "oracle": oracle_m,
        "grid": grid_metrics,
        "best_fixed_w_T": best_fixed[0],
        "best_fixed_metrics": best_fixed[1],
        "optimal_w_T_distribution": dict(ctr),
    }


async def main():
    benches = [
        (
            "mixed_cue",
            "mixed_cue_docs.jsonl",
            "mixed_cue_queries.jsonl",
            "mixed_cue_gold.jsonl",
            "v7l-mixed_cue",
            "v7l-mixed_cue",
        ),
        (
            "dense_cluster",
            "dense_cluster_docs.jsonl",
            "dense_cluster_queries.jsonl",
            "dense_cluster_gold.jsonl",
            "v7l-dense_cluster",
            "v7l-dense_cluster",
        ),
        (
            "tempreason_small",
            "real_benchmark_small_docs.jsonl",
            "real_benchmark_small_queries.jsonl",
            "real_benchmark_small_gold.jsonl",
            "v7l-tempreason",
            "v7l-tempreason",
        ),
        (
            "hard_bench",
            "hard_bench_docs.jsonl",
            "hard_bench_queries.jsonl",
            "hard_bench_gold.jsonl",
            "v7l-hard_bench",
            "v7l-hard_bench",
        ),
    ]
    all_results = {}
    for name, *paths in benches:
        try:
            r = await run_bench(name, *paths)
            all_results[name] = r
        except Exception as e:
            print(f"  [{name}] failed: {e}")
            import traceback

            traceback.print_exc()

    out_path = ROOT / "results" / "v7l_weight_oracle.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")

    print("\n=== UPPER-BOUND SUMMARY ===")
    print(
        f"{'Benchmark':22} {'Base R@1':>10} {'Oracle R@1':>11} {'ΔR@1':>7} {'BestFix':>8} {'Optimal w_T mode':>18}"
    )
    for bname, r in all_results.items():
        base_r1 = r["baseline"]["r@1"]
        orc_r1 = r["oracle"]["r@1"]
        delta = orc_r1 - base_r1
        best_fix = r["best_fixed_w_T"]
        mode_w = max(r["optimal_w_T_distribution"].items(), key=lambda x: x[1])[0]
        print(
            f"{bname:22} {base_r1:>10.3f} {orc_r1:>11.3f} {delta:>+7.3f} {best_fix:>8.2f} {mode_w:>18.2f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
