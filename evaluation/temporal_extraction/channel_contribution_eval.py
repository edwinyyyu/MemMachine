"""Channel-contribution analysis.

Answers: how much does L add beyond T+S? Is just-L worse than just-T?

For each benchmark, computes R@1/R@5/MRR for:

  pure_S          — w_S=1, w_T=0, w_L=0 (semantic only)
  pure_T          — w_T=1, w_S=0, w_L=0 (T-channel only — interval Bhattacharyya)
  pure_L          — w_L=1, w_T=0, w_S=0 (lattice/axis tags only)

Then sweeps w_T over [0..1] in 0.1 steps under two parametrizations:

  T+S only        — w_L=0 throughout, w_S = 1 - w_T
  T+L+S "folded"  — w_L = (1-w_S)/2, w_T = (1-w_S)/2; single dial = w_S
                    Equivalently: w_temporal_input ∈ [0..1] split 50/50 T/L

Reports the best-fixed-weight configuration in each parametrization.

No LLM calls. ~1 minute total.
"""

from __future__ import annotations

import asyncio
import json
import sys
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


def rank_blend(t, s, l, w_T: float, w_S: float, w_L: float) -> list[str]:
    chans = {"T": t, "S": s, "L": l}
    weights = {"T": w_T, "S": w_S, "L": w_L}
    fused = score_blend(chans, weights, top_k_per=40, dispersion_cv_ref=0.20)
    ranked = [d for d, _ in fused]
    seen = set(ranked)
    tail = [
        d
        for d, _ in sorted(s.items(), key=lambda x: x[1], reverse=True)
        if d not in seen
    ]
    return ranked + tail


def metrics(rankings, gold, qids):
    r1 = r3 = r5 = r10 = 0
    mrr_sum = 0.0
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
        n += 1
    return {
        "r@1": r1 / n if n else 0,
        "r@3": r3 / n if n else 0,
        "r@5": r5 / n if n else 0,
        "r@10": r10 / n if n else 0,
        "mrr": mrr_sum / n if n else 0,
        "n": n,
    }


async def run_bench(name, docs_path, queries_path, gold_path, cache_doc, cache_q):
    docs = [json.loads(l) for l in open(DATA_DIR / docs_path)]
    queries = [json.loads(l) for l in open(DATA_DIR / queries_path)]
    gold_rows = [json.loads(l) for l in open(DATA_DIR / gold_path)]
    gold = {g["query_id"]: g["relevant_doc_ids"] for g in gold_rows}

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

    lat_db = ROOT / "cache" / "channel_contrib" / f"lat_{name}.sqlite"
    lat_db.parent.mkdir(parents=True, exist_ok=True)
    if lat_db.exists():
        lat_db.unlink()
    lat = LatticeStore(str(lat_db))
    ingest_lattice(lat, doc_ext)
    per_q_l = lattice_scores_for_query(lat, q_ext, [q["query_id"] for q in queries])

    qids = [q["query_id"] for q in queries]

    def metrics_at(weights_for_q):
        ranks = {
            qid: rank_blend(
                per_q_t[qid], per_q_s[qid], per_q_l[qid], *weights_for_q(qid)
            )
            for qid in qids
        }
        return metrics(ranks, gold, qids)

    pure_S = metrics_at(lambda qid: (0.0, 1.0, 0.0))
    pure_T = metrics_at(lambda qid: (1.0, 0.0, 0.0))
    pure_L = metrics_at(lambda qid: (0.0, 0.0, 1.0))

    print(
        f"  Single-channel R@1: pure_S={pure_S['r@1']:.3f}  pure_T={pure_T['r@1']:.3f}  pure_L={pure_L['r@1']:.3f}"
    )
    print(
        f"                R@5: pure_S={pure_S['r@5']:.3f}  pure_T={pure_T['r@5']:.3f}  pure_L={pure_L['r@5']:.3f}"
    )

    # T+S sweep (w_L=0)
    print("\n  T+S sweep (w_L=0):")
    print(f"  {'w_T':>5} {'w_S':>5} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'MRR':>6}")
    ts_grid = {}
    for i in range(11):
        w_T = round(i / 10, 1)
        w_S = round(1.0 - w_T, 1)
        m = metrics_at(lambda qid, wt=w_T, ws=w_S: (wt, ws, 0.0))
        ts_grid[w_T] = m
        print(
            f"  {w_T:>5.1f} {w_S:>5.1f} {m['r@1']:>6.3f} {m['r@3']:>6.3f} {m['r@5']:>6.3f} {m['mrr']:>6.3f}"
        )
    best_ts = max(
        ts_grid.items(), key=lambda kv: (kv[1]["r@5"], kv[1]["r@1"], kv[1]["mrr"])
    )

    # T+L+S folded sweep: w_temporal_input ∈ [0..1] → (w_T, w_L) = (w/2, w/2), w_S = 1-w
    print("\n  T+L+S folded sweep (w_T = w_L = w_temp/2, w_S = 1-w_temp):")
    print(f"  {'w_temp':>6} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'MRR':>6}")
    folded_grid = {}
    for i in range(11):
        w_temp = round(i / 10, 1)
        w_T = round(w_temp / 2, 3)
        w_L = round(w_temp / 2, 3)
        w_S = round(1.0 - w_temp, 3)
        m = metrics_at(lambda qid, wt=w_T, wl=w_L, ws=w_S: (wt, ws, wl))
        folded_grid[w_temp] = m
        print(
            f"  {w_temp:>6.1f} {m['r@1']:>6.3f} {m['r@3']:>6.3f} {m['r@5']:>6.3f} {m['mrr']:>6.3f}"
        )
    best_folded = max(
        folded_grid.items(), key=lambda kv: (kv[1]["r@5"], kv[1]["r@1"], kv[1]["mrr"])
    )

    print(
        f"\n  best T+S    : w_T={best_ts[0]:.1f}  R@1={best_ts[1]['r@1']:.3f} R@5={best_ts[1]['r@5']:.3f}"
    )
    print(
        f"  best folded : w_temp={best_folded[0]:.1f}  R@1={best_folded[1]['r@1']:.3f} R@5={best_folded[1]['r@5']:.3f}"
    )
    print(
        f"  L value-add over best T+S: ΔR@1={best_folded[1]['r@1'] - best_ts[1]['r@1']:+.3f}  "
        f"ΔR@5={best_folded[1]['r@5'] - best_ts[1]['r@5']:+.3f}"
    )

    return {
        "pure_S": pure_S,
        "pure_T": pure_T,
        "pure_L": pure_L,
        "T_S_grid": {k: v for k, v in ts_grid.items()},
        "best_T_S": {"w_T": best_ts[0], **best_ts[1]},
        "folded_grid": {k: v for k, v in folded_grid.items()},
        "best_folded": {"w_temp": best_folded[0], **best_folded[1]},
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

    out_path = ROOT / "results" / "channel_contribution.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")

    print("\n=== SUMMARY (R@1) ===")
    print(
        f"{'Benchmark':22} {'pure_S':>7} {'pure_T':>7} {'pure_L':>7} {'best_TS':>10} {'best_folded':>14} {'ΔR@1':>7}"
    )
    for bname, r in all_results.items():
        pS = r["pure_S"]["r@1"]
        pT = r["pure_T"]["r@1"]
        pL = r["pure_L"]["r@1"]
        bts = r["best_T_S"]
        bf = r["best_folded"]
        delta = bf["r@1"] - bts["r@1"]
        print(
            f"{bname:22} {pS:>7.3f} {pT:>7.3f} {pL:>7.3f} "
            f"w={bts['w_T']:.1f}/R@1={bts['r@1']:.3f}  "
            f"w={bf['w_temp']:.1f}/R@1={bf['r@1']:.3f}  "
            f"{delta:>+7.3f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
