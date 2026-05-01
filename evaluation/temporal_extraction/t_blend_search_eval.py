"""Search T-blend configurations including L (lattice) for cross-benchmark robustness.

Currently T_full = 0.5*interval + 0.35*axis + 0.15*tag (no L). Earlier work
rejected L because pure_L < pure_T everywhere — but that didn't test L as a
*minority component* in a blend. Maybe L adds value as a 4th component when
weighted modestly.

Tests blend configs (α_iv, β_axis, γ_tag, δ_L) summing to 1.0, on all 4 benchmarks.
For each config, sweeps w_T over [0..1], records best fixed R@1 per benchmark.

Robustness metric: sum-of-R@1 across benchmarks (higher = more robust).
A robust config wins on average without big regressions.
"""

from __future__ import annotations

import asyncio
import json
import sys
from itertools import product
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
    axis_score_fn,
    build_memory,
    embed_all,
    interval_pair_best,
    parse_iso,
    rank_semantic,
    run_v2_extract,
    tag_score,
)

WEIGHT_GRID = [round(i * 0.1, 1) for i in range(11)]


def _compute_l_scores(store, q_extracted, query_ids):
    out = {}
    for qid in query_ids:
        tes = q_extracted.get(qid, [])
        scores, _ = lattice_retrieve_multi(store, tes, down_levels=1)
        out[qid] = scores
    return out


def make_t_scores(q_mem, doc_mem, l_per_doc, alpha, beta, gamma, delta):
    """alpha*interval + beta*axis + gamma*tag + delta*L blend."""
    qa = q_mem.get("axes_merged") or {}
    q_tags = q_mem.get("multi_tags") or set()
    q_ivs = q_mem.get("intervals") or []
    raw_iv = {
        did: interval_pair_best(q_ivs, b["intervals"]) for did, b in doc_mem.items()
    }
    max_iv = max(raw_iv.values()) if raw_iv else 0.0
    out = {}
    for did, b in doc_mem.items():
        iv_norm = raw_iv[did] / max_iv if max_iv > 0 else 0.0
        a_sc = axis_score_fn(qa, b["axes_merged"]) if beta > 0 else 0.0
        t_sc = tag_score(q_tags, b["multi_tags"]) if gamma > 0 else 0.0
        l_sc = l_per_doc.get(did, 0.0) if delta > 0 else 0.0
        out[did] = alpha * iv_norm + beta * a_sc + gamma * t_sc + delta * l_sc
    return out


def rank_blend_ts(t, s, w_T):
    w_S = max(0.0, 1.0 - w_T)
    chans = {"T": t, "S": s}
    weights = {"T": w_T, "S": w_S}
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
        n += 1
    return {"r@1": r1 / n if n else 0, "r@5": r5 / n if n else 0, "n": n}


# Generate blend configs: (α, β, γ, δ) with each in {0, 0.2, 0.4, 0.6, 0.8, 1.0} summing to ~1.0
def gen_configs():
    grid = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    configs = []
    for a, b, g, d in product(grid, grid, grid, grid):
        if abs(a + b + g + d - 1.0) < 1e-6:
            configs.append((a, b, g, d))
    return configs


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

    qids = [q["query_id"] for q in queries]
    per_q_s = {qid: rank_semantic(qid, q_embs, doc_embs) for qid in qids}

    # Lattice store
    lat_db = ROOT / "cache" / "blend_search" / f"lat_{name}.sqlite"
    lat_db.parent.mkdir(parents=True, exist_ok=True)
    if lat_db.exists():
        lat_db.unlink()
    lat = LatticeStore(str(lat_db))
    for doc_id, tes in doc_ext.items():
        for te in tes:
            ts = lattice_tags_for_expression(te)
            lat.insert(doc_id, ts.absolute, ts.cyclical)
    per_q_l = _compute_l_scores(lat, q_ext, qids)

    configs = gen_configs()
    print(
        f"  Testing {len(configs)} blend configs × 11 w_T values = {len(configs) * 11} evaluations"
    )

    bench_results = {}
    for cfg in configs:
        a, b, g, d = cfg
        per_q_t = {
            qid: make_t_scores(
                q_mem.get(
                    qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()}
                ),
                doc_mem,
                per_q_l.get(qid, {}),
                a,
                b,
                g,
                d,
            )
            for qid in qids
        }
        best = (None, None)
        for w_T in WEIGHT_GRID:
            ranks = {
                qid: rank_blend_ts(per_q_t[qid], per_q_s[qid], w_T) for qid in qids
            }
            m = metrics(ranks, gold, qids)
            if best[1] is None or (m["r@5"], m["r@1"]) > (
                best[1]["r@5"],
                best[1]["r@1"],
            ):
                best = (w_T, m)
        bench_results[cfg] = {
            "best_w_T": best[0],
            "r@1": best[1]["r@1"],
            "r@5": best[1]["r@5"],
        }
    return bench_results


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

    # Aggregate: sum-of-R@1, min-R@1, mean-R@1 per config
    configs = list(next(iter(all_results.values())).keys())
    agg = []
    for cfg in configs:
        r1s = [all_results[bn][cfg]["r@1"] for bn in all_results]
        agg.append(
            {
                "cfg": cfg,
                "sum_r@1": sum(r1s),
                "min_r@1": min(r1s),
                "mean_r@1": sum(r1s) / len(r1s),
                "by_bench": {bn: all_results[bn][cfg]["r@1"] for bn in all_results},
            }
        )

    # Top-10 by sum
    agg.sort(key=lambda x: (x["sum_r@1"], x["min_r@1"]), reverse=True)

    print("\n=== TOP-15 BLEND CONFIGS BY SUM-R@1 (4 benchmarks) ===")
    print(
        f"{'config (α/β/γ/δ)':22} {'sum':>6} {'min':>6} {'mean':>6}  {'mixed_cue':>10} {'dense_cluster':>13} {'tempreason':>11} {'hard_bench':>11}"
    )
    for entry in agg[:15]:
        a, b, g, d = entry["cfg"]
        cfg_str = f"{a:.1f}/{b:.1f}/{g:.1f}/{d:.1f}"
        bb = entry["by_bench"]
        print(
            f"{cfg_str:22} {entry['sum_r@1']:>6.3f} {entry['min_r@1']:>6.3f} {entry['mean_r@1']:>6.3f}  "
            f"{bb['mixed_cue']:>10.3f} {bb['dense_cluster']:>13.3f} "
            f"{bb['tempreason_small']:>11.3f} {bb['hard_bench']:>11.3f}"
        )

    # T_full reference (0.5/0.35/0.15/0) — not in grid, compute separately
    print(
        "\n=== Reference: T_full (0.5/0.35/0.15/0) is NOT in grid (γ=0.15 not on 0.2 step). Closest grid configs above. ==="
    )

    # Best per single benchmark
    print("\n=== Per-benchmark best ===")
    for bn in all_results:
        per_b = sorted(
            all_results[bn].items(),
            key=lambda kv: (kv[1]["r@5"], kv[1]["r@1"]),
            reverse=True,
        )[:3]
        print(f"  {bn}:")
        for cfg, m in per_b:
            a, b, g, d = cfg
            print(
                f"    {a:.1f}/{b:.1f}/{g:.1f}/{d:.1f}  R@1={m['r@1']:.3f} R@5={m['r@5']:.3f} (w_T={m['best_w_T']})"
            )

    out_path = ROOT / "results" / "t_blend_search.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "all_results": {
                    bn: {
                        f"{cfg[0]}_{cfg[1]}_{cfg[2]}_{cfg[3]}": v
                        for cfg, v in d.items()
                    }
                    for bn, d in all_results.items()
                },
                "top_by_sum": agg[:30],
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
