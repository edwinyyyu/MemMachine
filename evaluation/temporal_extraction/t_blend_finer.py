"""Finer-grained blend search around (0.2, 0.0, 0.2, 0.6).

Confirms whether T_lblend is a real optimum or a coarse-grid artifact.
Also reports pure-component baselines (0/0/0/1, etc.) for sanity.
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
from salience_eval import (
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


def make_t_scores(q_mem, doc_mem, l_per_doc, alpha, beta, gamma, delta):
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
    r1 = r5 = 0
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
            if hit <= 5:
                r5 += 1
        n += 1
    return {"r@1": r1 / n if n else 0, "r@5": r5 / n if n else 0}


# Finer grid: 0.05 steps near (0.2, 0, 0.2, 0.6)
def gen_configs():
    grid = [round(x * 0.05, 2) for x in range(21)]  # 0.0..1.0 in 0.05 steps
    configs = []
    # Constrained to neighborhood of (0.2, 0, 0.2, 0.6)
    for a in [0.1, 0.15, 0.2, 0.25, 0.3]:
        for b in [0.0, 0.05, 0.1, 0.15]:
            for g in [0.1, 0.15, 0.2, 0.25, 0.3]:
                d = round(1.0 - a - b - g, 4)
                if 0.4 <= d <= 0.75:
                    configs.append((a, b, g, d))
    # Also include pure-component baselines
    pure_baselines = [
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
        (0.5, 0.0, 0.0, 0.5),
        (0.0, 0.5, 0.0, 0.5),
        (0.0, 0.0, 0.5, 0.5),
        (0.5, 0.35, 0.15, 0.0),  # T_full reference
        (0.2, 0.0, 0.2, 0.6),  # Coarse-grid winner
    ]
    seen = set(configs)
    for c in pure_baselines:
        if c not in seen:
            configs.append(c)
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

    lat_db = ROOT / "cache" / "blend_finer" / f"lat_{name}.sqlite"
    lat_db.parent.mkdir(parents=True, exist_ok=True)
    if lat_db.exists():
        lat_db.unlink()
    lat = LatticeStore(str(lat_db))
    for doc_id, tes in doc_ext.items():
        for te in tes:
            ts = lattice_tags_for_expression(te)
            lat.insert(doc_id, ts.absolute, ts.cyclical)
    per_q_l = {
        qid: lattice_retrieve_multi(lat, q_ext.get(qid, []), down_levels=1)[0]
        for qid in qids
    }

    configs = gen_configs()
    print(f"  Testing {len(configs)} configs × 11 w_T values")

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

    configs = list(next(iter(all_results.values())).keys())
    agg = []
    for cfg in configs:
        r1s = [all_results[bn][cfg]["r@1"] for bn in all_results]
        agg.append(
            {
                "cfg": cfg,
                "sum_r1": sum(r1s),
                "min_r1": min(r1s),
                "by_bench": {bn: all_results[bn][cfg]["r@1"] for bn in all_results},
            }
        )
    agg.sort(key=lambda x: (x["sum_r1"], x["min_r1"]), reverse=True)

    print("\n=== TOP-15 (finer grid + pure baselines) ===")
    print(
        f"{'config (α/β/γ/δ)':22} {'sum':>6} {'min':>6}  {'mixed':>6} {'dense':>6} {'temp':>6} {'hard':>6}"
    )
    for entry in agg[:15]:
        a, b, g, d = entry["cfg"]
        cfg_str = f"{a:.2f}/{b:.2f}/{g:.2f}/{d:.2f}"
        bb = entry["by_bench"]
        print(
            f"{cfg_str:22} {entry['sum_r1']:>6.3f} {entry['min_r1']:>6.3f}  "
            f"{bb['mixed_cue']:>6.3f} {bb['dense_cluster']:>6.3f} "
            f"{bb['tempreason_small']:>6.3f} {bb['hard_bench']:>6.3f}"
        )

    print("\n=== Pure-component baselines for sanity ===")
    pure = [
        ("iv only", (1.0, 0.0, 0.0, 0.0)),
        ("axis only", (0.0, 1.0, 0.0, 0.0)),
        ("tag only", (0.0, 0.0, 1.0, 0.0)),
        ("L only", (0.0, 0.0, 0.0, 1.0)),
        ("T_full", (0.5, 0.35, 0.15, 0.0)),
        ("T_lblend (coarse winner)", (0.2, 0.0, 0.2, 0.6)),
    ]
    print(f"{'name':28} {'mixed':>6} {'dense':>6} {'temp':>6} {'hard':>6} {'sum':>6}")
    for label, cfg in pure:
        if cfg in {entry["cfg"] for entry in agg}:
            entry = next(e for e in agg if e["cfg"] == cfg)
            bb = entry["by_bench"]
            print(
                f"{label:28} {bb['mixed_cue']:>6.3f} {bb['dense_cluster']:>6.3f} "
                f"{bb['tempreason_small']:>6.3f} {bb['hard_bench']:>6.3f} {entry['sum_r1']:>6.3f}"
            )

    out_path = ROOT / "results" / "t_blend_finer.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"top_by_sum": agg[:30]}, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
