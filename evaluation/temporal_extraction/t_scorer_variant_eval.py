"""T-scorer variant comparison across all 4 benchmarks.

Tests multiple T-scoring approaches as the T channel:

  T_iv_jcomposite   = interval pair-best with Jaccard-composite (current T_iv)
  T_iv_gaussian     = interval pair-best with pure Gaussian similarity
  T_iv_hard         = interval pair-best with binary overlap
  T_axis            = multi-axis Bhattacharyya alone (current best on hard_bench)
  T_full_default    = 0.5 jaccard_composite + 0.35 axis + 0.15 tag (current default)

For each scorer, sweep w_T over [0..1] per benchmark, report best fixed R@1.
This shows which scorer is uniformly competitive vs distribution-dependent.

If a scorer wins on one benchmark but regresses on others, flag distribution sensitivity.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rag_fusion import score_blend
from salience_eval import (  # type: ignore
    AXES,
    DATA_DIR,
    AxisDistribution,
    axis_score_fn,
    build_memory,
    embed_all,
    parse_iso,
    rank_semantic,
    run_v2_extract,
    tag_score,
)
from scorer import (
    score_gaussian,
    score_gaussian_integrated,
    score_hard_overlap,
    score_jaccard_composite,
)

SCORERS = {
    "jaccard_composite": score_jaccard_composite,
    "gaussian": score_gaussian,
    "gaussian_integrated": score_gaussian_integrated,
    "hard_overlap": score_hard_overlap,
}

WEIGHT_GRID = [round(i * 0.1, 1) for i in range(11)]


def interval_pair_best_with_scorer(q_ivs, d_ivs, scorer):
    if not q_ivs or not d_ivs:
        return 0.0
    total = 0.0
    for qi in q_ivs:
        best = 0.0
        for si in d_ivs:
            s = scorer(qi, si)
            if s > best:
                best = s
        total += best
    return total / len(q_ivs)


def t_iv_with_scorer(q_mem, doc_mem, scorer):
    q_ivs = q_mem.get("intervals") or []
    raw = {
        did: interval_pair_best_with_scorer(q_ivs, b["intervals"], scorer)
        for did, b in doc_mem.items()
    }
    max_v = max(raw.values()) if raw else 0.0
    return {did: (v / max_v if max_v > 0 else 0.0) for did, v in raw.items()}


def t_axis_only(q_mem, doc_mem):
    qa = q_mem.get("axes_merged") or {}
    return {did: axis_score_fn(qa, b["axes_merged"]) for did, b in doc_mem.items()}


def t_full_default(q_mem, doc_mem):
    """0.5 * jaccard_composite + 0.35 * axis + 0.15 * tag"""
    qa = q_mem.get("axes_merged") or {}
    q_tags = q_mem.get("multi_tags") or set()
    q_ivs = q_mem.get("intervals") or []
    raw = {
        did: interval_pair_best_with_scorer(
            q_ivs, b["intervals"], score_jaccard_composite
        )
        for did, b in doc_mem.items()
    }
    max_v = max(raw.values()) if raw else 0.0
    out = {}
    for did, b in doc_mem.items():
        iv_norm = raw[did] / max_v if max_v > 0 else 0.0
        a_sc = axis_score_fn(qa, b["axes_merged"])
        t_sc = tag_score(q_tags, b["multi_tags"])
        out[did] = 0.5 * iv_norm + 0.35 * a_sc + 0.15 * t_sc
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
        "r@5": r5 / n if n else 0,
        "mrr": mrr_sum / n if n else 0,
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

    qids = [q["query_id"] for q in queries]
    per_q_s = {qid: rank_semantic(qid, q_embs, doc_embs) for qid in qids}

    # Build T per scorer variant
    variants = {}
    for sname, scorer in SCORERS.items():
        per_q_t = {
            qid: t_iv_with_scorer(
                q_mem.get(
                    qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()}
                ),
                doc_mem,
                scorer,
            )
            for qid in qids
        }
        variants[f"iv_{sname}"] = per_q_t
    variants["axis_only"] = {
        qid: t_axis_only(
            q_mem.get(qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()}),
            doc_mem,
        )
        for qid in qids
    }
    variants["full_default"] = {
        qid: t_full_default(
            q_mem.get(qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()}),
            doc_mem,
        )
        for qid in qids
    }

    # pure_S baseline
    pure_S_rank = {qid: rank_blend_ts({}, per_q_s[qid], 0.0) for qid in qids}
    pure_S_m = metrics(pure_S_rank, gold, qids)

    # Sweep w_T per variant; report best fixed R@1
    print(f"  pure_S: R@1={pure_S_m['r@1']:.3f} R@5={pure_S_m['r@5']:.3f}")
    print(f"  {'T variant':22} {'best w_T':>9} {'R@1':>6} {'R@5':>6} {'MRR':>6}")
    bench_rows = {"pure_S": pure_S_m}
    for vname, per_q_t in variants.items():
        best = (None, None)
        for w_T in WEIGHT_GRID:
            ranks = {
                qid: rank_blend_ts(per_q_t[qid], per_q_s[qid], w_T) for qid in qids
            }
            m = metrics(ranks, gold, qids)
            if best[1] is None or (m["r@5"], m["r@1"], m["mrr"]) > (
                best[1]["r@5"],
                best[1]["r@1"],
                best[1]["mrr"],
            ):
                best = (w_T, m)
        bench_rows[vname] = {"best_w_T": best[0], **best[1]}
        print(
            f"  {vname:22} {best[0]:>9.1f} {best[1]['r@1']:>6.3f} {best[1]['r@5']:>6.3f} {best[1]['mrr']:>6.3f}"
        )
    return bench_rows


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

    out_path = ROOT / "results" / "t_scorer_variants.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")

    print("\n=== T SCORER COMPARISON (best fixed R@1 per benchmark) ===")
    cols = [
        "pure_S",
        "iv_jaccard_composite",
        "iv_gaussian",
        "iv_gaussian_integrated",
        "iv_hard_overlap",
        "axis_only",
        "full_default",
    ]
    print(f"{'Benchmark':22}" + "".join(f"{c[-15:]:>17}" for c in cols))
    for bname, r in all_results.items():
        row = []
        for c in cols:
            entry = r.get(c, {})
            v = entry.get("r@1") if isinstance(entry, dict) else 0.0
            row.append(f"{v:.3f}" if v else "  -  ")
        print(f"{bname:22}" + "".join(f"{x:>17}" for x in row))


if __name__ == "__main__":
    asyncio.run(main())
