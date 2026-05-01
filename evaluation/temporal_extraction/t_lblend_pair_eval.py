"""Test the L-heavy blend (0.2/0.0/0.2/0.6) with blind_pair tuning on all 4 benchmarks.

Compares:
  T_full + blind_pair (current best ship recipe)
  T_lblend (0.2 interval + 0.2 tag + 0.6 lattice) fixed
  T_lblend + blind_pair
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import v7l_ts_blind_eval as base
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
from v7l_ts_blind_eval import BlindJudge, metrics, run_pair

T_VARIANTS = {
    "T_full": (0.5, 0.35, 0.15, 0.0),
    "T_lblend": (0.2, 0.0, 0.2, 0.6),
}


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


base.rank_blend_ts = rank_blend_ts


class CacheVariant:
    def __init__(self, t, s, doc_text):
        self.t, self.s, self.doc_text = t, s, doc_text
        self._cache = {}

    def get(self, w_T):
        key = round(max(0.0, min(1.0, w_T)), 3)
        if key not in self._cache:
            ranked = rank_blend_ts(self.t, self.s, key)
            top5_text = [
                self.doc_text.get(d, "")[: base.MAX_TEXT_LEN] for d in ranked[:5]
            ]
            self._cache[key] = (ranked, top5_text)
        return self._cache[key]


WEIGHT_GRID = [round(i * 0.1, 1) for i in range(11)]


async def run_bench(
    name, docs_path, queries_path, gold_path, cache_doc, cache_q, judge
):
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

    doc_text = {d["doc_id"]: d["text"] for d in docs}
    q_text = {q["query_id"]: q["text"] for q in queries}
    doc_texts = [d["text"] for d in docs]
    q_texts = [q["text"] for q in queries]
    doc_embs_arr = await embed_all(doc_texts)
    q_embs_arr = await embed_all(q_texts)
    doc_embs = {d["doc_id"]: doc_embs_arr[i] for i, d in enumerate(docs)}
    q_embs = {q["query_id"]: q_embs_arr[i] for i, q in enumerate(queries)}

    qids = [q["query_id"] for q in queries]
    per_q_s = {qid: rank_semantic(qid, q_embs, doc_embs) for qid in qids}

    # L scores
    lat_db = ROOT / "cache" / "lblend" / f"lat_{name}.sqlite"
    lat_db.parent.mkdir(parents=True, exist_ok=True)
    if lat_db.exists():
        lat_db.unlink()
    lat = LatticeStore(str(lat_db))
    for doc_id, tes in doc_ext.items():
        for te in tes:
            ts = lattice_tags_for_expression(te)
            lat.insert(doc_id, ts.absolute, ts.cyclical)
    per_q_l = {}
    for qid in qids:
        tes = q_ext.get(qid, [])
        scores, _ = lattice_retrieve_multi(lat, tes, down_levels=1)
        per_q_l[qid] = scores

    results = {}
    for vname, (a, b, g, d) in T_VARIANTS.items():
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

        # Best fixed w_T
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
        fixed_w, fixed_m = best

        # blind_pair
        opt_results = {}

        async def run_one(qid, ptt=per_q_t):
            cache = CacheVariant(ptt[qid], per_q_s[qid], doc_text)
            r, _ = await run_pair(qid, q_text[qid], cache, judge, with_references=False)
            opt_results[qid] = r

        await asyncio.gather(*(run_one(qid) for qid in qids))
        bm = metrics(opt_results, gold, qids)

        results[vname] = {
            "fixed_w_T": fixed_w,
            "fixed": {
                "r@1": fixed_m["r@1"],
                "r@5": fixed_m["r@5"],
                "mrr": fixed_m["mrr"],
            },
            "blind_pair": {"r@1": bm["r@1"], "r@5": bm["r@5"], "mrr": bm["mrr"]},
        }
        print(
            f"  {vname:10}  fixed (w={fixed_w:.1f}): R@1={fixed_m['r@1']:.3f} R@5={fixed_m['r@5']:.3f}  "
            f"|  + blind_pair: R@1={bm['r@1']:.3f} R@5={bm['r@5']:.3f}"
        )
    return results


async def main():
    judge = BlindJudge()
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
            r = await run_bench(name, *paths, judge=judge)
            all_results[name] = r
        except Exception as e:
            print(f"  [{name}] failed: {e}")
            import traceback

            traceback.print_exc()

    judge.save()
    out_path = ROOT / "results" / "t_lblend_pair.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"\nLLM calls: {judge.calls}, failed: {judge.failed}")

    print("\n=== SUMMARY (R@1) ===")
    print(
        f"{'Benchmark':22} {'T_full fixed':>13} {'T_full+pair':>13} {'T_lblend fixed':>15} {'T_lblend+pair':>15}"
    )
    for bname, r in all_results.items():
        ff = r["T_full"]["fixed"]["r@1"]
        fp = r["T_full"]["blind_pair"]["r@1"]
        lf = r["T_lblend"]["fixed"]["r@1"]
        lp = r["T_lblend"]["blind_pair"]["r@1"]
        print(f"{bname:22} {ff:>13.3f} {fp:>13.3f} {lf:>15.3f} {lp:>15.3f}")


if __name__ == "__main__":
    asyncio.run(main())
