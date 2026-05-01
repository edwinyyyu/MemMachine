"""fuse_T_R + blind_pair on all 4 benchmarks.

Tests: does per-query LLM tuning of the (w_T, w_R) fusion weight help generalize
across distributions, or is fixed (0.4, 0.6) sufficient?

Variants:
  fuse_T_R (w_T=0.4)        — fixed
  fuse_T_R best fixed       — sweep w_T per benchmark, report best
  fuse_T_R + blind_pair     — model picks better of {prev, curr} per round
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
from salience_eval import (  # type: ignore
    AXES,
    DATA_DIR,
    AxisDistribution,
    build_memory,
    embed_all,
    interval_pair_best,
    parse_iso,
    rank_semantic,
    run_v2_extract,
    tag_score,
)
from v7l_ts_blind_eval import BlindJudge, metrics, run_pair

T_ALPHA, T_GAMMA, T_DELTA = 0.20, 0.20, 0.60
RERANK_TOP_K = 50
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def make_t_scores(q_mem, doc_mem, l_per_doc):
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
        l_sc = l_per_doc.get(did, 0.0)
        out[did] = (
            T_ALPHA * iv_norm
            + T_GAMMA * tag_score(q_tags, b["multi_tags"])
            + T_DELTA * l_sc
        )
    return out


def fuse_T_R_rank(t_scores, r_scores, s_scores_for_tail, w_T):
    """fuse_T_R: score_blend({T, R}, {w_T, 1-w_T}); tail with pure_S."""
    fused = score_blend(
        {"T": t_scores, "R": r_scores},
        {"T": w_T, "R": 1.0 - w_T},
        top_k_per=40,
        dispersion_cv_ref=0.20,
    )
    primary = [d for d, _ in fused]
    seen = set(primary)
    tail = sorted(s_scores_for_tail.items(), key=lambda x: x[1], reverse=True)
    return primary + [d for d, _ in tail if d not in seen]


def topk_from_scores(scores, k):
    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [d for d, _ in items[:k]]


# Patch base.rank_blend_ts so RetrievalCache uses fuse_T_R semantics
def _make_rank_blend_ts(t_scores, r_scores, s_scores):
    def fn(t_unused, s_unused, w_T):
        return fuse_T_R_rank(t_scores, r_scores, s_scores, w_T)

    return fn


class FusionCache:
    """Per-query cache: w_T → (ranked, top5_text) under fuse_T_R."""

    def __init__(self, t_scores, r_scores, s_scores, doc_text):
        self.t, self.r, self.s = t_scores, r_scores, s_scores
        self.doc_text = doc_text
        self._cache = {}

    def get(self, w_T):
        key = round(max(0.0, min(1.0, w_T)), 3)
        if key not in self._cache:
            ranked = fuse_T_R_rank(self.t, self.r, self.s, key)
            top5_text = [
                self.doc_text.get(d, "")[: base.MAX_TEXT_LEN] for d in ranked[:5]
            ]
            self._cache[key] = (ranked, top5_text)
        return self._cache[key]


async def rerank_topk(reranker, query_text, doc_ids, doc_text, k):
    cand_ids = doc_ids[:k]
    cand_texts = [doc_text.get(did, "")[:1000] for did in cand_ids]
    scores = await reranker.score(query_text, cand_texts)
    return {did: float(s) for did, s in zip(cand_ids, scores)}


WEIGHT_GRID = [round(i * 0.1, 1) for i in range(11)]


async def run_bench(
    name, docs_path, queries_path, gold_path, cache_doc, cache_q, reranker, judge
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

    lat_db = ROOT / "cache" / "fuse_blind" / f"lat_{name}.sqlite"
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

    per_q_t = {
        qid: make_t_scores(
            q_mem.get(qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()}),
            doc_mem,
            per_q_l.get(qid, {}),
        )
        for qid in qids
    }

    print(f"  reranking {RERANK_TOP_K} candidates per query...")
    per_q_r = {}
    for qid in qids:
        s_top = topk_from_scores(per_q_s[qid], RERANK_TOP_K)
        t_top = topk_from_scores(per_q_t[qid], RERANK_TOP_K)
        union = list(dict.fromkeys(s_top + t_top))[:RERANK_TOP_K]
        per_q_r[qid] = await rerank_topk(
            reranker, q_text[qid], union, doc_text, len(union)
        )

    # Variant 1: fixed at w_T=0.4
    fixed_04 = {
        qid: fuse_T_R_rank(per_q_t[qid], per_q_r[qid], per_q_s[qid], 0.4)
        for qid in qids
    }

    # Variant 2: best fixed per benchmark (sweep)
    best_w = 0.4
    best_m = None
    for w in WEIGHT_GRID:
        ranks = {
            qid: fuse_T_R_rank(per_q_t[qid], per_q_r[qid], per_q_s[qid], w)
            for qid in qids
        }
        m = metrics(ranks, gold, qids)
        if best_m is None or (m["r@5"], m["r@1"], m["mrr"]) > (
            best_m["r@5"],
            best_m["r@1"],
            best_m["mrr"],
        ):
            best_m = m
            best_w = w
    best_fixed = {
        qid: fuse_T_R_rank(per_q_t[qid], per_q_r[qid], per_q_s[qid], best_w)
        for qid in qids
    }

    # Variant 3: blind_pair tuning
    bp_results = {}

    async def run_one(qid):
        cache = FusionCache(per_q_t[qid], per_q_r[qid], per_q_s[qid], doc_text)
        r, _ = await run_pair(qid, q_text[qid], cache, judge, with_references=False)
        bp_results[qid] = r

    await asyncio.gather(*(run_one(qid) for qid in qids))

    m_fixed_04 = metrics(fixed_04, gold, qids)
    m_best = metrics(best_fixed, gold, qids)
    m_bp = metrics(bp_results, gold, qids)

    print(
        f"  fixed w_T=0.4              R@1={m_fixed_04['r@1']:.3f} R@5={m_fixed_04['r@5']:.3f} MRR={m_fixed_04['mrr']:.3f}"
    )
    print(
        f"  best fixed (w_T={best_w:.1f})        R@1={m_best['r@1']:.3f} R@5={m_best['r@5']:.3f} MRR={m_best['mrr']:.3f}"
    )
    print(
        f"  blind_pair                  R@1={m_bp['r@1']:.3f} R@5={m_bp['r@5']:.3f} MRR={m_bp['mrr']:.3f}"
    )
    return {"fixed_04": m_fixed_04, "best_fixed": (best_w, m_best), "blind_pair": m_bp}


async def main():
    print("Loading cross-encoder...")
    from memmachine_server.common.reranker.cross_encoder_reranker import (
        CrossEncoderReranker,
        CrossEncoderRerankerParams,
    )
    from sentence_transformers import CrossEncoder

    ce = CrossEncoder(CROSS_ENCODER_MODEL, device="cpu")
    reranker = CrossEncoderReranker(
        CrossEncoderRerankerParams(
            cross_encoder=ce,
            max_input_length=512,
        )
    )
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
            r = await run_bench(name, *paths, reranker=reranker, judge=judge)
            all_results[name] = r
        except Exception as e:
            print(f"  [{name}] failed: {e}")
            import traceback

            traceback.print_exc()

    judge.save()
    out_path = ROOT / "results" / "fuse_T_R_blind_pair.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                bn: {
                    "fixed_04": r["fixed_04"],
                    "best_fixed_w": r["best_fixed"][0],
                    "best_fixed": r["best_fixed"][1],
                    "blind_pair": r["blind_pair"],
                }
                for bn, r in all_results.items()
            },
            f,
            indent=2,
        )
    print(f"\nWrote {out_path}")
    print(f"\nLLM calls: {judge.calls}")

    print("\n=== SUMMARY (R@1) ===")
    print(
        f"{'Benchmark':22} {'fixed=0.4':>10} {'best fixed (w)':>16} {'blind_pair':>12}"
    )
    for bn, r in all_results.items():
        bw, bm = r["best_fixed"]
        print(
            f"{bn:22} {r['fixed_04']['r@1']:>10.3f} {bm['r@1']:.3f} (w={bw:.1f})  {r['blind_pair']['r@1']:>12.3f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
