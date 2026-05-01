"""Compare temporal channel (T_lblend) vs reranker vs fusion.

Variants:
  pure_S            - semantic embedding only
  T_lblend          - current best temporal recipe (no reranker)
  rerank_only       - pure_S top-K → cross-encoder rerank → final ranking
  T_lblend+rerank   - T_lblend's top-K → rerank → final ranking
  fuse_T_R          - score_blend({T_lblend, rerank, S}) — full hybrid

Cross-encoder: cross-encoder/ms-marco-MiniLM-L-6-v2 (local, ~80 MB).

Tests on all 4 benchmarks.
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
    axis_score_fn,
    build_memory,
    embed_all,
    interval_pair_best,
    parse_iso,
    rank_semantic,
    run_v2_extract,
    tag_score,
)

# T_lblend coefficients (winning blend)
T_ALPHA, T_BETA, T_GAMMA, T_DELTA = 0.20, 0.0, 0.20, 0.60
RERANK_TOP_K = 50  # rerank top-K from initial candidates
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def make_t_scores(
    q_mem, doc_mem, l_per_doc, alpha=T_ALPHA, beta=T_BETA, gamma=T_GAMMA, delta=T_DELTA
):
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


def rank_blend(channels, weights):
    fused = score_blend(channels, weights, top_k_per=40, dispersion_cv_ref=0.20)
    return [d for d, _ in fused]


def metrics(rankings, gold, qids):
    r1 = r5 = 0
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
            if hit <= 5:
                r5 += 1
            mrr_sum += 1.0 / hit
        n += 1
    return {
        "r@1": r1 / n if n else 0,
        "r@5": r5 / n if n else 0,
        "mrr": mrr_sum / n if n else 0,
        "n": n,
    }


def topk_from_scores(scores: dict[str, float], k: int) -> list[str]:
    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [d for d, _ in items[:k]]


async def rerank_topk(reranker, query_text, doc_ids, doc_text, k):
    """Rerank top-K doc_ids; return dict {doc_id: rerank_score}."""
    cand_ids = doc_ids[:k]
    cand_texts = [doc_text.get(did, "")[:1000] for did in cand_ids]
    scores = await reranker.score(query_text, cand_texts)
    return {did: float(s) for did, s in zip(cand_ids, scores)}


def topk_score_dict(d: dict[str, float], k: int) -> dict[str, float]:
    """Keep only top-K scores; others get 0."""
    items = sorted(d.items(), key=lambda x: x[1], reverse=True)[:k]
    return {did: s for did, s in items}


def merge_with_tail(primary_ranking, tail_score_dict):
    """Append docs from tail not already in primary, ordered by tail score."""
    seen = set(primary_ranking)
    tail = sorted(
        ((did, s) for did, s in tail_score_dict.items() if did not in seen),
        key=lambda x: x[1],
        reverse=True,
    )
    return primary_ranking + [d for d, _ in tail]


async def run_bench(
    name, docs_path, queries_path, gold_path, cache_doc, cache_q, reranker
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

    # Lattice for T_lblend
    lat_db = ROOT / "cache" / "rerank" / f"lat_{name}.sqlite"
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

    # T_lblend scores per query
    per_q_t = {
        qid: make_t_scores(
            q_mem.get(qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()}),
            doc_mem,
            per_q_l.get(qid, {}),
        )
        for qid in qids
    }

    # Compute candidate union: top-K of pure_S ∪ top-K of T_lblend
    print(
        f"  reranking {RERANK_TOP_K} candidates per query (~{RERANK_TOP_K * len(qids)} pairs)..."
    )
    per_q_rerank: dict[str, dict[str, float]] = {}
    for qid in qids:
        s_top = topk_from_scores(per_q_s[qid], RERANK_TOP_K)
        t_top = topk_from_scores(per_q_t[qid], RERANK_TOP_K)
        union = list(dict.fromkeys(s_top + t_top))[:RERANK_TOP_K]
        per_q_rerank[qid] = await rerank_topk(
            reranker, q_text[qid], union, doc_text, len(union)
        )

    # Variants
    variants = {}

    # 1. pure_S
    variants["pure_S"] = {
        qid: rank_blend({"S": per_q_s[qid]}, {"S": 1.0}) for qid in qids
    }

    # 2. T_lblend at best-fixed w_T per benchmark (sweep 0..1, pick best by R@5/R@1/MRR)
    best_w_T = 0.2
    best_m = None
    for w_try in [round(i * 0.1, 1) for i in range(11)]:
        ranks = {
            qid: rank_blend(
                {"T": per_q_t[qid], "S": per_q_s[qid]}, {"T": w_try, "S": 1.0 - w_try}
            )
            for qid in qids
        }
        m = metrics(ranks, gold, qids)
        if best_m is None or (m["r@5"], m["r@1"], m["mrr"]) > (
            best_m["r@5"],
            best_m["r@1"],
            best_m["mrr"],
        ):
            best_m = m
            best_w_T = w_try
    print(f"  T_lblend best fixed w_T = {best_w_T}")
    variants["T_lblend"] = {
        qid: rank_blend(
            {"T": per_q_t[qid], "S": per_q_s[qid]}, {"T": best_w_T, "S": 1.0 - best_w_T}
        )
        for qid in qids
    }

    # 3. Rerank-only: pure_S top-K → reranker scores → tail with pure_S
    rerank_only = {}
    for qid in qids:
        s_top_50 = topk_from_scores(per_q_s[qid], RERANK_TOP_K)
        rs = {did: per_q_rerank[qid].get(did, 0.0) for did in s_top_50}
        ranked = sorted(rs.items(), key=lambda x: x[1], reverse=True)
        primary = [d for d, _ in ranked]
        rerank_only[qid] = merge_with_tail(primary, per_q_s[qid])
    variants["rerank_only"] = rerank_only

    # 4. T_lblend → reranker on its top-K → re-sort by reranker
    t_then_rerank = {}
    for qid in qids:
        t_top = topk_from_scores(per_q_t[qid], RERANK_TOP_K)
        rs = {did: per_q_rerank[qid].get(did, 0.0) for did in t_top}
        ranked = sorted(rs.items(), key=lambda x: x[1], reverse=True)
        primary = [d for d, _ in ranked]
        t_then_rerank[qid] = merge_with_tail(primary, per_q_s[qid])
    variants["T_then_rerank"] = t_then_rerank

    # 5. Full fusion: score_blend({T, rerank, S}, {0.3, 0.4, 0.3})
    fuse_T_R_S = {}
    for qid in qids:
        # rerank scores limited to its candidates; default 0 elsewhere
        r_scores = per_q_rerank[qid]
        fuse_T_R_S[qid] = rank_blend(
            {"T": per_q_t[qid], "R": r_scores, "S": per_q_s[qid]},
            {"T": 0.3, "R": 0.4, "S": 0.3},
        )
        # Tail with pure_S
        seen = set(fuse_T_R_S[qid])
        tail = sorted(per_q_s[qid].items(), key=lambda x: x[1], reverse=True)
        fuse_T_R_S[qid] = fuse_T_R_S[qid] + [d for d, _ in tail if d not in seen]
    variants["fuse_T_R_S"] = fuse_T_R_S

    # 6. Fusion: T + Rerank only (drop direct S, since rerank already encodes semantic)
    fuse_T_R = {}
    for qid in qids:
        r_scores = per_q_rerank[qid]
        fuse_T_R[qid] = rank_blend(
            {"T": per_q_t[qid], "R": r_scores},
            {"T": 0.4, "R": 0.6},
        )
        seen = set(fuse_T_R[qid])
        tail = sorted(per_q_s[qid].items(), key=lambda x: x[1], reverse=True)
        fuse_T_R[qid] = fuse_T_R[qid] + [d for d, _ in tail if d not in seen]
    variants["fuse_T_R"] = fuse_T_R

    results = {var: metrics(ranks, gold, qids) for var, ranks in variants.items()}
    print(f"  {'Variant':16} {'R@1':>6} {'R@5':>6} {'MRR':>6}")
    for var, m in results.items():
        print(f"  {var:16} {m['r@1']:>6.3f} {m['r@5']:>6.3f} {m['mrr']:>6.3f}")
    return results


async def main():
    print("Loading cross-encoder model...")
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
            r = await run_bench(name, *paths, reranker=reranker)
            all_results[name] = r
        except Exception as e:
            print(f"  [{name}] failed: {e}")
            import traceback

            traceback.print_exc()

    out_path = ROOT / "results" / "t_vs_reranker.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {out_path}")

    print("\n=== SUMMARY (R@1) ===")
    cols = [
        "pure_S",
        "T_lblend",
        "rerank_only",
        "T_then_rerank",
        "fuse_T_R_S",
        "fuse_T_R",
    ]
    print(f"{'Benchmark':22}" + "".join(f"{c:>16}" for c in cols))
    for bname, r in all_results.items():
        row = [r[c]["r@1"] for c in cols]
        print(f"{bname:22}" + "".join(f"{x:>16.3f}" for x in row))


if __name__ == "__main__":
    asyncio.run(main())
