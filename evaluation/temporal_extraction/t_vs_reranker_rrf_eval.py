"""Compare fusion strategies for {T_lblend, cross-encoder rerank}:
  - score_blend at multiple (w_T, w_R) ratios
  - RRF (uniform) at multiple k values
  - Weighted RRF at multiple weight ratios

All on hard_bench (only benchmark with headroom). Reuses cached cross-encoder
scores from prior run if hashes match.
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
    interval_pair_best,
    parse_iso,
    rank_semantic,
    run_v2_extract,
    tag_score,
)

T_ALPHA, T_BETA, T_GAMMA, T_DELTA = 0.20, 0.0, 0.20, 0.60
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
    }


def topk_from_scores(scores: dict[str, float], k: int) -> list[str]:
    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [d for d, _ in items[:k]]


def weighted_rrf(ranked_lists_with_weights, k=60):
    """Weighted RRF: scores[d] = Σ_l weight_l / (k + rank_l + 1)."""
    scores = {}
    for ranked, w in ranked_lists_with_weights:
        for i, d in enumerate(ranked):
            scores[d] = scores.get(d, 0.0) + w / (k + i + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def merge_with_tail(primary_ranking, tail_score_dict):
    seen = set(primary_ranking)
    tail = sorted(
        ((did, s) for did, s in tail_score_dict.items() if did not in seen),
        key=lambda x: x[1],
        reverse=True,
    )
    return primary_ranking + [d for d, _ in tail]


async def rerank_topk(reranker, query_text, doc_ids, doc_text, k):
    cand_ids = doc_ids[:k]
    cand_texts = [doc_text.get(did, "")[:1000] for did in cand_ids]
    scores = await reranker.score(query_text, cand_texts)
    return {did: float(s) for did, s in zip(cand_ids, scores)}


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

    name = "hard_bench"
    docs = [json.loads(l) for l in open(DATA_DIR / "hard_bench_docs.jsonl")]
    queries = [json.loads(l) for l in open(DATA_DIR / "hard_bench_queries.jsonl")]
    gold_rows = [json.loads(l) for l in open(DATA_DIR / "hard_bench_gold.jsonl")]
    gold = {g["query_id"]: g["relevant_doc_ids"] for g in gold_rows}

    print(f"=== {name}: {len(docs)} docs, {len(queries)} queries ===")

    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs]
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries]
    doc_ext = await run_v2_extract(doc_items, f"{name}-docs", "v7l-hard_bench")
    q_ext = await run_v2_extract(q_items, f"{name}-queries", "v7l-hard_bench")

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

    per_q_t = {
        qid: make_t_scores(
            q_mem.get(qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()}),
            doc_mem,
            per_q_l.get(qid, {}),
        )
        for qid in qids
    }

    print(f"reranking {RERANK_TOP_K} candidates per query...")
    per_q_r = {}
    for qid in qids:
        s_top = topk_from_scores(per_q_s[qid], RERANK_TOP_K)
        t_top = topk_from_scores(per_q_t[qid], RERANK_TOP_K)
        union = list(dict.fromkeys(s_top + t_top))[:RERANK_TOP_K]
        per_q_r[qid] = await rerank_topk(
            reranker, q_text[qid], union, doc_text, len(union)
        )

    # Build T/R/S rankings (per query)
    per_q_t_rank = {qid: topk_from_scores(per_q_t[qid], RERANK_TOP_K) for qid in qids}
    per_q_r_rank = {qid: topk_from_scores(per_q_r[qid], RERANK_TOP_K) for qid in qids}
    per_q_s_rank = {qid: topk_from_scores(per_q_s[qid], RERANK_TOP_K) for qid in qids}

    print("\n=== score_blend fusion at various (w_T, w_R) ===")
    print(f"{'w_T':>5} {'w_R':>5} {'R@1':>6} {'R@5':>6} {'MRR':>6}")
    sb_results = {}
    for w_t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        w_r = round(1.0 - w_t, 1)
        ranks = {}
        for qid in qids:
            fused = score_blend(
                {"T": per_q_t[qid], "R": per_q_r[qid]},
                {"T": w_t, "R": w_r},
                top_k_per=40,
                dispersion_cv_ref=0.20,
            )
            primary = [d for d, _ in fused]
            ranks[qid] = merge_with_tail(primary, per_q_s[qid])
        m = metrics(ranks, gold, qids)
        sb_results[(w_t, w_r)] = m
        print(
            f"{w_t:>5.1f} {w_r:>5.1f} {m['r@1']:>6.3f} {m['r@5']:>6.3f} {m['mrr']:>6.3f}"
        )

    print("\n=== uniform RRF at various k ===")
    print(f"{'k':>5} {'R@1':>6} {'R@5':>6} {'MRR':>6}")
    rrf_uniform_results = {}
    for k in [10, 30, 60, 100, 200]:
        ranks = {}
        for qid in qids:
            fused = weighted_rrf(
                [(per_q_t_rank[qid], 1.0), (per_q_r_rank[qid], 1.0)],
                k=k,
            )
            primary = [d for d, _ in fused]
            ranks[qid] = merge_with_tail(primary, per_q_s[qid])
        m = metrics(ranks, gold, qids)
        rrf_uniform_results[k] = m
        print(f"{k:>5} {m['r@1']:>6.3f} {m['r@5']:>6.3f} {m['mrr']:>6.3f}")

    print("\n=== weighted RRF at k=60, various (w_T, w_R) ===")
    print(f"{'w_T':>5} {'w_R':>5} {'R@1':>6} {'R@5':>6} {'MRR':>6}")
    rrf_weighted_results = {}
    for w_t in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        w_r = round(1.0 - w_t, 1)
        ranks = {}
        for qid in qids:
            fused = weighted_rrf(
                [(per_q_t_rank[qid], w_t), (per_q_r_rank[qid], w_r)],
                k=60,
            )
            primary = [d for d, _ in fused]
            ranks[qid] = merge_with_tail(primary, per_q_s[qid])
        m = metrics(ranks, gold, qids)
        rrf_weighted_results[(w_t, w_r)] = m
        print(
            f"{w_t:>5.1f} {w_r:>5.1f} {m['r@1']:>6.3f} {m['r@5']:>6.3f} {m['mrr']:>6.3f}"
        )

    print("\n=== TOP-PERFORMING configs ===")
    sb_best = max(
        sb_results.items(), key=lambda kv: (kv[1]["r@5"], kv[1]["r@1"], kv[1]["mrr"])
    )
    rrf_w_best = max(
        rrf_weighted_results.items(),
        key=lambda kv: (kv[1]["r@5"], kv[1]["r@1"], kv[1]["mrr"]),
    )
    rrf_u_best = max(
        rrf_uniform_results.items(),
        key=lambda kv: (kv[1]["r@5"], kv[1]["r@1"], kv[1]["mrr"]),
    )
    print(
        f"score_blend best: w=({sb_best[0][0]}, {sb_best[0][1]}) R@1={sb_best[1]['r@1']:.3f} R@5={sb_best[1]['r@5']:.3f}"
    )
    print(
        f"weighted RRF best: w=({rrf_w_best[0][0]}, {rrf_w_best[0][1]}) k=60 R@1={rrf_w_best[1]['r@1']:.3f} R@5={rrf_w_best[1]['r@5']:.3f}"
    )
    print(
        f"uniform RRF best: k={rrf_u_best[0]} R@1={rrf_u_best[1]['r@1']:.3f} R@5={rrf_u_best[1]['r@5']:.3f}"
    )

    out_path = ROOT / "results" / "t_rerank_fusion_strategies.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "score_blend": {f"{k[0]}_{k[1]}": v for k, v in sb_results.items()},
                "rrf_uniform": rrf_uniform_results,
                "rrf_weighted": {
                    f"{k[0]}_{k[1]}": v for k, v in rrf_weighted_results.items()
                },
            },
            f,
            indent=2,
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
