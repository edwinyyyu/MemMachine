"""LongMemEval-S non-temporal subset: does fuse_T_R hold up when temporal
signal is weak / absent?

Takes first 10 non-temporal questions from longmemeval_s_50q.json. Each
question's haystack (~48 sessions) is its own retrieval task: docs = sessions,
query = question, gold = answer_session_ids.

Compares: pure_S, T_lblend, rerank_only, fuse_T_R.
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
    AxisDistribution,
    build_memory,
    embed_all,
    interval_pair_best,
    parse_iso,
    rank_semantic,
    run_v2_extract,
    tag_score,
)

T_ALPHA, T_GAMMA, T_DELTA = 0.20, 0.20, 0.60
RERANK_TOP_K = 50
N_QUESTIONS = 10
NON_TEMPORAL_TYPES = {
    "single-session-preference",
    "single-session-user",
    "single-session-assistant",
    "knowledge-update",
    "multi-session",
}


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


def topk_from_scores(scores, k):
    return [d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]]


def merge_with_tail(primary, tail_scores):
    seen = set(primary)
    tail = sorted(
        ((d, s) for d, s in tail_scores.items() if d not in seen),
        key=lambda x: x[1],
        reverse=True,
    )
    return primary + [d for d, _ in tail]


async def rerank_topk(reranker, query_text, doc_ids, doc_text, k):
    cand_ids = doc_ids[:k]
    cand_texts = [doc_text.get(did, "")[:1000] for did in cand_ids]
    scores = await reranker.score(query_text, cand_texts)
    return {did: float(s) for did, s in zip(cand_ids, scores)}


def session_to_text(session):
    """Concat turns into a single string."""
    return " ".join(t.get("content", "") for t in session)


def hit_rank(ranking, gold_set, k=10):
    for i, d in enumerate(ranking[:k]):
        if d in gold_set:
            return i + 1
    return None


async def evaluate_question(
    qid, q_text, q_date, sessions_dict, session_dates, gold_ids, reranker
):
    """Evaluate all variants for a single question."""
    doc_ids = list(sessions_dict.keys())
    doc_text = sessions_dict
    gold_set = set(gold_ids)

    # Per-doc extraction
    doc_items = [
        (did, doc_text[did], parse_iso(session_dates.get(did, q_date)))
        for did in doc_ids
    ]
    q_items = [(qid, q_text, parse_iso(q_date))]

    cache_label = f"lme-q-{qid}"
    doc_ext = await run_v2_extract(doc_items, cache_label + "-docs", cache_label)
    q_ext = await run_v2_extract(q_items, cache_label + "-queries", cache_label)

    doc_mem = build_memory(doc_ext)
    q_mem = build_memory(q_ext)
    for did in doc_ids:
        doc_mem.setdefault(
            did,
            {
                "intervals": [],
                "axes_merged": {
                    a: AxisDistribution(axis=a, values={}, informative=False)
                    for a in AXES
                },
                "multi_tags": set(),
            },
        )

    # Embeddings
    doc_texts = [doc_text[did] for did in doc_ids]
    q_texts = [q_text]
    doc_embs_arr = await embed_all(doc_texts)
    q_embs_arr = await embed_all(q_texts)
    doc_embs = {did: doc_embs_arr[i] for i, did in enumerate(doc_ids)}
    q_embs = {qid: q_embs_arr[0]}

    s_scores = rank_semantic(qid, q_embs, doc_embs)

    # Lattice
    lat_db = ROOT / "cache" / "lme_subset" / f"lat_{qid}.sqlite"
    lat_db.parent.mkdir(parents=True, exist_ok=True)
    if lat_db.exists():
        lat_db.unlink()
    lat = LatticeStore(str(lat_db))
    for did, tes in doc_ext.items():
        for te in tes:
            ts = lattice_tags_for_expression(te)
            lat.insert(did, ts.absolute, ts.cyclical)
    l_scores, _ = lattice_retrieve_multi(lat, q_ext.get(qid, []), down_levels=1)

    t_scores = make_t_scores(
        q_mem.get(qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()}),
        doc_mem,
        l_scores,
    )

    # Rerank top-50 from union of S and T candidates
    s_top = topk_from_scores(s_scores, RERANK_TOP_K)
    t_top = topk_from_scores(t_scores, RERANK_TOP_K)
    union = list(dict.fromkeys(s_top + t_top))[:RERANK_TOP_K]
    r_scores = await rerank_topk(reranker, q_text, union, doc_text, len(union))

    # Variants
    variants = {}

    # 1. pure_S
    variants["pure_S"] = topk_from_scores(s_scores, len(doc_ids))

    # 2. T_lblend at w_T=0.4
    fused = score_blend(
        {"T": t_scores, "S": s_scores},
        {"T": 0.4, "S": 0.6},
        top_k_per=40,
        dispersion_cv_ref=0.20,
    )
    primary = [d for d, _ in fused]
    variants["T_lblend"] = merge_with_tail(primary, s_scores)

    # 3. rerank_only on top-50 of pure_S
    s_top_50 = topk_from_scores(s_scores, RERANK_TOP_K)
    rs_for_s = {did: r_scores.get(did, 0.0) for did in s_top_50}
    primary = [d for d, _ in sorted(rs_for_s.items(), key=lambda x: x[1], reverse=True)]
    variants["rerank_only"] = merge_with_tail(primary, s_scores)

    # 4. fuse_T_R at w_T=0.4
    fused = score_blend(
        {"T": t_scores, "R": r_scores},
        {"T": 0.4, "R": 0.6},
        top_k_per=40,
        dispersion_cv_ref=0.20,
    )
    primary = [d for d, _ in fused]
    variants["fuse_T_R"] = merge_with_tail(primary, s_scores)

    # Per-variant: gold rank
    return {var: hit_rank(rk, gold_set) for var, rk in variants.items()}


async def main():
    print("Loading cross-encoder...")
    from memmachine_server.common.reranker.cross_encoder_reranker import (
        CrossEncoderReranker,
        CrossEncoderRerankerParams,
    )
    from sentence_transformers import CrossEncoder

    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    reranker = CrossEncoderReranker(
        CrossEncoderRerankerParams(
            cross_encoder=ce,
            max_input_length=512,
        )
    )

    data = json.load(
        open(ROOT.parent / "associative_recall" / "data" / "longmemeval_s_50q.json")
    )
    non_temp = [q for q in data if q["question_type"] in NON_TEMPORAL_TYPES][
        :N_QUESTIONS
    ]

    print(f"Selected {len(non_temp)} non-temporal questions:")
    for q in non_temp:
        print(f"  [{q['question_type']}] {q['question'][:80]}")

    all_results = []  # list of {variant: hit_rank or None}
    for i, q in enumerate(non_temp):
        qid = q["question_id"]
        q_text = q["question"]
        q_date = (
            q["question_date"].split(" ")[0].replace("/", "-")
        )  # 2023/05/30 → 2023-05-30
        gold_ids = q["answer_session_ids"]

        # Build sessions dict
        sessions_dict = {
            sid: session_to_text(sess)
            for sid, sess in zip(q["haystack_session_ids"], q["haystack_sessions"])
        }
        session_dates = {
            sid: d.split(" ")[0].replace("/", "-")
            for sid, d in zip(q["haystack_session_ids"], q["haystack_dates"])
        }

        print(f"\n[{i + 1}/{len(non_temp)}] {q['question_type']}: {q_text[:80]}")
        print(f"  haystack: {len(sessions_dict)} sessions, gold: {gold_ids}")
        try:
            result = await evaluate_question(
                qid,
                q_text,
                q_date,
                sessions_dict,
                session_dates,
                gold_ids,
                reranker,
            )
            all_results.append(
                {"qid": qid, "type": q["question_type"], "result": result}
            )
            print(f"  hit ranks: {result}")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()

    # Aggregate
    print(f"\n=== AGGREGATE on {len(all_results)} questions ===")
    variants = ["pure_S", "T_lblend", "rerank_only", "fuse_T_R"]
    for var in variants:
        ranks = [e["result"][var] for e in all_results]
        r1 = sum(1 for r in ranks if r is not None and r <= 1)
        r5 = sum(1 for r in ranks if r is not None and r <= 5)
        r10 = sum(1 for r in ranks if r is not None and r <= 10)
        mrr = sum(1.0 / r for r in ranks if r is not None) / len(ranks)
        print(
            f"  {var:14} R@1={r1}/{len(ranks)} ({r1 / len(ranks):.3f})  "
            f"R@5={r5}/{len(ranks)} ({r5 / len(ranks):.3f})  "
            f"R@10={r10}/{len(ranks)} ({r10 / len(ranks):.3f})  MRR={mrr:.3f}"
        )

    out_path = ROOT / "results" / "longmemeval_subset.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
