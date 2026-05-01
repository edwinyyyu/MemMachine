"""Per-query gating: model picks between fuse_T_R and rerank_only result sets.

The structural problem with fuse_T_R: it adds T (good on date-anchored corpora,
bad on non-temporal). Score_blend's CV gate doesn't distinguish noisy variance
from informative variance.

This test: per query, show the model {rerank_only top-5, fuse_T_R top-5} blinded
and shuffled. Model picks better set. Use that as the final ranking.

If model picks correctly:
  - hard_bench: should pick fuse_T_R (recovers 0.893 R@1)
  - longmemeval non-temp: should pick rerank_only (recovers 0.700 R@1)

If model picks well, gated approach beats fixed-recipe. If not, no LLM-in-loop
approach can solve the distribution-drift problem.
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
from v7l_ts_blind_eval import BlindJudge

T_ALPHA, T_GAMMA, T_DELTA = 0.20, 0.20, 0.60
RERANK_TOP_K = 50
W_T_FUSE = 0.4


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


def hit_rank(ranking, gold_set, k=10):
    for i, d in enumerate(ranking[:k]):
        if d in gold_set:
            return i + 1
    return None


async def evaluate_query(
    qid, q_text, doc_text, doc_ids, gold_set, t_scores, s_scores, r_scores, judge
):
    """Compute rerank_only and fuse_T_R rankings, then have model pick between them."""
    # rerank_only
    s_top_50 = topk_from_scores(s_scores, RERANK_TOP_K)
    rs = {did: r_scores.get(did, 0.0) for did in s_top_50}
    rerank_only_primary = [
        d for d, _ in sorted(rs.items(), key=lambda x: x[1], reverse=True)
    ]
    rerank_only_full = merge_with_tail(rerank_only_primary, s_scores)

    # fuse_T_R
    fused = score_blend(
        {"T": t_scores, "R": r_scores},
        {"T": W_T_FUSE, "R": 1.0 - W_T_FUSE},
        top_k_per=40,
        dispersion_cv_ref=0.20,
    )
    fuse_primary = [d for d, _ in fused]
    fuse_T_R_full = merge_with_tail(fuse_primary, s_scores)

    # Show top-5 of each to model
    rerank_top5 = [
        doc_text.get(d, "")[: base.MAX_TEXT_LEN] for d in rerank_only_full[:5]
    ]
    fuse_top5 = [doc_text.get(d, "")[: base.MAX_TEXT_LEN] for d in fuse_T_R_full[:5]]

    seed = hash((qid, "gate")) & 0xFFFFFFFF
    pick_idx = await judge.pick_best(q_text, [rerank_top5, fuse_top5], rng_seed=seed)

    if pick_idx == 0:
        gated = rerank_only_full
        decision = "rerank_only"
    elif pick_idx == 1:
        gated = fuse_T_R_full
        decision = "fuse_T_R"
    else:
        # Tie → default to rerank_only (the more robust choice)
        gated = rerank_only_full
        decision = "tie→rerank_only"

    return {
        "rerank_only": hit_rank(rerank_only_full, gold_set),
        "fuse_T_R": hit_rank(fuse_T_R_full, gold_set),
        "gated": hit_rank(gated, gold_set),
        "decision": decision,
    }


async def run_temporal_bench(
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

    lat_db = ROOT / "cache" / "gate" / f"lat_{name}.sqlite"
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

    print("  reranking + gating...")
    per_q_r = {}
    for qid in qids:
        s_top = topk_from_scores(per_q_s[qid], RERANK_TOP_K)
        t_top = topk_from_scores(per_q_t[qid], RERANK_TOP_K)
        union = list(dict.fromkeys(s_top + t_top))[:RERANK_TOP_K]
        per_q_r[qid] = await rerank_topk(
            reranker, q_text[qid], union, doc_text, len(union)
        )

    results = []
    for q in queries:
        qid = q["query_id"]
        gold_set = set(gold.get(qid, []))
        if not gold_set:
            continue
        r = await evaluate_query(
            qid,
            q_text[qid],
            doc_text,
            list(doc_text.keys()),
            gold_set,
            per_q_t[qid],
            per_q_s[qid],
            per_q_r[qid],
            judge,
        )
        results.append(r)

    return aggregate(results, name)


async def run_lme_bench(judge, reranker):
    """LongMemEval non-temporal subset."""
    NON_TEMPORAL_TYPES = {
        "single-session-preference",
        "single-session-user",
        "single-session-assistant",
        "knowledge-update",
        "multi-session",
    }
    data = json.load(
        open(ROOT.parent / "associative_recall" / "data" / "longmemeval_s_50q.json")
    )
    non_temp = [q for q in data if q["question_type"] in NON_TEMPORAL_TYPES][:10]

    print(f"\n=== longmemeval (non-temp): {len(non_temp)} queries ===")

    results = []
    for q in non_temp:
        qid = q["question_id"]
        q_text = q["question"]
        q_date = q["question_date"].split(" ")[0].replace("/", "-")
        gold_ids = q["answer_session_ids"]
        gold_set = set(gold_ids)

        sessions_dict = {
            sid: " ".join(t.get("content", "") for t in sess)
            for sid, sess in zip(q["haystack_session_ids"], q["haystack_sessions"])
        }
        session_dates = {
            sid: d.split(" ")[0].replace("/", "-")
            for sid, d in zip(q["haystack_session_ids"], q["haystack_dates"])
        }
        doc_ids = list(sessions_dict.keys())
        doc_text = sessions_dict

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

        doc_embs_arr = await embed_all([doc_text[did] for did in doc_ids])
        q_embs_arr = await embed_all([q_text])
        doc_embs = {did: doc_embs_arr[i] for i, did in enumerate(doc_ids)}
        q_embs = {qid: q_embs_arr[0]}
        s_scores = rank_semantic(qid, q_embs, doc_embs)

        lat_db = ROOT / "cache" / "gate_lme" / f"lat_{qid}.sqlite"
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

        s_top = topk_from_scores(s_scores, RERANK_TOP_K)
        t_top = topk_from_scores(t_scores, RERANK_TOP_K)
        union = list(dict.fromkeys(s_top + t_top))[:RERANK_TOP_K]
        r_scores = await rerank_topk(reranker, q_text, union, doc_text, len(union))

        r = await evaluate_query(
            qid,
            q_text,
            doc_text,
            doc_ids,
            gold_set,
            t_scores,
            s_scores,
            r_scores,
            judge,
        )
        results.append(r)

    return aggregate(results, "longmemeval (non-temp)")


def aggregate(results, label):
    print(f"\n=== {label} ===")
    variants = ["rerank_only", "fuse_T_R", "gated"]
    n = len(results)
    for var in variants:
        ranks = [r[var] for r in results]
        r1 = sum(1 for x in ranks if x is not None and x <= 1)
        r5 = sum(1 for x in ranks if x is not None and x <= 5)
        mrr = sum(1.0 / x for x in ranks if x is not None) / n if n else 0
        print(
            f"  {var:14} R@1={r1}/{n} ({r1 / n:.3f})  R@5={r5}/{n} ({r5 / n:.3f})  MRR={mrr:.3f}"
        )
    decisions = {}
    for r in results:
        decisions[r["decision"]] = decisions.get(r["decision"], 0) + 1
    print(f"  decisions: {decisions}")
    return {"results": results, "decisions": decisions}


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
    judge = BlindJudge()

    out = {}
    out["hard_bench"] = await run_temporal_bench(
        "hard_bench",
        "hard_bench_docs.jsonl",
        "hard_bench_queries.jsonl",
        "hard_bench_gold.jsonl",
        "v7l-hard_bench",
        "v7l-hard_bench",
        reranker,
        judge,
    )
    out["longmemeval_nontemp"] = await run_lme_bench(judge, reranker)

    judge.save()
    out_path = ROOT / "results" / "per_query_gate.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"LLM calls: {judge.calls}")


if __name__ == "__main__":
    asyncio.run(main())
