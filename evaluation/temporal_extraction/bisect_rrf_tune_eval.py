"""Bisect_thirds with RRF for the tuning loop, score_blend for serving.

Hypothesis: under the chrono+set+1-token format only top-5 SET MEMBERSHIP
matters to the judge. score_blend has step-shaped thresholds where adjacent
w_T values produce identical top-5 sets (the bisection's no-signal trap).
Weighted RRF interpolates rank reciprocals smoothly → more frequent top-5
set transitions per Δw_T → bisection always has signal.

Final serving keeps score_blend so the cross-encoder's continuous relevance
gradient is preserved (RRF squashes it; -0.20 R@1 if RRF used end-to-end).

Variants tested:
  bisect_score    bisect_thirds with score_blend tune+serve (baseline; matches
                  the prior set_pickers_eval result)
  bisect_rrf_tune bisect_thirds with weighted RRF tune, score_blend serve

Both compared against:
  rerank_only, fuse_T_R w=0.6 (oracle), gate (new format)

Tested on hard_bench, mixed_cue, dense_cluster, tempreason, LME-nontemp.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT.parents[1] / ".env")

from force_pick_optimizers_eval import (
    RERANK_TOP_K,
    fuse_at_w,
    make_t_scores,
    merge_with_tail,
    rerank_topk,
    topk_from_scores,
)
from lattice_cells import tags_for_expression as lattice_tags_for_expression
from lattice_retrieval import retrieve_multi as lattice_retrieve_multi
from lattice_store import LatticeStore
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
from set_pickers_eval import hit_rank, pick_n, run_bisect_thirds, run_gate
from v7l_ts_blind_eval import BlindJudge

RRF_K = 60


def weighted_rrf(ranked_lists_with_weights, k=RRF_K):
    """Weighted RRF: each ranked list contributes (weight / (k + rank+1))."""
    scores = {}
    for rl, w in ranked_lists_with_weights:
        if w <= 0:
            continue
        for i, d in enumerate(rl):
            scores[d] = scores.get(d, 0.0) + w * 1.0 / (k + i + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def fuse_at_w_rrf(t_scores, r_scores, s_scores, w_T, top_k_per=40):
    """Weighted RRF fusion of T and R rankings. Used ONLY for tuning loop's
    candidate-set generation. Returns ranked doc list (no scores), tail
    appended from S for completeness."""
    t_top = [
        d
        for d, _ in sorted(t_scores.items(), key=lambda x: x[1], reverse=True)[
            :top_k_per
        ]
    ]
    r_top = [
        d
        for d, _ in sorted(r_scores.items(), key=lambda x: x[1], reverse=True)[
            :top_k_per
        ]
    ]
    fused = weighted_rrf([(t_top, w_T), (r_top, 1.0 - w_T)])
    primary = [d for d, _ in fused]
    seen = set(primary)
    tail = sorted(s_scores.items(), key=lambda x: x[1], reverse=True)
    return primary + [d for d, _ in tail if d not in seen]


def get_set_at_w_rrf(t_scores, r_scores, s_scores, w_T, doc_text, doc_dates):
    ranked = fuse_at_w_rrf(t_scores, r_scores, s_scores, w_T)
    return [(d, doc_dates.get(d, ""), doc_text.get(d, "")) for d in ranked[:5]]


# ---------- bisect with RRF tune, score_blend serve ----------


async def run_bisect_rrf_tune(
    qid,
    query_text,
    t_scores,
    r_scores,
    s_scores,
    doc_text,
    doc_dates,
    judge,
    lo_init=0.0,
    hi_init=0.7,
    max_rounds=4,
):
    lo, hi = lo_init, hi_init
    last_picked_w = (lo + hi) / 2
    history = []
    for r in range(max_rounds):
        L = hi - lo
        c_left = lo + L / 3.0
        c_right = lo + 2.0 * L / 3.0
        # RRF for tuning candidate-set generation
        s_left = get_set_at_w_rrf(
            t_scores, r_scores, s_scores, c_left, doc_text, doc_dates
        )
        s_right = get_set_at_w_rrf(
            t_scores, r_scores, s_scores, c_right, doc_text, doc_dates
        )
        ids_left = tuple(d for d, _, _ in s_left)
        ids_right = tuple(d for d, _, _ in s_right)
        if ids_left == ids_right:
            history.append({"round": r + 1, "skipped": "identical"})
            break
        seed = hash((qid, "bisect_rrf_tune", r, lo, hi)) & 0xFFFFFFFF
        idx = await pick_n(judge, query_text, [s_left, s_right], seed)
        if idx == 0:
            last_picked_w = c_left
            hi = c_right
            history.append(
                {
                    "round": r + 1,
                    "winner": "left",
                    "lo": lo,
                    "hi": hi,
                    "picked": last_picked_w,
                }
            )
        else:
            last_picked_w = c_right
            lo = c_left
            history.append(
                {
                    "round": r + 1,
                    "winner": "right",
                    "lo": lo,
                    "hi": hi,
                    "picked": last_picked_w,
                }
            )
    # Serve: score_blend at picked w_T
    return fuse_at_w(t_scores, r_scores, s_scores, last_picked_w), {
        "final_w_T": last_picked_w,
        "history": history,
    }


# ---------- per-query evaluation ----------


async def evaluate_query(
    qid, q_text, doc_text, doc_dates, gold_set, t_scores, s_scores, r_scores, judge
):
    s_top_50 = topk_from_scores(s_scores, RERANK_TOP_K)
    rs = {did: r_scores.get(did, 0.0) for did in s_top_50}
    rerank_only = merge_with_tail(
        [d for d, _ in sorted(rs.items(), key=lambda x: x[1], reverse=True)], s_scores
    )
    fuse_06 = fuse_at_w(t_scores, r_scores, s_scores, 0.6)

    bisect_score_rank, bsd = await run_bisect_thirds(
        qid, q_text, t_scores, r_scores, s_scores, doc_text, doc_dates, judge
    )
    bisect_rrf_rank, brd = await run_bisect_rrf_tune(
        qid, q_text, t_scores, r_scores, s_scores, doc_text, doc_dates, judge
    )
    gate_rank = await run_gate(
        qid, q_text, t_scores, r_scores, s_scores, doc_text, doc_dates, judge
    )

    return {
        "rerank_only": hit_rank(rerank_only, gold_set),
        "fuse_T_R_w06": hit_rank(fuse_06, gold_set),
        "bisect_score": hit_rank(bisect_score_rank, gold_set),
        "bisect_rrf_tune": hit_rank(bisect_rrf_rank, gold_set),
        "gate_new": hit_rank(gate_rank, gold_set),
        "bisect_score_w": bsd["final_w_T"],
        "bisect_rrf_w": brd["final_w_T"],
    }


# ---------- benchmark loaders ----------


async def run_temporal_bench(name, docs_path, queries_path, gold_path, reranker, judge):
    docs = [json.loads(l) for l in open(DATA_DIR / docs_path)]
    queries = [json.loads(l) for l in open(DATA_DIR / queries_path)]
    gold_rows = [json.loads(l) for l in open(DATA_DIR / gold_path)]
    gold = {g["query_id"]: g["relevant_doc_ids"] for g in gold_rows}

    print(f"\n=== {name}: {len(docs)} docs, {len(queries)} queries ===")
    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs]
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries]
    doc_ext = await run_v2_extract(doc_items, f"{name}-docs", f"v7l-{name}")
    q_ext = await run_v2_extract(q_items, f"{name}-queries", f"v7l-{name}")
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
    doc_dates = {d["doc_id"]: d["ref_time"][:10] for d in docs}
    q_text = {q["query_id"]: q["text"] for q in queries}
    doc_embs_arr = await embed_all([d["text"] for d in docs])
    q_embs_arr = await embed_all([q["text"] for q in queries])
    doc_embs = {d["doc_id"]: doc_embs_arr[i] for i, d in enumerate(docs)}
    q_embs = {q["query_id"]: q_embs_arr[i] for i, q in enumerate(queries)}
    qids = [q["query_id"] for q in queries]
    per_q_s = {qid: rank_semantic(qid, q_embs, doc_embs) for qid in qids}

    lat_db = ROOT / "cache" / "force" / f"lat_{name}.sqlite"
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

    print("  reranking + bisect rrf-tune...")
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
            doc_dates,
            gold_set,
            per_q_t[qid],
            per_q_s[qid],
            per_q_r[qid],
            judge,
        )
        results.append(r)
    return aggregate(results, name)


async def run_lme_bench(judge, reranker, label, types):
    data = json.load(
        open(ROOT.parent / "associative_recall" / "data" / "longmemeval_s_50q.json")
    )
    queries = [q for q in data if q["question_type"] in types]
    print(f"\n=== longmemeval ({label}): {len(queries)} queries ===")
    results = []
    skipped = 0
    for q in queries:
        qid = q["question_id"]
        q_text = q["question"]
        q_date = q["question_date"].split(" ")[0].replace("/", "-")
        gold_ids = q["answer_session_ids"]
        gold_set = set(gold_ids)
        # Truncate to 24000 chars to stay under embedding model's 8192-token limit
        sessions_dict = {
            sid: " ".join(t.get("content", "") for t in sess)[:24000]
            for sid, sess in zip(q["haystack_session_ids"], q["haystack_sessions"])
        }
        session_dates = {
            sid: d.split(" ")[0].replace("/", "-")
            for sid, d in zip(q["haystack_session_ids"], q["haystack_dates"])
        }
        doc_ids = list(sessions_dict.keys())
        doc_text = sessions_dict
        doc_dates = dict(session_dates)

        doc_items = [
            (did, doc_text[did], parse_iso(session_dates.get(did, q_date)))
            for did in doc_ids
        ]
        q_items = [(qid, q_text, parse_iso(q_date))]
        cache_label = f"lme-q-{qid}"
        try:
            doc_ext = await run_v2_extract(
                doc_items, cache_label + "-docs", cache_label
            )
            q_ext = await run_v2_extract(q_items, cache_label + "-queries", cache_label)
            doc_mem = build_memory(doc_ext)
            q_mem = build_memory(q_ext)
        except (OverflowError, ValueError) as e:
            skipped += 1
            print(f"  skipping {qid}: {type(e).__name__}: {str(e)[:80]}")
            continue
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

        lat_db = ROOT / "cache" / "force_lme" / f"lat_{qid}.sqlite"
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
            doc_dates,
            gold_set,
            t_scores,
            s_scores,
            r_scores,
            judge,
        )
        results.append(r)
    if skipped:
        print(f"  ({skipped} queries skipped due to extraction errors)")
    return aggregate(results, f"longmemeval ({label})")


def aggregate(results, label):
    print(f"\n=== {label} ===")
    variants = [
        "rerank_only",
        "fuse_T_R_w06",
        "bisect_score",
        "bisect_rrf_tune",
        "gate_new",
    ]
    n = len(results)
    for var in variants:
        ranks = [r[var] for r in results]
        r1 = sum(1 for x in ranks if x is not None and x <= 1)
        r5 = sum(1 for x in ranks if x is not None and x <= 5)
        mrr = sum(1.0 / x for x in ranks if x is not None) / n if n else 0
        print(
            f"  {var:18} R@1={r1}/{n} ({r1 / n:.3f})  R@5={r5}/{n} ({r5 / n:.3f})  MRR={mrr:.3f}"
        )
    import statistics

    for key in ["bisect_score_w", "bisect_rrf_w"]:
        ws = [r[key] for r in results]
        print(
            f"  {key:18}: mean={statistics.mean(ws):.3f}  unique={sorted(set(round(x, 3) for x in ws))}"
        )
    return {"results": results}


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

    benches = [
        (
            "goldilocks",
            "goldilocks_docs.jsonl",
            "goldilocks_queries.jsonl",
            "goldilocks_gold.jsonl",
        ),
        (
            "hard_bench",
            "hard_bench_docs.jsonl",
            "hard_bench_queries.jsonl",
            "hard_bench_gold.jsonl",
        ),
        (
            "temporal_essential",
            "temporal_essential_docs.jsonl",
            "temporal_essential_queries.jsonl",
            "temporal_essential_gold.jsonl",
        ),
        (
            "tempreason_small",
            "real_benchmark_small_docs.jsonl",
            "real_benchmark_small_queries.jsonl",
            "real_benchmark_small_gold.jsonl",
        ),
    ]
    out = {}
    for name, dp, qp, gp in benches:
        try:
            out[name] = await run_temporal_bench(name, dp, qp, gp, reranker, judge)
        except Exception as e:
            print(f"  [{name}] failed: {e}")
            out[name] = {"error": str(e)}
    NON_TEMP_TYPES = {
        "single-session-preference",
        "single-session-user",
        "single-session-assistant",
        "knowledge-update",
        "multi-session",
    }
    TEMP_TYPES = {"temporal-reasoning"}
    out["lme_nontemp_full"] = await run_lme_bench(
        judge, reranker, "non-temp", NON_TEMP_TYPES
    )
    out["lme_temp"] = await run_lme_bench(judge, reranker, "temp", TEMP_TYPES)

    judge.save()
    out_path = ROOT / "results" / "bisect_rrf_tune.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"LLM calls: {judge.calls}")


if __name__ == "__main__":
    asyncio.run(main())
