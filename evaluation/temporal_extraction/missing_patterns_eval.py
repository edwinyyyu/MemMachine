"""Eval T_lblend vs rerank_only on the 4 missing-pattern benchmarks.

Produces R@1 per benchmark per approach, plus a per-query diagnostic showing
which T component (interval / tag / lattice) scored gold above/below each
distractor, to surface which capability is missing.

Usage:
  /Users/eyu/edwinyyyu/mmcc/extra_memory/.venv/bin/python missing_patterns_eval.py
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
    interval_pair_best,
    parse_iso,
    rank_semantic,
    run_v2_extract,
    tag_score,
)

BENCHES = [
    (
        "causal_relative",
        "causal_relative_docs.jsonl",
        "causal_relative_queries.jsonl",
        "causal_relative_gold.jsonl",
    ),
    (
        "latest_recent",
        "latest_recent_docs.jsonl",
        "latest_recent_queries.jsonl",
        "latest_recent_gold.jsonl",
    ),
    (
        "open_ended_date",
        "open_ended_date_docs.jsonl",
        "open_ended_date_queries.jsonl",
        "open_ended_date_gold.jsonl",
    ),
    (
        "negation_temporal",
        "negation_temporal_docs.jsonl",
        "negation_temporal_queries.jsonl",
        "negation_temporal_gold.jsonl",
    ),
]


def hit_rank_set(ranking, gold_set, k=10):
    for i, d in enumerate(ranking[:k]):
        if d in gold_set:
            return i + 1
    return None


def t_components(q_mem_q, doc_mem_d, l_per_doc_d):
    """Decompose T into its three components for one (query, doc) pair."""
    q_ivs = q_mem_q.get("intervals") or []
    q_tags = q_mem_q.get("multi_tags") or set()
    raw_iv = interval_pair_best(q_ivs, doc_mem_d["intervals"])
    return {
        "iv_raw": raw_iv,
        "tag": tag_score(q_tags, doc_mem_d["multi_tags"]),
        "lat": l_per_doc_d,
    }


async def run_bench(name, docs_path, queries_path, gold_path, reranker):
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
    q_text = {q["query_id"]: q["text"] for q in queries}

    doc_embs_arr = await embed_all([d["text"] for d in docs])
    q_embs_arr = await embed_all([q["text"] for q in queries])
    doc_embs = {d["doc_id"]: doc_embs_arr[i] for i, d in enumerate(docs)}
    q_embs = {q["query_id"]: q_embs_arr[i] for i, q in enumerate(queries)}
    qids = [q["query_id"] for q in queries]
    per_q_s = {qid: rank_semantic(qid, q_embs, doc_embs) for qid in qids}

    lat_db = ROOT / "cache" / "missing" / f"lat_{name}.sqlite"
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

    print("  reranking...")
    per_q_r = {}
    for qid in qids:
        s_top = topk_from_scores(per_q_s[qid], RERANK_TOP_K)
        t_top = topk_from_scores(per_q_t[qid], RERANK_TOP_K)
        union = list(dict.fromkeys(s_top + t_top))[:RERANK_TOP_K]
        per_q_r[qid] = await rerank_topk(
            reranker, q_text[qid], union, doc_text, len(union)
        )

    per_q_results = []
    for q in queries:
        qid = q["query_id"]
        gold_set = set(gold.get(qid, []))
        if not gold_set:
            continue

        # rerank_only ranking: rerank top-50 semantic, then tail by semantic
        s_top_50 = topk_from_scores(per_q_s[qid], RERANK_TOP_K)
        rs = {did: per_q_r[qid].get(did, 0.0) for did in s_top_50}
        rerank_only = merge_with_tail(
            [d for d, _ in sorted(rs.items(), key=lambda x: x[1], reverse=True)],
            per_q_s[qid],
        )

        # T_lblend ranking: rank purely by T scores
        t_lblend = sorted(per_q_t[qid].items(), key=lambda x: x[1], reverse=True)
        t_lblend_ranked = [d for d, _ in t_lblend]
        # tail for T-zero ties: append docs not seen by semantic order
        t_lblend_full = merge_with_tail(t_lblend_ranked, per_q_s[qid])

        ro_rank = hit_rank_set(rerank_only, gold_set)
        tl_rank = hit_rank_set(t_lblend_full, gold_set)

        # Diagnostic: top-3 docs by T_lblend, and gold's T components
        gold_id = next(iter(gold_set))
        gold_t = per_q_t[qid].get(gold_id, 0.0)
        gold_components = t_components(
            q_mem.get(qid, {"intervals": [], "multi_tags": set()}),
            doc_mem.get(gold_id, {"intervals": [], "multi_tags": set()}),
            per_q_l.get(qid, {}).get(gold_id, 0.0),
        )

        # Show top 3 by T and their components
        top3_t = []
        for did, sc in t_lblend[:3]:
            comp = t_components(
                q_mem.get(qid, {"intervals": [], "multi_tags": set()}),
                doc_mem.get(did, {"intervals": [], "multi_tags": set()}),
                per_q_l.get(qid, {}).get(did, 0.0),
            )
            top3_t.append(
                {
                    "doc_id": did,
                    "is_gold": did in gold_set,
                    "t_total": sc,
                    "iv_raw": comp["iv_raw"],
                    "tag": comp["tag"],
                    "lat": comp["lat"],
                    "text_snip": doc_text[did][:90],
                }
            )

        per_q_results.append(
            {
                "qid": qid,
                "query": q_text[qid],
                "gold_id": gold_id,
                "gold_text": doc_text.get(gold_id, "")[:90],
                "rerank_only_rank": ro_rank,
                "t_lblend_rank": tl_rank,
                "gold_t": gold_t,
                "gold_components": gold_components,
                "top3_by_t": top3_t,
            }
        )

    n = len(per_q_results)
    ro_r1 = sum(
        1
        for r in per_q_results
        if r["rerank_only_rank"] is not None and r["rerank_only_rank"] <= 1
    )
    tl_r1 = sum(
        1
        for r in per_q_results
        if r["t_lblend_rank"] is not None and r["t_lblend_rank"] <= 1
    )
    ro_r5 = sum(
        1
        for r in per_q_results
        if r["rerank_only_rank"] is not None and r["rerank_only_rank"] <= 5
    )
    tl_r5 = sum(
        1
        for r in per_q_results
        if r["t_lblend_rank"] is not None and r["t_lblend_rank"] <= 5
    )

    print(
        f"  rerank_only  R@1={ro_r1}/{n} ({ro_r1 / n:.3f})  R@5={ro_r5}/{n} ({ro_r5 / n:.3f})"
    )
    print(
        f"  T_lblend     R@1={tl_r1}/{n} ({tl_r1 / n:.3f})  R@5={tl_r5}/{n} ({tl_r5 / n:.3f})"
    )
    return {
        "name": name,
        "n": n,
        "rerank_only_r1": ro_r1,
        "rerank_only_r5": ro_r5,
        "t_lblend_r1": tl_r1,
        "t_lblend_r5": tl_r5,
        "per_query": per_q_results,
    }


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

    out = {}
    for name, dp, qp, gp in BENCHES:
        try:
            out[name] = await run_bench(name, dp, qp, gp, reranker)
        except Exception as e:
            import traceback

            traceback.print_exc()
            out[name] = {"error": str(e)}

    out_path = ROOT / "results" / "missing_patterns_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {out_path}")

    print("\n=== Summary ===")
    print(f"{'Benchmark':<22} {'rerank_only R@1':<20} {'T_lblend R@1':<20}")
    for name, *_ in BENCHES:
        if "error" in out.get(name, {}):
            print(f"{name:<22} ERROR: {out[name]['error']}")
            continue
        r = out[name]
        print(
            f"{name:<22} {r['rerank_only_r1']}/{r['n']} ({r['rerank_only_r1'] / r['n']:.3f})  "
            f"     {r['t_lblend_r1']}/{r['n']} ({r['t_lblend_r1'] / r['n']:.3f})"
        )


if __name__ == "__main__":
    asyncio.run(main())
