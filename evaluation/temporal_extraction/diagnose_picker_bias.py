"""Diagnose WHY the model picks lower w_T when higher w_T produces correct rankings.

For hard_bench queries where:
  - fuse_T_R_w06 hits the gold at rank 1
  - bisect_thirds_capped misses (final w_T near 0.069)

Print the result sets at the two compared weights in round 1 (c_left=0.233, c_right=0.467
of [0, 0.7]) and inspect:
  - Is the gold present in either/both?
  - At what rank?
  - What does the model see (text snippets) that drives its pick?

Then ask: could a smarter judge pick correctly without seeing gold?
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
    get_top5_at_w,
    make_t_scores,
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


async def main():
    from memmachine_server.common.reranker.cross_encoder_reranker import (
        CrossEncoderReranker,
        CrossEncoderRerankerParams,
    )
    from sentence_transformers import CrossEncoder

    name = "hard_bench"
    docs = [json.loads(l) for l in open(DATA_DIR / "hard_bench_docs.jsonl")]
    queries = [json.loads(l) for l in open(DATA_DIR / "hard_bench_queries.jsonl")]
    gold_rows = [json.loads(l) for l in open(DATA_DIR / "hard_bench_gold.jsonl")]
    gold = {g["query_id"]: g["relevant_doc_ids"] for g in gold_rows}

    print("Loading cross-encoder...")
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    reranker = CrossEncoderReranker(
        CrossEncoderRerankerParams(
            cross_encoder=ce,
            max_input_length=512,
        )
    )

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

    print("Reranking...")
    per_q_r = {}
    for qid in qids:
        s_top = topk_from_scores(per_q_s[qid], RERANK_TOP_K)
        t_top = topk_from_scores(per_q_t[qid], RERANK_TOP_K)
        union = list(dict.fromkeys(s_top + t_top))[:RERANK_TOP_K]
        per_q_r[qid] = await rerank_topk(
            reranker, q_text[qid], union, doc_text, len(union)
        )

    # Find queries where w=0.6 hits and w~0.07 misses
    diagnostic = []
    for q in queries:
        qid = q["query_id"]
        gold_set = set(gold.get(qid, []))
        if not gold_set:
            continue
        rank_06 = fuse_at_w(per_q_t[qid], per_q_r[qid], per_q_s[qid], 0.6)
        rank_07 = fuse_at_w(per_q_t[qid], per_q_r[qid], per_q_s[qid], 0.069)
        hit_06 = next((i + 1 for i, d in enumerate(rank_06[:5]) if d in gold_set), None)
        hit_07 = next((i + 1 for i, d in enumerate(rank_07[:5]) if d in gold_set), None)
        if hit_06 == 1 and (hit_07 is None or hit_07 > 1):
            diagnostic.append(
                {
                    "qid": qid,
                    "q_text": q_text[qid],
                    "gold": list(gold_set),
                    "hit_06": hit_06,
                    "hit_07": hit_07,
                    "rank_06": rank_06[:5],
                    "rank_07": rank_07[:5],
                }
            )

    print(f"\n{len(diagnostic)} queries where w=0.6 wins rank-1 and w=0.069 doesn't")
    print("=" * 100)

    # Round 1 of bisect_thirds_capped over [0, 0.7]:
    #   L=0.7, c_left = 0.7/3 ≈ 0.233, c_right = 2*0.7/3 ≈ 0.467
    c_left, c_right = 0.7 / 3.0, 2 * 0.7 / 3.0
    print(f"\nRound 1 comparison: c_left={c_left:.3f} vs c_right={c_right:.3f}")
    print("(higher c_right means more weight on T = interval+tag+lattice)")
    print()

    # Pick 5 representative cases for deep inspection
    for case in diagnostic[:5]:
        qid = case["qid"]
        gold_set = set(case["gold"])
        print(f"\n{'=' * 100}")
        print(f"QUERY [{qid}]: {case['q_text']}")
        print(f"GOLD doc(s): {case['gold']}")
        print(f"  At w=0.6: gold at rank {case['hit_06']}")
        print(f"  At w=0.069: gold at rank {case['hit_07']}")

        for w, label in [
            (c_left, "c_LEFT  (w_T=0.233)"),
            (c_right, "c_RIGHT (w_T=0.467)"),
        ]:
            ranking, _ = get_top5_at_w(
                per_q_t[qid], per_q_r[qid], per_q_s[qid], w, doc_text
            )
            print(f"\n  --- {label} ---")
            for rk, did in enumerate(ranking, 1):
                marker = " *** GOLD ***" if did in gold_set else ""
                t_s = per_q_t[qid].get(did, 0.0)
                r_s = per_q_r[qid].get(did, 0.0)
                snippet = doc_text.get(did, "")[:120].replace("\n", " ")
                print(f"    {rk}. [{did}] T={t_s:.3f} R={r_s:.3f}{marker}")
                print(f"       {snippet}")

    # Save full set
    out_path = ROOT / "results" / "diagnostic_picker_bias.json"
    out_data = []
    for case in diagnostic:
        qid = case["qid"]
        gold_set = set(case["gold"])
        case_out = dict(case)
        for w, key in [(c_left, "left_w0233"), (c_right, "right_w0467")]:
            ranking, _ = get_top5_at_w(
                per_q_t[qid], per_q_r[qid], per_q_s[qid], w, doc_text
            )
            case_out[key] = [
                {
                    "doc_id": did,
                    "is_gold": did in gold_set,
                    "T": per_q_t[qid].get(did, 0.0),
                    "R": per_q_r[qid].get(did, 0.0),
                    "snippet": doc_text.get(did, "")[:300],
                }
                for did in ranking
            ]
        out_data.append(case_out)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nFull diagnostic written to {out_path}")
    print(f"Total {len(diagnostic)} cases for analysis.")


if __name__ == "__main__":
    asyncio.run(main())
