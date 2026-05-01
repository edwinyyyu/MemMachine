"""Evaluate T_lblend on the 4 edge benchmarks. Compare T-only vs rerank_only vs
oracle (knows gold). For each benchmark, compute:
  - R@1 / R@5 / MRR for T_lblend
  - R@1 / R@5 / MRR for rerank_only (cross-encoder over union)
  - R@1 / R@5 / MRR for semantic_only (no rerank)
  - per-query top-5 with T components decomposed (interval / tag / lattice)
    so we can see which component drove a wrong pick.

Writes JSON dump to results/edge_T_components_{name}.json.
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
    T_ALPHA,
    T_DELTA,
    T_GAMMA,
    rerank_topk,
    topk_from_scores,
)
from lattice_cells import tags_for_expression as lattice_tags_for_expression
from lattice_retrieval import retrieve_multi as lattice_retrieve_multi
from lattice_store import LatticeStore
from salience_eval import (
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


def hit_rank(ranking, gold_set, k=10):
    for i, d in enumerate(ranking[:k]):
        if d in gold_set:
            return i + 1
    return None


def make_t_components(q_mem, doc_mem, l_per_doc):
    """Same as make_t_scores but returns each component separately."""
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
        tg = tag_score(q_tags, b["multi_tags"])
        ll = l_per_doc.get(did, 0.0)
        total = T_ALPHA * iv_norm + T_GAMMA * tg + T_DELTA * ll
        out[did] = {
            "iv_norm": iv_norm,
            "tag": tg,
            "lattice": ll,
            "total": total,
        }
    return out


async def eval_bench(name, reranker):
    docs_path = DATA_DIR / f"edge_{name}_docs.jsonl"
    queries_path = DATA_DIR / f"edge_{name}_queries.jsonl"
    gold_path = DATA_DIR / f"edge_{name}_gold.jsonl"
    docs = [json.loads(l) for l in open(docs_path)]
    queries = [json.loads(l) for l in open(queries_path)]
    gold_rows = [json.loads(l) for l in open(gold_path)]
    gold = {g["query_id"]: g["relevant_doc_ids"] for g in gold_rows}

    print(f"\n=== {name}: {len(docs)} docs, {len(queries)} queries ===")
    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs]
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries]
    doc_ext = await run_v2_extract(doc_items, f"edge-{name}-docs", f"edge-{name}")
    q_ext = await run_v2_extract(q_items, f"edge-{name}-queries", f"edge-{name}")
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

    lat_db = ROOT / "cache" / "edge" / f"lat_{name}.sqlite"
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

    per_q_t_components = {
        qid: make_t_components(
            q_mem.get(qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()}),
            doc_mem,
            per_q_l.get(qid, {}),
        )
        for qid in qids
    }
    per_q_t = {
        qid: {d: c["total"] for d, c in comps.items()}
        for qid, comps in per_q_t_components.items()
    }

    # rerank: union of top-50 semantic + top-50 T
    print("  reranking...")
    per_q_r = {}
    for qid in qids:
        s_top = topk_from_scores(per_q_s[qid], RERANK_TOP_K)
        t_top = topk_from_scores(per_q_t[qid], RERANK_TOP_K)
        union = list(dict.fromkeys(s_top + t_top))[:RERANK_TOP_K]
        per_q_r[qid] = await rerank_topk(
            reranker, q_text[qid], union, doc_text, len(union)
        )

    # Compute hit ranks per variant
    results = []
    per_q_dump = {}
    for q in queries:
        qid = q["query_id"]
        gold_set = set(gold.get(qid, []))
        if not gold_set:
            continue
        # T_lblend ranking
        t_rank = sorted(per_q_t[qid].items(), key=lambda x: x[1], reverse=True)
        t_ranking = [d for d, _ in t_rank]
        # rerank ranking, with semantic tail
        r_rank = sorted(per_q_r[qid].items(), key=lambda x: x[1], reverse=True)
        seen = set(d for d, _ in r_rank)
        s_tail = sorted(per_q_s[qid].items(), key=lambda x: x[1], reverse=True)
        rerank_ranking = [d for d, _ in r_rank] + [
            d for d, _ in s_tail if d not in seen
        ]
        # semantic only
        s_rank = sorted(per_q_s[qid].items(), key=lambda x: x[1], reverse=True)
        sem_ranking = [d for d, _ in s_rank]

        results.append(
            {
                "qid": qid,
                "gold": list(gold_set),
                "t_lblend_rank": hit_rank(t_ranking, gold_set),
                "rerank_only_rank": hit_rank(rerank_ranking, gold_set),
                "semantic_only_rank": hit_rank(sem_ranking, gold_set),
            }
        )
        # for failure analysis: top-5 under T_lblend with components
        top5_t = []
        for d, total in t_rank[:5]:
            comp = per_q_t_components[qid][d]
            top5_t.append(
                {
                    "doc_id": d,
                    "is_gold": d in gold_set,
                    "text": doc_text[d][:200],
                    "iv_norm": round(comp["iv_norm"], 3),
                    "tag": round(comp["tag"], 3),
                    "lattice": round(comp["lattice"], 3),
                    "total": round(comp["total"], 3),
                }
            )
        # gold doc rank+components if not in top5
        gold_in_top5 = any(x["is_gold"] for x in top5_t)
        gold_info = None
        if not gold_in_top5:
            for gid in gold_set:
                gpos = next((i for i, (d, _) in enumerate(t_rank) if d == gid), None)
                gcomp = per_q_t_components[qid].get(gid, {})
                gold_info = {
                    "doc_id": gid,
                    "rank": (gpos + 1) if gpos is not None else None,
                    "text": doc_text.get(gid, "")[:200],
                    "iv_norm": round(gcomp.get("iv_norm", 0.0), 3),
                    "tag": round(gcomp.get("tag", 0.0), 3),
                    "lattice": round(gcomp.get("lattice", 0.0), 3),
                    "total": round(gcomp.get("total", 0.0), 3),
                }
        per_q_dump[qid] = {
            "query": q_text[qid],
            "ref_time": q["ref_time"],
            "top5_t_lblend": top5_t,
            "gold_outside_top5": gold_info,
        }
    return results, per_q_dump


def aggregate(results, label):
    n = len(results)
    out = {}
    for var in ("t_lblend_rank", "rerank_only_rank", "semantic_only_rank"):
        ranks = [r[var] for r in results]
        r1 = sum(1 for x in ranks if x is not None and x <= 1)
        r5 = sum(1 for x in ranks if x is not None and x <= 5)
        mrr = sum(1.0 / x for x in ranks if x is not None) / n if n else 0.0
        out[var] = {
            "R@1": r1 / n,
            "R@5": r5 / n,
            "MRR": mrr,
            "r1_count": r1,
            "r5_count": r5,
            "n": n,
        }
    print(f"\n=== {label} ===")
    for var, m in out.items():
        print(
            f"  {var:20} R@1={m['r1_count']:2}/{m['n']} ({m['R@1']:.3f})  R@5={m['r5_count']:2}/{m['n']} ({m['R@5']:.3f})  MRR={m['MRR']:.3f}"
        )
    return out


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

    benches = ["era_refs", "relative_time", "conjunctive_temporal", "multi_te_doc"]
    summary = {}
    for name in benches:
        try:
            results, per_q = await eval_bench(name, reranker)
            agg = aggregate(results, name)
            summary[name] = {"agg": agg, "results": results}
            out_path = ROOT / "results" / f"edge_T_components_{name}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump({"agg": agg, "per_q": per_q, "results": results}, f, indent=2)
            print(f"  -> {out_path}")
        except Exception as e:
            import traceback

            traceback.print_exc()
            summary[name] = {"error": str(e)}

    out_path = ROOT / "results" / "edge_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
