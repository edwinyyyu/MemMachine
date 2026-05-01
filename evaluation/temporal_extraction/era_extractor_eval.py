"""T_era_extractor — wire era_extractor as fallback when standard
ExtractorV2 produces no TEs for a query.

Pipeline:
  1. Run standard ExtractorV2 on docs and queries (run_v2_extract).
  2. For queries where ExtractorV2 returned [], run EraExtractor as fallback;
     merge any extracted TEs into q_ext.
  3. (Doc-side era extraction is OFF by default — the era_refs benchmark
     docs all have explicit calendar dates, so ExtractorV2 already picks
     them up; running era on docs is wasted compute and risks adding noisy
     wide intervals.)
  4. Reuse the same fuse_T_R + recency_additive pipeline as multi_channel_eval.py.

Reports R@1 for era_refs (target benchmark) and the regression set:
hard_bench, temporal_essential, tempreason_small, conjunctive_temporal,
multi_te_doc.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path

# Strip proxy env vars set by sandbox.
for _k in (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "FTP_PROXY",
    "ftp_proxy",
):
    os.environ.pop(_k, None)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT.parents[1] / ".env")

from advanced_common import LLMCaller
from era_extractor import EraExtractor
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
from rag_fusion import score_blend
from recency import (
    has_recency_cue,
    lambda_for_half_life,
    recency_scores_for_docs,
)
from salience_eval import (
    AXES,
    DATA_DIR,
    AxisDistribution,
    build_memory,
    embed_all,
    parse_iso,
    rank_semantic,
    run_v2_extract,
)
from schema import to_us

HALF_LIFE_DAYS = 21.0
CV_REF = 0.20
W_T_FUSE_TR = 0.4
W_R_FUSE_TR = 0.6
ADDITIVE_ALPHA = 0.5

ERA_TIMEOUT_S = 60.0


# -----------------------------------------------------------------------------
# Era fallback
# -----------------------------------------------------------------------------
async def run_era_fallback(items, label: str, llm: LLMCaller):
    """Run EraExtractor on each (id, text, ref_time) item; return {id: [TE]}."""
    ex = EraExtractor(llm)
    results: dict = {}

    async def one(iid, text, ref):
        try:
            tes = await asyncio.wait_for(ex.extract(text, ref), timeout=ERA_TIMEOUT_S)
            return iid, tes
        except asyncio.TimeoutError:
            return iid, []
        except Exception as e:
            print(f"  [{label}] era failed for {iid}: {e}", flush=True)
            return iid, []

    if not items:
        return results
    print(f"  [{label}] era-fallback for {len(items)} items...", flush=True)
    pairs = await asyncio.gather(*(one(*it) for it in items))
    for iid, tes in pairs:
        results[iid] = tes
    return results


# Switch regexes (copied from multi_channel_eval.py)
_MONTHS = (
    r"january|february|march|april|may|june|july|august|"
    r"september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec"
)
_SEASONS = r"spring|summer|autumn|fall|winter"
_TEMPORAL_ANCHOR_RE = re.compile(
    r"\b("
    r"(?:19|20)\d{2}"
    r"|q[1-4]\b"
    r"|(?:" + _MONTHS + r")\b"
    r"|(?:" + _SEASONS + r")\b"
    r"|(?:early|mid|late)\s+(?:19|20)?\d{2}s?"
    r"|(?:early|mid|late)\s+(?:" + _MONTHS + r"|" + _SEASONS + r")"
    r"|\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}/\d{1,2}/\d{2,4}"
    r"|\d{1,2}\s+(?:" + _MONTHS + r")"
    r"|(?:" + _MONTHS + r")\s+\d{1,2}(?:st|nd|rd|th)?"
    r")\b",
    re.IGNORECASE,
)


def has_temporal_anchor(query_text: str) -> bool:
    if not query_text:
        return False
    return _TEMPORAL_ANCHOR_RE.search(query_text) is not None


def hit_rank(ranking, gold_set, k=10):
    for i, d in enumerate(ranking[:k]):
        if d in gold_set:
            return i + 1
    return None


def rank_from_scores(scores):
    return [d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def normalize_rerank_full(rerank_partial, all_doc_ids, tail_score=0.0):
    if not rerank_partial:
        return dict.fromkeys(all_doc_ids, tail_score)
    vals = list(rerank_partial.values())
    rmin, rmax = min(vals), max(vals)
    span = (rmax - rmin) or 1.0
    out = {}
    for did in all_doc_ids:
        if did in rerank_partial:
            out[did] = (rerank_partial[did] - rmin) / span
        else:
            out[did] = tail_score
    return out


def fuse_T_R_blend(t_scores, r_scores, w_T=W_T_FUSE_TR):
    fused = score_blend(
        {"T": t_scores, "R": r_scores},
        {"T": w_T, "R": 1.0 - w_T},
        top_k_per=40,
        dispersion_cv_ref=CV_REF,
    )
    return [d for d, _ in fused]


def additive_with_recency(base_scores, rec_scores, cue, alpha=ADDITIVE_ALPHA):
    if not cue:
        return dict(base_scores)
    docs = set(base_scores) | set(rec_scores)
    out = {}
    for d in docs:
        out[d] = (1 - alpha) * base_scores.get(d, 0.0) + alpha * rec_scores.get(d, 0.0)
    return out


# -----------------------------------------------------------------------------
# Bench loop
# -----------------------------------------------------------------------------
async def run_bench(
    name,
    docs_path,
    queries_path,
    gold_path,
    cache_label,
    reranker,
    llm: LLMCaller,
    era_on_queries: bool = True,
    era_on_docs: bool = False,
):
    docs = [json.loads(l) for l in open(DATA_DIR / docs_path)]
    queries = [json.loads(l) for l in open(DATA_DIR / queries_path)]
    gold_rows = [json.loads(l) for l in open(DATA_DIR / gold_path)]
    gold = {g["query_id"]: g["relevant_doc_ids"] for g in gold_rows}

    print(f"\n=== {name}: {len(docs)} docs, {len(queries)} queries ===", flush=True)
    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs]
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries]
    doc_ext = await run_v2_extract(doc_items, f"{name}-docs", cache_label)
    q_ext = await run_v2_extract(q_items, f"{name}-queries", cache_label)

    # ---- Era fallback for queries with empty TE list -------------------------
    n_era_q_fired = 0
    n_era_q_added = 0
    era_q_qids: list[str] = []
    if era_on_queries:
        empty_q = [it for it in q_items if not q_ext.get(it[0])]
        era_q_qids = [it[0] for it in empty_q]
        if empty_q:
            era_results = await run_era_fallback(empty_q, f"{name}-q-era", llm)
            for qid, tes in era_results.items():
                if tes:
                    n_era_q_fired += 1
                    n_era_q_added += len(tes)
                    q_ext[qid] = tes
        print(
            f"  [{name}] era fallback: {n_era_q_fired}/{len(empty_q)} empty queries got TEs"
            f" (added {n_era_q_added} TEs)",
            flush=True,
        )

    n_era_d_fired = 0
    if era_on_docs:
        empty_d = [it for it in doc_items if not doc_ext.get(it[0])]
        if empty_d:
            era_d = await run_era_fallback(empty_d, f"{name}-d-era", llm)
            for did, tes in era_d.items():
                if tes:
                    n_era_d_fired += 1
                    doc_ext[did] = tes

    doc_text = {d["doc_id"]: d["text"] for d in docs}
    q_text = {q["query_id"]: q["text"] for q in queries}
    doc_ref_us = {d["doc_id"]: to_us(parse_iso(d["ref_time"])) for d in docs}
    q_ref_us = {q["query_id"]: to_us(parse_iso(q["ref_time"])) for q in queries}

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

    # Lattice for T_lblend
    lat_db = ROOT / "cache" / "era_extractor" / f"lat_{name}.sqlite"
    lat_db.parent.mkdir(parents=True, exist_ok=True)
    if lat_db.exists():
        lat_db.unlink()
    lat = LatticeStore(str(lat_db))
    for doc_id, tes in doc_ext.items():
        for te in tes:
            ts = lattice_tags_for_expression(te)
            lat.insert(doc_id, ts.absolute, ts.cyclical)
    qids = [q["query_id"] for q in queries]
    per_q_l = {
        qid: lattice_retrieve_multi(lat, q_ext.get(qid, []), down_levels=1)[0]
        for qid in qids
    }

    doc_bundles_for_rec: dict[str, list[dict]] = {}
    for did, mem in doc_mem.items():
        ivs = mem.get("intervals") or []
        if ivs:
            doc_bundles_for_rec[did] = [{"intervals": ivs}]
        else:
            doc_bundles_for_rec[did] = []

    print("  embedding + reranking...", flush=True)
    doc_embs_arr = await embed_all([d["text"] for d in docs])
    q_embs_arr = await embed_all([q["text"] for q in queries])
    doc_embs = {d["doc_id"]: doc_embs_arr[i] for i, d in enumerate(docs)}
    q_embs = {q["query_id"]: q_embs_arr[i] for i, q in enumerate(queries)}
    per_q_s = {qid: rank_semantic(qid, q_embs, doc_embs) for qid in qids}

    per_q_t = {
        qid: make_t_scores(
            q_mem.get(qid, {"intervals": [], "axes_merged": {}, "multi_tags": set()}),
            doc_mem,
            per_q_l.get(qid, {}),
        )
        for qid in qids
    }
    for qid in qids:
        for d in docs:
            per_q_t[qid].setdefault(d["doc_id"], 0.0)

    per_q_r_full: dict[str, dict[str, float]] = {}
    per_q_r_partial: dict[str, dict[str, float]] = {}
    for qid in qids:
        s_top = topk_from_scores(per_q_s[qid], RERANK_TOP_K)
        t_top = topk_from_scores(per_q_t[qid], RERANK_TOP_K)
        union = list(dict.fromkeys(s_top + t_top))[:RERANK_TOP_K]
        rs = await rerank_topk(reranker, q_text[qid], union, doc_text, len(union))
        per_q_r_partial[qid] = rs
        per_q_r_full[qid] = normalize_rerank_full(
            rs, [d["doc_id"] for d in docs], tail_score=0.0
        )

    lam = lambda_for_half_life(HALF_LIFE_DAYS)

    results = []
    for q in queries:
        qid = q["query_id"]
        gold_set = set(gold.get(qid, []))
        if not gold_set:
            continue

        T_active = has_temporal_anchor(q["text"])
        Recency_active = has_recency_cue(q["text"])

        t_scores = per_q_t[qid]
        r_full = per_q_r_full[qid]
        rerank_partial = per_q_r_partial[qid]
        s_scores = per_q_s[qid]

        rec_scores = recency_scores_for_docs(
            doc_bundles_for_rec,
            doc_ref_us,
            q_ref_us[qid],
            lam,
        )

        # Variant 1: rerank_only baseline
        rerank_only_rank = merge_with_tail(
            [
                d
                for d, _ in sorted(
                    rerank_partial.items(), key=lambda x: x[1], reverse=True
                )
            ],
            s_scores,
        )

        # Variant 2: fuse_T_R + recency_additive (current shipping baseline)
        fused_TR_scores = dict(
            score_blend(
                {"T": t_scores, "R": r_full},
                {"T": W_T_FUSE_TR, "R": W_R_FUSE_TR},
                top_k_per=40,
                dispersion_cv_ref=CV_REF,
            )
        )
        fused_TR_with_rec = additive_with_recency(
            fused_TR_scores,
            rec_scores,
            cue=Recency_active,
            alpha=ADDITIVE_ALPHA,
        )
        primary = rank_from_scores(fused_TR_with_rec)
        rank_fuse_TR_recAdd = primary + [
            d for d in rank_from_scores(s_scores) if d not in set(primary)
        ]

        results.append(
            {
                "qid": qid,
                "qtext": q.get("text", "")[:200],
                "gold": list(gold_set),
                "T_active": T_active,
                "Recency_active": Recency_active,
                "era_q_used": qid in era_q_qids,
                "rerank_only": hit_rank(rerank_only_rank, gold_set),
                "fuse_T_R_recAdd": hit_rank(rank_fuse_TR_recAdd, gold_set),
            }
        )

    return aggregate(results, name, n_era_q_fired, n_era_d_fired)


def aggregate(results, label, n_era_q_fired, n_era_d_fired):
    n = len(results)
    out = {
        "label": label,
        "n": n,
        "n_era_q_fired": n_era_q_fired,
        "n_era_d_fired": n_era_d_fired,
        "per_q": results,
    }
    for var in ("rerank_only", "fuse_T_R_recAdd"):
        ranks = [r[var] for r in results]
        r1 = sum(1 for x in ranks if x is not None and x <= 1)
        r5 = sum(1 for x in ranks if x is not None and x <= 5)
        mrr_v = sum(1.0 / x for x in ranks if x is not None) / n if n else 0.0
        out[var] = {
            "R@1": r1 / n if n else 0.0,
            "R@5": r5 / n if n else 0.0,
            "MRR": mrr_v,
            "r1_count": r1,
            "r5_count": r5,
        }
    print(f"  n={n}  era_q_fired={n_era_q_fired}", flush=True)
    for var in ("rerank_only", "fuse_T_R_recAdd"):
        d = out[var]
        print(
            f"  {var:22s}  R@1={d['R@1']:.3f} ({d['r1_count']}/{n})  "
            f"R@5={d['R@5']:.3f} ({d['r5_count']}/{n})  MRR={d['MRR']:.3f}",
            flush=True,
        )
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
async def main():
    print("Loading cross-encoder...", flush=True)
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

    # Single LLMCaller — its JSONCache persists across benches for free.
    llm = LLMCaller(concurrency=8)

    benches_main = [
        (
            "era_refs",
            "edge_era_refs_docs.jsonl",
            "edge_era_refs_queries.jsonl",
            "edge_era_refs_gold.jsonl",
            "edge-era_refs",
        ),
        (
            "hard_bench",
            "hard_bench_docs.jsonl",
            "hard_bench_queries.jsonl",
            "hard_bench_gold.jsonl",
            "v7l-hard_bench",
        ),
        (
            "temporal_essential",
            "temporal_essential_docs.jsonl",
            "temporal_essential_queries.jsonl",
            "temporal_essential_gold.jsonl",
            "v7l-temporal_essential",
        ),
        (
            "tempreason_small",
            "real_benchmark_small_docs.jsonl",
            "real_benchmark_small_queries.jsonl",
            "real_benchmark_small_gold.jsonl",
            "v7l-tempreason_small",
        ),
        (
            "conjunctive_temporal",
            "edge_conjunctive_temporal_docs.jsonl",
            "edge_conjunctive_temporal_queries.jsonl",
            "edge_conjunctive_temporal_gold.jsonl",
            "edge-conjunctive_temporal",
        ),
        (
            "multi_te_doc",
            "edge_multi_te_doc_docs.jsonl",
            "edge_multi_te_doc_queries.jsonl",
            "edge_multi_te_doc_gold.jsonl",
            "edge-multi_te_doc",
        ),
    ]

    out = {"benches": {}}
    for name, dp, qp, gp, cache_label in benches_main:
        if not (DATA_DIR / dp).exists():
            print(f"  [{name}] missing {dp} - skipping", flush=True)
            continue
        try:
            agg = await run_bench(
                name,
                dp,
                qp,
                gp,
                cache_label,
                reranker,
                llm,
                era_on_queries=True,
                era_on_docs=False,
            )
            out["benches"][name] = agg
        except Exception as e:
            import traceback

            traceback.print_exc()
            out["benches"][name] = {"error": str(e), "n": 0}

    llm.cache.save()

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "T_era_extractor.json"
    json_safe = {"benches": {}}
    for k, v in out["benches"].items():
        if "error" in v:
            json_safe["benches"][k] = v
            continue
        v2 = {kk: vv for kk, vv in v.items() if kk != "per_q"}
        v2["per_q"] = v.get("per_q", [])
        json_safe["benches"][k] = v2
    with open(json_path, "w") as f:
        json.dump(json_safe, f, indent=2, default=str)
    print(f"\nWrote {json_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
