"""Ambiguous-year diagnostic: runs the v4.4c pipeline on the ambiguous_year
bench and reports fusion-aware metrics.

Standard R@K saturates trivially when there are 3 gold docs per query (any
of them in top-K hits). This harness reports:
  - R@K (any gold in top K)         -- sanity, should be high
  - all_recall@K (frac of gold in top K)  -- HEADLINE: fusion test
  - year_coverage@K (distinct gold years in top K / # gold years) -- direct fusion test
  - per-query gold ranks            -- diagnostic

Pipeline mirrors `_v4_full_eval.py` v4.4c (DNF planner v4.0 + corpus-anchor
for "the X" + phrase-class gating + multiplicative recency).
"""

from __future__ import annotations

PIPELINE_VERSION = "pipeline-v4.4c"
PLANNER_PROMPT_VERSION = "v4.0"

import asyncio
import json
import os
import sys
from pathlib import Path

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

from _v3_q1_retrieval_ablation import doc_passes_filter
from _v3_q10_hybrid import build_pool
from _v4_full_eval import looks_anaphoric, looks_calendar, strip_fabricated_year
from composition_eval_v3 import (
    EXTREMUM_MULT_ALPHA,
    linear_recency_scores,
    normalize_dict,
    normalize_rerank_full,
    rank_from_scores,
    rank_semantic,
)
from force_pick_optimizers_eval import rerank_topk
from query_planner_v4 import QueryPlannerV4, QueryPlanV4, evaluate_dnf_mask
from salience_eval import (
    DATA_DIR,
    embed_all,
    flatten_intervals,
    parse_iso,
    run_v2_extract,
)
from schema import to_us

BENCH_NAME = "ambiguous_year"
DOCS_FILE = "ambiguous_year_docs.jsonl"
QUERIES_FILE = "ambiguous_year_queries.jsonl"
GOLD_FILE = "ambiguous_year_gold.jsonl"
CACHE_LABEL = "edge-ambiguous_year"

CONF_FLOOR = 0.5
K_VALUES = (1, 3, 5, 10)


def _gold_year(doc_id: str, doc_ref_us: dict[str, int]) -> int:
    """ambiguous_year docs have ref_time == the doc's actual date, so we
    can recover the year directly from doc_ref_us."""
    import datetime

    us = doc_ref_us[doc_id]
    return datetime.datetime.fromtimestamp(
        us / 1_000_000, tz=datetime.timezone.utc
    ).year


async def main():
    print("Loading cross-encoder...", flush=True)
    from memmachine_server.common.reranker.cross_encoder_reranker import (
        CrossEncoderReranker,
        CrossEncoderRerankerParams,
    )
    from sentence_transformers import CrossEncoder

    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    reranker = CrossEncoderReranker(
        CrossEncoderRerankerParams(cross_encoder=ce, max_input_length=512)
    )
    planner = QueryPlannerV4(prompt_version=PLANNER_PROMPT_VERSION)

    docs = [json.loads(l) for l in open(DATA_DIR / DOCS_FILE)]
    queries = [json.loads(l) for l in open(DATA_DIR / QUERIES_FILE)]
    gold_rows = [json.loads(l) for l in open(DATA_DIR / GOLD_FILE)]
    gold = {g["query_id"]: g["relevant_doc_ids"] for g in gold_rows}

    print(
        f"\n=== {BENCH_NAME}: {len(docs)} docs, {len(queries)} queries ===", flush=True
    )

    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs]
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries]
    doc_ext = await run_v2_extract(doc_items, f"{BENCH_NAME}-docs", CACHE_LABEL)
    q_ext = await run_v2_extract(q_items, f"{BENCH_NAME}-queries", CACHE_LABEL)

    doc_text = {d["doc_id"]: d["text"] for d in docs}
    q_text = {q["query_id"]: q["text"] for q in queries}
    doc_ref_us = {d["doc_id"]: to_us(parse_iso(d["ref_time"])) for d in docs}
    all_dids = list(doc_ref_us.keys())

    plan_items = [(q["query_id"], q["text"], q["ref_time"]) for q in queries]
    plans: dict[str, QueryPlanV4] = await planner.plan_many(plan_items)

    # Print plans for diagnostic
    print("\n--- Plans ---", flush=True)
    for q in queries:
        qid = q["query_id"]
        p = plans.get(qid)
        if not p:
            print(f"  {qid}: <no plan>")
            continue
        leaves = [
            (li, leaf.phrase, leaf.direction)
            for cl in p.expr
            for li, leaf in enumerate(cl)
        ]
        print(f"  {qid}: latest={p.latest_intent} leaves={leaves}")

    win_items = []
    for q in queries:
        qid = q["query_id"]
        ref = parse_iso(q["ref_time"])
        plan = plans.get(qid)
        if not plan:
            continue
        for ci, clause in enumerate(plan.expr):
            for li, leaf in enumerate(clause):
                tag = f"{qid}__c{ci}__l{li}"
                win_items.append((tag, leaf.phrase, ref))
    win_ext = (
        await run_v2_extract(
            win_items, f"{BENCH_NAME}-constraints-v4", f"{CACHE_LABEL}-constraints-v4"
        )
        if win_items
        else {}
    )

    # Print constraint extractions for diagnostic
    print("\n--- Extractions ---", flush=True)
    for tag, _, _ in win_items:
        tes = win_ext.get(tag, [])
        confs = [te.confidence for te in tes]
        ivs = []
        for te in tes:
            ivs.extend(flatten_intervals(te))
        print(f"  {tag}: {len(ivs)} intervals, conf={confs}")

    doc_ivs_flat = {}
    for did, tes in doc_ext.items():
        ivs = []
        for te in tes:
            ivs.extend(flatten_intervals(te))
        doc_ivs_flat[did] = ivs
    for d in docs:
        doc_ivs_flat.setdefault(d["doc_id"], [])

    doc_bundles_for_rec = {}
    for did, tes in doc_ext.items():
        ivs = []
        for te in tes:
            ivs.extend(flatten_intervals(te))
        doc_bundles_for_rec[did] = [{"intervals": ivs}] if ivs else []
    for d in docs:
        doc_bundles_for_rec.setdefault(d["doc_id"], [])

    doc_embs_arr = await embed_all([d["text"] for d in docs])
    q_embs_arr = await embed_all([q["text"] for q in queries])
    doc_embs = {d["doc_id"]: doc_embs_arr[i] for i, d in enumerate(docs)}
    q_embs = {q["query_id"]: q_embs_arr[i] for i, q in enumerate(queries)}

    qids = [q["query_id"] for q in queries]
    per_q_s = {qid: rank_semantic(qid, q_embs, doc_embs) for qid in qids}

    # corpus-anchor for anaphoric "the X" leaves
    anchor_keys_to_resolve = []
    for q in queries:
        qid = q["query_id"]
        plan = plans.get(qid)
        if not plan:
            continue
        for ci, clause in enumerate(plan.expr):
            for li, leaf in enumerate(clause):
                if looks_anaphoric(leaf.phrase):
                    anchor_keys_to_resolve.append((qid, ci, li, leaf.phrase))
    corpus_anchor_ivs = {}
    if anchor_keys_to_resolve:
        phrase_texts = [ph for _, _, _, ph in anchor_keys_to_resolve]
        phrase_embs = await embed_all(phrase_texts)
        import numpy as np

        doc_emb_norms = {
            did: (v, np.linalg.norm(v) or 1e-9) for did, v in doc_embs.items()
        }
        for (qid, ci, li, phrase), pemb in zip(anchor_keys_to_resolve, phrase_embs):
            pn = np.linalg.norm(pemb) or 1e-9
            best_did, best_sim = None, -1.0
            for did, (v, vn) in doc_emb_norms.items():
                sim = float(np.dot(pemb, v) / (pn * vn))
                if sim > best_sim:
                    best_sim = sim
                    best_did = did
            if best_did is not None:
                ivs = []
                for te in doc_ext.get(best_did, []):
                    ivs.extend(flatten_intervals(te))
                if ivs:
                    corpus_anchor_ivs[(qid, ci, li)] = ivs

    def leaf_anchor_from_extraction(qid, ci, li, leaf):
        eff_phrase = strip_fabricated_year(leaf.phrase, q_text[qid])
        if not looks_calendar(eff_phrase):
            return [], 0.0
        tag = f"{qid}__c{ci}__l{li}"
        tes = win_ext.get(tag, [])
        max_conf = max((te.confidence for te in tes), default=0.0)
        if max_conf < CONF_FLOOR:
            return [], max_conf
        ivs = []
        for te in tes:
            ivs.extend(flatten_intervals(te))
        return ivs, max_conf

    rows = []
    for q in queries:
        qid = q["query_id"]
        plan = plans.get(qid) or QueryPlanV4()

        valid_includes_filt = []
        valid_excludes_filt = []
        for ci, clause in enumerate(plan.expr):
            for li, leaf in enumerate(clause):
                anchor_ivs, _ = leaf_anchor_from_extraction(qid, ci, li, leaf)
                if not anchor_ivs:
                    continue
                if leaf.direction == "not_in":
                    valid_excludes_filt.append(anchor_ivs)
                else:
                    valid_includes_filt.append((leaf.direction, anchor_ivs))

        eligible_filt = [
            did
            for did in doc_ref_us
            if doc_passes_filter(
                doc_ivs_flat.get(did, []), valid_includes_filt, valid_excludes_filt
            )
        ]
        pool = build_pool("R-S_half_SF_half", per_q_s[qid], all_dids, eligible_filt)
        rs_partial = await rerank_topk(reranker, q_text[qid], pool, doc_text, len(pool))
        r_full = normalize_rerank_full(rs_partial, [d["doc_id"] for d in docs], 0.0)

        def leaf_resolver(ci, li, leaf, qid=qid):
            corpus_ivs = corpus_anchor_ivs.get((qid, ci, li))
            if looks_anaphoric(leaf.phrase) and corpus_ivs:
                return corpus_ivs
            anchor_ivs, _max_conf = leaf_anchor_from_extraction(qid, ci, li, leaf)
            return anchor_ivs

        mask = {
            did: evaluate_dnf_mask(plan, doc_ivs_flat.get(did, []), leaf_resolver)
            for did in pool
        }

        plan_latest = plan.latest_intent
        plan_earliest = plan.earliest_intent
        mask_passers = [did for did in pool if mask[did] >= 0.5]
        if (plan_latest or plan_earliest) and len(mask_passers) >= 2:
            rec_lin_mode = linear_recency_scores(
                {did: doc_bundles_for_rec.get(did, []) for did in mask_passers},
                {did: doc_ref_us[did] for did in mask_passers},
            )
        elif (plan_latest or plan_earliest) and len(pool) >= 2:
            rec_lin_mode = linear_recency_scores(
                {did: doc_bundles_for_rec.get(did, []) for did in pool},
                {did: doc_ref_us[did] for did in pool},
            )
        else:
            rec_lin_mode = {}

        r_pool = {did: r_full.get(did, 0.0) for did in pool}
        base = normalize_dict(r_pool)
        rs = {}
        for did in pool:
            b = base.get(did, 0.0) * mask[did]
            if plan_latest or plan_earliest:
                r = rec_lin_mode.get(did, 0.0)
                if plan_earliest:
                    r = 1.0 - r
                b *= 1.0 + EXTREMUM_MULT_ALPHA * r
            rs[did] = b

        pool_set = set(pool)
        ranking = [d for d in rank_from_scores(rs) if d in pool_set and rs[d] > 0.0]
        gold_set = set(gold.get(qid, []))

        # Per-query metrics
        gold_ranks = {}
        for g in gold_set:
            try:
                gold_ranks[g] = ranking.index(g) + 1
            except ValueError:
                gold_ranks[g] = None

        gold_years = {_gold_year(g, doc_ref_us) for g in gold_set}

        per_k = {}
        for k in K_VALUES:
            top_k = ranking[:k]
            top_k_set = set(top_k)
            n_gold_in_topk = len(top_k_set & gold_set)
            years_in_topk = {
                _gold_year(d, doc_ref_us) for d in top_k_set if d in gold_set
            }
            per_k[k] = {
                "any_hit": n_gold_in_topk > 0,
                "all_recall": n_gold_in_topk / len(gold_set) if gold_set else 0.0,
                "year_coverage": (
                    len(years_in_topk) / len(gold_years) if gold_years else 0.0
                ),
                "n_gold_in_topk": n_gold_in_topk,
                "n_gold": len(gold_set),
                "n_years_in_topk": len(years_in_topk),
                "n_gold_years": len(gold_years),
            }

        rows.append(
            {
                "qid": qid,
                "query": q_text[qid],
                "gold_in_pool": bool(gold_set & pool_set),
                "n_gold": len(gold_set),
                "gold_ranks": {g: r for g, r in gold_ranks.items()},
                "gold_year_for": {g: _gold_year(g, doc_ref_us) for g in gold_set},
                "per_k": per_k,
            }
        )

    # Aggregates
    n = len(rows)
    print("\n" + "=" * 80, flush=True)
    print(f"{'metric':28s} " + " ".join(f"@{k:<3d}" for k in K_VALUES))
    print("-" * 80)
    for metric_name in ("any_hit", "all_recall", "year_coverage"):
        vals = []
        for k in K_VALUES:
            avg = sum(r["per_k"][k][metric_name] for r in rows) / n
            vals.append(avg)
        print(f"{metric_name:28s} " + " ".join(f"{v:>4.3f}" for v in vals))
    print("=" * 80)
    print(f"n_queries={n}, n_gold_per_query={rows[0]['n_gold']}")
    print()

    # Per-query breakdown
    print("\n--- Per-query gold ranks ---")
    print(f"{'qid':12s} {'query':50s} {'ranks (year:rank)'}")
    print("-" * 100)
    for r in rows:
        rank_str_parts = []
        for g, rk in sorted(
            r["gold_ranks"].items(), key=lambda kv: r["gold_year_for"][kv[0]]
        ):
            yr = r["gold_year_for"][g]
            rank_str_parts.append(f"{yr}:{rk if rk is not None else 'X'}")
        rank_str = " ".join(rank_str_parts)
        q_short = r["query"][:48]
        print(f"{r['qid']:12s} {q_short:50s} {rank_str}")

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "T_v4_ambiguous_year.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "pipeline_version": PIPELINE_VERSION,
                "planner_prompt_version": PLANNER_PROMPT_VERSION,
                "n_queries": n,
                "rows": rows,
                "macro": {
                    metric_name: {
                        f"@{k}": sum(r["per_k"][k][metric_name] for r in rows) / n
                        for k in K_VALUES
                    }
                    for metric_name in ("any_hit", "all_recall", "year_coverage")
                },
                "planner_stats": planner.stats(),
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nWrote {json_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
