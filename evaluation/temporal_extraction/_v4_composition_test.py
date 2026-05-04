"""Focused composition test with V4 planner (DNF: model emits the full
boolean tree). Mask is evaluated as max-over-clauses, min-over-leaves.

Composition R@1 across 5 types (A: extremum+window, B: window+negation,
C: extremum+after-event, D: window+event-bracket, E: range+negation).

Pipeline version log (for tracking iterations on the test stack):
  pipeline-v4.0:
    - planner: query_planner_v4.QueryPlannerV4 (DNF), prompt v4.0
    - retrieval pool: hybrid R-S/2 + R-SF/2 with topup, K=10
    - filter: build_filter_constraints over leaf extractions only
              (no corpus-anchor — pool enrichment vs precise mask)
    - mask: evaluate_dnf_mask (max-over-clauses, min-over-leaves)
    - corpus-anchor: top-1 doc by phrase embedding, fires when leaf
                     extraction empty OR phrase looks anaphoric
    - anaphoric heuristic: starts-with-"the " AND no year token
    - extremum boost: within-mask-passers linear recency × (1 + 3·rec)

Composition R@1: 0.60 (15/25). Baseline q10 hybrid + v2 planner: 0.40.
"""

from __future__ import annotations

PIPELINE_VERSION = "pipeline-v4.0"

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

from _v3_q1_retrieval_ablation import (
    doc_passes_filter,
)
from _v3_q10_hybrid import build_pool
from composition_eval_v3 import (
    EXTREMUM_MULT_ALPHA,
    hit_rank,
    linear_recency_scores,
    normalize_dict,
    normalize_rerank_full,
    rank_from_scores,
)
from force_pick_optimizers_eval import rerank_topk
from query_planner_v2 import Constraint, QueryPlan
from query_planner_v4 import QueryPlannerV4, QueryPlanV4, evaluate_dnf_mask
from salience_eval import (
    DATA_DIR,
    embed_all,
    flatten_intervals,
    parse_iso,
    rank_semantic,
    run_v2_extract,
)
from schema import to_us


def plan_v4_to_v2_for_filter(plan_v4: QueryPlanV4) -> QueryPlan:
    """Build a v2-style flat plan from a DNF v4 plan, for the
    DB-pushed retrieval filter. The filter side ANDs all leaves anyway
    (it pre-filters the candidate set), so we union all leaves across
    clauses. This errs toward over-inclusion, which is what we want at
    retrieval (mask handles the precise scoring later)."""
    flat_constraints = []
    seen = set()
    for clause in plan_v4.expr:
        for leaf in clause:
            key = (leaf.phrase, leaf.direction)
            if key in seen:
                continue
            seen.add(key)
            flat_constraints.append(
                Constraint(phrase=leaf.phrase, direction=leaf.direction)
            )
    return QueryPlan(constraints=flat_constraints, extremum=plan_v4.extremum)


async def run_bench(reranker, planner: QueryPlannerV4):
    docs = [json.loads(l) for l in open(DATA_DIR / "composition_docs.jsonl")]
    queries = [json.loads(l) for l in open(DATA_DIR / "composition_queries.jsonl")]
    gold_rows = [json.loads(l) for l in open(DATA_DIR / "composition_gold.jsonl")]
    gold = {g["query_id"]: g["relevant_doc_ids"] for g in gold_rows}

    print(
        f"\n=== composition (v4 DNF): {len(docs)} docs, {len(queries)} queries ===",
        flush=True,
    )

    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs]
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries]
    doc_ext = await run_v2_extract(doc_items, "comp-docs", "edge-composition")
    q_ext = await run_v2_extract(q_items, "comp-queries", "edge-composition")

    doc_text = {d["doc_id"]: d["text"] for d in docs}
    q_text = {q["query_id"]: q["text"] for q in queries}
    q_type = {q["query_id"]: q["comp_type"] for q in queries}
    doc_ref_us = {d["doc_id"]: to_us(parse_iso(d["ref_time"])) for d in docs}
    all_dids = list(doc_ref_us.keys())

    print("  planning (v4)...", flush=True)
    plan_items = [(q["query_id"], q["text"], q["ref_time"]) for q in queries]
    plans: dict[str, QueryPlanV4] = await planner.plan_many(plan_items)

    # Per-leaf constraint extraction. Tag = qid__cCi__lLj
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
            win_items, "comp-constraints", "edge-composition-constraints-v4"
        )
        if win_items
        else {}
    )

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

    print("  embedding...", flush=True)
    doc_embs_arr = await embed_all([d["text"] for d in docs])
    q_embs_arr = await embed_all([q["text"] for q in queries])
    doc_embs = {d["doc_id"]: doc_embs_arr[i] for i, d in enumerate(docs)}
    q_embs = {q["query_id"]: q_embs_arr[i] for i, q in enumerate(queries)}

    qids = [q["query_id"] for q in queries]
    per_q_s = {qid: rank_semantic(qid, q_embs, doc_embs) for qid in qids}

    # Corpus-anchor resolution for leaves whose extraction returned 0 ivs.
    print("  resolving corpus anchors...", flush=True)
    import re

    def looks_anaphoric(phrase: str) -> bool:
        """A phrase is anaphoric (refers to a corpus event) when it
        starts with the definite article 'the' and has no explicit year.
        Anaphoric phrases prefer corpus-anchor resolution because the
        extractor's deictic interpretation guesses a window relative to
        ref_time and is often wrong. Calendar phrases ('Q4 2024'),
        deictic phrases ('yesterday', 'last quarter'), and resolved
        event-anchors ('2007') all skip this preference."""
        if re.search(r"\b(19|20|21)\d{2}\b", phrase):
            return False
        return phrase.strip().lower().startswith("the ")

    anchor_keys_to_resolve = []  # list of (qid, ci, li, phrase)
    for q in queries:
        qid = q["query_id"]
        plan = plans.get(qid)
        if not plan:
            continue
        for ci, clause in enumerate(plan.expr):
            for li, leaf in enumerate(clause):
                tag = f"{qid}__c{ci}__l{li}"
                tes = win_ext.get(tag, [])
                has_ivs = any(flatten_intervals(te) for te in tes)
                # Resolve via corpus-anchor when extraction is empty
                # OR when phrase is anaphoric (corpus-anchor is more
                # reliable than deictic for "the X" event refs).
                if not has_ivs or looks_anaphoric(leaf.phrase):
                    anchor_keys_to_resolve.append((qid, ci, li, leaf.phrase))
    corpus_anchor_ivs: dict[tuple[str, int, int], list] = {}
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
                    print(
                        f"    {qid} c{ci}l{li} ({phrase!r}) -> {best_did} (sim={best_sim:.3f}, {len(ivs)} ivs)"
                    )

    print("  retrieving + scoring...", flush=True)
    rows = []
    for q in queries:
        qid = q["query_id"]
        plan = plans.get(qid) or QueryPlanV4()
        # Build retrieval filter from leaf extractions ONLY (skip
        # corpus-anchor). The filter is candidate enrichment — applying
        # corpus-anchor here would make it too strict and shrink the pool
        # below R-SF expectations. Corpus-anchor is reserved for scoring.
        valid_includes_filt = []
        valid_excludes_filt = []
        for ci, clause in enumerate(plan.expr):
            for li, leaf in enumerate(clause):
                tag = f"{qid}__c{ci}__l{li}"
                tes = win_ext.get(tag, [])
                anchor_ivs = []
                for te in tes:
                    anchor_ivs.extend(flatten_intervals(te))
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

        # DNF mask resolver — for anaphoric phrases ("the X"), prefer
        # corpus-anchor (the doc that defines the event) over the
        # extractor's deictic guess. For calendar phrases ("Q4 2024"),
        # use extraction.
        def leaf_resolver(ci, li, leaf, qid=qid):
            corpus_ivs = corpus_anchor_ivs.get((qid, ci, li))
            if looks_anaphoric(leaf.phrase) and corpus_ivs:
                return corpus_ivs
            tag = f"{qid}__c{ci}__l{li}"
            tes = win_ext.get(tag, [])
            anchor_ivs = []
            for te in tes:
                anchor_ivs.extend(flatten_intervals(te))
            if not anchor_ivs and corpus_ivs:
                anchor_ivs = corpus_ivs
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
        rank = [d for d in rank_from_scores(rs) if d in pool_set and rs[d] > 0.0]
        gold_set = set(gold.get(qid, []))
        h = hit_rank(rank, gold_set, k=10)
        rows.append(
            {
                "qid": qid,
                "type": q_type[qid],
                "qtext": q_text[qid],
                "plan": plan.to_dict(),
                "rank": h,
                "top5": rank[:5],
                "gold": list(gold_set),
                "gold_in_pool": bool(gold_set & pool_set),
                "pool_size": len(pool),
            }
        )

    return rows


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
    planner = QueryPlannerV4()
    rows = await run_bench(reranker, planner)

    by_t = {}
    for r in rows:
        by_t.setdefault(r["type"], []).append(r)
    print("\n" + "=" * 80)
    print(f"{'qid':22s} {'type':4s} {'rank':>4s}  {'gold_in':7s}  qtext")
    print("-" * 80)
    for r in rows:
        rk = str(r["rank"]) if r["rank"] else "-"
        gp = "Y" if r["gold_in_pool"] else "N"
        print(f"{r['qid']:22s} {r['type']:4s} {rk:>4s}  {gp:7s}  {r['qtext']}")

    print("\n" + "=" * 60)
    print("Per-comp_type R@1:")
    for t in sorted(by_t):
        rs = by_t[t]
        n = len(rs)
        r1 = sum(1 for r in rs if r["rank"] is not None and r["rank"] <= 1)
        r5 = sum(1 for r in rs if r["rank"] is not None and r["rank"] <= 5)
        print(f"  {t}: R@1={r1}/{n}={r1 / n:.2f} R@5={r5}/{n}={r5 / n:.2f}")
    n = len(rows)
    r1 = sum(1 for r in rows if r["rank"] is not None and r["rank"] <= 1)
    r5 = sum(1 for r in rows if r["rank"] is not None and r["rank"] <= 5)
    print(f"\n  OVERALL: R@1={r1}/{n}={r1 / n:.2f} R@5={r5}/{n}={r5 / n:.2f}")

    print("\n" + "=" * 60)
    print("Failures (rank>1 or null):")
    for r in rows:
        if r["rank"] == 1:
            continue
        print(f"\n  {r['qid']} ({r['type']}, rank={r['rank']})")
        print(f"    Q: {r['qtext']}")
        print(f"    plan: {r['plan']}")
        print(f"    top5: {r['top5']}")
        print(
            f"    gold: {r['gold']}, in_pool: {r['gold_in_pool']}, pool_size: {r['pool_size']}"
        )

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "T_composition_test_v4.json"
    with open(json_path, "w") as f:
        json.dump(
            {"rows": rows, "planner_stats": planner.stats()}, f, indent=2, default=str
        )
    print(f"\nWrote {json_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
