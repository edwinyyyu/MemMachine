"""Composition per-type failure breakdown for v5.1.

The composition bench has 5 types (5 queries each):
  A. recency × absolute      -- "latest from Q4 2023"
  B. negation × absolute     -- "in 2024 not in summer"
  C. causal × recency        -- "most recent after the migration"
  D. causal × absolute       -- "in Q3 2023 after the launch"
  E. open_ended × negation   -- "after 2020 but not in 2023"

Reuses the v5.1 pipeline (LLM phrase classifier + DNF planner + B-RMX
scoring). For each query, dumps:
  - plan (DNF leaves)
  - classifier kind per leaf
  - extraction confidence
  - mask source (calendar_pin / anaphoric_event / no-op)
  - gold rank
  - top-3 retrieved docs (id + rerank score + mask)

Aggregates R@1 / R@5 per type to point at the weakest composition mode.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections import defaultdict
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
from composition_eval_v3 import (
    EXTREMUM_MULT_ALPHA,
    hit_rank,
    linear_recency_scores,
    normalize_dict,
    normalize_rerank_full,
    rank_from_scores,
    rank_semantic,
)
from force_pick_optimizers_eval import rerank_topk
from phrase_classifier import PhraseClassifier
from query_planner_v4 import QueryPlannerV4, QueryPlanV4, evaluate_dnf_mask
from salience_eval import (
    DATA_DIR,
    embed_all,
    flatten_intervals,
    parse_iso,
    run_v2_extract,
)
from schema import to_us

CONF_FLOOR = 0.5
PLANNER_PROMPT_VERSION = "v4.0"
CACHE_LABEL = "edge-composition"


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
    classifier = PhraseClassifier()

    docs = [json.loads(l) for l in open(DATA_DIR / "composition_docs.jsonl")]
    queries = [json.loads(l) for l in open(DATA_DIR / "composition_queries.jsonl")]
    gold_rows = [json.loads(l) for l in open(DATA_DIR / "composition_gold.jsonl")]
    gold = {g["query_id"]: g["relevant_doc_ids"] for g in gold_rows}

    print(
        f"\n=== composition: {len(docs)} docs, {len(queries)} queries ===", flush=True
    )

    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs]
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries]
    doc_ext = await run_v2_extract(doc_items, "composition-docs", CACHE_LABEL)
    _ = await run_v2_extract(q_items, "composition-queries", CACHE_LABEL)

    doc_text = {d["doc_id"]: d["text"] for d in docs}
    q_text = {q["query_id"]: q["text"] for q in queries}
    q_ref = {q["query_id"]: q["ref_time"] for q in queries}
    q_type = {q["query_id"]: q.get("comp_type", "?") for q in queries}
    doc_ref_us = {d["doc_id"]: to_us(parse_iso(d["ref_time"])) for d in docs}
    all_dids = list(doc_ref_us.keys())

    plan_items = [(q["query_id"], q["text"], q["ref_time"]) for q in queries]
    plans: dict[str, QueryPlanV4] = await planner.plan_many(plan_items)

    classify_items = []
    leaf_lookup = {}
    for q in queries:
        qid = q["query_id"]
        plan = plans.get(qid)
        if not plan:
            continue
        for ci, clause in enumerate(plan.expr):
            for li, leaf in enumerate(clause):
                tag = f"{qid}__c{ci}__l{li}"
                classify_items.append(
                    (tag, q_text[qid], q_ref[qid], leaf.phrase, leaf.direction)
                )
                leaf_lookup[tag] = (qid, ci, li, leaf)
    classes = await classifier.classify_many(classify_items) if classify_items else {}

    win_items = []
    for tag, (qid, ci, li, leaf) in leaf_lookup.items():
        cls = classes.get(tag)
        if cls and cls.kind == "calendar_pin":
            win_items.append((tag, leaf.phrase, parse_iso(q_ref[qid])))
    win_ext = (
        await run_v2_extract(
            win_items, "composition-constraints-v5", f"{CACHE_LABEL}-constraints-v5"
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

    doc_embs_arr = await embed_all([d["text"] for d in docs])
    q_embs_arr = await embed_all([q["text"] for q in queries])
    doc_embs = {d["doc_id"]: doc_embs_arr[i] for i, d in enumerate(docs)}
    q_embs = {q["query_id"]: q_embs_arr[i] for i, q in enumerate(queries)}
    qids = [q["query_id"] for q in queries]
    per_q_s = {qid: rank_semantic(qid, q_embs, doc_embs) for qid in qids}

    anchor_keys_to_resolve = []
    for tag, (qid, ci, li, leaf) in leaf_lookup.items():
        cls = classes.get(tag)
        if cls and cls.kind == "anaphoric_event":
            anchor_keys_to_resolve.append((qid, ci, li, leaf.phrase))
    corpus_anchor_ivs = {}
    corpus_anchor_doc = {}
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
                    best_sim, best_did = sim, did
            if best_did is not None:
                ivs = []
                for te in doc_ext.get(best_did, []):
                    ivs.extend(flatten_intervals(te))
                if ivs:
                    corpus_anchor_ivs[(qid, ci, li)] = ivs
                    corpus_anchor_doc[(qid, ci, li)] = (best_did, best_sim)

    def leaf_anchor(qid, ci, li, leaf):
        tag = f"{qid}__c{ci}__l{li}"
        cls = classes.get(tag)
        kind = cls.kind if cls else "recurring_period"
        if kind == "calendar_pin":
            tes = win_ext.get(tag, [])
            max_conf = max((te.confidence for te in tes), default=0.0)
            if max_conf < CONF_FLOOR:
                return [], "low_conf", max_conf
            ivs = []
            for te in tes:
                ivs.extend(flatten_intervals(te))
            return ivs, "calendar_pin", max_conf
        if kind == "anaphoric_event":
            ivs = corpus_anchor_ivs.get((qid, ci, li), [])
            return ivs, "anaphoric_event", 0.0
        return [], kind, 0.0

    rows = []
    for q in queries:
        qid = q["query_id"]
        plan = plans.get(qid) or QueryPlanV4()

        leaf_traces = []
        valid_includes_filt = []
        valid_excludes_filt = []
        for ci, clause in enumerate(plan.expr):
            for li, leaf in enumerate(clause):
                anchor_ivs, src, conf = leaf_anchor(qid, ci, li, leaf)
                anchor_doc = corpus_anchor_doc.get((qid, ci, li))
                leaf_traces.append(
                    {
                        "phrase": leaf.phrase,
                        "dir": leaf.direction,
                        "src": src,
                        "n_intervals": len(anchor_ivs),
                        "conf": conf,
                        "anchor_doc": anchor_doc,
                    }
                )
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

        # === Pool diagnostic: per-channel gold containment ===
        gold_set_local = set(gold.get(qid, []))
        sem_top5 = sorted(per_q_s[qid].items(), key=lambda x: x[1], reverse=True)[:5]
        sem_top5_ids = {d for d, _ in sem_top5}
        sem_top5_has_gold = bool(gold_set_local & sem_top5_ids)
        sem_top10 = sorted(per_q_s[qid].items(), key=lambda x: x[1], reverse=True)[:10]
        sem_top10_ids = {d for d, _ in sem_top10}
        sem_top10_has_gold = bool(gold_set_local & sem_top10_ids)
        gold_in_eligible = bool(gold_set_local & set(eligible_filt))
        # Filter-survivor semantic top 5
        eligible_set = set(eligible_filt)
        sf_top5 = [
            (d, s)
            for d, s in sorted(per_q_s[qid].items(), key=lambda x: x[1], reverse=True)
            if d in eligible_set
        ][:5]
        sf_top5_ids = {d for d, _ in sf_top5}
        sf_top5_has_gold = bool(gold_set_local & sf_top5_ids)
        # Where in pure semantic ranking does gold sit?
        sem_ranking = [
            d for d, _ in sorted(per_q_s[qid].items(), key=lambda x: x[1], reverse=True)
        ]
        gold_sem_ranks = {
            g: (sem_ranking.index(g) + 1 if g in sem_ranking else None)
            for g in gold_set_local
        }
        rs_partial = await rerank_topk(reranker, q_text[qid], pool, doc_text, len(pool))
        r_full = normalize_rerank_full(rs_partial, [d["doc_id"] for d in docs], 0.0)

        def leaf_resolver(ci, li, leaf, qid=qid):
            ivs, _, _ = leaf_anchor(qid, ci, li, leaf)
            return ivs

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
        h = hit_rank(ranking, gold_set, k=10)
        top3 = [(d, round(rs[d], 4), round(mask[d], 3)) for d in ranking[:3]]

        rows.append(
            {
                "qid": qid,
                "type": q_type[qid],
                "query": q_text[qid],
                "rank": h,
                "n_pool": len(pool),
                "n_mask_passers": len(mask_passers),
                "extremum": (
                    "latest" if plan_latest else "earliest" if plan_earliest else None
                ),
                "leaves": leaf_traces,
                "top3": top3,
                "gold_in_pool": bool(gold_set & pool_set),
                "gold": list(gold_set),
                # Channel diagnostics
                "n_eligible_filt": len(eligible_filt),
                "gold_in_eligible_filt": gold_in_eligible,
                "gold_in_sem_top5": sem_top5_has_gold,
                "gold_in_sem_top10": sem_top10_has_gold,
                "gold_in_sf_top5": sf_top5_has_gold,
                "gold_sem_ranks": gold_sem_ranks,
            }
        )

    # Aggregate
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)

    print("\n" + "=" * 80)
    print(
        f"{'Type':6s} {'n':>3s} {'R@1':>6s} {'R@5':>6s} {'R@10':>6s} {'gold-in-pool':>12s}"
    )
    print("-" * 80)
    for t in sorted(by_type.keys()):
        rs_ = by_type[t]
        n = len(rs_)
        r1 = sum(1 for r in rs_ if r["rank"] is not None and r["rank"] <= 1)
        r5 = sum(1 for r in rs_ if r["rank"] is not None and r["rank"] <= 5)
        r10 = sum(1 for r in rs_ if r["rank"] is not None and r["rank"] <= 10)
        gp = sum(1 for r in rs_ if r["gold_in_pool"])
        print(
            f"{t:6s} {n:>3d} {r1 / n:>6.3f} {r5 / n:>6.3f} {r10 / n:>6.3f} {gp:>4d}/{n}"
        )
    print("-" * 80)
    n_all = len(rows)
    r1_all = sum(1 for r in rows if r["rank"] is not None and r["rank"] <= 1)
    r5_all = sum(1 for r in rows if r["rank"] is not None and r["rank"] <= 5)
    print(f"{'ALL':6s} {n_all:>3d} {r1_all / n_all:>6.3f} {r5_all / n_all:>6.3f}")

    # Channel-attribution summary
    print("\n" + "=" * 80)
    print("CHANNEL ATTRIBUTION (where does gold appear?)")
    print("-" * 80)
    print(
        f"{'qid':16s} {'type':4s} {'pool?':5s} {'sem-top5?':9s} {'sem-top10?':10s} "
        f"{'filt?':5s} {'sf-top5?':8s} {'sem-rank':>10s}"
    )
    for r in rows:
        sem_ranks = r["gold_sem_ranks"]
        sr = ",".join(str(v) if v is not None else "X" for v in sem_ranks.values())
        print(
            f"{r['qid']:16s} {r['type']:4s} "
            f"{'Y' if r['gold_in_pool'] else 'N':5s} "
            f"{'Y' if r['gold_in_sem_top5'] else 'N':9s} "
            f"{'Y' if r['gold_in_sem_top10'] else 'N':10s} "
            f"{'Y' if r['gold_in_eligible_filt'] else 'N':5s} "
            f"{'Y' if r['gold_in_sf_top5'] else 'N':8s} "
            f"{sr:>10s}"
        )

    print("\n" + "=" * 80)
    print("FAILING QUERIES (gold not in top 5):")
    print("=" * 80)
    for t in sorted(by_type.keys()):
        rs_ = by_type[t]
        failing = [r for r in rs_ if r["rank"] is None or r["rank"] > 5]
        if not failing:
            continue
        print(f"\n--- Type {t} (failing {len(failing)}/{len(rs_)}) ---")
        for r in failing:
            print(
                f"  {r['qid']} [{r['type']}] rank={r['rank']} extremum={r['extremum']}"
                f" pool={r['n_pool']} mask_passers={r['n_mask_passers']}"
                f" gold_in_pool={r['gold_in_pool']}"
                f" gold_in_eligible_filt={r['gold_in_eligible_filt']}"
                f" gold_in_sem_top10={r['gold_in_sem_top10']}"
                f" sem_ranks={r['gold_sem_ranks']}"
            )
            print(f"    Q: {r['query']}")
            for lf in r["leaves"]:
                anchor = lf.get("anchor_doc")
                anchor_str = (
                    f" anchor_doc={anchor[0]} sim={anchor[1]:.3f}" if anchor else ""
                )
                print(
                    f"    leaf: phrase={lf['phrase']!r:30s} dir={lf['dir']:8s}"
                    f" src={lf['src']:18s} n_ivs={lf['n_intervals']}"
                    f" conf={lf['conf']:.2f}{anchor_str}"
                )
            print(f"    top3: {r['top3']}")
            print(f"    gold: {r['gold']}")

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "T_v5_composition_breakdown.json", "w") as f:
        json.dump(
            {"rows": rows, "by_type": {t: len(rs_) for t, rs_ in by_type.items()}},
            f,
            indent=2,
            default=str,
        )


if __name__ == "__main__":
    asyncio.run(main())
