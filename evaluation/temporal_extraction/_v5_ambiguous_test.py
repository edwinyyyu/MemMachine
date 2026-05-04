"""Ambiguous-year diagnostic for v5.0 (LLM phrase classifier).

Runs the v5 pipeline on either the basic ambiguous_year bench or the
adversarial extension. Reports fusion-aware metrics:
  - any_hit@K: rank of first gold (sanity)
  - all_recall@K: fraction of gold in top K (HEADLINE — fusion test)
  - year_coverage@K: distinct gold years/instances in top K / # gold instances

Usage:
  uv run python _v5_ambiguous_test.py basic    # ambiguous_year (12 q)
  uv run python _v5_ambiguous_test.py adv      # ambiguous_year_adv (12 q)
"""

from __future__ import annotations

PIPELINE_VERSION = "pipeline-v5.0"
PLANNER_PROMPT_VERSION = "v4.0"
CLASSIFIER_PROMPT_VERSION = "v5.0"

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
from composition_eval_v3 import (
    EXTREMUM_MULT_ALPHA,
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
K_VALUES = (1, 3, 5, 10)


VARIANTS = {
    "basic": {
        "name": "ambiguous_year",
        "docs": "ambiguous_year_docs.jsonl",
        "queries": "ambiguous_year_queries.jsonl",
        "gold": "ambiguous_year_gold.jsonl",
        "cache_label": "edge-ambiguous_year",
    },
    "adv": {
        "name": "ambiguous_year_adv",
        "docs": "ambiguous_year_adv_docs.jsonl",
        "queries": "ambiguous_year_adv_queries.jsonl",
        "gold": "ambiguous_year_adv_gold.jsonl",
        "cache_label": "edge-ambiguous_year_adv",
    },
}


def _gold_year(doc_id, doc_ref_us):
    import datetime

    us = doc_ref_us[doc_id]
    return datetime.datetime.fromtimestamp(
        us / 1_000_000, tz=datetime.timezone.utc
    ).year


async def main(variant: str = "basic"):
    cfg = VARIANTS[variant]
    print(f"Loading cross-encoder for variant={variant}...", flush=True)
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

    docs = [json.loads(l) for l in open(DATA_DIR / cfg["docs"])]
    queries = [json.loads(l) for l in open(DATA_DIR / cfg["queries"])]
    gold_rows = [json.loads(l) for l in open(DATA_DIR / cfg["gold"])]
    gold = {g["query_id"]: g["relevant_doc_ids"] for g in gold_rows}

    print(
        f"\n=== {cfg['name']}: {len(docs)} docs, {len(queries)} queries ===", flush=True
    )

    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs]
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries]
    doc_ext = await run_v2_extract(doc_items, f"{cfg['name']}-docs", cfg["cache_label"])
    _ = await run_v2_extract(q_items, f"{cfg['name']}-queries", cfg["cache_label"])

    doc_text = {d["doc_id"]: d["text"] for d in docs}
    q_text = {q["query_id"]: q["text"] for q in queries}
    q_ref = {q["query_id"]: q["ref_time"] for q in queries}
    doc_ref_us = {d["doc_id"]: to_us(parse_iso(d["ref_time"])) for d in docs}
    all_dids = list(doc_ref_us.keys())

    plan_items = [(q["query_id"], q["text"], q["ref_time"]) for q in queries]
    plans: dict[str, QueryPlanV4] = await planner.plan_many(plan_items)

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
        print(f"  {qid}: leaves={leaves}")

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

    print("\n--- Phrase classifications ---", flush=True)
    for tag, c in classes.items():
        qid, ci, li, leaf = leaf_lookup[tag]
        print(
            f"  {tag}: phrase={leaf.phrase!r:30s} kind={c.kind:18s} reason={c.rationale[:60]}"
        )

    # Extract intervals only for calendar_pin leaves.
    win_items = []
    for tag, (qid, ci, li, leaf) in leaf_lookup.items():
        cls = classes.get(tag)
        if cls and cls.kind == "calendar_pin":
            win_items.append((tag, leaf.phrase, parse_iso(q_ref[qid])))
    win_ext = (
        await run_v2_extract(
            win_items,
            f"{cfg['name']}-constraints-v5",
            f"{cfg['cache_label']}-constraints-v5",
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

    def leaf_anchor(qid, ci, li, leaf):
        tag = f"{qid}__c{ci}__l{li}"
        cls = classes.get(tag)
        kind = cls.kind if cls else "recurring_period"
        if kind == "calendar_pin":
            tes = win_ext.get(tag, [])
            max_conf = max((te.confidence for te in tes), default=0.0)
            if max_conf < CONF_FLOOR:
                return [], "low_conf"
            ivs = []
            for te in tes:
                ivs.extend(flatten_intervals(te))
            return ivs, "calendar_pin"
        if kind == "anaphoric_event":
            return corpus_anchor_ivs.get((qid, ci, li), []), "anaphoric_event"
        return [], kind

    rows = []
    for q in queries:
        qid = q["query_id"]
        plan = plans.get(qid) or QueryPlanV4()

        valid_includes_filt = []
        valid_excludes_filt = []
        for ci, clause in enumerate(plan.expr):
            for li, leaf in enumerate(clause):
                anchor_ivs, _ = leaf_anchor(qid, ci, li, leaf)
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
            ivs, _ = leaf_anchor(qid, ci, li, leaf)
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

        gold_ranks = {}
        for g in gold_set:
            try:
                gold_ranks[g] = ranking.index(g) + 1
            except ValueError:
                gold_ranks[g] = None

        gold_years = {_gold_year(g, doc_ref_us) for g in gold_set}

        per_k = {}
        for k in K_VALUES:
            top_k_set = set(ranking[:k])
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
            }

        rows.append(
            {
                "qid": qid,
                "query": q_text[qid],
                "n_gold": len(gold_set),
                "gold_ranks": gold_ranks,
                "gold_year_for": {g: _gold_year(g, doc_ref_us) for g in gold_set},
                "per_k": per_k,
            }
        )

    n = len(rows)
    print("\n" + "=" * 80, flush=True)
    print(f"{'metric':28s} " + " ".join(f"@{k:<3d}" for k in K_VALUES))
    print("-" * 80)
    for metric_name in ("any_hit", "all_recall", "year_coverage"):
        vals = [sum(r["per_k"][k][metric_name] for r in rows) / n for k in K_VALUES]
        print(f"{metric_name:28s} " + " ".join(f"{v:>4.3f}" for v in vals))
    print("=" * 80)

    print("\n--- Per-query gold ranks ---")
    for r in rows:
        rank_parts = []
        for g, rk in sorted(
            r["gold_ranks"].items(), key=lambda kv: r["gold_year_for"][kv[0]]
        ):
            yr = r["gold_year_for"][g]
            rank_parts.append(f"{yr}:{rk if rk is not None else 'X'}")
        rank_str = " ".join(rank_parts)
        q_short = r["query"][:48]
        print(f"{r['qid']:14s} {q_short:50s} {rank_str}")

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"T_v5_{cfg['name']}.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "pipeline_version": PIPELINE_VERSION,
                "classifier_prompt_version": CLASSIFIER_PROMPT_VERSION,
                "variant": variant,
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
                "classifier_stats": classifier.stats(),
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nWrote {json_path}", flush=True)


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "basic"
    asyncio.run(main(variant))
