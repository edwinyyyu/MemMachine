"""v5.0 — phrase-class gating via LLM, no regex.

Replaces v4.5's looks_calendar / looks_anaphoric / _PERSONAL_ERA_RE /
_RECURRING_PERIOD_WORD_RE / strip_fabricated_year regexes with a single
LLM phrase classifier (`phrase_classifier.PhraseClassifier`). Each
planner-emitted leaf is classified into one of:

    calendar_pin     -> use extraction-based mask
    anaphoric_event  -> use corpus-anchor mask
    recurring_period -> no-op (fuse via rerank)
    personal_era     -> no-op (fuse via rerank)
    generic_skip     -> no-op

Iteration log:
  v5.0 (CURRENT): regex-free phrase classifier. Targets ambiguous_year +
        ambiguous_year_adv benches, expected to maintain 12-bench parity
        with v4.5 (0.802 macro R@1, 0.933 macro R@5).

Runs over 14 benches (12 standard + ambiguous_year + ambiguous_year_adv).
The two ambiguous benches use a separate `all_recall@K` metric (see
`_v5_ambiguous_test.py`); standard `R@K` (any-gold-in-top-K) is reported
here for compatibility but it saturates trivially on multi-gold queries.
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
    hit_rank,
    linear_recency_scores,
    normalize_dict,
    normalize_rerank_full,
    rank_from_scores,
)
from force_pick_optimizers_eval import rerank_topk
from phrase_classifier import PhraseClassifier
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

BENCHES = [
    (
        "composition",
        "composition_docs.jsonl",
        "composition_queries.jsonl",
        "composition_gold.jsonl",
        "edge-composition",
    ),
    (
        "hard_bench",
        "hard_bench_docs.jsonl",
        "hard_bench_queries.jsonl",
        "hard_bench_gold.jsonl",
        "hard_bench",
    ),
    (
        "temporal_essential",
        "temporal_essential_docs.jsonl",
        "temporal_essential_queries.jsonl",
        "temporal_essential_gold.jsonl",
        "temporal_essential",
    ),
    (
        "tempreason_small",
        "real_benchmark_small_docs.jsonl",
        "real_benchmark_small_queries.jsonl",
        "real_benchmark_small_gold.jsonl",
        "real_benchmark_small",
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
    (
        "relative_time",
        "edge_relative_time_docs.jsonl",
        "edge_relative_time_queries.jsonl",
        "edge_relative_time_gold.jsonl",
        "edge-relative_time",
    ),
    (
        "era_refs",
        "edge_era_refs_docs.jsonl",
        "edge_era_refs_queries.jsonl",
        "edge_era_refs_gold.jsonl",
        "edge-era_refs",
    ),
    (
        "open_ended_date",
        "open_ended_date_docs.jsonl",
        "open_ended_date_queries.jsonl",
        "open_ended_date_gold.jsonl",
        "edge-open_ended_date",
    ),
    (
        "causal_relative",
        "causal_relative_docs.jsonl",
        "causal_relative_queries.jsonl",
        "causal_relative_gold.jsonl",
        "edge-causal_relative",
    ),
    (
        "latest_recent",
        "latest_recent_docs.jsonl",
        "latest_recent_queries.jsonl",
        "latest_recent_gold.jsonl",
        "edge-latest_recent",
    ),
    (
        "negation_temporal",
        "negation_temporal_docs.jsonl",
        "negation_temporal_queries.jsonl",
        "negation_temporal_gold.jsonl",
        "edge-negation_temporal",
    ),
]


CONF_FLOOR = 0.5


async def run_bench(
    name,
    docs_path,
    queries_path,
    gold_path,
    cache_label,
    reranker,
    planner: QueryPlannerV4,
    classifier: PhraseClassifier,
):
    docs = [json.loads(l) for l in open(DATA_DIR / docs_path)]
    queries = [json.loads(l) for l in open(DATA_DIR / queries_path)]
    gold_rows = [json.loads(l) for l in open(DATA_DIR / gold_path)]
    gold = {g["query_id"]: g["relevant_doc_ids"] for g in gold_rows}

    print(f"\n=== {name}: {len(docs)} docs, {len(queries)} queries ===", flush=True)

    doc_items = [(d["doc_id"], d["text"], parse_iso(d["ref_time"])) for d in docs]
    q_items = [(q["query_id"], q["text"], parse_iso(q["ref_time"])) for q in queries]
    doc_ext = await run_v2_extract(doc_items, f"{name}-docs", cache_label)
    _ = await run_v2_extract(q_items, f"{name}-queries", cache_label)

    doc_text = {d["doc_id"]: d["text"] for d in docs}
    q_text = {q["query_id"]: q["text"] for q in queries}
    doc_ref_us = {d["doc_id"]: to_us(parse_iso(d["ref_time"])) for d in docs}
    all_dids = list(doc_ref_us.keys())

    plan_items = [(q["query_id"], q["text"], q["ref_time"]) for q in queries]
    plans: dict[str, QueryPlanV4] = await planner.plan_many(plan_items)

    # Classify every leaf phrase via LLM. Tag uses the same scheme as
    # extraction so we can join later: f"{qid}__c{ci}__l{li}".
    classify_items = []
    leaf_lookup: dict[str, tuple] = {}  # tag -> (qid, ci, li, leaf)
    for q in queries:
        qid = q["query_id"]
        plan = plans.get(qid)
        if not plan:
            continue
        for ci, clause in enumerate(plan.expr):
            for li, leaf in enumerate(clause):
                tag = f"{qid}__c{ci}__l{li}"
                classify_items.append(
                    (tag, q_text[qid], q["ref_time"], leaf.phrase, leaf.direction)
                )
                leaf_lookup[tag] = (qid, ci, li, leaf)
    classes = await classifier.classify_many(classify_items) if classify_items else {}

    # Build win_items only for calendar_pin leaves (others don't need a
    # mask interval extracted).
    win_items = []
    for tag, (qid, ci, li, leaf) in leaf_lookup.items():
        cls = classes.get(tag)
        if cls and cls.kind == "calendar_pin":
            ref = parse_iso(
                next(q["ref_time"] for q in queries if q["query_id"] == qid)
            )
            win_items.append((tag, leaf.phrase, ref))
    win_ext = (
        await run_v2_extract(
            win_items, f"{name}-constraints-v5", f"{cache_label}-constraints-v5"
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

    # Corpus-anchor lookup for anaphoric_event leaves only.
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
                    best_sim = sim
                    best_did = did
            if best_did is not None:
                ivs = []
                for te in doc_ext.get(best_did, []):
                    ivs.extend(flatten_intervals(te))
                if ivs:
                    corpus_anchor_ivs[(qid, ci, li)] = ivs

    def leaf_anchor(qid, ci, li, leaf):
        """Return (intervals, source) for a leaf based on its class.
        intervals=[] means no-op (fuse via rerank)."""
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
        return [], kind  # recurring_period / personal_era / generic_skip

    rows = []
    for q in queries:
        qid = q["query_id"]
        plan = plans.get(qid) or QueryPlanV4()

        valid_includes_filt = []
        valid_excludes_filt = []
        for ci, clause in enumerate(plan.expr):
            for li, leaf in enumerate(clause):
                anchor_ivs, _src = leaf_anchor(qid, ci, li, leaf)
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
        rank = [d for d in rank_from_scores(rs) if d in pool_set and rs[d] > 0.0]
        gold_set = set(gold.get(qid, []))
        h = hit_rank(rank, gold_set, k=10)
        rows.append({"qid": qid, "rank": h, "gold_in_pool": bool(gold_set & pool_set)})

    n = len(rows)
    r1 = sum(1 for r in rows if r["rank"] is not None and r["rank"] <= 1)
    r5 = sum(1 for r in rows if r["rank"] is not None and r["rank"] <= 5)
    return {"n": n, "r1": r1, "r5": r5, "R@1": r1 / n, "R@5": r5 / n, "rows": rows}


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

    out = {}
    for spec in BENCHES:
        try:
            res = await run_bench(
                *spec, reranker=reranker, planner=planner, classifier=classifier
            )
            out[spec[0]] = res
            print(
                f"  {spec[0]:20s} R@1={res['R@1']:.3f} ({res['r1']}/{res['n']})  "
                f"R@5={res['R@5']:.3f}",
                flush=True,
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            out[spec[0]] = {"error": str(e), "n": 0, "R@1": 0.0, "R@5": 0.0}

    print("\n" + "=" * 80)
    print(f"{'bench':22s} {'n':>4s} {'R@1':>7s}  {'R@5':>7s}")
    print("-" * 80)
    valid = [
        (name, r) for name, r in out.items() if "error" not in r and r.get("n", 0) > 0
    ]
    for name, r in valid:
        print(f"{name:22s} {r['n']:>4d} {r['R@1']:>7.3f}  {r['R@5']:>7.3f}")
    macro_r1 = sum(r["R@1"] for _, r in valid) / max(1, len(valid))
    macro_r5 = sum(r["R@5"] for _, r in valid) / max(1, len(valid))
    print("-" * 80)
    print(f"{'MACRO':22s} {len(valid):>4d} {macro_r1:>7.3f}  {macro_r5:>7.3f}")

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "T_v5_full_eval.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "pipeline_version": PIPELINE_VERSION,
                "classifier_prompt_version": CLASSIFIER_PROMPT_VERSION,
                "benches": out,
                "macro_r1": macro_r1,
                "macro_r5": macro_r5,
                "planner_stats": planner.stats(),
                "classifier_stats": classifier.stats(),
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nWrote {json_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
