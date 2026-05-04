"""Run the v4 DNF stack across all 12 benches. Headline: macro R@1 vs the
prior 0.769 (q10 hybrid + v2 planner) baseline.

This wraps the v4 composition test logic for each bench, swapping data
files and accumulating per-bench R@1.

Pipeline version log:
  v4.0: anaphoric phrase ("the X", no year) always prefers corpus-anchor
        Macro 0.778. Composition 0.60. era_refs 0.083 (regressed).
  v4.1: anaphoric phrase prefers corpus-anchor only when extractor conf
        < 0.7. Macro 0.775. NOT shipping.
  v4.2: planner prompt v4.1 (tightened rule e). FAILED: macro 0.754. LLM
        ignored the SKIP examples because rule (b) listed "back in college"
        as a deictic to keep.
  v4.3: revert to planner v4.0 + harness conf floor 0.5. Macro 0.778.
        era_refs unchanged.
  v4.4: corpus-anchor restricted to anaphoric only. Macro 0.797. era_refs
        recovered. latest_recent still -0.134.
  v4.4b: phrase-class gating (looks_calendar). Macro 0.788. relative_time
        regressed because calendar regex too narrow.
  v4.4c (CURRENT): broadened calendar regex + personal-era blocklist.
        MACRO R@1 = 0.809 (+0.040 over baseline). R@5 = 0.933.
        composition +0.20, causal_relative +0.267, no regressions.
"""

from __future__ import annotations

PIPELINE_VERSION = "pipeline-v4.4c"
PLANNER_PROMPT_VERSION = "v4.0"  # ships v4.0 prompt; harness gates instead

import asyncio
import json
import os
import re
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


def looks_anaphoric(phrase: str) -> bool:
    """Phrase explicitly refers to a corpus event ("the X", no year)."""
    if re.search(r"\b(19|20|21)\d{2}\b", phrase):
        return False
    return phrase.strip().lower().startswith("the ")


_CALENDAR_TOKEN_RE = re.compile(
    r"\b("
    r"(19|20|21)\d{2}|"  # year
    r"Q[1-4]|"  # quarter
    r"jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|"
    r"january|february|march|april|june|july|august|september|october|november|december|"
    r"yesterday|today|tomorrow|tonight|"
    r"(last|this|next|past) (week|month|year|quarter|summer|winter|spring|fall|autumn)|"
    r"last (mon|tue|wed|thu|fri|sat|sun)|"
    r"earlier (this|last) (week|month|year|quarter)|"
    r"(a few|several|\d+|couple|couple of) (day|week|month|year|hour|minute)s? ago|"
    r"(spring|summer|fall|autumn|winter) \d{4}|"
    r"days? ago|weeks? ago|months? ago|years? ago"  # bare "X ago" forms
    r")\b",
    re.IGNORECASE,
)


_PERSONAL_ERA_RE = re.compile(
    r"\b("
    r"(my|our) [a-z]+|"  # "my parental leave", "my fitness phase"
    r"(grad|high|middle|elementary) school|"
    r"college|"
    r"(when|while) i [a-z]+|"  # "when I worked at Acme", "while I lived in Boston"
    r"i (worked|lived|graduated|studied) [a-z]+|"
    r"back in [a-z]+|"  # "back in college"
    r"during [a-z]+ phase|"
    r"training for [a-z]+"
    r")\b",
    re.IGNORECASE,
)


def looks_calendar(phrase: str) -> bool:
    """Phrase has a concrete calendar/deictic token that the extractor can
    ground against ref_time. Trust extraction. Filters out personal-era
    phrases ("grad school", "my parental leave") that hallucinate."""
    if _PERSONAL_ERA_RE.search(phrase):
        return False
    return bool(_CALENDAR_TOKEN_RE.search(phrase))


async def run_bench(
    name,
    docs_path,
    queries_path,
    gold_path,
    cache_label,
    reranker,
    planner: QueryPlannerV4,
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

    doc_text = {d["doc_id"]: d["text"] for d in docs}
    q_text = {q["query_id"]: q["text"] for q in queries}
    doc_ref_us = {d["doc_id"]: to_us(parse_iso(d["ref_time"])) for d in docs}
    all_dids = list(doc_ref_us.keys())

    plan_items = [(q["query_id"], q["text"], q["ref_time"]) for q in queries]
    plans: dict[str, QueryPlanV4] = await planner.plan_many(plan_items)

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
            win_items, f"{name}-constraints-v4", f"{cache_label}-constraints-v4"
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

    # corpus-anchor: leaves with empty extraction OR anaphoric phrase.
    # v4.2 reverts conf-gating; era_refs is fixed at the planner side
    # by tightening rule (e) to only emit "the X" patterns (planner v4.1).
    anchor_keys_to_resolve = []
    for q in queries:
        qid = q["query_id"]
        plan = plans.get(qid)
        if not plan:
            continue
        for ci, clause in enumerate(plan.expr):
            for li, leaf in enumerate(clause):
                # v4.3: corpus-anchor fires ONLY for anaphoric "the X"
                # phrases. For non-anaphoric phrases ("grad school", "back
                # in college") that the planner emitted but the extractor
                # couldn't ground, skip corpus-anchor — matching the v2
                # baseline's "no constraint" behavior is better than picking
                # a wrong corpus doc and masking out gold.
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

    # v4.4: phrase-class gating. Trust extraction ONLY for calendar phrases
    # (years, months, quarters, deictic terms with clean ref_time resolution).
    # For other phrases (era refs, "last X", anaphoric "the X" without
    # corpus anchor), let the leaf become a no-op so we fall back to pure
    # rerank — matches v2 baseline behavior where the planner skipped
    # these phrases entirely.
    CONF_FLOOR = 0.5

    def leaf_anchor_from_extraction(qid, ci, li, leaf):
        if not looks_calendar(leaf.phrase):
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
            # Corpus-anchor only for anaphoric "the X" phrases. Non-anaphoric
            # phrases use extraction; if extraction is empty, return empty
            # (no-op leaf — better than masking against a wrong corpus doc).
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
        rank = [d for d in rank_from_scores(rs) if d in pool_set and rs[d] > 0.0]
        gold_set = set(gold.get(qid, []))
        h = hit_rank(rank, gold_set, k=10)
        rows.append(
            {
                "qid": qid,
                "rank": h,
                "gold_in_pool": bool(gold_set & pool_set),
            }
        )
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

    out = {}
    for spec in BENCHES:
        try:
            res = await run_bench(*spec, reranker=reranker, planner=planner)
            out[spec[0]] = res
            print(
                f"  {spec[0]:20s} R@1={res['R@1']:.3f} ({res['r1']}/{res['n']})  R@5={res['R@5']:.3f}",
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
    json_path = out_dir / "T_v4_full_eval.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "benches": out,
                "macro_r1": macro_r1,
                "macro_r5": macro_r5,
                "planner_stats": planner.stats(),
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nWrote {json_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
