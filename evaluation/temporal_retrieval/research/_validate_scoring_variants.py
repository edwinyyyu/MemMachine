"""Multi-variant scoring A/B across the 35-bench suite.

Runs six scoring variants (baseline / frac_max / frac_min /
prob_andor / relation_wt / specificity) on the SAME planned queries
and extracted intervals — the planner and extractor are run once per
(query, doc); only the DNF-match scoring step varies. This makes the
whole sweep cheap and apples-to-apples.

Outputs R@1, R@5, all_R@5 per variant per bench, plus macro.

Run from ``evaluation/``::

    uv run python -m temporal_retrieval.research._validate_scoring_variants
"""
from __future__ import annotations

import asyncio
import json

import numpy as np

from temporal_retrieval import Doc, TemporalRetriever
from temporal_retrieval.core import (
    Interval,
    build_pool,
    doc_passes_filter,
    flatten_intervals,
    normalize_dict,
    normalize_rerank_full,
    recency_scores,
)
from temporal_retrieval.extractor_v3_3 import TemporalExtractorV3_3
from temporal_retrieval.planner import QueryPlanner
from temporal_retrieval.schema import parse_iso, to_us

from ._common import (
    ROOT,
    load_bench_jsonl,
    make_embed_fn,
    setup_env,
)
from .scoring_variants import FLOOR, VARIANTS

# Per-variant empty_doc_match. For fractional variants whose floor for
# weak temporal matches is FLOOR (0.3), use the same FLOOR for timeless
# docs so a weakly-matching temporal doc doesn't rank below an
# irrelevant timeless doc. Baseline and the older exploratory variants
# keep the current shipped 1.0.
EMPTY_DOC_MATCH = {
    "baseline":          1.0,
    "frac_max":          1.0,
    "frac_min":          1.0,
    "prob_andor":        1.0,
    "relation_wt":       1.0,
    "specificity":       1.0,
    # proximity_ab uses BINARY intersect — there's no inversion to
    # compensate for, so keep empty at 1.0 (was wrongly set to FLOOR
    # in the first iteration; b3zxnrym6 showed this cost −0.114 R@1
    # on adversarial).
    "proximity_ab":      1.0,
    # The two doc_frac variants do change intersect to fractional with
    # floor — empty=FLOOR keeps the inversion check (weak temporal
    # match vs timeless).
    "doc_frac":          FLOOR,
    "doc_frac_prox":     FLOOR,
    # Lean variants: raw doc-fraction (no floor), empty=0 — timeless
    # docs get no temporal boost, cosine alone carries them.
    "doc_frac_raw":      0.0,
    "doc_frac_raw_prox": 0.0,
}

setup_env()


BENCH_NAMES = [
    "adversarial", "allen", "ambiguous_year", "ambiguous_year_adv",
    "axis", "causal_relative", "composition", "cotemporal",
    "dense_cluster", "disc", "edge_conjunctive_temporal", "edge_era_refs",
    "edge_multi_te_doc", "edge_relative_time", "era", "goldilocks",
    "goldilocks_v2", "hard_bench", "hard_dense_cluster", "latest_recent",
    "lattice", "mixed_cue", "negation_temporal", "notin_multi_interval",
    "open_ended_date", "polarity", "precedents", "realq", "realq_deictic",
    "realq_v2", "sensitivity_curated", "speculative_anchors",
    "temporal_essential", "timeless_policies", "utterance",
    # NEW: targeted tests for fractional-vs-binary distinctions.
    "fractional_intent",
]

BENCHES = {name: (f"{name}_docs.jsonl", f"{name}_queries.jsonl", f"{name}_gold.jsonl")
           for name in BENCH_NAMES}


def make_cosine_rerank_fn(embed_fn):
    async def cosine_rerank(query, doc_texts):
        if not doc_texts:
            return []
        qe = (await embed_fn([query]))[0]
        des = await embed_fn(doc_texts)
        qn = float(np.linalg.norm(qe)) or 1e-9
        out = []
        for de in des:
            den = float(np.linalg.norm(de)) or 1e-9
            out.append(float(np.dot(qe, de) / (qn * den)))
        return out
    return cosine_rerank


async def evaluate_bench(bench_name, embed_fn, rerank_fn):
    """Returns {variant_name: {R@1, R@5, all_R@5, n}}."""
    docs_file, queries_file, gold_file = BENCHES[bench_name]
    try:
        docs_jsonl, queries, gold_rows = load_bench_jsonl(
            docs_file, queries_file, gold_file)
    except FileNotFoundError as e:
        return {"_error": str(e)}

    gold = {g["query_id"]: set(g["relevant_doc_ids"]) for g in gold_rows}
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"]) for d in docs_jsonl]

    extractor = TemporalExtractorV3_3()
    planner = QueryPlanner()
    retriever = TemporalRetriever(
        embed_fn=embed_fn,
        rerank_fn=rerank_fn,
        extractor=extractor,
        planner=planner,
    )
    await retriever.index(docs)

    # Per-variant counters
    variant_stats = {
        v: {"n_eval": 0, "n_r1": 0, "n_r5": 0, "all_r5": []}
        for v in VARIANTS
    }

    all_dids = list(retriever._doc_ref_us.keys())

    for q in queries:
        qid = q.get("query_id", "")
        gold_set = gold.get(qid, set())
        if not gold_set:
            continue

        # ===== Shared work =====
        plan = await planner.plan(q["text"], q["ref_time"])

        # Resolve each leaf's anchor (extractor on leaf phrase + utterance fallback)
        leaves_flat = [
            (ci, li, leaf)
            for ci, clause in enumerate(plan.expr)
            for li, leaf in enumerate(clause)
        ]
        anchors: dict[tuple[int, int], list[Interval]] = {}
        if leaves_flat:
            ref_dt = parse_iso(q["ref_time"])
            coros = [extractor.extract(leaf.phrase, ref_dt) for _, _, leaf in leaves_flat]
            results = await asyncio.gather(*coros)
            for (ci, li, _leaf), envs in zip(leaves_flat, results, strict=False):
                # Full-dir extractor returns TimeEnvelope; convert to Interval.
                anchors[(ci, li)] = flatten_intervals(envs)

        # Build filter sets
        valid_includes: list[tuple[str, list[Interval]]] = []
        valid_excludes: list[list[Interval]] = []
        for ci, li, leaf in leaves_flat:
            ivs = anchors.get((ci, li), [])
            if not ivs:
                continue
            if leaf.relation == "disjoint":
                valid_excludes.append(ivs)
            else:
                valid_includes.append((leaf.relation, ivs))

        eligible = [
            did for did in all_dids
            if doc_passes_filter(
                retriever._doc_ivs.get(did, []) or [],
                valid_includes, valid_excludes)
        ]

        # Hybrid pool (shared across variants)
        q_emb = (await embed_fn([q["text"]]))[0]
        q_emb = np.asarray(q_emb, dtype=np.float32)
        sem_scores: dict[str, float] = {}
        for did, demb in retriever._doc_emb.items():
            qn = float(np.linalg.norm(q_emb)) or 1e-9
            dn = float(np.linalg.norm(demb)) or 1e-9
            sem_scores[did] = float(np.dot(q_emb, demb) / (qn * dn))
        pool = build_pool(sem_scores, all_dids, eligible, pool_size=10)

        # Rerank over the pool (cross-encoder or cosine; here cosine)
        pool_texts = [retriever._docs[did].text for did in pool]
        rerank_raw = await rerank_fn(q["text"], pool_texts)
        rerank_partial = dict(zip(pool, rerank_raw, strict=False))
        rerank_norm = normalize_rerank_full(rerank_partial, pool, tail_score=0.0)
        base_norm = normalize_dict(rerank_norm)

        # Recency scores over pool (using doc bundles)
        recency_norm: dict[str, float] = {}
        if plan.latest_intent or plan.earliest_intent:
            direction = "latest" if plan.latest_intent else "earliest"
            doc_bundles = {
                did: [{"intervals": retriever._doc_ivs.get(did, [])}]
                for did in pool
            }
            doc_ref_us = {
                did: retriever._doc_ref_us.get(did, 0) for did in pool
            }
            recency_norm = recency_scores(doc_bundles, doc_ref_us, direction=direction)

        # Resolver for each variant's match evaluator
        def resolver(ci: int, li: int, _leaf) -> list[Interval]:
            return anchors.get((ci, li), [])

        # ===== Per-variant scoring =====
        for vname, vfn in VARIANTS.items():
            empty_score = EMPTY_DOC_MATCH.get(vname, 1.0)
            match_scores: dict[str, float] = {}
            for did in pool:
                doc_ivs = retriever._doc_ivs.get(did, []) or []
                if not doc_ivs:
                    match_scores[did] = empty_score
                else:
                    match_scores[did] = vfn(plan, doc_ivs, resolver)
            # Combine additively: pool_norm(base) + match + recency
            combined: dict[str, float] = {}
            for did in pool:
                s = base_norm.get(did, 0.0) + match_scores.get(did, 0.0)
                if recency_norm:
                    s += recency_norm.get(did, 0.0)
                combined[did] = s
            ranking = sorted(combined.keys(), key=lambda d: combined[d], reverse=True)

            variant_stats[vname]["n_eval"] += 1
            first = next((i + 1 for i, d in enumerate(ranking) if d in gold_set), None)
            if first is not None:
                if first <= 1:
                    variant_stats[vname]["n_r1"] += 1
                if first <= 5:
                    variant_stats[vname]["n_r5"] += 1
            top5 = set(ranking[:5])
            variant_stats[vname]["all_r5"].append(len(top5 & gold_set) / len(gold_set))

    return {
        v: {
            "R@1": s["n_r1"] / max(1, s["n_eval"]),
            "R@5": s["n_r5"] / max(1, s["n_eval"]),
            "all_R@5": sum(s["all_r5"]) / max(1, len(s["all_r5"])),
            "n_eval": s["n_eval"],
        }
        for v, s in variant_stats.items()
    }


async def main():
    print(f"Loading embed_fn... ({len(BENCHES)} benches, {len(VARIANTS)} variants)",
          flush=True)
    embed_fn = await make_embed_fn()
    rerank_fn = make_cosine_rerank_fn(embed_fn)

    all_results: dict[str, dict] = {}
    for bench in BENCHES:
        print(f"\n=== {bench} ===", flush=True)
        try:
            res = await evaluate_bench(bench, embed_fn, rerank_fn)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            all_results[bench] = {"_error": str(e)}
            continue
        if "_error" in res:
            print(f"  ERROR: {res['_error']}", flush=True)
            all_results[bench] = res
            continue
        all_results[bench] = res
        for v, m in res.items():
            print(f"  {v:14s}  R@1={m['R@1']:.3f}  R@5={m['R@5']:.3f}  "
                  f"all_R@5={m['all_R@5']:.3f}  n={m['n_eval']}", flush=True)

    # ===== Per-bench table + macro =====
    print("\n" + "=" * 120, flush=True)
    variants = list(VARIANTS.keys())
    macro = {v: {"R@1": [], "R@5": [], "all_R@5": []} for v in variants}

    # Per-bench R@1 table
    print(f"\n{'bench':28s}" + "".join(f"  {v[:11]:>11s}" for v in variants)
          + "    (R@1)", flush=True)
    print("-" * 120, flush=True)
    for bench in BENCHES:
        r = all_results.get(bench, {})
        if "_error" in r:
            continue
        row = f"{bench:28s}"
        for v in variants:
            row += f"  {r[v]['R@1']:>11.3f}"
            macro[v]["R@1"].append(r[v]["R@1"])
            macro[v]["R@5"].append(r[v]["R@5"])
            macro[v]["all_R@5"].append(r[v]["all_R@5"])
        print(row, flush=True)

    # Macro summary
    print("\n" + "=" * 120, flush=True)
    print("MACRO:", flush=True)
    print(f"{'variant':14s}  {'R@1':>8s}  {'R@5':>8s}  {'all_R@5':>8s}",
          flush=True)
    print("-" * 50, flush=True)
    base_r1 = sum(macro["baseline"]["R@1"]) / max(1, len(macro["baseline"]["R@1"]))
    base_r5 = sum(macro["baseline"]["R@5"]) / max(1, len(macro["baseline"]["R@5"]))
    base_ar5 = sum(macro["baseline"]["all_R@5"]) / max(1, len(macro["baseline"]["all_R@5"]))
    for v in variants:
        m1 = sum(macro[v]["R@1"]) / max(1, len(macro[v]["R@1"]))
        m5 = sum(macro[v]["R@5"]) / max(1, len(macro[v]["R@5"]))
        ma = sum(macro[v]["all_R@5"]) / max(1, len(macro[v]["all_R@5"]))
        d1 = m1 - base_r1
        d5 = m5 - base_r5
        da = ma - base_ar5
        suffix = ""
        if v != "baseline":
            suffix = f"   ΔR@1={d1:+.3f} ΔR@5={d5:+.3f} Δall={da:+.3f}"
        print(f"{v:14s}  {m1:>8.3f}  {m5:>8.3f}  {ma:>8.3f}{suffix}", flush=True)

    out_path = ROOT / "scoring_variants_validation.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
