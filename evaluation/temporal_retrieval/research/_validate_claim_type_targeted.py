"""Targeted version of _validate_claim_type: only 7 key benches, 4 variants.

Run from ``evaluation/``::

    uv run python -m temporal_retrieval.research._validate_claim_type_targeted
"""
from __future__ import annotations

import asyncio
import json

import numpy as np

from temporal_retrieval import Doc
from temporal_retrieval.core import (
    Interval,
    build_pool,
    constraint_factor_for_doc,
    doc_passes_filter,
    excluded_containment,
    normalize_dict,
    normalize_rerank_full,
    recency_scores,
)
from temporal_retrieval.extractor_v3_4 import TemporalExtractorV3_4
from temporal_retrieval.planner import QueryPlanner, evaluate_dnf_match
from temporal_retrieval.schema import parse_iso, to_us

from ._common import (
    ROOT,
    load_bench_jsonl,
    make_embed_fn,
    setup_env,
)
from .scoring_variants import claim_type_scoring, frac_min_binab

setup_env()


# Subset: the benches that regressed under claim_type + key state-distinction benches
BENCH_NAMES = [
    "adversarial",       # after/before; regressed with proximity
    "allen",             # interval algebra; regressed
    "cotemporal",        # state-tagged docs
    "era",               # 10y-wide state docs
    "mixed_cue",         # mixed event/state docs
    "fractional_intent", # state-inversion bench (our target win)
    "realq",             # frac_min won big here previously
    "realq_v2",          # ditto
    "sensitivity_curated", # ditto
    "timeless_policies", # doc_frac tanked here
    "lattice",           # frac_min won here
    "edge_relative_time",# frac_min lost here
]
BENCHES = {n: (f"{n}_docs.jsonl", f"{n}_queries.jsonl", f"{n}_gold.jsonl")
           for n in BENCH_NAMES}


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


def _baseline_match(plan, doc_ivs, resolver) -> float:
    return evaluate_dnf_match(plan, doc_ivs, resolver)


async def evaluate_bench(bench, embed_fn, rerank_fn):
    docs_file, queries_file, gold_file = BENCHES[bench]
    try:
        docs_jsonl, queries, gold_rows = load_bench_jsonl(
            docs_file, queries_file, gold_file)
    except FileNotFoundError as e:
        return {"_error": str(e)}
    gold = {g["query_id"]: set(g["relevant_doc_ids"]) for g in gold_rows}
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]

    extractor = TemporalExtractorV3_4()
    planner = QueryPlanner()

    async def _extract(d):
        try:
            refs = await extractor.extract(d.text, parse_iso(d.ref_time))
        except Exception:
            refs = []
        return d.id, refs
    results = await asyncio.gather(*(_extract(d) for d in docs))
    doc_refs: dict[str, list[dict]] = {did: r for did, r in results}
    doc_ivs: dict[str, list[Interval]] = {
        did: [Interval(earliest_us=r["earliest_us"], latest_us=r["latest_us"])
              for r in rs]
        for did, rs in doc_refs.items()
    }
    extractor.save_caches()

    embs = await embed_fn([d.text for d in docs])
    doc_emb = {d.id: np.asarray(e, dtype=np.float32)
               for d, e in zip(docs, embs, strict=False)}
    doc_ref_us = {d.id: to_us(parse_iso(d.ref_time)) for d in docs}
    docs_by_id = {d.id: d for d in docs}

    variants_all = ("baseline", "ct_evbinary", "ct_evbin_binab",
                    "ct_evdocf_binab", "frac_min_binab")
    variant_stats = {v: {"n_eval": 0, "n_r1": 0, "n_r5": 0, "all_r5": []}
                     for v in variants_all}
    all_dids = list(doc_ref_us.keys())

    for q in queries:
        qid = q.get("query_id", "")
        gold_set = gold.get(qid, set())
        if not gold_set:
            continue

        plan = await planner.plan(q["text"], q["ref_time"])
        leaves_flat = [
            (ci, li, leaf)
            for ci, clause in enumerate(plan.expr)
            for li, leaf in enumerate(clause)
        ]

        anchors: dict[tuple[int, int], list[Interval]] = {}
        if leaves_flat:
            ref_dt = parse_iso(q["ref_time"])
            res = await asyncio.gather(
                *(extractor.extract(leaf.phrase, ref_dt) for _, _, leaf in leaves_flat)
            )
            for (ci, li, _l), refs in zip(leaves_flat, res, strict=False):
                anchors[(ci, li)] = [
                    Interval(earliest_us=r["earliest_us"],
                             latest_us=r["latest_us"])
                    for r in refs
                ]

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
            if doc_passes_filter(doc_ivs.get(did, []), valid_includes, valid_excludes)
        ]

        q_emb = np.asarray((await embed_fn([q["text"]]))[0], dtype=np.float32)
        sem_scores: dict[str, float] = {}
        qn = float(np.linalg.norm(q_emb)) or 1e-9
        for did, demb in doc_emb.items():
            dn = float(np.linalg.norm(demb)) or 1e-9
            sem_scores[did] = float(np.dot(q_emb, demb) / (qn * dn))
        pool = build_pool(sem_scores, all_dids, eligible, pool_size=10)

        pool_texts = [docs_by_id[did].text for did in pool]
        rerank_raw = await rerank_fn(q["text"], pool_texts)
        rerank_partial = dict(zip(pool, rerank_raw, strict=False))
        rerank_norm = normalize_rerank_full(rerank_partial, pool, tail_score=0.0)
        base_norm = normalize_dict(rerank_norm)

        recency_norm: dict[str, float] = {}
        if plan.latest_intent or plan.earliest_intent:
            direction = "latest" if plan.latest_intent else "earliest"
            bundles = {did: [{"intervals": doc_ivs.get(did, [])}] for did in pool}
            refs_us = {did: doc_ref_us.get(did, 0) for did in pool}
            recency_norm = recency_scores(bundles, refs_us, direction=direction)

        def resolver(ci, li, _leaf):
            return anchors.get((ci, li), [])

        for vname in variants_all:
            match_scores: dict[str, float] = {}
            for did in pool:
                d_refs = doc_refs.get(did, []) or []
                d_ivs = doc_ivs.get(did, []) or []
                if not d_refs:
                    # Use the production empty_doc_match=1.0 for ALL variants
                    # — fair comparison. Empty doc = "timeless"; full match.
                    # (Bug fix: previously gave 1.0 only to baseline, which
                    # tanked variants on benches with empty-ref gold docs
                    # like allen 13/18 or adversarial 6/40.)
                    match_scores[did] = 1.0
                else:
                    if vname == "baseline":
                        match_scores[did] = _baseline_match(plan, d_ivs, resolver)
                    elif vname == "ct_evbinary":
                        match_scores[did] = claim_type_scoring(
                            plan, d_refs, resolver, event_binary=True)
                    elif vname == "ct_evbin_binab":
                        match_scores[did] = claim_type_scoring(
                            plan, d_refs, resolver, event_binary=True,
                            binary_ab=True)
                    elif vname == "ct_evdocf_binab":
                        match_scores[did] = claim_type_scoring(
                            plan, d_refs, resolver, event_docfrac=True,
                            binary_ab=True)
                    else:
                        match_scores[did] = frac_min_binab(
                            plan, d_ivs, resolver)
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
    print(f"Loading embed_fn... ({len(BENCHES)} benches)", flush=True)
    embed_fn = await make_embed_fn()
    rerank_fn = make_cosine_rerank_fn(embed_fn)

    all_results: dict[str, dict] = {}
    for bench in BENCHES:
        print(f"\n=== {bench} ===", flush=True)
        try:
            res = await evaluate_bench(bench, embed_fn, rerank_fn)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            continue
        if "_error" in res:
            print(f"  ERROR: {res['_error']}", flush=True)
            continue
        all_results[bench] = res
        for v, m in res.items():
            print(f"  {v:16s}  R@1={m['R@1']:.3f}  R@5={m['R@5']:.3f}  "
                  f"all_R@5={m['all_R@5']:.3f}  n={m['n_eval']}", flush=True)

    variants_all = ("baseline", "ct_evbinary", "ct_evbin_binab",
                    "ct_evdocf_binab", "frac_min_binab")
    print("\n" + "=" * 130, flush=True)
    header = f"{'bench':24s}"
    for v in variants_all:
        header += f"  {v[:8]:>8s}"
    header += "  |"
    for v in variants_all[1:]:
        header += f"  Δ{v[:6]:>6s}"
    print(header, flush=True)
    print("-" * 130, flush=True)
    macro = {v: {"R@1": [], "R@5": [], "all_R@5": []} for v in variants_all}
    for bench in BENCHES:
        if bench not in all_results:
            continue
        r = all_results[bench]
        for k in ("R@1", "R@5", "all_R@5"):
            for v in macro:
                macro[v][k].append(r[v][k])
        row = f"{bench:24s}"
        for v in variants_all:
            row += f"  {r[v]['R@1']:>8.3f}"
        row += "  |"
        for v in variants_all[1:]:
            d = r[v]["R@1"] - r["baseline"]["R@1"]
            row += f"  {d:>+7.3f}"
        print(row, flush=True)

    print("\nMACRO (subset, not full bench):", flush=True)
    print(f"{'variant':18s} {'R@1':>8s} {'R@5':>8s} {'all_R@5':>10s}", flush=True)
    for v in variants_all:
        m1 = sum(macro[v]["R@1"]) / max(1, len(macro[v]["R@1"]))
        m5 = sum(macro[v]["R@5"]) / max(1, len(macro[v]["R@5"]))
        ma = sum(macro[v]["all_R@5"]) / max(1, len(macro[v]["all_R@5"]))
        print(f"{v:18s} {m1:>8.3f} {m5:>8.3f} {ma:>10.3f}", flush=True)

    out = ROOT / "claim_type_targeted_validation.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
