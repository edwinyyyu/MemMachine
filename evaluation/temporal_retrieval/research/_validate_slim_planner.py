"""Minimal-viable planner prompt vs current 200-line production prompt.

Audit (project_temporal_and_or_dropped) showed plan-structure distribution:
- 73.4% single-leaf
- 19.2% empty (skip-rules fired)
- 6.4% 1-clause-2-leaf (composition rule "in X not in Y")
- 0.9% multi-clause OR
- 0.1% 1-clause-3-leaf

The production prompt is ~200 lines explaining the DNF shape, leaf
extraction rules (a)-(f), composition rule for "in X not in Y", a long
enumeration of TEMPORAL-LOOKING FRAMINGS THAT ARE NOT SCOPING, and ~30
worked examples. Each section serves a real query pattern.

Question: is the verbosity load-bearing, or could a principle-based
slim prompt match it? Slim prompt below distills to ~50 lines:
- DNF shape: 2 sentences
- Leaf extraction: principle "calendar-concrete date or anaphoric event"
- Skip rule: principle "topical references that NAME but don't NARROW"
- Composition: principle "relative period inside a year takes that year"
- Extremum: 2 sentences
- ~10 representative examples

Prediction: slim regresses most on `speculative_anchors` (whose 12
queries all rely on the long skip enumeration) and `timeless_policies`
(which depend on detecting non-scoping language).

Uses QueryPlanner(prompt_template=SLIM, cache_subdir="planner_slim")
to avoid invalidating the production cache.

Run from ``evaluation/``::

    uv run python -m temporal_retrieval.research._validate_slim_planner
"""
from __future__ import annotations

import asyncio
import json

import numpy as np

from temporal_retrieval import Doc
from temporal_retrieval.core import (
    build_pool,
    doc_passes_filter,
    flatten_intervals,
    normalize_dict,
    normalize_rerank_full,
    recency_scores,
)
from temporal_retrieval.extractor_v3_3 import TemporalExtractorV3_3
from temporal_retrieval.planner import QueryPlanner, evaluate_dnf_match
from temporal_retrieval.schema import parse_iso, to_us

from ._common import (
    ROOT,
    load_bench_jsonl,
    make_embed_fn,
    setup_env,
)

setup_env()


# Minimal-viable planner prompt — principle-based, ~50 lines.
# Designed in good faith: every section captures the equivalent
# instruction in the production prompt, just terser.
SLIM_PROMPT = """Extract temporal scoping constraints from a query as DNF.

Query: {query}
Reference time: {ref_time}

OUTPUT
======
Return JSON: {{"expr": [[<leaf>, ...], ...], "extremum": "latest"|"earliest"|null}}.
- Outer list = OR clauses; inner list = AND of leaves.
- Each leaf: {{"phrase": <text>, "relation": "intersect"|"after"|"before"|"disjoint"}}.
- expr=[] when the query has NO temporal scope (the time words name a topic, not a date narrowing).

LEAF PHRASE
===========
The phrase should be CALENDAR-CONCRETE (e.g., "Q4 2023", "March 2024",
"summer 2024") or an anaphoric event reference (e.g., "the launch").
- Direct date phrases: copy verbatim. Do NOT splice in a year from
  Reference time unless the query has a deictic cue ("last", "this",
  "next"). "March" alone stays bare.
- Event-anchor + offset: resolve in-place if you know the date
  ("four days after Election Day 2020" -> "November 7, 2020").

RELATION
========
- "intersect": the phrase NAMES the time of interest (default; covers
  "in", "during", deictic phrases, resolved event+offset)
- "after" / "before": directional, only when user wants open-ended search
- "disjoint": "not in", "outside", "excluding", "except"

SKIP PRINCIPLE
==============
Emit expr=[] when the temporal language names the TOPIC or FRAMING
rather than narrowing the answer's date scope. Test: does the phrasing
filter the answer BY date, or just name what the answer is ABOUT? If
only naming, skip. Examples that skip: "notes from the offsite",
"lessons of the launch", "aftermath of X", "when did X happen?",
"how did X go?", "the future of AI", "during the day in a beehive".

COMPOSITION
===========
When a bare period (season, month, quarter without year) appears WITH
another year-qualified leaf in the same clause, resolve against that
year:
  "in 2024 not in summer" -> [[{{"phrase":"2024","relation":"intersect"}},{{"phrase":"summer 2024","relation":"disjoint"}}]]

EXTREMUM
========
Set "latest" or "earliest" ONLY when the query asks to pick the most-
recent / oldest from MULTIPLE candidates. Cues: "latest", "most recent",
"earliest". NOT for "first X" / "last X" describing a specific event.

EXAMPLES
========
"in Q4 2023" -> {{"expr":[[{{"phrase":"Q4 2023","relation":"intersect"}}]],"extremum":null}}
"after 2020" -> {{"expr":[[{{"phrase":"2020","relation":"after"}}]],"extremum":null}}
"in March" -> {{"expr":[[{{"phrase":"March","relation":"intersect"}}]],"extremum":null}}
"in 2024 not in summer" -> {{"expr":[[{{"phrase":"2024","relation":"intersect"}},{{"phrase":"summer 2024","relation":"disjoint"}}]],"extremum":null}}
"in Q1 or Q4 of 2023" -> {{"expr":[[{{"phrase":"Q1 2023","relation":"intersect"}}],[{{"phrase":"Q4 2023","relation":"intersect"}}]],"extremum":null}}
"Most recent change since the redesign shipped" -> {{"expr":[[{{"phrase":"the redesign","relation":"after"}}]],"extremum":"latest"}}
"Notes from the team retreat" -> {{"expr":[],"extremum":null}}
"Lessons from the v3 launch" -> {{"expr":[],"extremum":null}}
"When did the v3 launch happen?" -> {{"expr":[],"extremum":null}}
"what did I do recently" -> {{"expr":[],"extremum":"latest"}}
"""


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
]
BENCHES = {n: (f"{n}_docs.jsonl", f"{n}_queries.jsonl", f"{n}_gold.jsonl")
           for n in BENCH_NAMES}

VARIANTS = ["production", "slim"]


def _intersect_min_norm(d_ivs, a_ivs) -> float:
    best = 0.0
    for di in d_ivs:
        d_w = di.latest_us - di.earliest_us
        if d_w <= 0:
            d_w = 1
        for ai in a_ivs:
            a_w = ai.latest_us - ai.earliest_us
            if a_w <= 0:
                a_w = 1
            lo = max(di.earliest_us, ai.earliest_us)
            hi = min(di.latest_us, ai.latest_us)
            inter = max(0, hi - lo)
            denom = min(d_w, a_w)
            f = min(1.0, inter / denom)
            if f > best:
                best = f
    return best


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


async def eval_with_planner(planner, bench, embed_fn, rerank_fn):
    docs_file, queries_file, gold_file = BENCHES[bench]
    try:
        docs_jsonl, queries, gold_rows = load_bench_jsonl(
            docs_file, queries_file, gold_file)
    except FileNotFoundError as e:
        return {"_error": str(e)}, {}
    gold = {g["query_id"]: set(g["relevant_doc_ids"]) for g in gold_rows}
    docs = [Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"])
            for d in docs_jsonl]

    extractor = TemporalExtractorV3_3()

    async def _extract(d):
        try:
            envs = await extractor.extract(d.text, parse_iso(d.ref_time))
        except Exception:
            envs = []
        return d.id, envs
    results = await asyncio.gather(*(_extract(d) for d in docs))
    doc_envs = {did: envs for did, envs in results}
    doc_ivs = {did: flatten_intervals(envs) for did, envs in doc_envs.items()}
    extractor.save_caches()

    embs = await embed_fn([d.text for d in docs])
    doc_emb = {d.id: np.asarray(e, dtype=np.float32)
               for d, e in zip(docs, embs, strict=False)}
    doc_ref_us = {d.id: to_us(parse_iso(d.ref_time)) for d in docs}
    docs_by_id = {d.id: d for d in docs}
    all_dids = list(doc_ref_us.keys())

    stats = {"n_eval": 0, "n_r1": 0, "n_r5": 0, "all_r5": []}
    plan_struct = {"1c1l": 0, "empty": 0, "1c_multi": 0, "multi_c": 0}

    for q in queries:
        qid = q.get("query_id", "")
        gold_set = gold.get(qid, set())
        if not gold_set:
            continue

        plan = await planner.plan(q["text"], q["ref_time"])

        nclauses = len(plan.expr)
        nleaves = sum(len(c) for c in plan.expr)
        if nclauses == 0:
            plan_struct["empty"] += 1
        elif nclauses == 1 and nleaves == 1:
            plan_struct["1c1l"] += 1
        elif nclauses == 1:
            plan_struct["1c_multi"] += 1
        else:
            plan_struct["multi_c"] += 1

        leaves_flat = [(ci, li, leaf)
                       for ci, clause in enumerate(plan.expr)
                       for li, leaf in enumerate(clause)]

        anchors = {}
        if leaves_flat:
            ref_dt = parse_iso(q["ref_time"])
            res = await asyncio.gather(
                *(extractor.extract(leaf.phrase, ref_dt)
                  for _, _, leaf in leaves_flat))
            for (ci, li, _l), envs in zip(leaves_flat, res, strict=False):
                anchors[(ci, li)] = flatten_intervals(envs)

        valid_includes = []
        valid_excludes = []
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
            if doc_passes_filter(doc_ivs.get(did, []), valid_includes,
                                 valid_excludes)
        ]
        q_emb = np.asarray((await embed_fn([q["text"]]))[0], dtype=np.float32)
        sem_scores = {}
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

        recency_norm = {}
        if plan.latest_intent or plan.earliest_intent:
            direction = "latest" if plan.latest_intent else "earliest"
            bundles = {did: [{"intervals": doc_ivs.get(did, [])}] for did in pool}
            refs_us = {did: doc_ref_us.get(did, 0) for did in pool}
            recency_norm = recency_scores(bundles, refs_us, direction=direction)

        def resolver(ci, li, _leaf):
            return anchors.get((ci, li), [])

        match_scores = {}
        for did in pool:
            d_ivs = doc_ivs.get(did, [])
            if not d_ivs and plan.expr:
                match_scores[did] = 1.0
            elif not plan.expr:
                match_scores[did] = 1.0
            else:
                match_scores[did] = evaluate_dnf_match(
                    plan, d_ivs, resolver, intersect_leaf="min_norm")

        combined = {}
        for did in pool:
            s = base_norm.get(did, 0.0) + match_scores.get(did, 0.0)
            if recency_norm:
                s += recency_norm.get(did, 0.0)
            combined[did] = s
        ranking = sorted(combined.keys(), key=lambda d: combined[d], reverse=True)

        stats["n_eval"] += 1
        first = next((i + 1 for i, d in enumerate(ranking) if d in gold_set), None)
        if first is not None:
            if first <= 1:
                stats["n_r1"] += 1
            if first <= 5:
                stats["n_r5"] += 1
        top5 = set(ranking[:5])
        stats["all_r5"].append(len(top5 & gold_set) / len(gold_set))

    return {
        "R@1": stats["n_r1"] / max(1, stats["n_eval"]),
        "R@5": stats["n_r5"] / max(1, stats["n_eval"]),
        "all_R@5": sum(stats["all_r5"]) / max(1, len(stats["all_r5"])),
        "n_eval": stats["n_eval"],
    }, plan_struct


async def main():
    print(f"Loading embed_fn... ({len(BENCHES)} benches, 2 planner variants)",
          flush=True)
    embed_fn = await make_embed_fn()
    rerank_fn = make_cosine_rerank_fn(embed_fn)

    prod_planner = QueryPlanner()  # production prompt + production cache
    slim_planner = QueryPlanner(prompt_template=SLIM_PROMPT,
                                cache_subdir="planner_slim")

    all_results = {}
    all_structs = {"production": {"1c1l": 0, "empty": 0, "1c_multi": 0, "multi_c": 0},
                   "slim":       {"1c1l": 0, "empty": 0, "1c_multi": 0, "multi_c": 0}}

    for bench in BENCHES:
        print(f"\n=== {bench} ===", flush=True)
        try:
            prod_res, prod_struct = await eval_with_planner(
                prod_planner, bench, embed_fn, rerank_fn)
            slim_res, slim_struct = await eval_with_planner(
                slim_planner, bench, embed_fn, rerank_fn)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            continue
        if "_error" in prod_res or "_error" in slim_res:
            print(f"  ERROR: {prod_res.get('_error')} / {slim_res.get('_error')}",
                  flush=True)
            continue
        all_results[bench] = {"production": prod_res, "slim": slim_res,
                              "prod_struct": prod_struct, "slim_struct": slim_struct}
        for k in prod_struct:
            all_structs["production"][k] += prod_struct[k]
            all_structs["slim"][k] += slim_struct[k]
        d_r1 = slim_res["R@1"] - prod_res["R@1"]
        d_r5 = slim_res["R@5"] - prod_res["R@5"]
        print(f"  production  R@1={prod_res['R@1']:.3f}  R@5={prod_res['R@5']:.3f}  "
              f"all={prod_res['all_R@5']:.3f}", flush=True)
        print(f"  slim        R@1={slim_res['R@1']:.3f}  R@5={slim_res['R@5']:.3f}  "
              f"all={slim_res['all_R@5']:.3f}  Δr1={d_r1:+.3f}  Δr5={d_r5:+.3f}",
              flush=True)
        if prod_struct != slim_struct:
            print(f"  STRUCT prod={prod_struct} slim={slim_struct}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("PLAN-STRUCTURE TOTALS:", flush=True)
    print(f"  production: {all_structs['production']}", flush=True)
    print(f"  slim:       {all_structs['slim']}", flush=True)

    print("\nMACRO (35 benches, frac_min intersect):", flush=True)
    macro = {v: {"R@1": [], "R@5": [], "all_R@5": []} for v in VARIANTS}
    for bench, r in all_results.items():
        for k in ("R@1", "R@5", "all_R@5"):
            for v in VARIANTS:
                macro[v][k].append(r[v][k])
    print(f"{'variant':12s} {'R@1':>8s} {'R@5':>8s} {'all_R@5':>10s} {'Δr1':>10s} {'Δr5':>10s}",
          flush=True)
    p1 = sum(macro["production"]["R@1"]) / max(1, len(macro["production"]["R@1"]))
    p5 = sum(macro["production"]["R@5"]) / max(1, len(macro["production"]["R@5"]))
    for v in VARIANTS:
        m1 = sum(macro[v]["R@1"]) / max(1, len(macro[v]["R@1"]))
        m5 = sum(macro[v]["R@5"]) / max(1, len(macro[v]["R@5"]))
        ma = sum(macro[v]["all_R@5"]) / max(1, len(macro[v]["all_R@5"]))
        d1 = m1 - p1
        d5 = m5 - p5
        print(f"{v:12s} {m1:>8.3f} {m5:>8.3f} {ma:>10.3f} {d1:>+10.3f} {d5:>+10.3f}",
              flush=True)

    # Where did plan-structure diverge?
    print("\n" + "=" * 70, flush=True)
    print("Benches where plan-structure diverged:", flush=True)
    for bench, r in all_results.items():
        if r["prod_struct"] != r["slim_struct"]:
            print(f"  {bench}: prod={r['prod_struct']}  slim={r['slim_struct']}",
                  flush=True)

    out = ROOT / "slim_planner_validation.json"
    with open(out, "w") as f:
        json.dump({
            "results": all_results,
            "macro": {v: {k: sum(macro[v][k]) / max(1, len(macro[v][k]))
                          for k in ("R@1", "R@5", "all_R@5")}
                      for v in VARIANTS},
            "structs": all_structs,
        }, f, indent=2, default=str)
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
