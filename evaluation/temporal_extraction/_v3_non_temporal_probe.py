"""Non-temporal probe: run the planner on 50 atemporal queries and report
empty-plan rate, false-positive constraints, and per-category breakdown.

Goal: verify the architecture's by-construction safety claim (empty plan ->
identity-on-rerank) holds empirically. If the planner emits constraints on
truly atemporal queries, those constraints will apply a mask < 1.0 and may
regress ranking vs rerank_only.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections import Counter
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

from query_planner_v3 import QueryPlannerV3
from salience_eval import DATA_DIR


async def main():
    queries = [
        json.loads(l) for l in open(DATA_DIR / "non_temporal_probe_queries.jsonl")
    ]
    print(f"Loaded {len(queries)} non-temporal queries", flush=True)

    planner = QueryPlannerV3()
    plan_items = [(q["query_id"], q["text"], q["ref_time"]) for q in queries]
    plans = await planner.plan_many(plan_items)

    cat = {q["query_id"]: q["category"] for q in queries}
    qtext = {q["query_id"]: q["text"] for q in queries}

    rows = []
    for qid in [q["query_id"] for q in queries]:
        p = plans[qid]
        n_constraints = len(p.constraints)
        has_extremum = p.latest_intent or p.earliest_intent
        empty = n_constraints == 0 and not has_extremum
        rows.append(
            {
                "qid": qid,
                "category": cat[qid],
                "text": qtext[qid],
                "n_constraints": n_constraints,
                "constraint_phrases": [
                    f"{c.direction}:{c.phrase}" for c in p.constraints
                ],
                "extremum": "latest"
                if p.latest_intent
                else ("earliest" if p.earliest_intent else None),
                "empty": empty,
            }
        )

    # Stats
    n = len(rows)
    n_empty = sum(1 for r in rows if r["empty"])
    n_with_constraints = sum(1 for r in rows if r["n_constraints"] > 0)
    n_with_extremum = sum(1 for r in rows if r["extremum"] is not None)

    print(f"\n{'=' * 80}")
    print(f"Overall: {n_empty}/{n} = {n_empty / n:.2f} EMPTY plans (safe)")
    print(f"  {n_with_constraints} plans have >=1 constraint")
    print(f"  {n_with_extremum} plans have extremum")

    # Per category
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    print("\nPer-category empty-plan rate:")
    for c, rs in by_cat.items():
        ne = sum(1 for r in rs if r["empty"])
        print(f"  {c:18s}: {ne}/{len(rs)} = {ne / len(rs):.2f}")

    # Show all non-empty plans (false positives)
    print(f"\n{'=' * 80}")
    print("NON-EMPTY plans (potential false positives):")
    fp = [r for r in rows if not r["empty"]]
    if not fp:
        print("  (none — all plans empty, by-construction safety confirmed)")
    else:
        for r in fp:
            print(f"\n  [{r['category']}] {r['qid']}: {r['text']}")
            print(f"    constraints: {r['constraint_phrases']}")
            print(f"    extremum: {r['extremum']}")

    # Constraint phrase histogram
    if fp:
        phrases = Counter()
        for r in fp:
            for p in r["constraint_phrases"]:
                phrases[p] += 1
        print(f"\n{'=' * 80}")
        print("Top constraint phrases emitted:")
        for ph, c in phrases.most_common(20):
            print(f"  {c:3d}  {ph}")

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "T_non_temporal_probe.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "n": n,
                "n_empty": n_empty,
                "empty_rate": n_empty / n,
                "n_with_constraints": n_with_constraints,
                "n_with_extremum": n_with_extremum,
                "rows": rows,
                "planner_stats": planner.stats(),
            },
            f,
            indent=2,
        )
    print(f"\nWrote {json_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
