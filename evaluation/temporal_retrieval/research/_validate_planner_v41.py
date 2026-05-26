"""A/B test the v4.1 planner prompt (tightened against speculative anchors)
vs the production v4.0 prompt.

Tests across:
- speculative_anchors: new reproducer bench targeting Mode 4 (queries with
  framing words like "from", "aftermath", "look back" that the v4.0
  planner can misread as temporal scopes).
- 7 existing benches: regression check.

Both planners share the same QueryPlanner code; only the PLAN_PROMPT
template differs. Each variant uses an isolated cache subdir to avoid
cache contamination.

Run from ``evaluation/``::

    uv run python -m temporal_retrieval.research._validate_planner_v41
"""

from __future__ import annotations

import asyncio
import json

from temporal_retrieval import Doc, TemporalRetriever
from temporal_retrieval.planner import PLAN_PROMPT, QueryPlanner

from ._common import (
    ROOT,
    load_bench_jsonl,
    make_embed_fn,
    make_rerank_fn,
    setup_env,
)

setup_env()

# Reconstruct the production v4.0 PLAN_PROMPT by stripping the v4.1
# additions (the new TEMPORAL-LOOKING FRAMINGS gate + new examples).
# v4.0 had no "TEMPORAL-LOOKING FRAMINGS" block in rule (e) and 7 fewer
# example queries.
V41_FRAMING_GATE = """\

        TEMPORAL-LOOKING FRAMINGS THAT ARE NOT SCOPING (skip → expr=[]):
        These words look temporal but only NAME a topic or PROVENANCE; the
        user is asking ABOUT the event, not for content scoped relative to it.
        - "from": "notes from the offsite", "lessons from the launch" —
          provenance/topic, not scope. → expr=[]
        - "of" (when the head is "aftermath", "outcomes", "lessons",
          "story of", "review of", "recap of", "wake of"): topical. → expr=[]
          "aftermath of the launch", "lessons of the migration"
        - "look back at", "looking back at", "thinking back to":
          narrative framing, not scope. → expr=[]
        - "behind" / "story behind": topical. → expr=[]
        - "when did X happen?", "when was X?": user wants the DATE OF X;
          retrieving docs about X is the answer, not filtering BY X. → expr=[]
        - "how did X go?", "what was X like?": narrative, not scope. → expr=[]

        The test: does this phrasing NARROW the time window of the answer,
        or just NAME what the answer is about? If it only names the topic,
        emit expr=[]."""

V41_NEW_EXAMPLES = """

Query: "Notes from the team retreat"
{{"expr":[],"extremum":null}}

Query: "Lessons from the v3 launch"
{{"expr":[],"extremum":null}}

Query: "Aftermath of the v3 launch"
{{"expr":[],"extremum":null}}

Query: "Look back at the regression"
{{"expr":[],"extremum":null}}

Query: "When did the v3 launch happen?"
{{"expr":[],"extremum":null}}

Query: "How did the migration go?"
{{"expr":[],"extremum":null}}

Query: "Recent migration plan"
{{"expr":[],"extremum":"latest"}}"""

# Reverse-engineer the v4.0 prompt by stripping v4.1 additions.
PROMPT_V41 = PLAN_PROMPT  # current code reflects v4.1
PROMPT_V40 = PLAN_PROMPT.replace(V41_FRAMING_GATE, "").replace(V41_NEW_EXAMPLES, "")
assert "TEMPORAL-LOOKING FRAMINGS" not in PROMPT_V40, (
    "v4.0 reconstruction failed — framing gate not stripped"
)
assert "Notes from the team retreat" not in PROMPT_V40, (
    "v4.0 reconstruction failed — new examples not stripped"
)
assert "TEMPORAL-LOOKING FRAMINGS" in PROMPT_V41, "v4.1 prompt missing framing gate"


BENCHES = {
    "speculative_anchors": (
        "speculative_anchors_docs.jsonl",
        "speculative_anchors_queries.jsonl",
        "speculative_anchors_gold.jsonl",
    ),
    "timeless_policies": (
        "timeless_policies_docs.jsonl",
        "timeless_policies_queries.jsonl",
        "timeless_policies_gold.jsonl",
    ),
    "precedents": (
        "precedents_docs.jsonl",
        "precedents_queries.jsonl",
        "precedents_gold.jsonl",
    ),
    "composition": (
        "composition_docs.jsonl",
        "composition_queries.jsonl",
        "composition_gold.jsonl",
    ),
    "adversarial": (
        "adversarial_docs.jsonl",
        "adversarial_queries.jsonl",
        "adversarial_gold.jsonl",
    ),
    "realq_v2": (
        "realq_v2_docs.jsonl",
        "realq_v2_queries.jsonl",
        "realq_v2_gold.jsonl",
    ),
    "hard_bench": (
        "hard_bench_docs.jsonl",
        "hard_bench_queries.jsonl",
        "hard_bench_gold.jsonl",
    ),
    "ambiguous_year": (
        "ambiguous_year_docs.jsonl",
        "ambiguous_year_queries.jsonl",
        "ambiguous_year_gold.jsonl",
    ),
}


VARIANTS = {
    "v40": {"prompt_template": PROMPT_V40, "cache_subdir": "planner_v40_test"},
    "v41": {"prompt_template": PROMPT_V41, "cache_subdir": "planner_v41_test"},
}


async def evaluate(
    bench: str,
    variant_name: str,
    variant_kw: dict,
    embed_fn,
    rerank_fn,
) -> dict:
    docs_file, queries_file, gold_file = BENCHES[bench]
    docs_jsonl, queries, gold_rows = load_bench_jsonl(
        docs_file, queries_file, gold_file
    )
    gold = {g["query_id"]: set(g["relevant_doc_ids"]) for g in gold_rows}

    docs = [
        Doc(id=d["doc_id"], text=d["text"], ref_time=d["ref_time"]) for d in docs_jsonl
    ]

    planner = QueryPlanner(
        prompt_template=variant_kw["prompt_template"],
        cache_subdir=variant_kw["cache_subdir"],
    )
    retriever = TemporalRetriever(
        embed_fn=embed_fn,
        rerank_fn=rerank_fn,
        planner=planner,
    )
    await retriever.index(docs)

    n_eval = 0
    n_r5 = 0
    n_r1 = 0
    all_recall_at_5: list[float] = []
    n_speculative = 0  # plans with non-empty expr
    for q in queries:
        qid = q.get("query_id", "")
        gold_set = gold.get(qid, set())
        if not gold_set:
            continue
        n_eval += 1
        plan = await planner.plan(q["text"], q["ref_time"])
        if plan.expr:
            n_speculative += 1
        results = await retriever.query(q["text"], q["ref_time"], k=10)
        ranking = [r.doc_id for r in results]
        first_gold = next((i + 1 for i, d in enumerate(ranking) if d in gold_set), None)
        if first_gold is not None:
            if first_gold <= 1:
                n_r1 += 1
            if first_gold <= 5:
                n_r5 += 1
        top5 = set(ranking[:5])
        all_recall_at_5.append(len(top5 & gold_set) / len(gold_set))

    return {
        "bench": bench,
        "variant": variant_name,
        "R@1": n_r1 / max(1, n_eval),
        "R@5": n_r5 / max(1, n_eval),
        "all_recall@5": sum(all_recall_at_5) / max(1, len(all_recall_at_5)),
        "n_eval": n_eval,
        "n_with_temporal_plan": n_speculative,
    }


async def main() -> None:
    print("Loading embed_fn + rerank_fn...", flush=True)
    embed_fn = await make_embed_fn()
    rerank_fn = await make_rerank_fn()

    rows: list[dict] = []
    for bench in BENCHES:
        print(f"\n=== bench: {bench} ===", flush=True)
        for vname, vkw in VARIANTS.items():
            print(f"  {vname}...", flush=True)
            r = await evaluate(bench, vname, vkw, embed_fn, rerank_fn)
            print(
                f"    R@1={r['R@1']:.3f} R@5={r['R@5']:.3f} "
                f"all_R@5={r['all_recall@5']:.3f} n_with_plan={r['n_with_temporal_plan']}/{r['n_eval']}",
                flush=True,
            )
            rows.append(r)

    print("\n" + "=" * 110, flush=True)
    print(
        f"{'bench':22s} {'variant':6s} {'R@1':>6s} {'R@5':>6s} "
        f"{'all_R@5':>8s} {'n_plan':>8s} {'delta_R@5':>10s} {'delta_R@1':>10s}",
        flush=True,
    )
    print("-" * 110, flush=True)
    for bench in BENCHES:
        baseline = next(
            r for r in rows if r["bench"] == bench and r["variant"] == "v40"
        )
        for vname in VARIANTS:
            r = next(
                row for row in rows if row["bench"] == bench and row["variant"] == vname
            )
            d5 = r["R@5"] - baseline["R@5"]
            d1 = r["R@1"] - baseline["R@1"]
            tag5 = "" if vname == "v40" else f"{d5:+.3f}"
            tag1 = "" if vname == "v40" else f"{d1:+.3f}"
            print(
                f"{bench:22s} {vname:6s} {r['R@1']:>6.3f} {r['R@5']:>6.3f} "
                f"{r['all_recall@5']:>8.3f} {r['n_with_temporal_plan']:>4d}/{r['n_eval']:<3d} "
                f"{tag5:>10s} {tag1:>10s}",
                flush=True,
            )
        print("-" * 110, flush=True)

    out_path = ROOT / "planner_v41_validation.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
