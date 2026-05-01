"""End-to-end pipeline evaluation.

Runs the pipeline over 4 scenarios, issues the ground-truth queries, and
uses LLM-as-judge to score each rubric point PASS/FAIL.

Also runs a small ablation study on S1 across 3 trigger/batch policies.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pipeline import (  # type: ignore
    BatchPolicy,
    Budget,
    Event,
    PipelineConfig,
    SemanticMemoryPipeline,
    _extract_json,
    llm_call,
)

SCENARIOS_FILE = HERE / "scenarios" / "scenarios.json"
RESULTS_DIR = HERE / "results"
CACHE_DIR = HERE / "cache"

JUDGE_PROMPT_TMPL = """You are grading whether an answer satisfies a specific rubric point.

QUERY: __QUERY__

ANSWER:
__ANSWER__

RUBRIC POINT: __RUBRIC_POINT__

Does the answer satisfy this rubric point? Output JSON only, no markdown:
  {"pass": true, "reason": "<1-sentence justification>"}
  or
  {"pass": false, "reason": "<1-sentence justification>"}
"""


def _fill_judge(template: str, **kwargs: str) -> str:
    out = template
    for k, v in kwargs.items():
        out = out.replace(f"__{k.upper()}__", v)
    return out


def load_scenarios() -> list[dict[str, Any]]:
    return json.loads(SCENARIOS_FILE.read_text())["scenarios"]


def run_scenario(
    scenario: dict[str, Any],
    config: PipelineConfig | None,
    budget: Budget,
    cache_subdir: str,
) -> dict[str, Any]:
    pipeline = SemanticMemoryPipeline(
        cache_dir=CACHE_DIR / cache_subdir,
        budget=budget,
        config=config,
    )
    events = [Event(**t) for t in scenario["turns"]]
    ingest_stats = pipeline.ingest(events)

    # Support single-query and multi-query scenarios.
    if "queries" in scenario:
        queries = scenario["queries"]
    else:
        queries = [
            {"query": scenario["query"], "rubric_points": scenario["rubric_points"]}
        ]

    per_query_results = []
    for qblock in queries:
        q = qblock["query"]
        rubric = qblock["rubric_points"]
        qresult = pipeline.query(q)
        # Grade each rubric point.
        judges = []
        passes = 0
        for point in rubric:
            judge_prompt = _fill_judge(
                JUDGE_PROMPT_TMPL,
                query=q,
                answer=qresult["answer"],
                rubric_point=point,
            )
            judge_raw = llm_call(
                judge_prompt, pipeline.llm_cache, budget, reasoning_effort="low"
            )
            judge_obj = _extract_json(judge_raw)
            passed = (
                bool(judge_obj.get("pass")) if isinstance(judge_obj, dict) else False
            )
            if passed:
                passes += 1
            judges.append(
                {
                    "point": point,
                    "pass": passed,
                    "reason": (judge_obj or {}).get("reason", "")
                    if isinstance(judge_obj, dict)
                    else judge_raw[:200],
                }
            )
        per_query_results.append(
            {
                "query": q,
                "answer": qresult["answer"],
                "topics": qresult["topics"],
                "rubric_total": len(rubric),
                "rubric_passes": passes,
                "rubric_accuracy": passes / max(1, len(rubric)),
                "judges": judges,
            }
        )

    pipeline.llm_cache.save()
    pipeline.embed_cache.save()

    return {
        "scenario_id": scenario["id"],
        "description": scenario["description"],
        "ingest_stats": ingest_stats,
        "num_topics": len(pipeline.topics),
        "num_entries": len(pipeline.entries),
        "topics": {
            name: [
                {
                    "id": e.entry_id,
                    "text": e.text,
                    "rel": e.relation,
                    "refs": e.refs,
                    "consolidated": e.consolidated,
                }
                for e in log.entries
            ]
            for name, log in pipeline.topics.items()
        },
        "queries": per_query_results,
    }


def run_rollback_test(budget: Budget) -> dict[str, Any]:
    """Verify rollback primitive: ingest a fact then rollback the batch."""
    pipeline = SemanticMemoryPipeline(
        cache_dir=CACHE_DIR / "rollback",
        budget=budget,
    )
    events = [
        Event(ts="2026-04-23T10:00:00", source="user", text="My SSN is 123-45-6789."),
        Event(ts="2026-04-23T10:00:10", source="assistant", text="Noted."),
    ]
    stats = pipeline.ingest(events)
    n_entries_before = len(pipeline.entries)
    # Rollback the last batch.
    last_batch_id = pipeline.batches[-1]["batch_id"]
    removed = pipeline.rollback_batch(last_batch_id)
    n_entries_after = len(pipeline.entries)
    pipeline.llm_cache.save()
    pipeline.embed_cache.save()
    return {
        "ingest_stats": stats,
        "entries_before_rollback": n_entries_before,
        "entries_removed": removed,
        "entries_after_rollback": n_entries_after,
        "success": n_entries_after == 0,
    }


def run_ablation(scenario: dict[str, Any], budget: Budget) -> list[dict[str, Any]]:
    """Run S1 under 3 batch policies; measure extraction quality + cost."""
    variants = [
        (
            "per_turn",
            PipelineConfig(
                batch_policy=BatchPolicy(n_turns=1, silence_gap_minutes=1.0)
            ),
        ),
        (
            "per_5_turns",
            PipelineConfig(
                batch_policy=BatchPolicy(n_turns=5, silence_gap_minutes=30.0)
            ),
        ),
        (
            "silence_only",
            PipelineConfig(
                batch_policy=BatchPolicy(n_turns=1000, silence_gap_minutes=15.0)
            ),
        ),
    ]
    results = []
    for name, config in variants:
        before = budget.llm_calls
        r = run_scenario(scenario, config, budget, cache_subdir=f"ablation_{name}")
        after = budget.llm_calls
        r["variant"] = name
        r["llm_calls_used"] = after - before
        results.append(r)
    return results


def run_no_prefilter(scenario: dict[str, Any], budget: Budget) -> dict[str, Any]:
    """Run S1 without the salience pre-filter; contrast."""
    config = PipelineConfig(salience_prefilter=False)
    before = budget.llm_calls
    r = run_scenario(scenario, config, budget, cache_subdir="no_prefilter")
    after = budget.llm_calls
    r["variant"] = "no_prefilter"
    r["llm_calls_used"] = after - before
    return r


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    budget = Budget()
    scenarios = load_scenarios()

    print("Running 4 scenarios + rollback test + ablations.")
    print(
        f"Budget: {budget.max_llm} LLM / {budget.max_embed} embed. "
        f"Stop at {budget.stop_at_llm} / {budget.stop_at_embed}."
    )

    results: dict[str, Any] = {
        "scenarios": {},
        "rollback_test": None,
        "ablation_s1_batching": None,
        "no_prefilter_s1": None,
        "budget": None,
    }

    try:
        # Main scenarios with default config.
        for scen in scenarios:
            print(f"\n=== Scenario {scen['id']} ===")
            r = run_scenario(scen, None, budget, cache_subdir=f"main_{scen['id']}")
            results["scenarios"][scen["id"]] = r
            total_passes = sum(q["rubric_passes"] for q in r["queries"])
            total_points = sum(q["rubric_total"] for q in r["queries"])
            print(
                f"  topics={r['num_topics']} entries={r['num_entries']} "
                f"ingest_stats={r['ingest_stats']}"
            )
            print(
                f"  rubric: {total_passes}/{total_points} "
                f"({total_passes / max(1, total_points):.0%})"
            )
            print(
                f"  budget: {budget.llm_calls} LLM / {budget.embed_calls} embed "
                f"(~${budget.approx_cost():.2f})"
            )

        # Rollback test.
        print("\n=== Rollback primitive test ===")
        results["rollback_test"] = run_rollback_test(budget)
        print(f"  {results['rollback_test']}")

        # Ablation: batching policy on S1.
        print("\n=== Ablation: batching policy (S1) ===")
        results["ablation_s1_batching"] = run_ablation(scenarios[0], budget)
        for r in results["ablation_s1_batching"]:
            total_passes = sum(q["rubric_passes"] for q in r["queries"])
            total_points = sum(q["rubric_total"] for q in r["queries"])
            print(
                f"  {r['variant']}: calls={r['llm_calls_used']} "
                f"rubric={total_passes}/{total_points}"
            )

        # No-prefilter: S1 without the salience filter.
        print("\n=== Ablation: no salience pre-filter (S1) ===")
        results["no_prefilter_s1"] = run_no_prefilter(scenarios[0], budget)
        r = results["no_prefilter_s1"]
        total_passes = sum(q["rubric_passes"] for q in r["queries"])
        total_points = sum(q["rubric_total"] for q in r["queries"])
        print(
            f"  no_prefilter: calls={r['llm_calls_used']} "
            f"rubric={total_passes}/{total_points}"
        )

    finally:
        results["budget"] = {
            "llm_calls": budget.llm_calls,
            "embed_calls": budget.embed_calls,
            "approx_cost": budget.approx_cost(),
        }
        (RESULTS_DIR / "results.json").write_text(
            json.dumps(results, indent=2, default=str)
        )
        print(f"\nSaved {RESULTS_DIR / 'results.json'}")
        print(
            f"Final: {budget.llm_calls} LLM / {budget.embed_calls} embed "
            f"(~${budget.approx_cost():.2f})"
        )

    write_report(results)


def write_report(results: dict[str, Any]) -> None:
    md: list[str] = ["# Round 6 Semantic Memory Pipeline -- Results\n"]
    md.append("End-to-end pipeline with salience pre-filter, 5-turn/silence batching, ")
    md.append("LLM extraction into append-only logs with fuzzy topic routing, ")
    md.append("and background consolidation at 8 live entries.\n\n")
    md.append(
        f"Budget used: {results['budget']['llm_calls']} LLM + "
        f"{results['budget']['embed_calls']} embed "
        f"(~${results['budget']['approx_cost']:.2f})\n\n"
    )

    md.append("## Scenario accuracy\n")
    md.append("| Scenario | Topics | Entries | Extraction calls | Rubric |")
    md.append("|----------|--------|---------|------------------|--------|")
    for sid, r in results["scenarios"].items():
        total_passes = sum(q["rubric_passes"] for q in r["queries"])
        total_points = sum(q["rubric_total"] for q in r["queries"])
        md.append(
            f"| {sid} | {r['num_topics']} | {r['num_entries']} | "
            f"{r['ingest_stats']['num_extraction_calls']} | "
            f"{total_passes}/{total_points} "
            f"({total_passes / max(1, total_points):.0%}) |"
        )
    md.append("")

    # Detail per scenario
    for sid, r in results["scenarios"].items():
        md.append(f"\n### {sid}\n")
        md.append(f"- {r['description']}")
        md.append(f"- Ingest stats: `{r['ingest_stats']}`")
        md.append(f"- Topics: {list(r['topics'].keys())}")
        for q in r["queries"]:
            md.append(f"\n**Query**: {q['query']}\n")
            md.append(f"**Answer**: {q['answer']}\n")
            md.append(f"**Rubric**: {q['rubric_passes']}/{q['rubric_total']}")
            md.append("\n| Point | Pass | Reason |")
            md.append("|-------|------|--------|")
            for j in q["judges"]:
                reason = j["reason"].replace("|", r"\|")[:200]
                md.append(
                    f"| {j['point'][:80]} | {'Y' if j['pass'] else 'N'} | {reason} |"
                )
        md.append("")

    # Rollback
    md.append("\n## Rollback primitive\n")
    rb = results["rollback_test"]
    md.append(f"- success: {rb['success']}")
    md.append(f"- entries before: {rb['entries_before_rollback']}")
    md.append(f"- entries removed: {rb['entries_removed']}")
    md.append(f"- entries after: {rb['entries_after_rollback']}")

    # Ablations
    md.append("\n## Ablation: batching policy on S1\n")
    md.append("| Variant | LLM calls | Extraction calls | Rubric | Topics |")
    md.append("|---------|-----------|------------------|--------|--------|")
    for r in results["ablation_s1_batching"]:
        total_passes = sum(q["rubric_passes"] for q in r["queries"])
        total_points = sum(q["rubric_total"] for q in r["queries"])
        md.append(
            f"| {r['variant']} | {r['llm_calls_used']} | "
            f"{r['ingest_stats']['num_extraction_calls']} | "
            f"{total_passes}/{total_points} "
            f"({total_passes / max(1, total_points):.0%}) | "
            f"{r['num_topics']} |"
        )

    # No-prefilter
    md.append("\n## Ablation: no salience pre-filter on S1\n")
    r = results["no_prefilter_s1"]
    total_passes = sum(q["rubric_passes"] for q in r["queries"])
    total_points = sum(q["rubric_total"] for q in r["queries"])
    md.append(f"- LLM calls: {r['llm_calls_used']}")
    md.append(f"- Extraction calls: {r['ingest_stats']['num_extraction_calls']}")
    md.append(f"- Rubric: {total_passes}/{total_points}")
    md.append(f"- Topics: {r['num_topics']}")

    (RESULTS_DIR / "REPORT.md").write_text("\n".join(md))
    print(f"Saved {RESULTS_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
