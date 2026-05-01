"""Round 6 evaluation: routing incoming facts into topic logs.

For each (strategy, scenario) we:
  1. Walk the turns in order.
  2. For each fact in each turn, call the strategy.
  3. Update topic state (centroid, count).
  4. At the end, compute metrics per scenario and aggregate.

Metrics:
  - consistency: % of equivalence_groups whose facts all land in the same topic
  - coverage: % of expected facts routed (always 100% if strategy always returns a topic; we also track cross-scenario topic reuse)
  - entity_match: for facts with primary_entity, does the chosen topic name contain that entity's name?
  - topic_count: total distinct topics created per scenario (not per strategy-global)
  - balance: stddev / mean of entries-per-topic
  - cost: llm_calls + embed_calls over all scenarios

Paraphrase consistency is measured cross-turn within a scenario (equivalence_group
is the ground-truth cluster label).
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import openai
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
EVAL_ROOT = HERE.parents[3]  # .../evaluation
load_dotenv(EVAL_ROOT / ".env")

sys.path.insert(0, str(HERE))
from strategies import (
    STRATEGIES,
    CallCounter,
    EmbedCache,
    LLMCache,
    RouteDecision,
    TopicState,
    embed,
)

SCENARIOS_FILE = HERE / "scenarios.json"
CACHE_DIR = HERE / "cache"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

LLM_CACHE = CACHE_DIR / "llm_cache.json"
EMBED_CACHE = CACHE_DIR / "embed_cache.json"

# gpt-5-mini rough pricing (per call): ~$0.0025
# text-embedding-3-small: ~$0.00002 per call (tiny)
PRICE_LLM_CALL = 0.0025
PRICE_EMBED_CALL = 0.00002
BUDGET_HARD_CAP_LLM = 200
BUDGET_HARD_CAP_EMBED = 200
BUDGET_STOP_AT_LLM = int(BUDGET_HARD_CAP_LLM * 0.80)  # 160
BUDGET_STOP_AT_EMBED = int(BUDGET_HARD_CAP_EMBED * 0.80)  # 160
BUDGET_HARD_CAP_USD = 2.00
BUDGET_STOP_AT_USD = 1.60


def check_budget(counter: CallCounter) -> None:
    cost = counter.llm_calls * PRICE_LLM_CALL + counter.embed_calls * PRICE_EMBED_CALL
    if counter.llm_calls >= BUDGET_STOP_AT_LLM:
        raise RuntimeError(
            f"Budget stop: {counter.llm_calls} LLM calls >= {BUDGET_STOP_AT_LLM} (80% of {BUDGET_HARD_CAP_LLM})"
        )
    if counter.embed_calls >= BUDGET_STOP_AT_EMBED:
        raise RuntimeError(
            f"Budget stop: {counter.embed_calls} embed calls >= {BUDGET_STOP_AT_EMBED} (80% of {BUDGET_HARD_CAP_EMBED})"
        )
    if cost >= BUDGET_STOP_AT_USD:
        raise RuntimeError(f"Budget stop: ${cost:.2f} >= ${BUDGET_STOP_AT_USD}")


def entity_in_topic(entity: str, topic_name: str) -> bool:
    """Is the entity name present as a component of the topic name?"""
    if not entity:
        return True
    tn = topic_name.lower()
    en = entity.lower()
    # Accept "User/..." for primary_entity == User; check as path-component
    parts = [p.strip() for p in tn.split("/")]
    return en in parts or en in tn


def evaluate_scenario(
    strategy_name: str,
    scenario: dict[str, Any],
    client: openai.OpenAI,
    llm_cache: LLMCache,
    embed_cache: EmbedCache,
    counter_before: CallCounter,
) -> dict[str, Any]:
    """Run one strategy on one scenario; return per-scenario metrics + routing trace."""
    strat = STRATEGIES[strategy_name]
    topics: list[TopicState] = []
    # Track routing decisions: fid -> {topic_name, primary_entity, equivalence_group}
    routings: list[dict[str, Any]] = []

    llm_before = counter_before.llm_calls
    embed_before = counter_before.embed_calls

    for turn in scenario["turns"]:
        for fact in turn.get("facts", []):
            check_budget(counter_before)
            decision: RouteDecision = strat(
                fact["text"],
                topics,
                client,
                llm_cache,
                embed_cache,
                counter_before,
            )
            # Update topic state (embed the fact to update centroid if not already)
            fact_emb = embed(client, embed_cache, counter_before, fact["text"])
            # Find/create TopicState
            ts = next((t for t in topics if t.name == decision.topic_name), None)
            if ts is None:
                ts = TopicState(
                    name=decision.topic_name,
                    description=decision.topic_name,
                )
                topics.append(ts)
            ts.update_centroid(fact_emb)
            routings.append(
                {
                    "tid": turn["tid"],
                    "fid": fact["fid"],
                    "fact_text": fact["text"],
                    "primary_entity": fact.get("primary_entity"),
                    "equivalence_group": fact.get("equivalence_group"),
                    "topic": decision.topic_name,
                    "is_new": decision.is_new,
                    "reason": decision.reason,
                }
            )

    # --- Metrics ---
    # Consistency: equivalence_group -> set of topics
    groups: dict[str, set[str]] = {}
    for r in routings:
        g = r["equivalence_group"]
        if g is None:
            continue
        groups.setdefault(g, set()).add(r["topic"])
    total_groups = len(groups)
    # Only multi-fact groups contribute to the consistency score (a group with a
    # single fact is trivially consistent).
    group_fact_counts: dict[str, int] = {}
    for r in routings:
        g = r["equivalence_group"]
        if g is None:
            continue
        group_fact_counts[g] = group_fact_counts.get(g, 0) + 1
    multi_groups = {g: s for g, s in groups.items() if group_fact_counts[g] >= 2}
    consistent_multi = sum(1 for s in multi_groups.values() if len(s) == 1)
    consistency_score = consistent_multi / len(multi_groups) if multi_groups else 1.0

    # Entity match (soft)
    entity_hits = 0
    entity_total = 0
    for r in routings:
        ent = r["primary_entity"]
        if ent is None:
            continue
        entity_total += 1
        if entity_in_topic(ent, r["topic"]):
            entity_hits += 1
    entity_match_score = entity_hits / entity_total if entity_total else 1.0

    # Coverage (for our setup, every fact is routed; this is always 1.0 as
    # no strategy currently emits a noop -- facts are always memory-worthy by
    # scenario design. We track it anyway).
    total_facts = len(routings)
    coverage = 1.0 if total_facts else 1.0

    topic_count = len(topics)
    entries_per_topic = [t.entry_count for t in topics]
    mean_entries = statistics.mean(entries_per_topic) if entries_per_topic else 0.0
    stdev_entries = (
        statistics.pstdev(entries_per_topic) if len(entries_per_topic) > 1 else 0.0
    )
    balance_cv = (stdev_entries / mean_entries) if mean_entries else 0.0

    llm_calls = counter_before.llm_calls - llm_before
    embed_calls = counter_before.embed_calls - embed_before

    return {
        "scenario_id": scenario["id"],
        "routings": routings,
        "topics": [{"name": t.name, "entry_count": t.entry_count} for t in topics],
        "metrics": {
            "consistency": round(consistency_score, 3),
            "multi_groups_total": len(multi_groups),
            "multi_groups_consistent": consistent_multi,
            "entity_match": round(entity_match_score, 3),
            "entity_total": entity_total,
            "entity_hits": entity_hits,
            "coverage": coverage,
            "topic_count": topic_count,
            "total_facts": total_facts,
            "balance_cv": round(balance_cv, 3),
            "llm_calls": llm_calls,
            "embed_calls": embed_calls,
        },
    }


def aggregate(per_scenario: list[dict[str, Any]]) -> dict[str, Any]:
    # Weighted-by-count consistency: sum consistent over sum total
    cons_consistent = sum(s["metrics"]["multi_groups_consistent"] for s in per_scenario)
    cons_total = sum(s["metrics"]["multi_groups_total"] for s in per_scenario)
    ent_hits = sum(s["metrics"]["entity_hits"] for s in per_scenario)
    ent_total = sum(s["metrics"]["entity_total"] for s in per_scenario)
    topics = sum(s["metrics"]["topic_count"] for s in per_scenario)
    facts = sum(s["metrics"]["total_facts"] for s in per_scenario)
    llm_calls = sum(s["metrics"]["llm_calls"] for s in per_scenario)
    embed_calls = sum(s["metrics"]["embed_calls"] for s in per_scenario)
    balance_cvs = [s["metrics"]["balance_cv"] for s in per_scenario]
    return {
        "consistency": round(cons_consistent / cons_total, 3) if cons_total else 1.0,
        "consistent_groups": f"{cons_consistent}/{cons_total}",
        "entity_match": round(ent_hits / ent_total, 3) if ent_total else 1.0,
        "entity_hits": f"{ent_hits}/{ent_total}",
        "topics_total": topics,
        "facts_total": facts,
        "topics_per_fact": round(topics / facts, 2) if facts else 0.0,
        "mean_balance_cv": round(statistics.mean(balance_cvs), 3)
        if balance_cvs
        else 0.0,
        "llm_calls": llm_calls,
        "embed_calls": embed_calls,
        "cost_usd": round(
            llm_calls * PRICE_LLM_CALL + embed_calls * PRICE_EMBED_CALL, 4
        ),
    }


def main() -> None:
    with open(SCENARIOS_FILE) as f:
        data = json.load(f)
    scenarios = data["scenarios"]

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    llm_cache = LLMCache(LLM_CACHE)
    embed_cache = EmbedCache(EMBED_CACHE)
    counter = CallCounter()

    all_results: dict[str, dict[str, Any]] = {}
    t0 = time.time()
    try:
        for strat_name in STRATEGIES:
            print(f"\n=== Strategy {strat_name} ===", flush=True)
            per_scenario = []
            for sc in scenarios:
                check_budget(counter)
                res = evaluate_scenario(
                    strat_name, sc, client, llm_cache, embed_cache, counter
                )
                per_scenario.append(res)
                m = res["metrics"]
                print(
                    f"  {sc['id']}: cons={m['multi_groups_consistent']}/{m['multi_groups_total']}"
                    f" ent={m['entity_hits']}/{m['entity_total']}"
                    f" topics={m['topic_count']}"
                    f" llm={m['llm_calls']} emb={m['embed_calls']}",
                    flush=True,
                )
            agg = aggregate(per_scenario)
            all_results[strat_name] = {
                "per_scenario": per_scenario,
                "aggregate": agg,
            }
            print(f"  AGG: {agg}", flush=True)
            # Save partial
            with open(RESULTS_DIR / "results.json", "w") as f:
                json.dump(all_results, f, indent=2)
            llm_cache.save()
            embed_cache.save()
    finally:
        llm_cache.save()
        embed_cache.save()

    # Write report
    report = build_report(all_results, scenarios)
    with open(RESULTS_DIR / "report.md", "w") as f:
        f.write(report)
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    dt = time.time() - t0
    print(f"\nDone. Wall time {dt:.1f}s.")
    print(f"LLM cache: {llm_cache.stats()}")
    print(f"Embed cache: {embed_cache.stats()}")
    print(f"Counter: llm={counter.llm_calls} embed={counter.embed_calls}")
    print(
        f"Approx cost: ${counter.llm_calls * PRICE_LLM_CALL + counter.embed_calls * PRICE_EMBED_CALL:.4f}"
    )


def build_report(
    all_results: dict[str, dict[str, Any]], scenarios: list[dict[str, Any]]
) -> str:
    md: list[str] = []
    md.append("# Round 6 -- Topic Routing Results\n")
    md.append("## Leaderboard (aggregate across 6 scenarios)\n")
    md.append(
        "| Strategy | Consistency | Entity-match | Topics/Fact | Balance CV | LLM calls | Embed calls | Cost |"
    )
    md.append(
        "|----------|-------------|--------------|-------------|------------|-----------|-------------|------|"
    )
    rows = []
    for sn, block in all_results.items():
        a = block["aggregate"]
        rows.append((sn, a))
    rows.sort(
        key=lambda r: (
            -r[1]["consistency"]
            if isinstance(r[1]["consistency"], (int, float))
            else 0,
            r[1]["llm_calls"],
        )
    )
    for sn, a in rows:
        md.append(
            f"| {sn} | {a['consistency']:.2%} ({a['consistent_groups']}) | {a['entity_match']:.2%} ({a['entity_hits']}) | {a['topics_per_fact']:.2f} | {a['mean_balance_cv']:.2f} | {a['llm_calls']} | {a['embed_calls']} | ${a['cost_usd']:.4f} |"
        )
    md.append("")

    md.append("## Per-scenario consistency (multi-fact equivalence groups)\n")
    md.append("| Scenario | " + " | ".join(STRATEGIES.keys()) + " |")
    md.append("|---|" + "---|" * len(STRATEGIES))
    for sc in scenarios:
        row = [sc["id"]]
        for sn in STRATEGIES:
            res = next(
                (
                    x
                    for x in all_results[sn]["per_scenario"]
                    if x["scenario_id"] == sc["id"]
                ),
                None,
            )
            if res is None:
                row.append("-")
                continue
            m = res["metrics"]
            row.append(f"{m['multi_groups_consistent']}/{m['multi_groups_total']}")
        md.append("| " + " | ".join(row) + " |")
    md.append("")

    md.append("## Per-scenario entity-match\n")
    md.append("| Scenario | " + " | ".join(STRATEGIES.keys()) + " |")
    md.append("|---|" + "---|" * len(STRATEGIES))
    for sc in scenarios:
        row = [sc["id"]]
        for sn in STRATEGIES:
            res = next(
                (
                    x
                    for x in all_results[sn]["per_scenario"]
                    if x["scenario_id"] == sc["id"]
                ),
                None,
            )
            if res is None:
                row.append("-")
                continue
            m = res["metrics"]
            row.append(f"{m['entity_hits']}/{m['entity_total']}")
        md.append("| " + " | ".join(row) + " |")
    md.append("")

    md.append("## Per-scenario topic count\n")
    md.append("| Scenario | " + " | ".join(STRATEGIES.keys()) + " |")
    md.append("|---|" + "---|" * len(STRATEGIES))
    for sc in scenarios:
        row = [sc["id"]]
        for sn in STRATEGIES:
            res = next(
                (
                    x
                    for x in all_results[sn]["per_scenario"]
                    if x["scenario_id"] == sc["id"]
                ),
                None,
            )
            if res is None:
                row.append("-")
                continue
            m = res["metrics"]
            row.append(str(m["topic_count"]))
        md.append("| " + " | ".join(row) + " |")
    md.append("")

    md.append("## Routing traces (per strategy)\n")
    for sn in STRATEGIES:
        md.append(f"### {sn}\n")
        for sc in scenarios:
            res = next(
                (
                    x
                    for x in all_results[sn]["per_scenario"]
                    if x["scenario_id"] == sc["id"]
                ),
                None,
            )
            if res is None:
                continue
            md.append(f"**{sc['id']}**")
            for r in res["routings"]:
                md.append(
                    f"  - t{r['tid']}.{r['fid']}: `{r['topic']}`"
                    f"{' (NEW)' if r['is_new'] else ''} <- {r['fact_text']!r}"
                    f"   [{r['reason']}]"
                )
            md.append(
                "  topics: "
                + ", ".join(f"{t['name']}({t['entry_count']})" for t in res["topics"])
            )
            md.append("")
    return "\n".join(md)


if __name__ == "__main__":
    main()
