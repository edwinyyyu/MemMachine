"""Run world-state framing experiment.

Evaluates F1-F4 framings on:
- baseline scenarios (6 scenarios, 32 facts) -- reuses round 6's scenarios.json
- coreference-stress scenarios (2 scenarios, ~12 facts) -- scenarios_coreference.json

Per-framing metrics mirror round 6's grader (consistency on equivalence_groups,
entity_match on primary_entity). F3_state_change_multi additionally reports
multi-label behavior: were multi-label emissions appropriate?
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
R6_DIR = HERE.parent
EVAL_ROOT = HERE.parents[4]
load_dotenv(EVAL_ROOT / ".env")

sys.path.insert(0, str(R6_DIR))
sys.path.insert(0, str(HERE))

from framings import FRAMINGS
from strategies import (
    CallCounter,
    EmbedCache,
    LLMCache,
    RouteDecision,
    TopicState,
    embed,
)

SCENARIOS_BASELINE = R6_DIR / "scenarios.json"
SCENARIOS_COREF = HERE / "scenarios_coreference.json"
CACHE_DIR = HERE / "cache"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

LLM_CACHE_FILE = CACHE_DIR / "llm_cache.json"
EMBED_CACHE_FILE = CACHE_DIR / "embed_cache.json"

PRICE_LLM = 0.0025
PRICE_EMBED = 0.00002

BUDGET_HARD_CAP_LLM = 200
BUDGET_STOP_AT_LLM = int(BUDGET_HARD_CAP_LLM * 0.80)  # 160
BUDGET_HARD_CAP_USD = 1.00
BUDGET_STOP_AT_USD = 0.80


def check_budget(counter: CallCounter) -> None:
    cost = counter.llm_calls * PRICE_LLM + counter.embed_calls * PRICE_EMBED
    if counter.llm_calls >= BUDGET_STOP_AT_LLM:
        raise RuntimeError(
            f"Budget stop: {counter.llm_calls} LLM calls >= {BUDGET_STOP_AT_LLM} (80% of {BUDGET_HARD_CAP_LLM})"
        )
    if cost >= BUDGET_STOP_AT_USD:
        raise RuntimeError(f"Budget stop: ${cost:.2f} >= ${BUDGET_STOP_AT_USD}")


def entity_in_topic(entity: str, topic_name: str) -> bool:
    if not entity:
        return True
    tn = topic_name.lower()
    en = entity.lower()
    parts = [p.strip() for p in tn.split("/")]
    return en in parts or en in tn


def evaluate_scenario(
    framing_name: str,
    scenario: dict[str, Any],
    client: openai.OpenAI,
    llm_cache: LLMCache,
    embed_cache: EmbedCache,
    counter: CallCounter,
) -> dict[str, Any]:
    strat = FRAMINGS[framing_name]
    topics: list[TopicState] = []
    routings: list[dict[str, Any]] = []

    llm_before = counter.llm_calls
    embed_before = counter.embed_calls

    for turn in scenario["turns"]:
        for fact in turn.get("facts", []):
            check_budget(counter)
            decision: RouteDecision = strat(
                fact["text"], topics, client, llm_cache, embed_cache, counter
            )
            fact_emb = embed(client, embed_cache, counter, fact["text"])

            # Build the set of topics this fact routes to. Multi-label: register
            # under all emitted topics (updates multiple centroids).
            target_names: list[str] = [decision.topic_name]
            if decision.multi_topics:
                for tn in decision.multi_topics:
                    if tn not in target_names:
                        target_names.append(tn)

            for tn in target_names:
                ts = next((t for t in topics if t.name == tn), None)
                if ts is None:
                    ts = TopicState(name=tn, description=tn)
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
                    "multi_topics": decision.multi_topics,
                    "is_new": decision.is_new,
                    "reason": decision.reason,
                }
            )

    # --- Metrics ---
    # Consistency on multi-fact equivalence groups
    groups: dict[str, set[str]] = {}
    group_fact_counts: dict[str, int] = {}
    for r in routings:
        g = r["equivalence_group"]
        if g is None:
            continue
        groups.setdefault(g, set()).add(r["topic"])
        group_fact_counts[g] = group_fact_counts.get(g, 0) + 1
    multi_groups = {g: s for g, s in groups.items() if group_fact_counts[g] >= 2}
    consistent_multi = sum(1 for s in multi_groups.values() if len(s) == 1)
    consistency = consistent_multi / len(multi_groups) if multi_groups else 1.0

    # Entity match (soft) -- considers multi_topics as well (hit if ANY route contains the entity)
    entity_hits = 0
    entity_total = 0
    entity_hits_primary_only = 0
    for r in routings:
        ent = r["primary_entity"]
        if ent is None:
            continue
        entity_total += 1
        # Multi-label entity match: any emitted topic contains the entity
        any_topic_hits = entity_in_topic(ent, r["topic"])
        if r.get("multi_topics"):
            any_topic_hits = any_topic_hits or any(
                entity_in_topic(ent, t) for t in r["multi_topics"]
            )
        if any_topic_hits:
            entity_hits += 1
        if entity_in_topic(ent, r["topic"]):
            entity_hits_primary_only += 1

    # Multi-label stats
    multi_label_emissions = sum(1 for r in routings if r.get("multi_topics"))

    topic_count = len(topics)
    entries_per_topic = [t.entry_count for t in topics]
    mean_entries = statistics.mean(entries_per_topic) if entries_per_topic else 0.0
    stdev_entries = (
        statistics.pstdev(entries_per_topic) if len(entries_per_topic) > 1 else 0.0
    )
    balance_cv = (stdev_entries / mean_entries) if mean_entries else 0.0

    return {
        "scenario_id": scenario["id"],
        "routings": routings,
        "topics": [{"name": t.name, "entry_count": t.entry_count} for t in topics],
        "metrics": {
            "consistency": round(consistency, 3),
            "multi_groups_total": len(multi_groups),
            "multi_groups_consistent": consistent_multi,
            "entity_match": round(entity_hits / entity_total, 3)
            if entity_total
            else 1.0,
            "entity_match_primary_only": round(
                entity_hits_primary_only / entity_total, 3
            )
            if entity_total
            else 1.0,
            "entity_total": entity_total,
            "entity_hits": entity_hits,
            "multi_label_emissions": multi_label_emissions,
            "topic_count": topic_count,
            "total_facts": len(routings),
            "balance_cv": round(balance_cv, 3),
            "llm_calls": counter.llm_calls - llm_before,
            "embed_calls": counter.embed_calls - embed_before,
        },
    }


def aggregate(per_scenario: list[dict[str, Any]]) -> dict[str, Any]:
    cons_c = sum(s["metrics"]["multi_groups_consistent"] for s in per_scenario)
    cons_t = sum(s["metrics"]["multi_groups_total"] for s in per_scenario)
    ent_h = sum(s["metrics"]["entity_hits"] for s in per_scenario)
    ent_t = sum(s["metrics"]["entity_total"] for s in per_scenario)
    topics = sum(s["metrics"]["topic_count"] for s in per_scenario)
    facts = sum(s["metrics"]["total_facts"] for s in per_scenario)
    ml_emissions = sum(s["metrics"]["multi_label_emissions"] for s in per_scenario)
    llm_c = sum(s["metrics"]["llm_calls"] for s in per_scenario)
    emb_c = sum(s["metrics"]["embed_calls"] for s in per_scenario)
    return {
        "consistency": round(cons_c / cons_t, 3) if cons_t else 1.0,
        "consistent_groups": f"{cons_c}/{cons_t}",
        "entity_match": round(ent_h / ent_t, 3) if ent_t else 1.0,
        "entity_hits": f"{ent_h}/{ent_t}",
        "topics_total": topics,
        "facts_total": facts,
        "topics_per_fact": round(topics / facts, 2) if facts else 0.0,
        "multi_label_emissions": ml_emissions,
        "llm_calls": llm_c,
        "embed_calls": emb_c,
        "cost_usd": round(llm_c * PRICE_LLM + emb_c * PRICE_EMBED, 4),
    }


def run_framing_on_scenarios(
    framing_name: str,
    scenarios: list[dict[str, Any]],
    client: openai.OpenAI,
    llm_cache: LLMCache,
    embed_cache: EmbedCache,
    counter: CallCounter,
) -> dict[str, Any]:
    per_scenario = []
    for sc in scenarios:
        check_budget(counter)
        res = evaluate_scenario(
            framing_name, sc, client, llm_cache, embed_cache, counter
        )
        per_scenario.append(res)
        m = res["metrics"]
        print(
            f"  {sc['id']}: cons={m['multi_groups_consistent']}/{m['multi_groups_total']}"
            f" ent={m['entity_hits']}/{m['entity_total']}"
            f" topics={m['topic_count']}"
            f" ml={m['multi_label_emissions']}"
            f" llm={m['llm_calls']} emb={m['embed_calls']}",
            flush=True,
        )
    return {"per_scenario": per_scenario, "aggregate": aggregate(per_scenario)}


def main() -> None:
    with open(SCENARIOS_BASELINE) as f:
        baseline = json.load(f)["scenarios"]
    with open(SCENARIOS_COREF) as f:
        coref = json.load(f)["scenarios"]

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    llm_cache = LLMCache(LLM_CACHE_FILE)
    embed_cache = EmbedCache(EMBED_CACHE_FILE)
    counter = CallCounter()

    all_results: dict[str, dict[str, Any]] = {}
    t0 = time.time()

    try:
        for framing_name in FRAMINGS:
            print(f"\n=== {framing_name} :: BASELINE ===", flush=True)
            baseline_block = run_framing_on_scenarios(
                framing_name, baseline, client, llm_cache, embed_cache, counter
            )
            print(f"  AGG baseline: {baseline_block['aggregate']}", flush=True)

            print(f"\n=== {framing_name} :: COREFERENCE ===", flush=True)
            coref_block = run_framing_on_scenarios(
                framing_name, coref, client, llm_cache, embed_cache, counter
            )
            print(f"  AGG coreference: {coref_block['aggregate']}", flush=True)

            all_results[framing_name] = {
                "baseline": baseline_block,
                "coreference": coref_block,
            }
            # Save incrementally
            with open(RESULTS_DIR / "results.json", "w") as f:
                json.dump(all_results, f, indent=2)
            llm_cache.save()
            embed_cache.save()

            print(
                f"  RUNNING COUNTER: llm={counter.llm_calls} embed={counter.embed_calls} "
                f"cost=${counter.llm_calls * PRICE_LLM + counter.embed_calls * PRICE_EMBED:.4f}",
                flush=True,
            )
    finally:
        llm_cache.save()
        embed_cache.save()

    dt = time.time() - t0
    print(
        f"\nDone. Wall {dt:.1f}s. llm={counter.llm_calls} embed={counter.embed_calls}"
    )
    print(
        f"Cost ${counter.llm_calls * PRICE_LLM + counter.embed_calls * PRICE_EMBED:.4f}"
    )
    print(f"LLM cache stats: {llm_cache.stats()}")
    print(f"Embed cache stats: {embed_cache.stats()}")


if __name__ == "__main__":
    main()
