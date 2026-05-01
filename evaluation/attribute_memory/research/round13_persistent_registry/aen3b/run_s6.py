"""Run S6 (long-delay descriptor recovery) on aen3b vs aen3 vs baseline."""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND13 = HERE.parent
ROUND12 = ROUND13.parent / "round12_entity_registry"
ROUND11 = ROUND13.parent / "round11_writer_stress"
ROUND7 = ROUND13.parent / "round7"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROUND13 / "architectures"))
sys.path.insert(0, str(ROUND13 / "scenarios"))
sys.path.insert(0, str(ROUND12 / "architectures"))
sys.path.insert(0, str(ROUND12 / "scenarios"))
sys.path.insert(0, str(ROUND11 / "architectures"))
sys.path.insert(0, str(ROUND7 / "experiments"))

import aen3_persistent  # noqa: E402
import aen3b_persistent  # noqa: E402
from _common import Budget, Cache  # noqa: E402
from long_delay import scenario_s6  # noqa: E402

# Re-use grading utilities from run_s3
from run_s3 import (  # noqa: E402
    baseline_coref,
    coref_log_decisions,
    grade_scenario,
)

CACHE_DIR = HERE / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_aen3b(scenario, cache, budget):
    pairs = [(t.idx, t.text) for t in scenario.turns]
    pre_llm = budget.llm_calls
    pre_emb = budget.embed_calls
    log, idx, reg, coref_log = aen3b_persistent.ingest_turns_with_registry(
        pairs,
        cache,
        budget,
        batch_size=5,
        rebuild_index_every=100,
        lru_size=20,
        top_k=5,
        run_writer=False,
    )
    cache.save()
    decisions = coref_log_decisions(coref_log)
    grade = grade_scenario(scenario, decisions)
    delta_llm = budget.llm_calls - pre_llm
    delta_emb = budget.embed_calls - pre_emb
    n_embed_searches = sum(
        1
        for decs in coref_log.values()
        for d in decs
        if getattr(d, "used_embedding_search", False)
    )
    return {
        "arch": "aen3b_persistent",
        "grade": grade,
        "log_size": len(log),
        "n_entities": len(reg.by_id),
        "n_embed_searches": n_embed_searches,
        "registry": aen3b_persistent.registry_snapshot(reg),
        "coref_decisions": {
            str(t): [asdict(d) for d in decs] for t, decs in coref_log.items()
        },
        "llm_calls_delta": delta_llm,
        "embed_calls_delta": delta_emb,
        "llm_per_turn": delta_llm / max(1, len(scenario.turns)),
        "embed_per_turn": delta_emb / max(1, len(scenario.turns)),
    }


def run_aen3(scenario, cache, budget):
    pairs = [(t.idx, t.text) for t in scenario.turns]
    pre_llm = budget.llm_calls
    pre_emb = budget.embed_calls
    log, idx, reg, coref_log = aen3_persistent.ingest_turns_with_registry(
        pairs,
        cache,
        budget,
        batch_size=5,
        rebuild_index_every=100,
        lru_size=20,
        top_k=5,
        run_writer=False,
    )
    cache.save()
    decisions = coref_log_decisions(coref_log)
    grade = grade_scenario(scenario, decisions)
    delta_llm = budget.llm_calls - pre_llm
    delta_emb = budget.embed_calls - pre_emb
    n_embed_searches = sum(
        1
        for decs in coref_log.values()
        for d in decs
        if getattr(d, "used_embedding_search", False)
    )
    return {
        "arch": "aen3_persistent",
        "grade": grade,
        "log_size": len(log),
        "n_entities": len(reg.by_id),
        "n_embed_searches": n_embed_searches,
        "registry": aen3_persistent.registry_snapshot(reg),
        "llm_calls_delta": delta_llm,
        "embed_calls_delta": delta_emb,
        "llm_per_turn": delta_llm / max(1, len(scenario.turns)),
        "embed_per_turn": delta_emb / max(1, len(scenario.turns)),
    }


def run_baseline(scenario):
    decisions = baseline_coref(scenario)
    grade = grade_scenario(scenario, decisions)
    return {"arch": "aen1_simple", "grade": grade}


def main() -> None:
    budget = Budget(max_llm=200, max_embed=100, stop_at_llm=199, stop_at_embed=99)
    s = scenario_s6()
    print(f"=== Scenario: {s.name} ({len(s.turns)} turns) ===")
    s_results: dict[str, dict] = {}

    aen3_cache_path = ROUND13 / "cache" / f"aen3_{s.name}.json"
    aen3b_cache_path = CACHE_DIR / f"aen3b_{s.name}.json"
    if aen3_cache_path.exists() and not aen3b_cache_path.exists():
        shutil.copy(aen3_cache_path, aen3b_cache_path)
        print(f"  warmed aen3b cache from {aen3_cache_path.name}")

    cache_a3b = Cache(aen3b_cache_path)
    print("  [aen3b_persistent] running...")
    s_results["aen3b_persistent"] = run_aen3b(s, cache_a3b, budget)
    g = s_results["aen3b_persistent"]["grade"]
    es = s_results["aen3b_persistent"]["n_embed_searches"]
    print(
        f"    acc={g['accuracy']:.2%} "
        f"named={g['named']['rate']:.2%} "
        f"pron={g['pronoun']['rate']:.2%} "
        f"desc={g['descriptor']['rate']:.2%} "
        f"({g['n_correct']}/{g['n_total']}) "
        f"LLM/turn={s_results['aen3b_persistent']['llm_per_turn']:.2f} "
        f"emb/turn={s_results['aen3b_persistent']['embed_per_turn']:.2f} "
        f"embed_searches={es}"
    )

    if aen3_cache_path.exists():
        cache_a3 = Cache(aen3_cache_path)
    else:
        cache_a3 = Cache(CACHE_DIR / f"aen3_{s.name}.json")
    print("  [aen3_persistent] running...")
    s_results["aen3_persistent"] = run_aen3(s, cache_a3, budget)
    g = s_results["aen3_persistent"]["grade"]
    es = s_results["aen3_persistent"]["n_embed_searches"]
    print(
        f"    acc={g['accuracy']:.2%} "
        f"named={g['named']['rate']:.2%} "
        f"pron={g['pronoun']['rate']:.2%} "
        f"desc={g['descriptor']['rate']:.2%} "
        f"({g['n_correct']}/{g['n_total']}) "
        f"LLM/turn={s_results['aen3_persistent']['llm_per_turn']:.2f} "
        f"emb/turn={s_results['aen3_persistent']['embed_per_turn']:.2f} "
        f"embed_searches={es}"
    )

    print("  [aen1_simple baseline] grading only...")
    s_results["aen1_simple"] = run_baseline(s)
    g = s_results["aen1_simple"]["grade"]
    print(
        f"    acc={g['accuracy']:.2%} "
        f"named={g['named']['rate']:.2%} "
        f"pron={g['pronoun']['rate']:.2%} "
        f"desc={g['descriptor']['rate']:.2%} "
        f"({g['n_correct']}/{g['n_total']})"
    )

    out = RESULTS_DIR / f"{s.name}.json"
    out.write_text(json.dumps(s_results, indent=2, default=str))
    print(f"\n  wrote {out}")
    print(
        f"  budget so far: LLM={budget.llm_calls} "
        f"embed={budget.embed_calls} cost=${budget.cost():.3f}"
    )


if __name__ == "__main__":
    main()
