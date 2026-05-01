"""Round 13: aen3_persistent vs aen2_registry vs aen1_simple baseline.

Reuses round 12's grading harness (align_ids_to_gt + grade_scenario).

Scenarios: S1-S5 (round 12) + S6 (long delay, this round).

Special focus: S3 descriptor accuracy (was 0% in aen2) and S6 (entirely new
descriptor-recovery scenario).
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND13 = HERE.parent
ROUND12 = ROUND13.parent / "round12_entity_registry"
ROUND11 = ROUND13.parent / "round11_writer_stress"
ROUND7 = ROUND13.parent / "round7"
sys.path.insert(0, str(ROUND13 / "architectures"))
sys.path.insert(0, str(ROUND13 / "scenarios"))
sys.path.insert(0, str(ROUND12 / "architectures"))
sys.path.insert(0, str(ROUND12 / "scenarios"))
sys.path.insert(0, str(ROUND11 / "architectures"))
sys.path.insert(0, str(ROUND7 / "experiments"))

import aen1_simple  # noqa: E402
import aen2_registry  # noqa: E402
import aen3_persistent  # noqa: E402
from _common import Budget, Cache  # noqa: E402
from coref_stress import Scenario
from coref_stress import all_scenarios as round12_scenarios  # noqa: E402
from long_delay import scenario_s6  # noqa: E402

CACHE_DIR = ROUND13 / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = ROUND13 / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Surface-form -> id decisions for grading
# ---------------------------------------------------------------------------


def baseline_coref(scenario: Scenario) -> dict[int, dict[str, str | None]]:
    """Surface-form == entity (aen1_simple status quo)."""
    decisions: dict[int, dict[str, str | None]] = {}
    for t in scenario.turns:
        per_turn: dict[str, str | None] = {}
        for m in t.mentions:
            if m.kind == "pronoun":
                per_turn[m.surface] = None
            else:
                per_turn[m.surface] = "surf::" + m.surface.lower().strip()
        decisions[t.idx] = per_turn
    return decisions


def coref_log_decisions(coref_log: dict) -> dict[int, dict[str, str | None]]:
    out: dict[int, dict[str, str | None]] = {}
    for tidx, decs in coref_log.items():
        per_turn: dict[str, str | None] = {}
        for d in decs:
            per_turn[d.surface] = d.entity_id
        out[tidx] = per_turn
    return out


# ---------------------------------------------------------------------------
# Alignment + grading (same as round 12)
# ---------------------------------------------------------------------------


def align_ids_to_gt(
    scenario: Scenario,
    decisions: dict[int, dict[str, str | None]],
) -> dict[str | None, str | None]:
    cooc: dict[str | None, Counter] = defaultdict(Counter)
    for t in scenario.turns:
        per_turn = decisions.get(t.idx, {})
        for m in t.mentions:
            arch_id = per_turn.get(m.surface)
            if arch_id is None or m.entity_id is None:
                continue
            cooc[arch_id][m.entity_id] += 1
    mapping: dict[str | None, str | None] = {}
    used_gt: set[str] = set()
    arch_ids_sorted = sorted(cooc.keys(), key=lambda a: -sum(cooc[a].values()))
    for arch_id in arch_ids_sorted:
        candidates = cooc[arch_id].most_common()
        chosen = None
        for gt, _ in candidates:
            if gt not in used_gt:
                chosen = gt
                break
        if chosen is None and candidates:
            chosen = candidates[0][0]
        mapping[arch_id] = chosen
        if chosen is not None:
            used_gt.add(chosen)
    return mapping


def grade_scenario(
    scenario: Scenario,
    decisions: dict[int, dict[str, str | None]],
) -> dict:
    mapping = align_ids_to_gt(scenario, decisions)
    n_total = n_correct = 0
    n_named_total = n_named_correct = 0
    n_pron_total = n_pron_correct = 0
    n_desc_total = n_desc_correct = 0
    failures: list[dict] = []
    for t in scenario.turns:
        per_turn = decisions.get(t.idx, {})
        for m in t.mentions:
            if m.entity_id is None:
                continue
            arch_id = per_turn.get(m.surface)
            mapped = mapping.get(arch_id) if arch_id else None
            ok = mapped == m.entity_id
            n_total += 1
            n_correct += int(ok)
            if m.kind == "named":
                n_named_total += 1
                n_named_correct += int(ok)
            elif m.kind == "pronoun":
                n_pron_total += 1
                n_pron_correct += int(ok)
            elif m.kind == "descriptor":
                n_desc_total += 1
                n_desc_correct += int(ok)
            if not ok:
                failures.append(
                    {
                        "turn_idx": t.idx,
                        "surface": m.surface,
                        "kind": m.kind,
                        "expected_gt": m.entity_id,
                        "arch_id": arch_id,
                        "mapped_gt": mapped,
                        "turn_text": t.text,
                        "turn_kind": t.kind,
                    }
                )
    return {
        "n_total": n_total,
        "n_correct": n_correct,
        "accuracy": (n_correct / n_total) if n_total else 0.0,
        "named": {
            "total": n_named_total,
            "correct": n_named_correct,
            "rate": (n_named_correct / n_named_total) if n_named_total else 0.0,
        },
        "pronoun": {
            "total": n_pron_total,
            "correct": n_pron_correct,
            "rate": (n_pron_correct / n_pron_total) if n_pron_total else 0.0,
        },
        "descriptor": {
            "total": n_desc_total,
            "correct": n_desc_correct,
            "rate": (n_desc_correct / n_desc_total) if n_desc_total else 0.0,
        },
        "id_mapping": {str(k): v for k, v in mapping.items()},
        "failures": failures,
    }


# ---------------------------------------------------------------------------
# Architectures under test
# ---------------------------------------------------------------------------


def run_aen3(
    scenario: Scenario, cache: Cache, budget: Budget, run_writer: bool = False
) -> dict:
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
        run_writer=run_writer,
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
        "coref_decisions": {
            str(t): [asdict(d) for d in decs] for t, decs in coref_log.items()
        },
        "llm_calls_delta": delta_llm,
        "embed_calls_delta": delta_emb,
        "llm_per_turn": delta_llm / max(1, len(scenario.turns)),
        "embed_per_turn": delta_emb / max(1, len(scenario.turns)),
    }


def run_aen2(scenario: Scenario, cache: Cache, budget: Budget) -> dict:
    pairs = [(t.idx, t.text) for t in scenario.turns]
    pre_llm = budget.llm_calls
    pre_emb = budget.embed_calls
    log, idx, reg, coref_log = aen2_registry.ingest_turns_with_registry(
        pairs,
        cache,
        budget,
        batch_size=5,
        rebuild_index_every=100,
        lru_size=20,
    )
    cache.save()
    decisions = coref_log_decisions(coref_log)
    grade = grade_scenario(scenario, decisions)
    delta_llm = budget.llm_calls - pre_llm
    delta_emb = budget.embed_calls - pre_emb
    return {
        "arch": "aen2_registry",
        "grade": grade,
        "log_size": len(log),
        "n_entities": len(reg.by_id),
        "llm_calls_delta": delta_llm,
        "embed_calls_delta": delta_emb,
        "llm_per_turn": delta_llm / max(1, len(scenario.turns)),
    }


def run_baseline(
    scenario: Scenario, cache: Cache, budget: Budget, run_writer: bool = False
) -> dict:
    pairs = [(t.idx, t.text) for t in scenario.turns]
    pre_llm = budget.llm_calls
    pre_emb = budget.embed_calls
    if run_writer:
        log, idx_obj = aen1_simple.ingest_turns(
            pairs,
            cache,
            budget,
            batch_size=5,
            rebuild_index_every=100,
        )
    else:
        log = []
    cache.save()
    decisions = baseline_coref(scenario)
    grade = grade_scenario(scenario, decisions)
    delta_llm = budget.llm_calls - pre_llm
    delta_emb = budget.embed_calls - pre_emb
    return {
        "arch": "aen1_simple",
        "grade": grade,
        "log_size": len(log),
        "llm_calls_delta": delta_llm,
        "embed_calls_delta": delta_emb,
        "llm_per_turn": delta_llm / max(1, len(scenario.turns)),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    budget = Budget(max_llm=500, max_embed=200, stop_at_llm=480, stop_at_embed=190)
    scenarios = round12_scenarios() + [scenario_s6()]
    summary: dict[str, dict] = {}

    # We RE-USE round 12's caches for aen2 by pointing to the same cache files.
    # And for aen3 we maintain our own cache files. Baseline reuses round 12's
    # cache too (still aen1_simple writer + scoring is deterministic).
    for s in scenarios:
        print(f"\n=== Scenario: {s.name} ({len(s.turns)} turns) ===")
        s_results: dict[str, dict] = {}

        # --- aen3 (this round) ---
        cache_a3 = Cache(CACHE_DIR / f"aen3_{s.name}.json")
        try:
            print("  [aen3_persistent] running...")
            s_results["aen3_persistent"] = run_aen3(
                s, cache_a3, budget, run_writer=False
            )
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
        except Exception as e:
            print(f"    AEN3 ERROR: {e}")
            import traceback

            traceback.print_exc()
            s_results["aen3_persistent"] = {"error": str(e)}

        # --- aen2 (round 12) ---
        # For round 12 scenarios we have warm caches in round12/cache; copy
        # those into our results by reading them. Simpler: just point Cache
        # at round 12's cache file directly so it reuses everything.
        round12_cache_path = ROUND12 / "cache" / f"reg_{s.name}.json"
        if round12_cache_path.exists():
            cache_a2 = Cache(round12_cache_path)
        else:
            cache_a2 = Cache(CACHE_DIR / f"aen2_{s.name}.json")
        try:
            print("  [aen2_registry] running...")
            s_results["aen2_registry"] = run_aen2(s, cache_a2, budget)
            g = s_results["aen2_registry"]["grade"]
            print(
                f"    acc={g['accuracy']:.2%} "
                f"named={g['named']['rate']:.2%} "
                f"pron={g['pronoun']['rate']:.2%} "
                f"desc={g['descriptor']['rate']:.2%} "
                f"({g['n_correct']}/{g['n_total']}) "
                f"LLM/turn={s_results['aen2_registry']['llm_per_turn']:.2f}"
            )
        except Exception as e:
            print(f"    AEN2 ERROR: {e}")
            import traceback

            traceback.print_exc()
            s_results["aen2_registry"] = {"error": str(e)}

        # --- aen1_simple baseline (no writer to keep cost low) ---
        cache_b = Cache(CACHE_DIR / f"base_{s.name}.json")
        try:
            print("  [aen1_simple baseline] grading only...")
            s_results["aen1_simple"] = run_baseline(
                s, cache_b, budget, run_writer=False
            )
            g = s_results["aen1_simple"]["grade"]
            print(
                f"    acc={g['accuracy']:.2%} "
                f"named={g['named']['rate']:.2%} "
                f"pron={g['pronoun']['rate']:.2%} "
                f"desc={g['descriptor']['rate']:.2%} "
                f"({g['n_correct']}/{g['n_total']})"
            )
        except Exception as e:
            print(f"    BASELINE ERROR: {e}")
            s_results["aen1_simple"] = {"error": str(e)}

        summary[s.name] = s_results
        out = RESULTS_DIR / f"{s.name}.json"
        out.write_text(json.dumps(s_results, indent=2, default=str))

        # Bail if we're approaching the budget cap.
        print(
            f"  budget so far: LLM={budget.llm_calls} "
            f"embed={budget.embed_calls} cost=${budget.cost():.3f}"
        )

    summary_path = RESULTS_DIR / "summary.json"
    overall = {
        "total_llm": budget.llm_calls,
        "total_embed": budget.embed_calls,
        "total_cost": budget.cost(),
        "per_scenario": {
            name: {
                arch: (
                    {
                        "accuracy": v.get("grade", {}).get("accuracy"),
                        "named": v.get("grade", {}).get("named", {}).get("rate"),
                        "named_total": v.get("grade", {}).get("named", {}).get("total"),
                        "pronoun": v.get("grade", {}).get("pronoun", {}).get("rate"),
                        "pronoun_total": v.get("grade", {})
                        .get("pronoun", {})
                        .get("total"),
                        "descriptor": v.get("grade", {})
                        .get("descriptor", {})
                        .get("rate"),
                        "descriptor_total": v.get("grade", {})
                        .get("descriptor", {})
                        .get("total"),
                        "llm_per_turn": v.get("llm_per_turn"),
                        "embed_per_turn": v.get("embed_per_turn"),
                        "n_embed_searches": v.get("n_embed_searches"),
                        "n_entities": v.get("n_entities"),
                        "n_total": v.get("grade", {}).get("n_total"),
                    }
                    if "grade" in v
                    else {"error": v.get("error")}
                )
                for arch, v in arches.items()
            }
            for name, arches in summary.items()
        },
    }
    summary_path.write_text(json.dumps(overall, indent=2, default=str))
    print(f"\n=== Wrote {summary_path} ===")
    print(
        f"Total LLM={budget.llm_calls} embed={budget.embed_calls} "
        f"cost=${budget.cost():.3f}"
    )


if __name__ == "__main__":
    main()
