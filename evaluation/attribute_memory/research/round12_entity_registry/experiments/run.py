"""Run aen2_registry vs aen1_simple baseline on coref_stress S1-S5.

Per scenario, both architectures process the same turns. We grade:

  - Disambiguation accuracy: did the system place each surface mention into
    the correct ground-truth entity bucket?
  - Alias recognition: when an alias is added, do later mentions of the new
    surface form resolve to the same internal id?
  - Pronoun resolution accuracy: pronouns get resolved to correct entities?
  - LRU behavior: when an entity is evicted and re-mentioned, does the
    system reattach to the original id?
  - Cost per turn: extra LLM calls for coref vs baseline.

Mapping from architecture-internal ids -> ground-truth labels is done by a
majority-vote alignment: for each internal id, find the GT label that the
mentions assigned to it most often agree with; for the baseline architecture
which has no internal id, we cluster by surface alias under the assumption
that "same surface" = "same entity" (which IS the baseline's behavior).
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND12 = HERE.parent
ROUND11 = ROUND12.parent / "round11_writer_stress"
ROUND7 = ROUND12.parent / "round7"
sys.path.insert(0, str(ROUND12 / "architectures"))
sys.path.insert(0, str(ROUND12 / "scenarios"))
sys.path.insert(0, str(ROUND11 / "architectures"))
sys.path.insert(0, str(ROUND7 / "experiments"))

import aen1_simple  # noqa: E402
import aen2_registry  # noqa: E402
from _common import Budget, Cache  # noqa: E402
from coref_stress import Scenario, all_scenarios  # noqa: E402

CACHE_DIR = ROUND12 / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = ROUND12 / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Baseline coref: surface-form == entity (the aen1_simple status quo).
# ---------------------------------------------------------------------------


def baseline_coref(scenario: Scenario) -> dict[int, dict[str, str]]:
    """For each turn, return surface -> assigned-internal-id.

    Baseline: every distinct surface form is its own entity id. Pronouns
    are NOT resolved (we mark them as None — baseline can't bind them).
    """
    decisions: dict[int, dict[str, str | None]] = {}
    for t in scenario.turns:
        per_turn: dict[str, str | None] = {}
        for m in t.mentions:
            if m.kind == "pronoun":
                per_turn[m.surface] = None  # baseline doesn't resolve pronouns
            else:
                # Use lowercased canonical surface as the "id"
                per_turn[m.surface] = "surf::" + m.surface.lower().strip()
        decisions[t.idx] = per_turn
    return decisions


def registry_coref_decisions(coref_log: dict) -> dict[int, dict[str, str | None]]:
    """Convert aen2 coref_log into surface -> id mapping."""
    out: dict[int, dict[str, str | None]] = {}
    for tidx, decs in coref_log.items():
        per_turn: dict[str, str | None] = {}
        for d in decs:
            per_turn[d.surface] = d.entity_id
        out[tidx] = per_turn
    return out


# ---------------------------------------------------------------------------
# Alignment + grading
# ---------------------------------------------------------------------------


def align_ids_to_gt(
    scenario: Scenario,
    decisions: dict[int, dict[str, str | None]],
) -> dict[str | None, str | None]:
    """Find best mapping from architecture-internal id -> GT label.

    For each internal id, count which GT label its surface mentions
    correspond to (excluding mentions whose GT is None or where the
    architecture said None). Pick the majority GT label.
    """
    cooc: dict[str | None, Counter] = defaultdict(Counter)
    for t in scenario.turns:
        per_turn = decisions.get(t.idx, {})
        for m in t.mentions:
            arch_id = per_turn.get(m.surface)
            if arch_id is None:
                continue
            if m.entity_id is None:
                continue
            cooc[arch_id][m.entity_id] += 1
    mapping: dict[str | None, str | None] = {}
    used_gt: set[str] = set()
    # Sort architecture-ids by total support so the strongest claims win.
    arch_ids_sorted = sorted(cooc.keys(), key=lambda a: -sum(cooc[a].values()))
    for arch_id in arch_ids_sorted:
        # Pick best GT label not yet claimed (so different arch ids prefer
        # different GT entities — important for S1's two-Alice case).
        candidates = cooc[arch_id].most_common()
        chosen = None
        for gt, _ in candidates:
            if gt not in used_gt:
                chosen = gt
                break
        if chosen is None and candidates:
            chosen = candidates[0][0]  # fallback: collision allowed
        mapping[arch_id] = chosen
        if chosen is not None:
            used_gt.add(chosen)
    return mapping


def grade_scenario(
    scenario: Scenario,
    decisions: dict[int, dict[str, str | None]],
) -> dict:
    mapping = align_ids_to_gt(scenario, decisions)

    n_total = 0
    n_correct = 0
    n_named_correct = 0
    n_named_total = 0
    n_pron_correct = 0
    n_pron_total = 0
    n_desc_correct = 0
    n_desc_total = 0
    failures: list[dict] = []

    for t in scenario.turns:
        per_turn = decisions.get(t.idx, {})
        for m in t.mentions:
            if m.entity_id is None:
                continue  # don't grade unscored mentions
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


def run_registry(scenario: Scenario, cache: Cache, budget: Budget) -> dict:
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
    decisions = registry_coref_decisions(coref_log)
    grade = grade_scenario(scenario, decisions)
    delta_llm = budget.llm_calls - pre_llm
    delta_emb = budget.embed_calls - pre_emb
    return {
        "arch": "aen2_registry",
        "grade": grade,
        "log_size": len(log),
        "n_entities": len(reg.by_id),
        "registry": aen2_registry.registry_snapshot(reg),
        "coref_decisions": {
            str(t): [asdict(d) for d in decs] for t, decs in coref_log.items()
        },
        "llm_calls_delta": delta_llm,
        "embed_calls_delta": delta_emb,
        "llm_per_turn": delta_llm / max(1, len(scenario.turns)),
    }


def run_baseline(scenario: Scenario, cache: Cache, budget: Budget) -> dict:
    """The baseline aen1_simple has NO coref pass. We just score what the
    surface-form-as-id strategy would produce, and we also run the writer to
    measure cost.

    For pronouns, baseline simply returns None (can't resolve) — those count
    as misses. For named/descriptor mentions, two distinct entities sharing a
    surface form will collide.
    """
    pairs = [(t.idx, t.text) for t in scenario.turns]
    pre_llm = budget.llm_calls
    pre_emb = budget.embed_calls
    log, idx_obj = aen1_simple.ingest_turns(
        pairs,
        cache,
        budget,
        batch_size=5,
        rebuild_index_every=100,
    )
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
    budget = Budget(max_llm=500, max_embed=100, stop_at_llm=480, stop_at_embed=95)
    scenarios = all_scenarios()
    summary: dict[str, dict] = {}

    for s in scenarios:
        print(f"\n=== Scenario: {s.name} ({len(s.turns)} turns) ===")
        s_results: dict[str, dict] = {}
        # Registry
        cache_reg = Cache(CACHE_DIR / f"reg_{s.name}.json")
        try:
            print("  [registry] running...")
            s_results["registry"] = run_registry(s, cache_reg, budget)
            g = s_results["registry"]["grade"]
            print(
                f"    acc={g['accuracy']:.2%} "
                f"named={g['named']['rate']:.2%} "
                f"pron={g['pronoun']['rate']:.2%} "
                f"desc={g['descriptor']['rate']:.2%} "
                f"({g['n_correct']}/{g['n_total']}) "
                f"LLM/turn={s_results['registry']['llm_per_turn']:.2f}"
            )
        except Exception as e:
            print(f"    REGISTRY ERROR: {e}")
            s_results["registry"] = {"error": str(e)}

        # Baseline
        cache_base = Cache(CACHE_DIR / f"base_{s.name}.json")
        try:
            print("  [baseline] running...")
            s_results["baseline"] = run_baseline(s, cache_base, budget)
            g = s_results["baseline"]["grade"]
            print(
                f"    acc={g['accuracy']:.2%} "
                f"named={g['named']['rate']:.2%} "
                f"pron={g['pronoun']['rate']:.2%} "
                f"desc={g['descriptor']['rate']:.2%} "
                f"({g['n_correct']}/{g['n_total']}) "
                f"LLM/turn={s_results['baseline']['llm_per_turn']:.2f}"
            )
        except Exception as e:
            print(f"    BASELINE ERROR: {e}")
            s_results["baseline"] = {"error": str(e)}

        summary[s.name] = s_results
        # Save partial results after every scenario
        out = RESULTS_DIR / f"{s.name}.json"
        out.write_text(json.dumps(s_results, indent=2, default=str))

    # Top-level summary
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
                        "pronoun": v.get("grade", {}).get("pronoun", {}).get("rate"),
                        "descriptor": v.get("grade", {})
                        .get("descriptor", {})
                        .get("rate"),
                        "llm_per_turn": v.get("llm_per_turn"),
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
