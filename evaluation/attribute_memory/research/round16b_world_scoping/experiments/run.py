"""Round 16b runner — world scoping evaluation.

Hard cap: 300 LLM + 50 embed, $3. Stop at 240 LLM (80%).

For each of 4 scenarios:
  - Ingest with aen1_worlds (classify-then-write per batch)
  - Compute world-classifier accuracy (per batch)
  - Compute cross-pollination rate (entry-level)
  - Run Q/A with world scoping AND without world scoping
  - Judge Q/A answers via deterministic must_include/must_exclude
  - Save per-scenario results, then aggregate
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND16B = HERE.parent
RESEARCH = ROUND16B.parent
ROUND15 = RESEARCH / "round15_active_chains"
ROUND11 = RESEARCH / "round11_writer_stress"
ROUND7 = RESEARCH / "round7"
sys.path.insert(0, str(ROUND16B / "architectures"))
sys.path.insert(0, str(ROUND16B))
sys.path.insert(0, str(ROUND15 / "architectures"))
sys.path.insert(0, str(ROUND11 / "architectures"))
sys.path.insert(0, str(ROUND7 / "experiments"))

import aen1_worlds  # noqa: E402
from _common import Budget, Cache  # noqa: E402
from scenarios import (  # noqa: E402
    fantasy_roleplay,
    mixed,
    novel_writing,
)
from scenarios import (
    hypothetical as hyp_mod,
)

CACHE_DIR = ROUND16B / "cache"
RESULTS_DIR = ROUND16B / "results"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 5


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def expected_batch_world(turns_in_batch) -> str:
    """Pick the modal expected_world among the turns in the batch (ties go to
    the LAST turn since worlds are sticky-from-end-of-batch). If a batch
    contains both real and a fictional world, treat the dominant signal as
    ground truth — but flag it as an ambiguous batch in telemetry."""
    counts = Counter(t.expected_world for t in turns_in_batch)
    max_n = max(counts.values())
    candidates = [w for w, n in counts.items() if n == max_n]
    if len(candidates) == 1:
        return candidates[0]
    # Tie: pick the last turn's expected world
    return turns_in_batch[-1].expected_world


def classifier_accuracy(scenario, telemetry: list[dict]):
    n_total = 0
    n_correct = 0
    n_partial = 0  # right "axis" (fiction vs real) but wrong slug
    rows = []
    turns = scenario.turns
    for tele in telemetry:
        b = tele["batch_no"]
        i = b * BATCH_SIZE
        batch_turns = turns[i : i + BATCH_SIZE]
        if not batch_turns:
            continue
        expected = expected_batch_world(batch_turns)
        actual = tele["classified_world"]
        is_correct = expected == actual

        # Partial-credit: real-vs-non-real axis correct
        def axis(w):
            return "real" if w == "real" else "non-real"

        is_axis_correct = axis(expected) == axis(actual)

        rows.append(
            {
                "batch_no": b,
                "expected": expected,
                "actual": actual,
                "correct": is_correct,
                "axis_correct": is_axis_correct,
                "reason": tele.get("classify_reason", ""),
                "turn_text_sample": batch_turns[0].text[:80],
            }
        )
        n_total += 1
        if is_correct:
            n_correct += 1
        if is_axis_correct:
            n_partial += 1
    return {
        "n_batches": n_total,
        "exact": n_correct,
        "axis_correct": n_partial,
        "exact_pct": n_correct / n_total if n_total else 0.0,
        "axis_pct": n_partial / n_total if n_total else 0.0,
        "rows": rows,
    }


def cross_pollination_rate(scenario, log: list) -> dict:
    """For each FactCheck, count:
      - n_correct_world: entries in `world` that contain a must_contain phrase
      - n_leaked: entries in must_not_world that contain a must_contain phrase
    Aggregate cross-pollination = leaks / (leaks + in-world hits) across all
    fact checks.
    """
    per_check = []
    total_in_world = 0
    total_leaked = 0
    for fc in scenario.fact_checks:
        in_world = 0
        leaked_to = []
        for e in log:
            text_low = e.text.lower()
            hit = any(p.lower() in text_low for p in fc.must_contain)
            if not hit:
                continue
            if e.world == fc.world:
                in_world += 1
            elif fc.must_not_world and e.world in fc.must_not_world:
                leaked_to.append({"uuid": e.uuid, "world": e.world, "text": e.text})
        per_check.append(
            {
                "description": fc.description,
                "expected_world": fc.world,
                "n_in_world": in_world,
                "n_leaked": len(leaked_to),
                "leaks": leaked_to[:5],
            }
        )
        total_in_world += in_world
        total_leaked += len(leaked_to)
    denom = total_in_world + total_leaked
    rate = total_leaked / denom if denom else 0.0
    return {
        "n_fact_checks": len(scenario.fact_checks),
        "total_in_world": total_in_world,
        "total_leaked": total_leaked,
        "cross_pollination_rate": rate,
        "per_check": per_check,
    }


def world_distribution(log: list) -> dict:
    counts = Counter(e.world for e in log)
    return dict(counts)


def grade_qa(qa_results: list[dict]) -> dict:
    """Each result: {qid, question, expected_world, answer, must_include, must_exclude}.
    Pass = at least one must_include substring is in answer (case-insensitive)
           AND no must_exclude substring is in answer.
    """
    n = 0
    n_pass = 0
    rows = []
    for r in qa_results:
        ans = (r["answer"] or "").lower()
        inc_hit = any(p.lower() in ans for p in r["must_include"])
        exc_hit = any(p.lower() in ans for p in (r.get("must_exclude") or []))
        passed = inc_hit and not exc_hit
        rows.append(
            {
                "qid": r["qid"],
                "question": r["question"],
                "expected_world": r["expected_world"],
                "answer": r["answer"],
                "must_include_hit": inc_hit,
                "must_exclude_hit": exc_hit,
                "passed": passed,
            }
        )
        n += 1
        if passed:
            n_pass += 1
    return {
        "n": n,
        "n_pass": n_pass,
        "pass_pct": n_pass / n if n else 0.0,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------


def run_scenario(scenario, cache: Cache, budget: Budget):
    print(f"\n=== scenario: {scenario.name} ({len(scenario.turns)} turns) ===")
    pairs = [(t.idx, t.text) for t in scenario.turns]

    t0 = time.time()
    n_llm_before = budget.llm_calls
    n_emb_before = budget.embed_calls

    log, idx, telemetry = aen1_worlds.ingest_turns(
        pairs,
        cache,
        budget,
        batch_size=BATCH_SIZE,
        rebuild_index_every=4,
        max_active_state_size=80,
    )
    cache.save()

    ingest_llm = budget.llm_calls - n_llm_before
    ingest_emb = budget.embed_calls - n_emb_before
    print(
        f"  [ingest] entries={len(log)} llm={ingest_llm} emb={ingest_emb} "
        f"t={time.time() - t0:.1f}s cost=${budget.cost():.3f}"
    )

    # World distribution
    wd = world_distribution(log)
    print(f"  [world distribution] {wd}")

    # Classifier accuracy
    cacc = classifier_accuracy(scenario, telemetry)
    print(
        f"  [classifier] exact={cacc['exact']}/{cacc['n_batches']} "
        f"({cacc['exact_pct']:.0%})  axis={cacc['axis_correct']}/{cacc['n_batches']} "
        f"({cacc['axis_pct']:.0%})"
    )
    for r in cacc["rows"]:
        if not r["correct"]:
            print(
                f"    miss b{r['batch_no']}: exp={r['expected']} "
                f"got={r['actual']} | {r['turn_text_sample']!r} | "
                f"reason={r['reason']!r}"
            )

    # Cross-pollination
    xp = cross_pollination_rate(scenario, log)
    print(
        f"  [cross-pollination] in_world={xp['total_in_world']} "
        f"leaked={xp['total_leaked']} rate={xp['cross_pollination_rate']:.2%}"
    )
    for c in xp["per_check"]:
        if c["n_leaked"] > 0:
            print(
                f"    LEAK '{c['description']}' "
                f"(world={c['expected_world']}): "
                f"{c['n_leaked']} leaks; e.g. {c['leaks'][0]}"
            )

    # Q/A with world scoping
    qa_with = []
    qa_without = []
    print("  [QA running with world scoping...]")
    for q in scenario.qa_checks:
        try:
            ans = aen1_worlds.answer_question(
                q.question,
                idx,
                cache,
                budget,
                top_k=14,
                world=q.expected_world,
            )
        except RuntimeError as e:
            print(f"    !! budget stop during QA-with: {e}")
            ans = "[budget_stop]"
        qa_with.append(
            {
                "qid": q.qid,
                "question": q.question,
                "expected_world": q.expected_world,
                "answer": ans,
                "must_include": q.must_include,
                "must_exclude": q.must_exclude,
            }
        )
    cache.save()

    print("  [QA running without world scoping...]")
    for q in scenario.qa_checks:
        try:
            ans = aen1_worlds.answer_question_no_world(
                q.question,
                idx,
                cache,
                budget,
                top_k=14,
            )
        except RuntimeError as e:
            print(f"    !! budget stop during QA-without: {e}")
            ans = "[budget_stop]"
        qa_without.append(
            {
                "qid": q.qid,
                "question": q.question,
                "expected_world": q.expected_world,
                "answer": ans,
                "must_include": q.must_include,
                "must_exclude": q.must_exclude,
            }
        )
    cache.save()

    grade_with = grade_qa(qa_with)
    grade_without = grade_qa(qa_without)
    print(
        f"  [QA grade] with-world: {grade_with['n_pass']}/{grade_with['n']} "
        f"({grade_with['pass_pct']:.0%})"
    )
    print(
        f"  [QA grade] no-world:   {grade_without['n_pass']}/{grade_without['n']} "
        f"({grade_without['pass_pct']:.0%})"
    )

    return {
        "scenario": scenario.name,
        "n_turns": len(scenario.turns),
        "n_entries": len(log),
        "ingest_llm_calls": ingest_llm,
        "ingest_embed_calls": ingest_emb,
        "world_distribution": wd,
        "classifier_accuracy": cacc,
        "cross_pollination": xp,
        "qa_with_world": grade_with,
        "qa_without_world": grade_without,
        "log_dump": [
            {
                "uuid": e.uuid,
                "ts": e.ts,
                "world": e.world,
                "predicate": e.predicate,
                "mentions": e.mentions,
                "refs": e.refs,
                "text": e.text,
            }
            for e in log
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # Hard cap: 300 LLM + 50 embed, $3. Stop at 240 LLM = 80%.
    budget = Budget(max_llm=300, max_embed=50, stop_at_llm=240, stop_at_embed=45)

    scenarios = [
        ("fantasy_roleplay", fantasy_roleplay.generate()),
        ("novel_writing", novel_writing.generate()),
        ("hypothetical", hyp_mod.generate()),
        ("mixed", mixed.generate()),
    ]

    all_results = {"scenarios": {}, "budget": {}}

    for name, scen in scenarios:
        cache_path = CACHE_DIR / f"{name}.json"
        cache = Cache(cache_path)
        try:
            res = run_scenario(scen, cache, budget)
            all_results["scenarios"][name] = res
        except RuntimeError as e:
            print(f"\n!!! Budget stop in {name}: {e}")
            traceback.print_exc()
            all_results["scenarios"][name] = {"error": str(e), "budget_stop": True}
            cache.save()
            break
        except Exception as e:
            print(f"\n!!! Unexpected error in {name}: {e}")
            traceback.print_exc()
            all_results["scenarios"][name] = {"error": str(e)}
            cache.save()

        # Checkpoint after each scenario
        all_results["budget"] = {
            "llm_calls": budget.llm_calls,
            "embed_calls": budget.embed_calls,
            "cost": budget.cost(),
        }
        out = RESULTS_DIR / "run.json"
        out.write_text(json.dumps(all_results, indent=2, default=str))
        print(
            f"  [checkpoint] cost=${budget.cost():.3f} "
            f"llm={budget.llm_calls} embed={budget.embed_calls}"
        )

    # Aggregate
    agg = aggregate(all_results)
    all_results["aggregate"] = agg
    all_results["budget"] = {
        "llm_calls": budget.llm_calls,
        "embed_calls": budget.embed_calls,
        "cost": budget.cost(),
    }
    out = RESULTS_DIR / "run.json"
    out.write_text(json.dumps(all_results, indent=2, default=str))

    print("\n=== AGGREGATE ===")
    for k, v in agg.items():
        print(f"  {k}: {v}")
    print(
        f"\n[done] cost=${budget.cost():.3f} llm={budget.llm_calls} "
        f"embed={budget.embed_calls}"
    )


def aggregate(results) -> dict:
    out = {}
    scens = [
        v
        for v in results["scenarios"].values()
        if isinstance(v, dict) and "classifier_accuracy" in v
    ]
    if not scens:
        return out
    n_b = sum(s["classifier_accuracy"]["n_batches"] for s in scens)
    n_exact = sum(s["classifier_accuracy"]["exact"] for s in scens)
    n_axis = sum(s["classifier_accuracy"]["axis_correct"] for s in scens)
    n_in = sum(s["cross_pollination"]["total_in_world"] for s in scens)
    n_leak = sum(s["cross_pollination"]["total_leaked"] for s in scens)
    qa_w = sum(s["qa_with_world"]["n_pass"] for s in scens)
    qa_w_n = sum(s["qa_with_world"]["n"] for s in scens)
    qa_o = sum(s["qa_without_world"]["n_pass"] for s in scens)
    qa_o_n = sum(s["qa_without_world"]["n"] for s in scens)

    out["classifier_exact_overall"] = (
        f"{n_exact}/{n_b} ({n_exact / n_b:.0%})" if n_b else "n/a"
    )
    out["classifier_axis_overall"] = (
        f"{n_axis}/{n_b} ({n_axis / n_b:.0%})" if n_b else "n/a"
    )
    out["cross_pollination_rate_overall"] = (
        n_leak / (n_in + n_leak) if (n_in + n_leak) else 0.0
    )
    out["qa_with_world_overall"] = (
        f"{qa_w}/{qa_w_n} ({qa_w / qa_w_n:.0%})" if qa_w_n else "n/a"
    )
    out["qa_without_world_overall"] = (
        f"{qa_o}/{qa_o_n} ({qa_o / qa_o_n:.0%})" if qa_o_n else "n/a"
    )
    return out


if __name__ == "__main__":
    main()
