"""Round 16B v2 runner — simpler world category sets.

Compares against round 16B v1 (4-category: real / hypothetical / fiction:pyrrhus /
fiction:novel_project) and against the no-scoping baseline.

Two variants:
  - n_cats=3: {real, hypothetical, fiction}
  - n_cats=2: {real, non_real}

Hard cap: 250 LLM + 50 embed, $2. Stop at 200 LLM = 80%.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND16B_V2 = HERE.parent
RESEARCH = ROUND16B_V2.parent
ROUND16B = RESEARCH / "round16b_world_scoping"
ROUND15 = RESEARCH / "round15_active_chains"
ROUND11 = RESEARCH / "round11_writer_stress"
ROUND7 = RESEARCH / "round7"
sys.path.insert(0, str(ROUND16B_V2 / "architectures"))
sys.path.insert(0, str(ROUND16B))
sys.path.insert(0, str(ROUND16B / "architectures"))
sys.path.insert(0, str(ROUND15 / "architectures"))
sys.path.insert(0, str(ROUND11 / "architectures"))
sys.path.insert(0, str(ROUND7 / "experiments"))

import aen1_worlds_simple  # noqa: E402
from _common import Budget, Cache  # noqa: E402
from scenarios import (  # noqa: E402
    fantasy_roleplay,
    mixed,
    novel_writing,
)
from scenarios import (
    hypothetical as hyp_mod,
)

CACHE_DIR = ROUND16B_V2 / "cache"
RESULTS_DIR = ROUND16B_V2 / "results"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 5


# ---------------------------------------------------------------------------
# Map a v1 expected_world (real / hypothetical / joke / fiction:* / game:*)
# into the variant's world space so we can score classifier accuracy.
# ---------------------------------------------------------------------------


def map_expected(expected: str, n_cats: int) -> str:
    if expected == "real":
        return "real"
    if n_cats == 2:
        return "non_real"
    # 3-cat
    if expected == "hypothetical":
        return "hypothetical"
    if expected == "joke":
        # Round 16B treats jokes as their own bucket. Under the simpler 3-cat
        # space, jokes about obvious untruths live with "fiction".
        return "fiction"
    if expected.startswith("fiction:") or expected.startswith("game:"):
        return "fiction"
    return expected


def map_must_not_world(must_not: list[str], n_cats: int) -> list[str]:
    return [map_expected(w, n_cats) for w in must_not]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def expected_batch_world(turns_in_batch, n_cats: int) -> str:
    """Pick the modal expected_world (mapped) among turns; ties go to last turn."""
    mapped = [map_expected(t.expected_world, n_cats) for t in turns_in_batch]
    counts = Counter(mapped)
    max_n = max(counts.values())
    candidates = [w for w, n in counts.items() if n == max_n]
    if len(candidates) == 1:
        return candidates[0]
    return mapped[-1]


def classifier_accuracy(scenario, telemetry: list[dict], n_cats: int):
    n_total = 0
    n_correct = 0
    n_axis = 0
    rows = []
    turns = scenario.turns
    for tele in telemetry:
        b = tele["batch_no"]
        i = b * BATCH_SIZE
        batch_turns = turns[i : i + BATCH_SIZE]
        if not batch_turns:
            continue
        expected = expected_batch_world(batch_turns, n_cats)
        actual = tele["classified_world"]
        is_correct = expected == actual

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
            n_axis += 1
    return {
        "n_batches": n_total,
        "exact": n_correct,
        "axis_correct": n_axis,
        "exact_pct": n_correct / n_total if n_total else 0.0,
        "axis_pct": n_axis / n_total if n_total else 0.0,
        "rows": rows,
    }


def cross_pollination_rate(scenario, log: list, n_cats: int) -> dict:
    """Variant-aware cross-pollination: remap fact-check worlds into the
    variant's world space first, then score."""
    per_check = []
    total_in_world = 0
    total_leaked = 0
    for fc in scenario.fact_checks:
        target_world = map_expected(fc.world, n_cats)
        forbidden = set(map_must_not_world(fc.must_not_world, n_cats))
        # Don't penalize ourselves for a "leak" into the same mapped world.
        forbidden.discard(target_world)
        in_world = 0
        leaked_to = []
        for e in log:
            text_low = e.text.lower()
            hit = any(p.lower() in text_low for p in fc.must_contain)
            if not hit:
                continue
            if e.world == target_world:
                in_world += 1
            elif forbidden and e.world in forbidden:
                leaked_to.append({"uuid": e.uuid, "world": e.world, "text": e.text})
        per_check.append(
            {
                "description": fc.description,
                "expected_world": target_world,
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
    return dict(Counter(e.world for e in log))


def grade_qa(qa_results: list[dict]) -> dict:
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
                "expected_world_v1": r["expected_world_v1"],
                "expected_world_mapped": r["expected_world_mapped"],
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


def run_scenario(scenario, cache: Cache, budget: Budget, n_cats: int):
    print(
        f"\n=== scenario: {scenario.name} (n_cats={n_cats}, "
        f"{len(scenario.turns)} turns) ==="
    )
    pairs = [(t.idx, t.text) for t in scenario.turns]

    t0 = time.time()
    n_llm_before = budget.llm_calls
    n_emb_before = budget.embed_calls

    log, idx, telemetry = aen1_worlds_simple.ingest_turns(
        pairs,
        cache,
        budget,
        n_cats=n_cats,
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

    wd = world_distribution(log)
    print(f"  [world distribution] {wd}")

    cacc = classifier_accuracy(scenario, telemetry, n_cats)
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

    xp = cross_pollination_rate(scenario, log, n_cats)
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

    qa_with = []
    qa_without = []
    print("  [QA running with world scoping...]")
    for q in scenario.qa_checks:
        mapped_world = map_expected(q.expected_world, n_cats)
        try:
            ans = aen1_worlds_simple.answer_question(
                q.question,
                idx,
                cache,
                budget,
                n_cats=n_cats,
                top_k=14,
                world=mapped_world,
            )
        except RuntimeError as e:
            print(f"    !! budget stop during QA-with: {e}")
            ans = "[budget_stop]"
        qa_with.append(
            {
                "qid": q.qid,
                "question": q.question,
                "expected_world_v1": q.expected_world,
                "expected_world_mapped": mapped_world,
                "answer": ans,
                "must_include": q.must_include,
                "must_exclude": q.must_exclude,
            }
        )
    cache.save()

    print("  [QA running without world scoping...]")
    for q in scenario.qa_checks:
        try:
            ans = aen1_worlds_simple.answer_question_no_world(
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
                "expected_world_v1": q.expected_world,
                "expected_world_mapped": map_expected(q.expected_world, n_cats),
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
        "n_cats": n_cats,
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
    # Expected ~208 LLM calls across both variants (32 batches × 2 prompts × 2
    # variants for ingest + 20 QA × 2 paths × 2 variants for QA = ~208). Cap
    # generously to leave headroom; stop at 80% of cap.
    budget = Budget(max_llm=600, max_embed=200, stop_at_llm=560, stop_at_embed=190)

    scenarios = [
        ("fantasy_roleplay", fantasy_roleplay.generate()),
        ("novel_writing", novel_writing.generate()),
        ("hypothetical", hyp_mod.generate()),
        ("mixed", mixed.generate()),
    ]

    all_results = {"variants": {}, "budget": {}}

    for n_cats in (3, 2):
        variant_key = f"{n_cats}cat"
        print(f"\n############ VARIANT n_cats={n_cats} ############")
        all_results["variants"][variant_key] = {"scenarios": {}}
        stop_variant = False
        for name, scen in scenarios:
            cache_path = CACHE_DIR / f"{name}_{variant_key}.json"
            cache = Cache(cache_path)
            try:
                res = run_scenario(scen, cache, budget, n_cats=n_cats)
                all_results["variants"][variant_key]["scenarios"][name] = res
            except RuntimeError as e:
                print(f"\n!!! Budget stop in {variant_key}/{name}: {e}")
                traceback.print_exc()
                all_results["variants"][variant_key]["scenarios"][name] = {
                    "error": str(e),
                    "budget_stop": True,
                }
                cache.save()
                stop_variant = True
            except Exception as e:
                print(f"\n!!! Unexpected error in {variant_key}/{name}: {e}")
                traceback.print_exc()
                all_results["variants"][variant_key]["scenarios"][name] = {
                    "error": str(e),
                }
                cache.save()

            all_results["budget"] = {
                "llm_calls": budget.llm_calls,
                "embed_calls": budget.embed_calls,
                "cost": budget.cost(),
            }
            (RESULTS_DIR / "run.json").write_text(
                json.dumps(all_results, indent=2, default=str)
            )
            print(
                f"  [checkpoint] cost=${budget.cost():.3f} "
                f"llm={budget.llm_calls} embed={budget.embed_calls}"
            )
            if stop_variant:
                break
        if stop_variant:
            print(f"\n!!! Stopping after variant {variant_key} due to budget.")
            break

    # Aggregate per variant
    for variant_key, payload in all_results["variants"].items():
        agg = aggregate(payload)
        payload["aggregate"] = agg

    all_results["budget"] = {
        "llm_calls": budget.llm_calls,
        "embed_calls": budget.embed_calls,
        "cost": budget.cost(),
    }
    (RESULTS_DIR / "run.json").write_text(
        json.dumps(all_results, indent=2, default=str)
    )

    # Print head-to-head
    print("\n\n############ HEAD-TO-HEAD ############")
    print(
        f"{'scenario':<22} {'variant':<6} {'cls_exact':>10} {'cls_axis':>10} "
        f"{'xpoll':>8} {'qa_w':>6} {'qa_no':>6}"
    )
    for variant_key in ("3cat", "2cat"):
        payload = all_results["variants"].get(variant_key, {})
        for name, s in payload.get("scenarios", {}).items():
            if "classifier_accuracy" not in s:
                continue
            ca = s["classifier_accuracy"]
            cp = s["cross_pollination"]
            print(
                f"{name:<22} {variant_key:<6} "
                f"{ca['exact']}/{ca['n_batches']:>2} ({ca['exact_pct']:>3.0%}) "
                f"{ca['axis_correct']}/{ca['n_batches']:>2} ({ca['axis_pct']:>3.0%}) "
                f"{cp['cross_pollination_rate']:>7.1%} "
                f"{s['qa_with_world']['pass_pct']:>5.0%} "
                f"{s['qa_without_world']['pass_pct']:>5.0%}"
            )

    print(
        f"\n[done] cost=${budget.cost():.3f} llm={budget.llm_calls} "
        f"embed={budget.embed_calls}"
    )


def aggregate(variant_payload) -> dict:
    out = {}
    scens = [
        v
        for v in variant_payload.get("scenarios", {}).values()
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
