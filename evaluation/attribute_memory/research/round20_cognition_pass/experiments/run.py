"""Round 20 — cognition pass alongside variable-binding writer.

Tests two things:
  1. Regression: does the cognition pass HURT v2's 8/8 on multi_batch_coref?
  2. Improvement: does the cognition pass help on hypothetical_becomes_real
     (the new scenario where prior expectations should be retrievable)?

Variants:
  - cog_off: enable_cognition=False (matches v2 baseline)
  - cog_on:  enable_cognition=True (cognition pass after each writer)
"""

from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND20 = HERE.parent
RESEARCH = ROUND20.parent
ROUND19 = RESEARCH / "round19_variable_binding"
ROUND16A = RESEARCH / "round16a_sliding_window"
ROUND14 = RESEARCH / "round14_chain_density"
ROUND7 = RESEARCH / "round7"

sys.path.insert(0, str(ROUND20 / "architectures"))
sys.path.insert(0, str(ROUND20 / "scenarios"))
sys.path.insert(0, str(ROUND16A / "scenarios"))
sys.path.insert(0, str(ROUND7 / "experiments"))

import aen3_cognition as ac  # noqa: E402
import hypothetical_becomes_real as hbr  # noqa: E402
import multi_batch_coref  # noqa: E402
from _common import Budget, Cache  # noqa: E402


def _load_module(alias: str, path: Path):
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


r14_run = _load_module("r14_run", ROUND14 / "experiments" / "run.py")

CACHE_DIR = ROUND20 / "cache"
RESULTS_DIR = ROUND20 / "results"


def grade_qa(qs, idx, cache, budget) -> dict:
    answers = {}
    for q in qs:
        a = ac.answer_question(q.question, idx, cache, budget, top_k=14)
        answers[q.qid] = a
    cache.save()
    verdicts = r14_run.grade_deterministic(qs, answers)
    det_pass = sum(1 for v in verdicts if v["passed"])
    judged = r14_run.judge_with_llm(verdicts, cache, budget)
    cache.save()
    judge_pass = sum(1 for v in judged if v["judge_correct"])
    return {
        "answers": answers,
        "verdicts": judged,
        "deterministic_pass": det_pass,
        "judge_pass": judge_pass,
        "total": len(verdicts),
    }


def run_variant(name, *, turns, gt, qs, w_past, w_future, k, enable_cognition, budget):
    cache = Cache(CACHE_DIR / f"{name}.json")
    pairs = [(t.idx, t.text) for t in turns]
    print(
        f"\n=== {name} (n_turns={len(pairs)}, w_past={w_past}, K={k}, "
        f"w_future={w_future}, cognition={enable_cognition}) ==="
    )
    print(f"[budget] llm={budget.llm_calls} cost=${budget.cost():.3f}")
    out = {
        "variant": name,
        "n_turns": len(pairs),
        "w_past": w_past,
        "w_future": w_future,
        "k": k,
        "enable_cognition": enable_cognition,
    }
    try:
        log, resolutions, idx, telemetry = ac.ingest_turns(
            pairs,
            cache,
            budget,
            w_past=w_past,
            w_future=w_future,
            k=k,
            rebuild_index_every=4,
            enable_cognition=enable_cognition,
        )
    except RuntimeError as e:
        print(f"!!! ingest budget stop: {e}")
        cache.save()
        out["error"] = str(e)
        return out
    cache.save()
    n_obs = sum(1 for e in log if e.entry_type == "observation")
    n_cog = sum(1 for e in log if e.entry_type == "cognition")
    print(
        f"[ingest] entries={len(log)} (obs={n_obs} cog={n_cog}) "
        f"resolutions={len(resolutions)} clusters={len(idx.cluster_entries)} "
        f"chains={len(idx.chain_head)}"
    )
    out["log_size"] = len(log)
    out["n_observations"] = n_obs
    out["n_cognitions"] = n_cog
    out["n_resolutions"] = len(resolutions)
    out["n_clusters"] = len(idx.cluster_entries)
    out["n_chains"] = len(idx.chain_head)

    if qs:
        try:
            print(f"[QA] running {len(qs)} questions...")
            qa = grade_qa(qs, idx, cache, budget)
            out["qa"] = qa
            print(
                f"[QA] det={qa['deterministic_pass']}/{qa['total']}  judge={qa['judge_pass']}/{qa['total']}"
            )
            for v in qa["verdicts"]:
                marker = "✓" if v.get("judge_correct") else "✗"
                print(
                    f"  {marker} [{v['qid']}] {v.get('question', '')[:60]} -> {v.get('answer', '')[:80]}"
                )
        except RuntimeError as e:
            print(f"!!! QA budget stop: {e}")
            cache.save()
            out["qa_error"] = str(e)

    out["llm_calls_after"] = budget.llm_calls
    out["cost_after"] = budget.cost()
    return out


def save(results, budget):
    results["budget"] = {"cost": budget.cost(), "llm_calls": budget.llm_calls}
    p = RESULTS_DIR / "run.json"
    p.write_text(json.dumps(results, indent=2, default=str))
    print(f"[checkpoint] {p} cost=${budget.cost():.3f}")


def main():
    budget = Budget(max_llm=600, max_embed=300, stop_at_llm=550, stop_at_embed=280)

    coref_turns = multi_batch_coref.generate()
    coref_gt = multi_batch_coref.ground_truth(coref_turns)
    coref_qs = multi_batch_coref.build_questions(coref_gt)

    hbr_turns = hbr.generate()
    hbr_gt = hbr.ground_truth(hbr_turns)
    hbr_qs = hbr.build_questions(hbr_gt)

    print(f"[multi_batch_coref] turns={len(coref_turns)} Qs={len(coref_qs)}")
    print(f"[hypothetical_becomes_real] turns={len(hbr_turns)} Qs={len(hbr_qs)}")
    for p in hbr_gt.pairs:
        print(
            f"  pair: cond_t={p.get('conditional_turn')} -> trigger_t={p.get('trigger_turn')} -> named_t={p.get('named_turn')}"
        )

    results = {"variants": {}}

    plan = [
        # 1. Cognition OFF on coref (regression baseline — should equal v2's 8/8)
        ("coref_cog_off", coref_turns, coref_gt, coref_qs, 7, 7, 3, False),
        # 2. Cognition ON on coref (does cognition HURT factual coref?)
        ("coref_cog_on", coref_turns, coref_gt, coref_qs, 7, 7, 3, True),
        # 3. Cognition OFF on HBR (baseline for the hypothetical scenario)
        ("hbr_cog_off", hbr_turns, hbr_gt, hbr_qs, 7, 7, 3, False),
        # 4. Cognition ON on HBR (does cognition HELP on hypothetical-becomes-real?)
        ("hbr_cog_on", hbr_turns, hbr_gt, hbr_qs, 7, 7, 3, True),
    ]

    for vname, turns, gt, qs, w_past, w_future, k, cog in plan:
        try:
            res = run_variant(
                vname,
                turns=turns,
                gt=gt,
                qs=qs,
                w_past=w_past,
                w_future=w_future,
                k=k,
                enable_cognition=cog,
                budget=budget,
            )
        except RuntimeError as e:
            print(f"!!! {vname} budget hit: {e}")
            res = {"variant": vname, "error": str(e)}
        except Exception as e:
            print(f"!!! {vname} CRASHED: {e}")
            traceback.print_exc()
            res = {"variant": vname, "crash": str(e)}
        results["variants"][vname] = res
        save(results, budget)
        if budget.llm_calls >= budget.stop_at_llm - 5:
            print("[plan] near LLM cap; halting.")
            break

    save(results, budget)
    print(f"\n[done] cost=${budget.cost():.3f} llm={budget.llm_calls}")


if __name__ == "__main__":
    main()
