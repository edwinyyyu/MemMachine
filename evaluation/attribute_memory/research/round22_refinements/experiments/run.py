"""Round 22 — refinements on top of round 21 partitioned base.

Three deltas:
  1. Writer prompt: add possessions/hobbies/recurring routines/confirmed plans
     as chain-worthy. (Fix HBR Q05.)
  2. Cognition retrieval gate: include "end up", "actually", "did user end"
     patterns to surface cognition for confirmation-of-plan questions.
  3. Cognizer prompt: stricter triggers (CONDITIONAL/CONFIRMATION/CONTRADICTION/
     NAMED-HOPE-FEAR only); default empty; max 1 per K-block. (Reduce noise.)

Tests against round 21 baseline:
  - coref_part_cog_off (R21): 7/8 → expect maintained or improved
  - coref_part_cog_on  (R21): 7/8 → expect maintained, fewer cog entries
  - hbr_part_cog_off   (R21): 4/5 → expect 5/5 (Bianchi captured)
  - hbr_part_cog_on    (R21): 4/5 → expect 5/5
"""

from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND22 = HERE.parent
RESEARCH = ROUND22.parent
ROUND20 = RESEARCH / "round20_cognition_pass"
ROUND16A = RESEARCH / "round16a_sliding_window"
ROUND14 = RESEARCH / "round14_chain_density"
ROUND7 = RESEARCH / "round7"

sys.path.insert(0, str(ROUND22 / "architectures"))
sys.path.insert(0, str(ROUND20 / "scenarios"))
sys.path.insert(0, str(ROUND16A / "scenarios"))
sys.path.insert(0, str(ROUND7 / "experiments"))

import aen5_refined as ar  # noqa: E402
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

CACHE_DIR = ROUND22 / "cache"
RESULTS_DIR = ROUND22 / "results"


def grade_qa(qs, store, cache, budget) -> dict:
    answers = {}
    for q in qs:
        a = ar.answer_question(q.question, store, cache, budget, top_k=14)
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
        obs_log, cog_log, store, telemetry = ar.ingest_turns(
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
    obs_idx = store.get("observations")
    cog_idx = store.get("cognition")
    print(
        f"[ingest] obs={len(obs_log)} cog={len(cog_log)} "
        f"obs_clusters={len(obs_idx.cluster_entries) if obs_idx else 0} "
        f"obs_chains={len(obs_idx.chain_head) if obs_idx else 0}"
    )
    out["obs_log_size"] = len(obs_log)
    out["cog_log_size"] = len(cog_log)
    out["obs_clusters"] = len(obs_idx.cluster_entries) if obs_idx else 0
    out["obs_chains"] = len(obs_idx.chain_head) if obs_idx else 0
    out["cog_clusters"] = len(cog_idx.cluster_entries) if cog_idx else 0
    out["cog_chains"] = len(cog_idx.chain_head) if cog_idx else 0

    if qs:
        try:
            print(f"[QA] running {len(qs)} questions...")
            qa = grade_qa(qs, store, cache, budget)
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

    results = {"variants": {}}

    plan = [
        ("r22_coref_cog_off", coref_turns, coref_gt, coref_qs, 7, 7, 3, False),
        ("r22_coref_cog_on", coref_turns, coref_gt, coref_qs, 7, 7, 3, True),
        ("r22_hbr_cog_off", hbr_turns, hbr_gt, hbr_qs, 7, 7, 3, False),
        ("r22_hbr_cog_on", hbr_turns, hbr_gt, hbr_qs, 7, 7, 3, True),
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
