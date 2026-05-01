"""Round 22 generalization — test refined architecture on dense_chains + dormant_chains.

V2 was 17/23 on dense; v3 broader was 10/23 (regression from over-emission).
R22 prompt has the broader durable list (possessions, hobbies, routines) +
partitioned storage + cognizer. Question: does R22 hit a better Pareto point
than either?
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
ROUND14 = RESEARCH / "round14_chain_density"
ROUND16A = RESEARCH / "round16a_sliding_window"
ROUND7 = RESEARCH / "round7"

sys.path.insert(0, str(ROUND22 / "architectures"))
sys.path.insert(0, str(ROUND14 / "scenarios"))
sys.path.insert(0, str(ROUND16A / "scenarios"))
sys.path.insert(0, str(ROUND7 / "experiments"))

import aen5_refined as ar  # noqa: E402
import dense_chains  # noqa: E402
import dormant_chains  # noqa: E402
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
    out = {"variant": name, "n_turns": len(pairs), "enable_cognition": enable_cognition}
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
    print(
        f"[ingest] obs={len(obs_log)} cog={len(cog_log)} "
        f"obs_chains={len(obs_idx.chain_head) if obs_idx else 0}"
    )
    out["obs_log_size"] = len(obs_log)
    out["cog_log_size"] = len(cog_log)
    out["obs_chains"] = len(obs_idx.chain_head) if obs_idx else 0

    if qs:
        try:
            print(f"[QA] running {len(qs)} questions...")
            qa = grade_qa(qs, store, cache, budget)
            out["qa"] = qa
            print(
                f"[QA] det={qa['deterministic_pass']}/{qa['total']}  judge={qa['judge_pass']}/{qa['total']}"
            )
        except RuntimeError as e:
            print(f"!!! QA budget stop: {e}")
            cache.save()
            out["qa_error"] = str(e)
    out["cost_after"] = budget.cost()
    return out


def save(results, budget):
    results["budget"] = {"cost": budget.cost(), "llm_calls": budget.llm_calls}
    p = RESULTS_DIR / "run_generalize.json"
    p.write_text(json.dumps(results, indent=2, default=str))
    print(f"[checkpoint] {p} cost=${budget.cost():.3f}")


def main():
    budget = Budget(max_llm=700, max_embed=350, stop_at_llm=650, stop_at_embed=320)

    DC_LIMIT = 200
    dc_turns = dense_chains.generate()[:DC_LIMIT]
    dc_gt = dense_chains.ground_truth(dc_turns)
    dc_qs = dense_chains.build_questions(dc_gt)

    DORM_LIMIT = 200
    dorm_turns = dormant_chains.generate()[:DORM_LIMIT]
    dorm_gt = dormant_chains.ground_truth(dorm_turns)
    dorm_qs = dormant_chains.build_questions(dorm_gt)

    print(f"[dense:{DC_LIMIT}] turns={len(dc_turns)} Qs={len(dc_qs)}")
    print(f"[dorm:{DORM_LIMIT}] turns={len(dorm_turns)} Qs={len(dorm_qs)}")

    results = {"variants": {}}

    plan = [
        # Cognition off — apples-to-apples vs R19 v2/v3 baselines
        ("r22_dense_cog_off", dc_turns, dc_gt, dc_qs, 7, 7, 3, False),
        ("r22_dorm_cog_off", dorm_turns, dorm_gt, dorm_qs, 7, 7, 3, False),
        # Cognition on — does cognition help on factual chain Qs (no, by design)
        # but does it hurt? regression check.
        ("r22_dense_cog_on", dc_turns, dc_gt, dc_qs, 7, 7, 3, True),
        ("r22_dorm_cog_on", dorm_turns, dorm_gt, dorm_qs, 7, 7, 3, True),
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
