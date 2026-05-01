"""Round 19 v2 generalization — test on dense_chains and dormant_chains.

Validates that the v2 binding architecture (which won 8/8 on multi_batch_coref)
also handles the harder scenarios:
  - dense_chains: 14 active predicates, 86 transitions over hundreds of turns
  - dormant_chains: long quiet periods between updates

K=3 w7+w7 with truncated scenarios (limit 200 turns) to keep cost bounded.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND19 = HERE.parent
RESEARCH = ROUND19.parent
ROUND16A = RESEARCH / "round16a_sliding_window"
ROUND14 = RESEARCH / "round14_chain_density"
ROUND7 = RESEARCH / "round7"

sys.path.insert(0, str(ROUND19 / "architectures"))
sys.path.insert(0, str(ROUND16A / "scenarios"))
sys.path.insert(0, str(ROUND14 / "scenarios"))
sys.path.insert(0, str(ROUND7 / "experiments"))

import aen2_binding_v2 as ab  # noqa: E402
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

CACHE_DIR = ROUND19 / "cache"
RESULTS_DIR = ROUND19 / "results"


def grade_qa(qs, idx, cache, budget) -> dict:
    answers = {}
    for q in qs:
        a = ab.answer_question(q.question, idx, cache, budget, top_k=14)
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


def run_variant(name, *, turns, gt, qs, w_past, w_future, k, budget, do_qa=True):
    cache = Cache(CACHE_DIR / f"{name}.json")
    pairs = [(t.idx, t.text) for t in turns]
    print(
        f"\n=== {name} (n_turns={len(pairs)}, w_past={w_past}, K={k}, w_future={w_future}) ==="
    )
    print(f"[budget] llm={budget.llm_calls} cost=${budget.cost():.3f}")
    out = {
        "variant": name,
        "n_turns": len(pairs),
        "w_past": w_past,
        "w_future": w_future,
        "k": k,
    }
    try:
        log, resolutions, idx, telemetry = ab.ingest_turns(
            pairs,
            cache,
            budget,
            w_past=w_past,
            w_future=w_future,
            k=k,
            rebuild_index_every=4,
        )
    except RuntimeError as e:
        print(f"!!! ingest budget stop: {e}")
        cache.save()
        out["error"] = str(e)
        return out
    cache.save()
    print(
        f"[ingest] entries={len(log)} resolutions={len(resolutions)} "
        f"clusters={len(idx.cluster_entries)} chains={len(idx.chain_head)}"
    )
    out["log_size"] = len(log)
    out["n_resolutions"] = len(resolutions)
    out["n_clusters"] = len(idx.cluster_entries)
    out["n_chains"] = len(idx.chain_head)

    if do_qa and qs:
        try:
            print(f"[QA] running {len(qs)} questions...")
            qa = grade_qa(qs, idx, cache, budget)
            out["qa"] = qa
            print(
                f"[QA] det={qa['deterministic_pass']}/{qa['total']}  "
                f"judge={qa['judge_pass']}/{qa['total']}"
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
    p = RESULTS_DIR / "run_v2_generalize.json"
    p.write_text(json.dumps(results, indent=2, default=str))
    print(f"[checkpoint] {p} cost=${budget.cost():.3f}")


def main():
    budget = Budget(max_llm=600, max_embed=300, stop_at_llm=550, stop_at_embed=280)

    DC_LIMIT = 200
    dc_turns_full = dense_chains.generate()
    dc_turns = dc_turns_full[:DC_LIMIT]
    dc_gt = dense_chains.ground_truth(dc_turns)
    dc_qs = dense_chains.build_questions(dc_gt)

    DORM_LIMIT = 200
    dorm_turns_full = dormant_chains.generate()
    dorm_turns = dorm_turns_full[:DORM_LIMIT]
    dorm_gt = dormant_chains.ground_truth(dorm_turns)
    dorm_qs = dormant_chains.build_questions(dorm_gt)

    print(
        f"[dense_chains:{DC_LIMIT}] turns={len(dc_turns)} chains={len(dc_gt.chains)} "
        f"Qs={len(dc_qs)}"
    )
    print(
        f"[dormant_chains:{DORM_LIMIT}] turns={len(dorm_turns)} chains={len(dorm_gt.chains)} "
        f"Qs={len(dorm_qs)}"
    )

    results = {"variants": {}}

    plan = [
        ("v2_dense_K3_w7_w7", dc_turns, dc_gt, dc_qs, 7, 7, 3),
        ("v2_dorm_K3_w7_w7", dorm_turns, dorm_gt, dorm_qs, 7, 7, 3),
    ]

    for vname, turns, gt, qs, w_past, w_future, k in plan:
        try:
            res = run_variant(
                vname,
                turns=turns,
                gt=gt,
                qs=qs,
                w_past=w_past,
                w_future=w_future,
                k=k,
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
