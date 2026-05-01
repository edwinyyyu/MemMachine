"""Direct R13b (immediate entity resolution) vs R22 (variable binding) comparison.

R13b = persistent registry + LRU + lazy embedding pull + per-turn coref pass.
R22  = variable binding + partitioned storage + trigger-gated cognition.

Both run on multi_batch_coref + HBR + dense_chains[:200] + dorm[:200] for
apples-to-apples.

R22 results were already collected (run.json + run_generalize.json). This
script just runs R13b on the same scenarios and assembles a comparison table.
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
ROUND13 = RESEARCH / "round13_persistent_registry"
ROUND14 = RESEARCH / "round14_chain_density"
ROUND16A = RESEARCH / "round16a_sliding_window"
ROUND20 = RESEARCH / "round20_cognition_pass"
ROUND11 = RESEARCH / "round11_writer_stress"
ROUND7 = RESEARCH / "round7"

sys.path.insert(0, str(ROUND13 / "aen3b"))
sys.path.insert(0, str(ROUND13 / "architectures"))
sys.path.insert(0, str(ROUND11 / "architectures"))
sys.path.insert(0, str(ROUND7 / "experiments"))
sys.path.insert(0, str(ROUND14 / "scenarios"))
sys.path.insert(0, str(ROUND16A / "scenarios"))
sys.path.insert(0, str(ROUND20 / "scenarios"))

import aen1_simple  # noqa: E402
import aen3b_persistent as r13b  # noqa: E402
import dense_chains  # noqa: E402
import dormant_chains  # noqa: E402
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


def grade_qa(qs, idx, cache, budget) -> dict:
    answers = {}
    for q in qs:
        a = aen1_simple.answer_question(q.question, idx, cache, budget, top_k=14)
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


def run_r13b(name, *, turns, qs, budget):
    cache = Cache(CACHE_DIR / f"r13b_{name}.json")
    pairs = [(t.idx, t.text) for t in turns]
    print(f"\n=== r13b on {name} (n_turns={len(pairs)}) ===")
    print(f"[budget] llm={budget.llm_calls} cost=${budget.cost():.3f}")
    out = {"variant": f"r13b_{name}", "n_turns": len(pairs)}
    try:
        log, idx, reg, coref_log = r13b.ingest_turns_with_registry(
            pairs,
            cache,
            budget,
            batch_size=5,
            lru_size=20,
            top_k=5,
        )
    except RuntimeError as e:
        print(f"!!! ingest budget stop: {e}")
        cache.save()
        out["error"] = str(e)
        return out
    cache.save()
    print(f"[ingest] entries={len(log)} entities={len(reg.by_id)}")
    out["log_size"] = len(log)
    out["n_entities"] = len(reg.by_id)

    if qs and idx is not None:
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
    out["cost_after"] = budget.cost()
    return out


def save(results, budget):
    results["budget"] = {"cost": budget.cost(), "llm_calls": budget.llm_calls}
    p = RESULTS_DIR / "compare_r13b_vs_r22.json"
    p.write_text(json.dumps(results, indent=2, default=str))
    print(f"[checkpoint] {p} cost=${budget.cost():.3f}")


def main():
    budget = Budget(max_llm=900, max_embed=500, stop_at_llm=850, stop_at_embed=470)

    coref_turns = multi_batch_coref.generate()
    coref_qs = multi_batch_coref.build_questions(
        multi_batch_coref.ground_truth(coref_turns)
    )

    hbr_turns = hbr.generate()
    hbr_qs = hbr.build_questions(hbr.ground_truth(hbr_turns))

    DC_LIMIT = 200
    dc_turns = dense_chains.generate()[:DC_LIMIT]
    dc_qs = dense_chains.build_questions(dense_chains.ground_truth(dc_turns))

    DORM_LIMIT = 200
    dorm_turns = dormant_chains.generate()[:DORM_LIMIT]
    dorm_qs = dormant_chains.build_questions(dormant_chains.ground_truth(dorm_turns))

    print(f"[coref] turns={len(coref_turns)} Qs={len(coref_qs)}")
    print(f"[hbr]   turns={len(hbr_turns)} Qs={len(hbr_qs)}")
    print(f"[dense:{DC_LIMIT}] turns={len(dc_turns)} Qs={len(dc_qs)}")
    print(f"[dorm:{DORM_LIMIT}] turns={len(dorm_turns)} Qs={len(dorm_qs)}")

    results = {"variants": {}}

    plan = [
        ("coref", coref_turns, coref_qs),
        ("hbr", hbr_turns, hbr_qs),
        ("dorm", dorm_turns, dorm_qs),
        ("dense", dc_turns, dc_qs),
    ]

    for name, turns, qs in plan:
        try:
            res = run_r13b(name, turns=turns, qs=qs, budget=budget)
        except RuntimeError as e:
            print(f"!!! {name} budget hit: {e}")
            res = {"variant": f"r13b_{name}", "error": str(e)}
        except Exception as e:
            print(f"!!! {name} CRASHED: {e}")
            traceback.print_exc()
            res = {"variant": f"r13b_{name}", "crash": str(e)}
        results["variants"][f"r13b_{name}"] = res
        save(results, budget)
        if budget.llm_calls >= budget.stop_at_llm - 5:
            print("[plan] near LLM cap; halting.")
            break

    save(results, budget)
    print(f"\n[done] cost=${budget.cost():.3f} llm={budget.llm_calls}")


if __name__ == "__main__":
    main()
