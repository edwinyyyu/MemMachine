"""Round 23 v2 — refined writer prompt: hard filler skip + prefer-existing-entity merges."""

from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND23 = HERE.parent
RESEARCH = ROUND23.parent
ROUND14 = RESEARCH / "round14_chain_density"
ROUND16A = RESEARCH / "round16a_sliding_window"
ROUND20 = RESEARCH / "round20_cognition_pass"
ROUND7 = RESEARCH / "round7"

sys.path.insert(0, str(ROUND23 / "architectures"))
sys.path.insert(0, str(ROUND14 / "scenarios"))
sys.path.insert(0, str(ROUND16A / "scenarios"))
sys.path.insert(0, str(ROUND20 / "scenarios"))
sys.path.insert(0, str(ROUND7 / "experiments"))

import aen6_prose_v2 as ap  # noqa: E402
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

CACHE_DIR = ROUND23 / "cache"
RESULTS_DIR = ROUND23 / "results"


def grade_qa(qs, store, cache, budget) -> dict:
    answers = {}
    for q in qs:
        a = ap.answer_question(q.question, store, cache, budget, top_k=14)
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


def run_variant(name, *, turns, qs, budget):
    cache = Cache(CACHE_DIR / f"{name}.json")
    pairs = [(t.idx, t.text) for t in turns]
    print(f"\n=== {name} (n_turns={len(pairs)}) ===")
    print(f"[budget] llm={budget.llm_calls} cost=${budget.cost():.3f}")
    out = {"variant": name, "n_turns": len(pairs)}
    try:
        facts, mentions, store, telemetry = ap.ingest_turns(
            pairs,
            cache,
            budget,
            w_past=7,
            w_future=7,
            k=3,
            rebuild_index_every=4,
        )
    except RuntimeError as e:
        print(f"!!! ingest budget stop: {e}")
        cache.save()
        out["error"] = str(e)
        return out
    cache.save()
    n_entities = len(store.registry.entity_members)
    n_merges = sum(1 for b in store.registry.binding_events if b.op == "merge")
    print(
        f"[ingest] facts={len(facts)} mentions={len(mentions)} entities={n_entities} merges={n_merges}"
    )
    out["log_size"] = len(facts)
    out["n_entities"] = n_entities
    out["n_merges"] = n_merges

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
    out["cost_after"] = budget.cost()
    return out


def save(results, budget):
    results["budget"] = {"cost": budget.cost(), "llm_calls": budget.llm_calls}
    p = RESULTS_DIR / "run_v2.json"
    p.write_text(json.dumps(results, indent=2, default=str))
    print(f"[checkpoint] {p} cost=${budget.cost():.3f}")


def main():
    budget = Budget(max_llm=700, max_embed=350, stop_at_llm=650, stop_at_embed=320)

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
    print(f"[dorm:{DORM_LIMIT}] turns={len(dorm_turns)} Qs={len(dorm_qs)}")
    print(f"[dense:{DC_LIMIT}] turns={len(dc_turns)} Qs={len(dc_qs)}")

    results = {"variants": {}}

    plan = [
        ("r23v2_coref", coref_turns, coref_qs),
        ("r23v2_hbr", hbr_turns, hbr_qs),
        ("r23v2_dorm", dorm_turns, dorm_qs),
        ("r23v2_dense", dc_turns, dc_qs),
    ]

    for name, turns, qs in plan:
        try:
            res = run_variant(name, turns=turns, qs=qs, budget=budget)
        except RuntimeError as e:
            print(f"!!! {name} budget hit: {e}")
            res = {"variant": name, "error": str(e)}
        except Exception as e:
            print(f"!!! {name} CRASHED: {e}")
            traceback.print_exc()
            res = {"variant": name, "crash": str(e)}
        results["variants"][name] = res
        save(results, budget)
        if budget.llm_calls >= budget.stop_at_llm - 5:
            print("[plan] near LLM cap; halting.")
            break

    save(results, budget)
    print(f"\n[done] cost=${budget.cost():.3f} llm={budget.llm_calls}")


if __name__ == "__main__":
    main()
