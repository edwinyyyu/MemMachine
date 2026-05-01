"""Phase 1: run each architecture through the 110-turn scenario and 20 questions.

Compare per-kind accuracy and total cost.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND9 = HERE.parent
ROUND7 = ROUND9.parent / "round7"
sys.path.insert(0, str(ROUND9 / "architectures"))
sys.path.insert(0, str(ROUND9 / "scenarios"))
sys.path.insert(0, str(ROUND7 / "experiments"))

import aen1  # noqa: E402
import aen1_views  # noqa: E402
import partitioned  # noqa: E402
from _common import Budget, Cache  # noqa: E402
from grader import grade_all  # noqa: E402
from phase1 import QUESTIONS, TURNS  # noqa: E402

CACHE_DIR = ROUND9 / "cache"
CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR = ROUND9 / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_aen1(budget: Budget) -> dict:
    cache = Cache(CACHE_DIR / "aen1.json")
    print(f"[aen1] ingesting {len(TURNS)} turns...")
    log = aen1.ingest_scenario(TURNS, cache, budget)
    cache.save()
    print(f"[aen1] log size: {len(log)} entries")

    print(f"[aen1] answering {len(QUESTIONS)} questions...")
    answers = {}
    for q in QUESTIONS:
        ans = aen1.answer_question(q.question, log, cache, budget, top_k=12)
        answers[q.qid] = ans
    cache.save()
    verdicts = grade_all(QUESTIONS, answers)
    return {
        "arch": "aen1",
        "log_size": len(log),
        "answers": answers,
        "verdicts": [asdict(v) for v in verdicts],
        "cost": budget.cost(),
        "llm_calls": budget.llm_calls,
        "embed_calls": budget.embed_calls,
    }


def run_aen1_views(budget: Budget) -> dict:
    cache = Cache(CACHE_DIR / "aen1.json")  # shared with aen1 for ingestion re-use
    print("[aen1_views] building from aen1 ingest...")
    log, views, inv, sup = aen1_views.ingest_and_build(TURNS, cache, budget)
    cache.save()
    print(f"[aen1_views] log size: {len(log)}, views: {len(views)}")

    answers = {}
    for q in QUESTIONS:
        ans = aen1_views.answer_question_with_views(
            q.question,
            log,
            views,
            inv,
            sup,
            cache,
            budget,
            top_k_extra=6,
        )
        answers[q.qid] = ans
    cache.save()
    verdicts = grade_all(QUESTIONS, answers)
    return {
        "arch": "aen1_views",
        "log_size": len(log),
        "num_views": len(views),
        "answers": answers,
        "verdicts": [asdict(v) for v in verdicts],
        "cost": budget.cost(),
        "llm_calls": budget.llm_calls,
        "embed_calls": budget.embed_calls,
    }


def run_partitioned(budget: Budget) -> dict:
    cache = Cache(CACHE_DIR / "partitioned.json")
    print(f"[partitioned] ingesting {len(TURNS)} turns...")
    store = partitioned.ingest_scenario(TURNS, cache, budget)
    cache.save()
    print(
        f"[partitioned] topics: {len(store.topics)}, "
        f"total entries: {len(store.all_entries)}, slots: {len(store.slots)}"
    )

    answers = {}
    for q in QUESTIONS:
        ans = partitioned.answer_question(q.question, store, cache, budget, top_k=12)
        answers[q.qid] = ans
    cache.save()
    verdicts = grade_all(QUESTIONS, answers)
    return {
        "arch": "partitioned",
        "n_topics": len(store.topics),
        "n_entries": len(store.all_entries),
        "n_slots": len(store.slots),
        "answers": answers,
        "verdicts": [asdict(v) for v in verdicts],
        "cost": budget.cost(),
        "llm_calls": budget.llm_calls,
        "embed_calls": budget.embed_calls,
    }


def main() -> None:
    budget = Budget(max_llm=400, max_embed=200, stop_at_llm=380, stop_at_embed=180)

    results = {}
    for arch_name, runner in [
        ("aen1", run_aen1),
        ("aen1_views", run_aen1_views),
        ("partitioned", run_partitioned),
    ]:
        print(f"\n=== Running {arch_name} ===")
        try:
            r = runner(budget)
            results[arch_name] = r
            passed = sum(1 for v in r["verdicts"] if v["passed"])
            total = len(r["verdicts"])
            print(f"[{arch_name}] PASSED {passed}/{total}")
            print(
                f"[{arch_name}] cost so far: ${budget.cost():.3f}, "
                f"LLM={budget.llm_calls}, embed={budget.embed_calls}"
            )
        except Exception as e:
            print(f"[{arch_name}] ERROR: {e}")
            results[arch_name] = {"error": str(e)}

    out_path = RESULTS_DIR / "phase1.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
