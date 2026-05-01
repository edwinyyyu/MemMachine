"""Scale sweep: run AEN-1 plain vs AEN-1 indexed across scenarios × scales.

Deterministic entries — no LLM writer calls. Budget goes to:
  - 1 embedding call per scenario (batched) for entry texts
  - 1 embedding call per question (cached globally)
  - 1 reader LLM call per (arch, scenario_instance, question)
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND10 = HERE.parent
ROUND7 = ROUND10.parent / "round7"
sys.path.insert(0, str(ROUND10 / "architectures"))
sys.path.insert(0, str(ROUND10 / "scenarios"))
sys.path.insert(0, str(ROUND7 / "experiments"))

import aen1_indexed as indexed  # noqa: E402
from _common import Budget, Cache  # noqa: E402
from generators import GENERATORS  # noqa: E402
from grader import grade_all  # noqa: E402

CACHE_DIR = ROUND10 / "cache"
CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR = ROUND10 / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# (scenario_name, scale) pairs to run. We skip some larger runs if budget tight.
# Start with the primary sweep: 200/500/1000 for all four scenarios.
RUNS = [
    ("dense", 200),
    ("dense", 500),
    ("dense", 1000),
    ("distractors", 200),
    ("distractors", 500),
    ("distractors", 1000),
    ("interleaved", 200),
    ("interleaved", 500),
    ("interleaved", 1000),
    ("deep_chain", 200),
    ("deep_chain", 500),
    ("deep_chain", 1000),
]


def run_one(scen: str, scale: int, budget: Budget) -> dict:
    gen = GENERATORS[scen]
    # For deep_chain at larger scales we increase chain_len a bit
    if scen == "deep_chain":
        chain_len = {200: 10, 500: 12, 1000: 15}[scale]
        entries, questions = gen(scale, chain_len=chain_len)
    else:
        entries, questions = gen(scale)

    cache = Cache(CACHE_DIR / f"{scen}_{scale}.json")
    t0 = time.time()
    idx = indexed.build_index(entries, cache, budget)
    t_index = time.time() - t0

    results = {
        "scenario": scen,
        "scale": scale,
        "n_entries": len(entries),
        "n_questions": len(questions),
        "t_index_s": round(t_index, 2),
        "n_mention_index_keys": len(idx.mention_index),
        "n_supersede_head": len(idx.supersede_head),
        "n_superseded_by": len(idx.superseded_by),
    }

    # Run both architectures on the same log
    per_arch: dict[str, dict] = {}
    for arch_name in ["aen1_indexed", "aen1_plain"]:
        answers: dict[str, str] = {}
        for q in questions:
            try:
                if arch_name == "aen1_indexed":
                    ans = indexed.answer_indexed(q.question, idx, cache, budget)
                else:
                    ans = indexed.answer_plain(
                        q.question, entries, idx.embed_by_uuid, cache, budget
                    )
            except RuntimeError as e:
                ans = f"[BUDGET_STOP: {e}]"
            answers[q.qid] = ans
        cache.save()
        verdicts = grade_all(questions, answers)
        passed = sum(1 for v in verdicts if v.passed)
        per_arch[arch_name] = {
            "answers": answers,
            "verdicts": [asdict(v) for v in verdicts],
            "passed": passed,
            "total": len(verdicts),
        }
    results["archs"] = per_arch
    cache.save()
    return results


def main() -> None:
    budget = Budget(max_llm=500, max_embed=300, stop_at_llm=480, stop_at_embed=290)
    all_results: list[dict] = []
    try:
        for scen, scale in RUNS:
            print(f"\n=== {scen} @ {scale} ===")
            try:
                r = run_one(scen, scale, budget)
                all_results.append(r)
                # summarize
                for arch_name, arch in r["archs"].items():
                    print(f"  [{arch_name}] {arch['passed']}/{arch['total']}")
                print(
                    f"  Budget so far: LLM={budget.llm_calls}, embed={budget.embed_calls}, "
                    f"cost=${budget.cost():.3f}"
                )
            except RuntimeError as e:
                print(f"  BUDGET STOP: {e}")
                all_results.append({"scenario": scen, "scale": scale, "error": str(e)})
                break
    finally:
        out = RESULTS_DIR / "scale_sweep.json"
        out.write_text(
            json.dumps(
                {
                    "final_cost": budget.cost(),
                    "llm_calls": budget.llm_calls,
                    "embed_calls": budget.embed_calls,
                    "runs": all_results,
                },
                indent=2,
                default=str,
            )
        )
        print(f"\nWrote {out}")
        print(
            f"Final: LLM={budget.llm_calls}, embed={budget.embed_calls}, "
            f"cost=${budget.cost():.3f}"
        )


if __name__ == "__main__":
    main()
