"""Phase 1: simplified (1 ref type) vs typed (4 ref types) at 110-turn scale.

Uses round 9's phase1 scenario and its 20 state-tracking questions.

Runs:
  - aen1_simple  (single ref type, structural indexes)
  - aen1_typed_baseline  (4 relations, structural indexes — equivalent to round 10)

Same scenario, same questions, same grader.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND11 = HERE.parent
ROUND9 = ROUND11.parent / "round9"
ROUND7 = ROUND11.parent / "round7"
sys.path.insert(0, str(ROUND11 / "architectures"))
sys.path.insert(0, str(ROUND9 / "experiments"))
sys.path.insert(0, str(ROUND7 / "experiments"))

# Load round9 phase1 scenario directly to avoid circular import with
# round9's experiments/phase1.py (same module name).
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "round9_phase1_scenario", ROUND9 / "scenarios" / "phase1.py"
)
_scen = importlib.util.module_from_spec(_spec)
sys.modules["round9_phase1_scenario"] = _scen
_spec.loader.exec_module(_scen)
TURNS = _scen.TURNS
QUESTIONS = _scen.QUESTIONS

import aen1_simple  # noqa: E402
import aen1_typed_baseline  # noqa: E402
from _common import Budget, Cache  # noqa: E402
from grader import grade_all  # noqa: E402

CACHE_DIR = ROUND11 / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = ROUND11 / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _turn_pairs(turns):
    return [(t.idx, t.text) for t in turns]


def run(arch_name: str, module, budget: Budget) -> dict:
    cache = Cache(CACHE_DIR / f"phase1_{arch_name}.json")
    print(f"[{arch_name}] ingesting {len(TURNS)} turns in batches of 5...")
    pairs = _turn_pairs(TURNS)
    log, idx = module.ingest_turns(
        pairs, cache, budget, batch_size=5, rebuild_index_every=100
    )
    cache.save()
    print(f"[{arch_name}] log size: {len(log)} entries")

    answers = {}
    for q in QUESTIONS:
        ans = module.answer_question(q.question, idx, cache, budget, top_k=12)
        answers[q.qid] = ans
    cache.save()
    verdicts = grade_all(QUESTIONS, answers)
    passed = sum(1 for v in verdicts if v.passed)
    print(
        f"[{arch_name}] PASSED {passed}/{len(verdicts)}   "
        f"cost=${budget.cost():.3f}  LLM={budget.llm_calls}  embed={budget.embed_calls}"
    )
    return {
        "arch": arch_name,
        "log_size": len(log),
        "answers": answers,
        "verdicts": [asdict(v) for v in verdicts],
        "cost": budget.cost(),
        "llm_calls": budget.llm_calls,
        "embed_calls": budget.embed_calls,
    }


def main() -> None:
    # Budget for BOTH arches combined
    budget = Budget(max_llm=600, max_embed=200, stop_at_llm=580, stop_at_embed=190)
    results = {}
    for arch_name, module in [
        ("aen1_simple", aen1_simple),
        ("aen1_typed_baseline", aen1_typed_baseline),
    ]:
        print(f"\n=== Running {arch_name} ===")
        try:
            results[arch_name] = run(arch_name, module, budget)
        except Exception as e:
            print(f"[{arch_name}] ERROR: {e}")
            results[arch_name] = {"error": str(e)}
    out = RESULTS_DIR / "phase1.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out}")
    print(
        f"Total cost: ${budget.cost():.3f}  LLM={budget.llm_calls}  "
        f"embed={budget.embed_calls}"
    )


if __name__ == "__main__":
    main()
