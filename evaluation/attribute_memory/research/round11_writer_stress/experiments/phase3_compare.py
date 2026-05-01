"""Phase 3: typed vs simplified writer at 1000 turns — head-to-head.

Runs the typed-ref writer on the same 1000-turn conversation, then compares
ref-emission, chain integrity, and end-to-end Q/A vs phase2 simplified.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND11 = HERE.parent
ROUND7 = ROUND11.parent / "round7"
sys.path.insert(0, str(ROUND11 / "architectures"))
sys.path.insert(0, str(ROUND11 / "scenarios"))
sys.path.insert(0, str(ROUND7 / "experiments"))

import aen1_typed_baseline as typed  # noqa: E402
import long_conversation  # noqa: E402
from _common import Budget, Cache  # noqa: E402
from phase2_writer_stress import (  # noqa: E402
    atag_drift_analysis,
    grade_questions,
    ref_emission_metrics,
)

CACHE_DIR = ROUND11 / "cache"
RESULTS_DIR = ROUND11 / "results"


def run(budget: Budget, n_turns: int = 1000) -> dict:
    turns = long_conversation.generate(n_turns)
    gt = long_conversation.ground_truth(turns)
    questions = long_conversation.build_questions(gt)

    cache = Cache(CACHE_DIR / "phase3_typed.json")
    pairs = [(t.idx, t.text) for t in turns]
    print(f"Ingesting {len(pairs)} turns in batches of 5 with TYPED writer...")
    log, idx = typed.ingest_turns(
        pairs, cache, budget, batch_size=5, rebuild_index_every=40
    )
    cache.save()
    print(f"log size: {len(log)} entries")
    print(
        f"Cost so far: ${budget.cost():.3f}  LLM={budget.llm_calls} "
        f"embed={budget.embed_calls}"
    )

    # Adapt typed log to (mentions-only) format for ref_emission_metrics.
    # The metrics function only checks mentions, text, refs -> so convert Ref to
    # plain uuid strings for the purposes of "emitted_ref".
    class _Adapted:
        def __init__(self, e):
            self.uuid = e.uuid
            self.ts = e.ts
            self.text = e.text
            self.mentions = e.mentions
            self.refs = [r.uuid for r in e.refs]
            self.predicate = e.predicate

    adapted = [_Adapted(e) for e in log]

    ref_metrics = ref_emission_metrics(turns, gt, adapted)
    tag_drift = atag_drift_analysis(adapted)

    print(f"\nAnswering {len(questions)} questions...")
    answers = {}
    for q in questions:
        a = typed.answer_question(q.question, idx, cache, budget, top_k=12)
        answers[q.qid] = a
    cache.save()
    verdicts = grade_questions(questions, answers)
    passed = sum(1 for v in verdicts if v["passed"])
    print(f"PASSED {passed}/{len(verdicts)}")
    print(
        f"Cost: ${budget.cost():.3f}  LLM={budget.llm_calls}  "
        f"embed={budget.embed_calls}"
    )

    return {
        "arch": "aen1_typed_baseline",
        "n_turns": n_turns,
        "log_size": len(log),
        "num_supersede_heads": len(idx.supersede_head),
        "num_entities": len(idx.mention_index),
        "ref_metrics_summary": {
            "n_transitions_total": ref_metrics["n_transitions_total"],
            "n_transitions_non_first": ref_metrics["n_transitions_non_first"],
            "entry_emission_rate": ref_metrics["entry_emission_rate"],
            "ref_emission_rate": ref_metrics["ref_emission_rate"],
            "atag_rate": ref_metrics["atag_rate"],
            "bucket_stats": ref_metrics["bucket_stats"],
        },
        "tag_drift": tag_drift,
        "per_transition": ref_metrics["per_transition"],
        "answers": answers,
        "verdicts": verdicts,
        "qa_passed": passed,
        "qa_total": len(verdicts),
        "cost": budget.cost(),
        "llm_calls": budget.llm_calls,
        "embed_calls": budget.embed_calls,
    }


def compare(simple_path: Path, typed_path: Path) -> dict:
    s = json.loads(simple_path.read_text())
    t = json.loads(typed_path.read_text())
    out = {
        "simple": {
            "log_size": s["log_size"],
            "cost": s["cost"],
            "llm_calls": s["llm_calls"],
            "ref_emission_rate": s["ref_metrics_summary"]["ref_emission_rate"],
            "atag_rate": s["ref_metrics_summary"]["atag_rate"],
            "qa_passed": s["qa_passed"],
            "qa_total": s["qa_total"],
            "bucket_stats": s["ref_metrics_summary"]["bucket_stats"],
        },
        "typed": {
            "log_size": t["log_size"],
            "cost": t["cost"],
            "llm_calls": t["llm_calls"],
            "ref_emission_rate": t["ref_metrics_summary"]["ref_emission_rate"],
            "atag_rate": t["ref_metrics_summary"]["atag_rate"],
            "qa_passed": t["qa_passed"],
            "qa_total": t["qa_total"],
            "bucket_stats": t["ref_metrics_summary"]["bucket_stats"],
        },
    }
    return out


def main() -> None:
    budget = Budget(max_llm=700, max_embed=300, stop_at_llm=670, stop_at_embed=290)
    result = run(budget)
    out = RESULTS_DIR / "phase3.json"
    out.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nWrote {out}")

    comp = compare(RESULTS_DIR / "phase2.json", out)
    (RESULTS_DIR / "phase3_compare.json").write_text(json.dumps(comp, indent=2))
    print("Comparison:")
    print(json.dumps(comp, indent=2))


if __name__ == "__main__":
    main()
