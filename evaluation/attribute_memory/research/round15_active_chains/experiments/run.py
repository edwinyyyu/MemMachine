"""Round 15 - at-write-time fix for the writer ref-emission collapse.

Compares:
  - aen1_simple (round 14 baseline, loaded from disk)
  - aen1_active with max_active_state_size in {50, 100, 200}

Same scenario (dense_chains, seed=17), same metrics + grader as round 14.

Budget plan (hard cap 400 LLM, target $2):
  - cap=100, batch_size=5, FULL run (writer ~149 + QA 32 + judge ~16 = ~197 LLM)
  - cap=200, batch_size=5, writer-only (~149 LLM)
  - cap=50,  batch_size=5, writer-only (~149 LLM, will likely budget-stop -
    we wrap each in try/except so partial telemetry is preserved)

If budget stops mid-run, we save partial results and report what we have.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND15 = HERE.parent
RESEARCH = ROUND15.parent
ROUND14 = RESEARCH / "round14_chain_density"
ROUND11 = RESEARCH / "round11_writer_stress"
ROUND7 = RESEARCH / "round7"
sys.path.insert(0, str(ROUND15 / "architectures"))
sys.path.insert(0, str(ROUND14 / "scenarios"))
sys.path.insert(0, str(ROUND14 / "experiments"))
sys.path.insert(0, str(ROUND11 / "architectures"))
sys.path.insert(0, str(ROUND7 / "experiments"))

import aen1_active  # noqa: E402
import dense_chains  # noqa: E402

# Reuse round 14's metric helpers verbatim.
import run as r14_run  # noqa: E402  (this is round14's experiments/run.py)
from _common import Budget, Cache  # noqa: E402

CACHE_DIR = ROUND15 / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = ROUND15 / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def collect_metrics_for(turns, gt, log):
    return r14_run.collect_metrics(turns, gt, log, bucket_size=100)


def atag_drift_for(log, n_turns):
    return r14_run.atag_drift(log, n_turns=n_turns, bucket_size=100)


def grade_qa(qs, idx, cache, budget, *, answer_fn) -> dict:
    answers = {}
    for q in qs:
        a = answer_fn(q.question, idx, cache, budget, top_k=14)
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


def telemetry_summary(telemetry: list[dict]) -> dict:
    if not telemetry:
        return {}
    n = len(telemetry)
    avg_heads = sum(t["n_active_state_heads"] for t in telemetry) / n
    avg_active_chars = sum(t["active_state_chars"] for t in telemetry) / n
    avg_prompt_chars = sum(t["prompt_chars"] for t in telemetry) / n
    max_heads = max(t["n_active_state_heads"] for t in telemetry)
    max_active_chars = max(t["active_state_chars"] for t in telemetry)
    return {
        "n_batches": n,
        "avg_active_state_heads": avg_heads,
        "max_active_state_heads": max_heads,
        "avg_active_state_chars": avg_active_chars,
        "max_active_state_chars": max_active_chars,
        "avg_prompt_chars": avg_prompt_chars,
    }


def run_active_variant(
    variant_name: str,
    max_active_state_size: int,
    turns,
    gt,
    qs,
    budget: Budget,
    *,
    do_qa: bool,
):
    cache_path = CACHE_DIR / f"{variant_name}.json"
    cache = Cache(cache_path)
    pairs = [(t.idx, t.text) for t in turns]
    print(f"\n=== variant: {variant_name} (cap={max_active_state_size}) ===")
    print(f"[ingest] {len(pairs)} turns, batch_size=5")
    try:
        log, idx, telemetry = aen1_active.ingest_turns(
            pairs,
            cache,
            budget,
            batch_size=5,
            rebuild_index_every=4,
            max_active_state_size=max_active_state_size,
        )
    except RuntimeError as e:
        print(f"!!! Budget stop during ingest: {e}")
        cache.save()
        raise
    cache.save()
    print(f"[ingest] log size: {len(log)}  supersede_heads: {len(idx.supersede_head)}")
    print(
        f"[budget] cost=${budget.cost():.3f} llm={budget.llm_calls} "
        f"embed={budget.embed_calls}"
    )

    metrics = collect_metrics_for(turns, gt, log)
    drift = atag_drift_for(log, n_turns=len(turns))

    print(
        f"[metrics] ref_emission_rate (overall non-first): "
        f"{metrics['summary']['ref_emission_rate']:.3f}"
    )
    print(
        f"[metrics] ref_correctness_rate (overall non-first): "
        f"{metrics['summary']['ref_correctness_rate']:.3f}"
    )
    print(
        f"[metrics] entry_emission_rate (overall non-first): "
        f"{metrics['summary']['entry_emission_rate']:.3f}"
    )
    print("[metrics] bucket curve:")
    for b in metrics["summary"]["bucket_stats"]:
        rate_e = b["ref_emission_rate"]
        rate_c = b["ref_correctness_rate"]
        s_e = f"{rate_e:.2f}" if rate_e is not None else " -- "
        s_c = f"{rate_c:.2f}" if rate_c is not None else " -- "
        print(
            f"  {b['range']:>14s}  trans={b['n_transitions']:>3d}  "
            f"emit={s_e}  correct={s_c}"
        )

    tele_sum = telemetry_summary(telemetry)
    print(
        f"[telemetry] avg active-state heads: "
        f"{tele_sum['avg_active_state_heads']:.1f}  "
        f"avg chars: {tele_sum['avg_active_state_chars']:.0f}  "
        f"max chars: {tele_sum['max_active_state_chars']}"
    )

    qa = None
    if do_qa:
        try:
            print(f"\n[QA] running {len(qs)} questions...")
            qa = grade_qa(qs, idx, cache, budget, answer_fn=aen1_active.answer_question)
            print(f"[QA] deterministic pass: {qa['deterministic_pass']}/{qa['total']}")
            print(f"[QA] judge-graded pass: {qa['judge_pass']}/{qa['total']}")
        except RuntimeError as e:
            print(f"!!! Budget stop during QA: {e}")
            cache.save()

    return {
        "variant": variant_name,
        "max_active_state_size": max_active_state_size,
        "log_size": len(log),
        "num_supersede_heads": len(idx.supersede_head),
        "metrics_summary": metrics["summary"],
        "tag_drift": drift,
        "transitions": metrics["transitions"],
        "telemetry_summary": tele_sum,
        "telemetry_per_batch": telemetry,
        "qa": qa,
    }


def load_round14_baseline() -> dict:
    p = ROUND14 / "results" / "run.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return {
        "variant": "aen1_simple_round14_baseline",
        "n_turns": d["n_turns"],
        "log_size": d["log_size"],
        "num_supersede_heads": d.get("num_supersede_heads"),
        "metrics_summary": d["metrics_summary"],
        "qa": {
            "deterministic_pass": d["qa_deterministic_pass"],
            "judge_pass": d["qa_judge_pass"],
            "total": d["qa_total"],
            "verdicts": d.get("verdicts"),
            "answers": d.get("answers"),
        },
    }


def main() -> None:
    # Hard cap 400 LLM, 50 embed, $4. Target $2.
    budget = Budget(max_llm=420, max_embed=60, stop_at_llm=400, stop_at_embed=50)

    turns = dense_chains.generate()
    gt = dense_chains.ground_truth(turns)
    qs = dense_chains.build_questions(gt)
    print(f"[scenario] turns={len(turns)}  questions={len(qs)}")
    n_nf = sum(max(0, len(v) - 1) for v in gt.chains.values())
    print(f"  non-first transitions: {n_nf}")

    results = {
        "scenario": {
            "name": "dense_chains",
            "n_turns": len(turns),
            "n_questions": len(qs),
            "n_non_first_transitions": n_nf,
        },
        "baseline_round14": load_round14_baseline(),
        "active_variants": {},
        "budget": {},
    }

    # Order: cap=100 (primary) first with full QA, then cap=200 writer-only,
    # then cap=50 writer-only.
    plan = [
        ("aen1_active_cap100", 100, True),
        ("aen1_active_cap200", 200, False),
        ("aen1_active_cap50", 50, False),
    ]

    for variant_name, cap, do_qa in plan:
        try:
            res = run_active_variant(
                variant_name, cap, turns, gt, qs, budget, do_qa=do_qa
            )
            results["active_variants"][variant_name] = res
        except RuntimeError as e:
            print(f"\n!!! Skipping {variant_name} due to budget stop: {e}")
            results["active_variants"][variant_name] = {
                "skipped": True,
                "reason": str(e),
            }
            # Stop the plan; don't try further variants
            break
        # Save after each variant
        out = RESULTS_DIR / "run.json"
        results["budget"] = {
            "cost": budget.cost(),
            "llm_calls": budget.llm_calls,
            "embed_calls": budget.embed_calls,
        }
        out.write_text(json.dumps(results, indent=2, default=str))
        print(
            f"[checkpoint] wrote {out} (cost=${budget.cost():.3f}, "
            f"llm={budget.llm_calls})"
        )

    results["budget"] = {
        "cost": budget.cost(),
        "llm_calls": budget.llm_calls,
        "embed_calls": budget.embed_calls,
    }
    out = RESULTS_DIR / "run.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[done] wrote {out}")
    print(
        f"[done] cost=${budget.cost():.3f} llm={budget.llm_calls} "
        f"embed={budget.embed_calls}"
    )


if __name__ == "__main__":
    main()
