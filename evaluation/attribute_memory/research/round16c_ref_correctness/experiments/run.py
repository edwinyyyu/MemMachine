"""Round 16c — ref-correctness fix. Compares:
  - aen1_active cap=100 (round 15 baseline, loaded from results/run.json)
  - aen1_active_v2 cap=100 (post-hoc deterministic relinker + clarify-skip;
    writer prompt + cache UNCHANGED)
  - aen1_active_v3 cap=100 (v3 writer prompt with predicate-discipline
    rules + deterministic relinker; fresh writer LLM calls)

Budget plan:
  Round 15 already at $0.48. v2 uses round-15 writer cache (only QA/judge
  + a few embeds new). v3 needs a fresh writer pass (~149 calls) + QA/judge.
  Hard cap: 350 LLM, 50 embed -> $3. Stop at 80%: 280/40.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND16C = HERE.parent
RESEARCH = ROUND16C.parent
ROUND15 = RESEARCH / "round15_active_chains"
ROUND14 = RESEARCH / "round14_chain_density"
ROUND11 = RESEARCH / "round11_writer_stress"
ROUND7 = RESEARCH / "round7"
sys.path.insert(0, str(ROUND16C / "architectures"))
sys.path.insert(0, str(ROUND15 / "architectures"))
sys.path.insert(0, str(ROUND14 / "scenarios"))
sys.path.insert(0, str(ROUND14 / "experiments"))
sys.path.insert(0, str(ROUND11 / "architectures"))
sys.path.insert(0, str(ROUND7 / "experiments"))

import aen1_active_v2  # noqa: E402
import aen1_active_v3  # noqa: E402
import dense_chains  # noqa: E402
import run as r14_run  # noqa: E402
from _common import Budget, Cache  # noqa: E402

CACHE_DIR = ROUND16C / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = ROUND16C / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# v2 reuses round-15's cap=100 writer cache verbatim; we copy it once.
SEED_CACHE = ROUND15 / "cache" / "aen1_active_cap100.json"


def seed_cache(name: str) -> Path:
    dest = CACHE_DIR / f"{name}.json"
    if not dest.exists() and SEED_CACHE.exists():
        dest.write_bytes(SEED_CACHE.read_bytes())
    return dest


def grade_qa(qs, idx, cache, budget, *, answer_fn):
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


def run_variant(
    name: str,
    arch_module,
    turns,
    gt,
    qs,
    budget,
    *,
    skip_clarify: bool,
    normalize: bool,
    do_qa: bool,
    seed_from_round15: bool,
):
    if seed_from_round15:
        cache_path = seed_cache(name)
    else:
        cache_path = CACHE_DIR / f"{name}.json"
    cache = Cache(cache_path)
    pairs = [(t.idx, t.text) for t in turns]
    print(
        f"\n=== variant: {name} (skip_clarify={skip_clarify}, "
        f"normalize={normalize}) ==="
    )
    print(f"[ingest] {len(pairs)} turns, batch_size=5")
    log, idx, telemetry = arch_module.ingest_turns(
        pairs,
        cache,
        budget,
        batch_size=5,
        rebuild_index_every=4,
        max_active_state_size=100,
        skip_clarify=skip_clarify,
        normalize=normalize,
    )
    cache.save()
    print(f"[ingest] log size: {len(log)}  supersede_heads: {len(idx.supersede_head)}")
    print(
        f"[budget] cost=${budget.cost():.3f} llm={budget.llm_calls} "
        f"embed={budget.embed_calls}"
    )

    metrics = r14_run.collect_metrics(turns, gt, log, bucket_size=100)
    drift = r14_run.atag_drift(log, n_turns=len(turns), bucket_size=100)
    s = metrics["summary"]
    print(
        f"[metrics] ref_emission_rate={s['ref_emission_rate']:.3f}  "
        f"ref_correctness_rate={s['ref_correctness_rate']:.3f}  "
        f"entry_emission_rate={s['entry_emission_rate']:.3f}"
    )
    print("[metrics] bucket curve:")
    for b in s["bucket_stats"]:
        rate_e = b["ref_emission_rate"]
        rate_c = b["ref_correctness_rate"]
        s_e = f"{rate_e:.2f}" if rate_e is not None else " -- "
        s_c = f"{rate_c:.2f}" if rate_c is not None else " -- "
        print(
            f"  {b['range']:>14s}  trans={b['n_transitions']:>3d}  "
            f"emit={s_e}  correct={s_c}"
        )

    qa = None
    if do_qa:
        try:
            print(f"\n[QA] running {len(qs)} questions...")
            qa = grade_qa(qs, idx, cache, budget, answer_fn=arch_module.answer_question)
            print(f"[QA] deterministic pass: {qa['deterministic_pass']}/{qa['total']}")
            print(f"[QA] judge-graded pass: {qa['judge_pass']}/{qa['total']}")
        except RuntimeError as e:
            print(f"!!! Budget stop during QA: {e}")
            cache.save()

    return {
        "variant": name,
        "skip_clarify": skip_clarify,
        "normalize": normalize,
        "log_size": len(log),
        "num_supersede_heads": len(idx.supersede_head),
        "metrics_summary": s,
        "tag_drift": drift,
        "transitions": metrics["transitions"],
        "qa": qa,
    }


def main():
    # Hard cap 350 LLM, 50 embed = ~$1.05 LLM + small embed = ~$1.05.
    # 80% stop = 280 LLM, 40 embed.
    budget = Budget(max_llm=350, max_embed=50, stop_at_llm=280, stop_at_embed=40)

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
        "variants": {},
        "budget": {},
    }

    # Load round-15 baseline
    r15 = ROUND15 / "results" / "run.json"
    if r15.exists():
        d = json.loads(r15.read_text())
        b = d["active_variants"]["aen1_active_cap100"]
        results["baseline_round15_cap100"] = {
            "metrics_summary": b["metrics_summary"],
            "qa": b.get("qa"),
            "log_size": b.get("log_size"),
        }

    plan = [
        # v2: writer prompt unchanged, cache reused, post-hoc deterministic
        # relink only.
        ("aen1_active_v2_full", aen1_active_v2, True, False, True, True),
        # v3: predicate-discipline writer prompt + deterministic relink.
        # Fresh writer cache (we still seed from round-15 to capture any
        # accidentally-shared prompts; v3 prompt is different so most
        # writer calls miss).
        ("aen1_active_v3_full", aen1_active_v3, True, False, True, True),
    ]

    for name, mod, sc, nm, do_qa, seed in plan:
        try:
            res = run_variant(
                name,
                mod,
                turns,
                gt,
                qs,
                budget,
                skip_clarify=sc,
                normalize=nm,
                do_qa=do_qa,
                seed_from_round15=seed,
            )
            results["variants"][name] = res
        except RuntimeError as e:
            print(f"\n!!! Skipping {name} due to budget stop: {e}")
            results["variants"][name] = {"skipped": True, "reason": str(e)}
            break
        results["budget"] = {
            "cost": budget.cost(),
            "llm_calls": budget.llm_calls,
            "embed_calls": budget.embed_calls,
        }
        out = RESULTS_DIR / "run.json"
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
