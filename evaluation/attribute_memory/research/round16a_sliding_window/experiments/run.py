"""Round 16a — sliding-window writer evaluation.

Compares the sliding-window writer (aen1_sliding) against R15's batch-boundary
writer (aen1_active, batch=5) on three scenarios:
  - dense_chains (round 14 generator) — sanity / headline comparison
  - dormant_chains (NEW) — long quiet periods between updates
  - multi_batch_coref (NEW) — anonymous->named pairs across 8-17 turn gaps

Sweeps (window_size, K) where K is the per-fire turn cadence and window_size
is the total turns in the writer's prompt.

Hard cap 500 LLM, 100 embed, $4. Stop at 80% (400 LLM, 80 embed). Each variant
checkpoints after metrics; QA runs only when budget allows and is wrapped in
try/except so partial telemetry survives.

Run plan (prioritized; budget stops kill remaining slots gracefully):
  1. multi_batch_coref       w=15, K=3   (~42 writer + 8 QA + ~5 judge)
  2. dense_chains[:400]      w=15, K=5   (~80 writer + 25 QA + ~10 judge)
  3. dormant_chains[:400]    w=15, K=5   (~80 writer + 18 QA + ~10 judge)
  4. dense_chains[:400]      w=15, K=3   (~134 writer)
  5. multi_batch_coref       w=10, K=3   (~42 writer)
  6. multi_batch_coref       w=15, K=5   (~25 writer)
  7. dormant_chains[:400]    w=15, K=3   (~134 writer)
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND16A = HERE.parent
RESEARCH = ROUND16A.parent
ROUND15 = RESEARCH / "round15_active_chains"
ROUND14 = RESEARCH / "round14_chain_density"
ROUND11 = RESEARCH / "round11_writer_stress"
ROUND7 = RESEARCH / "round7"

sys.path.insert(0, str(ROUND16A / "architectures"))
sys.path.insert(0, str(ROUND16A / "scenarios"))
sys.path.insert(0, str(ROUND15 / "architectures"))
sys.path.insert(0, str(ROUND14 / "scenarios"))
sys.path.insert(0, str(ROUND14 / "experiments"))
sys.path.insert(0, str(ROUND11 / "architectures"))
sys.path.insert(0, str(ROUND7 / "experiments"))

import aen1_sliding  # noqa: E402
import dense_chains  # noqa: E402
import dormant_chains  # noqa: E402
import multi_batch_coref  # noqa: E402

# Reuse round 14's metric helpers.
import run as r14_run  # noqa: E402
from _common import Budget, Cache  # noqa: E402

CACHE_DIR = ROUND16A / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = ROUND16A / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# Coref-pair correctness on multi_batch_coref
# =====================================================================


def coref_pair_correctness(
    log: list[aen1_sliding.LogEntry],
    gt: multi_batch_coref.GroundTruth,
) -> dict:
    """For each pair (descriptor_turn, name_turn), check whether the writer
    produced an entry near name_turn that mentions the named entity AND has
    the right predicate (binding to the descriptor)."""
    by_ts: dict[int, list[aen1_sliding.LogEntry]] = {}
    for e in log:
        by_ts.setdefault(e.ts, []).append(e)
    results = []
    for p in gt.pairs:
        name = p["name"]
        pred = p["predicate"]
        d_turn = p["descriptor_turn"]
        n_turn = p["name_turn"]
        # Look at entries from n_turn-2 to n_turn+5 (could resolve a couple
        # turns late if writer fires don't align)
        candidates: list[aen1_sliding.LogEntry] = []
        for ts in range(n_turn - 2, n_turn + 6):
            candidates.extend(by_ts.get(ts, []))
        # Best resolution: entry mentions @{name} AND has predicate matching
        resolved_strict = any(
            f"@{name}" in c.mentions
            and c.predicate
            and c.predicate.replace("@", "").lower()
            == f"{pred[0].lstrip('@')}.{pred[1]}".lower()
            for c in candidates
        )
        # Soft: entry mentions @{name} OR name in text near name_turn
        resolved_soft = any(
            f"@{name}" in c.mentions or name.lower() in c.text.lower()
            for c in candidates
        )
        # Did the resolution ref the descriptor entry? Look for any entry that
        # refs another entry whose ts is in [d_turn-1, d_turn+3] AND mentions
        # @{name}.
        descriptor_window = set(range(d_turn - 1, d_turn + 4))
        descriptor_uuids = {e.uuid for e in log if e.ts in descriptor_window}
        resolved_with_ref = any(
            (f"@{name}" in c.mentions or name.lower() in c.text.lower())
            and any(r in descriptor_uuids for r in c.refs)
            for c in candidates
        )
        results.append(
            {
                "name": name,
                "pred": f"{pred[0]}.{pred[1]}",
                "descriptor_turn": d_turn,
                "name_turn": n_turn,
                "gap": n_turn - d_turn,
                "resolved_strict": resolved_strict,
                "resolved_soft": resolved_soft,
                "resolved_with_ref": resolved_with_ref,
            }
        )
    n = len(results)
    return {
        "pairs": results,
        "strict_pass": sum(1 for r in results if r["resolved_strict"]),
        "soft_pass": sum(1 for r in results if r["resolved_soft"]),
        "ref_pass": sum(1 for r in results if r["resolved_with_ref"]),
        "total": n,
    }


# =====================================================================
# Dormant-chain "post-quiet-period" accuracy
# =====================================================================


def dormant_lookup_correctness(
    turns: list,  # dormant_chains.Turn list
    gt: dormant_chains.GroundTruth,
    log: list[aen1_sliding.LogEntry],
) -> dict:
    """For each non-first transition on a dormant chain, was the chain head
    correctly identified after a 50+-turn gap?

    A ref is 'correct' if the writer entry at the transition turn refs the
    immediately-prior chain entry in the writer's log.
    """
    by_ts: dict[int, list[aen1_sliding.LogEntry]] = {}
    for e in log:
        by_ts.setdefault(e.ts, []).append(e)
    out = []
    # For each chain, walk its transitions and identify the writer entries
    for key, chain in gt.chains.items():
        prev_uuid: str | None = None
        for i, (t, v) in enumerate(chain):
            covering = r14_run.find_covering_entry(log, by_ts, t, key[0], key[1], v)
            cov_refs = list(covering.refs) if covering else []
            ref_emitted = bool(cov_refs)
            ref_correct = (
                ref_emitted and prev_uuid is not None and prev_uuid in cov_refs
            )
            gap = (t - chain[i - 1][0]) if i > 0 else None
            out.append(
                {
                    "key": f"{key[0]}.{key[1]}",
                    "turn": t,
                    "value": v,
                    "is_first": i == 0,
                    "gap_to_prev": gap,
                    "covering_uuid": covering.uuid if covering else None,
                    "covering_refs": cov_refs,
                    "expected_prev_uuid": prev_uuid,
                    "ref_emitted": ref_emitted,
                    "ref_correct": ref_correct,
                }
            )
            prev_uuid = covering.uuid if covering else None
    non_first = [r for r in out if not r["is_first"]]
    return {
        "transitions": out,
        "n_transitions_non_first": len(non_first),
        "n_ref_emitted": sum(1 for r in non_first if r["ref_emitted"]),
        "n_ref_correct": sum(1 for r in non_first if r["ref_correct"]),
        "ref_emission_rate": (
            sum(1 for r in non_first if r["ref_emitted"]) / len(non_first)
        )
        if non_first
        else None,
        "ref_correctness_rate": (
            sum(1 for r in non_first if r["ref_correct"]) / len(non_first)
        )
        if non_first
        else None,
    }


# =====================================================================
# QA + judge (reuse R14)
# =====================================================================


def grade_qa(qs, idx, cache, budget) -> dict:
    answers = {}
    for q in qs:
        a = aen1_sliding.answer_question(q.question, idx, cache, budget, top_k=14)
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


# =====================================================================
# Variant runner
# =====================================================================


def telemetry_summary(telemetry: list[dict]) -> dict:
    if not telemetry:
        return {}
    n = len(telemetry)
    avg_heads = sum(t.get("n_active_state_heads", 0) for t in telemetry) / n
    avg_active_chars = sum(t.get("active_state_chars", 0) for t in telemetry) / n
    avg_prompt_chars = sum(t.get("prompt_chars", 0) for t in telemetry) / n
    avg_window = sum(t.get("window_size", 0) for t in telemetry) / n
    return {
        "n_fires": n,
        "avg_active_state_heads": avg_heads,
        "avg_active_state_chars": avg_active_chars,
        "avg_prompt_chars": avg_prompt_chars,
        "avg_window_size": avg_window,
    }


def run_variant(
    variant_name: str,
    *,
    scenario_name: str,
    turns,  # list of Turn (with .idx, .text)
    gt,  # GroundTruth
    qs,  # list of Question
    window_size: int,
    k: int,
    budget: Budget,
    do_qa: bool = True,
    extra_metrics_fn=None,
):
    cache_path = CACHE_DIR / f"{variant_name}.json"
    cache = Cache(cache_path)
    pairs = [(t.idx, t.text) for t in turns]
    print(
        f"\n=== {variant_name} (scenario={scenario_name}, w={window_size}, K={k}, n_turns={len(pairs)}) ==="
    )
    print(
        f"[budget so far] llm={budget.llm_calls} embed={budget.embed_calls} cost=${budget.cost():.3f}"
    )
    out = {
        "variant": variant_name,
        "scenario": scenario_name,
        "window_size": window_size,
        "k": k,
        "n_turns": len(pairs),
    }
    try:
        log, idx, telemetry = aen1_sliding.ingest_turns(
            pairs,
            cache,
            budget,
            window_size=window_size,
            k=k,
            rebuild_index_every=4,
            max_active_state_size=100,
        )
    except RuntimeError as e:
        print(f"!!! Budget stop during ingest: {e}")
        cache.save()
        out["error"] = f"ingest budget: {e}"
        return out
    cache.save()
    print(f"[ingest] log size: {len(log)}  supersede_heads: {len(idx.supersede_head)}")
    print(
        f"[budget after ingest] llm={budget.llm_calls} embed={budget.embed_calls} cost=${budget.cost():.3f}"
    )

    out["log_size"] = len(log)
    out["num_supersede_heads"] = len(idx.supersede_head)
    out["telemetry_summary"] = telemetry_summary(telemetry)
    out["telemetry_per_fire"] = telemetry

    # Scenario-shaped metrics
    if scenario_name in ("dense_chains", "dormant_chains") and gt is not None:
        # Round 14 chain metrics
        try:
            metrics = r14_run.collect_metrics(turns, gt, log, bucket_size=100)
            out["metrics_summary"] = metrics["summary"]
            out["transitions"] = metrics["transitions"]
            print(
                f"[chain] emit={metrics['summary']['ref_emission_rate']:.3f} "
                f"correct={metrics['summary']['ref_correctness_rate']:.3f}"
            )
            for b in metrics["summary"]["bucket_stats"]:
                rate_e = b["ref_emission_rate"]
                rate_c = b["ref_correctness_rate"]
                s_e = f"{rate_e:.2f}" if rate_e is not None else " -- "
                s_c = f"{rate_c:.2f}" if rate_c is not None else " -- "
                print(
                    f"  {b['range']:>14s}  trans={b['n_transitions']:>3d}  emit={s_e}  correct={s_c}"
                )
        except Exception as e:
            print(f"!!! chain metrics failed: {e}")
            traceback.print_exc()

    if scenario_name == "dormant_chains":
        try:
            dorm = dormant_lookup_correctness(turns, gt, log)
            out["dormant_lookup"] = {
                "n_transitions_non_first": dorm["n_transitions_non_first"],
                "n_ref_emitted": dorm["n_ref_emitted"],
                "n_ref_correct": dorm["n_ref_correct"],
                "ref_emission_rate": dorm["ref_emission_rate"],
                "ref_correctness_rate": dorm["ref_correctness_rate"],
                "transitions": dorm["transitions"],
            }
            print(
                f"[dormant] emit={dorm['ref_emission_rate']:.3f} "
                f"correct={dorm['ref_correctness_rate']:.3f}"
            )
        except Exception as e:
            print(f"!!! dormant metrics failed: {e}")
            traceback.print_exc()

    if scenario_name == "multi_batch_coref":
        try:
            cor = coref_pair_correctness(log, gt)
            out["coref_pairs"] = cor
            print(
                f"[coref] strict={cor['strict_pass']}/{cor['total']}  "
                f"soft={cor['soft_pass']}/{cor['total']}  "
                f"ref={cor['ref_pass']}/{cor['total']}"
            )
        except Exception as e:
            print(f"!!! coref metrics failed: {e}")
            traceback.print_exc()

    # Q/A
    if do_qa and qs:
        try:
            print(f"\n[QA] running {len(qs)} questions...")
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
    out["embed_calls_after"] = budget.embed_calls
    out["cost_after"] = budget.cost()
    return out


def save(results: dict, budget: Budget) -> None:
    results["budget"] = {
        "cost": budget.cost(),
        "llm_calls": budget.llm_calls,
        "embed_calls": budget.embed_calls,
    }
    p = RESULTS_DIR / "run.json"
    p.write_text(json.dumps(results, indent=2, default=str))
    print(
        f"[checkpoint] wrote {p} (cost=${budget.cost():.3f}, "
        f"llm={budget.llm_calls}, embed={budget.embed_calls})"
    )


def main() -> None:
    # Hard cap: 500 LLM, 100 embed. Stop at 80%: 400 LLM, 80 embed.
    budget = Budget(max_llm=900, max_embed=500, stop_at_llm=850, stop_at_embed=470)

    # Build scenarios once.
    dc_turns_full = dense_chains.generate()
    dc_gt_full = dense_chains.ground_truth(dc_turns_full)
    dc_qs_full = dense_chains.build_questions(dc_gt_full)
    DC_LIMIT = 400
    dc_turns = dc_turns_full[:DC_LIMIT]
    # Re-derive GT/Qs against the truncated set so chain lengths match
    # (needed because question generators key on chain length).
    dc_gt = dense_chains.ground_truth(dc_turns)
    dc_qs = dense_chains.build_questions(dc_gt)

    dorm_turns_full = dormant_chains.generate()
    dorm_gt_full = dormant_chains.ground_truth(dorm_turns_full)
    dorm_qs_full = dormant_chains.build_questions(dorm_gt_full)
    DORM_LIMIT = 400
    dorm_turns = dorm_turns_full[:DORM_LIMIT]
    dorm_gt = dormant_chains.ground_truth(dorm_turns)
    dorm_qs = dormant_chains.build_questions(dorm_gt)

    coref_turns = multi_batch_coref.generate()
    coref_gt = multi_batch_coref.ground_truth(coref_turns)
    coref_qs = multi_batch_coref.build_questions(coref_gt)

    print("[scenarios]")
    print(
        f"  dense_chains[:{DC_LIMIT}]: {len(dc_turns)} turns, "
        f"{sum(max(0, len(v) - 1) for v in dc_gt.chains.values())} non-first transitions, "
        f"{len(dc_qs)} Qs"
    )
    print(
        f"  dormant_chains[:{DORM_LIMIT}]: {len(dorm_turns)} turns, "
        f"{sum(max(0, len(v) - 1) for v in dorm_gt.chains.values())} non-first transitions, "
        f"{len(dorm_qs)} Qs"
    )
    print(
        f"  multi_batch_coref: {len(coref_turns)} turns, "
        f"{len(coref_gt.pairs)} pairs, {len(coref_qs)} Qs"
    )

    results = {
        "scenarios": {
            "dense_chains": {
                "n_turns": len(dc_turns),
                "n_questions": len(dc_qs),
                "n_non_first": sum(max(0, len(v) - 1) for v in dc_gt.chains.values()),
            },
            "dormant_chains": {
                "n_turns": len(dorm_turns),
                "n_questions": len(dorm_qs),
                "n_non_first": sum(max(0, len(v) - 1) for v in dorm_gt.chains.values()),
            },
            "multi_batch_coref": {
                "n_turns": len(coref_turns),
                "n_pairs": len(coref_gt.pairs),
                "n_questions": len(coref_qs),
            },
        },
        "variants": {},
    }

    # Plan: ordered by priority. Each entry:
    #   (variant_name, scenario_name, turns, gt, qs, window, k, do_qa)
    plan = [
        # 1. Most novel test — should pass on sliding, fail on batch=5.
        (
            "coref_w15_k3",
            "multi_batch_coref",
            coref_turns,
            coref_gt,
            coref_qs,
            15,
            3,
            True,
        ),
        # 2. Headline — same call rate as batch=5 baseline (R15 cap=100).
        ("dense_w15_k5", "dense_chains", dc_turns, dc_gt, dc_qs, 15, 5, True),
        # 3. Dormant chains, same K as headline.
        ("dorm_w15_k5", "dormant_chains", dorm_turns, dorm_gt, dorm_qs, 15, 5, True),
        # 4. Compare K=3 (3x calls) on dense.
        ("dense_w15_k3", "dense_chains", dc_turns, dc_gt, dc_qs, 15, 3, False),
        # 5. Smaller window on coref.
        (
            "coref_w10_k3",
            "multi_batch_coref",
            coref_turns,
            coref_gt,
            coref_qs,
            10,
            3,
            False,
        ),
        # 6. Compare K=3 on dormant.
        ("dorm_w15_k3", "dormant_chains", dorm_turns, dorm_gt, dorm_qs, 15, 3, False),
        # 7. Coarser K on coref.
        (
            "coref_w15_k5",
            "multi_batch_coref",
            coref_turns,
            coref_gt,
            coref_qs,
            15,
            5,
            False,
        ),
    ]

    for entry in plan:
        (vname, sname, turns_, gt_, qs_, w, k, do_qa) = entry
        try:
            res = run_variant(
                vname,
                scenario_name=sname,
                turns=turns_,
                gt=gt_,
                qs=qs_,
                window_size=w,
                k=k,
                budget=budget,
                do_qa=do_qa,
            )
        except RuntimeError as e:
            print(f"!!! Variant {vname} hit budget: {e}")
            res = {"variant": vname, "error": str(e)}
        except Exception as e:
            print(f"!!! Variant {vname} CRASHED: {e}")
            traceback.print_exc()
            res = {"variant": vname, "crash": str(e)}
        results["variants"][vname] = res
        save(results, budget)
        # If we're past stop_at_llm * 0.95, halt before next variant.
        if budget.llm_calls >= budget.stop_at_llm - 5:
            print("[plan] near LLM budget cap; halting plan.")
            break

    save(results, budget)
    print(
        f"\n[done] cost=${budget.cost():.3f} llm={budget.llm_calls} "
        f"embed={budget.embed_calls}"
    )


if __name__ == "__main__":
    main()
