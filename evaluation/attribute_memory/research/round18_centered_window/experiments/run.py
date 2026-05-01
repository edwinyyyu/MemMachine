"""Round 18 — K-BLOCK CENTERED-WINDOW writer.

Generalization of the per-turn centered writer: window = w_past + K + w_future,
target block of K turns at the MIDDLE, slide by K. Cost = N/K LLM calls.

Natural-conversation lookahead reasoning:
  Working memory holds ~5-10 conversational turns; conversational repair
  ("oh by the way, his name is Marcus") typically lands within 3-7 turns of
  the introduction. Beyond that, listeners drop earlier threads. So:

    w_past  = 7  (no fudge — past is committed/observed; bounded only by prompt)
    w_future depends on fudge factor:
      - 1x natural: w_future = 7   (gap≤7 from name-turn writer's POV)
      - 2x fudge:   w_future = 14  (gap≤14, captures team/colleague/Priya/
                                    Sana/Alice/Theo/Nadia in multi_batch_coref)

  K=3 amortizes 3 turns per LLM call.

  Pair gaps in multi_batch_coref:
    Marcus     gap=17  (only K=3 + w_past=7 + w_future=14 PLUS K-block boost
                        captures it: name at block-end sees [N-w_past-K+1..N+w_future]
                        = up to gap K-1+w_past+w_future = 2+7+14 = 23)
    Theo       gap=14  (fits w_future=14)
    platform   gap=11  (fits both 7 and 14 once K-block boost is included)
    Alice      gap=13  (fits w_future=14, borderline at 7)
    Nadia      gap=12  (fits w_future=14)
    Priya      gap=9   (fits both with K-block boost)
    Sana       gap=8   (fits both with K-block boost)
    Quentin    gap=16  (fits w_future=14 + K-block boost: 16 ≤ 14+2=16)

Run plan (~42 fires per variant + 8 QA + judge):
  1. coref_centered_K3_w6_w6   (CONTROLLED A/B vs round 16a coref_w15_k3:
                                 same K=3, same total window=15, only the
                                 target is moved from END→MIDDLE. Round 16a
                                 baseline scored strict=1/8, ref=7/8 here.)
  2. coref_centered_K3_w7_w14  (2x natural-lookahead fudge factor on w_future,
                                 total window=24, captures longer-gap pairs)
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND18 = HERE.parent
RESEARCH = ROUND18.parent
ROUND16A = RESEARCH / "round16a_sliding_window"
ROUND15 = RESEARCH / "round15_active_chains"
ROUND14 = RESEARCH / "round14_chain_density"
ROUND11 = RESEARCH / "round11_writer_stress"
ROUND7 = RESEARCH / "round7"

sys.path.insert(0, str(ROUND18 / "architectures"))
sys.path.insert(0, str(ROUND16A / "architectures"))
sys.path.insert(0, str(ROUND16A / "scenarios"))
sys.path.insert(0, str(ROUND15 / "architectures"))
sys.path.insert(0, str(ROUND14 / "scenarios"))
sys.path.insert(0, str(ROUND14 / "experiments"))
sys.path.insert(0, str(ROUND11 / "architectures"))
sys.path.insert(0, str(ROUND7 / "experiments"))

import importlib.util  # noqa: E402

import aen1_centered  # noqa: E402
import multi_batch_coref  # noqa: E402
from _common import Budget, Cache  # noqa: E402


def _load_module(alias: str, path: Path):
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# Round 14 and 16a both name their experiment file `run.py`; load each by path.
r16a_run = _load_module("r16a_run", ROUND16A / "experiments" / "run.py")
r14_run = _load_module("r14_run", ROUND14 / "experiments" / "run.py")

CACHE_DIR = ROUND18 / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = ROUND18 / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def telemetry_summary(telemetry: list[dict]) -> dict:
    if not telemetry:
        return {}
    n = len(telemetry)
    return {
        "n_fires": n,
        "avg_active_state_heads": sum(
            t.get("n_active_state_heads", 0) for t in telemetry
        )
        / n,
        "avg_active_state_chars": sum(t.get("active_state_chars", 0) for t in telemetry)
        / n,
        "avg_prompt_chars": sum(t.get("prompt_chars", 0) for t in telemetry) / n,
        "avg_window_size": sum(t.get("window_size", 0) for t in telemetry) / n,
        "avg_w_past_actual": sum(t.get("w_past_actual", 0) for t in telemetry) / n,
        "avg_w_future_actual": sum(t.get("w_future_actual", 0) for t in telemetry) / n,
    }


def run_variant(
    variant_name: str,
    *,
    turns,
    gt,
    qs,
    w_past: int,
    w_future: int,
    k: int,
    budget: Budget,
    do_qa: bool = True,
):
    cache_path = CACHE_DIR / f"{variant_name}.json"
    cache = Cache(cache_path)
    pairs = [(t.idx, t.text) for t in turns]
    total_w = w_past + k + w_future
    print(
        f"\n=== {variant_name} (multi_batch_coref, w_past={w_past}, K={k}, "
        f"w_future={w_future}, total_window={total_w}, n_turns={len(pairs)}) ==="
    )
    print(
        f"[budget so far] llm={budget.llm_calls} embed={budget.embed_calls} "
        f"cost=${budget.cost():.3f}"
    )
    out = {
        "variant": variant_name,
        "scenario": "multi_batch_coref",
        "w_past": w_past,
        "w_future": w_future,
        "k": k,
        "total_window": total_w,
        "n_turns": len(pairs),
    }
    try:
        log, idx, telemetry = aen1_centered.ingest_turns(
            pairs,
            cache,
            budget,
            w_past=w_past,
            w_future=w_future,
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
        f"[budget after ingest] llm={budget.llm_calls} embed={budget.embed_calls} "
        f"cost=${budget.cost():.3f}"
    )
    out["log_size"] = len(log)
    out["num_supersede_heads"] = len(idx.supersede_head)
    out["telemetry_summary"] = telemetry_summary(telemetry)
    out["telemetry_per_fire"] = telemetry

    # Coref-pair correctness (reuse R16a metric)
    try:
        cor = r16a_run.coref_pair_correctness(log, gt)
        out["coref_pairs"] = cor
        print(
            f"[coref] strict={cor['strict_pass']}/{cor['total']}  "
            f"soft={cor['soft_pass']}/{cor['total']}  "
            f"ref={cor['ref_pass']}/{cor['total']}"
        )
        for r in cor["pairs"]:
            print(
                f"  {r['name']:<10s} pred={r['pred']:<25s} gap={r['gap']:>3d}  "
                f"strict={r['resolved_strict']!s:<5s} ref={r['resolved_with_ref']!s:<5s}"
            )
    except Exception as e:
        print(f"!!! coref metrics failed: {e}")
        traceback.print_exc()

    # Q/A
    if do_qa and qs:
        try:
            print(f"\n[QA] running {len(qs)} questions...")
            qa = r16a_run.grade_qa(qs, idx, cache, budget)
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
    # K=3 writer: ~42 fires per variant on the 126-turn coref scenario.
    # Two variants -> ~85 ingest + ~16 QA + ~30 judge = ~130 LLM calls total.
    budget = Budget(max_llm=400, max_embed=200, stop_at_llm=370, stop_at_embed=180)

    coref_turns = multi_batch_coref.generate()
    coref_gt = multi_batch_coref.ground_truth(coref_turns)
    coref_qs = multi_batch_coref.build_questions(coref_gt)

    print("[scenario]")
    print(
        f"  multi_batch_coref: {len(coref_turns)} turns, "
        f"{len(coref_gt.pairs)} pairs, {len(coref_qs)} Qs"
    )
    print("  pair gaps:")
    for p in coref_gt.pairs:
        gap = p["name_turn"] - p["descriptor_turn"]
        print(
            f"    {p['name']:<10s} {p['predicate'][0]}.{p['predicate'][1]:<14s}  gap={gap}"
        )

    results = {
        "scenario": {
            "name": "multi_batch_coref",
            "n_turns": len(coref_turns),
            "n_pairs": len(coref_gt.pairs),
            "n_questions": len(coref_qs),
            "pair_gaps": [
                {
                    "name": p["name"],
                    "predicate": f"{p['predicate'][0]}.{p['predicate'][1]}",
                    "descriptor_turn": p["descriptor_turn"],
                    "name_turn": p["name_turn"],
                    "gap": p["name_turn"] - p["descriptor_turn"],
                }
                for p in coref_gt.pairs
            ],
        },
        "variants": {},
    }

    plan = [
        # 1. CONTROLLED A/B vs round 16a coref_w15_k3 (which scored strict=1/8).
        #    Same K=3, same total window=15. ONLY the target position changes:
        #    round 16a put it at the END; this puts it at the MIDDLE.
        ("coref_centered_K3_w6_w6", 6, 6, 3, True),
        # 2. 2x natural-lookahead fudge factor on w_future. total_window=24.
        ("coref_centered_K3_w7_w14", 7, 14, 3, True),
    ]

    for vname, w_past, w_future, k, do_qa in plan:
        try:
            res = run_variant(
                vname,
                turns=coref_turns,
                gt=coref_gt,
                qs=coref_qs,
                w_past=w_past,
                w_future=w_future,
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
