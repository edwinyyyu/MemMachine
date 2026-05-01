"""Hard scenarios runner — stress-test R24 on harder benchmarks.

Currently:
  - same_name_disambig: 3 different Alices in different contexts.

Future:
  - indirect_chain
  - silent_contradiction
  - multi_session
"""

from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND24 = HERE.parent
RESEARCH = ROUND24.parent
ROUND14 = RESEARCH / "round14_chain_density"
ROUND7 = RESEARCH / "round7"

sys.path.insert(0, str(ROUND24 / "architectures"))
sys.path.insert(0, str(ROUND24 / "scenarios"))
sys.path.insert(0, str(ROUND7 / "experiments"))

import aen7_recursive as ar  # noqa: E402
import indirect_chain  # noqa: E402
import same_name_disambig  # noqa: E402
import silent_contradiction  # noqa: E402
from _common import Budget, Cache  # noqa: E402


def _load_module(alias: str, path: Path):
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


r14_run = _load_module("r14_run", ROUND14 / "experiments" / "run.py")

CACHE_DIR = ROUND24 / "cache"
RESULTS_DIR = ROUND24 / "results"


def grade_qa(qs, store, cache, budget) -> dict:
    answers = {}
    for q in qs:
        a = ar.answer_question(q.question, store, cache, budget, top_k=14)
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


def run_variant(name, *, turns, qs, budget, enable_reflection=True):
    cache = Cache(CACHE_DIR / f"{name}.json")
    pairs = [(t.idx, t.text) for t in turns]
    print(f"\n=== {name} (n_turns={len(pairs)}) ===")
    print(f"[budget] llm={budget.llm_calls} cost=${budget.cost():.3f}")
    out = {"variant": name, "n_turns": len(pairs)}
    try:
        obs_facts, obs_mentions, cog_facts, cog_mentions, store, telemetry = (
            ar.ingest_turns(
                pairs,
                cache,
                budget,
                w_past=7,
                w_future=7,
                k=3,
                rebuild_index_every=4,
                reflection_budget=2,
                reflection_max=3,
                enable_reflection=enable_reflection,
            )
        )
    except RuntimeError as e:
        print(f"!!! ingest budget stop: {e}")
        cache.save()
        out["error"] = str(e)
        return out
    cache.save()

    n_entities = len(store.registry.entity_members)
    n_merges = sum(1 for b in store.registry.binding_events if b.op == "merge")
    n_refl = sum(t.get("n_reflection_calls", 0) for t in telemetry)
    print(
        f"[ingest] obs={len(obs_facts)} cog={len(cog_facts)} "
        f"entities={n_entities} merges={n_merges} reflections={n_refl}"
    )
    out["obs_facts"] = len(obs_facts)
    out["cog_facts"] = len(cog_facts)
    out["n_entities"] = n_entities
    out["n_merges"] = n_merges
    out["n_reflection_calls"] = n_refl

    # Dump obs facts mentioning each Alice for diagnostic
    print("[diagnostic] all obs facts:")
    for f in obs_facts:
        eids = [store.registry.get_canonical(m) for m in f.mention_ids]
        print(f"  [{f.fact_uuid} t={f.ts}] entities={eids} :: {f.text[:120]}")
    print("[diagnostic] all cog facts:")
    for f in cog_facts:
        eids = [store.registry.get_canonical(m) for m in f.mention_ids]
        print(f"  [{f.fact_uuid} t={f.ts}] entities={eids} :: {f.text[:120]}")

    if qs:
        try:
            qa = grade_qa(qs, store, cache, budget)
            out["qa"] = qa
            print(
                f"[QA] det={qa['deterministic_pass']}/{qa['total']}  "
                f"judge={qa['judge_pass']}/{qa['total']}"
            )
            for v in qa["verdicts"]:
                marker = "✓" if v.get("judge_correct") else "✗"
                print(
                    f"  {marker} [{v['qid']}] {v.get('question', '')[:60]} -> {v.get('answer', '')[:120]}"
                )
        except RuntimeError as e:
            print(f"!!! QA budget stop: {e}")
            cache.save()
            out["qa_error"] = str(e)

    out["cost_after"] = budget.cost()
    return out


def save(results, budget, fname="run_hard.json"):
    results["budget"] = {"cost": budget.cost(), "llm_calls": budget.llm_calls}
    p = RESULTS_DIR / fname
    p.write_text(json.dumps(results, indent=2, default=str))
    print(f"[checkpoint] {p} cost=${budget.cost():.3f}")


def main():
    budget = Budget(max_llm=900, max_embed=400, stop_at_llm=850, stop_at_embed=380)

    snd_turns = same_name_disambig.generate()
    snd_qs = same_name_disambig.build_questions(
        same_name_disambig.ground_truth(snd_turns)
    )
    ic_turns = indirect_chain.generate()
    ic_qs = indirect_chain.build_questions(indirect_chain.ground_truth(ic_turns))
    sc_turns = silent_contradiction.generate()
    sc_qs = silent_contradiction.build_questions(
        silent_contradiction.ground_truth(sc_turns)
    )

    print(f"[same_name_disambig]    turns={len(snd_turns)} Qs={len(snd_qs)}")
    print(f"[indirect_chain]        turns={len(ic_turns)} Qs={len(ic_qs)}")
    print(f"[silent_contradiction]  turns={len(sc_turns)} Qs={len(sc_qs)}")

    results = {"variants": {}}

    plan = [
        ("r24_hard_snd", snd_turns, snd_qs, True),
        ("r24_hard_ic", ic_turns, ic_qs, True),
        ("r24_hard_sc", sc_turns, sc_qs, True),
    ]

    for name, turns, qs, enable_refl in plan:
        try:
            res = run_variant(
                name,
                turns=turns,
                qs=qs,
                budget=budget,
                enable_reflection=enable_refl,
            )
        except Exception as e:
            print(f"!!! {name} CRASHED: {e}")
            traceback.print_exc()
            res = {"variant": name, "crash": str(e)}
        results["variants"][name] = res
        save(results, budget)

    save(results, budget)
    print(f"\n[done] cost=${budget.cost():.3f} llm={budget.llm_calls}")


if __name__ == "__main__":
    main()
