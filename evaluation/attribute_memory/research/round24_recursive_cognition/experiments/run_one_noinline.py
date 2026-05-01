"""Run a single hard scenario by name with inline_anchors=True.

Usage:
  uv run python run_one_anchored.py <scenario_name>

Same as run_one.py but threads inline_anchors=True into ingest_turns and
uses distinct cache/result names so baseline vs anchored can be compared.
"""

from __future__ import annotations

import importlib
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


def main():
    if len(sys.argv) < 2:
        print("Usage: run_one_anchored.py <scenario>")
        sys.exit(1)
    scenario_name = sys.argv[1]
    scenario = importlib.import_module(scenario_name)

    name = f"r24_noinline_{scenario_name}"
    cache = Cache(CACHE_DIR / f"{name}.json")
    budget = Budget(max_llm=400, max_embed=200, stop_at_llm=370, stop_at_embed=180)

    turns = scenario.generate()
    qs = scenario.build_questions(scenario.ground_truth(turns))
    pairs = [(t.idx, t.text) for t in turns]
    print(f"=== {name} (n_turns={len(pairs)}, n_qs={len(qs)}) ===")

    out = {"variant": name, "scenario": scenario_name, "n_turns": len(pairs)}
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
                enable_reflection=True,
                inline_anchors=False,
            )
        )
    except RuntimeError as e:
        print(f"!!! ingest budget stop: {e}")
        cache.save()
        out["error"] = str(e)
        _save(name, out, budget)
        return

    cache.save()

    n_entities = len(store.registry.entity_members)
    n_merges = sum(1 for b in store.registry.binding_events if b.op == "merge")
    n_refl = sum(t.get("n_reflection_calls", 0) for t in telemetry)
    print(
        f"[ingest] obs={len(obs_facts)} cog={len(cog_facts)} entities={n_entities} merges={n_merges} reflections={n_refl}"
    )
    out["obs_facts"] = len(obs_facts)
    out["cog_facts"] = len(cog_facts)
    out["n_entities"] = n_entities
    out["n_merges"] = n_merges
    out["n_reflection_calls"] = n_refl

    try:
        qa = grade_qa(qs, store, cache, budget)
        out["qa"] = qa
        print(
            f"[QA] det={qa['deterministic_pass']}/{qa['total']}  judge={qa['judge_pass']}/{qa['total']}"
        )
        for v in qa["verdicts"]:
            marker = "OK" if v.get("judge_correct") else "FAIL"
            print(
                f"  [{marker}] [{v['qid']}] {v.get('question', '')[:60]} -> {v.get('answer', '')[:120]}"
            )
    except RuntimeError as e:
        print(f"!!! QA budget stop: {e}")
        cache.save()
        out["qa_error"] = str(e)

    out["cost_after"] = budget.cost()

    out["obs_texts"] = [
        {
            "uuid": f.fact_uuid,
            "ts": f.ts,
            "text": f.text,
            "entities": [store.registry.get_canonical(m) for m in f.mention_ids],
        }
        for f in obs_facts
    ]
    out["cog_texts"] = [
        {
            "uuid": f.fact_uuid,
            "ts": f.ts,
            "text": f.text,
            "entities": [store.registry.get_canonical(m) for m in f.mention_ids],
        }
        for f in cog_facts
    ]

    _save(name, out, budget)


def _save(name, out, budget):
    out["budget"] = {"cost": budget.cost(), "llm_calls": budget.llm_calls}
    p = RESULTS_DIR / f"{name}.json"
    p.write_text(json.dumps(out, indent=2, default=str))
    print(f"[checkpoint] {p} cost=${budget.cost():.3f}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
