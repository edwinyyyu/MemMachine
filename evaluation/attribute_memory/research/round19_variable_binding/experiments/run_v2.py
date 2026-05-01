"""Round 19 V2 — simplified single-emit schema with stronger filler skip."""

from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND19 = HERE.parent
RESEARCH = ROUND19.parent
ROUND16A = RESEARCH / "round16a_sliding_window"
ROUND14 = RESEARCH / "round14_chain_density"
ROUND7 = RESEARCH / "round7"

sys.path.insert(0, str(ROUND19 / "architectures"))
sys.path.insert(0, str(ROUND16A / "scenarios"))
sys.path.insert(0, str(ROUND7 / "experiments"))

import aen2_binding_v2 as ab  # noqa: E402
import multi_batch_coref  # noqa: E402
from _common import Budget, Cache  # noqa: E402


def _load_module(alias: str, path: Path):
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


r14_run = _load_module("r14_run", ROUND14 / "experiments" / "run.py")

CACHE_DIR = ROUND19 / "cache"
RESULTS_DIR = ROUND19 / "results"


def binding_correctness(log, resolutions, idx, gt) -> dict:
    by_ts: dict[int, list[ab.LogEntry]] = {}
    for e in log:
        by_ts.setdefault(e.ts, []).append(e)
    res_by_ts: dict[int, list[ab.Resolution]] = {}
    for r in resolutions:
        res_by_ts.setdefault(r.ts, []).append(r)

    out = []
    for p in gt.pairs:
        name = p["name"]
        pred = p["predicate"]
        d_turn = p["descriptor_turn"]
        n_turn = p["name_turn"]
        target_pred_str = f"{pred[0]}.{pred[1]}".replace("@@", "@")
        target_subj = pred[0] if pred[0].startswith("@") else f"@{pred[0]}"

        d_window = list(range(d_turn - 1, d_turn + 6))
        anon_entries = [
            e
            for ts in d_window
            for e in by_ts.get(ts, [])
            if e.subject == target_subj
            and e.predicate is not None
            and e.predicate.lower().replace("@", "")
            == target_pred_str.lower().replace("@", "")
        ]
        anon_cluster_ids = {e.cluster_id for e in anon_entries}
        anon_at_descriptor = len(anon_entries) > 0

        n_window = list(range(n_turn - 2, n_turn + 6))
        bind_resolutions = [
            r
            for ts in n_window
            for r in res_by_ts.get(ts, [])
            if r.canonical_label.lower() == name.lower()
        ]
        resolution_at_name = len(bind_resolutions) > 0
        binding_links_correctly = any(
            r.cluster_id in anon_cluster_ids for r in bind_resolutions
        )

        chain_key = (target_subj, target_pred_str)
        head_cluster = idx.chain_head.get(chain_key)
        head_label = idx.cluster_label.get(head_cluster) if head_cluster else None
        chain_label_resolves = (
            head_label is not None and head_label.lower() == name.lower()
        )

        out.append(
            {
                "name": name,
                "pred": target_pred_str,
                "d_turn": d_turn,
                "n_turn": n_turn,
                "gap": n_turn - d_turn,
                "anon_at_descriptor": anon_at_descriptor,
                "anon_cluster_ids": sorted(anon_cluster_ids),
                "resolution_at_name": resolution_at_name,
                "binding_links_correctly": binding_links_correctly,
                "chain_head_cluster": head_cluster,
                "chain_head_label": head_label,
                "chain_label_resolves": chain_label_resolves,
            }
        )

    n = len(out)
    return {
        "pairs": out,
        "anon_at_descriptor_pass": sum(1 for r in out if r["anon_at_descriptor"]),
        "resolution_at_name_pass": sum(1 for r in out if r["resolution_at_name"]),
        "binding_links_pass": sum(1 for r in out if r["binding_links_correctly"]),
        "chain_label_pass": sum(1 for r in out if r["chain_label_resolves"]),
        "total": n,
    }


def grade_qa(qs, idx, cache, budget) -> dict:
    answers = {}
    for q in qs:
        a = ab.answer_question(q.question, idx, cache, budget, top_k=14)
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
    return {
        "n_fires": n,
        "avg_prompt_chars": sum(t.get("prompt_chars", 0) for t in telemetry) / n,
        "avg_active_state_chars": sum(t.get("active_state_chars", 0) for t in telemetry)
        / n,
        "n_entries_total": sum(t.get("n_entries_emitted", 0) for t in telemetry),
        "n_with_label_total": sum(t.get("n_with_label", 0) for t in telemetry),
    }


def run_variant(name, *, turns, gt, qs, w_past, w_future, k, budget, do_qa=True):
    cache = Cache(CACHE_DIR / f"{name}.json")
    pairs = [(t.idx, t.text) for t in turns]
    print(
        f"\n=== {name} (multi_batch_coref, w_past={w_past}, K={k}, "
        f"w_future={w_future}, total={w_past + k + w_future}, n_turns={len(pairs)}) ==="
    )
    print(f"[budget so far] llm={budget.llm_calls} cost=${budget.cost():.3f}")
    out = {
        "variant": name,
        "scenario": "multi_batch_coref",
        "w_past": w_past,
        "w_future": w_future,
        "k": k,
        "n_turns": len(pairs),
    }
    try:
        log, resolutions, idx, telemetry = ab.ingest_turns(
            pairs,
            cache,
            budget,
            w_past=w_past,
            w_future=w_future,
            k=k,
            rebuild_index_every=4,
        )
    except RuntimeError as e:
        print(f"!!! ingest budget stop: {e}")
        cache.save()
        out["error"] = str(e)
        return out
    cache.save()

    print(
        f"[ingest] entries={len(log)} resolutions={len(resolutions)} "
        f"clusters={len(idx.cluster_entries)} chains={len(idx.chain_head)}"
    )
    print(f"[budget after ingest] llm={budget.llm_calls} cost=${budget.cost():.3f}")
    out["log_size"] = len(log)
    out["n_resolutions"] = len(resolutions)
    out["n_clusters"] = len(idx.cluster_entries)
    out["n_chains"] = len(idx.chain_head)
    out["telemetry_summary"] = telemetry_summary(telemetry)
    out["telemetry_per_fire"] = telemetry

    try:
        bind = binding_correctness(log, resolutions, idx, gt)
        out["binding"] = bind
        print(
            f"[binding] anon@d={bind['anon_at_descriptor_pass']}/{bind['total']} "
            f"resolved@n={bind['resolution_at_name_pass']}/{bind['total']} "
            f"links={bind['binding_links_pass']}/{bind['total']} "
            f"chain_label_resolves={bind['chain_label_pass']}/{bind['total']}"
        )
        for r in bind["pairs"]:
            print(
                f"  {r['name']:<10s} {r['pred']:<22s} gap={r['gap']:>3d}  "
                f"anon_d={r['anon_at_descriptor']!s:<5s} "
                f"res_n={r['resolution_at_name']!s:<5s} "
                f"links={r['binding_links_correctly']!s:<5s} "
                f"chain_label={r['chain_head_label']!r}"
            )
    except Exception as e:
        print(f"!!! binding metrics failed: {e}")
        traceback.print_exc()

    if do_qa and qs:
        try:
            print(f"\n[QA] running {len(qs)} questions...")
            qa = grade_qa(qs, idx, cache, budget)
            out["qa"] = qa
            print(
                f"[QA] det={qa['deterministic_pass']}/{qa['total']}  judge={qa['judge_pass']}/{qa['total']}"
            )
        except RuntimeError as e:
            print(f"!!! QA budget stop: {e}")
            cache.save()
            out["qa_error"] = str(e)

    out["llm_calls_after"] = budget.llm_calls
    out["cost_after"] = budget.cost()
    return out


def save(results, budget):
    results["budget"] = {
        "cost": budget.cost(),
        "llm_calls": budget.llm_calls,
        "embed_calls": budget.embed_calls,
    }
    p = RESULTS_DIR / "run_v2.json"
    p.write_text(json.dumps(results, indent=2, default=str))
    print(f"[checkpoint] {p} cost=${budget.cost():.3f} llm={budget.llm_calls}")


def main():
    budget = Budget(max_llm=400, max_embed=200, stop_at_llm=370, stop_at_embed=180)

    coref_turns = multi_batch_coref.generate()
    coref_gt = multi_batch_coref.ground_truth(coref_turns)
    coref_qs = multi_batch_coref.build_questions(coref_gt)

    print(
        f"[scenario] multi_batch_coref: {len(coref_turns)} turns, "
        f"{len(coref_gt.pairs)} pairs, {len(coref_qs)} Qs"
    )

    results = {
        "scenario": {
            "name": "multi_batch_coref",
            "n_turns": len(coref_turns),
            "n_pairs": len(coref_gt.pairs),
            "n_questions": len(coref_qs),
        },
        "variants": {},
    }

    plan = [
        ("v2_binding_K3_w7_w14", 7, 14, 3, True),
        ("v2_binding_K3_w7_w7", 7, 7, 3, True),
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
            print(f"!!! {vname} budget hit: {e}")
            res = {"variant": vname, "error": str(e)}
        except Exception as e:
            print(f"!!! {vname} CRASHED: {e}")
            traceback.print_exc()
            res = {"variant": vname, "crash": str(e)}
        results["variants"][vname] = res
        save(results, budget)
        if budget.llm_calls >= budget.stop_at_llm - 5:
            print("[plan] near LLM cap; halting.")
            break

    save(results, budget)
    print(f"\n[done] cost=${budget.cost():.3f} llm={budget.llm_calls}")


if __name__ == "__main__":
    main()
