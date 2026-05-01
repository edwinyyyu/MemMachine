"""Round 14 — high-density chain stress test.

Reuses round 11's `aen1_simple` writer + structural-index pipeline.
Generates ~700 turns with ~85 non-first supersede transitions distributed
across tail buckets so we can probe chain integrity at depth.

Metrics per turn-bucket (100-wide):
  - n_transitions  (ground-truth)
  - n_entry_emitted
  - n_ref_emitted
  - n_ref_correct  (does ref point at the actual most-recent chain entry?)
  - n_atag_ok

Plus end-to-end Q/A on ~30 chain-traversal questions.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND14 = HERE.parent
RESEARCH = ROUND14.parent
ROUND11 = RESEARCH / "round11_writer_stress"
ROUND7 = RESEARCH / "round7"
sys.path.insert(0, str(ROUND11 / "architectures"))
sys.path.insert(0, str(ROUND14 / "scenarios"))
sys.path.insert(0, str(ROUND7 / "experiments"))

import aen1_simple  # noqa: E402
import dense_chains  # noqa: E402
from _common import Budget, Cache, llm  # noqa: E402

CACHE_DIR = ROUND14 / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = ROUND14 / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------
# Per-turn-bucket ref-emission metrics
# -----------------------------------------------------------------


def find_covering_entry(
    log: list[aen1_simple.LogEntry],
    by_ts: dict[int, list[aen1_simple.LogEntry]],
    turn_idx: int,
    entity_tag: str,
    pred: str,
    new_value: str,
) -> aen1_simple.LogEntry | None:
    """Find the writer entry that most plausibly represents this transition.
    Looks in batch window [t, t+5].
    """
    candidates: list[aen1_simple.LogEntry] = []
    for ts_i in range(turn_idx, turn_idx + 6):
        for e in by_ts.get(ts_i, []):
            if entity_tag in e.mentions:
                candidates.append(e)
    if not candidates:
        return None
    # Best: predicate match AND new_value present
    pred_match = f"{entity_tag.lstrip('@')}.{pred}".lower()
    for c in candidates:
        if (
            c.predicate
            and c.predicate.replace("@", "").lower() == pred_match
            and new_value.lower() in c.text.lower()
        ):
            return c
    # Next: new_value in text
    for c in candidates:
        if new_value.lower() in c.text.lower():
            return c
    # Next: predicate match
    for c in candidates:
        if c.predicate and c.predicate.replace("@", "").lower() == pred_match:
            return c
    return candidates[0]


def collect_metrics(
    turns: list[dense_chains.Turn],
    gt: dense_chains.GroundTruth,
    log: list[aen1_simple.LogEntry],
    bucket_size: int = 100,
) -> dict:
    by_ts: dict[int, list[aen1_simple.LogEntry]] = {}
    for e in log:
        by_ts.setdefault(e.ts, []).append(e)

    transitions = []  # list of dicts
    # For "ref correctness" we need to know what the prior chain entry's uuid is
    # — i.e. the writer entry we identified for the previous value of this key.
    # So we walk chains in order and remember the previous covering entry uuid.
    for key, chain in gt.chains.items():
        prev_uuid: str | None = None
        for i, (t, v) in enumerate(chain):
            covering = find_covering_entry(log, by_ts, t, key[0], key[1], v)
            is_first = i == 0
            cov_uuid = covering.uuid if covering else None
            cov_refs = list(covering.refs) if covering else []
            atag_ok = bool(covering and key[0] in covering.mentions)
            ref_emitted = bool(cov_refs)
            # Was the prior chain entry referenced?
            ref_correct = (
                ref_emitted and prev_uuid is not None and prev_uuid in cov_refs
            )
            transitions.append(
                {
                    "key": f"{key[0]}.{key[1]}",
                    "turn": t,
                    "value": v,
                    "is_first": is_first,
                    "covering_uuid": cov_uuid,
                    "covering_text": covering.text if covering else None,
                    "covering_refs": cov_refs,
                    "covering_predicate": covering.predicate if covering else None,
                    "expected_prev_uuid": prev_uuid,
                    "emitted_entry": cov_uuid is not None,
                    "emitted_ref": ref_emitted,
                    "ref_correct": ref_correct,
                    "atag_ok": atag_ok,
                }
            )
            prev_uuid = cov_uuid

    # Bucket by turn
    n_turns = len(turns)
    n_buckets = (n_turns + bucket_size - 1) // bucket_size
    buckets = [(i * bucket_size, (i + 1) * bucket_size) for i in range(n_buckets)]
    bucket_stats = []
    non_first = [r for r in transitions if not r["is_first"]]
    for lo, hi in buckets:
        in_b = [r for r in non_first if lo < r["turn"] <= hi]
        n_t = len(in_b)
        bucket_stats.append(
            {
                "range": f"({lo},{hi}]",
                "n_transitions": n_t,
                "n_entry": sum(1 for r in in_b if r["emitted_entry"]),
                "n_ref": sum(1 for r in in_b if r["emitted_ref"]),
                "n_correct": sum(1 for r in in_b if r["ref_correct"]),
                "n_atag": sum(1 for r in in_b if r["atag_ok"]),
                "ref_emission_rate": (sum(1 for r in in_b if r["emitted_ref"]) / n_t)
                if n_t
                else None,
                "ref_correctness_rate": (sum(1 for r in in_b if r["ref_correct"]) / n_t)
                if n_t
                else None,
            }
        )

    n_nf = len(non_first)
    summary = {
        "n_transitions_total": len(transitions),
        "n_transitions_non_first": n_nf,
        "entry_emission_rate": sum(1 for r in non_first if r["emitted_entry"]) / n_nf
        if n_nf
        else None,
        "ref_emission_rate": sum(1 for r in non_first if r["emitted_ref"]) / n_nf
        if n_nf
        else None,
        "ref_correctness_rate": sum(1 for r in non_first if r["ref_correct"]) / n_nf
        if n_nf
        else None,
        "atag_rate": sum(1 for r in non_first if r["atag_ok"]) / n_nf if n_nf else None,
        "bucket_stats": bucket_stats,
    }
    return {"summary": summary, "transitions": transitions}


def atag_drift(
    log: list[aen1_simple.LogEntry], n_turns: int, bucket_size: int = 100
) -> dict:
    known = set()
    for e in log:
        for m in e.mentions:
            if m.startswith("@"):
                known.add(m[1:])
    n_buckets = (n_turns + bucket_size - 1) // bucket_size
    buckets = [(i * bucket_size, (i + 1) * bucket_size) for i in range(n_buckets)]
    out = []
    for lo, hi in buckets:
        bucket_entries = [e for e in log if lo < e.ts <= hi]
        n_total = 0
        n_atag = 0
        for e in bucket_entries:
            for name in known:
                if re.search(rf"\b{name}\b", e.text):
                    n_total += 1
                    if f"@{name}" in e.mentions:
                        n_atag += 1
        out.append(
            {
                "range": f"({lo},{hi}]",
                "n_name_occurrences": n_total,
                "n_atag_correct": n_atag,
                "atag_rate": (n_atag / n_total) if n_total else None,
            }
        )
    return {"buckets": out, "known_names": sorted(known)}


# -----------------------------------------------------------------
# Q/A grading: deterministic + LLM judge
# -----------------------------------------------------------------


def grade_deterministic(qs, answers):
    verdicts = []
    for q in qs:
        ans = answers.get(q.qid, "")
        ans_low = ans.lower()
        missing = [p for p in q.expected_contains if p.lower() not in ans_low]
        forbidden = [p for p in q.expected_absent if p.lower() in ans_low]
        passed = (not missing) and (not forbidden)
        verdicts.append(
            {
                "qid": q.qid,
                "kind": q.kind,
                "passed": passed,
                "missing": missing,
                "forbidden": forbidden,
                "answer": ans,
                "question": q.question,
                "expected_contains": q.expected_contains,
            }
        )
    return verdicts


JUDGE_PROMPT = """You are a strict grader. Given a question, a model's answer,
and the expected facts, decide whether the answer is correct.

QUESTION: {question}
EXPECTED: the answer should reference: {expected_contains}
{absent_block}
MODEL ANSWER: {answer}

Reply with a single JSON object:
{{"correct": true | false, "reason": "<one short sentence>"}}
"""


def judge_with_llm(verdicts, cache, budget):
    out = []
    for v in verdicts:
        if v["passed"]:
            out.append(
                {**v, "judge_correct": True, "judge_reason": "deterministic pass"}
            )
            continue
        absent_block = ""
        prompt = JUDGE_PROMPT.format(
            question=v["question"],
            expected_contains=", ".join(v["expected_contains"]),
            absent_block=absent_block,
            answer=v["answer"],
        )
        raw = llm(prompt, cache, budget)
        from _common import extract_json

        obj = extract_json(raw)
        if isinstance(obj, dict) and "correct" in obj:
            out.append(
                {
                    **v,
                    "judge_correct": bool(obj["correct"]),
                    "judge_reason": obj.get("reason", ""),
                }
            )
        else:
            out.append(
                {**v, "judge_correct": False, "judge_reason": f"unparsable: {raw[:80]}"}
            )
    return out


# -----------------------------------------------------------------
# Driver
# -----------------------------------------------------------------


def run(budget: Budget) -> dict:
    turns = dense_chains.generate()
    gt = dense_chains.ground_truth(turns)
    qs = dense_chains.build_questions(gt)

    print(f"[scenario] turns={len(turns)}  questions={len(qs)}")
    print("  chains:")
    for k, vs in gt.chains.items():
        print(f"   {k[0]}.{k[1]}: {len(vs)}")
    n_nonfirst = sum(max(0, len(v) - 1) for v in gt.chains.values())
    print(f"  non-first transitions: {n_nonfirst}")

    cache = Cache(CACHE_DIR / "writer_reader.json")
    pairs = [(t.idx, t.text) for t in turns]
    print(f"\n[ingest] {len(pairs)} turns, batch_size=5...")
    log, idx = aen1_simple.ingest_turns(
        pairs, cache, budget, batch_size=5, rebuild_index_every=40
    )
    cache.save()
    print(f"[ingest] log size: {len(log)}  supersede_heads: {len(idx.supersede_head)}")
    print(
        f"[budget] cost=${budget.cost():.3f} llm={budget.llm_calls} embed={budget.embed_calls}"
    )

    # Metrics
    metrics = collect_metrics(turns, gt, log, bucket_size=100)
    drift = atag_drift(log, n_turns=len(turns), bucket_size=100)

    print(
        f"\n[metrics] ref_emission_rate (overall non-first): "
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
        s_e = f"{rate_e:.2f}" if rate_e is not None else " — "
        s_c = f"{rate_c:.2f}" if rate_c is not None else " — "
        print(
            f"  {b['range']:>14s}  trans={b['n_transitions']:>3d}  "
            f"emit={s_e}  correct={s_c}"
        )

    # Q/A
    print(f"\n[QA] running {len(qs)} questions...")
    answers = {}
    for q in qs:
        a = aen1_simple.answer_question(q.question, idx, cache, budget, top_k=14)
        answers[q.qid] = a
    cache.save()
    verdicts = grade_deterministic(qs, answers)
    det_pass = sum(1 for v in verdicts if v["passed"])
    print(f"[QA] deterministic pass: {det_pass}/{len(verdicts)}")

    # Judge LLM for ones that didn't pass deterministic check
    judged = judge_with_llm(verdicts, cache, budget)
    cache.save()
    judge_pass = sum(1 for v in judged if v["judge_correct"])
    print(f"[QA] judge-graded pass: {judge_pass}/{len(judged)}")

    return {
        "n_turns": len(turns),
        "log_size": len(log),
        "num_supersede_heads": len(idx.supersede_head),
        "num_entities": len(idx.mention_index),
        "gt_chains": {
            f"{k[0]}.{k[1]}": [v for _, v in vs] for k, vs in gt.chains.items()
        },
        "metrics_summary": metrics["summary"],
        "tag_drift": drift,
        "transitions": metrics["transitions"],
        "questions": [
            {
                "qid": q.qid,
                "kind": q.kind,
                "question": q.question,
                "expected": q.expected_contains,
            }
            for q in qs
        ],
        "answers": answers,
        "verdicts": judged,
        "qa_deterministic_pass": det_pass,
        "qa_judge_pass": judge_pass,
        "qa_total": len(verdicts),
        "cost": budget.cost(),
        "llm_calls": budget.llm_calls,
        "embed_calls": budget.embed_calls,
    }


def main() -> None:
    # Hard cap: 500 LLM, 50 embed, ~$1.50.
    # Stop at 80%: 400 LLM, 40 embed.
    budget = Budget(max_llm=500, max_embed=50, stop_at_llm=400, stop_at_embed=40)
    try:
        result = run(budget)
    except RuntimeError as e:
        print(f"\n!!! Budget stop: {e}")
        raise

    out = RESULTS_DIR / "run.json"
    out.write_text(json.dumps(result, indent=2, default=str))
    print(f"\n[done] wrote {out}")
    print(
        f"[done] cost=${result['cost']:.3f} llm={result['llm_calls']} "
        f"embed={result['embed_calls']}"
    )


if __name__ == "__main__":
    main()
