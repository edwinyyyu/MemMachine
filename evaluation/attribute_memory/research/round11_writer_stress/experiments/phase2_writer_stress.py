"""Phase 2: writer-stress at 1000 turns with the SIMPLIFIED writer.

Runs the simplified aen1 writer on a deterministic 1000-turn conversation,
measures:
  - Ref-emission rate (did writer emit a ref on each ground-truth supersede?)
  - Ref accuracy (did ref point at the correct chain-head?)
  - Chain integrity by turn-bucket (1-200, ..., 800-1000)
  - @-tag consistency (does writer keep using @Marcus or drop the @?)
  - End-to-end Q/A accuracy on 30 state-tracking questions
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND11 = HERE.parent
ROUND7 = ROUND11.parent / "round7"
sys.path.insert(0, str(ROUND11 / "architectures"))
sys.path.insert(0, str(ROUND11 / "scenarios"))
sys.path.insert(0, str(ROUND7 / "experiments"))

import aen1_simple  # noqa: E402
import long_conversation  # noqa: E402
from _common import Budget, Cache  # noqa: E402

CACHE_DIR = ROUND11 / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = ROUND11 / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def ref_emission_metrics(
    turns: list[long_conversation.Turn],
    gt: long_conversation.GroundTruth,
    log: list[aen1_simple.LogEntry],
) -> dict:
    """For each ground-truth supersede transition (second+ value in a chain),
    did at least one log entry written AT OR AFTER the transition turn-idx emit
    a ref?

    We define: a "covering entry" for a transition (turn_idx T, key K) is any
    log entry with ts in [T, T+5] (the batch window) whose mentions include
    the key's entity AND (ideally) whose predicate matches K or whose text
    mentions the new value.
    """
    # Build an index: turn_ts -> list of entries authored in that window
    by_ts = {}
    for e in log:
        by_ts.setdefault(e.ts, []).append(e)

    transitions = []  # (key, turn_idx, value, is_first)
    for key, chain in gt.chains.items():
        for i, (t, v) in enumerate(chain):
            transitions.append((key, t, v, i == 0))

    # For each transition (except the first), check:
    # - was a ref emitted?
    # - was the ref pointing at the (previous chain element's) entry?
    results = []
    for key, t, v, is_first in transitions:
        entity_tag, pred = key
        # Find covering entries (in the batch window [t, t+5])
        candidates = []
        for ts_i in range(t, t + 6):
            for e in by_ts.get(ts_i, []):
                if entity_tag in e.mentions:
                    candidates.append(e)
        # Try to find one that looks like "the writer's entry for this transition"
        # Use text-similarity: does the text mention v (the new value)?
        best = None
        for c in candidates:
            if v.lower() in c.text.lower():
                best = c
                break
        if best is None and candidates:
            # Fallback: any entry with matching predicate
            for c in candidates:
                if (
                    c.predicate
                    and c.predicate.replace("@", "").lower()
                    == f"{entity_tag.lstrip('@').lower()}.{pred.lower()}"
                ):
                    best = c
                    break
        if best is None and candidates:
            best = candidates[0]

        # Determine pass/fail criteria
        emitted_entry = best is not None
        # For "first" transitions (no prior value), refs aren't expected
        emitted_ref = bool(best.refs) if best else False
        # Also check @-tag was used (mentions include the full @Name)
        atag_ok = False
        if best:
            atag_ok = any(m == entity_tag for m in best.mentions)

        results.append(
            {
                "key": f"{key[0]}.{key[1]}",
                "turn": t,
                "value": v,
                "is_first": is_first,
                "covering_entry_uuid": best.uuid if best else None,
                "emitted_entry": emitted_entry,
                "emitted_ref": emitted_ref,
                "atag_ok": atag_ok,
            }
        )

    # Compute rates
    non_first = [r for r in results if not r["is_first"]]
    n_non_first = len(non_first)
    n_entry = sum(1 for r in non_first if r["emitted_entry"])
    n_ref = sum(1 for r in non_first if r["emitted_ref"])
    n_atag = sum(1 for r in non_first if r["atag_ok"])

    # Chain integrity by turn-bucket
    buckets = [(0, 200), (200, 400), (400, 600), (600, 800), (800, 1000)]
    bucket_stats = []
    for lo, hi in buckets:
        in_bucket = [r for r in non_first if lo < r["turn"] <= hi]
        bucket_stats.append(
            {
                "range": f"({lo},{hi}]",
                "n_transitions": len(in_bucket),
                "n_entry": sum(1 for r in in_bucket if r["emitted_entry"]),
                "n_ref": sum(1 for r in in_bucket if r["emitted_ref"]),
                "n_atag": sum(1 for r in in_bucket if r["atag_ok"]),
            }
        )

    return {
        "n_transitions_total": len(results),
        "n_transitions_non_first": n_non_first,
        "entry_emission_rate": n_entry / n_non_first if n_non_first else None,
        "ref_emission_rate": n_ref / n_non_first if n_non_first else None,
        "atag_rate": n_atag / n_non_first if n_non_first else None,
        "bucket_stats": bucket_stats,
        "per_transition": results,
    }


def atag_drift_analysis(log: list[aen1_simple.LogEntry]) -> dict:
    """Check if @-tag usage degrades over time.

    Specifically: for every known entity name (extracted from @mentions across
    the whole log), compute: fraction of log entries that mention the entity
    as a full @tag vs. just as a bare name in text.
    """
    # Gather known names
    known = set()
    for e in log:
        for m in e.mentions:
            if m.startswith("@"):
                known.add(m[1:])

    per_bucket = []
    buckets = [(0, 200), (200, 400), (400, 600), (600, 800), (800, 1000)]
    for lo, hi in buckets:
        bucket_entries = [e for e in log if lo < e.ts <= hi]
        n_total = 0
        n_atag_only = 0  # bare name in text BUT tag in mentions
        n_bare_only = 0  # bare name in text AND name NOT in mentions
        for e in bucket_entries:
            for name in known:
                # Does bare name appear in text?
                if re.search(rf"\b{name}\b", e.text):
                    n_total += 1
                    tag = f"@{name}"
                    if tag in e.mentions:
                        n_atag_only += 1
                    else:
                        n_bare_only += 1
        per_bucket.append(
            {
                "range": f"({lo},{hi}]",
                "n_name_occurrences": n_total,
                "n_atag_correct": n_atag_only,
                "n_bare_only": n_bare_only,
                "atag_rate": (n_atag_only / n_total) if n_total else None,
            }
        )
    return {"buckets": per_bucket, "known_names": sorted(known)}


def grade_questions(qs, answers):
    """Deterministic grader: expected_contains (AND), expected_absent."""
    verdicts = []
    for q in qs:
        ans = answers.get(q.qid, "")
        ans_low = ans.lower()
        missing = [p for p in q.expected_contains if p.lower() not in ans_low]
        forbidden = [p for p in q.expected_absent if p.lower() in ans_low]
        passed = not missing and not forbidden
        verdicts.append(
            {
                "qid": q.qid,
                "kind": q.kind,
                "passed": passed,
                "missing": missing,
                "forbidden": forbidden,
                "answer": ans,
            }
        )
    return verdicts


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(budget: Budget, n_turns: int = 1000) -> dict:
    # Generate scenario
    turns = long_conversation.generate(n_turns)
    gt = long_conversation.ground_truth(turns)
    questions = long_conversation.build_questions(gt)
    print(f"Generated {len(turns)} turns")
    print(f"Boss chain: {gt.chain_values(('@User', 'boss'))}")
    print(f"City chain: {gt.chain_values(('@User', 'location'))}")
    print(f"Employer chain: {gt.chain_values(('@User', 'employer'))}")
    print(f"{len(questions)} questions")

    # Ingest
    cache = Cache(CACHE_DIR / "phase2_simple.json")
    pairs = [(t.idx, t.text) for t in turns]
    print(f"\nIngesting {len(pairs)} turns in batches of 5...")
    log, idx = aen1_simple.ingest_turns(
        pairs, cache, budget, batch_size=5, rebuild_index_every=40
    )
    cache.save()
    print(f"log size: {len(log)} entries")
    print(f"supersede_head: {len(idx.supersede_head)} keys")
    print(f"mention_index: {len(idx.mention_index)} entities")
    print(
        f"Cost so far: ${budget.cost():.3f}  LLM={budget.llm_calls} "
        f"embed={budget.embed_calls}"
    )

    # Metrics
    ref_metrics = ref_emission_metrics(turns, gt, log)
    tag_drift = atag_drift_analysis(log)

    # Q/A
    print(f"\nAnswering {len(questions)} questions...")
    answers = {}
    for q in questions:
        a = aen1_simple.answer_question(q.question, idx, cache, budget, top_k=12)
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
        "arch": "aen1_simple",
        "n_turns": n_turns,
        "log_size": len(log),
        "num_supersede_heads": len(idx.supersede_head),
        "num_entities": len(idx.mention_index),
        "gt_chains": {f"{k[0]}.{k[1]}": gt.chain_values(k) for k in gt.chains},
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


def main() -> None:
    budget = Budget(max_llm=700, max_embed=300, stop_at_llm=670, stop_at_embed=290)
    result = run(budget)
    out = RESULTS_DIR / "phase2.json"
    # don't embed per_transition full list in main output, but DO include in companion
    big = dict(result)
    out.write_text(json.dumps(big, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
