"""Round 5: command-schema design for semantic memory.

Questions answered jointly:
  Q1: Is the model inherently bad at modify-in-place? (framing ablations)
  Q2: Does UPSERT supplant ADD without introducing new bias? (S2, S3)
  Q3: Is an append-only log a viable architecture? (S4, S5)
  Q4: Can we eliminate DELETE entirely? (S3 has no remove; compare)

Methodology:
  - 14 scenarios: 10 single-turn + 4 multi-turn.
  - Deterministic applier: row-state compared via rows_equal; append-log
    state graded with loose substring checks on the log.
  - For multi-turn scenarios, we drive state forward turn-by-turn. Each
    turn is an independent LLM call (no intra-turn memory other than
    the running state).

Budget: hard cap 250, stop at 80% = 200. Target ~150.

Candidates x scenarios:
  7 candidates x (10 single-turn + 4 multi-turn) x avg 4 turns per multi
  = 7 * (10 + 4*5) = 7 * 30 = 210 calls worst case.
  In practice: multi-turn averages ~6 turns per scenario. We'll tune.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # import common.py from update_commands/

from appliers import (
    LogEntry,
    apply_log_commands,
    apply_row_commands,
    grade_loose_multi_turn,
    render_log,
    render_sheet,
    rows_diff,
    rows_equal,
)
from candidates import CANDIDATES, Candidate
from common import (  # type: ignore
    MODEL,
    CallBudget,
    LLMCache,
    llm_call,
    make_client,
)

CACHE_FILE = HERE / "cache" / "round5_cache.json"
RESULTS_DIR = HERE / "results"
RESULTS_FILE = RESULTS_DIR / "round5_results.json"
REPORT_FILE = RESULTS_DIR / "round5_report.md"
SCENARIOS_FILE = HERE / "scenarios.json"


def extract_json_array(text: str) -> list[Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    try:
        v = json.loads(text)
        return v if isinstance(v, list) else None
    except Exception:
        pass
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            v = json.loads(m.group(0))
            return v if isinstance(v, list) else None
        except Exception:
            return None
    return None


def tally_ops(commands: list[dict[str, Any]] | None) -> dict[str, int]:
    t: dict[str, int] = {}
    if not commands:
        return t
    for c in commands:
        if isinstance(c, dict):
            op = c.get("op", "<missing_op>")
            t[op] = t.get(op, 0) + 1
    return t


def run_single_turn(
    cand: Candidate,
    scenario: dict[str, Any],
    client,
    cache,
    budget,
) -> dict[str, Any]:
    rows_before = scenario.get("rows_before", [])
    turn = scenario["turn"]

    if cand.schema_family == "row":
        prior_state = render_sheet(rows_before, show_cardinality=cand.show_cardinality)
    else:
        prior_state = render_log(
            []
        )  # single-turn starts with empty log for log schemas
        # Actually: for single-turn scenarios with rows_before, we need to bootstrap
        # the log from rows_before. Represent each row as a bootstrap log entry.
        bootstrap_log = bootstrap_log_from_rows(rows_before)
        prior_state = render_log(bootstrap_log)

    prompt = cand.build_prompt(prior_state, turn)
    raw = llm_call(client, cache, budget, prompt)
    cmds = extract_json_array(raw) or []
    tally = tally_ops(cmds)

    if cand.schema_family == "row":
        applied = apply_row_commands(rows_before, cmds)
        correct = rows_equal(applied.rows, scenario["rows_after"])
        diff = rows_diff(applied.rows, scenario["rows_after"]) if not correct else None
        return {
            "scenario_id": scenario["id"],
            "intent": scenario["intent"],
            "raw_output": raw,
            "parsed_commands": cmds,
            "op_tally": tally,
            "apply_errors": applied.errors,
            "state_correct": correct,
            "diff": diff,
            "output_chars": len(raw),
            "applied_rows": applied.rows,
        }
    bootstrap_log = bootstrap_log_from_rows(rows_before)
    new_log, errs = apply_log_commands(bootstrap_log, cmds)
    # Evaluate via loose_grading on single-turn scenarios too.
    # For single-turn, we check whether the expected rows_after facts
    # are covered by the log's live entries.
    loose = single_turn_loose_grade(scenario, new_log)
    return {
        "scenario_id": scenario["id"],
        "intent": scenario["intent"],
        "raw_output": raw,
        "parsed_commands": cmds,
        "op_tally": tally,
        "apply_errors": errs,
        "state_correct": loose["pass"],
        "diff": loose,
        "output_chars": len(raw),
        "log": [vars(e) for e in new_log],
    }


def run_multi_turn(
    cand: Candidate,
    scenario: dict[str, Any],
    client,
    cache,
    budget,
) -> dict[str, Any]:
    turns: list[str] = scenario["turns"]
    rows_before = scenario.get("rows_before", [])
    tally_acc: dict[str, int] = {}
    errs_acc: list[str] = []
    raw_outputs: list[str] = []
    all_commands: list[list[dict[str, Any]]] = []

    if cand.schema_family == "row":
        current_rows = [dict(r) for r in rows_before]
        for t in turns:
            prior_state = render_sheet(
                current_rows, show_cardinality=cand.show_cardinality
            )
            prompt = cand.build_prompt(prior_state, t)
            raw = llm_call(client, cache, budget, prompt)
            raw_outputs.append(raw)
            cmds = extract_json_array(raw) or []
            all_commands.append(cmds)
            for op, k in tally_ops(cmds).items():
                tally_acc[op] = tally_acc.get(op, 0) + k
            applied = apply_row_commands(current_rows, cmds)
            errs_acc += applied.errors
            current_rows = applied.rows
        # Grade
        grade = grade_loose_multi_turn(current_rows, None, scenario)
        return {
            "scenario_id": scenario["id"],
            "intent": scenario["intent"],
            "raw_outputs": raw_outputs,
            "all_commands": all_commands,
            "op_tally": tally_acc,
            "apply_errors": errs_acc,
            "state_correct": grade["pass"],
            "grade": grade,
            "final_rows": current_rows,
        }
    current_log = bootstrap_log_from_rows(rows_before)
    for t in turns:
        prior_state = render_log(current_log)
        prompt = cand.build_prompt(prior_state, t)
        raw = llm_call(client, cache, budget, prompt)
        raw_outputs.append(raw)
        cmds = extract_json_array(raw) or []
        all_commands.append(cmds)
        for op, k in tally_ops(cmds).items():
            tally_acc[op] = tally_acc.get(op, 0) + k
        new_log, errs = apply_log_commands(current_log, cmds)
        errs_acc += errs
        current_log = new_log
    grade = grade_loose_multi_turn([], current_log, scenario)
    return {
        "scenario_id": scenario["id"],
        "intent": scenario["intent"],
        "raw_outputs": raw_outputs,
        "all_commands": all_commands,
        "op_tally": tally_acc,
        "apply_errors": errs_acc,
        "state_correct": grade["pass"],
        "grade": grade,
        "final_log": [vars(e) for e in current_log],
    }


def bootstrap_log_from_rows(rows: list[dict]) -> list[LogEntry]:
    """Convert row-initial state into log bootstrap entries. Each row
    becomes one entry with a synthetic id. This is only used when a
    single-turn scenario has rows_before; multi-turn starts empty."""
    log: list[LogEntry] = []
    for i, r in enumerate(rows, start=1):
        topic = r["topic_category"]
        attr = r["attribute"]
        v = r["value"]
        vs = ", ".join(v) if isinstance(v, list) else str(v)
        conf = r.get("confidence") or "confirmed"
        text = f"{attr}: {vs} ({conf})"
        log.append(LogEntry(id=i, topic=topic, text=text, refs=[], relation=None))
    return log


def single_turn_loose_grade(scenario: dict, log: list[LogEntry]) -> dict[str, Any]:
    """Loose grading for single-turn scenarios evaluated against an
    append-only log. We check that each expected row's (attribute + value)
    appears as a keyword match in a live (non-invalidated) log entry, and
    that any rows NOT in expected_after don't appear in live log entries
    when they should have been retracted.

    This is not as strict as rows_equal but gives append-only schemas a
    fair grading surface.
    """
    expected = scenario.get("rows_after", [])
    rows_before = scenario.get("rows_before", [])
    checks: list[dict] = []

    live_texts = [e.text.lower() for e in log if not e.invalidated]
    all_texts = [e.text.lower() for e in log]

    # For each expected row, is its value present somewhere live?
    for r in expected:
        v = r["value"]
        if isinstance(v, list):
            # For sets, every member must appear in some live entry
            ok = True
            missing = []
            for m in v:
                token = m.lower().split()[0] if m.split() else m.lower()
                if not any(token in t for t in live_texts):
                    ok = False
                    missing.append(m)
            checks.append(
                {
                    "label": f"set {r['attribute']}",
                    "pass": ok,
                    "missing": missing,
                }
            )
        else:
            # Scalar: need to find a non-trivial token from value in live text.
            vs = str(v).lower()
            # Use the first meaningful token (skip common words)
            toks = [
                t for t in re.split(r"\W+", vs) if len(t) > 2 and t not in STOPWORDS
            ]
            token = toks[0] if toks else vs
            ok = any(token in t for t in live_texts)
            # Also: if expected confidence is hedged/negated, it should appear SOMEWHERE
            conf = r.get("confidence")
            if conf and conf != "confirmed":
                conf_ok = any(conf in t for t in live_texts)
                ok = ok and conf_ok
            checks.append(
                {"label": f"scalar {r['attribute']}", "pass": ok, "token": token}
            )

    # For rows that were in rows_before but NOT in expected (retraction),
    # they should not appear in live texts (or should be invalidated).
    expected_keys = {(r["topic_category"], r["attribute"]) for r in expected}
    for r in rows_before:
        k = (r["topic_category"], r["attribute"])
        if k in expected_keys:
            continue
        # This row was retracted
        v = r["value"]
        vs = ", ".join(v) if isinstance(v, list) else str(v)
        toks = [
            t for t in re.split(r"\W+", vs.lower()) if len(t) > 2 and t not in STOPWORDS
        ]
        if not toks:
            continue
        token = toks[0]
        leak = any(token in t for t in live_texts)
        checks.append(
            {"label": f"retracted {r['attribute']}", "pass": not leak, "token": token}
        )

    all_pass = all(c["pass"] for c in checks)
    return {"pass": all_pass, "checks": checks}


STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "as",
    "is",
    "was",
    "are",
    "were",
    "be",
    "been",
    "being",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "peanut",
    "allergy",
}  # last ones help debugging


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (HERE / "cache").mkdir(parents=True, exist_ok=True)

    client = make_client()
    cache = LLMCache(CACHE_FILE)
    budget = CallBudget(max_calls=250, stop_at=200)

    with open(SCENARIOS_FILE) as f:
        scenarios = json.load(f)["scenarios"]

    print(f"Round 5: {len(CANDIDATES)} candidates x {len(scenarios)} scenarios.")
    single_turn = [s for s in scenarios if "turns" not in s]
    multi_turn = [s for s in scenarios if "turns" in s]
    total_turns = len(CANDIDATES) * (
        len(single_turn) + sum(len(s["turns"]) for s in multi_turn)
    )
    print(
        f"  Single-turn scenarios: {len(single_turn)}; multi-turn: {len(multi_turn)}."
    )
    print(
        f"  Estimated worst-case calls: {total_turns} (hard cap {budget.max_calls}, stop at {budget.stop_at})."
    )

    results: dict[str, Any] = {"model": MODEL, "candidates": {}}

    # For framing ablations on S1, restrict to a subset that exercises the
    # 'preserve body verbatim' question (T06 weaken, T11 paraphrased correction,
    # T12 retraction, T14 evolution), plus a few single-turn sanity anchors.
    FRAMING_SCENARIO_IDS = {
        "T02_correction_value",
        "T03_retraction",
        "T04_set_add",
        "T05_set_remove",
        "T06_confidence_weaken",
        "T07_noop_weather",
        "T08_noop_joke",
        "T10_multi_change",
        "T11_paraphrased_correction_after_chain",
    }
    FRAMING_CANDIDATES = {"S1_diff_framing", "S1_archivist_framing"}

    try:
        for cand in CANDIDATES:
            print(
                f"\n=== Candidate: {cand.key} ({cand.name}) [{cand.schema_family}] ==="
            )
            per_scen: list[dict[str, Any]] = []
            correct_count = 0
            op_totals: dict[str, int] = {}
            total_output_chars = 0

            if cand.key in FRAMING_CANDIDATES:
                scens_for_cand = [
                    s for s in scenarios if s["id"] in FRAMING_SCENARIO_IDS
                ]
            else:
                scens_for_cand = scenarios

            for scen in scens_for_cand:
                if "turns" in scen:
                    r = run_multi_turn(cand, scen, client, cache, budget)
                    chars = sum(len(t) for t in r["raw_outputs"])
                    total_output_chars += chars
                else:
                    r = run_single_turn(cand, scen, client, cache, budget)
                    total_output_chars += r.get("output_chars", 0)

                for op, k in r["op_tally"].items():
                    op_totals[op] = op_totals.get(op, 0) + k

                if r["state_correct"]:
                    correct_count += 1
                per_scen.append(r)
                print(
                    f"  [{scen['id']}] correct={r['state_correct']}  tally={r['op_tally']}"
                )
                cache.save()

            results["candidates"][cand.key] = {
                "name": cand.name,
                "schema_family": cand.schema_family,
                "framing": (cand.framing.split("\n", 1)[0])[:80],
                "correct_count": correct_count,
                "total": len(scenarios),
                "accuracy": round(correct_count / len(scenarios), 3),
                "op_totals": op_totals,
                "total_output_chars": total_output_chars,
                "per_scenario": per_scen,
            }
            print(
                f"  -> {cand.key}: {correct_count}/{len(scenarios)} correct; "
                f"ops={op_totals}"
            )
            print(
                f"     budget so far: {budget.made} calls (~${budget.approx_cost():.2f})"
            )
    finally:
        cache.save()
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved {RESULTS_FILE}")
        print(f"Final budget: {budget.made} calls (~${budget.approx_cost():.2f})")

    write_report(results)


def write_report(results: dict[str, Any]) -> None:
    md: list[str] = ["# Round 5 -- Command Schema Design\n"]
    md.append(
        "Tests S1 (round-4 winner) vs. S2/S3 (upsert) vs. S4/S5 (append-only), with framing ablations on S1.\n"
    )
    md.append(
        f"Model: `{results['model']}`. Scenarios: 14 (10 single-turn + 4 multi-turn chains).\n"
    )

    # Leaderboard
    md.append("## Leaderboard\n")
    md.append("| Candidate | Family | Correct | Accuracy | Op totals |")
    md.append("|-----------|--------|---------|----------|-----------|")
    ranked = sorted(
        results["candidates"].items(),
        key=lambda kv: kv[1]["correct_count"],
        reverse=True,
    )
    for key, block in ranked:
        md.append(
            f"| `{key}` | {block['schema_family']} | "
            f"{block['correct_count']}/{block['total']} | "
            f"{block['accuracy']:.0%} | `{json.dumps(block['op_totals'])}` |"
        )
    md.append("")

    # Per-candidate detail (correctness matrix)
    md.append("## Per-scenario correctness\n")
    scenario_ids: list[str] = []
    if ranked:
        first_block = ranked[0][1]
        scenario_ids = [s["scenario_id"] for s in first_block["per_scenario"]]
    md.append("| Scenario | " + " | ".join(k for k, _ in ranked) + " |")
    md.append("|----------|" + "----|" * len(ranked))
    for sid in scenario_ids:
        row = [sid]
        for key, block in ranked:
            matched = [s for s in block["per_scenario"] if s["scenario_id"] == sid]
            if matched:
                row.append("Y" if matched[0]["state_correct"] else "N")
            else:
                row.append("-")
        md.append("| " + " | ".join(row) + " |")
    md.append("")

    # Per-candidate detail
    for key, block in ranked:
        md.append(f"\n## `{key}` -- {block['name']}\n")
        md.append(f"- Schema family: {block['schema_family']}\n")
        md.append(
            f"- Correct: **{block['correct_count']}/{block['total']}** ({block['accuracy']:.0%})\n"
        )
        md.append(f"- Op totals: `{json.dumps(block['op_totals'])}`\n")
        md.append(f"- Total output chars: {block['total_output_chars']}\n")
        md.append("\n| Scenario | Correct | Tally | Notes |")
        md.append("|----------|---------|-------|-------|")
        for pr in block["per_scenario"]:
            notes = ""
            if "apply_errors" in pr and pr.get("apply_errors"):
                notes = f"errors={len(pr['apply_errors'])}"
            if "diff" in pr and pr.get("diff"):
                d = pr["diff"]
                if isinstance(d, dict) and "missing" in d:
                    if d.get("missing") or d.get("extra") or d.get("wrong"):
                        notes += f" diff:missing={len(d.get('missing', []))},extra={len(d.get('extra', []))},wrong={len(d.get('wrong', []))}"
            if "grade" in pr and isinstance(pr["grade"], dict):
                failed = [c for c in pr["grade"].get("checks", []) if not c.get("pass")]
                if failed:
                    notes += f" failed_checks={[c['label'] for c in failed]}"
            md.append(
                f"| {pr['scenario_id']} | {pr['state_correct']} | `{json.dumps(pr['op_tally'])}` | {notes} |"
            )
        md.append("")

    with open(REPORT_FILE, "w") as f:
        f.write("\n".join(md))
    print(f"Saved {REPORT_FILE}")


if __name__ == "__main__":
    main()
