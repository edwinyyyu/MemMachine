"""Round 2: finalist stress test + prompt tightening.

Round 1 findings (see results/round1_report.md):
  - No schema dominates. Top 3 at 8/12: baseline_addel, member_ops, indexed_patch.
  - Universal failure: S07 weather chitchat — 5/5 schemas recorded "dislikes November"
    as a fact. This is a NOOP-DISCIPLINE failure, not a schema failure.
  - Confidence handling best in indexed_patch (2/2) and mixed in member_ops (1/2).
  - Baseline add/delete matches winners only because the LLM judge accepts
    paraphrase matches — it cannot emit noop/update/member verbs by design, so
    its "structural bias" is extreme: every command is add-or-delete.

Round 2 focus — finalists: member_ops and indexed_patch, with:
  (a) noop-discipline: explicit "weather/chitchat/vented feeling is not a fact"
      rule in the framing
  (b) confidence: exact, predictable confidence markers in the output value text
  (c) re-test on: S07 (noop weather), S05 (strengthen), S06 (weaken), S09 (multi-op),
      S02 (correction + introduce).

Budget: 2 finalists x 5 scenarios x (author + judge) = 20 calls.
After Round 1's 116, this lands at ~136 / 150 cap. We raise the soft stop to 140
(still well under the $1.50 hard cap).
"""

from __future__ import annotations

import json
from typing import Any

from common import (
    CACHE_DIR,
    MODEL,
    RESULTS_DIR,
    CallBudget,
    LLMCache,
    llm_call,
    load_scenarios,
    make_client,
)
from round1 import (
    Candidate,
    build_author_prompt,
    extract_json_array,
    judge,
    tally_ops,
)

CACHE_FILE = CACHE_DIR / "round2_cache.json"
RESULTS_FILE = RESULTS_DIR / "round2_results.json"
REPORT_FILE = RESULTS_DIR / "round2_report.md"


# ---------------------------------------------------------------------------
# Tightened candidates (finalists from Round 1 with noop-discipline + confidence
# marker discipline added).
# ---------------------------------------------------------------------------
MEMBER_OPS_V2_PROMPT = """\
You are a librarian maintaining a card catalog about a person. Each card lives \
under a topic.category heading and records one attribute. Some attributes hold a \
single value; others hold a set of members (e.g. allergies, pets, hobbies).

Before emitting any operation, decide: does this statement belong in a permanent \
factual record about the person? If not, emit noop.

DO NOT record (emit noop instead):
- Weather comments, seasonal gripes, chitchat ("the weather is gloomy", "I hate \
November").
- Transient moods or fleeting reactions ("I'm tired", "ugh", "cool", "thanks").
- Generic filler / acknowledgements ("haha yeah", "totally", "sure thing").
- Repetitions of facts already in the catalog that add no new detail.

DO record:
- Durable attributes (where they live, what they do, names, relationships).
- Durable preferences/traits (diet, allergies, hobbies, values).
- Plans/events the person commits to or reports completing.
- Any correction, addition, or removal to something already in the catalog.

Emit a JSON array of operations, each exactly one of:

  {"op": "add",           "category": "...", "attribute": "...", "value": "..."}
      // new single-valued card
  {"op": "update",        "category": "...", "attribute": "...", "value": "..."}
      // replace the value of an existing single-valued card
  {"op": "delete",        "category": "...", "attribute": "..."}
      // retire a card entirely
  {"op": "add_member",    "category": "...", "attribute": "...", "member": "..."}
      // add one member to a set-valued card
  {"op": "remove_member", "category": "...", "attribute": "...", "member": "..."}
      // remove one member from a set-valued card (match by meaning, not exact text)
  {"op": "noop"}
      // the statement carries no catalog-worthy information

Schema rules:
- Choose `add_member` / `remove_member` ONLY when the catalog already shows the \
attribute as a set (multiple comma-separated items). For single-valued attributes \
use update/delete.
- Confidence markers — when the statement changes the certainty of an existing \
value, use `update` and put a confidence tag at the END of the new value:
    confirmed:     "... (confirmed)"       (e.g. "summited Mt Rainier (confirmed)")
    hedged/trial:  "... (hedged)"          (e.g. "trying vegan for a month (hedged)")
    intended:      "... (intended)"        (for plans not yet acted on)
  Use EXACTLY one of: (confirmed), (hedged), (intended). No other parenthetical.
- Match existing cards by MEANING: if the person refers to a stored fact by \
paraphrase, treat it as the same card.
- When one turn carries multiple distinct changes, emit multiple operations \
(one per logical change).

CURRENT CATALOG:
{prior_state}

NEW STATEMENT:
{turn}

Emit ONLY the JSON array. No prose.
"""


INDEXED_PATCH_V2_PROMPT = """\
You are a copy editor marking up a numbered fact sheet. Each numbered line is one \
fact, in the form:
  [n] topic.category | attribute: value

Before emitting any edit, decide: does this statement contain something that \
BELONGS on a permanent fact sheet about the person? If not, emit noop.

DO NOT write to the sheet (emit noop instead):
- Weather comments, seasonal gripes, chitchat ("the weather is gloomy", "I hate \
November").
- Transient moods or fleeting reactions ("I'm tired", "ugh", "cool", "thanks").
- Generic filler / acknowledgements ("haha yeah", "totally", "sure thing").
- Repetitions of facts already on the sheet that add no new detail.

DO write to the sheet:
- Durable attributes (where they live, what they do, names, relationships).
- Durable preferences/traits (diet, allergies, hobbies, values).
- Plans/events the person commits to or reports completing.
- Any correction, addition, or removal to an existing fact.

Emit a JSON array of edits. Each edit is one of:

  {"op": "keep",   "index": n}
      // fact is still exactly right; keep verbatim (optional, rarely needed)
  {"op": "revise", "index": n, "new_text": "topic.category | attribute: new_value"}
      // replace the content of fact [n]
  {"op": "remove", "index": n}
      // strike fact [n] from the sheet
  {"op": "add",    "new_text": "topic.category | attribute: value"}
      // append a brand-new numbered line at the end
  {"op": "noop"}
      // the statement does not require any change

Schema rules:
- For set-valued attributes (the value looks like a comma-separated list), \
`revise` with the new full comma-separated list.
- Confidence markers — when confidence changes, `revise` the line and append \
EXACTLY one of: (confirmed), (hedged), (intended) at the end of the value. \
E.g. "user.activities | hiking_plan: summited Mt Rainier (confirmed)".
- When one turn carries multiple distinct changes, emit multiple edits \
(one per logical change).
- Match facts by MEANING: if the statement refers to fact [n] in paraphrase, \
that is still fact [n].
- An ambiguous referent (multiple facts could match) -> prefer noop over a \
blind edit.

CURRENT FACT SHEET:
{prior_state_numbered}

NEW STATEMENT:
{turn}

Emit ONLY the JSON array. No prose.
"""


CANDIDATES_V2: list[Candidate] = [
    Candidate(
        "member_ops_v2",
        "member ops + noop-discipline (librarian v2)",
        MEMBER_OPS_V2_PROMPT,
        "markdown",
    ),
    Candidate(
        "indexed_patch_v2",
        "numbered sheet + noop-discipline (copy editor v2)",
        INDEXED_PATCH_V2_PROMPT,
        "numbered",
    ),
]


# Focused subset: scenarios that previously failed or stressed set/confidence.
STRESS_SCENARIO_IDS = [
    "S02_correction_paraphrased",  # multi-change correction: universal miss
    "S05_strengthen_confidence",  # confidence up
    "S06_weaken_confidence",  # confidence down
    "S07_noop_weather",  # universal noop failure
    "S09_multi_op",  # multi-op mixing remove_member + replace
    "S10_ambiguous_referent",  # ambiguous
]


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    all_scenarios = {s["id"]: s for s in load_scenarios()}
    scenarios = [all_scenarios[i] for i in STRESS_SCENARIO_IDS]
    print(
        f"Round 2: {len(CANDIDATES_V2)} candidates x {len(scenarios)} stress scenarios"
    )
    print("Scenarios:", [s["id"] for s in scenarios])

    client = make_client()
    # Share cache with round1 so we don't duplicate any judge/author calls that
    # happen to be identical; round2 prompts differ, so they'll be new entries.
    cache = LLMCache(CACHE_FILE)
    # Raise soft stop to 140 for Round 2 (well under $1.50 hard cap).
    budget = CallBudget(max_calls=150, stop_at=140)
    # Account for Round 1's 116 prior calls:
    budget.made = 116

    all_results: dict[str, Any] = {"model": MODEL, "candidates": {}}

    try:
        for cand in CANDIDATES_V2:
            print(f"\n=== {cand.key} ({cand.name}) ===")
            per_scenario: list[dict[str, Any]] = []
            op_totals: dict[str, int] = {}
            correct_count = 0
            parse_failures = 0

            for s in scenarios:
                prompt = build_author_prompt(cand, s)
                raw = llm_call(client, cache, budget, prompt)
                cmds = extract_json_array(raw)
                if cmds is None:
                    parse_failures += 1
                tally = tally_ops(cmds)
                for op, n in tally.items():
                    op_totals[op] = op_totals.get(op, 0) + n

                verdict = judge(client, cache, budget, s, raw)
                if verdict.get("correct"):
                    correct_count += 1

                per_scenario.append(
                    {
                        "scenario_id": s["id"],
                        "intent": s["intent"],
                        "description": s["description"],
                        "raw_output": raw,
                        "parsed_commands": cmds,
                        "op_tally": tally,
                        "judge": verdict,
                    }
                )
                print(
                    f"  [{s['id']}] correct={verdict.get('correct')}  "
                    f"hits={verdict.get('expected_hits')}/{verdict.get('expected_total')}  "
                    f"spurious={verdict.get('spurious_changes')}  tally={tally}"
                )
                cache.save()

            all_results["candidates"][cand.key] = {
                "name": cand.name,
                "prompt_template": cand.prompt_template,
                "prior_state_mode": cand.prior_state_mode,
                "correct_count": correct_count,
                "total": len(scenarios),
                "accuracy": round(correct_count / len(scenarios), 3),
                "parse_failures": parse_failures,
                "op_totals": op_totals,
                "per_scenario": per_scenario,
            }
            print(
                f"  -> {cand.key}: correct {correct_count}/{len(scenarios)} "
                f"= {correct_count / len(scenarios):.0%}; op_totals={op_totals}"
            )
            print(
                f"     budget so far: {budget.made} calls (~${budget.approx_cost():.2f})"
            )
    finally:
        cache.save()
        with open(RESULTS_FILE, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved {RESULTS_FILE}")

    # Markdown report
    md: list[str] = [
        "# Round 2 — finalist stress test with noop-discipline + confidence markers\n"
    ]
    md.append(f"Scenarios: {STRESS_SCENARIO_IDS}\n")
    md.append(
        f"Model: `{MODEL}`. Budget used: ~{budget.made}/150 "
        f"(~${budget.approx_cost():.2f}).\n"
    )
    md.append("## Leaderboard\n")
    md.append("| Candidate | Correct | Accuracy | Op totals |")
    md.append("|-----------|---------|----------|-----------|")
    ranked = sorted(
        all_results["candidates"].items(),
        key=lambda kv: kv[1]["correct_count"],
        reverse=True,
    )
    for key, block in ranked:
        md.append(
            f"| `{key}` | {block['correct_count']}/{block['total']} | "
            f"{block['accuracy']:.0%} | {json.dumps(block['op_totals'])} |"
        )
    md.append("")

    for key, block in ranked:
        md.append(f"\n## `{key}`\n")
        md.append(
            f"Accuracy: **{block['correct_count']}/{block['total']}**. "
            f"Op totals: `{json.dumps(block['op_totals'])}`.\n"
        )
        md.append("| Scenario | Correct | Hits/Total | Spurious | Op tally | Notes |")
        md.append("|----------|---------|-----------|----------|----------|-------|")
        for pr in block["per_scenario"]:
            j = pr["judge"]
            md.append(
                f"| {pr['scenario_id']} | {j.get('correct')} | "
                f"{j.get('expected_hits')}/{j.get('expected_total')} | "
                f"{j.get('spurious_changes')} | `{json.dumps(pr['op_tally'])}` | "
                f"{j.get('notes', '')[:80]} |"
            )
        md.append("")

    # Per-scenario comparison
    md.append("\n## Side-by-side outputs\n")
    for s in scenarios:
        md.append(f"### {s['id']} — {s['description']}\n")
        md.append(f"Turn: `{s['turn']}`\n")
        md.append(f"Prior (json): `{json.dumps(s['prior_state_json'])}`\n")
        for key, block in all_results["candidates"].items():
            for pr in block["per_scenario"]:
                if pr["scenario_id"] == s["id"]:
                    j = pr["judge"]
                    md.append(
                        f"**`{key}`** — correct={j.get('correct')}  "
                        f"hits={j.get('expected_hits')}/{j.get('expected_total')}  "
                        f"notes: {j.get('notes', '')[:120]}\n"
                    )
                    md.append("```\n" + pr["raw_output"].strip() + "\n```\n")
        md.append("")

    with open(REPORT_FILE, "w") as f:
        f.write("\n".join(md))
    print(f"Saved {REPORT_FILE}")


if __name__ == "__main__":
    main()
