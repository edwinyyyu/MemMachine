"""Round 3: domain-neutral generalization check for the winner (indexed_patch_v2).

We feed the winning prompt (unchanged) a handful of scenarios in domains that
don't appear in Rounds 1-2 (travel logistics, personal finance, workplace
organization, multi-user family profile) to ensure the prompt generalizes and
is not silently overfit to the Round-1 examples it mentions.

Budget: 4 author + 4 judge = 8 calls. Lands at ~146/150.
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
    make_client,
)
from round1 import (
    extract_json_array,
    judge,
    render_numbered,
    tally_ops,
)
from round2 import INDEXED_PATCH_V2_PROMPT

CACHE_FILE = CACHE_DIR / "round3_cache.json"
RESULTS_FILE = RESULTS_DIR / "round3_results.json"
REPORT_FILE = RESULTS_DIR / "round3_report.md"


# Novel domain scenarios — deliberately different flavor from Round 1/2.
DOMAIN_SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "D01_travel_update",
        "intent": "correction",
        "description": "Travel booking update: flight time changed.",
        "prior_state_json": {
            "user.travel": {"upcoming_flight": "SFO -> NRT, Mar 15 9am, seat 12A"},
        },
        "turn": "User: Airline just rescheduled me — new departure is 11:30am, still Mar 15.",
        "expected_changes": [
            {
                "op": "replace",
                "attribute": "upcoming_flight",
                "old_value": "SFO -> NRT, Mar 15 9am, seat 12A",
                "new_value": "SFO -> NRT, Mar 15 11:30am, seat 12A",
            }
        ],
    },
    {
        "id": "D02_finance_setmember",
        "intent": "add_member",
        "description": "Add a new account to a list of financial accounts.",
        "prior_state_json": {
            "user.finance": {"accounts": ["Chase checking", "Vanguard 401k"]},
        },
        "turn": "User: I opened a Fidelity brokerage account this morning to start dollar-cost averaging into index funds.",
        "expected_changes": [
            {
                "op": "add_member",
                "attribute": "accounts",
                "member": "Fidelity brokerage (for DCA into index funds)",
            }
        ],
    },
    {
        "id": "D03_work_noop_venting",
        "intent": "noop",
        "description": "User vents about a meeting — no durable fact.",
        "prior_state_json": {
            "user.work": {"role": "product manager at Figma", "team": "design systems"},
        },
        "turn": "User: this meeting could've been an email, I'm losing my mind.",
        "expected_changes": [],
    },
    {
        "id": "D04_family_multi_member",
        "intent": "multi_op",
        "description": "Multi-user/family: remove one child from school list when they graduate; add a new hobby.",
        "prior_state_json": {
            "user.family": {"children_in_school": ["Maya (grade 8)", "Noah (grade 5)"]},
            "user.hobbies": {"hobbies": ["birding", "pickleball"]},
        },
        "turn": "User: Maya just graduated 8th grade so she's off to high school now, and I picked up woodworking last month.",
        "expected_changes": [
            {
                "op": "remove_member",
                "attribute": "children_in_school",
                "member": "Maya (grade 8)",
            },
            {
                "op": "add_or_update_member",
                "attribute": "children_in_school",
                "member": "Maya (high school)",
                "note": "acceptable either to add Maya (high school) or to not re-add her; must remove grade-8 entry",
            },
            {"op": "add_member", "attribute": "hobbies", "member": "woodworking"},
        ],
    },
]


def build_winner_prompt(scenario: dict[str, Any]) -> str:
    numbered = render_numbered(scenario["prior_state_json"])
    return INDEXED_PATCH_V2_PROMPT.replace("{prior_state_numbered}", numbered).replace(
        "{turn}", scenario["turn"]
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"Round 3: {len(DOMAIN_SCENARIOS)} domain-distinct scenarios for indexed_patch_v2"
    )

    client = make_client()
    cache = LLMCache(CACHE_FILE)
    budget = CallBudget(max_calls=150, stop_at=148)
    budget.made = 138  # From Round 2

    per_scenario: list[dict[str, Any]] = []
    op_totals: dict[str, int] = {}
    correct_count = 0
    parse_failures = 0

    for s in DOMAIN_SCENARIOS:
        prompt = build_winner_prompt(s)
        raw = llm_call(client, cache, budget, prompt)
        cmds = extract_json_array(raw)
        if cmds is None:
            parse_failures += 1
        tally = tally_ops(cmds)
        for op, n in tally.items():
            op_totals[op] = op_totals.get(op, 0) + n

        # Judge needs a prior_state_markdown rendering — build a cheap one.
        s_for_judge = dict(s)
        s_for_judge["prior_state_markdown"] = render_numbered(s["prior_state_json"])
        verdict = judge(client, cache, budget, s_for_judge, raw)
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
            f"spurious={verdict.get('spurious_changes')}  tally={tally}  "
            f"notes={verdict.get('notes', '')[:80]}"
        )
        cache.save()

    result = {
        "model": MODEL,
        "candidate": "indexed_patch_v2 (winner)",
        "correct_count": correct_count,
        "total": len(DOMAIN_SCENARIOS),
        "accuracy": round(correct_count / len(DOMAIN_SCENARIOS), 3),
        "parse_failures": parse_failures,
        "op_totals": op_totals,
        "per_scenario": per_scenario,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved {RESULTS_FILE}")
    print(f"Budget used: {budget.made}/150 (~${budget.approx_cost():.2f})")

    # Markdown
    md = ["# Round 3 — domain-neutral generalization check (indexed_patch_v2)\n"]
    md.append(
        f"Winner from Round 2 (`indexed_patch_v2`) tested on {len(DOMAIN_SCENARIOS)} "
        "scenarios with distinctly different domain flavor (travel, finance, "
        "work-venting, multi-user family). The prompt itself was NOT altered.\n"
    )
    md.append(
        f"Accuracy: **{correct_count}/{len(DOMAIN_SCENARIOS)} = "
        f"{correct_count / len(DOMAIN_SCENARIOS):.0%}**. "
        f"Op totals: `{json.dumps(op_totals)}`.\n"
    )
    md.append(
        "| Scenario | Intent | Correct | Hits/Total | Spurious | Op tally | Notes |"
    )
    md.append(
        "|----------|--------|---------|-----------|----------|----------|-------|"
    )
    for pr in per_scenario:
        j = pr["judge"]
        md.append(
            f"| {pr['scenario_id']} | {pr['intent']} | {j.get('correct')} | "
            f"{j.get('expected_hits')}/{j.get('expected_total')} | "
            f"{j.get('spurious_changes')} | `{json.dumps(pr['op_tally'])}` | "
            f"{j.get('notes', '')[:80]} |"
        )
    md.append("")

    md.append("\n## Per-scenario outputs\n")
    for s, pr in zip(DOMAIN_SCENARIOS, per_scenario):
        md.append(f"### {s['id']} — {s['description']}\n")
        md.append(f"Turn: `{s['turn']}`\n")
        md.append(f"Prior: `{json.dumps(s['prior_state_json'])}`\n")
        md.append("```\n" + pr["raw_output"].strip() + "\n```")
        md.append(f"Judge: `{json.dumps(pr['judge'])}`\n")

    with open(REPORT_FILE, "w") as f:
        f.write("\n".join(md))
    print(f"Saved {REPORT_FILE}")


if __name__ == "__main__":
    main()
