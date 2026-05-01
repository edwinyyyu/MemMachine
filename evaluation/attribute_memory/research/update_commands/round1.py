"""Round 1: compare 5 update-command schema x framing combinations.

Sample: all 12 scenarios in scenarios.json.
Per candidate: run the author prompt for each scenario -> collect commands.
Judging: for each (scenario, candidate), ask gpt-5-mini (separate judge prompt)
         whether the resulting state after applying the commands matches the
         scenario's expected_changes. Judge returns JSON with correctness +
         per-change breakdown + a summary verdict.

We also tally command-type distributions per schema to quantify bias.

Budget: 12 * 5 = 60 author calls + 12 * 5 = 60 judge calls = 120 total.
Stays under 80% of 150-call cap.

Storage representation for prior state: markdown bullet list grouped by
"topic.category" headings (rendered directly from prior_state_markdown).
JSON profile is also available in scenarios.json if a candidate prefers it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
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

CACHE_FILE = CACHE_DIR / "round1_cache.json"
RESULTS_FILE = RESULTS_DIR / "round1_results.json"
REPORT_FILE = RESULTS_DIR / "round1_report.md"


# ---------------------------------------------------------------------------
# Candidate 1 — BASELINE: raw add/delete over (category, attribute, value)
# Framing: direct memory-update system prompt (status quo).
# ---------------------------------------------------------------------------
BASELINE_PROMPT = """\
You maintain a structured memory of facts about a user. Each fact is a row with \
(category, attribute, value).

Given the current memory and a new user turn, emit a JSON array of commands. \
Each command is one of:
  {"op": "add",    "category": "...", "attribute": "...", "value": "..."}
  {"op": "delete", "category": "...", "attribute": "...", "value": "..."}

Delete requires exact (category, attribute, value) match with a currently-stored row.

CURRENT MEMORY:
{prior_state}

USER TURN:
{turn}

Emit ONLY the JSON array. No prose.
"""


# ---------------------------------------------------------------------------
# Candidate 2 — CUD_TRIPLE: add / update / delete + noop
# Framing: "dossier editor" — neutral, not told it's a memory system.
# ---------------------------------------------------------------------------
CUD_PROMPT = """\
You are a careful editor keeping a short dossier about a person up to date. After \
each new statement the person makes, you revise the dossier.

The dossier lists entries grouped by topic.category, one entry per attribute. An \
attribute's value may be a single item or a list of items (a set).

Given the CURRENT DOSSIER and the NEW STATEMENT, emit a JSON array of edits. Each \
edit is exactly one of:

  {"op": "add",    "category": "...", "attribute": "...", "value": "..."}
      // introduce a brand-new attribute
  {"op": "update", "category": "...", "attribute": "...", "value": "..."}
      // replace the current value of an EXISTING attribute
  {"op": "delete", "category": "...", "attribute": "...", "value": "..."}
      // remove an existing attribute entirely
  {"op": "noop"}
      // the statement carries no information worth recording

Notes:
- When an attribute currently holds a list of members, use "update" with the new \
full list (as a JSON array of strings) to add or remove members.
- If the statement changes only the CONFIDENCE of an existing attribute (e.g. \
from hedged to confirmed, or confirmed to hedged), use "update" with the same \
value-meaning and add a trailing qualifier like "(confirmed)" or "(trying / hedged)".
- Match existing attributes by MEANING, not exact text — paraphrases of a stored \
attribute value refer to the same row.

CURRENT DOSSIER:
{prior_state}

NEW STATEMENT:
{turn}

Emit ONLY the JSON array. No prose.
"""


# ---------------------------------------------------------------------------
# Candidate 3 — MEMBER_OPS: add / update / delete + add_member / remove_member + noop
# Framing: "librarian curating a card catalog".
# ---------------------------------------------------------------------------
MEMBER_OPS_PROMPT = """\
You are a librarian maintaining a card catalog about a person. Each card lives \
under a topic.category heading and records one attribute. Some attributes hold a \
single value; others hold a set of members (for example, allergies, pets, hobbies).

The person has just said something. Revise the catalog to reflect it.

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

Guidelines:
- Choose `add_member` / `remove_member` ONLY when the attribute already holds a \
set of members (look at the catalog). For single-valued attributes use update/delete.
- For confidence changes (hedged -> confirmed, confirmed -> hedged) use `update` \
with a value that mentions the new confidence in parentheses, e.g. \
"summited Mt Rainier (confirmed, past)" or "trying vegan for a month (hedged, trial)".
- Match existing cards by MEANING: if the person refers to a stored fact in \
paraphrase, treat it as the same card.
- Prefer fewer, more specific operations over many tiny ones.

CURRENT CATALOG:
{prior_state}

NEW STATEMENT:
{turn}

Emit ONLY the JSON array. No prose.
"""


# ---------------------------------------------------------------------------
# Candidate 4 — INTENT_OPS: intent-level verbs
# Framing: "biographer updating a draft chapter".
# ---------------------------------------------------------------------------
INTENT_OPS_PROMPT = """\
You are a biographer keeping a short factual draft about your subject up to date. \
After each thing the subject says, you emit a list of intents describing how the \
draft should change.

The draft groups facts by topic.category. An attribute's value may be a single \
item or a set of items.

Emit a JSON array of intents, each exactly one of:

  {"op": "introduce_fact",     "category": "...", "attribute": "...", "value": "..."}
      // a new attribute never seen in the draft
  {"op": "correct_fact",       "category": "...", "attribute": "...", "new_value": "..."}
      // revise the value of an existing single-valued attribute
  {"op": "add_to_set",         "category": "...", "attribute": "...", "member": "..."}
      // add one member to an existing set-valued attribute
  {"op": "remove_from_set",    "category": "...", "attribute": "...", "member": "..."}
      // remove one member from an existing set-valued attribute (match by meaning)
  {"op": "retire_fact",        "category": "...", "attribute": "..."}
      // drop the attribute from the draft entirely
  {"op": "strengthen_confidence", "category": "...", "attribute": "...", "new_value": "..."}
      // same-or-similar value, now confirmed / more certain
  {"op": "weaken_confidence",  "category": "...", "attribute": "...", "new_value": "..."}
      // same-or-similar value, now hedged / trial / tentative
  {"op": "nothing_to_change"}
      // the statement is not biographical; nothing to update

Guidelines:
- Pick the MOST SPECIFIC intent that applies. `introduce_fact` is only for truly \
new attributes; if the attribute already exists, use correct_fact / add_to_set / \
remove_from_set / *_confidence / retire_fact instead.
- For a set-valued attribute, NEVER use correct_fact — use add_to_set / \
remove_from_set (one command per member changed).
- Match existing facts by MEANING, not by exact text.

CURRENT DRAFT:
{prior_state}

NEW STATEMENT:
{turn}

Emit ONLY the JSON array. No prose.
"""


# ---------------------------------------------------------------------------
# Candidate 5 — INDEXED_PATCH: numbered existing facts + keep/revise/remove/add
# Framing: "copy editor marking up a numbered fact sheet".
# ---------------------------------------------------------------------------
INDEXED_PATCH_PROMPT = """\
You are a copy editor marking up a numbered fact sheet. Each numbered line is one \
fact, in the form:
  [n] topic.category | attribute: value

A new statement has just arrived. Emit a JSON array of edits. Each edit is one of:

  {"op": "keep", "index": n}
      // the fact is still exactly right; keep it verbatim
  {"op": "revise", "index": n, "new_text": "topic.category | attribute: new_value"}
      // replace the content of fact [n] with new_text (same [n] slot)
  {"op": "remove", "index": n}
      // strike fact [n] out of the sheet
  {"op": "add", "new_text": "topic.category | attribute: value"}
      // append a brand-new numbered line at the end
  {"op": "noop"}
      // the statement does not require any change to the sheet

Guidelines:
- You do NOT need to emit `keep` for facts you are not changing. Only emit `keep` \
if you want to make it explicit that a fact has been reviewed against the new \
statement.
- For set-valued attributes (where the value is a comma-separated list), `revise` \
with the new full comma-separated list.
- Confidence changes: `revise` the line, appending "(confirmed)" or "(hedged)" \
to the value. Match by meaning, not exact text.

CURRENT FACT SHEET:
{prior_state_numbered}

NEW STATEMENT:
{turn}

Emit ONLY the JSON array. No prose.
"""


# ---------------------------------------------------------------------------
# Candidate registry
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    key: str
    name: str
    prompt_template: str
    # which prior-state rendering to feed in: "markdown" or "numbered"
    prior_state_mode: str


CANDIDATES: list[Candidate] = [
    Candidate("baseline_addel", "Baseline add/delete", BASELINE_PROMPT, "markdown"),
    Candidate(
        "cud_triple",
        "add/update/delete + noop (dossier editor)",
        CUD_PROMPT,
        "markdown",
    ),
    Candidate(
        "member_ops",
        "add/update/delete + member ops + noop (librarian)",
        MEMBER_OPS_PROMPT,
        "markdown",
    ),
    Candidate(
        "intent_ops", "intent-level verbs (biographer)", INTENT_OPS_PROMPT, "markdown"
    ),
    Candidate(
        "indexed_patch",
        "numbered-sheet keep/revise/remove/add (copy editor)",
        INDEXED_PATCH_PROMPT,
        "numbered",
    ),
]


# ---------------------------------------------------------------------------
# Prior-state renderers
# ---------------------------------------------------------------------------
def render_numbered(prior_state_json: dict[str, Any]) -> str:
    """Render prior state as numbered fact sheet for indexed_patch candidate."""
    lines: list[str] = []
    i = 1
    if not prior_state_json:
        return "(empty fact sheet)"
    for topic_category, attrs in prior_state_json.items():
        for attribute, value in attrs.items():
            if isinstance(value, list):
                rendered = ", ".join(str(v) for v in value)
            elif isinstance(value, dict) and "value" in value:
                conf = value.get("confidence")
                rendered = f"{value['value']}" + (f" ({conf})" if conf else "")
            else:
                rendered = str(value)
            lines.append(f"[{i}] {topic_category} | {attribute}: {rendered}")
            i += 1
    return "\n".join(lines)


def build_author_prompt(candidate: Candidate, scenario: dict[str, Any]) -> str:
    if candidate.prior_state_mode == "numbered":
        prior = render_numbered(scenario["prior_state_json"])
        return candidate.prompt_template.replace(
            "{prior_state_numbered}", prior
        ).replace("{turn}", scenario["turn"])
    prior = scenario["prior_state_markdown"]
    return candidate.prompt_template.replace("{prior_state}", prior).replace(
        "{turn}", scenario["turn"]
    )


# ---------------------------------------------------------------------------
# Output parsing + command-type tallying
# ---------------------------------------------------------------------------
def extract_json_array(text: str) -> list[Any] | None:
    """Best-effort: pull the first top-level JSON array from text."""
    text = text.strip()
    # Strip code fences if present.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    try:
        v = json.loads(text)
        return v if isinstance(v, list) else None
    except Exception:
        pass
    # Fallback: regex for [ ... ]
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            v = json.loads(m.group(0))
            return v if isinstance(v, list) else None
        except Exception:
            return None
    return None


def tally_ops(commands: list[dict[str, Any]] | None) -> dict[str, int]:
    tally: dict[str, int] = {}
    if not commands:
        return tally
    for c in commands:
        if isinstance(c, dict):
            op = c.get("op", "<missing_op>")
            tally[op] = tally.get(op, 0) + 1
    return tally


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------
JUDGE_PROMPT = """\
You are grading a memory-update system. Given a short statement, the prior memory \
state, and the expected changes, judge whether the emitted commands, if applied, \
would produce the expected resulting state.

SCENARIO DESCRIPTION: {description}
INTENT LABEL (for your information): {intent}

PRIOR MEMORY (JSON):
{prior_state_json}

USER TURN:
{turn}

EXPECTED CHANGES (ground-truth list of logical edits the memory should end up \
reflecting, in any equivalent form):
{expected_changes_json}

EMITTED COMMANDS (JSON array produced by the candidate; treat their schema as \
self-describing, commands may be: add/update/delete/noop, add_member/remove_member, \
introduce_fact/correct_fact/add_to_set/remove_from_set/retire_fact/strengthen_confidence/\
weaken_confidence/nothing_to_change, or keep/revise/remove/add/noop with indices):
{emitted_commands_text}

Grading rubric — respond with ONE JSON object on a single line, no prose:
{{
  "correct": true | false,
  "expected_hits": <int: how many of the expected changes are achieved, 0..N>,
  "expected_total": <int: total expected changes>,
  "spurious_changes": <int: 0..K edits that were NOT in the expected set and \
would alter the state unjustifiably; noop/keep are never spurious>,
  "op_choice_quality": "good" | "wrong_op_type" | "right_effect_wrong_label",
  "notes": "<=30 words on the main failure mode, or '' if correct>"
}}

Rules for grading:
- Paraphrased values that clearly mean the same thing as the ground-truth value \
count as a HIT (e.g. "moved to Seattle for job" ~ "new job in Seattle").
- A no-op / nothing_to_change / empty-commands response is CORRECT only if the \
expected_changes list is empty (or the scenario marks the intent as \
ambiguous/clarify_or_skip).
- For the ambiguous scenario (S10), emitting noop is ACCEPTABLE; removing any \
ONE specific allergy is also ACCEPTABLE; removing ALL three is WRONG.
- If commands failed to parse or are missing, grade as correct=false with \
spurious_changes=0 and hits=0.
"""


def judge(
    client,
    cache: LLMCache,
    budget: CallBudget,
    scenario: dict[str, Any],
    emitted_raw: str,
) -> dict[str, Any]:
    prompt = (
        JUDGE_PROMPT.replace("{description}", scenario["description"])
        .replace("{intent}", scenario["intent"])
        .replace(
            "{prior_state_json}", json.dumps(scenario["prior_state_json"], indent=2)
        )
        .replace("{turn}", scenario["turn"])
        .replace(
            "{expected_changes_json}",
            json.dumps(scenario["expected_changes"], indent=2),
        )
        .replace("{emitted_commands_text}", emitted_raw.strip()[:4000])
    )
    text = llm_call(client, cache, budget, prompt, reasoning_effort="low")
    # Extract first JSON object
    try:
        # strip fences
        t = text.strip()
        if t.startswith("```"):
            t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
            t = re.sub(r"\n?```\s*$", "", t)
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        return json.loads(t)
    except Exception as e:
        return {
            "correct": False,
            "expected_hits": 0,
            "expected_total": len(scenario["expected_changes"]),
            "spurious_changes": 0,
            "op_choice_quality": "wrong_op_type",
            "notes": f"judge-parse-error: {e!r}; raw={text[:200]}",
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    scenarios = load_scenarios()
    print(f"Loaded {len(scenarios)} scenarios.")
    print(
        f"Running {len(CANDIDATES)} candidates x {len(scenarios)} scenarios "
        f"= {len(CANDIDATES) * len(scenarios)} author calls + "
        f"{len(CANDIDATES) * len(scenarios)} judge calls"
    )

    client = make_client()
    cache = LLMCache(CACHE_FILE)
    budget = CallBudget()

    all_results: dict[str, Any] = {
        "model": MODEL,
        "candidates": {},
    }

    try:
        for cand in CANDIDATES:
            print(f"\n=== Candidate: {cand.key} ({cand.name}) ===")
            per_scenario: list[dict[str, Any]] = []
            op_totals: dict[str, int] = {}
            correct_count = 0
            parse_failures = 0

            for s in scenarios:
                author_prompt = build_author_prompt(cand, s)
                raw = llm_call(client, cache, budget, author_prompt)
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
                    f"  [{s['id']}/{s['intent']}] correct={verdict.get('correct')}  "
                    f"hits={verdict.get('expected_hits')}/{verdict.get('expected_total')}  "
                    f"spurious={verdict.get('spurious_changes')}  "
                    f"tally={tally}"
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
                f"= {correct_count / len(scenarios):.0%}; op_totals={op_totals}; "
                f"parse_failures={parse_failures}"
            )
            print(
                f"     budget so far: {budget.made} calls (~${budget.approx_cost():.2f})"
            )
    finally:
        cache.save()
        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_FILE, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved {RESULTS_FILE}")

    # Markdown summary
    md: list[str] = ["# Round 1 — update-command schema comparison\n"]
    md.append(
        f"Model: `{MODEL}`. Scenarios: {len(scenarios)}. "
        f"LLM calls (new): {budget.made} (~${budget.approx_cost():.2f}).\n"
    )
    md.append("## Leaderboard (correct by LLM judge)\n")
    md.append("| Candidate | Correct | Accuracy | Parse Failures | Op totals |")
    md.append("|-----------|---------|----------|----------------|-----------|")
    ranked = sorted(
        all_results["candidates"].items(),
        key=lambda kv: kv[1]["correct_count"],
        reverse=True,
    )
    for key, block in ranked:
        md.append(
            f"| `{key}` | {block['correct_count']}/{block['total']} | "
            f"{block['accuracy']:.0%} | {block['parse_failures']} | "
            f"{json.dumps(block['op_totals'])} |"
        )
    md.append("")

    # Per-candidate detail
    for key, block in ranked:
        md.append(f"\n## `{key}` — {block['name']}\n")
        md.append(
            f"Accuracy: **{block['correct_count']}/{block['total']} "
            f"({block['accuracy']:.0%})**. Op totals: "
            f"`{json.dumps(block['op_totals'])}`. Parse failures: "
            f"{block['parse_failures']}.\n"
        )
        md.append(
            "| Scenario | Intent | Correct | Hits/Total | Spurious | Op tally | Notes |"
        )
        md.append(
            "|----------|--------|---------|-----------|----------|----------|-------|"
        )
        for pr in block["per_scenario"]:
            j = pr["judge"]
            md.append(
                f"| {pr['scenario_id']} | {pr['intent']} | {j.get('correct')} | "
                f"{j.get('expected_hits')}/{j.get('expected_total')} | "
                f"{j.get('spurious_changes')} | `{json.dumps(pr['op_tally'])}` | "
                f"{j.get('notes', '')[:80]} |"
            )
        md.append("")

    # Bias analysis block
    md.append("## Bias analysis\n")
    md.append(
        "Per-schema command-type distribution on the 12-scenario set. "
        "The `add`-only baseline cannot emit update/member/noop verbs at all, so "
        "its bias is structural. For the other schemas, skew toward any single "
        "op indicates schema-induced bias.\n"
    )
    for key, block in ranked:
        md.append(f"- **`{key}`** op totals: `{json.dumps(block['op_totals'])}`")
    md.append("")

    with open(REPORT_FILE, "w") as f:
        f.write("\n".join(md))
    print(f"Saved {REPORT_FILE}")


if __name__ == "__main__":
    main()
