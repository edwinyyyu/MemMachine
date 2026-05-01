"""Round 4: do finer-grained update commands beat the 4-op baseline?

Round 1-3 settled on a 4-op baseline (revise[n]/remove[n]/add/noop) over a
numbered fact sheet. This round asks whether adding ANY finer-grained op
buys us correctness, or whether it just invites verb-bias and new failure
modes.

Candidates (all share the same numbered fact-sheet rendering + framing;
only the verb set + per-verb instructions differ):

  C1 baseline      — revise[n], remove[n], add, noop (winner of R1-3)
  C2 +member_ops   — adds add_member[n] / remove_member[n] (only for rows
                     whose attribute is declared cardinality=set)
  C3 +string_patch — adds patch[n] old_substring new_substring
  C4 +append_to    — adds append_to[n] new_text (appends qualifiers to
                     existing value)
  C5 +conf_verbs   — adds strengthen[n] / weaken[n] (change only the
                     (confirmed)/(hedged)/(intended) tag)

Methodology
-----------

Rather than trust an LLM-as-judge (Round 1 judge was too lenient on
paraphrase), we build a tiny deterministic applier. Each op is a pure
function on a rows dict. We compare the resulting rows to the scenario's
`expected_rows_after` via strict normalized diff. This is the single
most important change from Round 1-3.

We measure, per (candidate, scenario):
  1. state_correct: does applied state match expected state exactly?
  2. op tally: what verbs did the author emit?
  3. for C3 only: did any patch's old_substring actually appear in the
     target row's value? (paraphrase-miss failure)
  4. estimated token cost of the emitted op list (len of raw output).

Budget: 5 candidates x 10 scenarios x 1 author call = 50 calls. No judge
calls (applier is deterministic). Hard stop at 80 calls (80% of 100).
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
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

CACHE_FILE = CACHE_DIR / "round4_cache.json"
RESULTS_FILE = RESULTS_DIR / "round4_results.json"
REPORT_FILE = RESULTS_DIR / "round4_report.md"


# ---------------------------------------------------------------------------
# Row model
# ---------------------------------------------------------------------------
# A "row" is (topic_category, attribute, value, confidence, cardinality).
# We carry cardinality in scenario data so member-op candidates know which
# rows accept add_member/remove_member.
#
# A scenario is a dict:
#   {
#     "id": str,
#     "description": str,
#     "intent": str,
#     "rows_before": list[Row],
#     "turn": str,
#     "rows_after": list[Row],      # expected resulting rows (order-insensitive for sets)
#   }
#
# Row dict:
#   {"topic_category": str, "attribute": str, "value": str | list[str],
#    "confidence": str | None, "cardinality": "single" | "set"}

CONF_TAGS = {"confirmed", "hedged", "intended"}


def normalize_row(r: dict[str, Any]) -> dict[str, Any]:
    """Normalize row for comparison: sort set values, strip spaces."""
    out = {
        "topic_category": r["topic_category"].strip(),
        "attribute": r["attribute"].strip(),
        "confidence": r.get("confidence"),
        "cardinality": r.get("cardinality", "single"),
    }
    v = r["value"]
    if out["cardinality"] == "set":
        if isinstance(v, str):
            v = [m.strip() for m in v.split(",") if m.strip()]
        out["value"] = sorted(v)
    else:
        out["value"] = str(v).strip() if not isinstance(v, list) else ", ".join(v)
    return out


def rows_equal(a: list[dict], b: list[dict]) -> bool:
    """Order-insensitive rows equality after normalization."""
    na = sorted(
        [normalize_row(r) for r in a],
        key=lambda r: (r["topic_category"], r["attribute"]),
    )
    nb = sorted(
        [normalize_row(r) for r in b],
        key=lambda r: (r["topic_category"], r["attribute"]),
    )
    return na == nb


def rows_diff(applied: list[dict], expected: list[dict]) -> dict[str, Any]:
    """Return a small structured diff between applied and expected."""
    na = {(r["topic_category"], r["attribute"]): normalize_row(r) for r in applied}
    nb = {(r["topic_category"], r["attribute"]): normalize_row(r) for r in expected}
    missing = [k for k in nb if k not in na]
    extra = [k for k in na if k not in nb]
    wrong = []
    for k in nb:
        if k in na and na[k] != nb[k]:
            wrong.append({"key": k, "got": na[k], "want": nb[k]})
    return {"missing": missing, "extra": extra, "wrong": wrong}


# ---------------------------------------------------------------------------
# Fact-sheet rendering: the LLM sees rows numbered [1]..[N]
# ---------------------------------------------------------------------------
def render_sheet(rows: list[dict[str, Any]], show_cardinality: bool = False) -> str:
    if not rows:
        return "(empty fact sheet)"
    lines: list[str] = []
    for i, r in enumerate(rows, start=1):
        tc = r["topic_category"]
        attr = r["attribute"]
        v = r["value"]
        card = r.get("cardinality", "single")
        conf = r.get("confidence")
        if card == "set":
            if isinstance(v, list):
                rendered = ", ".join(v)
            else:
                rendered = str(v)
        else:
            rendered = str(v)
        tag = f" ({conf})" if conf else ""
        card_hint = f"  [cardinality={card}]" if show_cardinality else ""
        lines.append(f"[{i}] {tc} | {attr}: {rendered}{tag}{card_hint}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Deterministic applier
# ---------------------------------------------------------------------------
@dataclass
class ApplyReport:
    rows: list[dict[str, Any]]
    errors: list[str] = field(default_factory=list)


def _parse_fact_line(text: str) -> tuple[str, str, str, str | None]:
    """Parse 'topic.category | attribute: value (conf)' into parts.

    Returns (topic_category, attribute, value, confidence_or_None).
    Strict: must contain ' | ' and ': '.
    """
    if " | " not in text:
        raise ValueError(f"missing ' | ' in {text!r}")
    tc, rest = text.split(" | ", 1)
    if ": " not in rest:
        raise ValueError(f"missing ': ' in {text!r}")
    attr, val = rest.split(": ", 1)
    val = val.strip()
    conf: str | None = None
    m = re.search(r"\s*\(([a-z]+)\)\s*$", val)
    if m and m.group(1).lower() in CONF_TAGS:
        conf = m.group(1).lower()
        val = val[: m.start()].strip()
    return tc.strip(), attr.strip(), val, conf


def apply_commands(
    rows_before: list[dict[str, Any]],
    commands: list[dict[str, Any]],
) -> ApplyReport:
    """Apply commands to rows_before and return resulting rows + errors.

    Row indices in commands are 1-based into rows_before (as rendered).
    Removals produce tombstones so subsequent indices remain stable.
    New rows from `add` are appended at the end (not indexable by prior ops).
    """
    # Copy and mark rows to preserve index stability.
    rows: list[dict[str, Any] | None] = [deepcopy(r) for r in rows_before]
    errors: list[str] = []

    def _get(idx: int) -> dict[str, Any] | None:
        if not isinstance(idx, int) or idx < 1 or idx > len(rows):
            errors.append(f"bad index {idx}")
            return None
        r = rows[idx - 1]
        if r is None:
            errors.append(f"index {idx} already removed")
            return None
        return r

    for cmd in commands:
        if not isinstance(cmd, dict):
            errors.append(f"non-dict command: {cmd!r}")
            continue
        op = cmd.get("op")
        try:
            if op == "noop" or op == "keep":
                continue

            if op == "remove":
                idx = cmd.get("index")
                if _get(idx) is None:
                    continue
                rows[idx - 1] = None

            elif op == "revise":
                idx = cmd.get("index")
                r = _get(idx)
                if r is None:
                    continue
                new_text = cmd.get("new_text") or ""
                try:
                    tc, attr, val, conf = _parse_fact_line(new_text)
                except ValueError as e:
                    errors.append(f"revise[{idx}] parse error: {e}")
                    continue
                r["topic_category"] = tc
                r["attribute"] = attr
                # Preserve prior confidence when new_text has no explicit tag.
                # (Author sees "(confirmed)" in the sheet; re-specifying on
                # every revise would be unnecessarily verbose.)
                if conf is not None:
                    r["confidence"] = conf
                # Preserve cardinality; re-split sets.
                if r.get("cardinality") == "set":
                    r["value"] = [m.strip() for m in val.split(",") if m.strip()]
                else:
                    r["value"] = val

            elif op == "add":
                new_text = cmd.get("new_text") or ""
                try:
                    tc, attr, val, conf = _parse_fact_line(new_text)
                except ValueError as e:
                    errors.append(f"add parse error: {e}")
                    continue
                # Default cardinality: if value has a comma => set, else single.
                card = "set" if "," in val else "single"
                value: str | list[str]
                if card == "set":
                    value = [m.strip() for m in val.split(",") if m.strip()]
                else:
                    value = val
                rows.append(
                    {
                        "topic_category": tc,
                        "attribute": attr,
                        "value": value,
                        "confidence": conf,
                        "cardinality": card,
                    }
                )

            # ----- C2: member ops -----
            elif op == "add_member":
                idx = cmd.get("index")
                r = _get(idx)
                if r is None:
                    continue
                if r.get("cardinality") != "set":
                    errors.append(f"add_member[{idx}] on non-set row")
                    continue
                member = str(cmd.get("member", "")).strip()
                if not member:
                    errors.append(f"add_member[{idx}] empty member")
                    continue
                if isinstance(r["value"], str):
                    r["value"] = [m.strip() for m in r["value"].split(",") if m.strip()]
                if member not in r["value"]:
                    r["value"].append(member)

            elif op == "remove_member":
                idx = cmd.get("index")
                r = _get(idx)
                if r is None:
                    continue
                if r.get("cardinality") != "set":
                    errors.append(f"remove_member[{idx}] on non-set row")
                    continue
                member = str(cmd.get("member", "")).strip()
                if isinstance(r["value"], str):
                    r["value"] = [m.strip() for m in r["value"].split(",") if m.strip()]
                # Meaning-loose match: exact first, else substring.
                if member in r["value"]:
                    r["value"].remove(member)
                else:
                    cand = [m for m in r["value"] if member.lower() in m.lower()]
                    if len(cand) == 1:
                        r["value"].remove(cand[0])
                    else:
                        errors.append(
                            f"remove_member[{idx}]: no unique match for {member!r}"
                        )

            # ----- C3: string patch -----
            elif op == "patch":
                idx = cmd.get("index")
                r = _get(idx)
                if r is None:
                    continue
                old_s = cmd.get("old_substring") or ""
                new_s = cmd.get("new_substring") or ""
                cur = r["value"]
                if isinstance(cur, list):
                    cur_str = ", ".join(cur)
                else:
                    cur_str = str(cur)
                if old_s not in cur_str:
                    errors.append(f"patch[{idx}]: old_substring {old_s!r} not in value")
                    continue
                new_val = cur_str.replace(old_s, new_s, 1)
                if r.get("cardinality") == "set":
                    r["value"] = [m.strip() for m in new_val.split(",") if m.strip()]
                else:
                    r["value"] = new_val

            # ----- C4: append_to -----
            elif op == "append_to":
                idx = cmd.get("index")
                r = _get(idx)
                if r is None:
                    continue
                suffix = cmd.get("new_text") or ""
                cur = r["value"]
                if isinstance(cur, list):
                    cur_str = ", ".join(cur)
                else:
                    cur_str = str(cur)
                sep = " " if not cur_str.endswith(" ") else ""
                new_val = f"{cur_str}{sep}{suffix}".strip()
                if r.get("cardinality") == "set":
                    # append_to on a set: append as new member
                    if isinstance(r["value"], str):
                        r["value"] = [
                            m.strip() for m in r["value"].split(",") if m.strip()
                        ]
                    r["value"].append(suffix.strip())
                else:
                    r["value"] = new_val

            # ----- C5: confidence verbs -----
            elif op == "strengthen":
                idx = cmd.get("index")
                r = _get(idx)
                if r is None:
                    continue
                # Advance confidence one step: hedged/intended -> confirmed
                r["confidence"] = "confirmed"

            elif op == "weaken":
                idx = cmd.get("index")
                r = _get(idx)
                if r is None:
                    continue
                # Weaken: confirmed -> hedged (trial-like)
                r["confidence"] = "hedged"

            else:
                errors.append(f"unknown op: {op!r}")
        except Exception as e:
            errors.append(f"apply {op}: {e!r}")

    resulting = [r for r in rows if r is not None]
    return ApplyReport(rows=resulting, errors=errors)


# ---------------------------------------------------------------------------
# Scenarios (10 — targeted to the fine-op question)
# ---------------------------------------------------------------------------
# Every row has topic_category, attribute, value, confidence, cardinality.

SCENARIOS: list[dict[str, Any]] = [
    # --- R01: Confidence change only (value body the same) ---
    {
        "id": "R01_conf_strengthen_only",
        "intent": "strengthen_confidence",
        "description": "User confirms a trait they previously hedged; value body unchanged.",
        "rows_before": [
            {
                "topic_category": "user.diet",
                "attribute": "diet",
                "value": "vegan",
                "confidence": "hedged",
                "cardinality": "single",
            },
        ],
        "turn": "User: I've been fully vegan for 3 years now, it's definitely sticking.",
        # Expected: same value text, confidence -> confirmed.
        "rows_after": [
            {
                "topic_category": "user.diet",
                "attribute": "diet",
                "value": "vegan",
                "confidence": "confirmed",
                "cardinality": "single",
            },
        ],
    },
    # --- R02: Small qualifier add ("peanut allergy" -> "...(severe)") ---
    {
        "id": "R02_small_qualifier_add",
        "intent": "append_qualifier",
        "description": "User adds a severity qualifier to an existing single-value fact.",
        "rows_before": [
            {
                "topic_category": "user.health",
                "attribute": "peanut_allergy",
                "value": "peanut allergy",
                "confidence": "confirmed",
                "cardinality": "single",
            },
        ],
        "turn": "User: By the way, my peanut allergy is severe — anaphylactic.",
        "rows_after": [
            {
                "topic_category": "user.health",
                "attribute": "peanut_allergy",
                "value": "peanut allergy (severe, anaphylactic)",
                "confidence": "confirmed",
                "cardinality": "single",
            },
        ],
    },
    # --- R03: Set member add (short set) ---
    {
        "id": "R03_set_add_member",
        "intent": "add_member",
        "description": "User adds a third allergen to a 2-member set.",
        "rows_before": [
            {
                "topic_category": "user.health",
                "attribute": "allergies",
                "value": ["peanuts", "shellfish"],
                "confidence": "confirmed",
                "cardinality": "set",
            },
        ],
        "turn": "User: Oh, I forgot — I also react to tree nuts, especially cashews.",
        "rows_after": [
            {
                "topic_category": "user.health",
                "attribute": "allergies",
                "value": ["peanuts", "shellfish", "tree nuts"],
                "confidence": "confirmed",
                "cardinality": "set",
            },
        ],
    },
    # --- R04: Set member remove ---
    {
        "id": "R04_set_remove_member",
        "intent": "remove_member",
        "description": "User reports a pet died; remove that one member only.",
        "rows_before": [
            {
                "topic_category": "user.pets",
                "attribute": "pets",
                "value": ["Luna (dog)", "Milo (cat)", "Rex (dog)"],
                "confidence": "confirmed",
                "cardinality": "set",
            },
        ],
        "turn": "User: Sad news this week — Milo passed away. Hardest part of the year.",
        "rows_after": [
            {
                "topic_category": "user.pets",
                "attribute": "pets",
                "value": ["Luna (dog)", "Rex (dog)"],
                "confidence": "confirmed",
                "cardinality": "set",
            },
        ],
    },
    # --- R05: Whole-value replace (cheap, simple) ---
    {
        "id": "R05_whole_replace",
        "intent": "correction",
        "description": "User corrects a single-value field; whole replacement is the cleanest.",
        "rows_before": [
            {
                "topic_category": "user.location",
                "attribute": "home_city",
                "value": "Seattle",
                "confidence": "confirmed",
                "cardinality": "single",
            },
        ],
        "turn": "User: Correction — I live in Portland now, moved last month.",
        "rows_after": [
            {
                "topic_category": "user.location",
                "attribute": "home_city",
                "value": "Portland",
                "confidence": "confirmed",
                "cardinality": "single",
            },
        ],
    },
    # --- R06: Multiple small edits across the sheet ---
    {
        "id": "R06_multi_small_edits",
        "intent": "multi_op",
        "description": "Two small changes in one turn: qualifier add + set member add.",
        "rows_before": [
            {
                "topic_category": "user.health",
                "attribute": "peanut_allergy",
                "value": "peanut allergy",
                "confidence": "confirmed",
                "cardinality": "single",
            },
            {
                "topic_category": "user.pets",
                "attribute": "pets",
                "value": ["Luna (dog)"],
                "confidence": "confirmed",
                "cardinality": "set",
            },
        ],
        "turn": "User: My peanut allergy is severe, by the way. Also, we just adopted a kitten named Tofu.",
        "rows_after": [
            {
                "topic_category": "user.health",
                "attribute": "peanut_allergy",
                "value": "peanut allergy (severe)",
                "confidence": "confirmed",
                "cardinality": "single",
            },
            {
                "topic_category": "user.pets",
                "attribute": "pets",
                "value": ["Luna (dog)", "Tofu (kitten)"],
                "confidence": "confirmed",
                "cardinality": "set",
            },
        ],
    },
    # --- R07: Looks like a small tweak but is actually a full replace ---
    {
        "id": "R07_looks_small_is_full",
        "intent": "replace_trap",
        "description": "Sounds like a qualifier, but the person's role and team both change.",
        "rows_before": [
            {
                "topic_category": "user.employment",
                "attribute": "role",
                "value": "senior software engineer on Search team",
                "confidence": "confirmed",
                "cardinality": "single",
            },
        ],
        "turn": "User: Big news — promoted and moved. I'm a principal engineer on the Ads platform team now.",
        # Patch-style "senior -> principal" and "Search -> Ads" misses the
        # platform bit; correct move is a whole-row revise.
        "rows_after": [
            {
                "topic_category": "user.employment",
                "attribute": "role",
                "value": "principal engineer on Ads platform team",
                "confidence": "confirmed",
                "cardinality": "single",
            },
        ],
    },
    # --- R08: LONG set; full-rewrite is token-expensive ---
    {
        "id": "R08_long_set_one_change",
        "intent": "add_member",
        "description": "10-member set; user adds just one new book.",
        "rows_before": [
            {
                "topic_category": "user.books",
                "attribute": "favorite_books",
                "value": [
                    "The Name of the Wind",
                    "Dune",
                    "Gödel, Escher, Bach",
                    "Neuromancer",
                    "The Left Hand of Darkness",
                    "Cloud Atlas",
                    "Piranesi",
                    "House of Leaves",
                    "A Fire Upon the Deep",
                    "Station Eleven",
                ],
                "confidence": "confirmed",
                "cardinality": "set",
            },
        ],
        "turn": "User: Oh, I just finished 'The Bone Clocks' by Mitchell — I'd add it to my favorites list.",
        "rows_after": [
            {
                "topic_category": "user.books",
                "attribute": "favorite_books",
                "value": [
                    "The Name of the Wind",
                    "Dune",
                    "Gödel, Escher, Bach",
                    "Neuromancer",
                    "The Left Hand of Darkness",
                    "Cloud Atlas",
                    "Piranesi",
                    "House of Leaves",
                    "A Fire Upon the Deep",
                    "Station Eleven",
                    "The Bone Clocks",
                ],
                "confidence": "confirmed",
                "cardinality": "set",
            },
        ],
    },
    # --- R09: Weaken confidence only ---
    {
        "id": "R09_conf_weaken_only",
        "intent": "weaken_confidence",
        "description": "Firm claim is now hedged; value body unchanged.",
        "rows_before": [
            {
                "topic_category": "user.fitness",
                "attribute": "running_routine",
                "value": "runs 5km every morning",
                "confidence": "confirmed",
                "cardinality": "single",
            },
        ],
        "turn": "User: Been slacking lately — the morning run is more aspirational than actual these days.",
        "rows_after": [
            {
                "topic_category": "user.fitness",
                "attribute": "running_routine",
                "value": "runs 5km every morning",
                "confidence": "hedged",
                "cardinality": "single",
            },
        ],
    },
    # --- R10: Patch-friendly surface edit (word-swap). Baseline must revise. ---
    {
        "id": "R10_surface_word_swap",
        "intent": "patch",
        "description": "One word changes in a long value string (language learning).",
        "rows_before": [
            {
                "topic_category": "user.learning",
                "attribute": "language_study",
                "value": "learning Spanish with Duolingo for 15 minutes every weekday",
                "confidence": "confirmed",
                "cardinality": "single",
            },
        ],
        "turn": "User: Switched from Duolingo to Anki for Spanish study — same schedule though.",
        "rows_after": [
            {
                "topic_category": "user.learning",
                "attribute": "language_study",
                "value": "learning Spanish with Anki for 15 minutes every weekday",
                "confidence": "confirmed",
                "cardinality": "single",
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# Candidate prompts
# ---------------------------------------------------------------------------

SHARED_FRAMING = """\
You are a copy editor marking up a numbered fact sheet. Each numbered line is one
fact, in the form:
  [n] topic.category | attribute: value (confidence_tag)

Confidence tag — if present, it is EXACTLY one of: (confirmed), (hedged), (intended).

Before emitting any edit, decide: does this statement contain something that
BELONGS on a permanent fact sheet about the person? If not, emit noop.

DO NOT write to the sheet (emit noop instead):
- Weather comments, seasonal gripes, chitchat.
- Transient moods or fleeting reactions.
- Generic filler / acknowledgements.
- Repetitions of facts already on the sheet that add no new detail.

DO write to the sheet:
- Durable attributes, preferences, traits, plans/events.
- Any correction, addition, or removal to an existing fact.

Match facts by MEANING, not exact text. An ambiguous referent -> prefer noop.
When multiple distinct changes land in one turn, emit multiple edits.
"""


# C1: baseline (winner of Round 1-3)
C1_PROMPT = (
    SHARED_FRAMING
    + """
Edit verbs:

  {"op": "revise", "index": n, "new_text": "topic.category | attribute: new_value"}
      // replace the content of fact [n] entirely. For set-valued attributes
      // (comma-separated list values), revise with the new full list.
  {"op": "remove", "index": n}
      // strike fact [n].
  {"op": "add", "new_text": "topic.category | attribute: value"}
      // append a brand-new numbered line.
  {"op": "noop"}
      // the statement does not require any change.

Confidence changes: revise the line and end the value with exactly one of
(confirmed), (hedged), (intended). E.g. "user.diet | diet: vegan (confirmed)".

CURRENT FACT SHEET:
{prior_state_numbered}

NEW STATEMENT:
{turn}

Emit ONLY the JSON array. No prose.
"""
)


# C2: + member_ops
C2_PROMPT = (
    SHARED_FRAMING
    + """
Some fact-sheet rows are declared cardinality=set (see tag in the sheet).
For those rows you may use member-level ops; for single-valued rows you must not.

Edit verbs:

  {"op": "revise", "index": n, "new_text": "topic.category | attribute: new_value"}
      // replace the content of fact [n] entirely.
  {"op": "remove", "index": n}
      // strike fact [n].
  {"op": "add", "new_text": "topic.category | attribute: value"}
      // append a brand-new numbered line.
  {"op": "add_member",    "index": n, "member": "..."}
      // add one member to a set-valued row [n].
  {"op": "remove_member", "index": n, "member": "..."}
      // remove one member from a set-valued row [n] (match by meaning).
  {"op": "noop"}

Use add_member / remove_member ONLY when cardinality=set for row [n].

Confidence changes: revise the line and end the value with exactly one of
(confirmed), (hedged), (intended).

CURRENT FACT SHEET (each line tagged with [cardinality=single|set]):
{prior_state_numbered}

NEW STATEMENT:
{turn}

Emit ONLY the JSON array. No prose.
"""
)


# C3: + string_patch
C3_PROMPT = (
    SHARED_FRAMING
    + """
Edit verbs:

  {"op": "revise", "index": n, "new_text": "topic.category | attribute: new_value"}
      // replace the content of fact [n] entirely.
  {"op": "remove", "index": n}
      // strike fact [n].
  {"op": "add", "new_text": "topic.category | attribute: value"}
      // append a brand-new numbered line.
  {"op": "patch",  "index": n, "old_substring": "...", "new_substring": "..."}
      // surgical edit: literally replace the FIRST occurrence of old_substring
      // in the value of row [n] with new_substring. old_substring MUST be a
      // verbatim substring of the current value — if you paraphrase, the
      // patch fails. Prefer revise when the change is large or paraphrased.
  {"op": "noop"}

Confidence changes: revise the line and end the value with exactly one of
(confirmed), (hedged), (intended).

CURRENT FACT SHEET:
{prior_state_numbered}

NEW STATEMENT:
{turn}

Emit ONLY the JSON array. No prose.
"""
)


# C4: + append_to
C4_PROMPT = (
    SHARED_FRAMING
    + """
Edit verbs:

  {"op": "revise", "index": n, "new_text": "topic.category | attribute: new_value"}
      // replace the content of fact [n] entirely.
  {"op": "remove", "index": n}
      // strike fact [n].
  {"op": "add", "new_text": "topic.category | attribute: value"}
      // append a brand-new numbered line.
  {"op": "append_to", "index": n, "new_text": "..."}
      // append new_text to the END of the value of row [n], preserving what
      // is already there. Use this when you are ONLY adding detail — a
      // qualifier, a parenthetical, or (for set-valued rows) a new member.
  {"op": "noop"}

Confidence changes: revise the line and end the value with exactly one of
(confirmed), (hedged), (intended).

CURRENT FACT SHEET:
{prior_state_numbered}

NEW STATEMENT:
{turn}

Emit ONLY the JSON array. No prose.
"""
)


# C5: + confidence verbs
C5_PROMPT = (
    SHARED_FRAMING
    + """
Edit verbs:

  {"op": "revise", "index": n, "new_text": "topic.category | attribute: new_value"}
      // replace the content of fact [n] entirely.
  {"op": "remove", "index": n}
      // strike fact [n].
  {"op": "add", "new_text": "topic.category | attribute: value"}
      // append a brand-new numbered line.
  {"op": "strengthen", "index": n}
      // set row [n]'s confidence tag to (confirmed). Use ONLY when the
      // value text is unchanged and only certainty has increased.
  {"op": "weaken", "index": n}
      // set row [n]'s confidence tag to (hedged). Use ONLY when the value
      // text is unchanged and only certainty has decreased.
  {"op": "noop"}

For any change that also alters the value text, use revise instead of
strengthen/weaken and end the value with exactly one of (confirmed), (hedged),
(intended).

CURRENT FACT SHEET:
{prior_state_numbered}

NEW STATEMENT:
{turn}

Emit ONLY the JSON array. No prose.
"""
)


@dataclass
class Candidate:
    key: str
    name: str
    prompt_template: str
    show_cardinality: bool = False


CANDIDATES: list[Candidate] = [
    Candidate("C1_baseline", "baseline (revise/remove/add/noop)", C1_PROMPT),
    Candidate(
        "C2_member_ops",
        "+ add_member / remove_member",
        C2_PROMPT,
        show_cardinality=True,
    ),
    Candidate("C3_string_patch", "+ patch old_substring new_substring", C3_PROMPT),
    Candidate("C4_append_to", "+ append_to", C4_PROMPT),
    Candidate("C5_conf_verbs", "+ strengthen / weaken", C5_PROMPT),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_author_prompt(cand: Candidate, scenario: dict[str, Any]) -> str:
    sheet = render_sheet(
        scenario["rows_before"], show_cardinality=cand.show_cardinality
    )
    return cand.prompt_template.replace("{prior_state_numbered}", sheet).replace(
        "{turn}", scenario["turn"]
    )


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


def count_patch_paraphrase_misses(
    commands: list[dict[str, Any]] | None,
    rows_before: list[dict[str, Any]],
) -> int:
    """For C3 only: count how many patch ops have an old_substring that is
    not a verbatim substring of the target row's value string."""
    if not commands:
        return 0
    n = 0
    for c in commands:
        if not isinstance(c, dict):
            continue
        if c.get("op") != "patch":
            continue
        idx = c.get("index")
        if not isinstance(idx, int) or idx < 1 or idx > len(rows_before):
            n += 1
            continue
        r = rows_before[idx - 1]
        cur = r["value"]
        cur_s = ", ".join(cur) if isinstance(cur, list) else str(cur)
        old_s = c.get("old_substring") or ""
        if old_s not in cur_s:
            n += 1
    return n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    client = make_client()
    cache = LLMCache(CACHE_FILE)
    # Override budget to Round-4 spec: 100 hard cap, stop at 80 (80%).
    budget = CallBudget(max_calls=100, stop_at=80)

    n = len(CANDIDATES) * len(SCENARIOS)
    print(
        f"Round 4: {len(CANDIDATES)} candidates x {len(SCENARIOS)} scenarios = {n} author calls."
    )
    print(f"No judge calls (deterministic applier). Hard stop at {budget.stop_at}.")

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
            paraphrase_misses = 0
            total_output_chars = 0

            for s in SCENARIOS:
                prompt = build_author_prompt(cand, s)
                raw = llm_call(client, cache, budget, prompt)
                total_output_chars += len(raw)
                cmds = extract_json_array(raw)
                if cmds is None:
                    parse_failures += 1
                tally = tally_ops(cmds)
                for op, k in tally.items():
                    op_totals[op] = op_totals.get(op, 0) + k

                if cand.key == "C3_string_patch":
                    paraphrase_misses += count_patch_paraphrase_misses(
                        cmds, s["rows_before"]
                    )

                applied = apply_commands(s["rows_before"], cmds or [])
                correct = rows_equal(applied.rows, s["rows_after"])
                if correct:
                    correct_count += 1
                diff = rows_diff(applied.rows, s["rows_after"]) if not correct else None

                per_scenario.append(
                    {
                        "scenario_id": s["id"],
                        "intent": s["intent"],
                        "description": s["description"],
                        "raw_output": raw,
                        "parsed_commands": cmds,
                        "op_tally": tally,
                        "apply_errors": applied.errors,
                        "state_correct": correct,
                        "diff": diff,
                        "output_chars": len(raw),
                    }
                )
                print(
                    f"  [{s['id']}] correct={correct}  "
                    f"tally={tally}  errors={len(applied.errors)}  "
                    f"out_chars={len(raw)}"
                )
                cache.save()

            all_results["candidates"][cand.key] = {
                "name": cand.name,
                "correct_count": correct_count,
                "total": len(SCENARIOS),
                "accuracy": round(correct_count / len(SCENARIOS), 3),
                "parse_failures": parse_failures,
                "paraphrase_misses": paraphrase_misses,  # C3 only, 0 otherwise
                "op_totals": op_totals,
                "total_output_chars": total_output_chars,
                "avg_output_chars": total_output_chars / max(1, len(SCENARIOS)),
                "per_scenario": per_scenario,
            }
            print(
                f"  -> {cand.key}: correct {correct_count}/{len(SCENARIOS)} "
                f"= {correct_count / len(SCENARIOS):.0%}; op_totals={op_totals}; "
                f"parse_failures={parse_failures}; paraphrase_misses={paraphrase_misses}; "
                f"avg_out_chars={total_output_chars / max(1, len(SCENARIOS)):.0f}"
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

    write_report(all_results)


def write_report(all_results: dict[str, Any]) -> None:
    md: list[str] = ["# Round 4 — fine-grained update operations\n"]
    md.append(
        "Question: beyond the 4-op baseline (`revise[n]` whole-value replace, "
        "`remove[n]`, `add`, `noop`), do ANY finer-grained verbs improve "
        "update quality on gpt-5-mini?\n"
    )
    md.append(
        f"Model: `{MODEL}`. Scenarios: {len(SCENARIOS)}. Grading: deterministic "
        "applier + strict diff vs. expected_rows_after (not LLM judge).\n"
    )

    # Leaderboard
    md.append("## Leaderboard (state-correctness on applier)\n")
    md.append(
        "| Candidate | Correct | Accuracy | Parse Fails | Paraphrase Misses | Avg out chars | Op totals |"
    )
    md.append(
        "|-----------|---------|----------|-------------|-------------------|---------------|-----------|"
    )
    ranked = sorted(
        all_results["candidates"].items(),
        key=lambda kv: kv[1]["correct_count"],
        reverse=True,
    )
    for key, block in ranked:
        md.append(
            f"| `{key}` | {block['correct_count']}/{block['total']} | "
            f"{block['accuracy']:.0%} | {block['parse_failures']} | "
            f"{block['paraphrase_misses']} | {block['avg_output_chars']:.0f} | "
            f"`{json.dumps(block['op_totals'])}` |"
        )
    md.append("")

    # Per-candidate detail
    for key, block in ranked:
        md.append(f"\n## `{key}` — {block['name']}\n")
        md.append(
            f"- Correct: **{block['correct_count']}/{block['total']} "
            f"({block['accuracy']:.0%})**\n"
            f"- Op totals: `{json.dumps(block['op_totals'])}`\n"
            f"- Parse failures: {block['parse_failures']}\n"
            f"- Paraphrase misses (C3 only): {block['paraphrase_misses']}\n"
            f"- Avg output chars (token-cost proxy): {block['avg_output_chars']:.0f}\n"
        )
        md.append(
            "| Scenario | Intent | Correct | Op tally | Apply errors | Diff summary |"
        )
        md.append(
            "|----------|--------|---------|----------|--------------|--------------|"
        )
        for pr in block["per_scenario"]:
            diff = pr["diff"]
            if diff is None:
                diff_summary = "—"
            else:
                bits = []
                if diff["missing"]:
                    bits.append(f"missing={diff['missing']}")
                if diff["extra"]:
                    bits.append(f"extra={diff['extra']}")
                if diff["wrong"]:
                    bits.append(f"wrong={len(diff['wrong'])}")
                diff_summary = "; ".join(bits) or "eq_rows_but_unequal"
            errs = pr["apply_errors"]
            errs_s = "; ".join(errs)[:80] if errs else ""
            md.append(
                f"| {pr['scenario_id']} | {pr['intent']} | "
                f"{pr['state_correct']} | `{json.dumps(pr['op_tally'])}` | "
                f"{errs_s} | {diff_summary} |"
            )
        md.append("")

    # Failure-mode / bias notes
    md.append("## Bias & failure-mode notes\n")
    for key, block in ranked:
        md.append(
            f"- `{key}` ({block['correct_count']}/{block['total']}): "
            f"ops=`{json.dumps(block['op_totals'])}`, "
            f"parse_fail={block['parse_failures']}, "
            f"paraphrase_miss={block['paraphrase_misses']}, "
            f"avg_out={block['avg_output_chars']:.0f}c"
        )
    md.append("")

    # Recommendation stub — we fill in based on data after run.
    md.append("## Recommendation\n")
    ranked_keys = [k for k, _ in ranked]
    best = ranked_keys[0]
    baseline = "C1_baseline"
    baseline_block = all_results["candidates"][baseline]
    best_block = all_results["candidates"][best]
    md.append(
        f"- Winner: `{best}` ({best_block['correct_count']}/{best_block['total']} correct; "
        f"avg {best_block['avg_output_chars']:.0f} chars/turn).\n"
        f"- Baseline `C1_baseline` = {baseline_block['correct_count']}/{baseline_block['total']} "
        f"correct; avg {baseline_block['avg_output_chars']:.0f} chars/turn.\n"
    )
    md.append(
        "Interpret with care: correctness on 10 scenarios has large CI. "
        "Prefer schemas that match baseline on correctness AND reduce "
        "token cost on long-set operations (R08), OR improve on a clear "
        "failure mode without introducing parse or bias regressions.\n"
    )

    with open(REPORT_FILE, "w") as f:
        f.write("\n".join(md))
    print(f"Saved {REPORT_FILE}")


if __name__ == "__main__":
    main()
