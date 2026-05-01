"""Deterministic appliers for Round 5 schemas.

Two families:

1. Row-based (S1 baseline, S2 upsert-replace, S3 upsert-only). State is a
   list of rows of shape
     {topic_category, attribute, value, confidence, cardinality}.
   Commands patch rows directly.

2. Append-log (S4 append-ref, S5 append-plain). State is a list of log
   entries of shape
     {id, ts, topic, text, refs: list[int], relation: str|None,
      invalidated: bool}.
   Current-state is derived by a materializer that:
     - For each (topic, inferred attribute) key, walks the log forward
       collecting non-invalidated claims.
     - Emits one row per key; the latest claim wins for scalars; for sets
       we union members minus invalidated ones.

The materializer is simple but sufficient for our 14 scenarios; we don't
try to model arbitrary inference. In doubt, the materializer prefers to
emit the latest non-invalidated claim text. This makes the append-only
schemas tractable for deterministic evaluation.
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

CONF_TAGS = {"confirmed", "hedged", "intended", "negated"}


# ---------------------------------------------------------------------------
# Row normalization (shared with round4_fine_ops.py)
# ---------------------------------------------------------------------------
def normalize_row(r: dict[str, Any]) -> dict[str, Any]:
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
    na = {(r["topic_category"], r["attribute"]): normalize_row(r) for r in applied}
    nb = {(r["topic_category"], r["attribute"]): normalize_row(r) for r in expected}
    missing = [list(k) for k in nb if k not in na]
    extra = [list(k) for k in na if k not in nb]
    wrong = []
    for k in nb:
        if k in na and na[k] != nb[k]:
            wrong.append({"key": list(k), "got": na[k], "want": nb[k]})
    return {"missing": missing, "extra": extra, "wrong": wrong}


# ---------------------------------------------------------------------------
# Fact-sheet rendering
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
            rendered = ", ".join(v) if isinstance(v, list) else str(v)
            cardlen = f" (n={len(v) if isinstance(v, list) else 0} total)"
        else:
            rendered = str(v)
            cardlen = ""
        tag = f" ({conf})" if conf else ""
        card_hint = f"  [cardinality={card}]" if show_cardinality else ""
        lines.append(f"[{i}] {tc} | {attr}: {rendered}{tag}{cardlen}{card_hint}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fact-line parser
# ---------------------------------------------------------------------------
def parse_fact_line(text: str) -> tuple[str, str, str, str | None]:
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


# ---------------------------------------------------------------------------
# S1 / row-based applier (baseline from round4)
# ---------------------------------------------------------------------------
@dataclass
class ApplyReport:
    rows: list[dict[str, Any]]
    errors: list[str] = field(default_factory=list)


def apply_row_commands(
    rows_before: list[dict[str, Any]],
    commands: list[dict[str, Any]],
) -> ApplyReport:
    """Apply S1-family row commands. Supports revise/remove/add/noop +
    add_member/remove_member + upsert."""
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

    def _find_by_key(tc: str, attr: str) -> int | None:
        """Return 1-based index of live row matching (tc, attr)."""
        for i, r in enumerate(rows, start=1):
            if r is None:
                continue
            if r["topic_category"].strip() == tc and r["attribute"].strip() == attr:
                return i
        return None

    for cmd in commands:
        if not isinstance(cmd, dict):
            errors.append(f"non-dict command: {cmd!r}")
            continue
        op = cmd.get("op")
        try:
            if op in ("noop", "keep"):
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
                    tc, attr, val, conf = parse_fact_line(new_text)
                except ValueError as e:
                    errors.append(f"revise[{idx}] parse: {e}")
                    continue
                r["topic_category"] = tc
                r["attribute"] = attr
                if conf is not None:
                    r["confidence"] = conf
                if r.get("cardinality") == "set":
                    r["value"] = [m.strip() for m in val.split(",") if m.strip()]
                else:
                    r["value"] = val

            elif op == "add":
                new_text = cmd.get("new_text") or ""
                try:
                    tc, attr, val, conf = parse_fact_line(new_text)
                except ValueError as e:
                    errors.append(f"add parse: {e}")
                    continue
                card = cmd.get("cardinality")
                if card not in ("single", "set"):
                    card = "set" if "," in val else "single"
                if card == "set":
                    value: Any = [m.strip() for m in val.split(",") if m.strip()]
                else:
                    value = val
                rows.append(
                    {
                        "topic_category": tc,
                        "attribute": attr,
                        "value": value,
                        "confidence": conf or "confirmed",
                        "cardinality": card,
                    }
                )

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

            elif op == "upsert":
                # {"op":"upsert","topic_category":..., "attribute":...,
                #  "value":..., "cardinality":..., "confidence":...}
                tc = str(cmd.get("topic_category", "")).strip()
                attr = str(cmd.get("attribute", "")).strip()
                if not tc or not attr:
                    errors.append(
                        f"upsert missing topic_category or attribute: {cmd!r}"
                    )
                    continue
                val = cmd.get("value", "")
                card = cmd.get("cardinality")
                if card not in ("single", "set"):
                    if isinstance(val, list) or (isinstance(val, str) and "," in val):
                        card = "set"
                    else:
                        card = "single"
                conf = cmd.get("confidence") or "confirmed"
                if card == "set":
                    if isinstance(val, str):
                        value_list = [m.strip() for m in val.split(",") if m.strip()]
                    else:
                        value_list = list(val) if isinstance(val, list) else [str(val)]
                    value_final: Any = value_list
                else:
                    value_final = str(val)
                existing = _find_by_key(tc, attr)
                if existing is None:
                    rows.append(
                        {
                            "topic_category": tc,
                            "attribute": attr,
                            "value": value_final,
                            "confidence": conf,
                            "cardinality": card,
                        }
                    )
                else:
                    r = rows[existing - 1]
                    if r is None:
                        continue
                    r["value"] = value_final
                    r["confidence"] = conf
                    r["cardinality"] = card

            else:
                errors.append(f"unknown op: {op!r}")
        except Exception as e:
            errors.append(f"apply {op}: {e!r}")

    resulting = [r for r in rows if r is not None]
    return ApplyReport(rows=resulting, errors=errors)


# ---------------------------------------------------------------------------
# S4/S5 append-log applier + materializer
# ---------------------------------------------------------------------------
@dataclass
class LogEntry:
    id: int
    topic: str
    text: str
    refs: list[int]
    relation: str | None  # clarify / refine / supersede / invalidate / None
    invalidated: bool = False


def apply_log_commands(
    log_before: list[LogEntry],
    commands: list[dict[str, Any]],
) -> tuple[list[LogEntry], list[str]]:
    """Apply append commands. Returns (new_log, errors).

    Supported ops:
      - {"op": "append", "topic": str, "text": str}
      - {"op": "append_ref", "topic": str, "refs": [int,...], "relation":
         "clarify"|"refine"|"supersede"|"invalidate", "text": str}
      - {"op": "noop"}
    """
    log: list[LogEntry] = [
        LogEntry(**vars(e)) if isinstance(e, LogEntry) else LogEntry(**e)
        for e in log_before
    ]
    errors: list[str] = []
    next_id = max([e.id for e in log], default=0) + 1

    for cmd in commands:
        if not isinstance(cmd, dict):
            errors.append(f"non-dict: {cmd!r}")
            continue
        op = cmd.get("op")
        try:
            if op == "noop" or op == "keep":
                continue
            if op == "append":
                topic = str(cmd.get("topic", "")).strip() or "user.general"
                text = str(cmd.get("text", "")).strip()
                if not text:
                    errors.append("append empty text")
                    continue
                log.append(
                    LogEntry(id=next_id, topic=topic, text=text, refs=[], relation=None)
                )
                next_id += 1
            elif op == "append_ref":
                topic = str(cmd.get("topic", "")).strip() or "user.general"
                text = str(cmd.get("text", "")).strip()
                relation = cmd.get("relation")
                refs_raw = cmd.get("refs") or cmd.get("ref_ids") or []
                if not isinstance(refs_raw, list):
                    errors.append("append_ref refs not list")
                    continue
                refs = [
                    int(x)
                    for x in refs_raw
                    if isinstance(x, (int, str)) and str(x).isdigit()
                ]
                if not text:
                    errors.append("append_ref empty text")
                    continue
                if relation not in ("clarify", "refine", "supersede", "invalidate"):
                    errors.append(f"bad relation: {relation!r}")
                    relation = relation or "clarify"
                new_entry = LogEntry(
                    id=next_id, topic=topic, text=text, refs=refs, relation=relation
                )
                log.append(new_entry)
                next_id += 1
                # If relation is invalidate/supersede, mark the referenced
                # entries as invalidated (and transitively, for supersede
                # it's still a single-hop, we don't chain).
                if relation in ("invalidate", "supersede"):
                    for rid in refs:
                        for e in log:
                            if e.id == rid:
                                e.invalidated = True
            else:
                errors.append(f"unknown op: {op!r}")
        except Exception as e:
            errors.append(f"apply {op}: {e!r}")

    return log, errors


def render_log(log: list[LogEntry]) -> str:
    """Render the log for the LLM as numbered entries, including invalidation."""
    if not log:
        return "(empty log)"
    lines: list[str] = []
    for e in log:
        ref_info = ""
        if e.refs:
            rel = e.relation or "ref"
            ref_info = f" [{rel} of {', '.join(str(r) for r in e.refs)}]"
        invtag = " [INVALIDATED]" if e.invalidated else ""
        lines.append(f"[{e.id}] ({e.topic}) {e.text}{ref_info}{invtag}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Materializer: log -> current state (rows)
# ---------------------------------------------------------------------------
# This is intentionally heuristic. The append schemas can't be evaluated
# with strict row-equality because their "state" is emergent. We provide
# two kinds of evaluation:
#
# 1. Materialize to a set of (topic, loose_key) claims. Check that each
#    expected fact is present (by substring match in the latest
#    non-invalidated claim for that rough key) and that retracted facts
#    are either absent or marked invalidated.
#
# 2. For the retrieval probe (T13), check that the old fact is still in
#    the log (non-invalidated entries are OK too -- we just need it to
#    be recoverable if someone asks "what did they used to say?").


def materialize_to_claims(log: list[LogEntry]) -> list[dict[str, Any]]:
    """Return the live claims (non-invalidated entries in chronological
    order). Used for substring-based fact checks."""
    return [
        {
            "id": e.id,
            "topic": e.topic,
            "text": e.text,
            "refs": e.refs,
            "relation": e.relation,
        }
        for e in log
        if not e.invalidated
    ]


def all_claims(log: list[LogEntry]) -> list[dict[str, Any]]:
    """Return every entry, including invalidated. For retrieval probes."""
    return [
        {
            "id": e.id,
            "topic": e.topic,
            "text": e.text,
            "refs": e.refs,
            "relation": e.relation,
            "invalidated": e.invalidated,
        }
        for e in log
    ]


# ---------------------------------------------------------------------------
# Loose grading for multi-turn and append-only scenarios
# ---------------------------------------------------------------------------
def loose_fact_present(claim_texts: list[str], fact_keywords: list[str]) -> bool:
    """A fact is 'present' if at least one claim contains ALL the keywords."""
    for text in claim_texts:
        low = text.lower()
        if all(kw.lower() in low for kw in fact_keywords):
            return True
    return False


def loose_fact_absent_or_retracted(
    live_texts: list[str], all_entries: list[dict], fact_keywords: list[str]
) -> bool:
    """Return True if the fact is either not in live text OR only appears
    in an invalidated entry."""
    if any(all(kw.lower() in t.lower() for kw in fact_keywords) for t in live_texts):
        return False
    return True  # Not in live texts -> absent or retracted


def grade_loose_multi_turn(
    applied_rows: list[dict[str, Any]],
    log: list[LogEntry] | None,
    scenario: dict[str, Any],
) -> dict[str, Any]:
    """Loose grading for multi-turn scenarios using keyword matching."""
    scenario_id = scenario["id"]
    grade = {"scenario_id": scenario_id, "checks": [], "pass": False}

    if scenario_id == "T11_paraphrased_correction_after_chain":
        # Required facts: gym_days includes Tue, Wed, Fri (or the set), leg_day=Tuesday,
        # gym=Equinox, wednesday involves swim.
        checks = []
        # 1. Leg day = Tuesday (not Monday)
        checks.append(
            _check_fact(
                applied_rows,
                log,
                fact_keywords=["tuesday"],
                attribute_hints=["leg"],
                label="leg_day is Tuesday",
            )
        )
        checks.append(
            _check_fact_absent(
                applied_rows,
                log,
                wrong_keywords=["leg", "monday"],
                label="leg_day is NOT Monday",
            )
        )
        # 2. Gym = Equinox
        checks.append(
            _check_fact(
                applied_rows,
                log,
                fact_keywords=["equinox"],
                attribute_hints=["gym"],
                label="gym is Equinox",
            )
        )
        # 3. Wednesdays involve swim
        checks.append(
            _check_fact(
                applied_rows,
                log,
                fact_keywords=["swim"],
                attribute_hints=["wednesday", "swim"],
                label="Wednesdays involve swimming",
            )
        )
        grade["checks"] = checks
        grade["pass"] = all(c["pass"] for c in checks)

    elif scenario_id == "T12_chain_with_retraction":
        checks = []
        # 1. Peanut allergy is retracted/negated/absent
        checks.append(
            _check_fact_absent(
                applied_rows,
                log,
                wrong_keywords=["peanut", "allerg"],
                label="peanut allergy is retracted",
            )
        )
        # 2. Lactose intolerant remains
        checks.append(
            _check_fact(
                applied_rows,
                log,
                fact_keywords=["lactose"],
                attribute_hints=["lactose", "intoler"],
                label="lactose intolerance retained",
            )
        )
        grade["checks"] = checks
        grade["pass"] = all(c["pass"] for c in checks)

    elif scenario_id == "T14_long_chain_preference_evolution":
        checks = []
        checks.append(
            _check_fact(
                applied_rows,
                log,
                fact_keywords=["half marathon"],
                attribute_hints=["half", "marathon", "goal"],
                label="half marathon goal",
            )
        )
        checks.append(
            _check_fact(
                applied_rows,
                log,
                fact_keywords=["10"],  # 10 mile
                attribute_hints=["run", "long"],
                label="10-mile long run",
            )
        )
        # Running frequency 4-5 days/week
        checks.append(
            _check_fact_any(
                applied_rows,
                log,
                keyword_sets=[["4"], ["5"], ["four"], ["five"]],
                attribute_hints=["run", "freq", "days"],
                label="running 4-5 days/week",
            )
        )
        grade["checks"] = checks
        grade["pass"] = all(c["pass"] for c in checks)

    elif scenario_id == "T13_retrieval_probe":
        # Need both: current state = Portland; old state (Seattle) still recoverable
        current_ok = _check_fact(
            applied_rows,
            log,
            fact_keywords=["portland"],
            attribute_hints=["home", "city", "location"],
            label="current home = Portland",
        )["pass"]
        seattle_present = False
        if log is not None:
            seattle_present = any("seattle" in e.text.lower() for e in log)
        else:
            # For row-based: need a bi-temporal note OR a prior-version row. Our
            # appliers don't track versions, so this is always False for row-based
            # in the current setup -- which is itself a finding.
            seattle_present = False
        grade["checks"] = [
            {"label": "current_portland", "pass": current_ok},
            {"label": "seattle_recoverable", "pass": seattle_present},
        ]
        grade["pass"] = current_ok and seattle_present

    return grade


def _stringify_row(r: dict) -> str:
    v = r.get("value")
    vs = ", ".join(v) if isinstance(v, list) else str(v)
    return f"{r.get('topic_category', '')} {r.get('attribute', '')} {vs} {r.get('confidence', '')}"


def _check_fact(applied_rows, log, fact_keywords, attribute_hints=None, label=""):
    """Check that any row/live-entry contains all fact_keywords."""
    texts = [_stringify_row(r).lower() for r in applied_rows]
    if log is not None:
        texts += [e.text.lower() for e in log if not e.invalidated]
    found = any(all(kw.lower() in t for kw in fact_keywords) for t in texts)
    return {"label": label, "pass": found, "keywords": fact_keywords}


def _check_fact_any(applied_rows, log, keyword_sets, attribute_hints=None, label=""):
    """Pass if ANY keyword_set is satisfied."""
    texts = [_stringify_row(r).lower() for r in applied_rows]
    if log is not None:
        texts += [e.text.lower() for e in log if not e.invalidated]
    for ks in keyword_sets:
        if any(all(kw.lower() in t for kw in ks) for t in texts):
            return {"label": label, "pass": True, "keywords": ks}
    return {"label": label, "pass": False, "keywords": keyword_sets}


def _check_fact_absent(applied_rows, log, wrong_keywords, label=""):
    """Pass if NO live row/entry has all wrong_keywords. For retractions:
    peanut allergy should not appear in LIVE rows (row-based) or in
    non-invalidated log entries (append)."""
    texts = [_stringify_row(r).lower() for r in applied_rows]
    # For row-based: negated confidence counts as 'absent'
    live_non_negated = [
        _stringify_row(r).lower()
        for r in applied_rows
        if r.get("confidence") != "negated"
    ]
    if log is not None:
        live_log_texts = [e.text.lower() for e in log if not e.invalidated]
    else:
        live_log_texts = []
    all_live = live_non_negated + live_log_texts
    leak = any(all(kw.lower() in t for kw in wrong_keywords) for t in all_live)
    return {"label": label, "pass": not leak, "wrong_keywords": wrong_keywords}
