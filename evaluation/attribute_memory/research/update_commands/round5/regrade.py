"""Post-hoc regrader for Round 5.

The initial grader was too strict in a few ways:

1. For append-only schemas on T12 (retraction chain), any mention of the
   word "peanut" in the live log caused a fail -- even when the final
   entry explicitly said "does NOT have a peanut allergy". The test
   should require: "do we correctly know, after reading the live log,
   that the user does NOT currently have a peanut allergy?"

2. For append-ref schemas on T11 (leg_day correction), the word
   "Mondays" appears in the gym_days entry. That's fine -- Mondays is a
   gym day. The leg_day check only needs to verify that the CORRECTED
   claim (Tuesday) is more recent than the original (Monday) claim.

3. For append-plain (S5), there's no relation mechanism, so the reader
   must derive state. We should credit "most-recent claim about X"
   rather than "any claim ever made about X".

4. For row schemas: if a row has confidence="negated", the fact is
   retracted. The grader should treat it that way.

We re-grade every per-scenario result using these improved rules. Output
goes to round5_regraded.json and round5_regraded_report.md.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_FILE = HERE / "results" / "round5_results.json"
OUT_RESULTS = HERE / "results" / "round5_regraded.json"
OUT_REPORT = HERE / "results" / "round5_regraded_report.md"


def latest_claim_about(
    entries: list[dict], keywords: list[str], invalidated_ok: bool = False
) -> dict | None:
    """Return the most recent non-invalidated entry whose text contains all keywords."""
    matches = []
    for e in entries:
        if not invalidated_ok and e.get("invalidated"):
            continue
        txt = e["text"].lower()
        if all(kw.lower() in txt for kw in keywords):
            matches.append(e)
    return matches[-1] if matches else None


def re_grade_row_scenario(scenario_id: str, per_scen_entry: dict) -> dict:
    """Regrade a row-family scenario. Use existing diff for single-turn;
    for multi-turn use the 'grade' fields but with row inspection."""
    rows = per_scen_entry.get("applied_rows") or per_scen_entry.get("final_rows") or []

    # For multi-turn scenarios, regrade based on final rows.
    if scenario_id == "T11_paraphrased_correction_after_chain":
        checks = []
        # leg_day currently = Tuesday
        leg_day_row = next(
            (
                r
                for r in rows
                if "leg" in r.get("attribute", "").lower()
                or "leg" in str(r.get("value", "")).lower()
            ),
            None,
        )
        if leg_day_row:
            v = str(leg_day_row.get("value", "")).lower()
            checks.append(
                {"label": "leg_day is Tuesday", "pass": "tuesday" in v, "detail": v}
            )
            checks.append(
                {
                    "label": "leg_day is NOT Monday",
                    "pass": "monday" not in v,
                    "detail": v,
                }
            )
        else:
            # Check all rows
            combined = " ".join(
                str(r.get("value", "")).lower()
                for r in rows
                if "leg" in (r.get("attribute", "") + str(r.get("value", ""))).lower()
            )
            checks.append(
                {"label": "leg_day is Tuesday", "pass": "tuesday" in combined}
            )
            checks.append(
                {"label": "leg_day is NOT Monday", "pass": "monday" not in combined}
            )
        gym = next(
            (
                r
                for r in rows
                if "gym" in r.get("attribute", "").lower()
                and "equinox" in str(r.get("value", "")).lower()
            ),
            None,
        )
        checks.append({"label": "gym is Equinox", "pass": gym is not None})
        swim_row = next(
            (
                r
                for r in rows
                if "swim" in str(r.get("value", "")).lower()
                or "wednesday" in r.get("attribute", "").lower()
            ),
            None,
        )
        if swim_row:
            checks.append(
                {
                    "label": "Wednesday swim",
                    "pass": "swim" in str(swim_row.get("value", "")).lower(),
                }
            )
        else:
            combined = " ".join(str(r.get("value", "")).lower() for r in rows)
            checks.append({"label": "Wednesday swim", "pass": "swim" in combined})
        all_pass = all(c["pass"] for c in checks)
        return {"scenario_id": scenario_id, "checks": checks, "pass": all_pass}

    if scenario_id == "T12_chain_with_retraction":
        checks = []
        # lactose intolerance retained
        lactose_row = next(
            (
                r
                for r in rows
                if "lactose" in r.get("attribute", "").lower()
                or "lactose" in str(r.get("value", "")).lower()
            ),
            None,
        )
        checks.append({"label": "lactose retained", "pass": lactose_row is not None})
        # peanut allergy retracted: either (a) no row contains peanut allergy, or
        # (b) the row's confidence is "negated"
        peanut_live = [
            r
            for r in rows
            if (
                "peanut" in r.get("attribute", "").lower()
                or "peanut" in str(r.get("value", "")).lower()
            )
            and r.get("confidence") != "negated"
        ]

        # Exclude "does not have peanut allergy" phrasings
        def is_retraction(r):
            v = str(r.get("value", "")).lower()
            a = r.get("attribute", "").lower()
            return any(
                tok in v
                for tok in [
                    "none",
                    "no allergy",
                    "not",
                    "negated",
                    "retract",
                    "ruled out",
                    "misdiagnos",
                ]
            ) or a in ("none",)

        peanut_actually_alive = [r for r in peanut_live if not is_retraction(r)]
        checks.append(
            {
                "label": "peanut allergy retracted",
                "pass": len(peanut_actually_alive) == 0,
                "detail": [r for r in peanut_actually_alive],
            }
        )
        return {
            "scenario_id": scenario_id,
            "checks": checks,
            "pass": all(c["pass"] for c in checks),
        }

    if scenario_id == "T14_long_chain_preference_evolution":
        checks = []
        combined_text = " ".join(
            f"{r.get('attribute', '')} {r.get('value', '')}" for r in rows
        ).lower()
        checks.append(
            {
                "label": "half marathon goal",
                "pass": "half marathon" in combined_text
                or "half-marathon" in combined_text,
            }
        )
        checks.append(
            {
                "label": "10-mile long run",
                "pass": "10" in combined_text or "ten" in combined_text,
            }
        )
        # frequency
        checks.append(
            {
                "label": "4-5 days/week",
                "pass": any(k in combined_text for k in ["4 ", "5 ", "four ", "five "]),
            }
        )
        return {
            "scenario_id": scenario_id,
            "checks": checks,
            "pass": all(c["pass"] for c in checks),
        }

    if scenario_id == "T13_retrieval_probe":
        # Current = Portland
        portland = any("portland" in str(r.get("value", "")).lower() for r in rows)
        # For row-based: seattle is lost (unless bi-temporal stores it).
        # Our applier doesn't store history, so seattle_recoverable = False.
        return {
            "scenario_id": scenario_id,
            "checks": [
                {"label": "current_portland", "pass": portland},
                {
                    "label": "seattle_recoverable",
                    "pass": False,
                    "note": "row-family: not recoverable; bi-temporal needed",
                },
            ],
            "pass": False,
        }  # Always fails for row-family (no history)

    # Single-turn: keep the original state_correct result.
    return {
        "scenario_id": scenario_id,
        "checks": [
            {"label": "original", "pass": per_scen_entry.get("state_correct", False)}
        ],
        "pass": per_scen_entry.get("state_correct", False),
    }


def re_grade_log_scenario(scenario_id: str, per_scen_entry: dict) -> dict:
    log = per_scen_entry.get("log") or per_scen_entry.get("final_log") or []

    if scenario_id == "T11_paraphrased_correction_after_chain":
        # For log schemas: find the latest non-invalidated claim about leg day.
        # Prefer a claim that has BOTH "leg" and "day" or "monday"/"tuesday".
        live = [e for e in log if not e.get("invalidated")]

        # Latest claim mentioning leg day.
        leg_claims = [e for e in live if "leg" in e["text"].lower()]
        latest_leg = leg_claims[-1] if leg_claims else None
        if latest_leg:
            leg_txt = latest_leg["text"].lower()
            is_tue = "tuesday" in leg_txt
            # Is monday mentioned *as being* leg day? The correction entry says
            # "leg day is on Tuesdays, not Mondays" -- monday is negated.
            # Accept if the entry says Tuesday AND (no monday OR in 'not monday' form).
            monday_context_ok = (
                "not mondays" in leg_txt
                or "not monday" in leg_txt
                or "monday" not in leg_txt
            )
            is_correct = is_tue and monday_context_ok
        else:
            is_correct = False
        checks = [
            {
                "label": "latest leg claim = Tuesday",
                "pass": is_correct,
                "latest": latest_leg["text"] if latest_leg else None,
            }
        ]

        # Gym = Equinox (latest gym claim)
        gym_claims = [
            e
            for e in live
            if "gym" in e["text"].lower()
            or "equinox" in e["text"].lower()
            or "24 hour" in e["text"].lower()
        ]
        latest_gym = gym_claims[-1] if gym_claims else None
        checks.append(
            {
                "label": "gym is Equinox",
                "pass": latest_gym is not None
                and "equinox" in latest_gym["text"].lower(),
            }
        )
        # Wednesday swim
        swim = any("swim" in e["text"].lower() for e in live)
        checks.append({"label": "Wednesday swim", "pass": swim})

        return {
            "scenario_id": scenario_id,
            "checks": checks,
            "pass": all(c["pass"] for c in checks),
        }

    if scenario_id == "T12_chain_with_retraction":
        live = [e for e in log if not e.get("invalidated")]
        # lactose intolerance
        lactose_live = any("lactose" in e["text"].lower() for e in live)
        # peanut allergy: if the latest claim says "does not" or similar, it's retracted.
        peanut_claims = [e for e in live if "peanut" in e["text"].lower()]
        if peanut_claims:
            last = peanut_claims[-1]["text"].lower()
            is_retracted = any(
                tok in last
                for tok in [
                    "does not",
                    "do not",
                    "not have",
                    "no peanut",
                    "not a peanut",
                    "negated",
                    "retract",
                    "misdiagnos",
                    "was not",
                    "isn't",
                ]
            )
        else:
            is_retracted = True
        return {
            "scenario_id": scenario_id,
            "checks": [
                {"label": "lactose retained", "pass": lactose_live},
                {
                    "label": "peanut allergy retracted",
                    "pass": is_retracted,
                    "latest_peanut": peanut_claims[-1]["text"]
                    if peanut_claims
                    else None,
                },
            ],
            "pass": lactose_live and is_retracted,
        }

    if scenario_id == "T14_long_chain_preference_evolution":
        live = [e for e in log if not e.get("invalidated")]
        combined = " ".join(e["text"].lower() for e in live)
        checks = [
            {
                "label": "half marathon goal",
                "pass": "half marathon" in combined or "half-marathon" in combined,
            },
            {
                "label": "10-mile long run",
                "pass": "10" in combined or "ten" in combined,
            },
            {
                "label": "4-5 days/week",
                "pass": any(k in combined for k in ["4 ", "5 ", "four ", "five "]),
            },
        ]
        return {
            "scenario_id": scenario_id,
            "checks": checks,
            "pass": all(c["pass"] for c in checks),
        }

    if scenario_id == "T13_retrieval_probe":
        # Current = Portland; old Seattle still in log
        live = [e for e in log if not e.get("invalidated")]
        portland = any("portland" in e["text"].lower() for e in live)
        # seattle_recoverable = anywhere in log, invalidated or not
        seattle = any("seattle" in e["text"].lower() for e in log)
        return {
            "scenario_id": scenario_id,
            "checks": [
                {"label": "current_portland", "pass": portland},
                {"label": "seattle_recoverable", "pass": seattle},
            ],
            "pass": portland and seattle,
        }

    # Single-turn: use original state_correct from the simpler grader.
    return {
        "scenario_id": scenario_id,
        "checks": [
            {"label": "original", "pass": per_scen_entry.get("state_correct", False)}
        ],
        "pass": per_scen_entry.get("state_correct", False),
    }


def main() -> None:
    with open(RESULTS_FILE) as f:
        results = json.load(f)

    regraded = {"model": results.get("model"), "candidates": {}}

    for cand_key, block in results["candidates"].items():
        family = block.get("schema_family", "row")
        new_per_scen = []
        correct = 0
        for pr in block["per_scenario"]:
            sid = pr["scenario_id"]
            if family == "log":
                rg = re_grade_log_scenario(sid, pr)
            else:
                rg = re_grade_row_scenario(sid, pr)
            pr2 = {**pr, "regrade": rg, "state_correct_regraded": rg["pass"]}
            new_per_scen.append(pr2)
            if rg["pass"]:
                correct += 1
        regraded["candidates"][cand_key] = {
            **block,
            "correct_count_regraded": correct,
            "accuracy_regraded": round(correct / len(new_per_scen), 3),
            "per_scenario": new_per_scen,
        }

    OUT_RESULTS.write_text(json.dumps(regraded, indent=2, default=str))

    # Report
    md: list[str] = ["# Round 5 -- Regraded Results\n"]
    md.append(
        "Regrading addresses three grader bugs in the original run:\n\n"
        "1. Log-based retraction check was failing on any entry mentioning the retracted topic, "
        "including the retraction entry itself. Fixed to check the LATEST live claim's semantics.\n"
        "2. T11 leg-day check was too coarse (substring of 'monday' in the gym_days entry). "
        "Fixed to look at the latest claim specifically about leg day.\n"
        "3. Row-family retraction-via-confidence=negated was counted as 'still present'. "
        "Fixed to treat negated rows as retracted.\n\n"
        "Single-turn scenarios keep their original grading from the deterministic applier.\n"
    )
    md.append("## Leaderboard (regraded)\n")
    md.append("| Candidate | Family | Correct | Accuracy |")
    md.append("|-----------|--------|---------|----------|")
    ranked = sorted(
        regraded["candidates"].items(),
        key=lambda kv: kv[1]["correct_count_regraded"],
        reverse=True,
    )
    for key, block in ranked:
        md.append(
            f"| `{key}` | {block.get('schema_family', '?')} | "
            f"{block['correct_count_regraded']}/{len(block['per_scenario'])} | "
            f"{block['accuracy_regraded']:.0%} |"
        )
    md.append("")

    md.append("## Per-scenario correctness (regraded)\n")
    sids = [
        s["scenario_id"]
        for s in next(iter(regraded["candidates"].values()))["per_scenario"]
    ]
    md.append("| Scenario | " + " | ".join(k for k, _ in ranked) + " |")
    md.append("|----------|" + "----|" * len(ranked))
    for sid in sids:
        row = [sid]
        for k, block in ranked:
            matches = [s for s in block["per_scenario"] if s["scenario_id"] == sid]
            if matches:
                row.append("Y" if matches[0]["state_correct_regraded"] else "N")
            else:
                row.append("-")
        md.append("| " + " | ".join(row) + " |")
    md.append("")

    md.append("## Op distribution\n")
    md.append("| Candidate | Op totals |")
    md.append("|-----------|-----------|")
    for k, block in ranked:
        md.append(f"| `{k}` | `{json.dumps(block.get('op_totals', {}))}` |")
    md.append("")

    OUT_REPORT.write_text("\n".join(md))
    print(f"Saved {OUT_RESULTS} and {OUT_REPORT}")


if __name__ == "__main__":
    main()
