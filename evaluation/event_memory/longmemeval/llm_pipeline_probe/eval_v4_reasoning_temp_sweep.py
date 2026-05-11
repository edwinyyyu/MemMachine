"""Reasoning + temperature sweep for the v4 anti-fragment deriver prompt.

Holds PROMPT_DERIVER from probe_deriver_v4_anti_fragment.py FIXED (no edits).
The earlier iter 3a/3b sub-experiments left an unresolved tension:

  - 3b (v3 prompt + reasoning bump): medium reasoning made over-decomposition
    WORSE — model executed the same splitty plan more diligently.
  - 3a (v4 prompt + low reasoning): the new anti-fragment principles fix
    F1/F2/F4 but with severe stochasticity (e.g. F4 ∈ {1, 3, 5, 9, 10}).

This sweep tests: under v4 (anti-fragment) prompt, does medium reasoning
apply the new constraints MORE faithfully (= less variance, fewer regressions)
or does it ALSO over-fragment like v3@medium did?

Empirical finding (probed via API): gpt-5.4-nano on the Responses API
REJECTS temperature with `Unsupported parameter: 'temperature' is not
supported with this model` for ANY value except the implicit default of 1.0.
So the temperature axis is unavailable; the sweep collapses to reasoning-only.

Cells:
  - gpt-5.4-nano @ low (control = current production setting for v4 iter 3a)
  - gpt-5.4-nano @ medium

Cases (11 total, EXACT fixtures from probe_deriver_v4_anti_fragment.py):
  - 4 failure: F1 drum-kit-JSON, F2 chess-25.g4, F3 Mario-prose, F4 D1-mustang
  - 4 clean: C1 Tokyo-Anne, C2 overloaded-trip, C4 markdown-table, C6 code-find-max
  - 1 adversarial: A4 abbreviations
  - 2 stress: D2 multi-topic, D12 numeric-list

Reps: 5 per (cell, case).

Run:
    uv run python eval_v4_reasoning_temp_sweep.py
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
from typing import Any

import openai
from dotenv import load_dotenv

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")

# Re-import the FIXED prompt + schema from the v4 probe.  We do NOT modify them.
from probe_deriver_v4_anti_fragment import (  # noqa: E402
    ADVERSARIAL_CASES,
    CRITICAL_CASES,
    DERIVATIVES_SCHEMA,
    DIVERSE_CASES,
    FAILURE_CASES,
    PROMPT_DERIVER,
)

# --------------------------------------------------------------------------
# CASES (11 total)
# --------------------------------------------------------------------------


def _segment_for_label(
    cases: list[tuple[str, str, str]], prefix: str
) -> tuple[str, str]:
    for label, seg, exp in cases:
        if label.startswith(prefix):
            return seg, exp
    raise KeyError(prefix)


F1_SEG, F1_EXP = _segment_for_label(FAILURE_CASES, "F1")
F2_SEG, F2_EXP = _segment_for_label(FAILURE_CASES, "F2")
F3_SEG, F3_EXP = _segment_for_label(FAILURE_CASES, "F3")
F4_SEG, F4_EXP = _segment_for_label(FAILURE_CASES, "F4")
C1_SEG, C1_EXP = _segment_for_label(CRITICAL_CASES, "C1")
C2_SEG, C2_EXP = _segment_for_label(CRITICAL_CASES, "C2")
C4_SEG, C4_EXP = _segment_for_label(CRITICAL_CASES, "C4")
C6_SEG, C6_EXP = _segment_for_label(CRITICAL_CASES, "C6")
A4_SEG, A4_EXP = _segment_for_label(ADVERSARIAL_CASES, "A4")
D2_SEG, D2_EXP = _segment_for_label(DIVERSE_CASES, "D2")
D12_SEG, D12_EXP = _segment_for_label(DIVERSE_CASES, "D12")


# (label, segment, expected_text, class, expected_n_low, expected_n_high)
# expected_n_low/high define the target count window for PASS verdict.
SWEEP_CASES: list[tuple[str, str, str, str, int, int]] = [
    ("F1-drum-kit-JSON", F1_SEG, F1_EXP, "fail", 1, 2),
    ("F2-chess-25.g4", F2_SEG, F2_EXP, "fail", 1, 1),
    ("F3-mario-prose", F3_SEG, F3_EXP, "fail", 1, 2),
    ("F4-D1-mustang", F4_SEG, F4_EXP, "fail", 1, 3),
    ("C1-tokyo-anne", C1_SEG, C1_EXP, "clean", 1, 1),
    ("C2-overloaded-trip", C2_SEG, C2_EXP, "clean", 2, 4),
    ("C4-markdown-table", C4_SEG, C4_EXP, "clean", 3, 5),  # artifact + 3 rows
    ("C6-code-find-max", C6_SEG, C6_EXP, "clean", 1, 2),
    ("A4-abbreviations", A4_SEG, A4_EXP, "adv", 1, 4),
    ("D2-multi-topic", D2_SEG, D2_EXP, "stress", 4, 6),
    ("D12-numeric-list", D12_SEG, D12_EXP, "stress", 1, 5),  # summary OR per-Q
]

# --------------------------------------------------------------------------
# CELLS
# --------------------------------------------------------------------------

CELLS: list[tuple[str, str]] = [
    ("gpt-5.4-nano", "low"),
    ("gpt-5.4-nano", "medium"),
]
TEMPERATURE_SUPPORTED = False  # confirmed: only temperature=1 (default) accepted

REPS = 5

# --------------------------------------------------------------------------
# CALL
# --------------------------------------------------------------------------


async def derive_timed(
    client: openai.AsyncOpenAI,
    segment: str,
    *,
    model: str,
    reasoning: str,
) -> tuple[list[str], float, str | None]:
    """Run derive once. Returns (derivatives, latency_seconds, error_or_None)."""
    t0 = time.perf_counter()
    try:
        resp = await client.responses.create(
            model=model,
            input=PROMPT_DERIVER.format(segment=segment),
            reasoning={"effort": reasoning},
            text={
                "format": {
                    "type": "json_schema",
                    "name": "derivatives",
                    "schema": DERIVATIVES_SCHEMA,
                    "strict": True,
                }
            },
        )
        elapsed = time.perf_counter() - t0
        payload = json.loads(resp.output_text)
        return list(payload.get("derivatives", [])), elapsed, None
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return [], elapsed, f"{type(e).__name__}: {e}"


# --------------------------------------------------------------------------
# SWEEP
# --------------------------------------------------------------------------


async def run_sweep() -> dict[str, Any]:
    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(8)

    async def one(cell_idx: int, case_idx: int, rep: int) -> dict[str, Any]:
        model, effort = CELLS[cell_idx]
        label, seg, _exp, _cls, _n_low, _n_high = SWEEP_CASES[case_idx]
        async with sem:
            derivs, latency, err = await derive_timed(
                client, seg, model=model, reasoning=effort
            )
        return {
            "model": model,
            "effort": effort,
            "case": label,
            "rep": rep,
            "derivatives": derivs,
            "latency_s": latency,
            "error": err,
        }

    tasks = []
    for ci in range(len(CELLS)):
        for casei in range(len(SWEEP_CASES)):
            for r in range(REPS):
                tasks.append(one(ci, casei, r))

    results = await asyncio.gather(*tasks)
    await client.close()
    return {
        "results": results,
        "cells": CELLS,
        "n_cases": len(SWEEP_CASES),
        "reps": REPS,
        "temperature_supported": TEMPERATURE_SUPPORTED,
    }


# --------------------------------------------------------------------------
# REPORTING
# --------------------------------------------------------------------------


def _short(s: str, n: int = 140) -> str:
    s = s.replace("\n", " \\n ")
    return s if len(s) <= n else s[: n - 1] + "..."


def _verdict(case: tuple, n_runs: list[int]) -> str:
    """Per-run verdict: PASS / PARTIAL / FAIL."""
    _label, _seg, _exp, _cls, lo, hi = case
    if not n_runs:
        return "?"
    in_window = sum(lo <= n <= hi for n in n_runs)
    if in_window == len(n_runs):
        return "PASS"
    if in_window >= max(1, len(n_runs) // 2):
        return "PARTIAL"
    return "FAIL"


def _per_cell_case(results: list[dict[str, Any]]) -> dict:
    out: dict = {}
    for r in results:
        key = (r["model"], r["effort"], r["case"])
        out.setdefault(key, {"n": [], "lat": [], "derivs": [], "err": []})
        out[key]["n"].append(len(r["derivatives"]))
        out[key]["lat"].append(r["latency_s"])
        out[key]["derivs"].append(r["derivatives"])
        out[key]["err"].append(r["error"])
    return out


def _stdev(xs: list[float]) -> float:
    return statistics.stdev(xs) if len(xs) >= 2 else 0.0


def _median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def print_report(payload: dict[str, Any]) -> None:
    results = payload["results"]
    per_cc = _per_cell_case(results)

    case_lookup: dict[str, tuple] = {c[0]: c for c in SWEEP_CASES}

    print()
    print("=" * 110)
    print("# v4 DERIVER REASONING SWEEP — RESULTS")
    print("=" * 110)
    print(f"# cells = {CELLS}")
    print(f"# n_cases = {len(SWEEP_CASES)}, reps = {REPS}")
    print(
        f"# temperature_supported = {payload['temperature_supported']}  "
        f"(only temperature=1.0 (default) accepted by gpt-5.4-nano on Responses API)"
    )

    # ---- 1) Detailed per-cell x per-case grid: 5 reps shown explicitly
    print(
        "\n## Per-cell x per-case detailed (n_derivs across 5 reps | median latency | verdict)"
    )
    print()
    header = f"  {'case':<22s} | {'cls':<6s} | {'expect':<8s} |"
    for m, e in CELLS:
        header += f" {m}/{e:<7s} {'reps':<19s} {'lat':<6s} {'verdict':<9s} |"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for label, seg, exp, cls, lo, hi in SWEEP_CASES:
        case_tuple = case_lookup[label]
        row = f"  {label:<22s} | {cls:<6s} | {f'{lo}-{hi}':<8s} |"
        for m, e in CELLS:
            agg = per_cc.get((m, e, label))
            if not agg:
                row += f" {'?':<43s} |"
                continue
            ns = agg["n"]
            lats = agg["lat"]
            ns_str = ",".join(str(x) for x in ns)
            verdict = _verdict(case_tuple, ns)
            row += (
                f" {f'[{ns_str}]':<28s} {f'{_median(lats):.2f}s':<6s} {verdict:<9s} |"
            )
        print(row)

    # ---- 2) Per-cell aggregate stats
    print("\n## Per-cell aggregate stats")
    print()
    print(
        f"  {'cell':<25s} | {'mean stdev(n)':<14s} | {'med latency':<12s} | "
        f"{'#PASS':<6s} | {'#PARTIAL':<8s} | {'#FAIL':<6s} | {'#regr-vs-low':<12s} | {'#fixes-vs-low':<12s}"
    )
    print("  " + "-" * 116)

    # We compute regressions/fixes by comparing each cell to the "low" cell.
    low_per_case_verdict: dict[str, str] = {}
    for label, seg, exp, cls, lo, hi in SWEEP_CASES:
        agg = per_cc.get(("gpt-5.4-nano", "low", label))
        if agg:
            low_per_case_verdict[label] = _verdict(case_lookup[label], agg["n"])

    for m, e in CELLS:
        all_stdevs = []
        all_lats = []
        n_pass = n_partial = n_fail = n_regr = n_fix = 0
        for label, seg, exp, cls, lo, hi in SWEEP_CASES:
            agg = per_cc.get((m, e, label))
            if not agg:
                continue
            all_stdevs.append(_stdev([float(x) for x in agg["n"]]))
            all_lats.extend(agg["lat"])
            v = _verdict(case_lookup[label], agg["n"])
            if v == "PASS":
                n_pass += 1
            elif v == "PARTIAL":
                n_partial += 1
            elif v == "FAIL":
                n_fail += 1
            # compare to low
            low_v = low_per_case_verdict.get(label)
            if low_v is not None and (m, e) != ("gpt-5.4-nano", "low"):
                # regression: was PASS at low, no longer PASS here
                if low_v == "PASS" and v != "PASS":
                    n_regr += 1
                # fix: not PASS at low, PASS here
                if low_v != "PASS" and v == "PASS":
                    n_fix += 1
        mean_sd = statistics.mean(all_stdevs) if all_stdevs else 0.0
        med_lat = _median(all_lats)
        print(
            f"  {f'{m}/{e}':<25s} | {mean_sd:<14.2f} | {f'{med_lat:.2f}s':<12s} | "
            f"{n_pass:<6d} | {n_partial:<8d} | {n_fail:<6d} | {n_regr:<12d} | {n_fix:<12d}"
        )

    # ---- 3) Latency multiplier vs low
    print("\n## Latency multiplier vs low (median latency across all cases x reps)")
    low_lats: list[float] = []
    for m, e in CELLS:
        if (m, e) == ("gpt-5.4-nano", "low"):
            for label, _, _, _, _, _ in SWEEP_CASES:
                agg = per_cc.get((m, e, label))
                if agg:
                    low_lats.extend(agg["lat"])
    low_med = _median(low_lats) if low_lats else float("nan")
    print(f"  low median latency (baseline) = {low_med:.2f}s")
    for m, e in CELLS:
        if (m, e) == ("gpt-5.4-nano", "low"):
            continue
        cell_lats: list[float] = []
        for label, _, _, _, _, _ in SWEEP_CASES:
            agg = per_cc.get((m, e, label))
            if agg:
                cell_lats.extend(agg["lat"])
        med = _median(cell_lats) if cell_lats else float("nan")
        mult = med / low_med if low_med else float("nan")
        print(f"  {m}/{e}: median {med:.2f}s, mult {mult:.2f}x")

    # ---- 4) Failure-case full dumps
    print("\n## Failure-case full outputs by cell")
    for label, seg, exp, cls, lo, hi in SWEEP_CASES:
        if cls != "fail":
            continue
        print(f"\n### CASE: {label}  (target n in [{lo},{hi}])")
        print(f"    expected: {exp}")
        for m, e in CELLS:
            agg = per_cc.get((m, e, label))
            if not agg:
                continue
            print(
                f"\n    [{m}/{e}]  n_runs={agg['n']}  lat_med={_median(agg['lat']):.2f}s"
            )
            for rep_idx, derivs in enumerate(agg["derivs"]):
                err = agg["err"][rep_idx]
                if err:
                    print(f"      rep{rep_idx}: ERROR {err}")
                    continue
                print(
                    f"      rep{rep_idx}: n={len(derivs)}  lat={agg['lat'][rep_idx]:.2f}s"
                )
                for i, d in enumerate(derivs):
                    print(f"        [{i}] {_short(d, 200)}")

    # ---- 5) Clean / adv / stress quick dump
    print("\n## Clean / adversarial / stress full outputs by cell")
    for label, seg, exp, cls, lo, hi in SWEEP_CASES:
        if cls == "fail":
            continue
        print(f"\n### CASE: {label} ({cls})  (target n in [{lo},{hi}])")
        print(f"    expected: {exp}")
        for m, e in CELLS:
            agg = per_cc.get((m, e, label))
            if not agg:
                continue
            print(
                f"\n    [{m}/{e}]  n_runs={agg['n']}  lat_med={_median(agg['lat']):.2f}s"
            )
            for rep_idx, derivs in enumerate(agg["derivs"]):
                err = agg["err"][rep_idx]
                if err:
                    print(f"      rep{rep_idx}: ERROR {err}")
                    continue
                print(
                    f"      rep{rep_idx}: n={len(derivs)}  lat={agg['lat'][rep_idx]:.2f}s"
                )
                for i, d in enumerate(derivs):
                    print(f"        [{i}] {_short(d, 200)}")


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------


async def main() -> None:
    print("# v4 DERIVER REASONING SWEEP")
    print(f"# cells = {CELLS}")
    print(f"# n_cases = {len(SWEEP_CASES)}  reps = {REPS}")
    print(f"# total_calls = {len(CELLS) * len(SWEEP_CASES) * REPS}")
    print("# (PROMPT_DERIVER held FIXED from probe_deriver_v4_anti_fragment.py)")
    print("# temperature parameter: NOT supported on gpt-5.4-nano Responses API")
    print(
        "#   (only the implicit default temperature=1.0 accepted; any other value 400s)"
    )

    payload = await run_sweep()

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "eval_v4_reasoning_temp_sweep.json",
    )
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n# wrote raw results -> {out_path}")

    print_report(payload)


if __name__ == "__main__":
    asyncio.run(main())
