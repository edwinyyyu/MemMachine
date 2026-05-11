"""Reasoning-effort sweep for the v3 deriver prompt.

Holds PROMPT_DERIVER from probe_deriver_v3_diverse.py FIXED. Sweeps
(model x reasoning_effort) and reports latency / quality across 9 cases:
  - 4 known failures (drum-kit JSON, chess-move, Super Mario prose, D1 long single-topic)
  - 5 known clean (C1, C2, C4, C6, D2)

Each (cell, case) is run twice for variance.

Run:
    uv run python eval_deriver_reasoning_sweep.py
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

# Re-import the FIXED prompt + schema from the v3 probe.  We do NOT modify them.
from probe_deriver_v3_diverse import (  # noqa: E402
    CRITICAL_CASES,
    DERIVATIVES_SCHEMA,
    DIVERSE_CASES,
    PROMPT_DERIVER,
)

# --------------------------------------------------------------------------
# CASES (9 total)
# --------------------------------------------------------------------------

# 4 known failures: drum-kit JSON, chess move, Super Mario prose, D1 long single-topic.
# Drum-kit / chess / Mario come from eval_deriver_real_corpus.json (idx 27, 6, 29).
# D1 is from probe_deriver_v3_diverse.DIVERSE_CASES.

DRUM_KIT_SEGMENT = (
    "Sure, here's an example of a JSON format that includes the manufacturer, "
    "model name, and possible versions of electronic drum kits:\n"
    "```json\n"
    "{\n"
    '  "electronic_drums": [\n'
    "    {\n"
    '      "manufacturer": "Roland",\n'
    '      "model": "TD-50KVX",\n'
    '      "versions": [\n'
    '        "TD-50KVX-S",\n'
    '        "TD-50KVX-WL",\n'
    '        "TD-50KVX-EXP",\n'
    '        "TD-50KVX-ECOM"\n'
    "      ]\n"
    "    },\n"
    "    {\n"
    '      "manufacturer": "Roland",\n'
    '      "model\n'
    "```"
)

CHESS_SEGMENT = (
    "Interesting game so far! My move will be:\n\n"
    "25. g4\n\n"
    "I aim to gain more space on the kingside and restrict your pawn structure. "
    "What is your response?"
)

MARIO_SEGMENT = (
    "As an AI language model, I don't have memories like human beings, but Super "
    "Mario Advance was indeed one of the most popular games for the Game Boy "
    "Advance. It was a remake of the classic Super Mario Bros. 2 game but with "
    "updated graphics and some new features. It also included four playable "
    "characters - Mario, Luigi, Toad, and Princess Peach - each with their "
    "unique abilities. The game was well received by fans and was a great "
    "addition to the Game Boy Advance's library of games."
)


def _segment_for_label(cases: list[tuple[str, str, str]], prefix: str) -> str:
    for label, seg, _ in cases:
        if label.startswith(prefix):
            return seg
    raise KeyError(prefix)


C1_SEGMENT = _segment_for_label(CRITICAL_CASES, "C1")
C2_SEGMENT = _segment_for_label(CRITICAL_CASES, "C2")
C4_SEGMENT = _segment_for_label(CRITICAL_CASES, "C4")
C6_SEGMENT = _segment_for_label(CRITICAL_CASES, "C6")
D1_SEGMENT = _segment_for_label(DIVERSE_CASES, "D1")
D2_SEGMENT = _segment_for_label(DIVERSE_CASES, "D2")


# class: "fail" or "clean"
SWEEP_CASES: list[tuple[str, str, str, str]] = [
    # (label, segment, expected, class)
    (
        "F-drum-kit-JSON",
        DRUM_KIT_SEGMENT,
        "1 artifact-desc derivative + maybe 1 verbatim; SHOULD NOT enumerate one-per-version (4-5 derivs is failure).",
        "fail",
    ),
    (
        "F-chess-25.g4",
        CHESS_SEGMENT,
        "1 derivative restating chess move 25.g4 + intent (kingside space, restrict pawns); 3+ near-paraphrases is failure.",
        "fail",
    ),
    (
        "F-mario",
        MARIO_SEGMENT,
        "1-2 grouped derivatives about Super Mario Advance (GBA remake of SMB2, four characters); 5 sentence-level fragments is failure.",
        "fail",
    ),
    (
        "F-D1-mustang",
        D1_SEGMENT,
        "Few atomic facts (car identity, engine rebuild, paint, transmission, debut); fan-out per paragraph/sentence is failure.",
        "fail",
    ),
    (
        "C-C1-tokyo",
        C1_SEGMENT,
        "1 derivative close to original; preserves Tokyo, Anne, March.",
        "clean",
    ),
    (
        "C-C2-overloaded",
        C2_SEGMENT,
        "~3 derivatives: Tokyo trip with Anne; Park Hyatt $400/5nt; Ichiran ramen.",
        "clean",
    ),
    ("C-C4-md-table", C4_SEGMENT, "Artifact desc + per-row prose (no pipes).", "clean"),
    (
        "C-C6-code",
        C6_SEGMENT,
        "Artifact desc: Python find_max recursive max-of-tree.",
        "clean",
    ),
    (
        "C-D2-multitopic",
        D2_SEGMENT,
        "~5 derivatives: Aisha birth; bathroom reno; dad cardiology; Stripe offer; Diwali dinner.",
        "clean",
    ),
]

# --------------------------------------------------------------------------
# CELLS (model x reasoning)
# --------------------------------------------------------------------------

CELLS: list[tuple[str, str]] = [
    ("gpt-5.4-nano", "low"),
    ("gpt-5.4-nano", "medium"),
    ("gpt-5.4-nano", "high"),
    ("gpt-5-nano", "low"),
    ("gpt-5-nano", "medium"),
]

REPS = 2  # repeat each (cell, case) REPS times for variance.

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
    """Run every (cell, case, rep). Bounded concurrency."""
    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(8)

    async def one(cell_idx: int, case_idx: int, rep: int) -> dict[str, Any]:
        model, effort = CELLS[cell_idx]
        label, seg, _exp, _cls = SWEEP_CASES[case_idx]
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
    return {"results": results}


# --------------------------------------------------------------------------
# REPORTING
# --------------------------------------------------------------------------


def _short(s: str, n: int = 140) -> str:
    s = s.replace("\n", " \\n ")
    return s if len(s) <= n else s[: n - 1] + "..."


def _agg_per_cell(
    results: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for r in results:
        key = (r["model"], r["effort"])
        out.setdefault(key, {"latencies": [], "n_errors": 0, "n_runs": 0})
        out[key]["latencies"].append(r["latency_s"])
        out[key]["n_runs"] += 1
        if r["error"]:
            out[key]["n_errors"] += 1
    return out


def _agg_per_cell_case(
    results: list[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in results:
        key = (r["model"], r["effort"], r["case"])
        out.setdefault(
            key,
            {
                "n_derivatives_runs": [],
                "latencies": [],
                "derivatives_runs": [],
                "errors": [],
            },
        )
        out[key]["n_derivatives_runs"].append(len(r["derivatives"]))
        out[key]["latencies"].append(r["latency_s"])
        out[key]["derivatives_runs"].append(r["derivatives"])
        out[key]["errors"].append(r["error"])
    return out


def _median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def _max(xs: list[float]) -> float:
    return max(xs) if xs else float("nan")


def print_report(payload: dict[str, Any]) -> None:
    results = payload["results"]
    per_cell = _agg_per_cell(results)
    per_cc = _agg_per_cell_case(results)

    case_class: dict[str, str] = {label: cls for label, _seg, _exp, cls in SWEEP_CASES}
    case_expected: dict[str, str] = {
        label: exp for label, _seg, exp, _cls in SWEEP_CASES
    }

    print()
    print("=" * 100)
    print("# DERIVER REASONING SWEEP -- RESULTS")
    print("=" * 100)

    # ---- 1) Tabular per-cell summary
    print(f"\n## Per-cell latency summary (across all 9 cases x {REPS} reps each)")
    print(
        "\n  cell                                 | n_runs | n_err | latency_med (s) | latency_max (s)"
    )
    print("  " + "-" * 95)
    for model, effort in CELLS:
        a = per_cell.get((model, effort), {"latencies": [], "n_errors": 0, "n_runs": 0})
        med = _median(a["latencies"])
        mx = _max(a["latencies"])
        cell_label = f"{model}/{effort}"
        print(
            f"  {cell_label:36s} | {a['n_runs']:6d} | {a['n_errors']:5d} | {med:14.3f} | {mx:14.3f}"
        )

    # ---- 2) Per-cell-per-case grid (n_derivatives, latency)
    print("\n## Per-cell x per-case grid (n_derivatives min..max | latency median sec)")
    case_order = [c[0] for c in SWEEP_CASES]
    header = "  case                  "
    for model, effort in CELLS:
        header += f" | {model[:10]}.{effort[:3]:6s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label in case_order:
        cls = case_class[label]
        row = f"  {('[' + cls + '] ' + label):24s}"
        for model, effort in CELLS:
            agg = per_cc.get((model, effort, label))
            if not agg:
                row += f" | {'?':>15s}"
                continue
            ns = agg["n_derivatives_runs"]
            lats = agg["latencies"]
            n_min, n_max = min(ns), max(ns)
            n_str = f"{n_min}" if n_min == n_max else f"{n_min}-{n_max}"
            row += f" | {n_str:>3s} ({_median(lats):4.1f}s) "
        print(row)

    # ---- 3) Latency on small (C1) and large (D1, drum-kit, Mario) cases
    print(
        "\n## Latency focus: small case (C-C1-tokyo) vs large cases (F-drum-kit-JSON, F-mario, F-D1-mustang)"
    )
    focus = ["C-C1-tokyo", "F-drum-kit-JSON", "F-mario", "F-D1-mustang"]
    print(
        "  cell                                 | "
        + " | ".join(f"{c:18s}" for c in focus)
    )
    print("  " + "-" * 130)
    for model, effort in CELLS:
        cells = []
        for c in focus:
            agg = per_cc.get((model, effort, c))
            if not agg:
                cells.append("?")
                continue
            cells.append(
                f"med {_median(agg['latencies']):4.1f} max {_max(agg['latencies']):4.1f}"
            )
        print(
            f"  {model + '/' + effort:36s} | " + " | ".join(f"{x:18s}" for x in cells)
        )

    # ---- 4) Full derivative dumps for failure cases (low vs medium vs high)
    print("\n## Failure-case outputs by cell (qualitative)")
    failure_cases = [c[0] for c in SWEEP_CASES if c[3] == "fail"]
    for label in failure_cases:
        print(f"\n### CASE: {label}")
        print(f"    Expected behavior: {case_expected[label]}")
        for model, effort in CELLS:
            agg = per_cc.get((model, effort, label))
            if not agg:
                continue
            print(f"\n    [{model}/{effort}]")
            for rep_idx, derivs in enumerate(agg["derivatives_runs"]):
                err = agg["errors"][rep_idx]
                if err:
                    print(f"      rep{rep_idx}: ERROR {err}")
                    continue
                print(
                    f"      rep{rep_idx}: n={len(derivs)} latency={agg['latencies'][rep_idx]:.2f}s"
                )
                for i, d in enumerate(derivs):
                    print(f"        [{i}] {_short(d, 200)}")

    # ---- 5) Full derivative dumps for clean cases (regression check)
    print("\n## Clean-case outputs by cell (regression check)")
    clean_cases = [c[0] for c in SWEEP_CASES if c[3] == "clean"]
    for label in clean_cases:
        print(f"\n### CASE: {label}")
        print(f"    Expected behavior: {case_expected[label]}")
        for model, effort in CELLS:
            agg = per_cc.get((model, effort, label))
            if not agg:
                continue
            print(f"\n    [{model}/{effort}]")
            for rep_idx, derivs in enumerate(agg["derivatives_runs"]):
                err = agg["errors"][rep_idx]
                if err:
                    print(f"      rep{rep_idx}: ERROR {err}")
                    continue
                print(
                    f"      rep{rep_idx}: n={len(derivs)} latency={agg['latencies'][rep_idx]:.2f}s"
                )
                for i, d in enumerate(derivs):
                    print(f"        [{i}] {_short(d, 200)}")

    # ---- 6) Latency cost vs low baseline per model
    print(
        "\n## Latency multiplier vs low (per-model, on FAILURE cases only - large segments)"
    )
    fail_labels = failure_cases
    for model in {m for (m, _e) in CELLS}:
        # gather mean latency across failure cases at each effort for this model
        per_effort: dict[str, list[float]] = {}
        for (m, e, c), agg in per_cc.items():
            if m != model or c not in fail_labels:
                continue
            per_effort.setdefault(e, []).extend(agg["latencies"])
        if "low" not in per_effort:
            continue
        low_mean = (
            statistics.mean(per_effort["low"]) if per_effort["low"] else float("nan")
        )
        line = f"  {model}: low_mean_latency_on_failure={low_mean:.2f}s"
        for e in ("medium", "high"):
            if per_effort.get(e):
                em = statistics.mean(per_effort[e])
                line += f"   |   {e}/low = {em / low_mean:.2f}x ({em:.2f}s)"
        print(line)


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------


async def main() -> None:
    print("# DERIVER REASONING-EFFORT SWEEP")
    print(f"# cells = {CELLS}")
    print(f"# n_cases = {len(SWEEP_CASES)}  reps = {REPS}")
    print(f"# total_calls = {len(CELLS) * len(SWEEP_CASES) * REPS}")
    print("# (PROMPT_DERIVER held FIXED from probe_deriver_v3_diverse.py)")

    payload = await run_sweep()

    # Save raw json next to script
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "eval_deriver_reasoning_sweep.json",
    )
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n# wrote raw results -> {out_path}")

    print_report(payload)


if __name__ == "__main__":
    asyncio.run(main())
