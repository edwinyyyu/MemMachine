"""Run the 10 hard scenarios (indices 10-19) of mid_execution_scenarios.json
through mid_execution_eval_e2.py serially. Inherits env vars (EM_INGEST_THINKING,
REASONING_MODE, EXECUTOR_BACKEND, etc.) from caller.

Usage:
    EM_INGEST_THINKING=1 uv run python evaluation/associative_recall/run_hard_set.py

Each scenario runs as its own subprocess invocation of mid_execution_eval_e2.py
--scenario <id>; results print to stdout. After all complete, prints a means
table identical in shape to the in-process summary.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).parent
SCENARIOS = json.loads((HERE / "data" / "mid_execution_scenarios.json").read_text())
HARD_IDS = [s["scenario_id"] for s in SCENARIOS[10:20]]


def main() -> None:
    modes = os.environ.get("MODES", "spreading_activation_full")
    K = os.environ.get("K", "1,3,5,10")
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    run_id = int(time.time())
    per_results: list[dict] = []

    for sid in HARD_IDS:
        print(f"[hard] {sid}", flush=True)
        out_path = out_dir / f"hard_{run_id}_{sid}.json"
        cmd = [
            "uv",
            "run",
            "python",
            str(HERE / "mid_execution_eval_e2.py"),
            "--scenario",
            sid,
            "--modes",
            modes,
            "--K",
            K,
            "--out",
            str(out_path),
        ]
        env = os.environ.copy()
        proc = subprocess.run(cmd, env=env)
        if proc.returncode != 0:
            print(f"[hard] {sid} FAILED rc={proc.returncode}", flush=True)
            continue
        try:
            d = json.loads(out_path.read_text())
            per_results.extend(d.get("scenarios", []))
        except Exception as e:
            print(f"[hard] {sid} parse error: {e}", flush=True)

    # Aggregate
    print("\n=== Hard-set means ===", flush=True)
    if not per_results:
        print("  (no results)", flush=True)
        return

    K_list = [int(x) for x in K.split(",") if x.strip()]
    mode_list = [m.strip() for m in modes.split(",") if m.strip()]
    print(f"  n={len(per_results)} scenarios", flush=True)
    for m in mode_list:
        cov_vals = []
        full_at = {k: [] for k in K_list}
        cond_at = {k: [] for k in K_list}
        for r in per_results:
            agg = r["per_mode"][m]["aggregates"]
            cov_vals.append(agg["coverage_rate"])
            for k in K_list:
                v = agg.get(f"triggered_recall_full@{k}")
                if v is not None:
                    full_at[k].append(v)
                v = agg.get(f"recall_given_covered@{k}")
                if v is not None:
                    cond_at[k].append(v)
        print(f"\n  {m}:", flush=True)
        print(f"    coverage_rate: {sum(cov_vals) / len(cov_vals):.3f}", flush=True)
        for k in K_list:
            if full_at[k]:
                print(
                    f"    full_R@{k}: {sum(full_at[k]) / len(full_at[k]):.3f}",
                    flush=True,
                )
        for k in K_list:
            if cond_at[k]:
                print(
                    f"    cond_R@{k}: {sum(cond_at[k]) / len(cond_at[k]):.3f}",
                    flush=True,
                )

    summary_path = out_dir / f"hard_{run_id}_SUMMARY.json"
    summary_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "scenarios": HARD_IDS,
                "modes": mode_list,
                "K_list": K_list,
                "per_results": per_results,
            },
            indent=2,
        )
    )
    print(f"\nWrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
