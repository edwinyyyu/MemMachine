"""Round 3 evaluation: systematically test new cue strategies.

Tests v19-v24 on both BEAM and LoCoMo at 30q, then scales up winners to 60q.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
WORKING_DIR = Path(__file__).resolve().parent

# Phase 1: 30q screening on both benchmarks
PHASE1_CONFIGS = [
    {"version": "v19", "desc": "HyDE-style full hypothetical answer"},
    {"version": "v20", "desc": "Perspective-taking (recall from inside)"},
    {"version": "v21", "desc": "Question decomposition"},
    {"version": "v22", "desc": "Cue-as-continuation"},
    {"version": "v23", "desc": "Minimal (zero guidance)"},
    {"version": "v24", "desc": "Self-monitoring + vocab extraction"},
]

BENCHMARKS = ["beam", "locomo"]


def run_evaluation(version: str, benchmark: str, label: str,
                   max_questions: int | None = None,
                   extra_args: list[str] | None = None) -> bool:
    """Run a single evaluation and return True if successful."""
    result_file = RESULTS_DIR / f"normalized_{label}.json"
    if result_file.exists():
        print(f"  SKIP (exists): {result_file.name}")
        return True

    cmd = [
        "uv", "run", "python", "evaluate_normalized.py",
        "--prompt-version", version,
        "--max-hops", "1",
        "--neighbor-radius", "1",
        "--data-suffix", "_extended",
        "--benchmark-filter", benchmark,
        "--label", label,
    ]
    if max_questions:
        cmd.extend(["--max-questions", str(max_questions)])
    if extra_args:
        cmd.extend(extra_args)

    print(f"  RUN: {label}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=WORKING_DIR)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  FAILED ({elapsed:.0f}s): {result.stderr[-500:]}")
        return False

    # Extract delta line
    lines = result.stdout.strip().split("\n")
    for line in lines:
        if "DELTAS" in line:
            idx = lines.index(line)
            if idx + 1 < len(lines):
                print(f"  DELTAS: {lines[idx+1].strip()}")
    print(f"  OK ({elapsed:.0f}s)")
    return True


def analyze_results(labels: list[str]) -> None:
    """Print comparison table."""
    cmd = ["uv", "run", "python", "analyze_normalized.py"] + labels
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=WORKING_DIR)
    # Print just the comparison table
    lines = result.stdout.strip().split("\n")
    in_comparison = False
    for line in lines:
        if "COMPARISON" in line:
            in_comparison = True
        if in_comparison:
            print(line)


def main() -> None:
    phase = sys.argv[1] if len(sys.argv) > 1 else "1"

    if phase == "1":
        print("=" * 80)
        print("ROUND 3 — Phase 1: 30q screening of v19-v24")
        print("=" * 80)

        for config in PHASE1_CONFIGS:
            version = config["version"]
            desc = config["desc"]
            print(f"\n--- {version}: {desc} ---")
            for benchmark in BENCHMARKS:
                label = f"{version}_nr1_h1_{benchmark}_ext_30q"
                print(f"\n  [{benchmark.upper()}]")
                run_evaluation(version, benchmark, label, max_questions=30)

        # Compare all
        print("\n\n" + "=" * 80)
        print("COMPARISON")
        print("=" * 80)
        labels = ["v15_nr1_h1_locomo_ext_30q", "v15_nr1_h1_beam_ext_30q"]
        for config in PHASE1_CONFIGS:
            for bm in BENCHMARKS:
                labels.append(f"{config['version']}_nr1_h1_{bm}_ext_30q")
        analyze_results(labels)

    elif phase == "2":
        print("=" * 80)
        print("ROUND 3 — Phase 2: 60q scale-up of winners")
        print("=" * 80)
        # Determine which versions to scale up based on Phase 1 results
        # This will be filled in after Phase 1 analysis
        versions = sys.argv[2:]
        if not versions:
            print("Usage: run_round3.py 2 v19 v20 ...")
            return
        for version in versions:
            for benchmark in BENCHMARKS:
                label = f"{version}_nr1_h1_{benchmark}_ext_60q"
                print(f"\n  [{benchmark.upper()}] {version} -> {label}")
                run_evaluation(version, benchmark, label, max_questions=60)
        labels = []
        for version in versions:
            for bm in BENCHMARKS:
                labels.append(f"{version}_nr1_h1_{bm}_ext_60q")
        labels += ["v15_nr1_h1_locomo_ext_60q", "v15_nr1_h1_beam_ext_60q"]
        analyze_results(labels)

    elif phase == "3":
        print("=" * 80)
        print("ROUND 3 — Phase 3: Rerank/backfill variants of winners")
        print("=" * 80)
        versions = sys.argv[2:]
        if not versions:
            print("Usage: run_round3.py 3 v19 v20 ...")
            return
        for version in versions:
            for benchmark in BENCHMARKS:
                # Rerank variant
                label = f"{version}_nr1_h1_{benchmark}_ext_30q_rerank"
                print(f"\n  [{benchmark.upper()}] {version} rerank -> {label}")
                run_evaluation(version, benchmark, label,
                               max_questions=30, extra_args=["--rerank"])
                # Backfill variant
                label = f"{version}_nr1_h1_{benchmark}_ext_30q_backfill"
                print(f"\n  [{benchmark.upper()}] {version} backfill -> {label}")
                run_evaluation(version, benchmark, label,
                               max_questions=30, extra_args=["--backfill"])

    elif phase == "scratchpad":
        print("=" * 80)
        print("ROUND 3 — Scratchpad experiment")
        print("=" * 80)
        # v25 will be the scratchpad variant: see associative_recall.py
        for benchmark in BENCHMARKS:
            label = f"v25_nr1_h1_{benchmark}_ext_30q"
            print(f"\n  [{benchmark.upper()}]")
            run_evaluation("v25", benchmark, label, max_questions=30)


if __name__ == "__main__":
    main()
