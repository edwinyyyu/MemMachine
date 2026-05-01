"""Run cue format exploration evaluations sequentially.

Tests multiple prompt versions (v8 baseline, v10-v16) on both BEAM and LoCoMo
benchmarks, with normalized evaluation.
"""

import subprocess
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Configurations to test
CONFIGS = [
    # Baseline re-run on full dataset
    {"version": "v8", "label_suffix": "full", "desc": "v8 keyword-dense (baseline)"},
    # New cue format explorations
    {
        "version": "v10",
        "label_suffix": "",
        "desc": "v10 HyDE-style hypothetical answer",
    },
    {"version": "v11", "label_suffix": "", "desc": "v11 Utterance-style chat messages"},
    {"version": "v12", "label_suffix": "", "desc": "v12 Narrative-style paragraphs"},
    {
        "version": "v13",
        "label_suffix": "",
        "desc": "v13 Freeform/Mixed (no format constraint)",
    },
    {
        "version": "v14",
        "label_suffix": "",
        "desc": "v14 Contrastive (target what's missing)",
    },
    {"version": "v15", "label_suffix": "", "desc": "v15 Self-monitoring + cues"},
    {"version": "v16", "label_suffix": "", "desc": "v16 Scratchpad + cues"},
]

BENCHMARKS = ["beam", "locomo"]


def run_evaluation(version: str, benchmark: str, label: str) -> bool:
    """Run a single evaluation and return True if successful."""
    result_file = RESULTS_DIR / f"normalized_{label}.json"
    if result_file.exists():
        print(f"  SKIP (exists): {result_file.name}")
        return True

    cmd = [
        "uv",
        "run",
        "python",
        "evaluate_normalized.py",
        "--prompt-version",
        version,
        "--max-hops",
        "1",
        "--neighbor-radius",
        "1",
        "--data-suffix",
        "_extended",
        "--benchmark-filter",
        benchmark,
        "--label",
        label,
    ]
    print(f"  RUN: {' '.join(cmd[-8:])}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  FAILED ({elapsed:.0f}s): {result.stderr[-500:]}")
        return False
    # Extract summary line
    lines = result.stdout.strip().split("\n")
    for line in lines:
        if "DELTAS" in line or "delta" in line.lower():
            print(f"  {line.strip()}")
    print(f"  OK ({elapsed:.0f}s)")
    return True


def main() -> None:
    print(f"{'=' * 80}")
    print("CUE FORMAT EXPLORATION - Sequential Evaluation")
    print(f"{'=' * 80}")

    for config in CONFIGS:
        version = config["version"]
        suffix = f"_{config['label_suffix']}" if config["label_suffix"] else ""
        desc = config["desc"]

        print(f"\n--- {desc} ---")
        for benchmark in BENCHMARKS:
            label = f"{version}_nr1_h1_{benchmark}_ext{suffix}"
            print(f"\n  [{benchmark.upper()}] {version} -> {label}")
            run_evaluation(version, benchmark, label)

    print(f"\n{'=' * 80}")
    print("All evaluations complete. Run analyze_normalized.py to compare.")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
