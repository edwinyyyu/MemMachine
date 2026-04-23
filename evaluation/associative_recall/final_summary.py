"""Generate final comprehensive summary of cue format exploration."""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load(label):
    path = RESULTS_DIR / f"normalized_{label}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def compute(results, bl="r@20"):
    b_vals = [r["baseline_recalls"][bl] for r in results]
    a_vals = [r["assoc_recalls"][bl] for r in results]
    delta = sum(a - b for a, b in zip(a_vals, b_vals)) / len(results)
    w = sum(1 for a, b in zip(a_vals, b_vals) if a > b + 0.001)
    l = sum(1 for a, b in zip(a_vals, b_vals) if b > a + 0.001)
    return delta, w, l


configs = [
    ("v8 keyword-dense", "v8_nr1_h1_{bench}_ext_30q"),
    ("v10 HyDE answer", "v10_nr1_h1_{bench}_ext_30q"),
    ("v11 Utterance-style", "v11_nr1_h1_{bench}_ext_30q"),
    ("v12 Narrative", "v12_nr1_h1_{bench}_ext_30q"),
    ("v13 Freeform", "v13_nr1_h1_{bench}_ext_30q"),
    ("v14 Contrastive", "v14_nr1_h1_{bench}_ext_30q"),
    ("v15 Self-monitoring", "v15_nr1_h1_{bench}_ext_30q"),
    ("v16 Scratchpad", "v16_nr1_h1_{bench}_ext_30q"),
    ("v17 Minimal", "v17_nr1_h1_{bench}_ext_30q"),
]

print("COMPREHENSIVE RESULTS TABLE (30 questions per benchmark)")
print("=" * 120)
header = (
    f"{'Config':25s} | "
    f"{'BEAM r@20':>9s} {'W/L':>5s} "
    f"{'r@50':>7s} "
    f"{'r@100':>7s} | "
    f"{'LoCo r@20':>9s} {'W/L':>5s} "
    f"{'r@50':>7s} "
    f"{'r@100':>7s} |"
)
print(header)
print("-" * 120)

for name, pattern in configs:
    beam = load(pattern.format(bench="beam"))
    locomo = load(pattern.format(bench="locomo"))

    print(f"{name:25s} |", end="")

    if beam:
        d20, w20, l20 = compute(beam, "r@20")
        d50, _, _ = compute(beam, "r@50")
        d100, _, _ = compute(beam, "r@100")
        print(f" {d20:>+8.1%} {w20:>2d}/{l20:<2d}", end="")
        print(f" {d50:>+6.1%}", end="")
        print(f" {d100:>+6.1%} |", end="")
    else:
        print(f" {'N/A':>9s} {'':>5s} {'':>7s} {'':>7s} |", end="")

    if locomo:
        d20, w20, l20 = compute(locomo, "r@20")
        d50, _, _ = compute(locomo, "r@50")
        d100, _, _ = compute(locomo, "r@100")
        print(f" {d20:>+8.1%} {w20:>2d}/{l20:<2d}", end="")
        print(f" {d50:>+6.1%}", end="")
        print(f" {d100:>+6.1%} |", end="")
    else:
        print(f" {'N/A':>9s} {'':>5s} {'':>7s} {'':>7s} |", end="")

    print()

print("-" * 120)

# Robustness checks
print("\nROBUSTNESS CHECKS (60 questions)")
print("-" * 80)
robustness = [
    ("v15 Self-mon", "v15_nr1_h1_{bench}_ext_60q"),
    ("v16 Scratchpad", "v16_nr1_h1_{bench}_ext_60q"),
    ("v13 Freeform", "v13_nr1_h1_{bench}_ext_60q"),
]
for name, pattern in robustness:
    beam = load(pattern.format(bench="beam"))
    locomo = load(pattern.format(bench="locomo"))
    print(f"  {name:20s}", end="")
    if beam:
        d20, w20, l20 = compute(beam, "r@20")
        d50, _, _ = compute(beam, "r@50")
        print(f"  BEAM: {d20:>+6.1%} W/L={w20}/{l20}", end="")
    if locomo:
        d20, w20, l20 = compute(locomo, "r@20")
        d50, _, _ = compute(locomo, "r@50")
        print(f"  LoCo: {d20:>+6.1%} W/L={w20}/{l20}", end="")
    print()


if __name__ == "__main__":
    pass
