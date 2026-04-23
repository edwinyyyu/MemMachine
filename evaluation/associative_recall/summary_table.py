"""Generate comprehensive summary table of all cue format exploration results."""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
BUDGETS = [20, 50, 100]

# (version_key, description, label_pattern)
CONFIGS = [
    ("v8", "v8 keyword-dense", "{v}_nr1_h1_{bench}_ext_30q"),
    ("v10", "v10 HyDE answer", "{v}_nr1_h1_{bench}_ext_30q"),
    ("v11", "v11 Utterance-style", "{v}_nr1_h1_{bench}_ext_30q"),
    ("v12", "v12 Narrative", "{v}_nr1_h1_{bench}_ext_30q"),
    ("v13", "v13 Freeform", "{v}_nr1_h1_{bench}_ext_30q"),
    ("v14", "v14 Contrastive", "{v}_nr1_h1_{bench}_ext_30q"),
    ("v15", "v15 Self-monitoring", "{v}_nr1_h1_{bench}_ext_30q"),
    ("v16", "v16 Scratchpad", "{v}_nr1_h1_{bench}_ext_30q"),
    ("v17", "v17 Minimal", "{v}_nr1_h1_{bench}_ext_30q"),
    ("v15", "v15 Self-mon 2-hop", "{v}_nr1_h2_{bench}_ext_30q"),
    ("v15", "v15 Self-mon no-nbr", "{v}_nr0_h1_{bench}_ext_30q"),
]


def load(label: str) -> list[dict] | None:
    path = RESULTS_DIR / f"normalized_{label}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def wtl(results: list[dict], budget_label: str) -> tuple[int, int, int]:
    w = t = l = 0
    for r in results:
        b = r["baseline_recalls"].get(budget_label, 0)
        a = r["assoc_recalls"].get(budget_label, 0)
        if a > b + 0.001:
            w += 1
        elif b > a + 0.001:
            l += 1
        else:
            t += 1
    return w, t, l


def main() -> None:
    for benchmark in ["beam", "locomo"]:
        print(f"\n{'='*120}")
        print(f"  {benchmark.upper()} BENCHMARK (30 questions)")
        print(f"{'='*120}")

        # Header
        print(f"\n  {'Config':25s}", end="")
        for b in BUDGETS:
            print(f"  {'delta r@'+str(b):>10s} {'W/T/L':>7s}", end="")
        print(f"  {'AvgSeg':>6s}")
        print("  " + "-" * 100)

        loaded = []
        for version, desc, pattern in CONFIGS:
            label = pattern.format(v=version, bench=benchmark)
            results = load(label)
            if results is None:
                continue

            loaded.append((desc, results))
            avg_segs = sum(r["total_assoc_retrieved"] for r in results) / len(results)
            print(f"  {desc:25s}", end="")
            for b in BUDGETS:
                bl = f"r@{b}"
                b_mean = sum(r["baseline_recalls"].get(bl, 0) for r in results) / len(results)
                a_mean = sum(r["assoc_recalls"].get(bl, 0) for r in results) / len(results)
                delta = a_mean - b_mean
                w, t, l = wtl(results, bl)
                print(f"  {delta:>+10.3f} {w}/{t}/{l}", end="")
            print(f"  {avg_segs:>6.0f}")

        # Combined ranking
        print(f"\n  Ranking by avg delta across r@20+r@50:")
        scores = []
        for desc, results in loaded:
            d20 = sum(r["assoc_recalls"]["r@20"] - r["baseline_recalls"]["r@20"] for r in results) / len(results)
            d50 = sum(r["assoc_recalls"]["r@50"] - r["baseline_recalls"]["r@50"] for r in results) / len(results)
            avg_d = (d20 + d50) / 2
            w20, _, l20 = wtl(results, "r@20")
            scores.append((avg_d, desc, d20, d50, w20, l20))

        scores.sort(reverse=True)
        for rank, (avg_d, desc, d20, d50, w20, l20) in enumerate(scores, 1):
            print(f"    {rank}. {desc:25s} avg={avg_d:+.3f} (r@20={d20:+.3f} r@50={d50:+.3f}) W/L@20={w20}/{l20}")


if __name__ == "__main__":
    main()
