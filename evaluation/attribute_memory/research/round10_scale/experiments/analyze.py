"""Analyze sweep output: build a degradation curve table and per-kind breakdown."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND10 = HERE.parent
RESULTS = ROUND10 / "results"


def main() -> None:
    sweep = json.loads((RESULTS / "scale_sweep.json").read_text())
    runs = sweep["runs"]

    # Per (scenario, scale, arch) -> (passed, total)
    summary = []
    # also per kind
    by_kind: dict[tuple[str, int, str], dict[str, list[bool]]] = defaultdict(
        lambda: defaultdict(list)
    )

    # Need to re-derive question kinds from the generators. Easiest: load them.
    import sys

    sys.path.insert(0, str(ROUND10 / "scenarios"))
    from generators import GENERATORS

    qid_to_kind: dict[str, str] = {}
    for name, gen in GENERATORS.items():
        if name == "deep_chain":
            _, qs = gen(200, chain_len=10)
        else:
            _, qs = gen(200)
        for q in qs:
            qid_to_kind[q.qid] = q.kind

    for run in runs:
        if "archs" not in run:
            continue
        scen = run["scenario"]
        scale = run["scale"]
        for arch_name, arch in run["archs"].items():
            passed = arch["passed"]
            total = arch["total"]
            summary.append((scen, scale, arch_name, passed, total))
            for v in arch["verdicts"]:
                kind = qid_to_kind.get(v["qid"], "unk")
                by_kind[(scen, scale, arch_name)][kind].append(v["passed"])

    # Print main table
    print("\n=== MAIN TABLE: arch x (scenario, scale) ===\n")
    print(f"{'scenario':<14}{'scale':>6}  {'plain':>8}  {'indexed':>10}")
    by_pair: dict[tuple[str, int], dict[str, tuple[int, int]]] = defaultdict(dict)
    for scen, scale, arch, p, t in summary:
        by_pair[(scen, scale)][arch] = (p, t)
    for (scen, scale), archs in sorted(by_pair.items()):
        plain = archs.get("aen1_plain", (0, 0))
        ind = archs.get("aen1_indexed", (0, 0))
        print(f"{scen:<14}{scale:>6}  {plain[0]}/{plain[1]:<6}  {ind[0]}/{ind[1]:<6}")

    print("\n=== ACCURACY (rate) ===\n")
    print(f"{'scenario':<14}{'scale':>6}  {'plain':>8}  {'indexed':>10}  {'delta':>8}")
    for (scen, scale), archs in sorted(by_pair.items()):
        plain = archs.get("aen1_plain", (0, 1))
        ind = archs.get("aen1_indexed", (0, 1))
        plain_rate = plain[0] / plain[1] if plain[1] else 0
        ind_rate = ind[0] / ind[1] if ind[1] else 0
        delta = ind_rate - plain_rate
        print(
            f"{scen:<14}{scale:>6}  {plain_rate:>8.2%}  {ind_rate:>10.2%}  {delta:>+8.2%}"
        )

    print("\n=== PER-KIND ACCURACY at scale=1000 ===\n")
    print(f"{'scenario':<14}{'kind':<12}{'plain':>10}  {'indexed':>12}")
    for (scen, scale, arch), kinds in sorted(by_kind.items()):
        if scale != 1000:
            continue
        for kind, results in sorted(kinds.items()):
            r = sum(results) / len(results) if results else 0
            print(f"{scen:<14}{kind:<12}{arch:<14} {r:.2%}")

    print(
        f"\nTotal cost: ${sweep['final_cost']:.3f}, "
        f"LLM={sweep['llm_calls']}, embed={sweep['embed_calls']}"
    )


if __name__ == "__main__":
    main()
