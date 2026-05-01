"""Aggregate ALL sweeps (scale, stress_v2, stress_v3, extreme) into one table."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND10 = HERE.parent
RESULTS = ROUND10 / "results"


def load(name: str) -> list[dict]:
    f = RESULTS / name
    if not f.exists():
        return []
    return json.loads(f.read_text())["runs"]


def main() -> None:
    all_runs = []
    for name in [
        "scale_sweep.json",
        "stress_v2_sweep.json",
        "stress_v3_sweep.json",
        "extreme_sweep.json",
    ]:
        all_runs.extend(load(name))

    # Aggregate by scenario, scale
    rows = []
    for r in all_runs:
        if "archs" not in r:
            continue
        scen = r.get("scenario", "extreme")
        scale = r.get("scale", "?")
        if "chain_len" in r:
            scen = f"extreme_c{r['chain_len']}"
        plain = r["archs"].get("aen1_plain", {"passed": 0, "total": 0})
        ind = r["archs"].get("aen1_indexed", {"passed": 0, "total": 0})
        rows.append(
            (scen, scale, plain["passed"], plain["total"], ind["passed"], ind["total"])
        )

    print("\n=== FULL ROUND-10 RESULTS ===\n")
    print(f"{'scenario':<18}{'scale':>6}  {'plain':>10}  {'indexed':>10}  {'delta':>8}")
    print("-" * 70)
    for scen, scale, pp, pt, ip, it in sorted(rows):
        pr = pp / pt if pt else 0
        ir = ip / it if it else 0
        d = ir - pr
        sd = f"{d:+.0%}" if d != 0 else "  ="
        print(f"{scen:<18}{scale:>6}  {pp:>3}/{pt:<6}  {ip:>3}/{it:<6}  {sd:>8}")

    # Per-architecture, per-scale aggregate
    print("\n=== AGGREGATE BY SCALE ===\n")
    by_scale = {}
    for scen, scale, pp, pt, ip, it in rows:
        if scale == "?":
            continue
        s = int(scale)
        if s not in by_scale:
            by_scale[s] = [0, 0, 0, 0]
        by_scale[s][0] += pp
        by_scale[s][1] += pt
        by_scale[s][2] += ip
        by_scale[s][3] += it
    print(f"{'scale':>6}  {'plain':>14}  {'indexed':>14}")
    for s in sorted(by_scale):
        pp, pt, ip, it = by_scale[s]
        pr = pp / pt if pt else 0
        ir = ip / it if it else 0
        print(f"{s:>6}  {pp:>3}/{pt:<3} ({pr:.0%})   {ip:>3}/{it:<3} ({ir:.0%})")


if __name__ == "__main__":
    main()
