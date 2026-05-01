"""Compare per-scenario results across three runs:
  - explicit-thinking baseline (no EM-ingest)
  - native-reasoning baseline (no EM-ingest)
  - explicit-thinking + EM_INGEST_THINKING=1
  - (optional) native-reasoning + EM_INGEST_THINKING=1

Usage:
    uv run python evaluation/associative_recall/compare_runs.py

The script auto-discovers result files by scenario_id from the per-scenario
result-file naming pattern. Pass --include glob/list to override.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results"

HARD = [
    "multi-hop-banquet-01",
    "vocab-bridge-trip-01",
    "narrative-meal-01",
    "adversarial-pricing-01",
    "stacked-event-planning-01",
    "supersession-vendor-decision-01",
    "negative-space-onboarding-01",
    "world-knowledge-bridge-01",
    "deductive-chain-procurement-01",
    "negation-and-inference-01",
]

# Manual mapping of timestamps → run-label for the explicit-thinking baseline
# and the native-reasoning baseline. Discovered by chronology.
EXPLICIT_BASELINE_TS = {
    "vocab-bridge-trip-01": 1777400776,
    "multi-hop-banquet-01": 1777401050,
    "narrative-meal-01": 1777401335,
    "adversarial-pricing-01": 1777401582,
    "stacked-event-planning-01": 1777401948,
    "supersession-vendor-decision-01": 1777402228,
    "negative-space-onboarding-01": 1777402482,
    "world-knowledge-bridge-01": 1777402684,
    "deductive-chain-procurement-01": 1777402907,
    "negation-and-inference-01": 1777403138,
}
NATIVE_TS = {
    "vocab-bridge-trip-01": 1777403411,
    "multi-hop-banquet-01": 1777403667,
    "narrative-meal-01": 1777403968,
    "adversarial-pricing-01": 1777404208,
    # stacked-event-planning-01 silent failure
    "supersession-vendor-decision-01": 1777404557,
    "negative-space-onboarding-01": 1777404763,
    "world-knowledge-bridge-01": 1777405025,
    "deductive-chain-procurement-01": 1777405308,
    "negation-and-inference-01": 1777405551,
}


def load_one(ts: int) -> dict:
    p = RESULTS / f"mid_execution_eval_e2_{ts}.json"
    return json.loads(p.read_text())


def per_scenario_metrics(d: dict, mode: str = "spreading_activation_full") -> dict:
    out: dict[str, dict] = {}
    for s in d["scenarios"]:
        sid = s["scenario_id"]
        if mode not in s["per_mode"]:
            continue
        agg = s["per_mode"][mode]["aggregates"]
        out[sid] = {
            "cov": agg.get("coverage_rate"),
            "full5": agg.get("triggered_recall_full@5"),
            "cond5": agg.get("recall_given_covered@5"),
        }
    return out


def discover_em_ingest_by_run(run_id: int) -> dict[str, dict]:
    """Find the per-scenario hard_<run_id>_<sid>.json files for a specific run."""
    out: dict[str, dict] = {}
    for sid in HARD:
        p = RESULTS / f"hard_{run_id}_{sid}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for s in d["scenarios"]:
            if s["scenario_id"] == sid:
                agg = (
                    s["per_mode"]
                    .get("spreading_activation_full", {})
                    .get("aggregates", {})
                )
                out[sid] = {
                    "cov": agg.get("coverage_rate"),
                    "full5": agg.get("triggered_recall_full@5"),
                    "cond5": agg.get("recall_given_covered@5"),
                    "_path": str(p),
                }
                break
    return out


def discover_em_ingest(after_ts: int = 1777405551) -> dict[str, dict]:
    """Find newer per-scenario result files (after the native run) and treat
    the most recent one per scenario as the EM-ingest result.
    """
    out: dict[str, dict] = {}
    for sid in HARD:
        # Scan for hard_RUNID_<sid>.json (preferred — created by run_hard_set.py)
        cands = sorted(
            RESULTS.glob(f"hard_*_{sid}.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if cands:
            d = json.loads(cands[0].read_text())
            for s in d["scenarios"]:
                if s["scenario_id"] == sid:
                    agg = (
                        s["per_mode"]
                        .get("spreading_activation_full", {})
                        .get("aggregates", {})
                    )
                    out[sid] = {
                        "cov": agg.get("coverage_rate"),
                        "full5": agg.get("triggered_recall_full@5"),
                        "cond5": agg.get("recall_given_covered@5"),
                        "_path": str(cands[0]),
                    }
                    break
            continue
        # Fallback: most-recent any-run file for this sid newer than `after_ts`
        cands = sorted(
            RESULTS.glob("mid_execution_eval_e2_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for p in cands:
            try:
                ts = int(p.stem.split("_")[-1])
            except ValueError:
                continue
            if ts <= after_ts:
                break
            d = json.loads(p.read_text())
            sids = [s["scenario_id"] for s in d["scenarios"]]
            if sid in sids:
                for s in d["scenarios"]:
                    if s["scenario_id"] == sid:
                        agg = (
                            s["per_mode"]
                            .get("spreading_activation_full", {})
                            .get("aggregates", {})
                        )
                        out[sid] = {
                            "cov": agg.get("coverage_rate"),
                            "full5": agg.get("triggered_recall_full@5"),
                            "cond5": agg.get("recall_given_covered@5"),
                            "_path": str(p),
                        }
                        break
                break
    return out


EM_INGEST_RUN_ID = 1777405618  # explicit-thinking + EM-ingest, no filter
EM_INGEST_FILTER_RUN_ID = 1777408191  # explicit-thinking + EM-ingest + retrieval filter
EM_COGNITION_CHANNEL_RUN_ID = (
    1777409739  # explicit-thinking + cognition channel (separate EM partition)
)


def main() -> None:
    explicit = {
        sid: per_scenario_metrics(load_one(ts)).get(sid)
        for sid, ts in EXPLICIT_BASELINE_TS.items()
    }
    native = {
        sid: per_scenario_metrics(load_one(ts)).get(sid)
        for sid, ts in NATIVE_TS.items()
    }
    em_ingest = discover_em_ingest_by_run(EM_INGEST_RUN_ID)
    em_filter = discover_em_ingest_by_run(EM_INGEST_FILTER_RUN_ID)
    em_cog = discover_em_ingest_by_run(EM_COGNITION_CHANNEL_RUN_ID)

    cols = [
        ("explicit", explicit),
        ("native", native),
        ("em_ingest", em_ingest),
        ("em_ing+flt", em_filter),
        ("cog_chan", em_cog),
    ]
    width = 14
    sep = "-" * (35 + len(cols) * (width + 3))
    head1 = f"{'scenario':<35s}" + "".join(f" | {label:<{width}s}" for label, _ in cols)
    head2 = f"{'':<35s}" + "".join(f" | {'cov / full@5':<{width}s}" for _ in cols)
    print(sep)
    print(head1)
    print(head2)
    print(sep)
    sums = {label: [0, 0, 0] for label, _ in cols}
    for sid in HARD:
        cells: list[str] = []
        for label, table in cols:
            x = table.get(sid)
            if x is None:
                cells.append("—")
            else:
                cells.append(f"{x['cov']:.3f} / {x['full5']:.3f}")
                if x.get("cov") is not None:
                    sums[label][0] += x["cov"]
                    sums[label][1] += x["full5"]
                    sums[label][2] += 1
        line = f"{sid:<35s}" + "".join(f" | {c:<{width}s}" for c in cells)
        print(line)
    print(sep)
    for label, _ in cols:
        c, f, n = sums[label]
        if n == 0:
            print(f"  {label:<11s} mean: (n=0)")
        else:
            print(f"  {label:<11s} mean (n={n}): cov={c / n:.3f}  full@5={f / n:.3f}")


if __name__ == "__main__":
    main()
