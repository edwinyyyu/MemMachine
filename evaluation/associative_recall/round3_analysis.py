"""Round 3 comprehensive analysis."""

import json
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load(label: str) -> list[dict]:
    with open(RESULTS_DIR / f"normalized_{label}.json") as f:
        return json.load(f)


def wtl(results: list[dict], budget: str) -> tuple[int, int, int]:
    wins = sum(1 for r in results if r["assoc_recalls"][budget] > r["baseline_recalls"][budget] + 0.001)
    losses = sum(1 for r in results if r["baseline_recalls"][budget] > r["assoc_recalls"][budget] + 0.001)
    ties = len(results) - wins - losses
    return wins, ties, losses


def delta(results: list[dict], budget: str) -> float:
    return sum(r["assoc_recalls"][budget] - r["baseline_recalls"][budget] for r in results) / len(results)


def print_section(title: str):
    print(f"\n{'='*80}")
    print(title)
    print(f"{'='*80}")


def main():
    # === SECTION 1: New prompt versions (30q LoCoMo) ===
    print_section("1. NEW CUE STRATEGIES vs v15 (30q LoCoMo)")
    print(f"{'Config':<35s} {'d@20':>7s} {'d@50':>7s} {'d@100':>7s} {'W/T/L@20':>10s}")
    print("-" * 75)

    locomo_configs = [
        ("v15 (self-monitoring) [BEST]", "v15_nr1_h1_locomo_ext_30q"),
        ("v25 (keyword-dense monitor)", "v25_nr1_h1_locomo_ext_30q"),
        ("v22 (cue-as-continuation)", "v22_nr1_h1_locomo_ext_30q"),
        ("v23 (minimal/zero guidance)", "v23_nr1_h1_locomo_ext_30q"),
        ("v21 (question decomposition)", "v21_nr1_h1_locomo_ext_30q"),
        ("v26 (scratchpad)", "v26_nr1_h1_locomo_ext_30q"),
        ("v19 (HyDE hypothetical)", "v19_nr1_h1_locomo_ext_30q"),
        ("v28 (adaptive strategy)", "v28_nr1_h1_locomo_ext_30q"),
        ("v24 (vocab extraction)", "v24_nr1_h1_locomo_ext_30q"),
        ("v20 (perspective-taking)", "v20_nr1_h1_locomo_ext_30q"),
    ]
    for name, label in locomo_configs:
        r = load(label)
        d20 = delta(r, "r@20")
        d50 = delta(r, "r@50")
        d100 = delta(r, "r@100")
        w, t, l = wtl(r, "r@20")
        print(f"{name:<35s} {d20:>+7.1%} {d50:>+7.1%} {d100:>+7.1%} {w:>2d}/{t:>2d}/{l:>2d}")

    # === SECTION 2: New prompt versions (30q BEAM) ===
    print_section("2. NEW CUE STRATEGIES vs v15 (30q BEAM)")
    print(f"{'Config':<35s} {'d@20':>7s} {'d@50':>7s} {'d@100':>7s} {'W/T/L@20':>10s}")
    print("-" * 75)

    beam_configs = [
        ("v15 (self-monitoring)", "v15_nr1_h1_beam_ext_30q"),
        ("v21 (question decomposition)", "v21_nr1_h1_beam_ext_30q"),
        ("v20 (perspective-taking)", "v20_nr1_h1_beam_ext_30q"),
        ("v23 (minimal/zero guidance)", "v23_nr1_h1_beam_ext_30q"),
        ("v25 (keyword-dense monitor)", "v25_nr1_h1_beam_ext_30q"),
        ("v28 (adaptive strategy)", "v28_nr1_h1_beam_ext_30q"),
        ("v26 (scratchpad)", "v26_nr1_h1_beam_ext_30q"),
        ("v19 (HyDE hypothetical)", "v19_nr1_h1_beam_ext_30q"),
        ("v24 (vocab extraction)", "v24_nr1_h1_beam_ext_30q"),
        ("v22 (cue-as-continuation)", "v22_nr1_h1_beam_ext_30q"),
    ]
    for name, label in beam_configs:
        r = load(label)
        d20 = delta(r, "r@20")
        d50 = delta(r, "r@50")
        d100 = delta(r, "r@100")
        w, t, l = wtl(r, "r@20")
        print(f"{name:<35s} {d20:>+7.1%} {d50:>+7.1%} {d100:>+7.1%} {w:>2d}/{t:>2d}/{l:>2d}")

    # === SECTION 3: Configuration variants ===
    print_section("3. v15 CONFIGURATION VARIANTS (30q LoCoMo)")
    print(f"{'Config':<40s} {'d@20':>7s} {'d@50':>7s} {'d@100':>7s} {'AvgSegs':>7s}")
    print("-" * 70)

    variant_configs = [
        ("v15 1h 2cues nr1 (DEFAULT)", "v15_nr1_h1_locomo_ext_30q"),
        ("v15 1h 2cues nr1 +backfill", "v15_nr1_h1_locomo_ext_30q_backfill"),
        ("v15 1h 2cues nr1 +rerank", "v15_nr1_h1_locomo_ext_30q_rerank"),
        ("v15 1h 2cues nr0 tk15", "v15_nr0_h1_locomo_ext_30q_fresh"),
        ("v15 2h 2cues nr1", "v15_nr1_h2_locomo_ext_30q"),
        ("v15 1h 3cues nr1", "v15_nr1_h1_3cues_locomo_ext_30q"),
        ("v15 1h 1cue nr1", "v15_nr1_h1_1cue_locomo_ext_30q"),
    ]
    for name, label in variant_configs:
        r = load(label)
        d20 = delta(r, "r@20")
        d50 = delta(r, "r@50")
        d100 = delta(r, "r@100")
        avg_segs = sum(x["total_assoc_retrieved"] for x in r) / len(r)
        print(f"{name:<40s} {d20:>+7.1%} {d50:>+7.1%} {d100:>+7.1%} {avg_segs:>7.0f}")

    # === SECTION 4: Full-scale results ===
    print_section("4. FULL-SCALE RESULTS (all questions)")
    print(f"{'Config':<45s} {'n':>3s} {'d@20':>7s} {'d@50':>7s} {'d@100':>7s} {'W/T/L@20':>10s}")
    print("-" * 80)

    full_configs = [
        ("v15 LoCoMo (full)", "v15_nr1_h1_locomo_ext_full"),
        ("v15+backfill LoCoMo (full)", "v15_nr1_h1_locomo_ext_full_backfill"),
        ("v15 BEAM (full)", "v15_nr1_h1_beam_ext_full"),
        ("v15+backfill BEAM (full)", "v15_nr1_h1_beam_ext_full_backfill"),
    ]
    for name, label in full_configs:
        r = load(label)
        d20 = delta(r, "r@20")
        d50 = delta(r, "r@50")
        d100 = delta(r, "r@100")
        w, t, l = wtl(r, "r@20")
        print(f"{name:<45s} {len(r):>3d} {d20:>+7.1%} {d50:>+7.1%} {d100:>+7.1%} {w:>2d}/{t:>2d}/{l:>2d}")

    # === SECTION 5: Benefit by conversation length ===
    print_section("5. BENEFIT BY CONVERSATION LENGTH (v15+backfill)")

    locomo_bf = load("v15_nr1_h1_locomo_ext_full_backfill")
    beam_bf = load("v15_nr1_h1_beam_ext_full_backfill")
    all_bf = locomo_bf + beam_bf

    by_len = defaultdict(list)
    for r in all_bf:
        cl = r["conv_length"]
        if cl < 250:
            by_len["short (<250)"].append(r)
        elif cl < 400:
            by_len["medium (250-400)"].append(r)
        else:
            by_len["long (400+)"].append(r)

    print(f"{'Length':<20s} {'n':>3s} {'Base@20':>8s} {'d@20':>7s} {'d@50':>7s} {'d@100':>7s} {'W/T/L@20':>10s}")
    print("-" * 70)
    for bucket in ["short (<250)", "medium (250-400)", "long (400+)"]:
        rows = by_len[bucket]
        b20 = sum(r["baseline_recalls"]["r@20"] for r in rows) / len(rows)
        d20 = delta(rows, "r@20")
        d50 = delta(rows, "r@50")
        d100 = delta(rows, "r@100")
        w, t, l = wtl(rows, "r@20")
        print(f"{bucket:<20s} {len(rows):>3d} {b20:>8.3f} {d20:>+7.1%} {d50:>+7.1%} {d100:>+7.1%} {w:>2d}/{t:>2d}/{l:>2d}")

    # === SECTION 6: Cue length analysis ===
    print_section("6. CUE LENGTH vs PERFORMANCE (v15, 182q LoCoMo)")

    locomo_full = load("v15_nr1_h1_locomo_ext_full")
    for r in locomo_full:
        cue_lens = []
        for hop in r["hop_details"]:
            if hop["hop"] > 0:
                for c in hop["cues"]:
                    cue_lens.append(len(c))
        r["avg_cue_len"] = sum(cue_lens) / len(cue_lens) if cue_lens else 0
        r["delta_r20"] = r["assoc_recalls"]["r@20"] - r["baseline_recalls"]["r@20"]

    sorted_by_len = sorted(locomo_full, key=lambda r: r["avg_cue_len"])
    n = len(sorted_by_len)
    quartiles = [
        ("Q1 (shortest)", sorted_by_len[: n // 4]),
        ("Q2", sorted_by_len[n // 4 : n // 2]),
        ("Q3", sorted_by_len[n // 2 : 3 * n // 4]),
        ("Q4 (longest)", sorted_by_len[3 * n // 4 :]),
    ]

    print(f"{'Quartile':<20s} {'AvgLen':>7s} {'d@20':>7s} {'W/T/L@20':>10s}")
    print("-" * 50)
    for label, group in quartiles:
        avg_len = sum(r["avg_cue_len"] for r in group) / len(group)
        avg_d20 = sum(r["delta_r20"] for r in group) / len(group)
        w = sum(1 for r in group if r["delta_r20"] > 0.001)
        l = sum(1 for r in group if r["delta_r20"] < -0.001)
        t = len(group) - w - l
        print(f"{label:<20s} {avg_len:>7.0f} {avg_d20:>+7.1%} {w:>2d}/{t:>2d}/{l:>2d}")


if __name__ == "__main__":
    main()
