"""Analyze eval_recall_at_k.json: print per-question table, aggregates, wins, losses.

Run:
    uv run python analyze_recall_at_k.py
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PATH = Path(__file__).parent / "eval_recall_at_k.json"


def short(s: str, n: int = 160) -> str:
    s = (s or "").replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def main() -> None:
    if not PATH.exists():
        print(f"missing {PATH}")
        sys.exit(1)
    data = json.load(PATH.open())
    results = data.get("results", [])
    print(f"loaded {len(results)} question results")

    # Per-question table.
    print("\n=== PER-QUESTION TABLE ===")
    print(
        f"{'qid':22s} {'qtype':40s} {'segs':5s} {'derivs':6s} "
        f"{'A@1':>3s} {'B@1':>3s} {'A@5':>3s} {'B@5':>3s} {'A@10':>4s} {'B@10':>4s} {'verdict':>8s}"
    )
    win, tie, loss = 0, 0, 0
    win_by_k = {1: [0, 0, 0], 5: [0, 0, 0], 10: [0, 0, 0]}
    for r in results:
        if "rA_at" not in r:
            continue
        rA, rB = r["rA_at"], r["rB_at"]
        delta_at_5 = rB[5] - rA[5]
        verdict = "WIN" if delta_at_5 > 0 else ("LOSS" if delta_at_5 < 0 else "TIE")
        if delta_at_5 > 0:
            win += 1
        elif delta_at_5 < 0:
            loss += 1
        else:
            tie += 1
        for k in (1, 5, 10):
            d = rB[k] - rA[k]
            if d > 0:
                win_by_k[k][0] += 1
            elif d < 0:
                win_by_k[k][2] += 1
            else:
                win_by_k[k][1] += 1
        print(
            f"{r['question_id']:22s} {r['question_type'][13:]:40s} "
            f"{r['n_segments']:5d} {r['n_derivatives']:6d} "
            f"{rA[1]:>3d} {rB[1]:>3d} {rA[5]:>3d} {rB[5]:>3d} {rA[10]:>4d} {rB[10]:>4d} "
            f"{verdict:>8s}"
        )
    print(f"\nat R@5: wins={win} ties={tie} losses={loss}")
    for k in (1, 5, 10):
        w, t, losses = win_by_k[k]
        print(f"at R@{k}: wins={w} ties={t} losses={losses}")

    # Overall aggregates.
    print("\n=== AGGREGATES ===")
    overall = defaultdict(list)
    for r in results:
        if "rA_at" not in r:
            continue
        for k in (1, 5, 10):
            overall[f"A@{k}"].append(r["rA_at"][k])
            overall[f"B@{k}"].append(r["rB_at"][k])
    n = len(overall["A@1"])
    print(f"n={n} (excludes abstention/no-gold questions)")
    for k in (1, 5, 10):
        a = statistics.mean(overall[f"A@{k}"])
        b = statistics.mean(overall[f"B@{k}"])
        delta = b - a
        # 95% CI on the mean difference (paired): t * stderr
        diffs = [
            br - ar
            for ar, br in zip(overall[f"A@{k}"], overall[f"B@{k}"], strict=False)
        ]
        if len(diffs) > 1:
            sd = statistics.stdev(diffs)
            se = sd / (len(diffs) ** 0.5)
            ci_lo, ci_hi = delta - 1.96 * se, delta + 1.96 * se
        else:
            ci_lo = ci_hi = delta
        print(
            f"R@{k:2d}: A={a:.3f}  B={b:.3f}  delta={delta:+.3f}  "
            f"95% CI=[{ci_lo:+.3f}, {ci_hi:+.3f}]  "
            f"({sum(1 for d in diffs if d > 0)} wins, "
            f"{sum(1 for d in diffs if d == 0)} ties, "
            f"{sum(1 for d in diffs if d < 0)} losses)"
        )

    # Per-type aggregates.
    print("\n=== PER-TYPE AGGREGATES ===")
    by_type = defaultdict(lambda: defaultdict(list))
    for r in results:
        if "rA_at" not in r:
            continue
        for k in (1, 5, 10):
            by_type[r["question_type"]][f"A@{k}"].append(r["rA_at"][k])
            by_type[r["question_type"]][f"B@{k}"].append(r["rB_at"][k])
    print(
        f"{'type':40s} {'n':>3s} "
        f"{'A@1':>5s} {'B@1':>5s} {'d@1':>6s}  "
        f"{'A@5':>5s} {'B@5':>5s} {'d@5':>6s}  "
        f"{'A@10':>5s} {'B@10':>5s} {'d@10':>6s}"
    )
    for t in sorted(by_type.keys()):
        v = by_type[t]
        nt = len(v["A@1"])
        cells = []
        for k in (1, 5, 10):
            a = statistics.mean(v[f"A@{k}"])
            b = statistics.mean(v[f"B@{k}"])
            cells.append((a, b, b - a))
        print(
            f"{t[13:]:40s} {nt:>3d} "
            f"{cells[0][0]:>5.2f} {cells[0][1]:>5.2f} {cells[0][2]:+.3f}  "
            f"{cells[1][0]:>5.2f} {cells[1][1]:>5.2f} {cells[1][2]:+.3f}  "
            f"{cells[2][0]:>5.2f} {cells[2][1]:>5.2f} {cells[2][2]:+.3f}"
        )

    # Embedding cost.
    print("\n=== EMBEDDING COST ===")
    n_segs = sum(r["n_segments"] for r in results if "rA_at" in r)
    n_derivs = sum(r["n_derivatives"] for r in results if "rA_at" in r)
    print(f"total verbatim segments embedded (Index A): {n_segs}")
    print(f"total derivatives embedded (Index B added): {n_derivs}")
    print(f"avg derivatives per segment: {n_derivs / max(1, n_segs):.2f}")
    print(
        f"Index B / Index A embedding ratio: {(n_segs + n_derivs) / max(1, n_segs):.2f}x"
    )

    # ---- Wins ----
    print("\n=== WINS (B finds gold; A misses at R@5) ===")
    wins = []
    for r in results:
        if "rA_at" not in r:
            continue
        if r["rB_at"][5] > r["rA_at"][5]:
            wins.append(r)
    wins.sort(
        key=lambda r: -(r["rB_at"][10] - r["rA_at"][10] + r["rB_at"][5] - r["rA_at"][5])
    )
    for w in wins[:6]:
        print()
        print(f"--- {w['question_id']} ({w['question_type']}) ---")
        print(f"  Q: {short(w['question'], 200)}")
        print(f"  ANSWER: {short(w['answer'], 200)}")
        for g in w["diag"]["gold_segments"]:
            print(
                f"  GOLD seg {g['segment_id']} role={g['role']} rankA={g['rank_A']} rankB={g['rank_B']}"
            )
            print(f"    text: {short(g['text'], 240)}")
            if g["derivatives"]:
                for d in g["derivatives"][:5]:
                    print(f"    DERIV: {short(d, 240)}")

    # ---- Losses ----
    print("\n=== LOSSES (A finds gold; B regresses at R@5) ===")
    losses = []
    for r in results:
        if "rA_at" not in r:
            continue
        if r["rB_at"][5] < r["rA_at"][5]:
            losses.append(r)
    losses.sort(
        key=lambda r: r["rB_at"][10] - r["rA_at"][10] + r["rB_at"][5] - r["rA_at"][5]
    )
    for w in losses[:6]:
        print()
        print(f"--- {w['question_id']} ({w['question_type']}) ---")
        print(f"  Q: {short(w['question'], 200)}")
        print(f"  ANSWER: {short(w['answer'], 200)}")
        for g in w["diag"]["gold_segments"]:
            print(
                f"  GOLD seg {g['segment_id']} role={g['role']} rankA={g['rank_A']} rankB={g['rank_B']}"
            )
            print(f"    text: {short(g['text'], 240)}")
            if g["derivatives"]:
                for d in g["derivatives"][:5]:
                    print(f"    DERIV: {short(d, 240)}")
        # show what B's top-5 are instead.
        print("  B's top-5 (not gold):")
        for i, src in enumerate(w["diag"]["topB_sources"][:5]):
            tag = "GOLD" if src["is_gold"] else "miss"
            srctype = "DERIV" if src["from_derivative"] else "VERB "
            print(
                f"    {i + 1}. [{tag} {srctype}] seg={src['segment_id']} text: {short(src['match_text'], 200)}"
            )


if __name__ == "__main__":
    main()
