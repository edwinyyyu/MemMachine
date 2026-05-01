"""Re-grade coref pairs with a LOCATION-AGNOSTIC metric.

The original `coref_pair_correctness` looks for the resolution near n_turn.
Centered writers can commit the resolution at d_turn (because future is
visible), which causes the metric to undercount.

This script:
  1. Re-runs ingest from cache (free, no LLM calls).
  2. Computes location-agnostic resolution metrics:
     - resolved_anywhere_strict: ANY entry in the log with @{name} mention +
       right predicate, with timestamp in [d_turn-1, n_turn+5].
     - resolved_in_d_window: resolution committed in the descriptor's window
       (d_turn-1 .. d_turn+w_future+1).
     - resolved_in_n_window: original metric (entries near n_turn).
     - resolved_descriptor_first: an entry at d_turn that ALREADY mentions
       @{name} (centered-writer hindsight commit).
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent

sys.path.insert(0, str(HERE / "architectures"))
sys.path.insert(0, str(RESEARCH / "round16a_sliding_window" / "architectures"))
sys.path.insert(0, str(RESEARCH / "round16a_sliding_window" / "scenarios"))
sys.path.insert(0, str(RESEARCH / "round15_active_chains" / "architectures"))
sys.path.insert(0, str(RESEARCH / "round11_writer_stress" / "architectures"))
sys.path.insert(0, str(RESEARCH / "round7" / "experiments"))

import aen1_centered  # noqa: E402
import multi_batch_coref  # noqa: E402
from _common import Budget, Cache  # noqa: E402

CACHE_DIR = HERE / "cache"


def regrade_anywhere(log, gt) -> dict:
    """Location-agnostic coref grading."""
    by_ts: dict[int, list] = {}
    for e in log:
        by_ts.setdefault(e.ts, []).append(e)

    out = []
    for p in gt.pairs:
        name = p["name"]
        pred = p["predicate"]  # ("@User", "boss")
        d_turn = p["descriptor_turn"]
        n_turn = p["name_turn"]
        full_pred = f"{pred[0].lstrip('@')}.{pred[1]}".lower()

        # ANY entry in the entire log
        all_entries_with_name = [
            e for e in log if f"@{name}" in e.mentions or name.lower() in e.text.lower()
        ]
        all_entries_with_strict = [
            e
            for e in all_entries_with_name
            if e.predicate and e.predicate.replace("@", "").lower() == full_pred
        ]
        # Window-of-the-pair: anywhere from d_turn-1 to n_turn+5
        pair_window_strict = [
            e for e in all_entries_with_strict if d_turn - 1 <= e.ts <= n_turn + 5
        ]
        # Did the writer commit early (at descriptor's window)?
        descriptor_window_strict = [
            e
            for e in all_entries_with_strict
            if d_turn - 1 <= e.ts <= d_turn + 8  # descriptor + ~K + a few
        ]
        # Did the writer commit at name-turn vicinity?
        name_window_strict = [
            e for e in all_entries_with_strict if n_turn - 2 <= e.ts <= n_turn + 5
        ]
        out.append(
            {
                "name": name,
                "pred": f"{pred[0]}.{pred[1]}",
                "d_turn": d_turn,
                "n_turn": n_turn,
                "gap": n_turn - d_turn,
                "any_in_log_strict": len(all_entries_with_strict) > 0,
                "in_pair_window_strict": len(pair_window_strict) > 0,
                "in_d_window_strict": len(descriptor_window_strict) > 0,
                "in_n_window_strict": len(name_window_strict) > 0,
                "n_strict_entries": len(all_entries_with_strict),
                "earliest_strict_ts": min(
                    (e.ts for e in all_entries_with_strict), default=None
                ),
                "latest_strict_ts": max(
                    (e.ts for e in all_entries_with_strict), default=None
                ),
            }
        )
    n = len(out)
    return {
        "pairs": out,
        "any_in_log_pass": sum(1 for r in out if r["any_in_log_strict"]),
        "in_pair_window_pass": sum(1 for r in out if r["in_pair_window_strict"]),
        "in_d_window_pass": sum(1 for r in out if r["in_d_window_strict"]),
        "in_n_window_pass": sum(1 for r in out if r["in_n_window_strict"]),
        "total": n,
    }


def regrade_variant(variant_name: str, w_past: int, w_future: int, k: int):
    cache_path = CACHE_DIR / f"{variant_name}.json"
    if not cache_path.exists():
        print(f"!!! cache missing: {cache_path}")
        return None
    cache = Cache(cache_path)
    # Use a budget with everything pre-cached; if any miss, abort.
    budget = Budget(
        max_llm=10000, max_embed=10000, stop_at_llm=9000, stop_at_embed=9000
    )

    coref_turns = multi_batch_coref.generate()
    coref_gt = multi_batch_coref.ground_truth(coref_turns)
    pairs = [(t.idx, t.text) for t in coref_turns]

    log, idx, telemetry = aen1_centered.ingest_turns(
        pairs,
        cache,
        budget,
        w_past=w_past,
        w_future=w_future,
        k=k,
        rebuild_index_every=4,
        max_active_state_size=100,
    )
    print(
        f"\n=== {variant_name} (n_entries={len(log)}, llm_calls={budget.llm_calls}) ==="
    )

    res = regrade_anywhere(log, coref_gt)
    print(
        f"any_in_log:      {res['any_in_log_pass']}/{res['total']}   "
        f"(resolution exists ANYWHERE in the log)"
    )
    print(
        f"in_pair_window:  {res['in_pair_window_pass']}/{res['total']}   "
        f"(resolution between d_turn-1 and n_turn+5)"
    )
    print(
        f"in_d_window:     {res['in_d_window_pass']}/{res['total']}   "
        f"(resolution committed AT descriptor's batch — centered's signature)"
    )
    print(
        f"in_n_window:     {res['in_n_window_pass']}/{res['total']}   "
        f"(original metric: resolution near n_turn)"
    )
    for r in res["pairs"]:
        first = r["earliest_strict_ts"]
        delta = (first - r["d_turn"]) if first is not None else None
        print(
            f"  {r['name']:<10s} pred={r['pred']:<22s} gap={r['gap']:>3d}  "
            f"any={r['any_in_log_strict']!s:<5s} "
            f"first_strict_ts={first} (d_turn+{delta if delta is not None else '--'})"
        )

    # Show entries about Marcus to verify
    print("\n  All entries mentioning @Marcus or 'marcus':")
    for e in log:
        if "@Marcus" in e.mentions or "marcus" in e.text.lower():
            print(
                f"    t{e.ts:>3d} [{e.uuid}] mentions={e.mentions} pred={e.predicate} :: {e.text[:90]}"
            )

    return res


if __name__ == "__main__":
    regrade_variant("coref_centered_K3_w6_w6", w_past=6, w_future=6, k=3)
