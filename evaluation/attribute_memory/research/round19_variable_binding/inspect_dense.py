"""Inspect dense_chains entries to diagnose under-extraction."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path.insert(0, str(HERE / "architectures"))
sys.path.insert(0, str(RESEARCH / "round14_chain_density" / "scenarios"))
sys.path.insert(0, str(RESEARCH / "round7" / "experiments"))

import aen2_binding_v2 as ab  # noqa: E402
import dense_chains  # noqa: E402
from _common import Budget, Cache  # noqa: E402

CACHE_DIR = HERE / "cache"


def main():
    cache = Cache(CACHE_DIR / "v2_dense_K3_w7_w7.json")
    budget = Budget(
        max_llm=10000, max_embed=10000, stop_at_llm=9999, stop_at_embed=9999
    )
    turns_full = dense_chains.generate()
    turns = turns_full[:200]
    pairs = [(t.idx, t.text) for t in turns]
    log, resolutions, idx, telemetry = ab.ingest_turns(
        pairs,
        cache,
        budget,
        w_past=7,
        w_future=7,
        k=3,
        rebuild_index_every=4,
    )

    gt = dense_chains.ground_truth(turns)
    print("\n=== dense_chains[:200] ===")
    print(
        f"entries={len(log)} resolutions={len(resolutions)} clusters={len(idx.cluster_entries)} chains={len(idx.chain_head)}\n"
    )

    print("=== CHAIN HEADS ===")
    for (subj, pred), cid in idx.chain_head.items():
        head_uuid = idx.chain_head_entry[(subj, pred)]
        head = idx.by_uuid[head_uuid]
        label = idx.cluster_label.get(cid)
        print(
            f"  {pred:<25s} -> cluster={cid:<25s} label={label!r}  head=[{head.uuid} t={head.ts}] {head.text[:80]}"
        )

    print("\n=== GT CHAINS (transition history) ===")
    for key, transitions in gt.chains.items():
        print(f"  {key}: {len(transitions)} transitions")
        for t, v in transitions:
            # Find any entry near t that mentions v
            matches = [
                e
                for e in log
                if abs(e.ts - t) <= 4
                and (
                    v.lower() in e.text.lower()
                    or any(v.lower() in m.lower() for m in e.mentions)
                )
            ]
            covers = ", ".join(
                f"e{e.ts}_{e.uuid[-3:]}={e.predicate or '-'}/cid={e.cluster_id}"
                for e in matches[:2]
            )
            tag = "✓" if matches else "✗"
            print(f"    {tag} t={t}: {v}  -- {covers or 'NO ENTRY'}")


if __name__ == "__main__":
    main()
