"""Re-ingest from cache and dump emitted entries + resolutions for diagnosis."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
sys.path.insert(0, str(HERE / "architectures"))
sys.path.insert(0, str(RESEARCH / "round16a_sliding_window" / "scenarios"))
sys.path.insert(0, str(RESEARCH / "round7" / "experiments"))

import aen2_binding  # noqa: E402
import multi_batch_coref  # noqa: E402
from _common import Budget, Cache  # noqa: E402

CACHE_DIR = HERE / "cache"


def dump(variant_name: str, w_past: int, w_future: int, k: int):
    cache = Cache(CACHE_DIR / f"{variant_name}.json")
    budget = Budget(
        max_llm=10000, max_embed=10000, stop_at_llm=9999, stop_at_embed=9999
    )
    coref_turns = multi_batch_coref.generate()
    pairs = [(t.idx, t.text) for t in coref_turns]
    log, resolutions, idx, telemetry = aen2_binding.ingest_turns(
        pairs,
        cache,
        budget,
        w_past=w_past,
        w_future=w_future,
        k=k,
        rebuild_index_every=4,
    )

    print(
        f"\n=== {variant_name}: {len(log)} entries, {len(resolutions)} resolutions ===\n"
    )
    print("=== ENTRIES ===")
    for e in log:
        print(
            f"  t={e.ts:>3d} [{e.uuid:<10s}] cluster={e.cluster_id:<22s}"
            f" subj={(e.subject or '-'):<8s} pred={(e.predicate or '-'):<22s}"
            f" :: {e.text[:100]}"
        )
    print("\n=== RESOLUTIONS ===")
    for r in resolutions:
        print(
            f"  t={r.ts:>3d} [{r.uuid:<10s}] cluster={r.cluster_id:<22s} -> {r.canonical_label!r}"
            f" evidence={r.evidence_entry_uuids}"
        )
    print("\n=== CLUSTERS ===")
    for cid, entries in idx.cluster_entries.items():
        label = idx.cluster_label.get(cid)
        print(f"  cluster={cid!r}  label={label!r}  n_entries={len(entries)}")


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "binding_K3_w7_w14"
    if variant == "binding_K3_w7_w14":
        dump(variant, w_past=7, w_future=14, k=3)
    elif variant == "binding_K3_w7_w7":
        dump(variant, w_past=7, w_future=7, k=3)
    else:
        print(f"unknown variant: {variant}")
