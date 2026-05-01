"""Grade the partial SQLite state from a halted ingest.

Reads MemoryEntry rows directly from the entry_store, runs the
cluster_id-based grader on the subset of dense_chains transitions whose turn
index is within the ingested prefix, and writes results/run.json.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import harness  # ensures sys.path + agamemnon patching is set up

sys.path.insert(
    0,
    str(harness.ROUND14 / "scenarios"),
)
import dense_chains
from memmachine_server.common.payload_codec.payload_codec_config import (
    PlaintextPayloadCodecConfig,
)
from memmachine_server.extra_memory.entry_store.data_types import (
    EntryStorePartitionConfig,
)
from memmachine_server.extra_memory.entry_store.sqlalchemy_entry_store import (
    SQLAlchemyEntryStore,
    SQLAlchemyEntryStoreParams,
)
from sqlalchemy.ext.asyncio import create_async_engine


async def load_entries(sqlite_path: Path, partition_key: str):
    engine = create_async_engine(f"sqlite+aiosqlite:///{sqlite_path}", future=True)
    store = SQLAlchemyEntryStore(SQLAlchemyEntryStoreParams(engine=engine))
    await store.startup()
    partition = await store.open_or_create_partition(
        partition_key,
        EntryStorePartitionConfig(payload_codec_config=PlaintextPayloadCodecConfig()),
    )
    # Use get_recent_entries with a high limit to dump everything
    entries = await partition.get_recent_entries(limit=10000)
    return entries


def main():
    sqlite_path = Path(harness.ROUND17 / "cache" / "extra_memory_a20.sqlite")
    if not sqlite_path.exists():
        print(f"missing {sqlite_path}")
        return

    entries = asyncio.run(load_entries(sqlite_path, "extra_memory_a20"))
    print(f"loaded {len(entries)} entries")

    if not entries:
        return

    # Determine the prefix of turns we actually ingested
    last_turn = max(harness._turn_idx_from_ts(e.timestamp) for e in entries)
    print(f"last ingested turn: {last_turn}")

    # Build a truncated scenario / GT by clipping turns and chains to last_turn
    turns = dense_chains.generate()
    gt = dense_chains.ground_truth(turns)
    truncated_turns = [t for t in turns if t.idx <= last_turn]
    truncated_gt = dense_chains.GroundTruth()
    for key, chain in gt.chains.items():
        clipped = [(t, v) for t, v in chain if t <= last_turn]
        if clipped:
            truncated_gt.chains[key] = clipped
    n_nf = sum(max(0, len(v) - 1) for v in truncated_gt.chains.values())
    print(f"truncated turns={len(truncated_turns)}  non-first transitions={n_nf}")

    metrics = harness.collect_metrics(
        truncated_turns, truncated_gt, entries, bucket_size=100
    )
    s = metrics["summary"]
    print("metrics:")
    print(f"  ref_emission_rate={s['ref_emission_rate']:.3f}")
    print(f"  ref_correctness_rate(refs)={s['ref_correctness_rate']:.3f}")
    print(f"  cluster_correctness_rate={s['cluster_correctness_rate']:.3f}")
    print(f"  entry_emission_rate={s['entry_emission_rate']:.3f}")
    print("buckets:")
    for b in s["bucket_stats"]:
        rate_e = b["ref_emission_rate"]
        rate_c = b["ref_correctness_rate"]
        rate_cl = b["cluster_correctness_rate"]
        s_e = f"{rate_e:.2f}" if rate_e is not None else " -- "
        s_c = f"{rate_c:.2f}" if rate_c is not None else " -- "
        s_cl = f"{rate_cl:.2f}" if rate_cl is not None else " -- "
        print(
            f"  {b['range']:>14s}  trans={b['n_transitions']:>3d}  "
            f"emit={s_e}  refs={s_c}  cluster={s_cl}"
        )

    out = {
        "partial": True,
        "n_turns_ingested": last_turn,
        "n_non_first_transitions": n_nf,
        "n_entries": len(entries),
        "log_size_bytes": sqlite_path.stat().st_size,
        "metrics_summary": s,
        "transitions": metrics["transitions"],
    }
    (harness.ROUND17 / "results" / "run_partial.json").write_text(
        json.dumps(out, indent=2, default=str)
    )
    print(f"wrote {harness.ROUND17 / 'results' / 'run_partial.json'}")


if __name__ == "__main__":
    main()
