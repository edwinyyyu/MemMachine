"""
Launch N parallel shards of the eviction ingestion script.

Usage:
    uv run longmemeval_ingest_eviction_parallel.py --shards 10 --log-dir logs \
        --data-path evaluation/data/longmemeval_s_cleaned.json \
        --derive-sentences --eviction-similarity-threshold 0.9

All arguments except --shards and --log-dir are forwarded to the ingest script.
Logs go to --log-dir (default: current directory), one file per shard.
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


async def initialize_stores():
    """Create DB tables and vector store collections before shards start."""
    from memmachine_server.common.vector_store.qdrant_vector_store import (
        QdrantVectorStore,
        QdrantVectorStoreParams,
    )
    from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import (
        SQLAlchemySegmentStore,
        SQLAlchemySegmentStoreParams,
    )
    from qdrant_client import AsyncQdrantClient
    from sqlalchemy.ext.asyncio import create_async_engine

    qdrant_client = AsyncQdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        prefer_grpc=True,
        timeout=300,
        port=int(os.getenv("QDRANT_PORT", "6333")),
        grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
    )
    vector_store = QdrantVectorStore(QdrantVectorStoreParams(client=qdrant_client))
    await vector_store.startup()

    engine = create_async_engine(os.getenv("SQL_URL"))
    segment_store = SQLAlchemySegmentStore(SQLAlchemySegmentStoreParams(engine=engine))
    await segment_store.startup()

    await segment_store.shutdown()
    await vector_store.shutdown()
    await engine.dispose()
    await qdrant_client.close()


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Launch parallel shards of eviction ingestion"
    )
    parser.add_argument("--shards", type=int, default=10, help="Number of shards")
    parser.add_argument(
        "--log-dir",
        default=".",
        help="Directory for shard log files",
    )
    args, passthrough_args = parser.parse_known_args()

    # Find --data-path in passthrough args to count questions.
    data_path = None
    for i, arg in enumerate(passthrough_args):
        if arg == "--data-path" and i + 1 < len(passthrough_args):
            data_path = passthrough_args[i + 1]
            break
    if data_path is None:
        print("Error: --data-path is required", file=sys.stderr)
        sys.exit(1)

    with open(data_path) as f:
        total_questions = len(json.load(f))

    per_shard = total_questions // args.shards
    remainder = total_questions % args.shards
    shard_sizes = [per_shard + (1 if i < remainder else 0) for i in range(args.shards)]

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing DB tables and vector store...", flush=True)
    asyncio.run(initialize_stores())
    print("Initialization done.", flush=True)

    print(f"\nTotal questions: {total_questions}")
    print(f"Shards: {args.shards} (sizes: {shard_sizes})")
    print(f"Log dir: {log_dir}")
    print(f"Ingest args: {passthrough_args}")
    print(flush=True)

    script = str(Path(__file__).parent / "longmemeval_ingest_eviction.py")
    processes: list[tuple[int, subprocess.Popen, Path]] = []

    offset = 0
    for i in range(args.shards):
        limit = shard_sizes[i]

        log_path = log_dir / f"eviction_shard_{i}.out"
        print(f"Shard {i}: offset={offset} limit={limit} -> {log_path}", flush=True)

        log_file = open(log_path, "w")
        proc = subprocess.Popen(
            [
                sys.executable,
                script,
                "--question-offset",
                str(offset),
                "--question-limit",
                str(limit),
                "--skip-startup",
                *passthrough_args,
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        processes.append((i, proc, log_path))
        offset += limit

    print(f"\nAll {args.shards} shards launched.", flush=True)
    print("Waiting for completion...", flush=True)
    start = time.monotonic()

    failed = 0
    for i, proc, log_path in processes:
        proc.wait()
        if proc.returncode == 0:
            print(f"Shard {i} finished successfully", flush=True)
        else:
            print(
                f"Shard {i} FAILED (exit code {proc.returncode}). See {log_path}",
                file=sys.stderr,
                flush=True,
            )
            failed += 1

    elapsed = time.monotonic() - start
    print(f"\nElapsed: {elapsed:.0f}s", flush=True)

    if failed == 0:
        print("All shards completed successfully.", flush=True)
    else:
        print(f"{failed} shard(s) failed. Check logs: eviction_shard_*.out", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
