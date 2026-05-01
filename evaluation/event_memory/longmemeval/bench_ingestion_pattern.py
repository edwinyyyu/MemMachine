"""
Benchmark that replicates the eviction ingestion pattern against Qdrant,
without actual episodes, SQL, or embeddings.

Simulates: 500 questions × ~48 sessions × ~110 derivatives per session.
Each session does: batch query (eviction) → batch upsert.
Serialized within each question (semaphore 2), concurrent across questions.
"""

import argparse
import asyncio
import random
import time
import uuid

import docker
import numpy as np
from qdrant_client import AsyncQdrantClient, models

# Scale parameters (from longmemeval_s_cleaned.json analysis)
NUM_QUESTIONS = 50  # subset for benchmarking
SESSIONS_PER_QUESTION = 48
DERIVATIVES_PER_SESSION = 110
DIMENSIONS = 1536
EVICTION_SEARCH_LIMIT = 20
EVICTION_TOP_K = 20
QUESTION_CONCURRENCY = 50
SESSION_CONCURRENCY = 2


async def run_benchmark(http_port: int, grpc_port: int, collection_name: str):
    client = AsyncQdrantClient(
        host="localhost",
        prefer_grpc=True,
        port=http_port,
        grpc_port=grpc_port,
        timeout=300,
    )

    # Pre-generate all vectors
    np.random.seed(42)
    print("Generating vectors...")
    total_vecs = NUM_QUESTIONS * SESSIONS_PER_QUESTION * DERIVATIVES_PER_SESSION
    print(
        f"  {NUM_QUESTIONS} questions × {SESSIONS_PER_QUESTION} sessions "
        f"× {DERIVATIVES_PER_SESSION} derivatives = {total_vecs} total"
    )

    # Stats tracking
    query_latencies: list[float] = []
    upsert_latencies: list[float] = []
    session_latencies: list[float] = []
    lock_wait_times: list[float] = []
    total_vectors_inserted = 0
    vectors_lock = asyncio.Lock()

    async def process_session(
        question_idx: int,
        session_idx: int,
        encode_lock: asyncio.Lock,
    ):
        nonlocal total_vectors_inserted

        # Simulate embedding step (outside lock) — just generate random vectors
        vecs = np.random.randn(DERIVATIVES_PER_SESSION, DIMENSIONS).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / norms

        session_start = time.monotonic()

        # Acquire lock (simulates serialize_encode)
        lock_wait_start = time.monotonic()
        async with encode_lock:
            lock_waited = time.monotonic() - lock_wait_start

            # Step 1: Eviction query (batch)
            query_start = time.monotonic()
            requests = [
                models.QueryRequest(
                    query=v.tolist(),
                    limit=EVICTION_TOP_K,
                    score_threshold=0.9,
                    with_payload=True,
                )
                for v in vecs
            ]
            await client.query_batch_points(collection_name, requests=requests)
            query_time = time.monotonic() - query_start

            # Step 2: Upsert
            upsert_start = time.monotonic()
            points = [
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=v.tolist(),
                    payload={
                        "_segment_uuid": str(uuid.uuid4()),
                        "_timestamp": "2024-01-15T12:00:00Z",
                    },
                )
                for v in vecs
            ]
            await client.upsert(collection_name=collection_name, points=points)
            upsert_time = time.monotonic() - upsert_start

        session_time = time.monotonic() - session_start

        async with vectors_lock:
            query_latencies.append(query_time)
            upsert_latencies.append(upsert_time)
            session_latencies.append(session_time)
            lock_wait_times.append(lock_waited)
            total_vectors_inserted += DERIVATIVES_PER_SESSION

    async def process_question(question_idx: int):
        encode_lock = asyncio.Lock()
        session_sem = asyncio.Semaphore(SESSION_CONCURRENCY)

        async def bounded_session(session_idx):
            async with session_sem:
                await process_session(question_idx, session_idx, encode_lock)

        await asyncio.gather(
            *[bounded_session(i) for i in range(SESSIONS_PER_QUESTION)]
        )

    # Run
    print("\nStarting benchmark...")
    print(
        f"  Question concurrency: {QUESTION_CONCURRENCY}, "
        f"Session concurrency per question: {SESSION_CONCURRENCY}"
    )

    question_sem = asyncio.Semaphore(QUESTION_CONCURRENCY)
    wall_start = time.monotonic()

    async def bounded_question(idx):
        async with question_sem:
            await process_question(idx)
            if (idx + 1) % 10 == 0:
                elapsed = time.monotonic() - wall_start
                rate = total_vectors_inserted / elapsed if elapsed > 0 else 0
                print(
                    f"  Questions {idx + 1}/{NUM_QUESTIONS} done, "
                    f"{total_vectors_inserted} vecs, {rate:.0f} vecs/sec"
                )

    await asyncio.gather(*[bounded_question(i) for i in range(NUM_QUESTIONS)])

    wall_time = time.monotonic() - wall_start
    throughput = total_vectors_inserted / wall_time

    # Report
    import statistics

    def pct(lst, p):
        s = sorted(lst)
        return s[int(len(s) * p)] * 1000

    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print(f"Total vectors inserted:  {total_vectors_inserted}")
    print(f"Wall time:               {wall_time:.1f}s")
    print(f"Throughput:              {throughput:.0f} vecs/sec")
    print("\nPer-session breakdown (ms):")
    print(
        f"  Query:      p50={pct(query_latencies, 0.5):.0f}  "
        f"p95={pct(query_latencies, 0.95):.0f}  "
        f"p99={pct(query_latencies, 0.99):.0f}"
    )
    print(
        f"  Upsert:     p50={pct(upsert_latencies, 0.5):.0f}  "
        f"p95={pct(upsert_latencies, 0.95):.0f}  "
        f"p99={pct(upsert_latencies, 0.99):.0f}"
    )
    print(
        f"  Lock wait:  p50={pct(lock_wait_times, 0.5):.0f}  "
        f"p95={pct(lock_wait_times, 0.95):.0f}  "
        f"p99={pct(lock_wait_times, 0.99):.0f}"
    )
    print(
        f"  Session:    p50={pct(session_latencies, 0.5):.0f}  "
        f"p95={pct(session_latencies, 0.95):.0f}  "
        f"p99={pct(session_latencies, 0.99):.0f}"
    )
    print(
        f"\n  Avg query:  {statistics.mean(query_latencies) * 1000:.0f}ms  "
        f"({statistics.mean(query_latencies) / DERIVATIVES_PER_SESSION * 1000:.2f}ms/vec)"
    )
    print(
        f"  Avg upsert: {statistics.mean(upsert_latencies) * 1000:.0f}ms  "
        f"({statistics.mean(upsert_latencies) / DERIVATIVES_PER_SESSION * 1000:.2f}ms/vec)"
    )

    await client.close()


def main():
    parser = argparse.ArgumentParser(description="Benchmark eviction ingestion pattern")
    parser.add_argument("--http-port", type=int, default=None, help="Qdrant HTTP port")
    parser.add_argument("--grpc-port", type=int, default=None, help="Qdrant gRPC port")
    parser.add_argument(
        "--collection",
        default="bench_ingestion",
        help="Collection name",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Spin up a fresh Qdrant testcontainer",
    )
    args = parser.parse_args()

    if args.fresh:
        client_docker = docker.from_env()
        cname = "bench-qdrant-" + str(random.randint(10000, 99999))
        print(f"Starting fresh Qdrant container: {cname}")
        container = client_docker.containers.run(
            "qdrant/qdrant:v1.17.0",
            detach=True,
            name=cname,
            ports={"6333/tcp": None, "6334/tcp": None},
            remove=True,
        )
        time.sleep(3)
        container.reload()
        http_port = int(container.ports["6333/tcp"][0]["HostPort"])
        grpc_port = int(container.ports["6334/tcp"][0]["HostPort"])
        print(f"Container ready on HTTP:{http_port} gRPC:{grpc_port}")
    else:
        http_port = args.http_port or 6333
        grpc_port = args.grpc_port or 6334
        container = None

    # Create collection
    from qdrant_client import QdrantClient

    sync_client = QdrantClient(host="localhost", port=http_port, timeout=300)
    if sync_client.collection_exists(args.collection):
        sync_client.delete_collection(args.collection)
    sync_client.create_collection(
        collection_name=args.collection,
        vectors_config=models.VectorParams(
            size=DIMENSIONS, distance=models.Distance.COSINE
        ),
    )
    print(f"Created collection '{args.collection}'")
    sync_client.close()

    try:
        asyncio.run(run_benchmark(http_port, grpc_port, args.collection))
    finally:
        if container:
            print("\nStopping container...")
            container.stop(timeout=5)
            print("Done.")


if __name__ == "__main__":
    main()
