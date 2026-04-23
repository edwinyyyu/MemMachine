"""Sweep question concurrency to find the optimal level."""

import asyncio
import random
import time
import uuid

import docker
import numpy as np
from qdrant_client import AsyncQdrantClient, QdrantClient, models

SESSIONS_PER_QUESTION = 48
DERIVATIVES_PER_SESSION = 110
DIMENSIONS = 1536
NUM_QUESTIONS = 20  # smaller for sweep


async def run_one(http_port, grpc_port, collection, q_conc, s_conc):
    client = AsyncQdrantClient(
        host="localhost",
        prefer_grpc=True,
        port=http_port,
        grpc_port=grpc_port,
        timeout=300,
    )

    total_inserted = 0
    insert_lock = asyncio.Lock()

    async def process_session(encode_lock):
        nonlocal total_inserted
        vecs = np.random.randn(DERIVATIVES_PER_SESSION, DIMENSIONS).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / norms

        async with encode_lock:
            requests = [
                models.QueryRequest(
                    query=v.tolist(),
                    limit=20,
                    score_threshold=0.9,
                    with_payload=True,
                )
                for v in vecs
            ]
            await client.query_batch_points(collection, requests=requests)

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
            await client.upsert(collection_name=collection, points=points)

        async with insert_lock:
            total_inserted += DERIVATIVES_PER_SESSION

    async def process_question():
        encode_lock = asyncio.Lock()
        sem = asyncio.Semaphore(s_conc)

        async def bounded(i):
            async with sem:
                await process_session(encode_lock)

        await asyncio.gather(*[bounded(i) for i in range(SESSIONS_PER_QUESTION)])

    sem = asyncio.Semaphore(q_conc)

    async def bounded_q(i):
        async with sem:
            await process_question()

    start = time.monotonic()
    await asyncio.gather(*[bounded_q(i) for i in range(NUM_QUESTIONS)])
    wall = time.monotonic() - start

    await client.close()
    return total_inserted, wall, total_inserted / wall


def main():
    client_docker = docker.from_env()
    cname = "bench-qdrant-sweep-" + str(random.randint(10000, 99999))
    print(f"Starting container: {cname}")
    container = client_docker.containers.run(
        "qdrant/qdrant:v1.17.0",
        detach=True,
        name=cname,
        ports={"6333/tcp": None, "6334/tcp": None},
        remove=True,
    )
    time.sleep(3)
    container.reload()
    hp = int(container.ports["6333/tcp"][0]["HostPort"])
    gp = int(container.ports["6334/tcp"][0]["HostPort"])
    print(f"Ports: HTTP={hp} gRPC={gp}")

    col = "bench_sweep"
    sync = QdrantClient(host="localhost", port=hp, timeout=300)

    try:
        print(f"\n{'q_conc':>6} {'s_conc':>6} {'vecs/s':>10} {'wall':>8}")
        print("-" * 35)

        for q_conc in [1, 5, 10, 25, 50]:
            for s_conc in [1, 2]:
                if sync.collection_exists(col):
                    sync.delete_collection(col)
                sync.create_collection(
                    col,
                    vectors_config=models.VectorParams(
                        size=DIMENSIONS, distance=models.Distance.COSINE
                    ),
                )

                np.random.seed(42)
                total, wall, rate = asyncio.run(run_one(hp, gp, col, q_conc, s_conc))
                print(f"{q_conc:>6} {s_conc:>6} {rate:>10.0f} {wall:>7.1f}s")
    finally:
        sync.close()
        container.stop(timeout=5)
        print("\nDone.")


if __name__ == "__main__":
    main()
