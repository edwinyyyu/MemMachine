"""Benchmark: cost of redundant Qdrant upserts (uuid5 dedup scenario)."""

import asyncio
import time
from uuid import UUID, uuid4

import numpy as np
from qdrant_client import AsyncQdrantClient, models
from testcontainers.qdrant import QdrantContainer

NAMESPACE = UUID("b7e3f1a2-8c4d-4f6e-9a1b-2d5e8f0c3a7b")
DIMS = 1536
COLLECTION = "bench"
NUM_VECTORS = 6000  # typical partition size
BATCH_SIZE = 500
DUP_RATIO = 0.15  # ~15% duplicates, matching what we observed


def make_vectors(n):
    vecs = np.random.default_rng(42).standard_normal((n, DIMS)).astype(np.float32)
    # normalize to unit length (cosine)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return (vecs / norms).tolist()


async def create_collection(client):
    await client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(size=DIMS, distance=models.Distance.COSINE),
        hnsw_config=models.HnswConfigDiff(m=0, payload_m=16),
    )
    await client.create_payload_index(
        collection_name=COLLECTION,
        field_name="partition",
        field_schema=models.KeywordIndexParams(
            type=models.KeywordIndexType.KEYWORD,
            is_tenant=True,
        ),
    )


async def bench_fresh_insert(client, vectors):
    """Insert all unique vectors (baseline)."""
    await client.delete_collection(COLLECTION)
    await create_collection(client)

    points = [
        models.PointStruct(id=str(uuid4()), vector=v, payload={"partition": "a"})
        for v in vectors
    ]

    start = time.monotonic()
    for i in range(0, len(points), BATCH_SIZE):
        await client.upsert(
            collection_name=COLLECTION, points=points[i : i + BATCH_SIZE]
        )
    elapsed = time.monotonic() - start
    return elapsed


async def bench_with_duplicates(client, vectors, dup_ratio):
    """Insert vectors, then re-upsert dup_ratio of them with same ID and vector (uuid5 scenario)."""
    await client.delete_collection(COLLECTION)
    await create_collection(client)

    # First pass: insert all
    ids = [str(uuid4()) for _ in vectors]
    points = [
        models.PointStruct(id=id_, vector=v, payload={"partition": "a"})
        for id_, v in zip(ids, vectors)
    ]

    start = time.monotonic()
    for i in range(0, len(points), BATCH_SIZE):
        await client.upsert(
            collection_name=COLLECTION, points=points[i : i + BATCH_SIZE]
        )

    # Second pass: re-upsert dup_ratio of them (simulates duplicate derivatives in next session)
    n_dups = int(len(vectors) * dup_ratio)
    dup_points = points[:n_dups]  # same IDs, same vectors
    for i in range(0, len(dup_points), BATCH_SIZE):
        await client.upsert(
            collection_name=COLLECTION, points=dup_points[i : i + BATCH_SIZE]
        )

    elapsed = time.monotonic() - start
    return elapsed


async def bench_extra_upsert_only(client, vectors, dup_ratio):
    """Measure ONLY the cost of the redundant upserts (not the initial insert)."""
    await client.delete_collection(COLLECTION)
    await create_collection(client)

    ids = [str(uuid4()) for _ in vectors]
    points = [
        models.PointStruct(id=id_, vector=v, payload={"partition": "a"})
        for id_, v in zip(ids, vectors)
    ]

    # Insert all first (not timed)
    for i in range(0, len(points), BATCH_SIZE):
        await client.upsert(
            collection_name=COLLECTION, points=points[i : i + BATCH_SIZE]
        )

    # Wait for indexing
    await asyncio.sleep(2)

    # Time only the redundant upserts
    n_dups = int(len(vectors) * dup_ratio)
    dup_points = points[:n_dups]

    start = time.monotonic()
    for i in range(0, len(dup_points), BATCH_SIZE):
        await client.upsert(
            collection_name=COLLECTION, points=dup_points[i : i + BATCH_SIZE]
        )
    elapsed = time.monotonic() - start
    return elapsed, n_dups


async def run():
    with QdrantContainer("qdrant/qdrant:latest") as qdrant:
        url = f"http://{qdrant.get_container_host_ip()}:{qdrant.get_exposed_port(6333)}"
        client = AsyncQdrantClient(url=url)

        print(f"Generating {NUM_VECTORS} vectors of dim {DIMS}...")
        vectors = make_vectors(NUM_VECTORS)

        # Warmup
        await bench_fresh_insert(client, vectors[:100])

        print(
            f"\n--- Benchmark: {NUM_VECTORS} vectors, {DUP_RATIO:.0%} duplicate ratio ---\n"
        )

        # Fresh insert (baseline)
        t1 = await bench_fresh_insert(client, vectors)
        print(f"Fresh insert ({NUM_VECTORS} unique):         {t1:.3f}s")

        # Insert + redundant upserts
        t2 = await bench_with_duplicates(client, vectors, DUP_RATIO)
        print(
            f"Insert + {DUP_RATIO:.0%} redundant upserts:    {t2:.3f}s  (overhead: {t2 - t1:.3f}s)"
        )

        # Just the redundant upserts
        t3, n = await bench_extra_upsert_only(client, vectors, DUP_RATIO)
        print(
            f"Redundant upsert only ({n} points):   {t3:.3f}s  ({t3 / n * 1000:.2f}ms/point)"
        )

        # Compare: fresh insert of same count
        t4 = await bench_fresh_insert(client, vectors[: int(NUM_VECTORS * DUP_RATIO)])
        print(
            f"Fresh insert of {int(NUM_VECTORS * DUP_RATIO)} points:       {t4:.3f}s  ({t4 / int(NUM_VECTORS * DUP_RATIO) * 1000:.2f}ms/point)"
        )

        await client.close()


asyncio.run(run())
