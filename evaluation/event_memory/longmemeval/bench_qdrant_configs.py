"""
Benchmark Qdrant configs with the real ingestion pattern (query+upsert),
monitoring hardware utilization via docker stats.
"""

import asyncio
import random
import statistics
import threading
import time
import uuid

import docker
import numpy as np
from qdrant_client import AsyncQdrantClient, QdrantClient, models

DIMENSIONS = 1536
NUM_VECTORS = 50000
BATCH_SIZE = 110
TOP_K = 20
NUM_QUESTIONS = 20
SESSIONS_PER_Q = 48
Q_CONC = 10
S_CONC = 2


def make_vecs(n):
    v = np.random.randn(n, DIMENSIONS).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def fill(sync, col, hnsw_m=16, hnsw_ef_construct=128, quantization=None):
    if sync.collection_exists(col):
        sync.delete_collection(col)
    sync.create_collection(
        col,
        vectors_config=models.VectorParams(
            size=DIMENSIONS, distance=models.Distance.COSINE
        ),
        hnsw_config=models.HnswConfigDiff(m=hnsw_m, ef_construct=hnsw_ef_construct),
        quantization_config=quantization,
    )
    data = make_vecs(NUM_VECTORS)
    for start in range(0, NUM_VECTORS, 1000):
        end = min(start + 1000, NUM_VECTORS)
        sync.upsert(
            col,
            points=[
                models.PointStruct(
                    id=i,
                    vector=data[i].tolist(),
                    payload={
                        "_segment_uuid": str(uuid.uuid4()),
                        "_timestamp": "2024-01-15T12:00:00Z",
                    },
                )
                for i in range(start, end)
            ],
        )
    time.sleep(5)
    info = sync.get_collection(col)
    print(
        f"  Filled: {info.points_count} pts, indexed={info.indexed_vectors_count}",
        flush=True,
    )


async def run_ingestion(hp, gp, col, search_ef):
    client = AsyncQdrantClient(
        host="localhost",
        prefer_grpc=True,
        port=hp,
        grpc_port=gp,
        timeout=300,
    )

    total_vecs = [0]

    async def session(encode_lock):
        vecs = make_vecs(BATCH_SIZE)
        async with encode_lock:
            reqs = [
                models.QueryRequest(
                    query=v.tolist(),
                    limit=TOP_K,
                    score_threshold=0.9,
                    with_payload=True,
                    params=models.SearchParams(hnsw_ef=search_ef),
                )
                for v in vecs
            ]
            await client.query_batch_points(col, requests=reqs)
            points = [
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=v.tolist(),
                    payload={"_timestamp": "2024-01-15T12:00:00Z"},
                )
                for v in vecs
            ]
            await client.upsert(collection_name=col, points=points)
        total_vecs[0] += BATCH_SIZE

    async def question():
        lock = asyncio.Lock()
        sem = asyncio.Semaphore(S_CONC)

        async def bounded(i):
            async with sem:
                await session(lock)

        await asyncio.gather(*[bounded(i) for i in range(SESSIONS_PER_Q)])

    q_sem = asyncio.Semaphore(Q_CONC)

    async def bounded_q(i):
        async with q_sem:
            await question()

    start = time.monotonic()
    await asyncio.gather(*[bounded_q(i) for i in range(NUM_QUESTIONS)])
    wall = time.monotonic() - start
    rate = total_vecs[0] / wall

    await client.close()
    return total_vecs[0], wall, rate


def main():
    np.random.seed(42)

    dc = docker.from_env()
    cname = f"bench-cfg-{random.randint(10000, 99999)}"
    container = dc.containers.run(
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
    print(f"Container {cname} on HTTP:{hp} gRPC:{gp}\n", flush=True)

    sync = QdrantClient(host="localhost", port=hp, timeout=300)

    # Monitor docker stats
    cpu_samples: list[float] = []
    mem_samples: list[float] = []
    monitoring = False

    def monitor():
        while monitoring:
            try:
                stats = container.stats(stream=False)
                cpu_d = (
                    stats["cpu_stats"]["cpu_usage"]["total_usage"]
                    - stats["precpu_stats"]["cpu_usage"]["total_usage"]
                )
                sys_d = (
                    stats["cpu_stats"]["system_cpu_usage"]
                    - stats["precpu_stats"]["system_cpu_usage"]
                )
                n = stats["cpu_stats"]["online_cpus"]
                if sys_d > 0:
                    cpu_samples.append((cpu_d / sys_d) * n * 100)
                mem_samples.append(stats["memory_stats"]["usage"] / (1024 * 1024))
            except Exception:
                pass
            time.sleep(1)

    configs = [
        (
            "default m=16 ef_c=128",
            dict(hnsw_m=16, hnsw_ef_construct=128, quantization=None),
            128,
        ),
        (
            "default + search_ef=64",
            dict(hnsw_m=16, hnsw_ef_construct=128, quantization=None),
            64,
        ),
        (
            "default + search_ef=32",
            dict(hnsw_m=16, hnsw_ef_construct=128, quantization=None),
            32,
        ),
        (
            "m=8 ef_c=64 + search_ef=64",
            dict(hnsw_m=8, hnsw_ef_construct=64, quantization=None),
            64,
        ),
        (
            "m=8 + scalar_quant + ef=64",
            dict(
                hnsw_m=8,
                hnsw_ef_construct=64,
                quantization=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8, always_ram=True
                    )
                ),
            ),
            64,
        ),
        (
            "m=8 + scalar_quant + ef=32",
            dict(
                hnsw_m=8,
                hnsw_ef_construct=64,
                quantization=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8, always_ram=True
                    )
                ),
            ),
            32,
        ),
    ]

    header = f"{'Config':<40s} {'ef':>4} {'vecs/s':>8} {'wall':>6} {'CPU avg%':>9} {'CPU max%':>9} {'MEM MB':>7}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    try:
        for label, cfg, search_ef in configs:
            col = "bench_cfg"
            fill(sync, col, **cfg)

            cpu_samples.clear()
            mem_samples.clear()
            monitoring = True
            t = threading.Thread(target=monitor, daemon=True)
            t.start()

            total, wall, rate = asyncio.run(run_ingestion(hp, gp, col, search_ef))

            monitoring = False
            t.join(timeout=3)

            avg_cpu = statistics.mean(cpu_samples) if cpu_samples else 0
            max_cpu = max(cpu_samples) if cpu_samples else 0
            avg_mem = statistics.mean(mem_samples) if mem_samples else 0

            print(
                f"{label:<40s} {search_ef:>4} {rate:>8.0f} {wall:>5.0f}s "
                f"{avg_cpu:>8.1f}% {max_cpu:>8.1f}% {avg_mem:>6.0f}",
                flush=True,
            )
    finally:
        sync.close()
        container.stop(timeout=5)
        print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
