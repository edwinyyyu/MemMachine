"""
Vector Database Query Throughput Benchmark

Compares Qdrant configurations and alternative vector databases for
batch similarity search throughput.

Workload:
- 50,000 vectors, 1536 dimensions, cosine similarity
- Batch queries: 50 vectors per batch, top-20, score threshold 0.9
- Concurrency levels: 1, 10, 25, 50
- 100 batches per concurrency level = 5,000 query vectors per test

Usage:
    uv run python evaluation/extra_memory/bench_vector_db.py --qdrant-only
    uv run python evaluation/extra_memory/bench_vector_db.py --qdrant-only --skip-qdrant-volume
    uv run python evaluation/extra_memory/bench_vector_db.py  # all DBs
"""

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

DIMENSIONS = 1536
NUM_VECTORS = 50_000
BATCH_SIZE = 50
TOP_K = 20
SCORE_THRESHOLD = 0.9
CONCURRENCY_LEVELS = [1, 10, 25, 50]
NUM_BATCHES = 100
COLLECTION_NAME = "bench_vectors"


# -- Data generation --


def generate_vectors(count, dim=DIMENSIONS, seed=42):
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((count, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


def generate_payloads(count):
    return [{"timestamp": 1700000000 + i} for i in range(count)]


# -- Benchmark framework --


@dataclass
class BenchmarkResult:
    engine: str
    variant: str
    concurrency: int
    query_vectors_per_second: float
    total_time_seconds: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float


def _timed_call(fn, *args):
    t0 = time.perf_counter()
    fn(*args)
    return (time.perf_counter() - t0) * 1000


def run_benchmark(query_fn, query_vectors, concurrency):
    rng = np.random.default_rng(123)
    batches = [
        query_vectors[rng.integers(0, len(query_vectors), size=BATCH_SIZE)]
        for _ in range(NUM_BATCHES)
    ]
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_timed_call, query_fn, b) for b in batches]
        latencies = [f.result() for f in as_completed(futures)]
    total_time = time.perf_counter() - start
    return total_time, sorted(latencies)


def make_result(engine, variant, concurrency, total_time, latencies):
    total_vectors = NUM_BATCHES * BATCH_SIZE
    n = len(latencies)
    return BenchmarkResult(
        engine=engine,
        variant=variant,
        concurrency=concurrency,
        query_vectors_per_second=round(total_vectors / total_time, 1),
        total_time_seconds=round(total_time, 2),
        latency_p50_ms=round(latencies[int(n * 0.50)], 1),
        latency_p95_ms=round(latencies[int(n * 0.95)], 1),
        latency_p99_ms=round(latencies[int(n * 0.99)], 1),
    )


def bench_all_concurrencies(engine, variant, query_fn, query_vectors):
    results = []
    for c in CONCURRENCY_LEVELS:
        logger.info(f"  [{engine}/{variant}] concurrency={c}")
        t, lats = run_benchmark(query_fn, query_vectors, c)
        r = make_result(engine, variant, c, t, lats)
        logger.info(
            f"    -> {r.query_vectors_per_second} qv/s, "
            f"p50={r.latency_p50_ms}ms, p99={r.latency_p99_ms}ms"
        )
        results.append(r)
    return results


# -- Qdrant helpers --


def qdrant_setup_collection(client, data_vectors, data_payloads, label="default"):
    from qdrant_client import models

    logger.info(f"Setting up Qdrant collection ({label})...")
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=DIMENSIONS, distance=models.Distance.COSINE
        ),
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
        optimizers_config=models.OptimizersConfigDiff(
            default_segment_number=0, indexing_threshold=10_000
        ),
    )
    for i in range(0, len(data_vectors), 1000):
        end = min(i + 1000, len(data_vectors))
        points = [
            models.PointStruct(
                id=j, vector=data_vectors[j].tolist(), payload=data_payloads[j]
            )
            for j in range(i, end)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
    logger.info(f"Inserted {len(data_vectors)} vectors, waiting for indexing...")
    qdrant_wait_optimization(client)
    info = client.get_collection(COLLECTION_NAME)
    logger.info(f"Ready: {info.points_count} points, {info.segments_count} segments")


def qdrant_wait_optimization(client, timeout=300):
    from qdrant_client.models import OptimizersStatusOneOf

    start = time.time()
    while time.time() - start < timeout:
        info = client.get_collection(COLLECTION_NAME)
        if info.optimizer_status == OptimizersStatusOneOf.OK:
            return
        time.sleep(2)
    logger.warning("Optimization did not complete within timeout")


def qdrant_wait_ready(host, port, timeout=60):
    import urllib.request

    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(f"http://{host}:{port}/collections", timeout=5)
            return
        except Exception:
            time.sleep(1)
    raise TimeoutError(f"Qdrant not ready at {host}:{port}")


def make_qdrant_query_fn(client, search_params=None):
    from qdrant_client import models

    def query_fn(batch_vectors):
        requests = []
        for vec in batch_vectors:
            kwargs = dict(
                query=vec.tolist(),
                limit=TOP_K,
                score_threshold=SCORE_THRESHOLD,
                with_payload=True,
            )
            if search_params:
                kwargs["params"] = search_params
            requests.append(models.QueryRequest(**kwargs))
        client.query_batch_points(collection_name=COLLECTION_NAME, requests=requests)

    return query_fn


# -- Qdrant benchmarks --


def bench_qdrant_baseline(data_vectors, data_payloads, query_vectors):
    from qdrant_client import QdrantClient

    logger.info("=== Qdrant Baseline (HTTP) ===")
    client = QdrantClient("localhost", port=6333, timeout=120)
    qdrant_setup_collection(client, data_vectors, data_payloads, "baseline-http")
    return bench_all_concurrencies(
        "Qdrant", "baseline (HTTP)", make_qdrant_query_fn(client), query_vectors
    )


def bench_qdrant_grpc(data_vectors, data_payloads, query_vectors):
    from qdrant_client import QdrantClient

    logger.info("=== Qdrant gRPC ===")
    client = QdrantClient("localhost", port=6334, prefer_grpc=True, timeout=120)
    return bench_all_concurrencies(
        "Qdrant", "gRPC", make_qdrant_query_fn(client), query_vectors
    )


def bench_qdrant_quantization(data_vectors, data_payloads, query_vectors):
    from qdrant_client import QdrantClient, models

    logger.info("=== Qdrant gRPC + Scalar Quantization (INT8) ===")
    client = QdrantClient("localhost", port=6334, prefer_grpc=True, timeout=120)
    try:
        client.update_collection(
            collection_name=COLLECTION_NAME,
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8, quantile=0.99, always_ram=True
                )
            ),
        )
        logger.info("Quantization enabled, waiting for optimization...")
        qdrant_wait_optimization(client)
    except Exception as e:
        logger.warning(f"Quantization setup failed: {e}")

    params = models.SearchParams(
        quantization=models.QuantizationSearchParams(rescore=True, oversampling=1.5)
    )
    results = bench_all_concurrencies(
        "Qdrant",
        "gRPC + int8 quantization",
        make_qdrant_query_fn(client, params),
        query_vectors,
    )

    # Disable quantization for next test
    try:
        client.update_collection(
            collection_name=COLLECTION_NAME, quantization_config=models.Disabled()
        )
        qdrant_wait_optimization(client)
    except Exception:
        pass
    return results


def bench_qdrant_hnsw_tuned(data_vectors, data_payloads, query_vectors):
    from qdrant_client import QdrantClient, models

    logger.info("=== Qdrant gRPC + HNSW tuned (m=32, ef=256) ===")
    client = QdrantClient("localhost", port=6334, prefer_grpc=True, timeout=120)
    try:
        client.update_collection(
            collection_name=COLLECTION_NAME,
            hnsw_config=models.HnswConfigDiff(m=32, ef_construct=256),
        )
        logger.info("HNSW params updated, re-indexing...")
        qdrant_wait_optimization(client)
    except Exception as e:
        logger.warning(f"HNSW update failed: {e}")

    params = models.SearchParams(hnsw_ef=256)
    results = bench_all_concurrencies(
        "Qdrant",
        "gRPC + HNSW(m=32,ef=256)",
        make_qdrant_query_fn(client, params),
        query_vectors,
    )

    try:
        client.update_collection(
            collection_name=COLLECTION_NAME,
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
        )
        qdrant_wait_optimization(client)
    except Exception:
        pass
    return results


def bench_qdrant_segments(data_vectors, data_payloads, query_vectors):
    from qdrant_client import QdrantClient, models

    logger.info("=== Qdrant gRPC + 4 segments (more parallelism) ===")
    client = QdrantClient("localhost", port=6334, prefer_grpc=True, timeout=120)
    try:
        client.update_collection(
            collection_name=COLLECTION_NAME,
            optimizer_config=models.OptimizersConfigDiff(
                default_segment_number=4, max_optimization_threads=4
            ),
        )
        logger.info("Segment config updated, waiting...")
        qdrant_wait_optimization(client)
        info = client.get_collection(COLLECTION_NAME)
        logger.info(f"Now has {info.segments_count} segments")
    except Exception as e:
        logger.warning(f"Segment config failed: {e}")

    results = bench_all_concurrencies(
        "Qdrant", "gRPC + 4 segments", make_qdrant_query_fn(client), query_vectors
    )

    try:
        client.update_collection(
            collection_name=COLLECTION_NAME,
            optimizer_config=models.OptimizersConfigDiff(
                default_segment_number=0, max_optimization_threads=0
            ),
        )
        qdrant_wait_optimization(client)
    except Exception:
        pass
    return results


def bench_qdrant_volume(data_vectors, data_payloads, query_vectors):
    import docker as docker_lib
    from qdrant_client import QdrantClient

    logger.info("=== Qdrant gRPC + Docker-managed volume (vs bind mount) ===")
    dc = docker_lib.from_env()

    volume_name = "qdrant_bench_volume"
    try:
        dc.volumes.get(volume_name).remove(force=True)
    except Exception:
        pass
    dc.volumes.create(name=volume_name)

    # Remove old container if exists
    try:
        old = dc.containers.get("qdrant-bench-volume")
        old.remove(force=True)
    except Exception:
        pass

    container = dc.containers.run(
        "qdrant/qdrant:v1.17.0",
        detach=True,
        ports={"6333/tcp": 16333, "6334/tcp": 16334},
        volumes={volume_name: {"bind": "/qdrant/storage", "mode": "rw"}},
        name="qdrant-bench-volume",
    )

    try:
        qdrant_wait_ready("localhost", 16333)
        client = QdrantClient("localhost", port=16334, prefer_grpc=True, timeout=120)
        qdrant_setup_collection(client, data_vectors, data_payloads, "docker-volume")
        results = bench_all_concurrencies(
            "Qdrant",
            "gRPC + Docker volume",
            make_qdrant_query_fn(client),
            query_vectors,
        )
    finally:
        container.stop()
        container.remove(force=True)
        try:
            dc.volumes.get(volume_name).remove(force=True)
        except Exception:
            pass
    return results


# -- Milvus --


def bench_milvus(data_vectors, data_payloads, query_vectors):
    import docker as docker_lib
    from pymilvus import DataType, MilvusClient

    logger.info("=== Milvus Standalone ===")
    dc = docker_lib.from_env()

    # Cleanup
    for name in ["milvus-bench-standalone", "milvus-bench-etcd", "milvus-bench-minio"]:
        try:
            c = dc.containers.get(name)
            c.remove(force=True)
        except Exception:
            pass
    network_name = "milvus-bench-net"
    try:
        dc.networks.get(network_name).remove()
    except Exception:
        pass
    network = dc.networks.create(network_name, driver="bridge")

    try:
        logger.info("Starting etcd...")
        dc.containers.run(
            "quay.io/coreos/etcd:v3.5.18",
            detach=True,
            name="milvus-bench-etcd",
            network=network_name,
            environment={
                "ETCD_AUTO_COMPACTION_MODE": "revision",
                "ETCD_AUTO_COMPACTION_RETENTION": "1000",
                "ETCD_QUOTA_BACKEND_BYTES": "4294967296",
                "ETCD_SNAPSHOT_COUNT": "50000",
            },
            command=[
                "etcd",
                "-advertise-client-urls=http://127.0.0.1:2379",
                "-listen-client-urls=http://0.0.0.0:2379",
                "--data-dir=/etcd",
            ],
        )
        logger.info("Starting MinIO...")
        dc.containers.run(
            "minio/minio:latest",
            detach=True,
            name="milvus-bench-minio",
            network=network_name,
            environment={
                "MINIO_ACCESS_KEY": "minioadmin",
                "MINIO_SECRET_KEY": "minioadmin",
            },
            command=["server", "/minio_data", "--console-address", ":9001"],
        )
        logger.info("Starting Milvus...")
        dc.containers.run(
            "milvusdb/milvus:v2.5.6",
            detach=True,
            name="milvus-bench-standalone",
            network=network_name,
            ports={"19530/tcp": 19530, "9091/tcp": 9091},
            environment={
                "ETCD_ENDPOINTS": "milvus-bench-etcd:2379",
                "MINIO_ADDRESS": "milvus-bench-minio:9000",
            },
            command=["milvus", "run", "standalone"],
        )

        logger.info("Waiting for Milvus to be ready...")
        start = time.time()
        while time.time() - start < 120:
            try:
                mc = MilvusClient(uri="http://localhost:19530")
                mc.list_collections()
                break
            except Exception:
                time.sleep(2)
        else:
            raise TimeoutError("Milvus not ready after 120s")

        mc = MilvusClient(uri="http://localhost:19530")
        try:
            mc.drop_collection(COLLECTION_NAME)
        except Exception:
            pass

        logger.info("Creating Milvus collection...")
        schema = mc.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(
            field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=DIMENSIONS
        )
        schema.add_field(field_name="timestamp", datatype=DataType.INT64)

        index_params = mc.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 100},
        )

        mc.create_collection(
            collection_name=COLLECTION_NAME, schema=schema, index_params=index_params
        )

        logger.info("Inserting data...")
        for i in range(0, len(data_vectors), 2000):
            end = min(i + 2000, len(data_vectors))
            data = [
                {
                    "id": j,
                    "vector": data_vectors[j].tolist(),
                    "timestamp": data_payloads[j]["timestamp"],
                }
                for j in range(i, end)
            ]
            mc.insert(collection_name=COLLECTION_NAME, data=data)

        mc.load_collection(COLLECTION_NAME)
        logger.info(f"Milvus loaded {len(data_vectors)} vectors")
        time.sleep(3)

        def query_fn(batch_vectors):
            mc.search(
                collection_name=COLLECTION_NAME,
                data=batch_vectors.tolist(),
                limit=TOP_K,
                output_fields=["timestamp"],
                search_params={"metric_type": "COSINE", "params": {"ef": 100}},
                anns_field="vector",
            )

        results = bench_all_concurrencies(
            "Milvus", "standalone (HNSW)", query_fn, query_vectors
        )

    finally:
        for name in [
            "milvus-bench-standalone",
            "milvus-bench-etcd",
            "milvus-bench-minio",
        ]:
            try:
                c = dc.containers.get(name)
                c.stop(timeout=5)
                c.remove(force=True)
            except Exception:
                pass
        try:
            network.remove()
        except Exception:
            pass

    return results


# -- Weaviate --


def bench_weaviate(data_vectors, data_payloads, query_vectors):
    import docker as docker_lib
    import weaviate
    from weaviate.classes.config import Configure, DataType, Property, VectorDistances
    from weaviate.classes.query import MetadataQuery

    logger.info("=== Weaviate ===")
    dc = docker_lib.from_env()

    try:
        old = dc.containers.get("weaviate-bench")
        old.remove(force=True)
    except Exception:
        pass

    container = dc.containers.run(
        "semitechnologies/weaviate:1.28.4",
        detach=True,
        name="weaviate-bench",
        ports={"8080/tcp": 18080, "50051/tcp": 50051},
        environment={
            "QUERY_DEFAULTS_LIMIT": "25",
            "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED": "true",
            "PERSISTENCE_DATA_PATH": "/var/lib/weaviate",
            "DEFAULT_VECTORIZER_MODULE": "none",
            "CLUSTER_HOSTNAME": "node1",
            "GOMAXPROCS": "8",
        },
    )

    try:
        logger.info("Waiting for Weaviate...")
        import urllib.request

        start = time.time()
        while time.time() - start < 60:
            try:
                urllib.request.urlopen(
                    "http://localhost:18080/v1/.well-known/ready", timeout=5
                )
                break
            except Exception:
                time.sleep(1)
        else:
            raise TimeoutError("Weaviate not ready")

        client = weaviate.connect_to_custom(
            http_host="localhost",
            http_port=18080,
            http_secure=False,
            grpc_host="localhost",
            grpc_port=50051,
            grpc_secure=False,
        )

        try:
            client.collections.delete(COLLECTION_NAME)
        except Exception:
            pass

        logger.info("Creating Weaviate collection...")
        collection = client.collections.create(
            name=COLLECTION_NAME,
            vectorizer_config=Configure.Vectorizer.none(),
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE,
                ef_construction=100,
                max_connections=16,
                ef=100,
            ),
            properties=[Property(name="timestamp", data_type=DataType.INT)],
        )

        logger.info("Inserting data...")
        with collection.batch.fixed_size(batch_size=1000) as batch:
            for i in range(len(data_vectors)):
                batch.add_object(
                    properties={"timestamp": data_payloads[i]["timestamp"]},
                    vector=data_vectors[i].tolist(),
                )
        logger.info(f"Inserted {len(data_vectors)} vectors")
        time.sleep(3)

        def query_fn(batch_vectors):
            for vec in batch_vectors:
                collection.query.near_vector(
                    near_vector=vec.tolist(),
                    limit=TOP_K,
                    distance=1 - SCORE_THRESHOLD,
                    return_metadata=MetadataQuery(distance=True),
                    return_properties=["timestamp"],
                )

        results = bench_all_concurrencies(
            "Weaviate", "standalone (HNSW, sequential)", query_fn, query_vectors
        )
        client.close()

    finally:
        container.stop()
        container.remove(force=True)

    return results


# -- Redis --


def bench_redis(data_vectors, data_payloads, query_vectors):
    import docker as docker_lib
    import redis as redis_lib
    from redis.commands.search.field import NumericField, VectorField
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType
    from redis.commands.search.query import Query

    logger.info("=== Redis Stack (Vector Search) ===")
    dc = docker_lib.from_env()

    try:
        old = dc.containers.get("redis-bench")
        old.remove(force=True)
    except Exception:
        pass

    container = dc.containers.run(
        "redis/redis-stack:latest",
        detach=True,
        name="redis-bench",
        ports={"6379/tcp": 16379},
    )

    try:
        logger.info("Waiting for Redis...")
        start = time.time()
        while time.time() - start < 60:
            try:
                r = redis_lib.Redis(host="localhost", port=16379)
                r.ping()
                break
            except Exception:
                time.sleep(1)
        else:
            raise TimeoutError("Redis not ready")

        r = redis_lib.Redis(host="localhost", port=16379, decode_responses=False)
        try:
            r.ft("bench_idx").dropindex(delete_documents=True)
        except Exception:
            pass

        logger.info("Creating Redis vector index...")
        schema = (
            VectorField(
                "vector",
                "HNSW",
                {
                    "TYPE": "FLOAT32",
                    "DIM": DIMENSIONS,
                    "DISTANCE_METRIC": "COSINE",
                    "M": 16,
                    "EF_CONSTRUCTION": 100,
                    "EF_RUNTIME": 100,
                },
            ),
            NumericField("timestamp"),
        )
        definition = IndexDefinition(prefix=["vec:"], index_type=IndexType.HASH)
        r.ft("bench_idx").create_index(fields=schema, definition=definition)

        logger.info("Inserting data...")
        pipe = r.pipeline(transaction=False)
        for i in range(len(data_vectors)):
            pipe.hset(
                f"vec:{i}",
                mapping={
                    "vector": data_vectors[i].tobytes(),
                    "timestamp": str(data_payloads[i]["timestamp"]),
                },
            )
            if (i + 1) % 1000 == 0:
                pipe.execute()
                pipe = r.pipeline(transaction=False)
        pipe.execute()
        logger.info(f"Inserted {len(data_vectors)} vectors")
        time.sleep(5)

        def query_fn(batch_vectors):
            pipe = r.pipeline(transaction=False)
            for vec in batch_vectors:
                q = (
                    Query(f"(*)=>[KNN {TOP_K} @vector $query_vector AS score]")
                    .sort_by("score")
                    .return_fields("timestamp", "score")
                    .dialect(2)
                )
                pipe.ft("bench_idx").search(
                    q, query_params={"query_vector": vec.tobytes()}
                )
            pipe.execute()

        results = bench_all_concurrencies(
            "Redis", "Stack (HNSW, pipeline)", query_fn, query_vectors
        )

    finally:
        container.stop()
        container.remove(force=True)

    return results


# -- Reporting --


def print_results(all_results):
    print("\n" + "=" * 115)
    print("VECTOR DATABASE QUERY THROUGHPUT BENCHMARK RESULTS")
    print(
        f"Workload: {NUM_VECTORS} vectors, {DIMENSIONS}d, cosine, "
        f"top-{TOP_K}, score>={SCORE_THRESHOLD}, batch={BATCH_SIZE}"
    )
    print(f"Batches per concurrency level: {NUM_BATCHES}")
    print("=" * 115)
    header = (
        f"{'Engine':<12} {'Variant':<32} {'Conc':>5} {'QV/s':>10} "
        f"{'Total(s)':>10} {'p50(ms)':>10} {'p95(ms)':>10} {'p99(ms)':>10}"
    )
    print(header)
    print("-" * 115)
    prev = None
    for r in all_results:
        key = f"{r.engine}/{r.variant}"
        if key != prev:
            if prev is not None:
                print("-" * 115)
            prev = key
        print(
            f"{r.engine:<12} {r.variant:<32} {r.concurrency:>5} "
            f"{r.query_vectors_per_second:>10.1f} {r.total_time_seconds:>10.2f} "
            f"{r.latency_p50_ms:>10.1f} {r.latency_p95_ms:>10.1f} "
            f"{r.latency_p99_ms:>10.1f}"
        )
    print("=" * 115)
    print("\nSUMMARY (concurrency 50):")
    print("-" * 80)
    c50 = sorted(
        [r for r in all_results if r.concurrency == 50],
        key=lambda r: r.query_vectors_per_second,
        reverse=True,
    )
    for rank, r in enumerate(c50, 1):
        print(
            f"  #{rank}: {r.engine} ({r.variant}) "
            f"- {r.query_vectors_per_second:.0f} qv/s, p99={r.latency_p99_ms:.0f}ms"
        )


def save_results(all_results, path):
    data = [
        {
            "engine": r.engine,
            "variant": r.variant,
            "concurrency": r.concurrency,
            "query_vectors_per_second": r.query_vectors_per_second,
            "total_time_seconds": r.total_time_seconds,
            "latency_p50_ms": r.latency_p50_ms,
            "latency_p95_ms": r.latency_p95_ms,
            "latency_p99_ms": r.latency_p99_ms,
        }
        for r in all_results
    ]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Results saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Vector DB Throughput Benchmark")
    parser.add_argument(
        "--qdrant-only", action="store_true", help="Only run Qdrant benchmarks"
    )
    parser.add_argument(
        "--alternatives-only",
        action="store_true",
        help="Only run Milvus/Weaviate/Redis",
    )
    parser.add_argument("--skip-milvus", action="store_true")
    parser.add_argument("--skip-weaviate", action="store_true")
    parser.add_argument("--skip-redis", action="store_true")
    parser.add_argument("--skip-qdrant-volume", action="store_true")
    parser.add_argument(
        "--output", default="evaluation/extra_memory/bench_vector_db_results.json"
    )
    args = parser.parse_args()

    logger.info("Generating test data...")
    data_vectors = generate_vectors(NUM_VECTORS)
    data_payloads = generate_payloads(NUM_VECTORS)
    query_vectors = generate_vectors(5000, seed=99)
    logger.info(
        f"Generated {NUM_VECTORS} data vectors, {len(query_vectors)} query vectors"
    )

    results = []

    # Qdrant benchmarks
    if not args.alternatives_only:
        for bench_fn in [
            bench_qdrant_baseline,
            bench_qdrant_grpc,
            bench_qdrant_quantization,
            bench_qdrant_hnsw_tuned,
            bench_qdrant_segments,
        ]:
            try:
                results.extend(bench_fn(data_vectors, data_payloads, query_vectors))
            except Exception as e:
                logger.error(f"{bench_fn.__name__} failed: {e}", exc_info=True)

        if not args.skip_qdrant_volume:
            try:
                results.extend(
                    bench_qdrant_volume(data_vectors, data_payloads, query_vectors)
                )
            except Exception as e:
                logger.error(f"bench_qdrant_volume failed: {e}", exc_info=True)

    # Alternative DB benchmarks
    if not args.qdrant_only:
        if not args.skip_milvus:
            try:
                results.extend(bench_milvus(data_vectors, data_payloads, query_vectors))
            except Exception as e:
                logger.error(f"bench_milvus failed: {e}", exc_info=True)
        if not args.skip_weaviate:
            try:
                results.extend(
                    bench_weaviate(data_vectors, data_payloads, query_vectors)
                )
            except Exception as e:
                logger.error(f"bench_weaviate failed: {e}", exc_info=True)
        if not args.skip_redis:
            try:
                results.extend(bench_redis(data_vectors, data_payloads, query_vectors))
            except Exception as e:
                logger.error(f"bench_redis failed: {e}", exc_info=True)

    if results:
        print_results(results)
        save_results(results, args.output)
    else:
        logger.warning("No benchmark results collected!")


if __name__ == "__main__":
    main()
