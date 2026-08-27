"""Benchmark EventMemory (QdrantVectorStore + SQLAlchemySegmentStore/Postgres).

Targets the memmachine-qdrant (qdrant/qdrant:v1.19.0) and memmachine-postgres
(pgvector/pg16) containers from the repo compose file, shipped settings.

Knob parity with bench_declarative.py:
  declarative search_scored(max_num_episodes=K) overfetches min(5K, 200)
  derivative vectors; we pass vector_search_limit=5K to EventMemory.query so
  the ANN stage does the same amount of work. expand_context E maps 1:1
  (1 message event = 1 segment with the passthrough segmenter).
  Reranker: None on event (embedding scores reused) vs IdentityReranker on
  declarative (no-op) -- neither does real scoring work.
"""

import argparse
import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    FakeEmbedder,
    category_of,
    make_queries,
    make_texts,
    run_workers,
    save_result,
    split_chunks,
    summarize,
    ts_of,
)
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import create_async_engine

from memmachine_server.common.data_types import SimilarityMetric
from memmachine_server.common.filter.filter_parser import Comparison
from memmachine_server.common.vector_store.data_types import (
    VectorStoreCollectionConfig,
)
from memmachine_server.common.vector_store.qdrant_vector_store import (
    QdrantVectorStore,
    QdrantVectorStoreParams,
)
from memmachine_server.episodic_memory.event_memory.data_types import (
    Event,
    ProducerContext,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.deriver.text_deriver import (
    WholeTextDeriver,
)
from memmachine_server.episodic_memory.event_memory.event_memory import (
    EventMemory,
    EventMemoryParams,
)
from memmachine_server.episodic_memory.event_memory.segment_store.data_types import (
    SegmentStorePartitionConfig,
)
from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import (
    SQLAlchemySegmentStore,
    SQLAlchemySegmentStoreParams,
)
from memmachine_server.episodic_memory.event_memory.segmenter.passthrough_segmenter import (
    PassthroughSegmenter,
)

from common import PG_DSN as PG_URL  # noqa: E402  (env-overridable)
NAMESPACE = "bench"

SCHEMA = {
    "_segment_uuid": str,
    "_timestamp": datetime,
    "category": str,
}


async def build_memory(
    vector_store: QdrantVectorStore,
    segment_store: SQLAlchemySegmentStore,
    session_key: str,
) -> EventMemory:
    collection = await vector_store.open_or_create_collection(
        namespace=NAMESPACE,
        name=session_key,
        config=VectorStoreCollectionConfig(
            vector_dimensions=FakeEmbedder().dimensions,
            similarity_metric=SimilarityMetric.COSINE,
            indexed_properties_schema=SCHEMA,
        ),
    )
    partition = await segment_store.open_or_create_partition(
        session_key, SegmentStorePartitionConfig()
    )
    return EventMemory(
        EventMemoryParams(
            segment_store_partition=partition,
            vector_store_collection=collection,
            segmenter=PassthroughSegmenter(),
            deriver=WholeTextDeriver(),
            embedder=FakeEmbedder(),
            reranker=None,
        )
    )


def make_events(n: int, seed: int) -> list[Event]:
    texts = make_texts(n, seed)
    return [
        Event(
            uuid=uuid4(),
            timestamp=ts_of(i),
            context=ProducerContext(producer="alice" if i % 2 == 0 else "bob"),
            blocks=[TextBlock(text=texts[i])],
            properties={"category": category_of(i)},
        )
        for i in range(n)
    ]


async def mode_ingest(args, vector_store, segment_store):
    events = make_events(args.n, seed=args.seed)
    if args.sessions_per_worker:
        per = split_chunks(events, args.concurrency)
        thunk_lists = []
        for w, evs in enumerate(per):
            mem = await build_memory(
                vector_store, segment_store, f"{args.session}_w{w}"
            )
            batches = [
                evs[i : i + args.batch] for i in range(0, len(evs), args.batch)
            ]
            thunk_lists.append(
                [lambda b=b, m=mem: m.encode_events(b) for b in batches]
            )
    else:
        mem = await build_memory(vector_store, segment_store, args.session)
        batches = [
            events[i : i + args.batch] for i in range(0, len(events), args.batch)
        ]
        thunk_lists = split_chunks(
            [lambda b=b: mem.encode_events(b) for b in batches], args.concurrency
        )

    lat, off, wall = await run_workers(thunk_lists)
    metrics = summarize(lat, wall, sum(len(t) for t in thunk_lists))
    metrics["episodes_per_s"] = round(args.n / wall, 2)
    save_result(
        args.label,
        vars(args) | {"backend": "event", "qdrant": "v1.19.0", "pg": "pgvector-pg16"},
        metrics,
        raw={"latencies": lat, "offsets": off},
    )


async def mode_query(args, vector_store, segment_store):
    mem = await build_memory(vector_store, segment_store, args.session)
    queries = make_queries(args.queries + args.warmup, seed=args.seed + 1)
    pfilter = (
        Comparison(field="m.category", op="=", value="cat_3")
        if args.filtered
        else None
    )

    async def one(q):
        await mem.query(
            q,
            vector_search_limit=args.vsl,
            expand_context=args.expand,
            property_filter=pfilter,
        )

    for q in queries[: args.warmup]:
        await one(q)

    thunk_lists = split_chunks(
        [lambda q=q: one(q) for q in queries[args.warmup :]], args.concurrency
    )
    lat, off, wall = await run_workers(thunk_lists)
    metrics = summarize(lat, wall, args.queries)
    save_result(
        args.label,
        vars(args) | {"backend": "event"},
        metrics,
        raw={"latencies": lat},
    )


async def mode_mixed(args, vector_store, segment_store):
    mem = await build_memory(vector_store, segment_store, args.session)
    queries = make_queries(10_000, seed=args.seed + 2)
    events = make_events(20_000, seed=args.seed + 3)
    stop = time.monotonic() + args.duration
    rlat: list[float] = []
    wlat: list[float] = []

    async def reader(qs):
        i = 0
        while time.monotonic() < stop:
            s = time.perf_counter()
            await mem.query(
                qs[i % len(qs)], vector_search_limit=args.vsl,
                expand_context=args.expand,
            )
            rlat.append(time.perf_counter() - s)
            i += 1

    async def writer(evs):
        i = 0
        while time.monotonic() < stop and i < len(evs):
            s = time.perf_counter()
            await mem.encode_events([evs[i]])
            wlat.append(time.perf_counter() - s)
            i += 1

    for q in queries[:5]:
        await mem.query(q, vector_search_limit=args.vsl)

    t0 = time.perf_counter()
    await asyncio.gather(
        *(reader(queries[i::args.readers]) for i in range(args.readers)),
        *(writer(events[i::args.writers]) for i in range(args.writers)),
    )
    wall = time.perf_counter() - t0
    metrics = {
        "read": summarize(rlat, wall, len(rlat)),
        "write": summarize(wlat, wall, len(wlat)) if wlat else None,
    }
    save_result(
        args.label,
        vars(args) | {"backend": "event"},
        metrics,
        raw={"read_latencies": rlat, "write_latencies": wlat},
    )


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["ingest", "query", "mixed"], required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--session", required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--sessions-per-worker", action="store_true")
    p.add_argument("--queries", type=int, default=200)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--vsl", type=int, default=50)
    p.add_argument("--expand", type=int, default=0)
    p.add_argument("--filtered", action="store_true")
    p.add_argument("--prefer-grpc", action="store_true")
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--readers", type=int, default=16)
    p.add_argument("--writers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    client = AsyncQdrantClient(
        host="localhost", port=6333, grpc_port=6334,
        prefer_grpc=args.prefer_grpc, https=False,
    )
    vector_store = QdrantVectorStore(QdrantVectorStoreParams(client=client))
    engine = create_async_engine(PG_URL, pool_size=48, max_overflow=16)
    segment_store = SQLAlchemySegmentStore(
        SQLAlchemySegmentStoreParams(engine=engine)
    )
    await segment_store.startup()
    try:
        if args.mode == "ingest":
            await mode_ingest(args, vector_store, segment_store)
        elif args.mode == "query":
            await mode_query(args, vector_store, segment_store)
        else:
            await mode_mixed(args, vector_store, segment_store)
    finally:
        await client.close()
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
