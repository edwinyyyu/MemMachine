"""RPS sweep of the event-stack operations, per operation type.

Modes (--ops, comma-separated; default all):
  search   : EventMemory.query(vsl=L, expand_context=0), L in --limits
  expand   : one raw get_segment_contexts call, 1 seed, window E split
             back E//3 / fwd rest, E in --windows
  combined : EventMemory.query(vsl=L, expand_context=6) -- NOTE: expands
             ALL ~L deduped seeds (no top-K cap exists in the API)
  paired   : one request = raw collection.query(vsl=50) + raw
             get_segment_contexts over the TOP-10 deduped seeds (2/4).
             Matches the fixed-Neo4j "full" probe shape; NOT the agentic
             pattern, and NOT the expand_context parameter.

Requires a session previously ingested by bench_event.py (default
--session ev_50k). One client process; c1 and c16 per cell.
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_event
from common import PG_DSN, make_queries, run_workers, split_chunks
from qdrant_client import AsyncQdrantClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from memmachine_server.common.vector_store.qdrant_vector_store import (
    QdrantVectorStore,
    QdrantVectorStoreParams,
)
from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import (
    SQLAlchemySegmentStore,
    SQLAlchemySegmentStoreParams,
)


async def cell(fn, c, n):
    thunks = [lambda i=i: fn(i) for i in range(n)]
    lat, _, wall = await run_workers(split_chunks(thunks, c))
    ls = sorted(lat)
    return n / wall, ls[n // 2] * 1000


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session", default="ev_50k")
    p.add_argument("--ops", default="search,expand,combined,paired")
    p.add_argument("--limits", default="5,10,20,50,100")
    p.add_argument("--windows", default="6,12,24,48")
    p.add_argument("--queries", type=int, default=300)
    args = p.parse_args()
    ops = args.ops.split(",")

    client = AsyncQdrantClient(host="localhost", port=6333, grpc_port=6334,
                               prefer_grpc=False)
    vs = QdrantVectorStore(QdrantVectorStoreParams(client=client))
    engine = create_async_engine(PG_DSN, pool_size=40)
    store = SQLAlchemySegmentStore(SQLAlchemySegmentStoreParams(engine=engine))
    await store.startup()
    mem = await bench_event.build_memory(vs, store, args.session)
    part = await store.open_partition(args.session)
    coll = await vs.open_collection(namespace=bench_event.NAMESPACE,
                                    name=args.session)
    qs = make_queries(700, seed=2222)
    from common import FakeEmbedder
    vecs = [FakeEmbedder()._embed_one(q) for q in qs]
    async with engine.connect() as conn:
        rows = await conn.execute(
            text("SELECT uuid FROM segment_store_sg WHERE partition_key = :pk "
                 "ORDER BY timestamp OFFSET 15000 LIMIT 20000"),
            {"pk": args.session})
        pool = [r[0] for r in rows]

    async def run(name, fn):
        await fn(0)
        r1, p1 = await cell(fn, 1, min(200, args.queries))
        r16, p16 = await cell(fn, 16, args.queries)
        print(f"{name:<28} c1 {r1:7.1f}/s {p1:7.2f}ms   "
              f"c16 {r16:7.1f}/s {p16:7.2f}ms")

    if "search" in ops:
        print("SEARCH ONLY: EventMemory.query(vsl=L, expand_context=0)")
        for L in map(int, args.limits.split(",")):
            async def one(i, L=L):
                await mem.query(qs[i % 600], vector_search_limit=L,
                                expand_context=0)
            await run(f"  vsl={L}", one)

    if "expand" in ops:
        print("EXPAND ONLY: raw get_segment_contexts, 1 seed")
        for E in map(int, args.windows.split(",")):
            b, f = E // 3, E - E // 3
            async def one(i, b=b, f=f):
                s = (i * 11) % (len(pool) - 1)
                await part.get_segment_contexts(
                    [pool[s]], max_backward_segments=b, max_forward_segments=f)
            await run(f"  E={E} ({b}/{f})", one)

    if "combined" in ops:
        print("COMBINED: EventMemory.query(vsl=L, expand_context=6) "
              "-- expands ALL ~L seeds")
        for L in map(int, args.limits.split(",")):
            async def one(i, L=L):
                await mem.query(qs[i % 600], vector_search_limit=L,
                                expand_context=6)
            await run(f"  vsl={L} E=6", one)

    if "paired" in ops:
        print("PAIRED (raw calls, matches fixed-Neo4j full shape): "
              "search vsl=50 + expand top-10 (2/4)")

        async def one(i):
            res = await coll.query(query_vectors=[vecs[i % 600]], limit=50,
                                   return_vector=False, return_properties=True)
            seen, seeds = set(), []
            for m in res[0].matches:
                su = m.record.properties["_segment_uuid"]
                if su not in seen:
                    seen.add(su)
                    seeds.append(UUID(su))
                if len(seeds) == 10:
                    break
            await part.get_segment_contexts(
                seeds, max_backward_segments=2, max_forward_segments=4)

        await run("  paired(50 -> top10, 2/4)", one)

    await client.close()
    await engine.dispose()


asyncio.run(main())
