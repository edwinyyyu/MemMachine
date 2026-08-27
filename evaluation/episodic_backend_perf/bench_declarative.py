"""Benchmark DeclarativeMemory (Neo4jVectorGraphStore) against live Neo4j.

Targets the memmachine-neo4j container started from the repo compose file
(neo4j:5.23-community, heap 512m/1G, bolt pool 2000 -- shipped settings).

Modes:
  ingest : C workers append batches of episodes to one session (or one
           session per worker with --sessions-per-worker).
  query  : C workers run queries against an existing session.
  mixed  : writers append while readers query, fixed duration.
"""

import argparse
import asyncio
import sys
import time
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
from neo4j import AsyncGraphDatabase

from memmachine_server.common.filter.filter_parser import Comparison
from memmachine_server.common.reranker.identity_reranker import IdentityReranker
from memmachine_server.common.vector_graph_store.neo4j_vector_graph_store import (
    Neo4jVectorGraphStore,
    Neo4jVectorGraphStoreParams,
)
from memmachine_server.episodic_memory.declarative_memory.data_types import (
    ContentType,
    Episode,
)
from memmachine_server.episodic_memory.declarative_memory.declarative_memory import (
    DeclarativeMemory,
    DeclarativeMemoryParams,
)

from common import NEO4J_AUTH, NEO4J_URI  # noqa: E402  (env-overridable)


def build_memory(store: Neo4jVectorGraphStore, session_id: str) -> DeclarativeMemory:
    return DeclarativeMemory(
        DeclarativeMemoryParams(
            session_id=session_id,
            vector_graph_store=store,
            embedder=FakeEmbedder(),
            reranker=IdentityReranker(),
        )
    )


def build_store(driver, threshold: int) -> Neo4jVectorGraphStore:
    return Neo4jVectorGraphStore(
        Neo4jVectorGraphStoreParams(
            driver=driver,
            range_index_hierarchies=[["uid"], ["timestamp", "uid"]],
            range_index_creation_threshold=threshold,
            vector_index_creation_threshold=threshold,
        )
    )


def make_episodes(n: int, seed: int) -> list[Episode]:
    texts = make_texts(n, seed)
    return [
        Episode(
            uid=str(uuid4()),
            timestamp=ts_of(i),
            source="alice" if i % 2 == 0 else "bob",
            content_type=ContentType.MESSAGE,
            content=texts[i],
            filterable_properties={"category": category_of(i)},
            user_metadata=None,
        )
        for i in range(n)
    ]


async def wait_indexes_online(driver, timeout_s: float = 300.0):
    """Poll until no index is in a non-ONLINE state."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        async with driver.session() as s:
            res = await s.run("SHOW INDEXES YIELD name, state RETURN name, state")
            records = await res.data()
        pending = [r for r in records if r["state"] != "ONLINE"]
        if not pending:
            return [r["name"] for r in records]
        await asyncio.sleep(1.0)
    raise TimeoutError(f"indexes not online: {pending}")


async def drain_background_tasks(store: Neo4jVectorGraphStore):
    while store._background_tasks:
        await asyncio.gather(*list(store._background_tasks), return_exceptions=True)


async def mode_ingest(args, driver):
    store = build_store(driver, args.threshold)
    episodes = make_episodes(args.n, seed=args.seed)
    thunks = []
    if args.sessions_per_worker:
        # one session per worker: server-realistic multi-session ingest
        per = split_chunks(episodes, args.concurrency)
        thunk_lists = []
        for w, eps in enumerate(per):
            mem = build_memory(store, f"{args.session}_w{w}")
            batches = [
                eps[i : i + args.batch] for i in range(0, len(eps), args.batch)
            ]
            thunk_lists.append(
                [lambda b=b, m=mem: m.add_episodes(b) for b in batches]
            )
    else:
        mem = build_memory(store, args.session)
        batches = [
            episodes[i : i + args.batch]
            for i in range(0, len(episodes), args.batch)
        ]
        thunks = [lambda b=b: mem.add_episodes(b) for b in batches]
        thunk_lists = split_chunks(thunks, args.concurrency)

    lat, off, wall = await run_workers(thunk_lists)
    await drain_background_tasks(store)
    metrics = summarize(lat, wall, sum(len(t) for t in thunk_lists))
    metrics["episodes_per_s"] = round(args.n / wall, 2)
    save_result(
        args.label,
        vars(args) | {"backend": "declarative", "neo4j": "5.23-community heap1G"},
        metrics,
        raw={"latencies": lat, "offsets": off},
    )


async def mode_query(args, driver):
    store = build_store(driver, args.threshold)
    mem = build_memory(store, args.session)
    queries = make_queries(args.queries + args.warmup, seed=args.seed + 1)
    pfilter = (
        Comparison(field="category", op="=", value="cat_3")
        if args.filtered
        else None
    )

    async def one(q):
        await mem.search_scored(
            q,
            max_num_episodes=args.k,
            expand_context=args.expand,
            property_filter=pfilter,
        )

    if args.wait_indexes:
        names = await wait_indexes_online(driver)
        print(f"[indexes online] {len(names)}")
    for q in queries[: args.warmup]:
        await one(q)

    thunk_lists = split_chunks(
        [lambda q=q: one(q) for q in queries[args.warmup :]], args.concurrency
    )
    lat, off, wall = await run_workers(thunk_lists)
    metrics = summarize(lat, wall, args.queries)
    save_result(
        args.label,
        vars(args) | {"backend": "declarative"},
        metrics,
        raw={"latencies": lat},
    )


async def mode_mixed(args, driver):
    store = build_store(driver, args.threshold)
    mem = build_memory(store, args.session)
    queries = make_queries(10_000, seed=args.seed + 2)
    episodes = make_episodes(20_000, seed=args.seed + 3)
    stop = time.monotonic() + args.duration
    rlat: list[float] = []
    wlat: list[float] = []

    async def reader(qs):
        i = 0
        while time.monotonic() < stop:
            s = time.perf_counter()
            await mem.search_scored(
                qs[i % len(qs)], max_num_episodes=args.k, expand_context=args.expand
            )
            rlat.append(time.perf_counter() - s)
            i += 1

    async def writer(eps):
        i = 0
        while time.monotonic() < stop and i < len(eps):
            s = time.perf_counter()
            await mem.add_episodes([eps[i]])
            wlat.append(time.perf_counter() - s)
            i += 1

    for q in queries[:5]:
        await mem.search_scored(q, max_num_episodes=args.k)

    t0 = time.perf_counter()
    await asyncio.gather(
        *(reader(queries[i::args.readers]) for i in range(args.readers)),
        *(writer(episodes[i::args.writers]) for i in range(args.writers)),
    )
    wall = time.perf_counter() - t0
    metrics = {
        "read": summarize(rlat, wall, len(rlat)),
        "write": summarize(wlat, wall, len(wlat)) if wlat else None,
    }
    save_result(
        args.label,
        vars(args) | {"backend": "declarative"},
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
    p.add_argument("--threshold", type=int, default=10_000)
    p.add_argument("--queries", type=int, default=200)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--expand", type=int, default=0)
    p.add_argument("--filtered", action="store_true")
    p.add_argument("--wait-indexes", action="store_true")
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--readers", type=int, default=16)
    p.add_argument("--writers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pool", type=int, default=None,
                   help="max_connection_pool_size (driver default 100)")
    args = p.parse_args()

    kwargs = {}
    if args.pool:
        kwargs["max_connection_pool_size"] = args.pool
    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH, **kwargs)
    try:
        if args.mode == "ingest":
            await mode_ingest(args, driver)
        elif args.mode == "query":
            await mode_query(args, driver)
        else:
            await mode_mixed(args, driver)
    finally:
        await driver.close()


if __name__ == "__main__":
    asyncio.run(main())
