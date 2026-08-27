"""Isolate segment-store expansion: time get_segment_contexts directly.

Runs against the ev_50k partition ingested by the main matrix (Postgres,
pgvector/pg16 container, shipped settings). Measures the store call alone:
seed row fetch + per-direction LATERAL window scan + row decode. No vector
search, no scoring, no rendering.

Matrix: seeds x (backward, forward) windows, sequential (30 reps/cell),
plus one c16 concurrency cell.
"""

import asyncio
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import (
    SQLAlchemySegmentStore,
    SQLAlchemySegmentStoreParams,
)

from common import PG_DSN as PG_URL  # noqa: E402
PARTITION = "ev_50k"
REPS = 30


async def main():
    engine = create_async_engine(PG_URL, pool_size=24, max_overflow=8)
    store = SQLAlchemySegmentStore(SQLAlchemySegmentStoreParams(engine=engine))
    await store.startup()
    part = await store.open_partition(PARTITION)
    assert part is not None, f"partition {PARTITION!r} not found -- run the matrix first"

    async with engine.connect() as conn:
        rows = await conn.execute(
            text(
                "SELECT uuid FROM segment_store_sg "
                "WHERE partition_key = :pk ORDER BY timestamp "
                "OFFSET 20000 LIMIT 5000"
            ),
            {"pk": PARTITION},
        )
        pool = [r[0] for r in rows]
    print(f"seed pool: {len(pool)} segment uuids from mid-timeline")

    async def one(seeds, back, fwd):
        s = time.perf_counter()
        ctxs = await part.get_segment_contexts(
            seeds, max_backward_segments=back, max_forward_segments=fwd
        )
        dt = time.perf_counter() - s
        return dt, sum(len(v) for v in ctxs.values())

    print(f"{'seeds':>6} {'back/fwd':>9} {'p50 ms':>8} {'mean ms':>8} {'rows':>6}")
    for n_seeds in (1, 10, 50):
        for back, fwd in ((0, 0), (2, 4), (8, 16)):
            xs, rows_out = [], 0
            for i in range(REPS):
                seeds = pool[i * n_seeds : (i + 1) * n_seeds]
                dt, nr = await one(seeds, back, fwd)
                xs.append(dt)
                rows_out = nr
            xs.sort()
            print(
                f"{n_seeds:>6} {f'{back}/{fwd}':>9} "
                f"{xs[REPS // 2] * 1000:8.2f} {statistics.fmean(xs) * 1000:8.2f} "
                f"{rows_out:>6}"
            )

    # concurrency: 16 in-flight expansions of the e6 full-query shape (50 seeds)
    async def timed(i):
        seeds = pool[i * 50 : (i + 1) * 50]
        s = time.perf_counter()
        await part.get_segment_contexts(
            seeds, max_backward_segments=2, max_forward_segments=4
        )
        return time.perf_counter() - s

    t0 = time.perf_counter()
    lat = await asyncio.gather(*(timed(i) for i in range(16)))
    wall = time.perf_counter() - t0
    ls = sorted(lat)
    print(
        f"c16, 50 seeds, 2/4: {16 / wall:.0f} expansions/s  "
        f"p50 {ls[8] * 1000:.2f} ms  max {ls[-1] * 1000:.2f} ms"
    )
    await engine.dispose()


asyncio.run(main())
