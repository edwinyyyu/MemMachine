"""Benchmark: separate property table vs inline JSONB properties.

Spins up a Postgres container via testcontainers. Optionally also benchmarks SQLite.

Usage:
    uv run python bench_properties.py
    uv run python bench_properties.py --include-sqlite
    uv run python bench_properties.py --batch-size 500 --num-batches 100
"""

import argparse
import asyncio
import random
import time
from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Uuid,
    func,
    insert,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, MappedColumn, mapped_column

_JSON_AUTO = JSON().with_variant(JSONB, "postgresql")


# ── Schema A: separate property table ────────────────────────────────────────


class BaseA(DeclarativeBase):
    pass


class SegmentA(BaseA):
    __tablename__ = "bench_a_segments"

    uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    partition_key: MappedColumn[str] = mapped_column(String, nullable=False)
    episode_uuid: MappedColumn[UUID] = mapped_column(Uuid, nullable=False)
    timestamp: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    block: MappedColumn[dict] = mapped_column(_JSON_AUTO, nullable=False)

    __table_args__ = (Index("bench_a_segments__pk_ts", "partition_key", "timestamp"),)


class PropertyA(BaseA):
    __tablename__ = "bench_a_properties"

    segment_uuid: MappedColumn[UUID] = mapped_column(
        Uuid,
        ForeignKey("bench_a_segments.uuid", ondelete="CASCADE"),
        primary_key=True,
    )
    key: MappedColumn[str] = mapped_column(String, primary_key=True)
    value_bool: MappedColumn[bool] = mapped_column(Boolean, nullable=True)
    value_int: MappedColumn[int] = mapped_column(Integer, nullable=True)
    value_float: MappedColumn[float] = mapped_column(Float, nullable=True)
    value_str: MappedColumn[str] = mapped_column(String, nullable=True)


class DerivativeA(BaseA):
    __tablename__ = "bench_a_derivatives"

    uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    segment_uuid: MappedColumn[UUID] = mapped_column(
        Uuid,
        ForeignKey("bench_a_segments.uuid", ondelete="CASCADE"),
        nullable=False,
    )

    __table_args__ = (Index("bench_a_derivatives__su", "segment_uuid"),)


# ── Schema B: inline JSONB properties ────────────────────────────────────────


class BaseB(DeclarativeBase):
    pass


class SegmentB(BaseB):
    __tablename__ = "bench_b_segments"

    uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    partition_key: MappedColumn[str] = mapped_column(String, nullable=False)
    episode_uuid: MappedColumn[UUID] = mapped_column(Uuid, nullable=False)
    timestamp: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    block: MappedColumn[dict] = mapped_column(_JSON_AUTO, nullable=False)
    properties: MappedColumn[dict] = mapped_column(_JSON_AUTO, nullable=False)

    __table_args__ = (Index("bench_b_segments__pk_ts", "partition_key", "timestamp"),)


class DerivativeB(BaseB):
    __tablename__ = "bench_b_derivatives"

    uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    segment_uuid: MappedColumn[UUID] = mapped_column(
        Uuid,
        ForeignKey("bench_b_segments.uuid", ondelete="CASCADE"),
        nullable=False,
    )

    __table_args__ = (Index("bench_b_derivatives__su", "segment_uuid"),)


# ── Schema C: type-tagged JSONB properties ───────────────────────────────────


class BaseC(DeclarativeBase):
    pass


class SegmentC(BaseC):
    __tablename__ = "bench_c_segments"

    uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    partition_key: MappedColumn[str] = mapped_column(String, nullable=False)
    episode_uuid: MappedColumn[UUID] = mapped_column(Uuid, nullable=False)
    timestamp: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    block: MappedColumn[dict] = mapped_column(_JSON_AUTO, nullable=False)
    properties: MappedColumn[dict] = mapped_column(_JSON_AUTO, nullable=False)

    __table_args__ = (Index("bench_c_segments__pk_ts", "partition_key", "timestamp"),)


class DerivativeC(BaseC):
    __tablename__ = "bench_c_derivatives"

    uuid: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    segment_uuid: MappedColumn[UUID] = mapped_column(
        Uuid,
        ForeignKey("bench_c_segments.uuid", ondelete="CASCADE"),
        nullable=False,
    )

    __table_args__ = (Index("bench_c_derivatives__su", "segment_uuid"),)


# ── Data generation ──────────────────────────────────────────────────────────


def make_batch(batch_size: int, num_derivatives_per_segment: int):
    """Generate a batch of test data."""
    now = datetime.now(timezone.utc)
    episode_uuid = uuid4()
    segments = []
    properties = []
    derivatives_a = []
    derivatives_b = []
    derivatives_c = []

    for i in range(batch_size):
        seg_uuid = uuid4()
        created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
        props = {
            "session_id": f"session_{i % 50}",
            "has_answer": i % 3 == 0,
            "turn_id": i,
            "created_at": created_at,
        }

        segments.append(
            {
                "uuid": seg_uuid,
                "partition_key": "bench",
                "episode_uuid": episode_uuid,
                "timestamp": now,
                "block": {"type": "text", "text": f"segment content {i}"},
            }
        )

        for key, value in props.items():
            row: dict = {"segment_uuid": seg_uuid, "key": key}
            if isinstance(value, bool):
                row["value_bool"] = value
            elif isinstance(value, int):
                row["value_int"] = value
            elif isinstance(value, str):
                row["value_str"] = value
            properties.append(row)

        for _ in range(num_derivatives_per_segment):
            d_uuid = uuid4()
            derivatives_a.append({"uuid": d_uuid, "segment_uuid": seg_uuid})
            derivatives_b.append({"uuid": d_uuid, "segment_uuid": seg_uuid})
            derivatives_c.append({"uuid": d_uuid, "segment_uuid": seg_uuid})

    # Schema B: plain JSONB (no type tags)
    segments_b = [
        {
            **seg,
            "properties": {
                "session_id": f"session_{i % 50}",
                "has_answer": i % 3 == 0,
                "turn_id": i,
                "created_at": datetime(2025, 1, 1, tzinfo=timezone.utc).isoformat(),
            },
        }
        for i, seg in enumerate(segments)
    ]

    # Schema C: type-tagged JSONB  {"v": value, "t": type_name}
    # Matches SQLAlchemySegmentLinkerPartition._encode_properties
    _V = "v"
    _T = "t"
    segments_c = [
        {
            **seg,
            "properties": {
                "session_id": {_V: f"session_{i % 50}", _T: "str"},
                "has_answer": {_V: i % 3 == 0, _T: "bool"},
                "turn_id": {_V: i, _T: "int"},
                "created_at": {
                    _V: datetime(2025, 1, 1, tzinfo=timezone.utc).isoformat(),
                    _T: "datetime",
                },
            },
        }
        for i, seg in enumerate(segments)
    ]

    return (
        segments,
        segments_b,
        segments_c,
        properties,
        derivatives_a,
        derivatives_b,
        derivatives_c,
    )


# ── Benchmark functions ──────────────────────────────────────────────────────


async def bench_write_a(
    session_maker: async_sessionmaker[AsyncSession],
    segments: list[dict],
    properties: list[dict],
    derivatives: list[dict],
) -> float:
    t0 = time.perf_counter()
    async with session_maker() as session, session.begin():
        await session.execute(insert(SegmentA), segments)
        await session.execute(insert(PropertyA), properties)
        await session.execute(insert(DerivativeA), derivatives)
    return time.perf_counter() - t0


async def bench_write_b(
    session_maker: async_sessionmaker[AsyncSession],
    segments_b: list[dict],
    derivatives: list[dict],
) -> float:
    t0 = time.perf_counter()
    async with session_maker() as session, session.begin():
        await session.execute(insert(SegmentB), segments_b)
        await session.execute(insert(DerivativeB), derivatives)
    return time.perf_counter() - t0


async def bench_read_a(
    session_maker: async_sessionmaker[AsyncSession],
    derivative_uuids: list[UUID],
) -> float:
    t0 = time.perf_counter()
    async with session_maker() as session:
        result = await session.execute(
            select(DerivativeA.uuid, SegmentA)
            .join(DerivativeA, DerivativeA.segment_uuid == SegmentA.uuid)
            .where(DerivativeA.uuid.in_(derivative_uuids))
            .order_by(SegmentA.timestamp)
        )
        rows = result.all()
        segment_uuids = [seg.uuid for _, seg in rows]

        if segment_uuids:
            prop_result = await session.execute(
                select(PropertyA).where(PropertyA.segment_uuid.in_(segment_uuids))
            )
            _prop_rows = prop_result.scalars().all()

    return time.perf_counter() - t0


async def bench_read_b(
    session_maker: async_sessionmaker[AsyncSession],
    derivative_uuids: list[UUID],
) -> float:
    t0 = time.perf_counter()
    async with session_maker() as session:
        result = await session.execute(
            select(DerivativeB.uuid, SegmentB)
            .join(DerivativeB, DerivativeB.segment_uuid == SegmentB.uuid)
            .where(DerivativeB.uuid.in_(derivative_uuids))
            .order_by(SegmentB.timestamp)
        )
        _rows = result.all()

    return time.perf_counter() - t0


async def bench_read_filtered_a(
    session_maker: async_sessionmaker[AsyncSession],
    derivative_uuids: list[UUID],
) -> float:
    t0 = time.perf_counter()
    async with session_maker() as session:
        prop_filter = (
            select(1)
            .select_from(PropertyA)
            .where(
                PropertyA.segment_uuid == SegmentA.uuid,
                PropertyA.key == "has_answer",
                PropertyA.value_bool == True,  # noqa: E712
            )
            .exists()
        )
        result = await session.execute(
            select(DerivativeA.uuid, SegmentA)
            .join(DerivativeA, DerivativeA.segment_uuid == SegmentA.uuid)
            .where(
                DerivativeA.uuid.in_(derivative_uuids),
                prop_filter,
            )
            .order_by(SegmentA.timestamp)
        )
        rows = result.all()
        segment_uuids = [seg.uuid for _, seg in rows]

        if segment_uuids:
            prop_result = await session.execute(
                select(PropertyA).where(PropertyA.segment_uuid.in_(segment_uuids))
            )
            _prop_rows = prop_result.scalars().all()

    return time.perf_counter() - t0


async def bench_read_filtered_b(
    session_maker: async_sessionmaker[AsyncSession],
    derivative_uuids: list[UUID],
    is_postgres: bool,
) -> float:
    t0 = time.perf_counter()
    async with session_maker() as session:
        if is_postgres:
            prop_filter = SegmentB.properties["has_answer"].as_boolean() == True  # noqa: E712
        else:
            prop_filter = func.json_extract(SegmentB.properties, "$.has_answer") == 1

        result = await session.execute(
            select(DerivativeB.uuid, SegmentB)
            .join(DerivativeB, DerivativeB.segment_uuid == SegmentB.uuid)
            .where(
                DerivativeB.uuid.in_(derivative_uuids),
                prop_filter,
            )
            .order_by(SegmentB.timestamp)
        )
        _rows = result.all()

    return time.perf_counter() - t0


# ── Schema C bench functions ─────────────────────────────────────────────────


async def bench_write_c(
    session_maker: async_sessionmaker[AsyncSession],
    segments_c: list[dict],
    derivatives: list[dict],
) -> float:
    t0 = time.perf_counter()
    async with session_maker() as session, session.begin():
        await session.execute(insert(SegmentC), segments_c)
        await session.execute(insert(DerivativeC), derivatives)
    return time.perf_counter() - t0


async def bench_read_c(
    session_maker: async_sessionmaker[AsyncSession],
    derivative_uuids: list[UUID],
) -> float:
    t0 = time.perf_counter()
    async with session_maker() as session:
        result = await session.execute(
            select(DerivativeC.uuid, SegmentC)
            .join(DerivativeC, DerivativeC.segment_uuid == SegmentC.uuid)
            .where(DerivativeC.uuid.in_(derivative_uuids))
            .order_by(SegmentC.timestamp)
        )
        _rows = result.all()

    return time.perf_counter() - t0


async def bench_read_filtered_c(
    session_maker: async_sessionmaker[AsyncSession],
    derivative_uuids: list[UUID],
    is_postgres: bool,
) -> float:
    """Matches SQLAlchemySegmentLinkerPartition._compile_comparison: type check + value cast."""
    from sqlalchemy import and_

    t0 = time.perf_counter()
    async with session_maker() as session:
        _V = "v"
        _T = "t"
        if is_postgres:
            type_check = SegmentC.properties["has_answer"][_T].as_string() == "bool"
            value_check = SegmentC.properties["has_answer"][_V].as_boolean() == True  # noqa: E712
        else:
            type_check = (
                func.json_extract(SegmentC.properties, f"$.has_answer.{_T}") == "bool"
            )
            value_check = (
                func.json_extract(SegmentC.properties, f"$.has_answer.{_V}") == 1
            )

        result = await session.execute(
            select(DerivativeC.uuid, SegmentC)
            .join(DerivativeC, DerivativeC.segment_uuid == SegmentC.uuid)
            .where(
                DerivativeC.uuid.in_(derivative_uuids),
                and_(type_check, value_check),
            )
            .order_by(SegmentC.timestamp)
        )
        _rows = result.all()

    return time.perf_counter() - t0


# ── Runner ───────────────────────────────────────────────────────────────────


async def run_benchmark(
    url: str,
    label: str,
    batch_size: int,
    num_batches: int,
    derivatives_per_segment: int,
    read_iterations: int,
) -> None:
    is_postgres = "postgresql" in url

    engine = create_async_engine(url, pool_size=5, max_overflow=5)
    session_maker = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(BaseA.metadata.drop_all)
        await conn.run_sync(BaseB.metadata.drop_all)
        await conn.run_sync(BaseC.metadata.drop_all)
        await conn.run_sync(BaseA.metadata.create_all)
        await conn.run_sync(BaseB.metadata.create_all)
        await conn.run_sync(BaseC.metadata.create_all)

    total_segments = batch_size * num_batches
    print(f"\n{'=' * 60}")
    print(f"Backend: {label}")
    print(
        f"Config: {total_segments} segments, {derivatives_per_segment} derivatives/segment, "
        f"{read_iterations} read iterations"
    )
    print(f"{'=' * 60}")

    all_derivative_uuids: list[UUID] = []

    # Warmup
    seg, seg_b, seg_c, props, der_a, der_b, der_c = make_batch(
        10, derivatives_per_segment
    )
    await bench_write_a(session_maker, seg, props, der_a)
    await bench_write_b(session_maker, seg_b, der_b)
    await bench_write_c(session_maker, seg_c, der_c)

    async with engine.begin() as conn:
        await conn.run_sync(BaseA.metadata.drop_all)
        await conn.run_sync(BaseB.metadata.drop_all)
        await conn.run_sync(BaseC.metadata.drop_all)
        await conn.run_sync(BaseA.metadata.create_all)
        await conn.run_sync(BaseB.metadata.create_all)
        await conn.run_sync(BaseC.metadata.create_all)

    # Write benchmark
    time_a_write = 0.0
    time_b_write = 0.0
    time_c_write = 0.0

    for batch_idx in range(num_batches):
        (
            segments,
            segments_b,
            segments_c,
            properties,
            derivatives_a,
            derivatives_b,
            derivatives_c,
        ) = make_batch(batch_size, derivatives_per_segment)
        all_derivative_uuids.extend(d["uuid"] for d in derivatives_a)

        time_a_write += await bench_write_a(
            session_maker, segments, properties, derivatives_a
        )
        time_b_write += await bench_write_b(session_maker, segments_b, derivatives_b)
        time_c_write += await bench_write_c(session_maker, segments_c, derivatives_c)

        if (batch_idx + 1) % 10 == 0:
            print(f"  Written {(batch_idx + 1) * batch_size} segments...", flush=True)

    print(f"\nWrite ({total_segments} segments, {total_segments * 4} properties):")
    print(f"  A  Separate table:  {time_a_write:.3f}s")
    print(f"  B  Inline JSONB:    {time_b_write:.3f}s")
    print(f"  C  Tagged JSONB:    {time_c_write:.3f}s")
    print(f"  A/B speedup:        {time_a_write / time_b_write:.2f}x")
    print(f"  A/C speedup:        {time_a_write / time_c_write:.2f}x")
    print(f"  B/C ratio:          {time_b_write / time_c_write:.2f}x")

    # Read benchmark
    random.seed(42)
    sample_sizes = [10, 30, 50]

    for sample_size in sample_sizes:
        time_a_read = 0.0
        time_b_read = 0.0
        time_c_read = 0.0
        time_a_filtered = 0.0
        time_b_filtered = 0.0
        time_c_filtered = 0.0

        for _ in range(read_iterations):
            sample = random.sample(
                all_derivative_uuids, min(sample_size, len(all_derivative_uuids))
            )

            time_a_read += await bench_read_a(session_maker, sample)
            time_b_read += await bench_read_b(session_maker, sample)
            time_c_read += await bench_read_c(session_maker, sample)
            time_a_filtered += await bench_read_filtered_a(session_maker, sample)
            time_b_filtered += await bench_read_filtered_b(
                session_maker, sample, is_postgres
            )
            time_c_filtered += await bench_read_filtered_c(
                session_maker, sample, is_postgres
            )

        print(
            f"\nRead unfiltered ({sample_size} derivatives, {read_iterations} iters):"
        )
        print(f"  A  Separate table:  {time_a_read:.3f}s")
        print(f"  B  Inline JSONB:    {time_b_read:.3f}s")
        print(f"  C  Tagged JSONB:    {time_c_read:.3f}s")
        print(f"  A/B speedup:        {time_a_read / time_b_read:.2f}x")
        print(f"  B/C ratio:          {time_b_read / time_c_read:.2f}x")

        print(f"\nRead filtered ({sample_size} derivatives, {read_iterations} iters):")
        print(f"  A  Separate table:  {time_a_filtered:.3f}s")
        print(f"  B  Inline JSONB:    {time_b_filtered:.3f}s")
        print(f"  C  Tagged JSONB:    {time_c_filtered:.3f}s")
        print(f"  A/B speedup:        {time_a_filtered / time_b_filtered:.2f}x")
        print(f"  B/C ratio:          {time_b_filtered / time_c_filtered:.2f}x")

    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(BaseA.metadata.drop_all)
        await conn.run_sync(BaseB.metadata.drop_all)
        await conn.run_sync(BaseC.metadata.drop_all)
    await engine.dispose()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--num-batches", type=int, default=50)
    parser.add_argument("--derivatives-per-segment", type=int, default=3)
    parser.add_argument("--read-iterations", type=int, default=100)
    parser.add_argument(
        "--include-sqlite", action="store_true", help="Also benchmark SQLite"
    )
    args = parser.parse_args()

    # Start Postgres container
    from testcontainers.postgres import PostgresContainer

    print("Starting Postgres container...", flush=True)
    with PostgresContainer("postgres:16") as pg:
        pg_url = pg.get_connection_url().replace(
            "postgresql+psycopg2://", "postgresql+asyncpg://"
        )
        print("Postgres ready.", flush=True)

        await run_benchmark(
            url=pg_url,
            label="PostgreSQL (testcontainer)",
            batch_size=args.batch_size,
            num_batches=args.num_batches,
            derivatives_per_segment=args.derivatives_per_segment,
            read_iterations=args.read_iterations,
        )

    if args.include_sqlite:
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "bench.db")
            sqlite_url = f"sqlite+aiosqlite:///{db_path}"

            await run_benchmark(
                url=sqlite_url,
                label="SQLite (temp file)",
                batch_size=args.batch_size,
                num_batches=args.num_batches,
                derivatives_per_segment=args.derivatives_per_segment,
                read_iterations=args.read_iterations,
            )


if __name__ == "__main__":
    asyncio.run(main())
