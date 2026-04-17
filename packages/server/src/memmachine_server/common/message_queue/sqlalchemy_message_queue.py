"""SQLAlchemy-backed :class:`MessageQueue` for SQLite and Postgres.

Schema
------
* ``message_queue_consumers``: registered consumer ids.
* ``message_queue_pending``: one row per ``(consumer_id, partition_id,
  message_id)`` tuple that has not yet been acked by that consumer.

A message is "present in the queue" iff at least one pending row
references it.  Acking deletes the (consumer, message) pending rows;
when the last one goes, the message is effectively gone — no
explicit cleanup step is needed because there is no separate
messages table.

Producing with no consumers registered is a no-op: the message has
nothing to enqueue against.

Dialect support
---------------
Uses SQLAlchemy's per-dialect ``insert`` with
``on_conflict_do_nothing()`` to handle duplicate-produce retries
safely on both Postgres and SQLite.
"""

from collections.abc import AsyncIterator, Iterable
from uuid import UUID

from sqlalchemy import Executable, String, Uuid, delete, func, select
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from memmachine_server.common.message_queue.message_queue import MessageQueue


class MessageQueueBase(DeclarativeBase):
    """Independent DeclarativeBase for message queue tables."""


class ConsumerRow(MessageQueueBase):
    """Registered consumer row keyed by ``consumer_id``."""

    __tablename__ = "message_queue_consumers"

    consumer_id: Mapped[str] = mapped_column(String, primary_key=True)


class PendingRow(MessageQueueBase):
    """One pending ``message_id`` for a given consumer and partition."""

    __tablename__ = "message_queue_pending"

    consumer_id: Mapped[str] = mapped_column(String, primary_key=True)
    partition_id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    message_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)


def _insert_ignore_consumer(dialect: str, consumer_id: str) -> Executable:
    if dialect == "postgresql":
        return (
            postgresql.insert(ConsumerRow)
            .values(consumer_id=consumer_id)
            .on_conflict_do_nothing()
        )
    if dialect == "sqlite":
        return (
            sqlite.insert(ConsumerRow)
            .values(consumer_id=consumer_id)
            .on_conflict_do_nothing()
        )
    raise NotImplementedError(
        f"SQLAlchemyMessageQueue: unsupported dialect {dialect!r}"
    )


def _insert_ignore_pending(dialect: str, rows: list[dict[str, object]]) -> Executable:
    if dialect == "postgresql":
        return postgresql.insert(PendingRow).values(rows).on_conflict_do_nothing()
    if dialect == "sqlite":
        return sqlite.insert(PendingRow).values(rows).on_conflict_do_nothing()
    raise NotImplementedError(
        f"SQLAlchemyMessageQueue: unsupported dialect {dialect!r}"
    )


class SQLAlchemyMessageQueue(MessageQueue):
    """:class:`MessageQueue` backed by SQLAlchemy (SQLite and Postgres)."""

    def __init__(self, engine: AsyncEngine) -> None:
        """Bind to an existing async engine; caller owns its lifecycle."""
        self._engine = engine
        self._session_factory = async_sessionmaker(engine, expire_on_commit=False)
        self._dialect = engine.dialect.name
        if self._dialect not in ("postgresql", "sqlite"):
            raise NotImplementedError(
                "SQLAlchemyMessageQueue: unsupported dialect "
                f"{self._dialect!r} (expected 'postgresql' or 'sqlite')"
            )

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def startup(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(MessageQueueBase.metadata.create_all)

    async def cleanup(self) -> None:
        return

    async def delete_all(self) -> None:
        async with self._session_factory() as session, session.begin():
            await session.execute(delete(PendingRow))
            await session.execute(delete(ConsumerRow))

    # ------------------------------------------------------------------ #
    # Consumer registration
    # ------------------------------------------------------------------ #

    async def register_consumer(self, consumer_id: str) -> None:
        stmt = _insert_ignore_consumer(self._dialect, consumer_id)
        async with self._session_factory() as session, session.begin():
            await session.execute(stmt)

    async def deregister_consumer(self, consumer_id: str) -> None:
        async with self._session_factory() as session, session.begin():
            await session.execute(
                delete(PendingRow).where(PendingRow.consumer_id == consumer_id)
            )
            await session.execute(
                delete(ConsumerRow).where(ConsumerRow.consumer_id == consumer_id)
            )

    async def list_consumers(self) -> tuple[str, ...]:
        stmt = select(ConsumerRow.consumer_id).order_by(ConsumerRow.consumer_id)
        async with self._session_factory() as session:
            result = await session.execute(stmt)
            return tuple(result.scalars())

    # ------------------------------------------------------------------ #
    # Producer side
    # ------------------------------------------------------------------ #

    async def produce(self, partition_id: str, message_id: UUID) -> None:
        async with self._session_factory() as session, session.begin():
            consumer_result = await session.execute(select(ConsumerRow.consumer_id))
            consumer_ids = list(consumer_result.scalars())
            if not consumer_ids:
                return
            rows: list[dict[str, object]] = [
                {
                    "consumer_id": cid,
                    "partition_id": partition_id,
                    "message_id": message_id,
                }
                for cid in consumer_ids
            ]
            await session.execute(_insert_ignore_pending(self._dialect, rows))

    # ------------------------------------------------------------------ #
    # Consumer side
    # ------------------------------------------------------------------ #

    async def list_pending(
        self,
        *,
        consumer_id: str,
        partition_ids: Iterable[str] | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[UUID]:
        stmt = (
            select(PendingRow.message_id)
            .where(PendingRow.consumer_id == consumer_id)
            .order_by(PendingRow.partition_id, PendingRow.message_id)
        )
        if partition_ids is not None:
            pids = list(partition_ids)
            if not pids:
                return
            stmt = stmt.where(PendingRow.partition_id.in_(pids))
        if limit is not None:
            stmt = stmt.limit(limit)
        async with self._session_factory() as session:
            result = await session.stream(stmt)
            async for row in result:
                yield row[0]

    async def count_pending(
        self,
        *,
        consumer_id: str,
        partition_ids: Iterable[str] | None = None,
    ) -> int:
        stmt = (
            select(func.count())
            .select_from(PendingRow)
            .where(PendingRow.consumer_id == consumer_id)
        )
        if partition_ids is not None:
            pids = list(partition_ids)
            if not pids:
                return 0
            stmt = stmt.where(PendingRow.partition_id.in_(pids))
        async with self._session_factory() as session:
            result = await session.execute(stmt)
            return result.scalar_one()

    async def ack(
        self,
        *,
        consumer_id: str,
        partition_id: str,
        message_ids: Iterable[UUID],
    ) -> None:
        mids = list(message_ids)
        if not mids:
            return
        stmt = delete(PendingRow).where(
            PendingRow.consumer_id == consumer_id,
            PendingRow.partition_id == partition_id,
            PendingRow.message_id.in_(mids),
        )
        async with self._session_factory() as session, session.begin():
            await session.execute(stmt)

    # ------------------------------------------------------------------ #
    # Scheduling discovery
    # ------------------------------------------------------------------ #

    async def list_partitions_with_pending(
        self,
        *,
        consumer_id: str,
    ) -> AsyncIterator[str]:
        stmt = (
            select(PendingRow.partition_id)
            .where(PendingRow.consumer_id == consumer_id)
            .distinct()
            .order_by(PendingRow.partition_id)
        )
        async with self._session_factory() as session:
            result = await session.stream(stmt)
            async for row in result:
                yield row[0]
