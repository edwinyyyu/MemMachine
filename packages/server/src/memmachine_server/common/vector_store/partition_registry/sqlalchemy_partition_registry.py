"""
A partition registry in a relational database, through SQLAlchemy.

A table of registered partitions keyed by vector store and partition key,
each with its incarnation and whether it is live (its storage prepared), and
a queue of deleted incarnations claimed in the order they come due. The
primary key arbitrates registration across processes, a conditional update
confirms a reservation, unregistration is one transaction, and a purge
claim is a lease on a tombstone that one short write takes and another ends.
"""

import asyncio
import contextlib
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import override
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, InstanceOf, JsonValue, field_validator
from sqlalchemy import (
    JSON,
    Boolean,
    ColumnElement,
    DateTime,
    Index,
    Integer,
    Interval,
    String,
    Uuid,
    bindparam,
    case,
    delete,
    func,
    insert,
    literal,
    or_,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.orm import DeclarativeBase, MappedColumn, mapped_column
from sqlalchemy.pool import StaticPool

from memmachine_server.common.utils import ensure_tz_aware
from memmachine_server.common.vector_store.data_types import (
    PartitionSchema,
    VectorStoreAttemptsExhaustedError,
    VectorStorePartitionAlreadyExistsError,
    VectorStorePartitionDeletedError,
    VectorStorePartitionHandleStaleError,
    VectorStorePartitionPendingError,
)
from memmachine_server.common.vector_store.utils import (
    _IDENTIFIER_MAX_BYTES,
    validate_identifier,
)

from .partition_registry import (
    PurgeRound,
    Registration,
    Reservation,
    VectorStorePartitionRegistry,
)

logger = logging.getLogger(__name__)

_MAX_MINT_ATTEMPTS = 10

# The first SQLite with RETURNING, which the registry uses.
_MIN_SQLITE_VERSION = (3, 35)

# Consecutive failed purge rounds after which a tombstone is dead-lettered.
_MAX_FAILED_PURGE_ROUNDS = 10

_JSON_AUTO = JSON().with_variant(JSONB, "postgresql")


class BasePartitionRegistry(DeclarativeBase):
    """Base class for partition registry tables."""


class PartitionRow(BasePartitionRegistry):
    """A registered partition of a vector store, pending or live."""

    __tablename__ = "partition_registry_pt"

    vector_store_name: MappedColumn[str] = mapped_column(
        String(_IDENTIFIER_MAX_BYTES), primary_key=True
    )
    partition_key: MappedColumn[str] = mapped_column(
        String(_IDENTIFIER_MAX_BYTES), primary_key=True
    )
    incarnation: MappedColumn[UUID] = mapped_column(Uuid, nullable=False, unique=True)
    # The dimensions, metric and declared schema the partition was created
    # under, so a store built with others fails loudly instead of filtering
    # on indexes that are not there.
    schema: MappedColumn[dict[str, JsonValue]] = mapped_column(
        _JSON_AUTO, nullable=False
    )
    # Whether the partition's storage is prepared; it is pending until then.
    live: MappedColumn[bool] = mapped_column(Boolean, nullable=False)
    registered_at: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class PurgeQueueRow(BasePartitionRegistry):
    """A deleted partition's tombstone: its incarnation awaiting purge."""

    __tablename__ = "partition_registry_gc"

    incarnation: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    vector_store_name: MappedColumn[str] = mapped_column(
        String(_IDENTIFIER_MAX_BYTES), nullable=False
    )
    # The partition's key, kept for inspection.
    partition_key: MappedColumn[str] = mapped_column(
        String(_IDENTIFIER_MAX_BYTES), nullable=False
    )
    enqueued_at: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    # Consecutive purge rounds on the tombstone that raised or never ended,
    # and when the last one raised or, if it never ended, was claimed, on the
    # database clock.
    failed_rounds: MappedColumn[int] = mapped_column(Integer, nullable=False, default=0)
    last_failed_at: MappedColumn[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # When the tombstone's latest claim was taken, on the database clock, and
    # None once its round ended. The claim holds the tombstone until the
    # purge lease has passed since.
    claimed_at: MappedColumn[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Incremented by each claim. A round ends its claim only while the
    # tombstone's generation is still its claim's.
    claim_generation: MappedColumn[int] = mapped_column(
        Integer, nullable=False, default=0
    )

    __table_args__ = (
        Index("partition_registry_gc__vs_ea", "vector_store_name", "enqueued_at"),
    )


class _RegistryInsertRejectedError(Exception):
    """A registry insert was rejected for a reason other than the key being taken."""


class SQLAlchemyVectorStorePartitionRegistryParams(BaseModel):
    """
    Parameters for SQLAlchemyVectorStorePartitionRegistry.

    Attributes:
        engine (AsyncEngine):
            Async SQLAlchemy engine, on PostgreSQL or SQLite.
        vector_store_name (str):
            The name of the store the registry serves, under which its rows are
            kept: registry objects with the same name on the same database are
            one registry, so a name identifies one store among all the stores
            whose registries share the database. It must match [a-z0-9_]+ and
            be at most 32 bytes.
        tombstone_retention_seconds (int):
            Seconds a deleted partition's records are kept before its purge
            starts, on the database clock. It must exceed, by orders of
            magnitude, the longest a write to the backend can be in flight and
            the delay before the store's reads reflect a write.
        purge_lease_seconds (int):
            Seconds a purge round's claim holds its tombstone, on the database
            clock; it should exceed the longest round (default: 300).
        base_purge_retry_backoff_seconds (int):
            Seconds before a tombstone whose purge round raised is claimed
            again, doubled for each further consecutive failure (default: 30).
        max_purge_retry_backoff_seconds (int):
            The longest the purge retry backoff grows to (default: 3600).
    """

    engine: InstanceOf[AsyncEngine] = Field(
        ..., description="Async SQLAlchemy engine, on PostgreSQL or SQLite"
    )
    vector_store_name: str = Field(
        ...,
        description=(
            "The name of the store the registry serves, under which its rows are "
            "kept: registry objects with the same name on the same database are "
            "one registry, so a name identifies one store among all the stores "
            "whose registries share the database. It must match [a-z0-9_]+ and "
            "be at most 32 bytes"
        ),
    )
    tombstone_retention_seconds: int = Field(
        ...,
        ge=0,
        description=(
            "Seconds a deleted partition's records are kept before its purge "
            "starts, on the database clock. It must exceed, by orders of "
            "magnitude, the longest a write to the backend can be in flight and "
            "the delay before the store's reads reflect a write"
        ),
    )
    purge_lease_seconds: int = Field(
        300,
        gt=0,
        description=(
            "Seconds a purge round's claim holds its tombstone, on the database "
            "clock; it should exceed the longest round"
        ),
    )
    base_purge_retry_backoff_seconds: int = Field(
        30,
        gt=0,
        description=(
            "Seconds before a tombstone whose purge round raised is claimed "
            "again, doubled for each further consecutive failure"
        ),
    )
    max_purge_retry_backoff_seconds: int = Field(
        3600, gt=0, description="The longest the purge retry backoff grows to"
    )

    @field_validator("engine")
    @classmethod
    def _validate_engine(cls, engine: AsyncEngine) -> AsyncEngine:
        if engine.dialect.name not in ("postgresql", "sqlite"):
            raise ValueError(
                f"Engine uses the {engine.dialect.name} dialect, which the "
                "registry does not support. Use PostgreSQL or SQLite."
            )
        # The registry's arbitration rests on transactions of their own.
        engine_shares_one_connection = isinstance(engine.pool, StaticPool)
        if engine_shares_one_connection:
            raise ValueError(
                "Engine uses StaticPool, which shares one connection across "
                "sessions. Use a multi-connection pool instead."
            )
        database = engine.url.database
        if engine.dialect.name == "sqlite" and database in (None, "", ":memory:"):
            raise ValueError(
                "Engine uses in-memory SQLite, where each connection gets a "
                "separate database. Use a file path instead."
            )
        if (
            engine.dialect.name == "sqlite"
            and sqlite3.sqlite_version_info < _MIN_SQLITE_VERSION
        ):
            minimum = ".".join(str(part) for part in _MIN_SQLITE_VERSION)
            raise ValueError(
                f"SQLite runtime {sqlite3.sqlite_version} lacks the RETURNING "
                f"support the registry depends on. Use SQLite {minimum} or newer."
            )
        return engine

    @field_validator("vector_store_name")
    @classmethod
    def _validate_vector_store_name(cls, vector_store_name: str) -> str:
        if not validate_identifier(vector_store_name):
            raise ValueError(
                f"Vector store name {vector_store_name!r} must match [a-z0-9_]+ "
                "and be at most 32 bytes"
            )
        return vector_store_name


class SQLAlchemyVectorStorePartitionRegistry(VectorStorePartitionRegistry):
    """
    The registry of one vector store's partitions.

    Registries of different vector stores share the tables, keyed by vector
    store name. A purge claim holds its tombstone for the purge lease, or
    until its round ends, and no transaction stays open during the round.
    After `_MAX_FAILED_PURGE_ROUNDS` consecutive failed rounds, a
    tombstone is dead-lettered: kept, its incarnation reserved, skipped by
    claims, and reported in an error log. Setting its
    `failed_rounds` back to 0 returns it to the purge.
    """

    def __init__(self, params: SQLAlchemyVectorStorePartitionRegistryParams) -> None:
        """Initialize with the provided parameters."""
        self._engine = params.engine
        self._is_sqlite = params.engine.dialect.name == "sqlite"
        self._vector_store_name = params.vector_store_name
        self._tombstone_retention = timedelta(
            seconds=params.tombstone_retention_seconds
        )
        self._purge_lease = timedelta(seconds=params.purge_lease_seconds)
        self._base_purge_retry_backoff = timedelta(
            seconds=params.base_purge_retry_backoff_seconds
        )
        self._max_purge_retry_backoff = timedelta(
            seconds=params.max_purge_retry_backoff_seconds
        )
        self._claim_releases: set[asyncio.Task[None]] = set()

    @override
    async def startup(self) -> None:
        async with self._engine.begin() as connection:
            await connection.run_sync(BasePartitionRegistry.metadata.create_all)

    @override
    async def reserve(self, partition_key: str, schema: PartitionSchema) -> Reservation:
        # The primary key arbitrates the partition key across processes.
        # An insert rejected for another reason, such as a minted incarnation
        # that is registered or awaiting purge, is tried again with a fresh one.
        attempts = 0
        while True:
            incarnation = uuid4()
            try:
                await self._insert(partition_key, incarnation, schema)
            except _RegistryInsertRejectedError as err:
                logger.warning(
                    "Reserving partition %r under incarnation %s was rejected: "
                    "%s; minting another",
                    partition_key,
                    incarnation,
                    err,
                )
                attempts += 1
                if attempts >= _MAX_MINT_ATTEMPTS:
                    raise VectorStoreAttemptsExhaustedError(
                        f"Creating partition {partition_key!r} made no progress "
                        f"after {_MAX_MINT_ATTEMPTS} attempts"
                    ) from err
                continue
            return _SQLAlchemyReservation(
                partition_key=partition_key,
                schema=schema,
                incarnation=incarnation,
                engine=self._engine,
                vector_store_name=self._vector_store_name,
            )

    async def _insert(
        self, partition_key: str, incarnation: UUID, schema: PartitionSchema
    ) -> None:
        # The queue check runs after the insert, so a concurrent deletion
        # queuing a colliding incarnation, which the insert waited on, is
        # visible to it; the locking read sees the latest committed state.
        try:
            async with self._engine.begin() as connection:
                await connection.execute(
                    insert(PartitionRow).values(
                        vector_store_name=self._vector_store_name,
                        partition_key=partition_key,
                        incarnation=incarnation,
                        schema=schema.model_dump(mode="json"),
                        live=False,
                        registered_at=func.now(),
                    )
                )
                queued = (
                    await connection.execute(
                        select(PurgeQueueRow.incarnation)
                        .where(PurgeQueueRow.incarnation == incarnation)
                        .with_for_update(read=True)
                    )
                ).scalar_one_or_none()
                if queued is not None:
                    raise _RegistryInsertRejectedError("the incarnation awaits purge")
        except IntegrityError as err:
            async with self._engine.connect() as connection:
                taken = (
                    await connection.execute(
                        select(PartitionRow.incarnation).where(
                            PartitionRow.vector_store_name == self._vector_store_name,
                            PartitionRow.partition_key == partition_key,
                        )
                    )
                ).scalar_one_or_none()
            if taken is not None:
                raise VectorStorePartitionAlreadyExistsError(
                    self._vector_store_name, partition_key
                ) from err
            raise _RegistryInsertRejectedError(
                "the insert failed and no row exists under the key"
            ) from err

    @override
    async def resolve(self, partition_key: str) -> Registration | None:
        async with self._engine.connect() as connection:
            row = (
                await connection.execute(
                    select(
                        PartitionRow.incarnation,
                        PartitionRow.schema,
                        PartitionRow.live,
                        PartitionRow.registered_at,
                    ).where(
                        PartitionRow.vector_store_name == self._vector_store_name,
                        PartitionRow.partition_key == partition_key,
                    )
                )
            ).one_or_none()
        if row is None:
            return None
        schema = PartitionSchema.model_validate(row.schema)
        if not row.live:
            raise VectorStorePartitionPendingError(
                self._vector_store_name,
                partition_key,
                ensure_tz_aware(row.registered_at),
                schema,
            )
        return _SQLAlchemyRegistration(
            partition_key=partition_key,
            schema=schema,
            incarnation=row.incarnation,
            engine=self._engine,
            vector_store_name=self._vector_store_name,
        )

    @override
    async def unregister(self, partition_key: str) -> None:
        await _unregister_where(
            self._engine,
            self._vector_store_name,
            PartitionRow.partition_key == partition_key,
        )

    @override
    async def run_purge_round(self, purge_round: PurgeRound) -> bool:
        # The claim is a lease: one committed write takes the oldest due
        # tombstone that no unexpired claim holds, and the round runs with no
        # transaction open, so no lock is held across its remote calls. The
        # lease only spreads rounds across purgers: a round is safe to repeat
        # and to run beside another, so a lease that ends early repeats work.
        # The retention, the backoff and the lease are applied when a claim is
        # decided, on the database clock, so a changed duration applies to
        # every tombstone. On PostgreSQL a concurrent claim skips the row
        # being claimed; SQLite drops the locking clause and serializes the
        # claims as writes. The writes that end a claim are conditioned on its
        # generation, so a round that outlasted its lease cannot end the claim
        # taken after it.
        oldest_due = (
            select(PurgeQueueRow.incarnation)
            .where(
                PurgeQueueRow.vector_store_name == self._vector_store_name,
                PurgeQueueRow.enqueued_at
                <= self._now_less(self._tombstone_retention, "retention"),
                PurgeQueueRow.failed_rounds < _MAX_FAILED_PURGE_ROUNDS,
                or_(
                    PurgeQueueRow.failed_rounds == 0,
                    PurgeQueueRow.last_failed_at <= self._backoff_cutoff(),
                ),
                or_(
                    PurgeQueueRow.claimed_at.is_(None),
                    PurgeQueueRow.claimed_at
                    <= self._now_less(self._purge_lease, "purge_lease"),
                ),
            )
            .order_by(PurgeQueueRow.enqueued_at)
            .limit(1)
            .with_for_update(skip_locked=True)
            .scalar_subquery()
        )
        # A claim that finds the tombstone's previous claim unended, its lease
        # passed, runs no round: it counts that round as failed, as of when it
        # was claimed, and ends it. SET reads the row as it was, so one
        # statement both tells the cases apart and records either.
        unended = PurgeQueueRow.claimed_at.is_not(None)
        async with self._engine.begin() as connection:
            claim = (
                await connection.execute(
                    update(PurgeQueueRow)
                    .where(PurgeQueueRow.incarnation == oldest_due)
                    .values(
                        claimed_at=case((unended, None), else_=func.now()),
                        claim_generation=PurgeQueueRow.claim_generation + 1,
                        failed_rounds=PurgeQueueRow.failed_rounds
                        + case((unended, 1), else_=0),
                        last_failed_at=case(
                            (unended, PurgeQueueRow.claimed_at),
                            else_=PurgeQueueRow.last_failed_at,
                        ),
                    )
                    .returning(
                        PurgeQueueRow.incarnation,
                        PurgeQueueRow.claim_generation,
                        PurgeQueueRow.claimed_at,
                        PurgeQueueRow.failed_rounds,
                        PurgeQueueRow.last_failed_at,
                    )
                )
            ).one_or_none()
        if claim is None:
            return False
        if claim.claimed_at is None:
            logger.warning(
                "Purge round on incarnation %s, claimed at %s, did not end "
                "within its %d s lease: its purger stopped, or the round runs "
                "on past the lease. It counts as a failed round.",
                claim.incarnation,
                claim.last_failed_at,
                int(self._purge_lease.total_seconds()),
            )
            self._report_dead_lettering(
                claim.incarnation,
                claim.failed_rounds,
                f"the round claimed at {claim.last_failed_at} did not end",
            )
            return True
        try:
            any_records_found = await purge_round(claim.incarnation)
        except Exception as error:
            await self._count_failed_round(
                claim.incarnation, claim.claim_generation, error
            )
            raise
        except BaseException:
            # A cancelled round ends its claim uncounted, shielded so a
            # cancelled purge still frees the tombstone at once. The task
            # reports its own failure, since a purge cancelled again stops
            # awaiting it; the claim then ends with its lease.
            release = asyncio.create_task(
                self._end_claim(claim.incarnation, claim.claim_generation)
            )
            self._claim_releases.add(release)

            def finish(task: asyncio.Task[None]) -> None:
                self._claim_releases.discard(task)
                if not task.cancelled() and task.exception() is not None:
                    logger.exception(
                        "Could not end the claim of a cancelled purge round on "
                        "incarnation %s; it ends with its lease",
                        claim.incarnation,
                        exc_info=task.exception(),
                    )

            release.add_done_callback(finish)
            with contextlib.suppress(Exception):
                await asyncio.shield(release)
            raise
        await self._record_round(
            claim.incarnation, claim.claim_generation, any_records_found
        )
        return True

    async def _record_round(
        self, incarnation: UUID, claim_generation: int, any_records_found: bool
    ) -> None:
        """Record a purge round's outcome.

        A round that found no records removes the tombstone, under whichever
        claim it ran: an incarnation found empty after the retention needs no
        more rounds. One that found records ends its claim and resets the
        tombstone's count of failed rounds, while its claim is the latest.
        """
        if not any_records_found:
            async with self._engine.begin() as connection:
                await connection.execute(
                    delete(PurgeQueueRow).where(
                        PurgeQueueRow.incarnation == incarnation
                    )
                )
            return
        async with self._engine.begin() as connection:
            ended = await connection.execute(
                update(PurgeQueueRow)
                .where(
                    PurgeQueueRow.incarnation == incarnation,
                    PurgeQueueRow.claim_generation == claim_generation,
                )
                .values(failed_rounds=0, claimed_at=None)
            )
        if ended.rowcount != 1:
            logger.warning(
                "Purge round on incarnation %s outlasted its claim's %d s lease: "
                "the tombstone was claimed again or removed while it ran",
                incarnation,
                int(self._purge_lease.total_seconds()),
            )

    async def _count_failed_round(
        self, incarnation: UUID, claim_generation: int, error: Exception
    ) -> None:
        """Count a raised purge round against its tombstone, ending its claim, and report its dead-lettering.

        Counted while the round's claim is the latest, so a round that
        outlasted its lease does not count against the claim taken after it.
        """
        async with self._engine.begin() as connection:
            failed_rounds = (
                await connection.execute(
                    update(PurgeQueueRow)
                    .where(
                        PurgeQueueRow.incarnation == incarnation,
                        PurgeQueueRow.claim_generation == claim_generation,
                    )
                    .values(
                        failed_rounds=PurgeQueueRow.failed_rounds + 1,
                        last_failed_at=func.now(),
                        claimed_at=None,
                    )
                    .returning(PurgeQueueRow.failed_rounds)
                )
            ).scalar_one_or_none()
        if failed_rounds is None:
            logger.warning(
                "Purge round on incarnation %s outlasted its claim's %d s lease, "
                "so its failure is not counted: the tombstone was claimed again "
                "or removed while it ran",
                incarnation,
                int(self._purge_lease.total_seconds()),
            )
            return
        self._report_dead_lettering(incarnation, failed_rounds, repr(error))

    async def _end_claim(self, incarnation: UUID, claim_generation: int) -> None:
        """End a round's claim without recording an outcome, while its claim is the latest."""
        async with self._engine.begin() as connection:
            await connection.execute(
                update(PurgeQueueRow)
                .where(
                    PurgeQueueRow.incarnation == incarnation,
                    PurgeQueueRow.claim_generation == claim_generation,
                )
                .values(claimed_at=None)
            )

    @staticmethod
    def _report_dead_lettering(
        incarnation: UUID, failed_rounds: int, last_error: str
    ) -> None:
        """Report a tombstone whose count of failed rounds reached the dead-letter bound."""
        # The claim takes only tombstones under the bound, so one at or past
        # it is dead-lettered.
        if failed_rounds >= _MAX_FAILED_PURGE_ROUNDS:
            logger.error(
                "Purge of incarnation %s failed %d rounds in a row and is "
                "dead-lettered: its records stay and it is no longer claimed. "
                "Last error: %s. Set its failed_rounds to 0 in %s to retry it.",
                incarnation,
                failed_rounds,
                last_error,
                PurgeQueueRow.__tablename__,
            )

    def _backoff_cutoff(self) -> ColumnElement:
        """The database clock's now, less each queue row's backoff, computed by the database."""
        # 1 << (f - 1) is 2 ** (f - 1) on both dialects. The claim evaluates
        # it for rows at 0 failures too, so the count is clamped at 0: a
        # shift by a negative count is undefined on PostgreSQL.
        doublings = literal(1, Integer).op("<<")(
            case(
                (PurgeQueueRow.failed_rounds > 0, PurgeQueueRow.failed_rounds - 1),
                else_=0,
            )
        )
        if self._is_sqlite:
            seconds = func.min(
                int(self._base_purge_retry_backoff.total_seconds()) * doublings,
                int(self._max_purge_retry_backoff.total_seconds()),
            )
            return func.datetime(
                "now",
                func.printf("-%d seconds", seconds),
                type_=DateTime(timezone=True),
            )
        return func.now() - func.least(
            bindparam(
                "base_retry_backoff", self._base_purge_retry_backoff, type_=Interval
            )
            * doublings,
            bindparam(
                "max_retry_backoff", self._max_purge_retry_backoff, type_=Interval
            ),
        )

    def _now_less(self, period: timedelta, key: str) -> ColumnElement:
        """The database clock's now, less `period`, computed by the database; `key` names its bound parameter."""
        if self._is_sqlite:
            # datetime() emits the form CURRENT_TIMESTAMP stamps with, so
            # the stamps and the cutoff compare as text in time order.
            seconds = int(period.total_seconds())
            return func.datetime(
                "now",
                bindparam(f"{key}_modifier", f"-{seconds} seconds"),
                type_=DateTime(timezone=True),
            )
        return func.now() - bindparam(key, period, type_=Interval)


@dataclass(frozen=True)
class _SQLAlchemyReservation(Reservation):
    """A reservation whose methods write the registry's tables directly."""

    engine: AsyncEngine = field(repr=False, compare=False)
    vector_store_name: str = field(repr=False)

    @override
    async def confirm(self) -> Registration:
        # Conditional on the incarnation and on the row being pending, so a
        # creation marks live only the partition it reserved.
        async with self.engine.begin() as connection:
            result = await connection.execute(
                update(PartitionRow)
                .where(
                    PartitionRow.vector_store_name == self.vector_store_name,
                    PartitionRow.partition_key == self.partition_key,
                    PartitionRow.incarnation == self.incarnation,
                    PartitionRow.live.is_(False),
                )
                .values(live=True)
            )
        if result.rowcount != 1:
            raise VectorStorePartitionDeletedError(
                self.vector_store_name, self.partition_key
            )
        return _SQLAlchemyRegistration(
            partition_key=self.partition_key,
            schema=self.schema,
            incarnation=self.incarnation,
            engine=self.engine,
            vector_store_name=self.vector_store_name,
        )

    @override
    async def cancel(self) -> None:
        # Conditional on the row being pending, as confirm is, so a cancel
        # after a confirmation that committed leaves the live partition.
        await _unregister_where(
            self.engine,
            self.vector_store_name,
            PartitionRow.incarnation == self.incarnation,
            PartitionRow.live.is_(False),
        )


@dataclass(frozen=True)
class _SQLAlchemyRegistration(Registration):
    """A registration whose method reads the registry's tables directly."""

    engine: AsyncEngine = field(repr=False, compare=False)
    vector_store_name: str = field(repr=False)

    @override
    async def require_current(self) -> None:
        async with self.engine.connect() as connection:
            current = (
                await connection.execute(
                    select(PartitionRow.incarnation).where(
                        PartitionRow.vector_store_name == self.vector_store_name,
                        PartitionRow.partition_key == self.partition_key,
                    )
                )
            ).scalar_one_or_none()
        if current != self.incarnation:
            raise VectorStorePartitionHandleStaleError(
                self.vector_store_name, self.partition_key
            )


async def _unregister_where(
    engine: AsyncEngine, vector_store_name: str, *conditions: ColumnElement[bool]
) -> None:
    """Delete the partition row of a vector store the conditions select, and queue its tombstone.

    One transaction, so the partition is unreachable once it commits. The
    DELETE goes first and takes the row's write lock, so racing deleters
    serialize on it and the loser finds no row and returns.
    """
    async with engine.begin() as connection:
        row = (
            await connection.execute(
                delete(PartitionRow)
                .where(PartitionRow.vector_store_name == vector_store_name, *conditions)
                .returning(PartitionRow.incarnation, PartitionRow.partition_key)
            )
        ).one_or_none()
        if row is None:
            return
        await connection.execute(
            insert(PurgeQueueRow).values(
                incarnation=row.incarnation,
                vector_store_name=vector_store_name,
                partition_key=row.partition_key,
                enqueued_at=func.now(),
            )
        )
