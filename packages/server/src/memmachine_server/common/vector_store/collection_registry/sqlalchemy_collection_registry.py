"""
A collection registry in a relational database, through SQLAlchemy.

Qdrant and Milvus have no transactions or unique constraints, so a catalog
kept in them cannot arbitrate two processes creating, deleting or
reclaiming the same logical collection. This registry keeps it in a
relational database: a table of live collections keyed by vector store,
namespace and name, each with its incarnation, and a queue of deleted
incarnations claimed in the order they come due. The primary key
arbitrates registration, unregistration is one transaction, and a purge
claim is a row lock.
"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import override
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, InstanceOf, JsonValue, field_validator
from sqlalchemy import (
    JSON,
    ColumnElement,
    DateTime,
    Index,
    Integer,
    Interval,
    String,
    Uuid,
    bindparam,
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
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine
from sqlalchemy.orm import DeclarativeBase, MappedColumn, mapped_column

from memmachine_server.common.vector_store.data_types import (
    VectorStoreAttemptsExhaustedError,
    VectorStoreCollectionAlreadyExistsError,
    VectorStoreCollectionConfig,
)
from memmachine_server.common.vector_store.utils import (
    _IDENTIFIER_MAX_BYTES,
    validate_identifier,
)

from .collection_registry import (
    PurgeClaim,
    RegisteredCollection,
    VectorStoreCollectionRegistry,
)

logger = logging.getLogger(__name__)

_MAX_MINT_ATTEMPTS = 10

# Consecutive failed purge rounds after which a tombstone is dead-lettered.
_MAX_FAILED_PURGE_ROUNDS = 10

_JSON_AUTO = JSON().with_variant(JSONB, "postgresql")


class BaseCollectionRegistry(DeclarativeBase):
    """Base class for collection registry tables."""


class CollectionRow(BaseCollectionRegistry):
    """A live collection of a vector store."""

    __tablename__ = "collection_registry_ct"

    vector_store_name: MappedColumn[str] = mapped_column(
        String(_IDENTIFIER_MAX_BYTES), primary_key=True
    )
    namespace: MappedColumn[str] = mapped_column(
        String(_IDENTIFIER_MAX_BYTES), primary_key=True
    )
    name: MappedColumn[str] = mapped_column(
        String(_IDENTIFIER_MAX_BYTES), primary_key=True
    )
    incarnation: MappedColumn[UUID] = mapped_column(Uuid, nullable=False, unique=True)
    # The configuration the collection was created with.
    config: MappedColumn[dict[str, JsonValue]] = mapped_column(
        _JSON_AUTO, nullable=False
    )


class PurgeQueueRow(BaseCollectionRegistry):
    """A deleted collection's tombstone: its incarnation awaiting purge."""

    __tablename__ = "collection_registry_gc"

    incarnation: MappedColumn[UUID] = mapped_column(Uuid, primary_key=True)
    vector_store_name: MappedColumn[str] = mapped_column(
        String(_IDENTIFIER_MAX_BYTES), nullable=False
    )
    namespace: MappedColumn[str] = mapped_column(
        String(_IDENTIFIER_MAX_BYTES), nullable=False
    )
    # The collection's name, kept for inspection.
    name: MappedColumn[str] = mapped_column(
        String(_IDENTIFIER_MAX_BYTES), nullable=False
    )
    # The configuration names the native collection the records are in.
    config: MappedColumn[dict[str, JsonValue]] = mapped_column(
        _JSON_AUTO, nullable=False
    )
    enqueued_at: MappedColumn[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    # Consecutive purge rounds on the tombstone that raised, and when the
    # last one did, on the database clock.
    failed_rounds: MappedColumn[int] = mapped_column(Integer, nullable=False, default=0)
    last_failed_at: MappedColumn[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        Index("collection_registry_gc__vs_ea", "vector_store_name", "enqueued_at"),
    )


class _RegistryInsertRejectedError(Exception):
    """A registry insert was rejected; retry with a fresh incarnation."""


class SQLAlchemyVectorStoreCollectionRegistryParams(BaseModel):
    """
    Parameters for SQLAlchemyVectorStoreCollectionRegistry.

    Attributes:
        engine (AsyncEngine):
            Async SQLAlchemy engine, on PostgreSQL or SQLite.
        vector_store_name (str):
            The name the registry's rows are kept under: registry objects with
            the same name on the same database are one registry. It must match
            [a-z0-9_]+ and be at most 32 bytes.
        tombstone_retention_seconds (int):
            Seconds a deleted collection's records are kept before its purge
            starts, on the database clock. It must exceed, by orders of
            magnitude, the longest a write to the backend can be in flight and
            the delay before the store's reads reflect a write.
        purge_retry_backoff_seconds (int):
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
            "The name the registry's rows are kept under: registry objects with "
            "the same name on the same database are one registry. It must match "
            "[a-z0-9_]+ and be at most 32 bytes"
        ),
    )
    tombstone_retention_seconds: int = Field(
        ...,
        ge=0,
        description=(
            "Seconds a deleted collection's records are kept before its purge "
            "starts, on the database clock. It must exceed, by orders of "
            "magnitude, the longest a write to the backend can be in flight and "
            "the delay before the store's reads reflect a write"
        ),
    )
    purge_retry_backoff_seconds: int = Field(
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


class SQLAlchemyVectorStoreCollectionRegistry(VectorStoreCollectionRegistry):
    """
    The registry of one vector store's collections.

    Registries of different vector stores share the tables, keyed by vector
    store name. After `_MAX_FAILED_PURGE_ROUNDS` consecutive failed rounds, a
    tombstone is dead-lettered: kept, its incarnation not minted again, no
    longer claimed, and reported by an error log. Setting its
    `failed_rounds` back to 0 returns it to the purge.
    """

    def __init__(self, params: SQLAlchemyVectorStoreCollectionRegistryParams) -> None:
        """Initialize with the provided parameters."""
        self._engine = params.engine
        self._is_sqlite = params.engine.dialect.name == "sqlite"
        self._vector_store_name = params.vector_store_name
        self._tombstone_retention = timedelta(
            seconds=params.tombstone_retention_seconds
        )
        self._purge_retry_backoff = timedelta(
            seconds=params.purge_retry_backoff_seconds
        )
        self._max_purge_retry_backoff = timedelta(
            seconds=params.max_purge_retry_backoff_seconds
        )

    @override
    async def startup(self) -> None:
        async with self._engine.begin() as connection:
            await connection.run_sync(BaseCollectionRegistry.metadata.create_all)

    @override
    async def register(
        self, namespace: str, name: str, config: VectorStoreCollectionConfig
    ) -> UUID:
        # The primary key arbitrates the (namespace, name) across processes;
        # the incarnation's unique constraint and the in-transaction queue
        # check reject an incarnation that is live or still awaiting purge.
        attempts = 0
        while True:
            incarnation = uuid4()
            try:
                await self._insert(namespace, name, incarnation, config)
            except _RegistryInsertRejectedError as err:
                attempts += 1
                if attempts >= _MAX_MINT_ATTEMPTS:
                    raise VectorStoreAttemptsExhaustedError(
                        f"Creating collection ({namespace!r}, {name!r}) made no "
                        f"progress after {_MAX_MINT_ATTEMPTS} attempts"
                    ) from err
                continue
            return incarnation

    async def _insert(
        self,
        namespace: str,
        name: str,
        incarnation: UUID,
        config: VectorStoreCollectionConfig,
    ) -> None:
        # The queue check runs after the insert, so a concurrent deletion
        # queuing a colliding incarnation, which the insert waited on, is
        # visible to it; the locking read sees the latest committed state.
        try:
            async with self._engine.begin() as connection:
                await connection.execute(
                    insert(CollectionRow).values(
                        vector_store_name=self._vector_store_name,
                        namespace=namespace,
                        name=name,
                        incarnation=incarnation,
                        config=config.model_dump(mode="json"),
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
                    logger.warning(
                        "Incarnation %s minted for collection (%r, %r) "
                        "collides with garbage awaiting purge; re-minting",
                        incarnation,
                        namespace,
                        name,
                    )
                    raise _RegistryInsertRejectedError(str(incarnation))
        except IntegrityError as err:
            if await self.get(namespace, name) is not None:
                raise VectorStoreCollectionAlreadyExistsError(namespace, name) from err
            logger.warning(
                "Registry insert for collection (%r, %r) with incarnation %s "
                "failed and no row exists under the key; retrying with a fresh "
                "incarnation",
                namespace,
                name,
                incarnation,
            )
            raise _RegistryInsertRejectedError(str(incarnation)) from err

    @override
    async def get(self, namespace: str, name: str) -> RegisteredCollection | None:
        async with self._engine.connect() as connection:
            row = (
                await connection.execute(
                    select(CollectionRow.incarnation, CollectionRow.config).where(
                        CollectionRow.vector_store_name == self._vector_store_name,
                        CollectionRow.namespace == namespace,
                        CollectionRow.name == name,
                    )
                )
            ).one_or_none()
        if row is None:
            return None
        return RegisteredCollection(
            incarnation=row.incarnation,
            config=VectorStoreCollectionConfig.model_validate(row.config),
        )

    @override
    async def is_live(self, incarnation: UUID) -> bool:
        async with self._engine.connect() as connection:
            row = (
                await connection.execute(
                    select(CollectionRow.name).where(
                        CollectionRow.incarnation == incarnation
                    )
                )
            ).scalar_one_or_none()
        return row is not None

    @override
    async def unregister(self, namespace: str, name: str) -> None:
        # One transaction: the collection is unreachable once it commits,
        # and the queue row is its incarnation's tombstone. The DELETE goes
        # first and takes the row's write lock, so racing deleters serialize
        # on it and the loser deletes nothing.
        async with self._engine.begin() as connection:
            row = (
                await connection.execute(
                    delete(CollectionRow)
                    .where(
                        CollectionRow.vector_store_name == self._vector_store_name,
                        CollectionRow.namespace == namespace,
                        CollectionRow.name == name,
                    )
                    .returning(CollectionRow.incarnation, CollectionRow.config)
                )
            ).one_or_none()
            if row is None:
                return
            await connection.execute(
                insert(PurgeQueueRow).values(
                    incarnation=row.incarnation,
                    vector_store_name=self._vector_store_name,
                    namespace=namespace,
                    name=name,
                    config=row.config,
                    enqueued_at=func.now(),
                )
            )

    @override
    @asynccontextmanager
    async def claim_purgeable_incarnation(
        self,
    ) -> AsyncGenerator[PurgeClaim | None, None]:
        # The retention is applied when a claim is decided, on the database
        # clock, so a changed retention applies to every tombstone. The
        # claim is a row lock held for the body: on PostgreSQL a concurrent
        # purger skips the locked row; on SQLite two purgers may run the
        # same round. A round that raises rolls back, then counts against
        # the tombstone in a transaction of its own.
        claimed: UUID | None = None
        try:
            async with self._engine.begin() as connection:
                row = (
                    await connection.execute(
                        select(
                            PurgeQueueRow.incarnation,
                            PurgeQueueRow.namespace,
                            PurgeQueueRow.config,
                            PurgeQueueRow.failed_rounds,
                        )
                        .where(
                            PurgeQueueRow.vector_store_name == self._vector_store_name,
                            PurgeQueueRow.enqueued_at <= self._retention_cutoff(),
                            PurgeQueueRow.failed_rounds < _MAX_FAILED_PURGE_ROUNDS,
                            or_(
                                PurgeQueueRow.failed_rounds == 0,
                                PurgeQueueRow.last_failed_at <= self._backoff_cutoff(),
                            ),
                        )
                        .order_by(PurgeQueueRow.enqueued_at)
                        .limit(1)
                        .with_for_update(skip_locked=True)
                    )
                ).one_or_none()
                if row is None:
                    yield None
                    return
                claimed = row.incarnation
                claim = PurgeClaim(
                    incarnation=row.incarnation,
                    namespace=row.namespace,
                    config=VectorStoreCollectionConfig.model_validate(row.config),
                )
                yield claim
                await self._record_round(connection, claim, row.failed_rounds)
        except Exception as error:
            if claimed is not None:
                await self._count_failed_round(claimed, error)
            raise

    async def _record_round(
        self, connection: AsyncConnection, claim: PurgeClaim, failed_rounds: int
    ) -> None:
        """Record a purge round's outcome, in the claim's transaction.

        A round that found no records removes the tombstone; one that found
        records resets its count of failed rounds.
        """
        if claim.any_records_found is None:
            raise RuntimeError(
                f"Purge round for incarnation {claim.incarnation} ended "
                "without setting any_records_found"
            )
        if not claim.any_records_found:
            await connection.execute(
                delete(PurgeQueueRow).where(
                    PurgeQueueRow.incarnation == claim.incarnation
                )
            )
        elif failed_rounds:
            await connection.execute(
                update(PurgeQueueRow)
                .where(PurgeQueueRow.incarnation == claim.incarnation)
                .values(failed_rounds=0)
            )

    async def _count_failed_round(self, incarnation: UUID, error: Exception) -> None:
        """Count a raised purge round against a tombstone, and report its dead-lettering."""
        async with self._engine.begin() as connection:
            failed_rounds = (
                await connection.execute(
                    update(PurgeQueueRow)
                    .where(PurgeQueueRow.incarnation == incarnation)
                    .values(
                        failed_rounds=PurgeQueueRow.failed_rounds + 1,
                        last_failed_at=func.now(),
                    )
                    .returning(PurgeQueueRow.failed_rounds)
                )
            ).scalar_one_or_none()
        if failed_rounds == _MAX_FAILED_PURGE_ROUNDS:
            logger.error(
                "Purge of incarnation %s failed %d rounds in a row and is "
                "dead-lettered: its records stay and it is no longer claimed. "
                "Last error: %r. Set its failed_rounds to 0 in %s to retry it.",
                incarnation,
                failed_rounds,
                error,
                PurgeQueueRow.__tablename__,
            )

    def _backoff_cutoff(self) -> ColumnElement:
        """The database clock's now, less each queue row's backoff, computed by the database."""
        # 1 << (f - 1) is 2 ** (f - 1) on both dialects.
        doublings = literal(1, Integer).op("<<")(PurgeQueueRow.failed_rounds - 1)
        if self._is_sqlite:
            seconds = func.min(
                int(self._purge_retry_backoff.total_seconds()) * doublings,
                int(self._max_purge_retry_backoff.total_seconds()),
            )
            return func.datetime(
                "now",
                func.printf("-%d seconds", seconds),
                type_=DateTime(timezone=True),
            )
        return func.now() - func.least(
            bindparam("retry_backoff", self._purge_retry_backoff, type_=Interval)
            * doublings,
            bindparam(
                "max_retry_backoff", self._max_purge_retry_backoff, type_=Interval
            ),
        )

    def _retention_cutoff(self) -> ColumnElement:
        """The database clock's now, less the retention, computed by the database."""
        if self._is_sqlite:
            # datetime() emits the form CURRENT_TIMESTAMP stamps with, so
            # the stamps and the cutoff compare as text in time order.
            seconds = int(self._tombstone_retention.total_seconds())
            return func.datetime(
                "now",
                bindparam("retention_modifier", f"-{seconds} seconds"),
                type_=DateTime(timezone=True),
            )
        return func.now() - bindparam(
            "retention", self._tombstone_retention, type_=Interval
        )
