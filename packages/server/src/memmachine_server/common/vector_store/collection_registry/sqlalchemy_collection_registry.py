"""
A collection registry in a relational database, through SQLAlchemy.

Qdrant and Milvus have no transactions, unique constraints or conditional
writes, so a catalog kept inside them cannot arbitrate two processes
creating, deleting or reclaiming the same logical collection. This
registry lives in a relational database instead: a table pair per vector
store, named by the store's name, with a row per live logical collection
keyed by namespace and name, whose incarnation is the value
every point of that life carries, and a purge queue of dead incarnations
claimed oldest-first. Creation is an insert the primary key arbitrates,
deletion is one transaction, and a purge claim is a row lock the database
hands to one purger at a time.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import override
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Index,
    MetaData,
    String,
    Table,
    Uuid,
    delete,
    func,
    insert,
    or_,
    select,
    update,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

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
    CollectionRegistry,
    PurgeClaim,
    RegisteredCollection,
)

logger = logging.getLogger(__name__)

_MAX_MINT_ATTEMPTS = 10


class _RegistryInsertRejectedError(Exception):
    """A registry insert was rejected; retry with a fresh incarnation."""


class SQLAlchemyCollectionRegistry(CollectionRegistry):
    """The registry of one vector store's collections, in its own table pair.

    `vector_store_name` names the vector store, and so the vector database
    deployment it reaches, whose collections the registry holds: its tables
    are `collection_registry_{vector_store_name}_ct` and `..._gc`, so the
    registries of different vector stores share a database without sharing
    a row, and registry objects under one name are one registry. It must
    match `[a-z0-9_]+` and be at most 32 bytes. `tombstone_retention` is
    how long a dead incarnation's entry outlives the first purge round that
    found nothing; it must exceed, by orders of magnitude, the longest a
    write to the backend can be in flight. The retention is measured on the
    database clock.
    """

    def __init__(
        self,
        *,
        engine: AsyncEngine,
        vector_store_name: str,
        tombstone_retention: timedelta,
    ) -> None:
        """Bind to the tables of the registry of the vector store `vector_store_name`."""
        if not validate_identifier(vector_store_name):
            raise ValueError(
                f"Vector store name {vector_store_name!r} must match [a-z0-9_]+ "
                "and be at most 32 bytes"
            )
        self._engine = engine
        self._tombstone_retention = tombstone_retention
        table_prefix = f"collection_registry_{vector_store_name}"
        metadata = MetaData()
        self._collections = Table(
            f"{table_prefix}_ct",
            metadata,
            Column("namespace", String(_IDENTIFIER_MAX_BYTES), primary_key=True),
            Column("name", String(_IDENTIFIER_MAX_BYTES), primary_key=True),
            Column("incarnation", Uuid, nullable=False, unique=True),
            # The configuration the collection was created with: the handle
            # is built from it, and open-or-create compares against it.
            Column("config_json", JSON, nullable=False),
        )
        self._purge_queue = Table(
            f"{table_prefix}_gc",
            metadata,
            Column("incarnation", Uuid, primary_key=True),
            Column("namespace", String(_IDENTIFIER_MAX_BYTES), nullable=False),
            Column("name", String(_IDENTIFIER_MAX_BYTES), nullable=False),
            # The configuration names the native collection the points are in.
            Column("config_json", JSON, nullable=False),
            Column("enqueued_at", DateTime(timezone=True), nullable=False),
            # When a purge round last found nothing under the incarnation;
            # cleared by a round that finds something. The entry is removed
            # by a round that finds nothing a retention after this.
            Column("clean_at", DateTime(timezone=True), nullable=True),
            Index(f"{table_prefix}_gc__ea", "enqueued_at"),
        )
        self._metadata = metadata

    @override
    async def startup(self) -> None:
        async with self._engine.begin() as connection:
            await connection.run_sync(self._metadata.create_all)

    @override
    async def create(
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
        # The queue check runs after the insert so that a concurrent
        # deletion moving a colliding row to the queue, which our insert
        # waited on, is already visible; after the check, no new queue
        # entry for this incarnation can appear before we commit, because
        # the only registry row carrying it is ours, uncommitted. The
        # locking read sees latest-committed state even on dialects whose
        # plain reads serve transaction-start snapshots.
        try:
            async with self._engine.begin() as connection:
                await connection.execute(
                    insert(self._collections).values(
                        namespace=namespace,
                        name=name,
                        incarnation=incarnation,
                        config_json=config.model_dump(mode="json"),
                    )
                )
                queued = (
                    await connection.execute(
                        select(self._purge_queue.c.incarnation)
                        .where(self._purge_queue.c.incarnation == incarnation)
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
                    select(
                        self._collections.c.incarnation,
                        self._collections.c.config_json,
                    ).where(
                        self._collections.c.namespace == namespace,
                        self._collections.c.name == name,
                    )
                )
            ).one_or_none()
        if row is None:
            return None
        return RegisteredCollection(
            incarnation=row.incarnation,
            config=VectorStoreCollectionConfig.model_validate(row.config_json),
        )

    @override
    async def is_live(self, incarnation: UUID) -> bool:
        async with self._engine.connect() as connection:
            row = (
                await connection.execute(
                    select(self._collections.c.name).where(
                        self._collections.c.incarnation == incarnation
                    )
                )
            ).scalar_one_or_none()
        return row is not None

    @override
    async def delete(self, namespace: str, name: str) -> None:
        # One transaction: the collection is unreachable as soon as it
        # commits, and the queue row is the incarnation's tombstone. The
        # segment store pins the row before its queue insert with the
        # fence its writes use; the registry has no such fence to reuse,
        # so the DELETE goes first: it takes the row's write lock, racing
        # deleters serialize on it and the loser deletes nothing, on
        # PostgreSQL and on SQLite alike, and RETURNING resolves the
        # incarnation in the same round trip.
        async with self._engine.begin() as connection:
            row = (
                await connection.execute(
                    delete(self._collections)
                    .where(
                        self._collections.c.namespace == namespace,
                        self._collections.c.name == name,
                    )
                    .returning(
                        self._collections.c.incarnation,
                        self._collections.c.config_json,
                    )
                )
            ).one_or_none()
            if row is None:
                return
            await connection.execute(
                insert(self._purge_queue).values(
                    incarnation=row.incarnation,
                    namespace=namespace,
                    name=name,
                    config_json=row.config_json,
                    enqueued_at=func.now(),
                )
            )

    @override
    @asynccontextmanager
    async def claim_oldest(self) -> AsyncIterator[PurgeClaim | None]:
        # The claim is a row lock held for the body: on PostgreSQL a
        # concurrent purger skips the locked entry and takes the next; on
        # SQLite the writers serialize at the end of the round, so a doubly
        # claimed entry costs a repeated, idempotent round and never a
        # missed one. Entries order by their enqueue stamp on the database
        # clock, so entries from every server order on one clock; entries
        # stamped in the same tick are unordered among themselves. The
        # retention is measured on the same clock.
        async with self._engine.begin() as connection:
            database_now = (
                await connection.execute(
                    select(func.now(type_=DateTime(timezone=True)))
                )
            ).scalar_one()
            row = (
                await connection.execute(
                    select(
                        self._purge_queue.c.incarnation,
                        self._purge_queue.c.namespace,
                        self._purge_queue.c.name,
                        self._purge_queue.c.config_json,
                        self._purge_queue.c.clean_at,
                    )
                    .where(
                        or_(
                            self._purge_queue.c.clean_at.is_(None),
                            self._purge_queue.c.clean_at
                            <= database_now - self._tombstone_retention,
                        ),
                    )
                    .order_by(self._purge_queue.c.enqueued_at)
                    .limit(1)
                    .with_for_update(skip_locked=True)
                )
            ).one_or_none()
            if row is None:
                yield None
                return
            claim = PurgeClaim(
                incarnation=row.incarnation,
                namespace=row.namespace,
                name=row.name,
                config=VectorStoreCollectionConfig.model_validate(row.config_json),
                clean_at=row.clean_at,
            )
            yield claim
            if claim.found is None:
                raise RuntimeError(
                    f"Purge round for incarnation {claim.incarnation} ended "
                    "without reporting what it found"
                )
            # A round that found nothing applies its outcome only to the
            # entry as the claim read it. Under the row lock that is always
            # the entry; on SQLite a doubly claimed entry is possible, and a
            # round that found nothing must neither stamp over nor remove
            # an entry another round has since un-stamped for the points it
            # found. Un-stamping restarts the protocol and is always safe.
            entry = self._purge_queue.c.incarnation == claim.incarnation
            if claim.found:
                await connection.execute(
                    update(self._purge_queue).where(entry).values(clean_at=None)
                )
            elif claim.clean_at is None:
                await connection.execute(
                    update(self._purge_queue)
                    .where(entry, self._purge_queue.c.clean_at.is_(None))
                    .values(clean_at=func.now())
                )
            else:
                await connection.execute(
                    delete(self._purge_queue).where(
                        entry, self._purge_queue.c.clean_at == claim.clean_at
                    )
                )
