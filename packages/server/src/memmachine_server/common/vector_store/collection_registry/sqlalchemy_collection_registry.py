"""
A collection registry in a relational database, through SQLAlchemy.

Qdrant and Milvus have no transactions, unique constraints or conditional
writes, so a catalog kept inside them cannot arbitrate two processes
creating, deleting or reclaiming the same logical collection. This
registry lives in the deployment's relational database instead: one table
pair per backend kind, shared by every store on that kind of backend,
with a row per live logical collection keyed by backend, namespace and
name, whose incarnation is the value every point of that life carries,
and a purge queue of dead incarnations claimed oldest-first. Creation is
an insert the primary key arbitrates, deletion is one transaction, and a
purge claim is a row lock the database hands to one purger at a time.
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
from memmachine_server.common.vector_store.utils import _IDENTIFIER_MAX_BYTES

from .collection_registry import CollectionRegistry, PurgeClaim, RegisteredCollection

logger = logging.getLogger(__name__)

_MAX_MINT_ATTEMPTS = 10
_BACKEND_ID_MAX_LENGTH = 255


class _RegistryInsertRejectedError(Exception):
    """A registry insert was rejected; retry with a fresh incarnation."""


class SQLAlchemyCollectionRegistry(CollectionRegistry):
    """The collection registry of one store, in a table pair shared by its backend kind.

    `table_prefix` names the backend kind: every store on that kind of
    backend shares `{prefix}_ct` and `{prefix}_gc`, and `backend`, the id
    of the configured backend the store is on, is part of every key, so
    stores on different backends keep apart in one database.
    `tombstone_retention` is how long a dead incarnation's entry outlives
    the first purge round that found nothing; it must exceed, by orders
    of magnitude, the longest a write to the backend can be in flight.
    The retention is measured on the database clock.
    """

    def __init__(
        self,
        *,
        engine: AsyncEngine,
        table_prefix: str,
        backend: str,
        tombstone_retention: timedelta,
    ) -> None:
        """Bind to the registry tables of one backend kind, for one backend."""
        self._engine = engine
        self._backend = backend
        self._tombstone_retention = tombstone_retention
        metadata = MetaData()
        self._collections = Table(
            f"{table_prefix}_ct",
            metadata,
            Column("backend", String(_BACKEND_ID_MAX_LENGTH), primary_key=True),
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
            Column("backend", String(_BACKEND_ID_MAX_LENGTH), nullable=False),
            Column("namespace", String(_IDENTIFIER_MAX_BYTES), nullable=False),
            Column("name", String(_IDENTIFIER_MAX_BYTES), nullable=False),
            # The configuration names the native collection the points are in.
            Column("config_json", JSON, nullable=False),
            Column("enqueued_at", DateTime(timezone=True), nullable=False),
            # When a purge round last found nothing under the incarnation;
            # cleared by a round that finds something. The entry is removed
            # by a round that finds nothing a retention after this.
            Column("clean_at", DateTime(timezone=True), nullable=True),
            Index(f"{table_prefix}_gc__ba_ea", "backend", "enqueued_at"),
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
                        f"Creating collection ({namespace!r}, {name!r}) on backend "
                        f"{self._backend!r} made no progress after "
                        f"{_MAX_MINT_ATTEMPTS} attempts"
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
                        backend=self._backend,
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
                        "Incarnation %s minted for collection (%r, %r) on backend %r "
                        "collides with garbage awaiting purge; re-minting",
                        incarnation,
                        namespace,
                        name,
                        self._backend,
                    )
                    raise _RegistryInsertRejectedError(str(incarnation))
        except IntegrityError as err:
            if await self.get(namespace, name) is not None:
                raise VectorStoreCollectionAlreadyExistsError(namespace, name) from err
            logger.warning(
                "Registry insert for collection (%r, %r) on backend %r with "
                "incarnation %s failed and no row exists under the key; "
                "retrying with a fresh incarnation",
                namespace,
                name,
                self._backend,
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
                        self._collections.c.backend == self._backend,
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
                        self._collections.c.backend == self._backend,
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
                    backend=self._backend,
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
        # missed one. The retention is measured on the database clock.
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
                        self._purge_queue.c.backend == self._backend,
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
                    f"Purge round for incarnation {claim.incarnation} on backend "
                    f"{self._backend!r} ended without reporting what it found"
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
