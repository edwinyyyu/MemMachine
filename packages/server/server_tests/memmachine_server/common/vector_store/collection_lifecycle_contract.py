"""
The collection lifecycle contract the registry-backed vector stores satisfy.

Each store's test module mixes `CollectionLifecycleContract` into a test
class and supplies the `store` fixture, a started store, and
`count_stored(store, namespace, config)`, the number of records the backend
physically holds in the native collection those name, which the purge tests
read past the store's own API. The backend may persist across tests (a
shared Qdrant or Milvus server): every test starts by deleting the
collections it uses, and the purge tests count relative to a drained
baseline.

A collection is identified to callers by its (namespace, name) and inside
the store by an incarnation minted per life of the pair. The contract: a
handle is bound to one incarnation and raises once that incarnation is
deleted; a collection re-created under a deleted name starts empty;
deletion is a registry write and `purge_deleted_collections` reclaims the
records afterward; a write can land under an incarnation that died while
it was in flight, and the operation then raises instead of reporting
success, with the incarnation's tombstone having the records reclaimed by
a later purge round.
"""

import math
from uuid import uuid4

import pytest

from memmachine_server.common.vector_store import (
    Record,
    VectorStoreCollectionConfig,
    VectorStoreCollectionHandleStaleError,
)

LIFECYCLE_NAMESPACE = "lifecycle_ns"
LIFECYCLE_NAME = "lifecycle"
LIFECYCLE_CONFIG = VectorStoreCollectionConfig(vector_dimensions=3)


def _unit(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(x * x for x in vector))
    return [x / magnitude for x in vector]


def _records(count: int) -> list[Record]:
    return [
        Record(uuid=uuid4(), vector=_unit([1.0, 0.01 * index, 0.0]))
        for index in range(count)
    ]


async def _fresh(store, name: str):
    """A new collection under `name`, whatever a persisted backend held for it."""
    await store.delete_collection(namespace=LIFECYCLE_NAMESPACE, name=name)
    await store.create_collection(
        namespace=LIFECYCLE_NAMESPACE, name=name, config=LIFECYCLE_CONFIG
    )
    collection = await store.open_collection(namespace=LIFECYCLE_NAMESPACE, name=name)
    assert collection is not None
    return collection


def _uuids(result) -> set:
    return {match.record.uuid for match in result.matches}


class CollectionLifecycleContract:
    """Mixed into a store's test class, with its `store` fixture and `count_stored`."""

    @staticmethod
    async def count_stored(store, namespace: str, config) -> int:
        """Records the backend physically holds in the native collection, live or dead."""
        raise NotImplementedError

    async def _drained_count(self, store) -> int:
        """`count_stored` once nothing deleted is left to reclaim."""
        while await store.purge_deleted_collections():
            pass
        return await self.count_stored(store, LIFECYCLE_NAMESPACE, LIFECYCLE_CONFIG)

    @pytest.mark.asyncio
    async def test_a_handle_is_stale_once_its_collection_is_deleted(self, store):
        collection = await _fresh(store, LIFECYCLE_NAME)
        record = _records(1)[0]
        await collection.upsert(records=[record])

        await store.delete_collection(
            namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME
        )

        assert (
            await store.open_collection(
                namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME
            )
            is None
        )
        with pytest.raises(VectorStoreCollectionHandleStaleError, match=LIFECYCLE_NAME):
            await collection.upsert(records=[record])
        with pytest.raises(VectorStoreCollectionHandleStaleError, match=LIFECYCLE_NAME):
            await collection.query(query_vectors=[record.vector], limit=5)
        with pytest.raises(VectorStoreCollectionHandleStaleError, match=LIFECYCLE_NAME):
            await collection.get(record_uuids=[record.uuid])
        with pytest.raises(VectorStoreCollectionHandleStaleError, match=LIFECYCLE_NAME):
            await collection.delete(record_uuids=[record.uuid])

    @pytest.mark.asyncio
    async def test_a_recreated_collection_starts_empty_and_the_old_handle_stays_stale(
        self, store
    ):
        old = await _fresh(store, LIFECYCLE_NAME)
        old_record, new_record = _records(2)
        await old.upsert(records=[old_record])

        await store.delete_collection(
            namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME
        )
        new = await _fresh(store, LIFECYCLE_NAME)

        [before] = await new.query(query_vectors=[old_record.vector], limit=5)
        assert before.matches == []
        assert await new.get(record_uuids=[old_record.uuid]) == []

        await new.upsert(records=[new_record])
        [after] = await new.query(query_vectors=[new_record.vector], limit=5)
        assert _uuids(after) == {new_record.uuid}

        # The old life's handle cannot reach the new life's records.
        with pytest.raises(VectorStoreCollectionHandleStaleError):
            await old.query(query_vectors=[new_record.vector], limit=5)
        with pytest.raises(VectorStoreCollectionHandleStaleError):
            await old.delete(record_uuids=[new_record.uuid])
        [still] = await new.query(query_vectors=[new_record.vector], limit=5)
        assert _uuids(still) == {new_record.uuid}

        await store.delete_collection(
            namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME
        )

    @pytest.mark.asyncio
    async def test_open_or_create_adopts_the_live_incarnation(self, store):
        """Opening an existing collection binds to its life; creating one mints a new life."""
        await store.delete_collection(
            namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME
        )
        first = await store.open_or_create_collection(
            namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME, config=LIFECYCLE_CONFIG
        )
        second = await store.open_or_create_collection(
            namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME, config=LIFECYCLE_CONFIG
        )
        record = _records(1)[0]
        await first.upsert(records=[record])
        [seen] = await second.query(query_vectors=[record.vector], limit=5)
        assert _uuids(seen) == {record.uuid}

        await store.delete_collection(
            namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME
        )
        third = await store.open_or_create_collection(
            namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME, config=LIFECYCLE_CONFIG
        )
        [empty] = await third.query(query_vectors=[record.vector], limit=5)
        assert empty.matches == []
        with pytest.raises(VectorStoreCollectionHandleStaleError):
            await first.query(query_vectors=[record.vector], limit=5)
        await store.delete_collection(
            namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME
        )

    @pytest.mark.asyncio
    async def test_deleting_twice_and_deleting_nothing_are_no_ops(self, store):
        await _fresh(store, LIFECYCLE_NAME)
        await store.delete_collection(
            namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME
        )
        await store.delete_collection(
            namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME
        )
        assert (
            await store.open_collection(
                namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME
            )
            is None
        )
        await store.delete_collection(
            namespace=LIFECYCLE_NAMESPACE, name=f"{LIFECYCLE_NAME}_never_created"
        )

    @pytest.mark.asyncio
    async def test_purge_reclaims_what_deletion_deferred(self, store):
        collection = await _fresh(store, LIFECYCLE_NAME)
        baseline = await self._drained_count(store)
        records = _records(5)
        await collection.upsert(records=records)
        assert (
            await self.count_stored(store, LIFECYCLE_NAMESPACE, LIFECYCLE_CONFIG)
            == baseline + 5
        )

        await store.delete_collection(
            namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME
        )
        # Unreachable at once; the records may still be held.
        assert (
            await store.open_collection(
                namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME
            )
            is None
        )

        assert await self._drained_count(store) == baseline
        # Nothing left to claim.
        assert await store.purge_deleted_collections() is False

    @pytest.mark.asyncio
    async def test_purge_leaves_a_live_collection_alone(self, store):
        live_name = f"{LIFECYCLE_NAME}_live"
        live = await _fresh(store, live_name)
        dead = await _fresh(store, LIFECYCLE_NAME)
        baseline = await self._drained_count(store)
        kept = _records(3)
        gone = _records(3)
        await live.upsert(records=kept)
        await dead.upsert(records=gone)

        await store.delete_collection(
            namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME
        )

        assert await self._drained_count(store) == baseline + 3
        [result] = await live.query(query_vectors=[kept[0].vector], limit=10)
        assert _uuids(result) == {record.uuid for record in kept}
        await store.delete_collection(namespace=LIFECYCLE_NAMESPACE, name=live_name)
        assert await self._drained_count(store) == baseline

    @pytest.mark.asyncio
    async def test_a_write_landing_under_a_dead_incarnation_raises_and_is_reclaimed(
        self, store
    ):
        collection = await _fresh(store, LIFECYCLE_NAME)
        baseline = await self._drained_count(store)
        records = _records(2)

        # The collection dies between the handle's check and its write.
        is_live = collection._is_live

        async def deleted_once_checked(incarnation) -> bool:
            live = await is_live(incarnation)
            if live:
                await store.delete_collection(
                    namespace=LIFECYCLE_NAMESPACE, name=LIFECYCLE_NAME
                )
            return live

        collection._is_live = deleted_once_checked

        with pytest.raises(VectorStoreCollectionHandleStaleError, match=LIFECYCLE_NAME):
            await collection.upsert(records=records)

        # The write landed, under an incarnation nothing can reach...
        assert (
            await self.count_stored(store, LIFECYCLE_NAMESPACE, LIFECYCLE_CONFIG)
            == baseline + 2
        )
        # ...and the tombstone's next round reclaims it.
        assert await store.purge_deleted_collections() is True
        assert await self._drained_count(store) == baseline

    @pytest.mark.asyncio
    async def test_a_trailing_newline_is_not_part_of_a_valid_identifier(self, store):
        for namespace, name in (
            (LIFECYCLE_NAMESPACE, f"{LIFECYCLE_NAME}\n"),
            (f"{LIFECYCLE_NAMESPACE}\n", LIFECYCLE_NAME),
        ):
            with pytest.raises(ValueError, match="must match"):
                await store.create_collection(
                    namespace=namespace, name=name, config=LIFECYCLE_CONFIG
                )
