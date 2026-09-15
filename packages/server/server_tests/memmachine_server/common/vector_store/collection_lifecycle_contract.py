"""
The collection lifecycle contract of a vector store that binds handles to an incarnation.

A store's test module mixes `CollectionLifecycleContract` into a test class
and supplies the `store` fixture, a started store, and
`count_stored(store, namespace, config)`, the number of records the backend
physically holds in the physical collection for `(namespace, config)`, which
the purge tests read past the store's own API. The backend may persist
across tests (a shared Qdrant server): every test starts by deleting the
collections it uses, and the purge tests count relative to a drained
baseline.

The contract: a handle is bound to one incarnation and raises once that
incarnation is deleted; a collection re-created under a deleted pair starts
empty; deletion is a registry write and `purge_deleted_collections` reclaims
the records afterward, one physical collection's worth at a time, leaving
every other logical collection alone.
"""

import math
from uuid import uuid4

import pytest

from memmachine_server.common.vector_store import (
    Record,
    VectorStoreCollectionConfig,
    VectorStoreCollectionHandleStaleError,
)

NAMESPACE = "lifecycle_ns"
NAME = "lifecycle"
DIMENSIONS = 3
CONFIG = VectorStoreCollectionConfig(vector_dimensions=DIMENSIONS)
OTHER_CONFIG = VectorStoreCollectionConfig(
    vector_dimensions=DIMENSIONS, indexed_properties_schema={"name": str}
)


def _unit(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(x * x for x in vector))
    return [x / magnitude for x in vector]


def _records(count: int) -> list[Record]:
    return [
        Record(
            uuid=uuid4(),
            vector=_unit([1.0] + [0.01 * index] + [0.0] * (DIMENSIONS - 2)),
        )
        for index in range(count)
    ]


async def _fresh(store, namespace: str, name: str, config=CONFIG):
    """A new collection under the pair, whatever a persisted backend held for it."""
    await store.delete_collection(namespace=namespace, name=name)
    await store.create_collection(namespace=namespace, name=name, config=config)
    collection = await store.get_collection(namespace=namespace, name=name)
    assert collection is not None
    return collection


class CollectionLifecycleContract:
    """Mixed into a store's test class, with its `store` fixture and `count_stored`."""

    @staticmethod
    async def count_stored(store, namespace: str, config) -> int:
        """Records the backend holds in the physical collection, live or dead."""
        raise NotImplementedError

    async def _drained_count(self, store, namespace: str, config) -> int:
        """`count_stored` once nothing deleted is left to reclaim."""
        while await store.purge_deleted_collections():
            pass
        return await self.count_stored(store, namespace, config)

    @pytest.mark.asyncio
    async def test_a_handle_is_stale_once_its_collection_is_deleted(self, store):
        collection = await _fresh(store, NAMESPACE, NAME)
        record = _records(1)[0]
        await collection.upsert(records=[record])

        await store.delete_collection(namespace=NAMESPACE, name=NAME)

        assert await store.get_collection(namespace=NAMESPACE, name=NAME) is None
        with pytest.raises(VectorStoreCollectionHandleStaleError, match=NAME):
            await collection.upsert(records=[record])
        with pytest.raises(VectorStoreCollectionHandleStaleError, match=NAME):
            await collection.query(query_vectors=[record.vector], limit=5)
        with pytest.raises(VectorStoreCollectionHandleStaleError, match=NAME):
            await collection.delete(record_uuids=[record.uuid])

    @pytest.mark.asyncio
    async def test_a_recreated_collection_starts_empty_and_the_old_handle_stays_stale(
        self, store
    ):
        old = await _fresh(store, NAMESPACE, NAME)
        old_record, new_record = _records(2)
        await old.upsert(records=[old_record])

        await store.delete_collection(namespace=NAMESPACE, name=NAME)
        new = await _fresh(store, NAMESPACE, NAME)

        [before] = await new.query(query_vectors=[old_record.vector], limit=5)
        assert before.matches == []

        await new.upsert(records=[new_record])
        [after] = await new.query(query_vectors=[new_record.vector], limit=5)
        assert {match.record_uuid for match in after.matches} == {new_record.uuid}

        # The old life's handle cannot reach the new life's records.
        with pytest.raises(VectorStoreCollectionHandleStaleError):
            await old.query(query_vectors=[new_record.vector], limit=5)
        with pytest.raises(VectorStoreCollectionHandleStaleError):
            await old.delete(record_uuids=[new_record.uuid])
        [still] = await new.query(query_vectors=[new_record.vector], limit=5)
        assert {match.record_uuid for match in still.matches} == {new_record.uuid}

        await store.delete_collection(namespace=NAMESPACE, name=NAME)

    @pytest.mark.asyncio
    async def test_deleting_twice_and_deleting_nothing_are_no_ops(self, store):
        await _fresh(store, NAMESPACE, NAME)
        await store.delete_collection(namespace=NAMESPACE, name=NAME)
        await store.delete_collection(namespace=NAMESPACE, name=NAME)
        assert await store.get_collection(namespace=NAMESPACE, name=NAME) is None
        await store.delete_collection(namespace=NAMESPACE, name=f"{NAME}_never")

    @pytest.mark.asyncio
    async def test_purge_reclaims_what_deletion_deferred(self, store):
        collection = await _fresh(store, NAMESPACE, NAME)
        baseline = await self._drained_count(store, NAMESPACE, CONFIG)
        records = _records(5)
        await collection.upsert(records=records)
        assert await self.count_stored(store, NAMESPACE, CONFIG) == baseline + 5

        await store.delete_collection(namespace=NAMESPACE, name=NAME)
        # Unreachable at once; the records may still be held.
        assert await store.get_collection(namespace=NAMESPACE, name=NAME) is None

        assert await self._drained_count(store, NAMESPACE, CONFIG) == baseline
        # Nothing left to claim.
        assert await store.purge_deleted_collections() is False

    @pytest.mark.asyncio
    async def test_purge_leaves_a_live_collection_of_the_same_config_alone(self, store):
        """Two collections of one config share a physical collection; the purge is scoped."""
        live_name = f"{NAME}_live"
        live = await _fresh(store, NAMESPACE, live_name)
        dead = await _fresh(store, NAMESPACE, NAME)
        baseline = await self._drained_count(store, NAMESPACE, CONFIG)
        kept = _records(3)
        gone = _records(3)
        await live.upsert(records=kept)
        await dead.upsert(records=gone)

        await store.delete_collection(namespace=NAMESPACE, name=NAME)

        assert await self._drained_count(store, NAMESPACE, CONFIG) == baseline + 3
        [result] = await live.query(query_vectors=[kept[0].vector], limit=10)
        assert {match.record_uuid for match in result.matches} == {
            record.uuid for record in kept
        }
        await store.delete_collection(namespace=NAMESPACE, name=live_name)
        assert await self._drained_count(store, NAMESPACE, CONFIG) == baseline

    @pytest.mark.asyncio
    async def test_purge_leaves_a_collection_of_another_config_alone(self, store):
        other_name = f"{NAME}_other"
        other = await _fresh(store, NAMESPACE, other_name, OTHER_CONFIG)
        dead = await _fresh(store, NAMESPACE, NAME)
        other_baseline = await self._drained_count(store, NAMESPACE, OTHER_CONFIG)
        baseline = await self.count_stored(store, NAMESPACE, CONFIG)
        await other.upsert(records=_records(2))
        await dead.upsert(records=_records(3))

        await store.delete_collection(namespace=NAMESPACE, name=NAME)

        assert await self._drained_count(store, NAMESPACE, CONFIG) == baseline
        assert (
            await self.count_stored(store, NAMESPACE, OTHER_CONFIG)
            == other_baseline + 2
        )
        await store.delete_collection(namespace=NAMESPACE, name=other_name)
        assert (
            await self._drained_count(store, NAMESPACE, OTHER_CONFIG) == other_baseline
        )

    @pytest.mark.asyncio
    async def test_recreating_under_another_config_while_the_predecessor_awaits_purge(
        self, store
    ):
        """The purge reclaims from the physical collection the dead incarnation lived in."""
        old = await _fresh(store, NAMESPACE, NAME)
        old_baseline = await self._drained_count(store, NAMESPACE, CONFIG)
        new_baseline = await self.count_stored(store, NAMESPACE, OTHER_CONFIG)
        await old.upsert(records=_records(3))

        await store.delete_collection(namespace=NAMESPACE, name=NAME)
        new = await _fresh(store, NAMESPACE, NAME, OTHER_CONFIG)
        await new.upsert(records=_records(2))
        with pytest.raises(VectorStoreCollectionHandleStaleError):
            await old.query(query_vectors=[_records(1)[0].vector], limit=5)

        assert await self._drained_count(store, NAMESPACE, CONFIG) == old_baseline
        assert (
            await self.count_stored(store, NAMESPACE, OTHER_CONFIG) == new_baseline + 2
        )
        await store.delete_collection(namespace=NAMESPACE, name=NAME)
        assert await self._drained_count(store, NAMESPACE, OTHER_CONFIG) == new_baseline

    @pytest.mark.asyncio
    async def test_one_sweep_serves_every_namespace(self, store):
        other_namespace = f"{NAMESPACE}_two"
        first = await _fresh(store, NAMESPACE, NAME)
        second = await _fresh(store, other_namespace, NAME)
        first_baseline = await self._drained_count(store, NAMESPACE, CONFIG)
        second_baseline = await self.count_stored(store, other_namespace, CONFIG)
        await first.upsert(records=_records(2))
        await second.upsert(records=_records(2))

        await store.delete_collection(namespace=NAMESPACE, name=NAME)
        await store.delete_collection(namespace=other_namespace, name=NAME)

        assert await self._drained_count(store, NAMESPACE, CONFIG) == first_baseline
        assert (
            await self.count_stored(store, other_namespace, CONFIG) == second_baseline
        )

    @pytest.mark.asyncio
    async def test_the_same_record_uuid_in_two_collections_stays_two_records(
        self, store
    ):
        """Point ids are per incarnation: a co-tenant's upsert never replaces a point."""
        other_name = f"{NAME}_other"
        first = await _fresh(store, NAMESPACE, NAME)
        second = await _fresh(store, NAMESPACE, other_name)
        shared_uuid = uuid4()
        first_vector = _unit([1.0, 0.0, 0.0])
        second_vector = _unit([0.0, 1.0, 0.0])
        await first.upsert(records=[Record(uuid=shared_uuid, vector=first_vector)])
        await second.upsert(records=[Record(uuid=shared_uuid, vector=second_vector)])

        [first_result] = await first.query(query_vectors=[first_vector], limit=5)
        [second_result] = await second.query(query_vectors=[second_vector], limit=5)
        assert [m.record_uuid for m in first_result.matches] == [shared_uuid]
        assert first_result.matches[0].cosine_similarity == pytest.approx(1.0)
        assert [m.record_uuid for m in second_result.matches] == [shared_uuid]
        assert second_result.matches[0].cosine_similarity == pytest.approx(1.0)

        await second.delete(record_uuids=[shared_uuid])
        [after] = await first.query(query_vectors=[first_vector], limit=5)
        assert [m.record_uuid for m in after.matches] == [shared_uuid]

        await store.delete_collection(namespace=NAMESPACE, name=NAME)
        await store.delete_collection(namespace=NAMESPACE, name=other_name)
