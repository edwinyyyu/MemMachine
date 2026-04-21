"""Tests for the :class:`AttributeMemory` orchestrator.

Uses a real :class:`SQLAlchemySemanticStorePartition` against SQLite,
an in-memory :class:`FakeVectorStoreCollection`, and scripted
:class:`FakeLanguageModel` / :class:`FakeEmbedder` / :class:`FakeReranker`
so every code path (ingest, retrieve, consolidate) can be driven
deterministically.
"""

from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine

from memmachine_server.common.filter.filter_parser import Comparison
from memmachine_server.semantic_memory.attribute_memory import (
    AttributeMemory,
    ClusteringConfig,
    SemanticAttribute,
)
from memmachine_server.semantic_memory.attribute_memory.data_types import (
    ClusterParams,
    Command,
    CommandType,
    Content,
    Event,
    MessageContext,
    Text,
)
from memmachine_server.semantic_memory.attribute_memory.semantic_store.sqlalchemy_semantic_store import (
    CategoryDefinition,
    PartitionSchema,
    SQLAlchemySemanticStore,
    SQLAlchemySemanticStoreParams,
    SQLAlchemySemanticStorePartition,
    TopicDefinition,
)

from .conftest import (
    FakeEmbedder,
    FakeLanguageModel,
    FakeReranker,
    FakeVectorStoreCollection,
)

PARTITION = "org_acme_42"


class SplitPerEvent:
    """Test splitter that deterministically breaks one cluster into singletons."""

    def __init__(self) -> None:
        self.calls = 0

    async def maybe_split_clusters(
        self,
        *,
        cluster_events,
        cluster_embeddings,
        state,
        reranker,
    ):
        del cluster_embeddings, reranker
        self.calls += 1
        split: list[tuple[str, list[Event]]] = []
        for cluster_id, events in cluster_events:
            for index, event in enumerate(events):
                child_id = cluster_id if index == 0 else f"{cluster_id}_split_{index}"
                split.append((child_id, [event]))
        return split, state


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def store(
    sqlite_engine: AsyncEngine,
) -> AsyncIterator[SQLAlchemySemanticStore]:
    s = SQLAlchemySemanticStore(SQLAlchemySemanticStoreParams(engine=sqlite_engine))
    await s.startup()
    yield s
    await s.shutdown()


@pytest_asyncio.fixture
async def partition(
    store: SQLAlchemySemanticStore,
) -> SQLAlchemySemanticStorePartition:
    return await store.open_or_create_partition(PARTITION)


def _schema(*topics: TopicDefinition) -> PartitionSchema:
    if not topics:
        topics = (
            TopicDefinition(
                name="Profile",
                description="User profile",
                categories=(
                    CategoryDefinition(name="food", description="food preferences"),
                    CategoryDefinition(name="music", description="music taste"),
                ),
            ),
        )
    return PartitionSchema(topics=topics)


def _config(**overrides) -> ClusteringConfig:
    # threshold=0 + FakeEmbedder's non-negative vectors → every event
    # clusters together (cosine similarity ≥ 0).
    base = ClusteringConfig(
        cluster_params=ClusterParams(similarity_threshold=0.0),
        trigger_messages=1,  # flush on every call by default
        trigger_age=None,
        idle_ttl=None,
        max_clusters_per_run=10,
        max_features_per_update=50,
        consolidation_threshold=0,  # disable auto-consolidate by default
    )
    return replace(base, **overrides)


def _memory(
    partition: SQLAlchemySemanticStorePartition,
    vector: FakeVectorStoreCollection,
    embedder: FakeEmbedder,
    llm: FakeLanguageModel,
    *,
    schema: PartitionSchema | None = None,
    clustering_config: ClusteringConfig | None = None,
    reranker: FakeReranker | None = None,
) -> AttributeMemory:
    return AttributeMemory(
        partition=partition,
        vector_collection=vector,
        embedder=embedder,
        language_model=llm,
        schema=schema or _schema(),
        clustering_config=clustering_config or _config(),
        reranker=reranker,
    )


def _event(
    text: str,
    *,
    timestamp: datetime | None = None,
    uuid: UUID | None = None,
    source: str = "user",
) -> Event:
    return Event(
        uuid=uuid or uuid4(),
        timestamp=timestamp or datetime.now(tz=UTC),
        body=Content(
            context=MessageContext(source=source),
            items=[Text(text=text)],
        ),
    )


def _attr(
    *,
    topic: str = "Profile",
    category: str = "food",
    attribute: str = "favorite_pizza",
    value: str = "margherita",
    properties: dict | None = None,
    citations: tuple[UUID, ...] | None = None,
) -> SemanticAttribute:
    return SemanticAttribute(
        id=uuid4(),
        topic=topic,
        category=category,
        attribute=attribute,
        value=value,
        properties=properties,
        citations=citations,
    )


def _feature_update(*commands: Command) -> dict:
    """Build the payload a FakeLanguageModel returns for _llm_extract_commands."""
    return {"commands": [c.model_dump() for c in commands]}


def _consolidation(
    *,
    consolidated: list[dict] | None = None,
    keep_indices: list[int] | None = None,
) -> dict:
    """Build the payload a FakeLanguageModel returns for _llm_consolidate."""
    return {
        "consolidated_memories": consolidated or [],
        "keep_indices": list(keep_indices or []),
    }


# ---------------------------------------------------------------------------
# add_attributes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_attributes_empty_is_noop(
    partition: SQLAlchemySemanticStorePartition,
    fake_vector_collection: FakeVectorStoreCollection,
    fake_embedder: FakeEmbedder,
    fake_llm: FakeLanguageModel,
) -> None:
    memory = _memory(partition, fake_vector_collection, fake_embedder, fake_llm)
    await memory.add_attributes([])
    assert fake_embedder.ingest_calls == []


@pytest.mark.asyncio
async def test_add_attributes_persists_to_both_stores(
    partition: SQLAlchemySemanticStorePartition,
    fake_vector_collection: FakeVectorStoreCollection,
    fake_embedder: FakeEmbedder,
    fake_llm: FakeLanguageModel,
) -> None:
    memory = _memory(partition, fake_vector_collection, fake_embedder, fake_llm)
    a = _attr()
    await memory.add_attributes([a])

    stored = await partition.get_attributes([a.id])
    assert a.id in stored
    vector_rows = await fake_vector_collection.get(record_uuids=[a.id])
    assert len(vector_rows) == 1
    assert vector_rows[0].properties is not None
    assert vector_rows[0].properties["_topic"] == "Profile"


@pytest.mark.asyncio
async def test_add_attributes_rejects_reserved_keys(
    partition: SQLAlchemySemanticStorePartition,
    fake_vector_collection: FakeVectorStoreCollection,
    fake_embedder: FakeEmbedder,
    fake_llm: FakeLanguageModel,
) -> None:
    memory = _memory(partition, fake_vector_collection, fake_embedder, fake_llm)
    a = _attr(properties={"_cluster_id": "c_0"})
    with pytest.raises(ValueError, match="reserved"):
        await memory.add_attributes([a])
    assert await partition.get_attributes([a.id]) == {}


@pytest.mark.asyncio
async def test_add_attributes_accepts_application_underscore_keys(
    partition: SQLAlchemySemanticStorePartition,
    fake_vector_collection: FakeVectorStoreCollection,
    fake_embedder: FakeEmbedder,
    fake_llm: FakeLanguageModel,
) -> None:
    """Narrow reservation: app-specific ``_foo`` is allowed."""
    memory = _memory(partition, fake_vector_collection, fake_embedder, fake_llm)
    a = _attr(properties={"_app_internal": "value"})
    await memory.add_attributes([a])
    got = (await partition.get_attributes([a.id])).get(a.id)
    assert got is not None
    assert got.properties == {"_app_internal": "value"}


@pytest.mark.asyncio
async def test_add_attributes_rejects_caller_supplied_citations(
    partition: SQLAlchemySemanticStorePartition,
    fake_vector_collection: FakeVectorStoreCollection,
    fake_embedder: FakeEmbedder,
    fake_llm: FakeLanguageModel,
) -> None:
    """Citations are managed by the memory (only :meth:`ingest` sets them)."""
    memory = _memory(partition, fake_vector_collection, fake_embedder, fake_llm)
    a = _attr(citations=(uuid4(), uuid4()))
    with pytest.raises(ValueError, match="managed by the memory"):
        await memory.add_attributes([a])
    # Nothing persisted.
    assert await partition.get_attributes([a.id]) == {}


@pytest.mark.asyncio
async def test_add_attributes_store_failure_skips_vector() -> None:
    partition_mock = AsyncMock()
    partition_mock.add_attributes.side_effect = RuntimeError("store down")
    vector_mock = AsyncMock()
    memory = AttributeMemory(
        partition=partition_mock,
        vector_collection=vector_mock,
        embedder=FakeEmbedder(),
        language_model=FakeLanguageModel(),
        schema=_schema(),
    )
    with pytest.raises(RuntimeError, match="store down"):
        await memory.add_attributes([_attr()])
    vector_mock.upsert.assert_not_awaited()


# ---------------------------------------------------------------------------
# delete_attributes / delete_attributes_matching
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_attributes_vector_first(
    partition: SQLAlchemySemanticStorePartition,
    fake_vector_collection: FakeVectorStoreCollection,
    fake_embedder: FakeEmbedder,
    fake_llm: FakeLanguageModel,
) -> None:
    memory = _memory(partition, fake_vector_collection, fake_embedder, fake_llm)
    a = _attr()
    await memory.add_attributes([a])
    await memory.delete_attributes([a.id])
    assert await partition.get_attributes([a.id]) == {}
    assert await fake_vector_collection.get(record_uuids=[a.id]) == []


@pytest.mark.asyncio
async def test_delete_attributes_empty_is_noop() -> None:
    partition_mock = AsyncMock()
    vector_mock = AsyncMock()
    memory = AttributeMemory(
        partition=partition_mock,
        vector_collection=vector_mock,
        embedder=FakeEmbedder(),
        language_model=FakeLanguageModel(),
        schema=_schema(),
    )
    await memory.delete_attributes([])
    vector_mock.delete.assert_not_awaited()
    partition_mock.delete_attributes.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_attributes_ordering() -> None:
    partition_mock = AsyncMock()
    vector_mock = AsyncMock()
    calls: list[str] = []
    vector_mock.delete.side_effect = lambda **_: calls.append("vector")
    partition_mock.delete_attributes.side_effect = lambda _: calls.append("store")

    memory = AttributeMemory(
        partition=partition_mock,
        vector_collection=vector_mock,
        embedder=FakeEmbedder(),
        language_model=FakeLanguageModel(),
        schema=_schema(),
    )
    await memory.delete_attributes([uuid4()])
    assert calls == ["vector", "store"]


@pytest.mark.asyncio
async def test_delete_attributes_matching(
    partition: SQLAlchemySemanticStorePartition,
    fake_vector_collection: FakeVectorStoreCollection,
    fake_embedder: FakeEmbedder,
    fake_llm: FakeLanguageModel,
) -> None:
    memory = _memory(partition, fake_vector_collection, fake_embedder, fake_llm)
    keep = _attr(category="music")
    drop = _attr(category="food")
    await memory.add_attributes([keep, drop])

    await memory.delete_attributes_matching(
        filter_expr=Comparison(field="category", op="=", value="food")
    )
    remaining = await partition.get_attributes([keep.id, drop.id])
    assert keep.id in remaining
    assert drop.id not in remaining


# ---------------------------------------------------------------------------
# retrieve
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_without_reranker_returns_vector_order(
    partition: SQLAlchemySemanticStorePartition,
    fake_vector_collection: FakeVectorStoreCollection,
    fake_embedder: FakeEmbedder,
    fake_llm: FakeLanguageModel,
) -> None:
    memory = _memory(partition, fake_vector_collection, fake_embedder, fake_llm)
    a = _attr(value="first")
    b = _attr(value="second")
    await memory.add_attributes([a, b])

    results = await memory.retrieve("query")
    assert len(results) == 2
    # FakeVectorStoreCollection scores by -|insertion_index| descending;
    # the first insert (a) ranks highest.
    assert results[0][0].id == a.id
    assert results[1][0].id == b.id


@pytest.mark.asyncio
async def test_retrieve_with_reranker_reorders_and_overfetches(
    partition: SQLAlchemySemanticStorePartition,
    fake_vector_collection: FakeVectorStoreCollection,
    fake_embedder: FakeEmbedder,
    fake_llm: FakeLanguageModel,
    fake_reranker: FakeReranker,
) -> None:
    memory = _memory(
        partition,
        fake_vector_collection,
        fake_embedder,
        fake_llm,
        reranker=fake_reranker,
    )
    # Insert 6 attributes; request top_k=2.  The FakeReranker scores
    # by reversed candidate order (first candidate = highest), so the
    # last-inserted attributes (which vector-rank lowest) win.
    attrs = [_attr(attribute=f"a_{i}", value=f"v_{i}") for i in range(6)]
    await memory.add_attributes(attrs)

    results = await memory.retrieve("query", top_k=2)
    assert len(results) == 2
    # Reranker was called with all 6 candidates (overfetched past top_k=2).
    assert len(fake_reranker.calls) == 1
    _query, candidates = fake_reranker.calls[0]
    assert len(candidates) == 6


@pytest.mark.asyncio
async def test_retrieve_filter_translates_for_vector_store(
    partition: SQLAlchemySemanticStorePartition,
    fake_vector_collection: FakeVectorStoreCollection,
    fake_embedder: FakeEmbedder,
    fake_llm: FakeLanguageModel,
) -> None:
    memory = _memory(partition, fake_vector_collection, fake_embedder, fake_llm)
    await memory.add_attributes([_attr(category="food"), _attr(category="music")])
    await memory.retrieve(
        "query",
        filter_expr=Comparison(field="category", op="=", value="food"),
    )
    sent = fake_vector_collection.last_property_filter
    assert isinstance(sent, Comparison)
    assert sent.field == "_category"


@pytest.mark.asyncio
async def test_retrieve_empty(
    partition: SQLAlchemySemanticStorePartition,
    fake_vector_collection: FakeVectorStoreCollection,
    fake_embedder: FakeEmbedder,
    fake_llm: FakeLanguageModel,
) -> None:
    memory = _memory(partition, fake_vector_collection, fake_embedder, fake_llm)
    assert await memory.retrieve("query") == []


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_empty_returns_empty(
    partition: SQLAlchemySemanticStorePartition,
    fake_vector_collection: FakeVectorStoreCollection,
    fake_embedder: FakeEmbedder,
    fake_llm: FakeLanguageModel,
) -> None:
    memory = _memory(partition, fake_vector_collection, fake_embedder, fake_llm)
    assert await memory.ingest([]) == ()


@pytest.mark.asyncio
async def test_ingest_without_clustering_uses_topic_wide_profile(
    partition: SQLAlchemySemanticStorePartition,
    fake_vector_collection: FakeVectorStoreCollection,
    fake_embedder: FakeEmbedder,
    fake_llm: FakeLanguageModel,
) -> None:
    memory = _memory(
        partition,
        fake_vector_collection,
        fake_embedder,
        fake_llm,
        clustering_config=_config(enabled=False),
    )
    fake_llm.parsed_responses.extend(
        [
            _feature_update(
                Command(
                    command=CommandType.ADD,
                    category="food",
                    attribute="favorite_pizza",
                    value="margherita",
                )
            ),
            _feature_update(
                Command(
                    command=CommandType.DELETE,
                    category="food",
                    attribute="favorite_pizza",
                    value="margherita",
                ),
                Command(
                    command=CommandType.ADD,
                    category="food",
                    attribute="favorite_pizza",
                    value="pepperoni",
                ),
            ),
        ]
    )

    e1 = _event("first preference")
    e2 = _event("updated preference")
    processed = await memory.ingest([e1, e2])

    assert processed == (e1.uuid, e2.uuid)
    assert await partition.get_cluster_state() is None

    uuids = await partition.list_attribute_uuids_matching()
    got = list((await partition.get_attributes(uuids)).values())
    assert len(got) == 1
    assert got[0].value == "pepperoni"
    assert got[0].properties is None


@pytest.mark.asyncio
async def test_ingest_below_trigger_stays_pending(
    partition: SQLAlchemySemanticStorePartition,
    fake_vector_collection: FakeVectorStoreCollection,
    fake_embedder: FakeEmbedder,
    fake_llm: FakeLanguageModel,
) -> None:
    memory = _memory(
        partition,
        fake_vector_collection,
        fake_embedder,
        fake_llm,
        clustering_config=_config(trigger_messages=5, trigger_age=None),
    )
    processed = await memory.ingest([_event("hi"), _event("there")])
    assert processed == ()
    # Cluster state is persisted with the two pending events.
    state = await partition.get_cluster_state()
    assert state is not None
    assert sum(len(events) for events in state.pending_events.values()) == 2


@pytest.mark.asyncio
async def test_ingest_size_trigger_flushes(
    partition: SQLAlchemySemanticStorePartition,
    fake_vector_collection: FakeVectorStoreCollection,
    fake_embedder: FakeEmbedder,
    fake_llm: FakeLanguageModel,
) -> None:
    # One-topic schema keeps LLM call count predictable.
    schema = _schema(
        TopicDefinition(
            name="Profile",
            description="Profile",
            categories=(CategoryDefinition(name="food"),),
        )
    )
    memory = _memory(
        partition,
        fake_vector_collection,
        fake_embedder,
        fake_llm,
        schema=schema,
        clustering_config=_config(trigger_messages=2, trigger_age=None),
    )
    fake_llm.parsed_responses.append(
        _feature_update(
            Command(
                command=CommandType.ADD,
                category="food",
                attribute="favorite_pizza",
                value="margherita",
            )
        )
    )

    e1 = _event("I love margherita pizza")
    e2 = _event("Thin crust, lots of basil")
    processed = await memory.ingest([e1, e2])
    assert set(processed) == {e1.uuid, e2.uuid}

    # One new attribute written with _cluster_id metadata.
    uuids = await partition.list_attribute_uuids_matching()
    assert len(uuids) == 1
    got = (await partition.get_attributes(uuids)).get(uuids[0])
    assert got is not None
    assert got.topic == "Profile"
    assert got.category == "food"
    assert got.value == "margherita"
    assert got.properties is not None
    assert got.properties.get("_cluster_id") is not None


@pytest.mark.asyncio
async def test_ingest_age_trigger_flushes(
    partition: SQLAlchemySemanticStorePartition,
    fake_vector_collection: FakeVectorStoreCollection,
    fake_embedder: FakeEmbedder,
    fake_llm: FakeLanguageModel,
) -> None:
    schema = _schema(
        TopicDefinition(
            name="Profile",
            description="Profile",
            categories=(CategoryDefinition(name="food"),),
        )
    )
    memory = _memory(
        partition,
        fake_vector_collection,
        fake_embedder,
        fake_llm,
        schema=schema,
        clustering_config=_config(
            trigger_messages=0,  # size trigger disabled
            trigger_age=timedelta(seconds=10),
        ),
    )
    fake_llm.parsed_responses.append(_feature_update())

    old = _event(
        "ancient message",
        timestamp=datetime.now(tz=UTC) - timedelta(minutes=5),
    )
    processed = await memory.ingest([old])
    assert set(processed) == {old.uuid}


@pytest.mark.asyncio
async def test_ingest_does_not_flush_when_size_trigger_disabled_and_age_not_met(
    partition: SQLAlchemySemanticStorePartition,
    fake_vector_collection: FakeVectorStoreCollection,
    fake_embedder: FakeEmbedder,
    fake_llm: FakeLanguageModel,
) -> None:
    schema = _schema(
        TopicDefinition(
            name="Profile",
            description="Profile",
            categories=(CategoryDefinition(name="food"),),
        )
    )
    memory = _memory(
        partition,
        fake_vector_collection,
        fake_embedder,
        fake_llm,
        schema=schema,
        clustering_config=_config(
            trigger_messages=0,  # size trigger disabled
            trigger_age=timedelta(days=1),
        ),
    )

    recent = _event("recent message", timestamp=datetime.now(tz=UTC))
    processed = await memory.ingest([recent])

    assert processed == ()
    assert fake_llm.parsed_calls == []


@pytest.mark.asyncio
async def test_ingest_persists_cluster_state_across_calls(
    partition: SQLAlchemySemanticStorePartition,
    fake_vector_collection: FakeVectorStoreCollection,
    fake_embedder: FakeEmbedder,
    fake_llm: FakeLanguageModel,
) -> None:
    schema = _schema(
        TopicDefinition(
            name="Profile",
            description="Profile",
            categories=(CategoryDefinition(name="food"),),
        )
    )
    memory = _memory(
        partition,
        fake_vector_collection,
        fake_embedder,
        fake_llm,
        schema=schema,
        clustering_config=_config(trigger_messages=3, trigger_age=None),
    )

    e1 = _event("one")
    e2 = _event("two")
    processed = await memory.ingest([e1, e2])
    assert processed == ()

    # Second call carries the first pair forward: trigger fires at 3.
    fake_llm.parsed_responses.append(_feature_update())
    e3 = _event("three")
    processed = await memory.ingest([e1, e2, e3])
    assert set(processed) == {e1.uuid, e2.uuid, e3.uuid}


@pytest.mark.asyncio
async def test_ingest_skips_topic_on_context_length(
    partition: SQLAlchemySemanticStorePartition,
    fake_vector_collection: FakeVectorStoreCollection,
    fake_embedder: FakeEmbedder,
    fake_llm: FakeLanguageModel,
) -> None:
    schema = _schema(
        TopicDefinition(
            name="Profile",
            description="Profile",
            categories=(CategoryDefinition(name="food"),),
        )
    )
    memory = _memory(
        partition,
        fake_vector_collection,
        fake_embedder,
        fake_llm,
        schema=schema,
        clustering_config=_config(trigger_messages=1, trigger_age=None),
    )
    fake_llm.raise_on_parsed = RuntimeError("context_length_exceeded")

    e = _event("too long")
    processed = await memory.ingest([e])
    # Cluster is still flushed (uid returned) so caller can ack; no
    # attributes written because the LLM failed.
    assert set(processed) == {e.uuid}
    assert await partition.list_attribute_uuids_matching() == ()


@pytest.mark.asyncio
async def test_ingest_caps_existing_features_per_update(
    partition: SQLAlchemySemanticStorePartition,
    fake_vector_collection: FakeVectorStoreCollection,
    fake_embedder: FakeEmbedder,
    fake_llm: FakeLanguageModel,
) -> None:
    schema = _schema(
        TopicDefinition(
            name="Profile",
            description="Profile",
            categories=(CategoryDefinition(name="food"),),
        )
    )
    config = _config(
        trigger_messages=1,
        trigger_age=None,
        max_features_per_update=3,
    )
    memory = _memory(
        partition,
        fake_vector_collection,
        fake_embedder,
        fake_llm,
        schema=schema,
        clustering_config=config,
    )

    # Pre-seed 5 attrs under the same cluster id so _list_features_for_cluster
    # has them to return.
    cluster_id = "cluster_0"
    preseed_attrs = [
        SemanticAttribute(
            id=uuid4(),
            topic="Profile",
            category="food",
            attribute=f"slot_{i}",
            value=f"val_{i}",
            properties={"_cluster_id": cluster_id},
        )
        for i in range(5)
    ]
    # Use the internal write path to bypass reserved-key validation.
    await memory._write_attributes(preseed_attrs)
    # Save a matching cluster state so ingest assigns new episodes to
    # the same cluster id on trigger.
    from memmachine_server.semantic_memory.attribute_memory.data_types import (
        ClusterInfo,
        ClusterState,
    )

    await partition.save_cluster_state(
        ClusterState(
            clusters={
                cluster_id: ClusterInfo(
                    centroid=[0.0, 0.0, 0.0],
                    count=1,
                    last_ts=datetime.now(tz=UTC),
                )
            },
            next_cluster_id=1,
        )
    )

    fake_llm.parsed_responses.append(_feature_update())
    await memory.ingest([_event("something")])

    # Inspect the user_prompt the LLM saw — it should include at most
    # max_features_per_update (=3) features in the "old profile".
    assert len(fake_llm.parsed_calls) == 1
    prompt = fake_llm.parsed_calls[0]["user_prompt"]
    # Each preseeded feature shows up as a slot_N key in the JSON.
    slot_mentions = sum(f"slot_{i}" in prompt for i in range(5))
    assert slot_mentions == 3


@pytest.mark.asyncio
async def test_ingest_invokes_configured_splitter(
    partition: SQLAlchemySemanticStorePartition,
    fake_vector_collection: FakeVectorStoreCollection,
    fake_embedder: FakeEmbedder,
    fake_llm: FakeLanguageModel,
) -> None:
    splitter = SplitPerEvent()
    memory = _memory(
        partition,
        fake_vector_collection,
        fake_embedder,
        fake_llm,
        clustering_config=_config(
            trigger_messages=2,
            trigger_age=None,
            splitter=splitter,
        ),
    )
    fake_llm.parsed_responses.extend(
        [
            _feature_update(
                Command(
                    command=CommandType.ADD,
                    category="food",
                    attribute="favorite_pizza",
                    value="margherita",
                )
            ),
            _feature_update(
                Command(
                    command=CommandType.ADD,
                    category="food",
                    attribute="favorite_pasta",
                    value="cacio_e_pepe",
                )
            ),
        ]
    )

    e1 = _event("pizza note")
    e2 = _event("pasta note")
    processed = await memory.ingest([e1, e2])

    assert set(processed) == {e1.uuid, e2.uuid}
    assert splitter.calls == 1

    stored = list(
        (
            await partition.get_attributes(
                await partition.list_attribute_uuids_matching()
            )
        ).values()
    )
    cluster_ids = {
        attr.properties["_cluster_id"]
        for attr in stored
        if attr.properties is not None and "_cluster_id" in attr.properties
    }
    assert cluster_ids == {"cluster_0", "cluster_0_split_1"}


# ---------------------------------------------------------------------------
# consolidate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consolidate_topic_and_category(
    partition: SQLAlchemySemanticStorePartition,
    fake_vector_collection: FakeVectorStoreCollection,
    fake_embedder: FakeEmbedder,
    fake_llm: FakeLanguageModel,
) -> None:
    schema = _schema(
        TopicDefinition(
            name="Profile",
            description="Profile",
            categories=(
                CategoryDefinition(name="food"),
                CategoryDefinition(name="music"),
            ),
        )
    )
    memory = _memory(
        partition, fake_vector_collection, fake_embedder, fake_llm, schema=schema
    )

    a = _attr(category="food", attribute="pizza", value="margherita")
    b = _attr(category="food", attribute="pizza", value="marg")
    c = _attr(category="music", attribute="genre", value="jazz")
    await memory.add_attributes([a, b, c])

    # LLM drops both food entries (keep_indices=[]) and returns one
    # consolidated entry.
    fake_llm.parsed_responses.append(
        _consolidation(
            consolidated=[
                {"category": "food", "attribute": "pizza", "value": "margherita"},
            ],
            keep_indices=[],
        )
    )

    await memory.consolidate(topic="Profile", category="food")

    remaining = {
        x.value
        async for x in partition.list_attributes(
            filter_expr=Comparison(field="category", op="=", value="food")
        )
    }
    assert remaining == {"margherita"}
    # Music is untouched.
    music = {
        x.id
        async for x in partition.list_attributes(
            filter_expr=Comparison(field="category", op="=", value="music")
        )
    }
    assert music == {c.id}


@pytest.mark.asyncio
async def test_consolidate_unknown_topic_raises(
    partition: SQLAlchemySemanticStorePartition,
    fake_vector_collection: FakeVectorStoreCollection,
    fake_embedder: FakeEmbedder,
    fake_llm: FakeLanguageModel,
) -> None:
    memory = _memory(partition, fake_vector_collection, fake_embedder, fake_llm)
    with pytest.raises(ValueError, match="not in the schema"):
        await memory.consolidate(topic="Unknown")


# ---------------------------------------------------------------------------
# Auto-consolidate at end of ingest
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_consolidate_triggers_at_threshold(
    partition: SQLAlchemySemanticStorePartition,
    fake_vector_collection: FakeVectorStoreCollection,
    fake_embedder: FakeEmbedder,
    fake_llm: FakeLanguageModel,
) -> None:
    schema = _schema(
        TopicDefinition(
            name="Profile",
            description="Profile",
            categories=(CategoryDefinition(name="food"),),
        )
    )
    memory = _memory(
        partition,
        fake_vector_collection,
        fake_embedder,
        fake_llm,
        schema=schema,
        clustering_config=_config(
            trigger_messages=1,
            trigger_age=None,
            consolidation_threshold=2,
        ),
    )

    # Seed two attributes so the (Profile, food) count reaches 2.
    pre = [
        _attr(category="food", attribute="x", value="v1"),
        _attr(category="food", attribute="y", value="v2"),
    ]
    await memory.add_attributes(pre)

    # Ingest will: run feature extraction once, then auto-consolidate
    # once.  Push matching responses.
    fake_llm.parsed_responses.append(_feature_update())  # extraction → no-op
    fake_llm.parsed_responses.append(
        _consolidation(
            consolidated=[
                {"category": "food", "attribute": "merged", "value": "merged"},
            ],
            keep_indices=[],
        )
    )

    await memory.ingest([_event("msg")])

    remaining = [
        x.value
        async for x in partition.list_attributes(
            filter_expr=Comparison(field="category", op="=", value="food")
        )
    ]
    assert remaining == ["merged"]


@pytest.mark.asyncio
async def test_auto_consolidate_skipped_below_threshold(
    partition: SQLAlchemySemanticStorePartition,
    fake_vector_collection: FakeVectorStoreCollection,
    fake_embedder: FakeEmbedder,
    fake_llm: FakeLanguageModel,
) -> None:
    schema = _schema(
        TopicDefinition(
            name="Profile",
            description="Profile",
            categories=(CategoryDefinition(name="food"),),
        )
    )
    memory = _memory(
        partition,
        fake_vector_collection,
        fake_embedder,
        fake_llm,
        schema=schema,
        clustering_config=_config(
            trigger_messages=1,
            trigger_age=None,
            consolidation_threshold=10,
        ),
    )
    # Single extraction call; no consolidation call should be made.
    fake_llm.parsed_responses.append(_feature_update())

    await memory.ingest([_event("msg")])

    # Only one parsed call (the extraction); no consolidation ran.
    assert len(fake_llm.parsed_calls) == 1


# ---------------------------------------------------------------------------
# Reads (pass-through)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_attributes_passthrough(
    partition: SQLAlchemySemanticStorePartition,
    fake_vector_collection: FakeVectorStoreCollection,
    fake_embedder: FakeEmbedder,
    fake_llm: FakeLanguageModel,
) -> None:
    memory = _memory(partition, fake_vector_collection, fake_embedder, fake_llm)
    a = _attr()
    b = _attr(category="music")
    await memory.add_attributes([a, b])
    got = await memory.get_attributes([a.id, b.id])
    assert set(got.keys()) == {a.id, b.id}


@pytest.mark.asyncio
async def test_list_attributes_passthrough(
    partition: SQLAlchemySemanticStorePartition,
    fake_vector_collection: FakeVectorStoreCollection,
    fake_embedder: FakeEmbedder,
    fake_llm: FakeLanguageModel,
) -> None:
    memory = _memory(partition, fake_vector_collection, fake_embedder, fake_llm)
    a = _attr()
    b = _attr(category="music")
    await memory.add_attributes([a, b])
    result = [x async for x in memory.list_attributes()]
    assert {x.id for x in result} == {a.id, b.id}
