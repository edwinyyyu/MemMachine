"""Attribute memory: per-partition semantic-memory engine.

An :class:`AttributeMemory` instance is bound to one logical partition
— one :class:`SemanticStorePartition` paired with one
:class:`VectorStoreCollection`, with embedder, language-model, and
(optional) reranker handles to drive the LLM-assisted paths.  It
exposes the full set of operations that used to be split across
``SemanticService``, ``SemanticSessionManager``, ``IngestionService``,
and the consolidation helpers, deduplicated into a single class.

Public surface
--------------
* :meth:`ingest` — cluster-aware LLM extraction from conversation
  episodes.  Loads cluster state, assigns new episodes to clusters,
  flushes ready clusters (by size or age) to the LLM, writes
  extracted attributes with citations, garbage-collects idle
  clusters, and persists updated state.  Returns the uuids that were
  actually flushed so callers can ack them on their message queue.
* :meth:`add_attributes` — write caller-supplied attributes (no LLM,
  no clustering); the memory embeds values internally.
* :meth:`retrieve` — text query → embed → vector search → enrich.
* :meth:`get_attributes`, :meth:`list_attributes` — low-level reads.
* :meth:`delete_attributes`, :meth:`delete_attributes_matching` —
  low-level + filter-based deletes.
* :meth:`consolidate` — LLM-driven dedup within one topic or one
  ``(topic, category)``.  Auto-runs at the end of :meth:`ingest` for
  any ``(topic, category)`` over ``ClusteringConfig.consolidation_threshold``.

Ordering rules
--------------
* **Add**: :class:`SemanticStorePartition` first, then
  :class:`VectorStoreCollection`.
* **Delete**: :class:`VectorStoreCollection` first, then
  :class:`SemanticStorePartition`.

Invariant: ``vector.uuids ⊆ store.uuids``.

System-defined metadata in ``properties``
-----------------------------------------
Keys in ``SemanticAttribute.properties`` that begin with ``_`` denote
**system metadata**.  "System" here is broad — it includes both this
library and any application building on top of it; only an end-user
entering free-form metadata is outside the system.  The library
claims a **specific named set** of ``_``-prefixed keys for itself
(see :attr:`AttributeMemory._RESERVED_PROPERTY_KEYS`) and rejects
those from :meth:`add_attributes`; every other ``_``-prefixed key is
available for applications to use.

The one key the memory currently reserves is ``_cluster_id``: every
attribute produced by :meth:`ingest` carries the id of the cluster
it was extracted from, so subsequent LLM calls for the same cluster
see only that cluster's attributes as their "current profile".
"""

import functools
import json
import logging
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from memmachine_server.common.data_types import PropertyValue
from memmachine_server.common.embedder import Embedder
from memmachine_server.common.filter.filter_parser import (
    And,
    Comparison,
    FilterExpr,
    demangle_user_metadata_key,
    map_filter_fields,
    normalize_filter_field,
)
from memmachine_server.common.language_model import LanguageModel
from memmachine_server.common.reranker import Reranker
from memmachine_server.common.vector_store import Record, VectorStoreCollection
from memmachine_server.semantic_memory.attribute_memory.clustering import (
    ClusterManager,
    NoOpClusterSplitter,
)
from memmachine_server.semantic_memory.attribute_memory.clustering_config import (
    ClusteringConfig,
)
from memmachine_server.semantic_memory.attribute_memory.data_types import (
    ClusterState,
    Command,
    CommandType,
    Content,
    Event,
    MessageContext,
    Text,
)
from memmachine_server.semantic_memory.attribute_memory.semantic_store.semantic_store import (
    SemanticAttribute,
    SemanticStorePartition,
)
from memmachine_server.semantic_memory.attribute_memory.semantic_store.sqlalchemy_semantic_store import (
    PartitionSchema,
    TopicDefinition,
)

logger = logging.getLogger(__name__)

_CLUSTER_METADATA_KEY = "_cluster_id"

# Rerank over-fetch tuning (mirrors DeclarativeMemory.retrieve_episodes):
# when a reranker is available, pull a bigger candidate window from the
# vector store and let the reranker narrow it down.
_RETRIEVE_OVERFETCH_MULTIPLIER = 5
_RETRIEVE_OVERFETCH_CAP = 200


class _LLMUpdateResult(BaseModel):
    """LLM response schema for attribute-update calls."""

    commands: list[Command] = Field(default_factory=list)


class _LLMConsolidatedAttribute(BaseModel):
    """LLM response shape for a consolidated attribute."""

    category: str
    attribute: str
    value: str


class _LLMConsolidateResult(BaseModel):
    """LLM response schema for consolidation calls.

    ``keep_indices`` names the positions in the input list whose
    attributes should be preserved (zero-based).  The LLM never sees
    attribute UUIDs — the caller maps indices back to UUIDs from the
    ordered input.
    """

    consolidated_memories: list[_LLMConsolidatedAttribute] = Field(default_factory=list)
    keep_indices: list[int] | None = None


def _is_context_length_exceeded_error(error: BaseException) -> bool:
    """Walk an exception chain looking for an LLM context-overflow signal.

    Ported from :mod:`semantic_memory.semantic_ingestion`.
    """
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        code = getattr(current, "code", None)
        if isinstance(code, str) and code.lower() == "context_length_exceeded":
            return True
        message = str(current).lower()
        if (
            "context_length_exceeded" in message
            or "exceeds the context window" in message
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


class AttributeMemory:
    """Per-partition semantic-memory engine."""

    _RESERVED_PREFIX = "_"
    _SYSTEM_FIELDS: tuple[str, ...] = (
        "topic",
        "category",
        "attribute",
        "value",
    )
    # Property-dict keys that the memory reserves for its own use.
    # Applications may set their own ``_``-prefixed keys in
    # ``SemanticAttribute.properties`` (they are part of "the system"
    # too); this set names the specific keys the memory library itself
    # owns, which user-facing API is not allowed to set via
    # :meth:`add_attributes`.
    _RESERVED_PROPERTY_KEYS: frozenset[str] = frozenset({_CLUSTER_METADATA_KEY})

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        *,
        partition: SemanticStorePartition,
        vector_collection: VectorStoreCollection,
        embedder: Embedder,
        language_model: LanguageModel,
        schema: PartitionSchema,
        clustering_config: ClusteringConfig | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        """Bind to partition-scoped storages and LLM/embedder handles."""
        self._partition = partition
        self._vector = vector_collection
        self._embedder = embedder
        self._llm = language_model
        self._schema = schema
        self._config = clustering_config or ClusteringConfig()
        self._reranker = reranker
        self._cluster_manager = ClusterManager(self._config.cluster_params)

    @classmethod
    def expected_vector_store_collection_schema(
        cls,
    ) -> dict[str, type[PropertyValue]]:
        """Properties-schema contribution for the paired vector collection.

        Includes both the hierarchy-field keys (``_topic`` etc.) and
        every key in :attr:`_RESERVED_PROPERTY_KEYS` the memory might
        write — applications building the collection merge this with
        their own user-metadata schema.
        """
        return {
            **{f"{cls._RESERVED_PREFIX}{name}": str for name in cls._SYSTEM_FIELDS},
            **dict.fromkeys(cls._RESERVED_PROPERTY_KEYS, str),
        }

    # ------------------------------------------------------------------ #
    # Ingest (high-level, cluster-aware)
    # ------------------------------------------------------------------ #

    async def ingest(self, events: Iterable[Event]) -> tuple[UUID, ...]:
        """Cluster-aware LLM extraction from events.

        Loads persisted cluster state, assigns any new events to
        clusters, and flushes ready clusters (size or age trigger) to
        the LLM.  Returns the uuids that were flushed.  Events whose
        clusters aren't yet ready remain in persisted ``pending_events``
        and will be flushed on a future call when the trigger fires.

        Callers should pass every event that is still ``pending`` on
        their upstream queue — the memory dedupes via
        ``state.event_to_cluster`` so repeated passes are idempotent.
        """
        events_list = list(events)
        state = await self._partition.get_cluster_state() or ClusterState()

        await self._assign_new_events(events_list, state)

        now = datetime.now(tz=UTC)
        ready_cluster_ids = self._select_ready_clusters(state, now)
        if not ready_cluster_ids:
            self._apply_cluster_idle_gc(state, now)
            await self._partition.save_cluster_state(state)
            return ()

        ready_cluster_ids = ready_cluster_ids[: self._config.max_clusters_per_run]

        events_by_uuid = {e.uuid: e for e in events_list}
        clustered = self._collect_ready_clusters(
            ready_cluster_ids, state, events_by_uuid
        )
        if not clustered:
            self._apply_cluster_idle_gc(state, now)
            await self._partition.save_cluster_state(state)
            return ()

        processed_uuids: list[UUID] = []
        for cluster_id, cluster_events in clustered:
            await self._process_cluster(cluster_id, cluster_events)
            state.pending_events.pop(cluster_id, None)
            processed_uuids.extend(e.uuid for e in cluster_events)

        _rebuild_event_to_cluster(state)
        self._apply_cluster_idle_gc(state, now)
        await self._partition.save_cluster_state(state)
        await self._auto_consolidate()
        return tuple(processed_uuids)

    async def _assign_new_events(
        self,
        events: Sequence[Event],
        state: ClusterState,
    ) -> None:
        """Embed unseen events and assign them to clusters in ``state``."""
        new_events = [e for e in events if e.uuid not in state.event_to_cluster]
        if not new_events:
            return
        new_events.sort(key=lambda e: (e.timestamp, e.uuid))
        vectors = await self._embedder.ingest_embed(
            [_event_text(e) for e in new_events]
        )
        for event, vector in zip(new_events, vectors, strict=True):
            assignment = self._cluster_manager.assign(
                event_id=event.uuid,
                embedding=vector,
                timestamp=event.timestamp,
                state=state,
            )
            state.pending_events.setdefault(assignment.cluster_id, {})[event.uuid] = (
                event.timestamp
            )
        _rebuild_event_to_cluster(state)

    def _select_ready_clusters(
        self,
        state: ClusterState,
        now: datetime,
    ) -> list[str]:
        """Return cluster_ids whose pending events have tripped a trigger."""
        ready: list[tuple[datetime, str]] = []
        for cluster_id, events in state.pending_events.items():
            if not events:
                continue
            count = len(events)
            oldest = min(events.values())
            size_ready = (
                self._config.trigger_messages <= 0
                or count >= self._config.trigger_messages
            )
            age_ready = (
                self._config.trigger_age is not None
                and (now - oldest) >= self._config.trigger_age
            )
            if size_ready or age_ready:
                ready.append((oldest, cluster_id))
        return [cluster_id for _oldest, cluster_id in sorted(ready)]

    def _collect_ready_clusters(
        self,
        cluster_ids: Sequence[str],
        state: ClusterState,
        events_by_uuid: Mapping[UUID, Event],
    ) -> list[tuple[str, list[Event]]]:
        """Pair ready cluster ids with their events from the current batch.

        Skips any cluster whose pending event set isn't fully present
        in the caller-provided batch — processing a cluster without
        all its content would corrupt the LLM's view of it.
        """
        clustered: list[tuple[str, list[Event]]] = []
        for cluster_id in cluster_ids:
            event_uuids = list(state.pending_events.get(cluster_id, {}).keys())
            if not event_uuids:
                continue
            cluster_events = [
                events_by_uuid[uid] for uid in event_uuids if uid in events_by_uuid
            ]
            if len(cluster_events) < len(event_uuids):
                logger.warning(
                    "Skipping ready cluster %s: %d of %d events missing from batch",
                    cluster_id,
                    len(event_uuids) - len(cluster_events),
                    len(event_uuids),
                )
                continue
            cluster_events.sort(key=lambda e: (e.timestamp, e.uuid))
            clustered.append((cluster_id, cluster_events))
        return clustered

    async def _process_cluster(
        self,
        cluster_id: str,
        events: Sequence[Event],
    ) -> None:
        """Run LLM extraction for one ready cluster across every topic.

        Context-length failures from the LLM are logged and skipped
        for the topic in question — the cluster's other topics still
        run, and the cluster is considered processed so its events
        won't loop back through ingest.
        """
        content = _format_cluster_content(events)
        citation_uuids = tuple(e.uuid for e in events)
        for topic_def in self._schema.topics:
            existing = await self._list_cluster_attributes(topic_def, cluster_id)
            try:
                commands = await self._llm_extract_commands(
                    existing=existing,
                    message_content=content,
                    update_prompt=topic_def.update_prompt,
                )
            except Exception as exc:
                if _is_context_length_exceeded_error(exc):
                    logger.warning(
                        "Skipping topic %s for cluster %s: LLM context "
                        "length exceeded (%d events, %d existing attributes)",
                        topic_def.name,
                        cluster_id,
                        len(events),
                        len(existing),
                    )
                    continue
                raise
            await self._apply_commands(
                topic_def.name,
                cluster_id,
                commands,
                citation_uuids,
            )

    def _apply_cluster_idle_gc(self, state: ClusterState, now: datetime) -> None:
        """Drop clusters inactive longer than ``idle_ttl`` with no pending events."""
        if self._config.idle_ttl is None:
            return
        for cluster_id, info in list(state.clusters.items()):
            if cluster_id in state.pending_events:
                continue
            if (now - info.last_ts) > self._config.idle_ttl:
                state.clusters.pop(cluster_id, None)
                state.split_records.pop(cluster_id, None)

    # ------------------------------------------------------------------ #
    # Mid-level write (no LLM, no clustering)
    # ------------------------------------------------------------------ #

    async def add_attributes(
        self,
        attributes: Iterable[SemanticAttribute],
    ) -> None:
        """Persist caller-supplied attributes to both stores.

        Ordering: store first, vector store second.

        Citations are managed by the memory itself — they flow from
        :meth:`ingest` where the memory knows which events an attribute
        was extracted from.  Attempting to pass ``citations`` on a
        caller-supplied attribute raises: the caller has no authority
        over provenance and silently dropping the field would hide
        bugs.  Leave ``citations=None``.

        Rejects any ``properties`` key in
        :attr:`_RESERVED_PROPERTY_KEYS` (system-defined keys like
        ``_cluster_id`` are the memory's to set, not the caller's).
        """
        attrs = list(attributes)
        if not attrs:
            return
        for attr in attrs:
            if attr.citations:
                raise ValueError(
                    "citations are managed by the memory (set from event "
                    "uuids during ingest); do not set them on caller-supplied "
                    "attributes"
                )
            self._validate_attribute_properties(attr.properties)
        await self._write_attributes(attrs)

    async def _write_attributes(
        self,
        attrs: Sequence[SemanticAttribute],
    ) -> None:
        """Internal write path.

        Skips reserved-key validation and honors ``attr.citations`` —
        both are required for :meth:`ingest` to attach ``_cluster_id``
        and link source-event uuids.
        """
        if not attrs:
            return
        await self._partition.add_attributes(attrs)

        vectors = await self._embedder.ingest_embed([attr.value for attr in attrs])
        records = [
            self._build_vector_record(attr, vec)
            for attr, vec in zip(attrs, vectors, strict=True)
        ]
        await self._vector.upsert(records=records)

        for attr in attrs:
            if attr.citations:
                await self._partition.add_citations(attr.id, attr.citations)

    # ------------------------------------------------------------------ #
    # High-level reads
    # ------------------------------------------------------------------ #

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
        filter_expr: FilterExpr | None = None,
        load_citations: bool = False,
    ) -> list[tuple[SemanticAttribute, float]]:
        """Embed ``query`` and return ranked attributes.

        When a reranker is configured, the vector store is
        over-fetched (``5 * top_k`` candidates, capped at 200 — same
        window as :class:`DeclarativeMemory.retrieve_episodes`) and
        the reranker scores the full candidate set before trimming to
        ``top_k``.
        """
        query_vectors = await self._embedder.search_embed([query])
        if not query_vectors:
            return []
        vector_filter = self._translate_filter_for_vector_store(filter_expr)
        vector_limit = top_k
        if self._reranker is not None and top_k is not None:
            vector_limit = min(
                top_k * _RETRIEVE_OVERFETCH_MULTIPLIER, _RETRIEVE_OVERFETCH_CAP
            )
        results = await self._vector.query(
            query_vectors=query_vectors,
            limit=vector_limit,
            score_threshold=score_threshold,
            property_filter=vector_filter,
            return_properties=False,
        )
        if not results or not results[0].matches:
            return []
        matches = results[0].matches
        attribute_map = await self._partition.get_attributes(
            [m.record.uuid for m in matches], load_citations=load_citations
        )
        scored = [
            (attribute_map[m.record.uuid], m.score)
            for m in matches
            if m.record.uuid in attribute_map
        ]
        if self._reranker is not None and scored:
            rerank_scores = await self._reranker.score(
                query, [attr.value for attr, _ in scored]
            )
            scored = sorted(
                zip(scored, rerank_scores, strict=True),
                key=lambda pair: pair[1],
                reverse=True,
            )
            scored = [(attr, new_score) for (attr, _old), new_score in scored]
        if top_k is not None:
            scored = scored[:top_k]
        return scored

    async def get_attributes(
        self,
        attribute_uuids: Iterable[UUID],
        *,
        load_citations: bool = False,
    ) -> Mapping[UUID, SemanticAttribute]:
        """Bulk-load attributes by uuid."""
        return await self._partition.get_attributes(
            attribute_uuids, load_citations=load_citations
        )

    async def list_attributes(
        self,
        *,
        filter_expr: FilterExpr | None = None,
        load_citations: bool = False,
    ) -> AsyncIterator[SemanticAttribute]:
        """Stream attributes matching a relational filter."""
        async for attribute in self._partition.list_attributes(
            filter_expr=filter_expr,
            load_citations=load_citations,
        ):
            yield attribute

    # ------------------------------------------------------------------ #
    # Deletes
    # ------------------------------------------------------------------ #

    async def delete_attributes(
        self,
        attribute_uuids: Iterable[UUID],
    ) -> None:
        """Delete attributes from both stores, vector first."""
        uuids = tuple(attribute_uuids)
        if not uuids:
            return
        await self._vector.delete(record_uuids=uuids)
        await self._partition.delete_attributes(uuids)

    async def delete_attributes_matching(
        self,
        *,
        filter_expr: FilterExpr | None = None,
    ) -> None:
        """Delete attributes matching a relational filter, vector first."""
        uuids = await self._partition.list_attribute_uuids_matching(
            filter_expr=filter_expr
        )
        if not uuids:
            return
        await self.delete_attributes(uuids)

    # ------------------------------------------------------------------ #
    # Consolidation
    # ------------------------------------------------------------------ #

    async def consolidate(
        self,
        *,
        topic: str,
        category: str | None = None,
    ) -> None:
        """LLM-merge near-duplicate attributes.

        Scoped to one topic by default; pass ``category`` to restrict
        further to a single ``(topic, category)`` pair.  This is the
        legacy's "consolidate by tag" scope and is what
        auto-consolidation triggers on.
        """
        topic_def = self._schema.topic(topic)
        if topic_def is None:
            raise ValueError(f"Topic {topic!r} is not in the schema")

        filter_parts: list[FilterExpr] = [
            Comparison(field="topic", op="=", value=topic),
        ]
        if category is not None:
            filter_parts.append(Comparison(field="category", op="=", value=category))
        filter_expr = functools.reduce(lambda a, b: And(left=a, right=b), filter_parts)

        existing = [
            attr
            async for attr in self._partition.list_attributes(filter_expr=filter_expr)
        ]
        if not existing:
            return

        result = await self._llm_consolidate(
            existing=existing,
            consolidate_prompt=topic_def.consolidation_prompt,
        )
        if result is None:
            return

        keep_indices = set(result.keep_indices or [])
        to_delete = [a.id for i, a in enumerate(existing) if i not in keep_indices]
        if to_delete:
            await self.delete_attributes(to_delete)

        new_attrs = [
            SemanticAttribute(
                id=uuid4(),
                topic=topic,
                category=m.category,
                attribute=m.attribute,
                value=m.value,
            )
            for m in result.consolidated_memories
        ]
        if new_attrs:
            await self.add_attributes(new_attrs)

    async def _auto_consolidate(self) -> None:
        """Run :meth:`consolidate` on every ``(topic, category)`` over threshold.

        Called at the end of :meth:`ingest` once clusters are flushed.
        Matches the legacy's ingestion-time auto-sweep semantics.
        """
        threshold = self._config.consolidation_threshold
        if threshold <= 0:
            return
        for topic_def in self._schema.topics:
            for category_def in topic_def.categories:
                filter_expr = And(
                    left=Comparison(field="topic", op="=", value=topic_def.name),
                    right=Comparison(field="category", op="=", value=category_def.name),
                )
                uuids = await self._partition.list_attribute_uuids_matching(
                    filter_expr=filter_expr
                )
                if len(uuids) < threshold:
                    continue
                try:
                    await self.consolidate(
                        topic=topic_def.name, category=category_def.name
                    )
                except Exception:
                    logger.exception(
                        "Auto-consolidation failed for (%s, %s)",
                        topic_def.name,
                        category_def.name,
                    )

    # ------------------------------------------------------------------ #
    # Ingest helpers
    # ------------------------------------------------------------------ #

    async def _list_cluster_attributes(
        self,
        topic_def: TopicDefinition,
        cluster_id: str,
    ) -> list[SemanticAttribute]:
        """Load attributes under (topic, cluster_id), capped for prompt safety."""
        filter_expr = And(
            left=Comparison(field="topic", op="=", value=topic_def.name),
            right=Comparison(
                field=f"m.{_CLUSTER_METADATA_KEY}", op="=", value=cluster_id
            ),
        )
        cap = self._config.max_features_per_update
        attrs: list[SemanticAttribute] = []
        async for attr in self._partition.list_attributes(filter_expr=filter_expr):
            attrs.append(attr)
            if len(attrs) >= cap:
                break
        return attrs

    async def _llm_extract_commands(
        self,
        *,
        existing: Sequence[SemanticAttribute],
        message_content: str,
        update_prompt: str,
    ) -> list[Command]:
        """LLM pass: old profile + new message → add/delete commands.

        The "old profile" is serialized as a two-level
        ``{category: {attribute: value}}`` dict.  No UUIDs cross the
        prompt boundary — commands identify their target by
        ``(category, attribute, value)`` tuples, which we later match
        against the partition to resolve real UUIDs for deletion.
        """
        profile: dict[str, dict[str, str]] = {}
        for attr in existing:
            profile.setdefault(attr.category, {})[attr.attribute] = attr.value
        user_prompt = (
            "The old feature set is provided below:\n"
            "<OLD_PROFILE>\n"
            f"{json.dumps(profile)}\n"
            "</OLD_PROFILE>\n"
            "\n"
            "The history is provided below:\n"
            "<HISTORY>\n"
            f"{message_content}\n"
            "</HISTORY>\n"
        )
        result = await self._llm.generate_parsed_response(
            system_prompt=update_prompt,
            user_prompt=user_prompt,
            output_format=_LLMUpdateResult,
        )
        if result is None:
            return []
        return result.commands

    async def _llm_consolidate(
        self,
        *,
        existing: Sequence[SemanticAttribute],
        consolidate_prompt: str,
    ) -> _LLMConsolidateResult | None:
        """LLM pass: current attribute set → consolidated set + keep-list.

        Each input attribute is serialized with its zero-based
        ``index`` (position in ``existing``).  The LLM references the
        preserved inputs by ``keep_indices`` in its response — the
        caller maps indices back to real UUIDs from ``existing``.
        """
        memories = [
            {
                "index": i,
                "category": attr.category,
                "attribute": attr.attribute,
                "value": attr.value,
            }
            for i, attr in enumerate(existing)
        ]
        return await self._llm.generate_parsed_response(
            system_prompt=consolidate_prompt,
            user_prompt=json.dumps(memories),
            output_format=_LLMConsolidateResult,
        )

    async def _apply_commands(
        self,
        topic: str,
        cluster_id: str,
        commands: Iterable[Command],
        citation_uuids: Sequence[UUID],
    ) -> None:
        """Apply LLM-emitted add/delete commands under (topic, cluster_id)."""
        commands = tuple(commands)
        if not commands:
            return

        delete_cmds = [c for c in commands if c.command == CommandType.DELETE]
        add_cmds = [c for c in commands if c.command == CommandType.ADD]

        delete_uuids: set[UUID] = set()
        for cmd in delete_cmds:
            delete_uuids.update(await self._match_uuids(topic, cluster_id, cmd))
        if delete_uuids:
            await self.delete_attributes(delete_uuids)

        if add_cmds:
            new_attrs = [
                SemanticAttribute(
                    id=uuid4(),
                    topic=topic,
                    category=cmd.category,
                    attribute=cmd.attribute,
                    value=cmd.value,
                    properties={_CLUSTER_METADATA_KEY: cluster_id},
                    citations=tuple(citation_uuids) or None,
                )
                for cmd in add_cmds
            ]
            await self._write_attributes(new_attrs)

    async def _match_uuids(
        self,
        topic: str,
        cluster_id: str,
        cmd: Command,
    ) -> tuple[UUID, ...]:
        """Resolve uuids matching a delete command, scoped to the cluster."""
        parts: list[FilterExpr] = [
            Comparison(field="topic", op="=", value=topic),
            Comparison(field="category", op="=", value=cmd.category),
            Comparison(field="attribute", op="=", value=cmd.attribute),
            Comparison(field="value", op="=", value=cmd.value),
            Comparison(field=f"m.{_CLUSTER_METADATA_KEY}", op="=", value=cluster_id),
        ]
        filter_expr = functools.reduce(lambda a, b: And(left=a, right=b), parts)
        return await self._partition.list_attribute_uuids_matching(
            filter_expr=filter_expr
        )

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #

    @classmethod
    def _validate_attribute_properties(
        cls,
        properties: Mapping[str, object] | None,
    ) -> None:
        """Reject properties whose keys are reserved for the memory library."""
        if properties is None:
            return
        reserved = sorted(k for k in properties if k in cls._RESERVED_PROPERTY_KEYS)
        if reserved:
            raise ValueError(
                f"Property keys {reserved!r} are reserved by the memory library "
                "for its own use; remove them before calling add_attributes()"
            )

    @classmethod
    def _build_vector_record(
        cls,
        attribute: SemanticAttribute,
        vector: list[float],
    ) -> Record:
        """Pack a paired vector-store record from an attribute + its vector."""
        prefix = cls._RESERVED_PREFIX
        props: dict[str, PropertyValue] = {
            f"{prefix}topic": attribute.topic,
            f"{prefix}category": attribute.category,
            f"{prefix}attribute": attribute.attribute,
            f"{prefix}value": attribute.value,
        }
        if attribute.properties:
            props.update(attribute.properties)
        return Record(uuid=attribute.id, vector=vector, properties=props)

    @classmethod
    def _translate_filter_for_vector_store(
        cls,
        filter_expr: FilterExpr | None,
    ) -> FilterExpr | None:
        """Rewrite a user-facing filter for vector-store property keys."""
        if filter_expr is None:
            return None

        def translate(field: str) -> str:
            internal, is_user = normalize_filter_field(field)
            if is_user:
                return demangle_user_metadata_key(internal)
            return f"{cls._RESERVED_PREFIX}{internal}"

        return map_filter_fields(filter_expr, translate)


# ---------------------------------------------------------------------- #
# Module-level helpers
# ---------------------------------------------------------------------- #


def _event_text(event: Event) -> str:
    """Extract the plain-text content of an event for embedding / prompt use.

    Handles the :class:`Content` body with :class:`Text` item blocks;
    non-text items (image/audio/video/file_ref) contribute nothing to
    the text payload.  :class:`ReadFile` bodies return an empty string.
    """
    body = event.body
    if not isinstance(body, Content):
        return ""
    return "\n".join(item.text for item in body.items if isinstance(item, Text))


def _format_cluster_content(events: Sequence[Event]) -> str:
    """Format a cluster's events as a single prompt string.

    Each event's text is prefixed with its source (from
    :class:`MessageContext`) when available, matching the legacy
    "``source: content``" framing for conversation transcripts.
    """
    parts: list[str] = []
    for event in events:
        text = _event_text(event)
        if not text:
            continue
        body = event.body
        source: str | None = None
        if isinstance(body, Content) and isinstance(body.context, MessageContext):
            source = body.context.source
        parts.append(f"{source}: {text}" if source else text)
    return "\n\n".join(parts)


def _rebuild_event_to_cluster(state: ClusterState) -> None:
    """Recompute ``event_to_cluster`` from the authoritative ``pending_events``."""
    state.event_to_cluster = {
        event_id: cluster_id
        for cluster_id, events in state.pending_events.items()
        for event_id in events
    }


__all__ = ["AttributeMemory", "NoOpClusterSplitter"]
