"""Event memory system for storing and retrieving events."""

import asyncio
import datetime
import json
import logging
import time
from collections.abc import Iterable, Sequence
from typing import ClassVar, cast
from uuid import UUID

from pydantic import BaseModel, Field, InstanceOf

from memmachine_core.common import (
    RESERVED_PROPERTY_KEY_PREFIX,
    PropertyValue,
    is_reserved_property_key,
    reserved_property_key,
    validate_caller_property_key,
)
from memmachine_core.common.embedder import Embedder
from memmachine_core.common.filter import (
    And,
    FilterExpr,
    Ordering,
    map_filter_fields,
)
from memmachine_core.common.metrics_factory import (
    MetricsFactory,
    OperationTracker,
)
from memmachine_core.common.vector_store import (
    Record,
    VectorStoreCollection,
)

from .data_types import (
    Block,
    Derivative,
    Event,
    FormatOptions,
    NullContext,
    ProducerContext,
    QueryResult,
    Segment,
    SegmentContextMatch,
    TextBlock,
)
from .deriver import Deriver
from .formatting import format_timestamp
from .segment_store import SegmentStorePartition
from .segmenter import Segmenter

logger = logging.getLogger(__name__)


class EventMemoryParams(BaseModel):
    """
    Parameters for EventMemory.

    Attributes:
        segment_store_partition (SegmentStorePartition):
            Segment store partition.
        vector_store_collection (VectorStoreCollection):
            Vector store collection.
        segmenter (Segmenter):
            Segmenter that segments events into segments.
        deriver (Deriver):
            Deriver that derives derivatives from segments.
        embedder (Embedder):
            Embedder instance for creating embeddings.
        metrics_factory (MetricsFactory | None):
            An instance of MetricsFactory for collecting usage metrics
            (default: None).
    """

    segment_store_partition: InstanceOf[SegmentStorePartition] = Field(
        ...,
        description="Segment store partition",
    )
    vector_store_collection: InstanceOf[VectorStoreCollection] = Field(
        ...,
        description="Vector store collection",
    )
    segmenter: InstanceOf[Segmenter] = Field(
        ...,
        description="Segmenter that segments events into segments",
    )
    deriver: InstanceOf[Deriver] = Field(
        ...,
        description="Deriver that derives derivatives from segments",
    )
    embedder: InstanceOf[Embedder] = Field(
        ...,
        description="Embedder instance for creating embeddings",
    )
    metrics_factory: InstanceOf[MetricsFactory] | None = Field(
        None,
        description="An instance of MetricsFactory for collecting usage metrics",
    )


class EventMemory:
    """Event memory system."""

    # The reserved namespace is shared by every MemMachine memory system, so a
    # key names the system that owns it. Built rather than written out so the
    # alphabet and length budget are checked at import time.
    _TIMESTAMP_FIELD_NAME: ClassVar[str] = reserved_property_key("event", "timestamp")

    _RESERVED_PROPERTY_SCHEMA: ClassVar[dict[str, type[PropertyValue]]] = {
        _TIMESTAMP_FIELD_NAME: cast(type[PropertyValue], datetime.datetime),
    }

    @classmethod
    def expected_vector_store_collection_schema(cls) -> dict[str, type[PropertyValue]]:
        """
        Return the vector store collection schema expected by EventMemory.

        Callers should merge this with their own properties when creating the
        collection so that EventMemory's reserved fields are efficiently filterable.
        """
        return dict(cls._RESERVED_PROPERTY_SCHEMA)

    @classmethod
    def _validated_filter_field(cls, field: str) -> str:
        """Return `field` unchanged, raising if it is not a legal property key."""
        validate_caller_property_key(field)
        return field

    @staticmethod
    def _with_time_range(
        property_filter: FilterExpr | None,
        since: datetime.datetime | None,
        before: datetime.datetime | None,
        field: str,
    ) -> FilterExpr | None:
        """
        Conjoin a half-open [since, before) range onto a caller filter.

        `since` is inclusive and `before` is exclusive, so adjacent ranges tile
        without overlapping or leaving a gap and a whole day is
        `since=<day>, before=<next day>`.
        """
        clauses: list[FilterExpr] = []
        if since is not None:
            clauses.append(Ordering(field=field, op=">=", value=since))
        if before is not None:
            clauses.append(Ordering(field=field, op="<", value=before))
        if property_filter is not None:
            clauses.append(property_filter)
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return And(tuple(clauses))

    def __init__(self, params: EventMemoryParams) -> None:
        """
        Initialize an EventMemory with the provided parameters.

        Args:
            params (EventMemoryParams):
                Parameters for the EventMemory.

        """
        self._segment_store_partition = params.segment_store_partition
        self._vector_store_collection = params.vector_store_collection
        self._segmenter = params.segmenter
        self._deriver = params.deriver
        self._embedder = params.embedder

        self._tracker = OperationTracker(
            params.metrics_factory,
            prefix="event_memory",
        )

        self._schema_fields = frozenset(
            params.vector_store_collection.config.indexed_properties_schema
        )

        declared_schema = (
            params.vector_store_collection.config.indexed_properties_schema
        )
        for key, expected_type in EventMemory._RESERVED_PROPERTY_SCHEMA.items():
            declared_type = declared_schema.get(key)
            if declared_type is None:
                raise ValueError(
                    f"Collection schema missing field required by EventMemory: {key}. "
                    f"Merge EventMemory.expected_vector_store_collection_schema() "
                    f"into the collection schema."
                )
            if declared_type is not expected_type:
                raise ValueError(
                    f"Collection schema declares {key} as {declared_type.__name__}, "
                    f"but EventMemory requires {expected_type.__name__}."
                )
        squatted = sorted(
            key
            for key in declared_schema
            if is_reserved_property_key(key)
            and key not in EventMemory._RESERVED_PROPERTY_SCHEMA
        )
        if squatted:
            raise ValueError(
                f"Collection schema declares reserved property keys: "
                f"{', '.join(squatted)}. Keys beginning with "
                f"{RESERVED_PROPERTY_KEY_PREFIX!r} belong to MemMachine."
            )

        self._encode_events_phase_seconds: MetricsFactory.Histogram | None = None
        self._query_phase_seconds: MetricsFactory.Histogram | None = None
        if params.metrics_factory is not None:
            self._encode_events_phase_seconds = params.metrics_factory.get_histogram(
                "event_memory_encode_events_phase_seconds",
                "Time spent in each phase of encode_events",
                label_names=("phase",),
            )
            self._query_phase_seconds = params.metrics_factory.get_histogram(
                "event_memory_query_phase_seconds",
                "Time spent in each phase of query",
                label_names=("phase",),
            )

    def _validate_events(self, events: Iterable[Event]) -> None:
        """
        Validate a batch of events before encoding.

        Raises ValueError if any event supplies a reserved field name in its properties,
        or if the collection schema is missing fields required by EventMemory.
        """
        events = list(events)

        for event in events:
            for key in event.properties:
                validate_caller_property_key(key)

    async def encode_events(
        self,
        events: Iterable[Event],
        *,
        format_options: FormatOptions | None = None,
    ) -> None:
        """
        Encode events.

        Args:
            events (Iterable[Event]): The events to encode.
            format_options (FormatOptions | None):
                Options for formatting.
                (default: None).

        Raises:
            ValueError:
                If any event supplies a reserved field name in its properties,
                or if the collection schema is missing fields required by any event's Context type.
        """
        async with self._tracker("encode_events"):
            await self._encode_events(events, format_options=format_options)

    async def _encode_events(
        self,
        events: Iterable[Event],
        *,
        format_options: FormatOptions | None,
    ) -> None:
        t_start = time.monotonic()

        events = list(events)
        self._validate_events(events)

        segment_lists = await asyncio.gather(
            *(
                self._segmenter.segment(event, format_options=format_options)
                for event in events
            )
        )
        segments = [
            segment for segment_list in segment_lists for segment in segment_list
        ]
        t_segmentation = time.monotonic()

        derivative_lists = await asyncio.gather(
            *(
                self._deriver.derive(segment, format_options=format_options)
                for segment in segments
            )
        )
        segments_to_derivatives: dict[Segment, list[Derivative]] = dict(
            zip(segments, derivative_lists, strict=True)
        )

        derivatives = [
            derivative
            for segment_derivatives in segments_to_derivatives.values()
            for derivative in segment_derivatives
        ]
        t_derivation = time.monotonic()

        derivative_texts: list[str] = []
        for derivative in derivatives:
            text = EventMemory._extract_text(derivative.block)
            if text is None:
                raise NotImplementedError(
                    f"Unsupported block type: {type(derivative.block).__name__}"
                )
            derivative_texts.append(text)

        derivative_embeddings = await self._embedder.ingest_embed(derivative_texts)
        t_embedding = time.monotonic()

        await self._segment_store_partition.add_segments(
            {
                segment: [derivative.uuid for derivative in segment_derivatives]
                for segment, segment_derivatives in segments_to_derivatives.items()
            }
        )
        t_segment_store = time.monotonic()

        derivative_records = [
            EventMemory._build_derivative_record(derivative, derivative_embedding)
            for derivative, derivative_embedding in zip(
                derivatives,
                derivative_embeddings,
                strict=True,
            )
        ]

        if derivative_records:
            await self._vector_store_collection.upsert(records=derivative_records)
        t_vector_store = time.monotonic()

        phase_durations = {
            "segmentation": t_segmentation - t_start,
            "derivation": t_derivation - t_segmentation,
            "embedding": t_embedding - t_derivation,
            "segment_store": t_segment_store - t_embedding,
            "vector_store": t_vector_store - t_segment_store,
        }

        logger.debug(
            "encode_events timing: %s total=%.3fs",
            " ".join(
                f"{phase}={duration:.3f}s"
                for phase, duration in phase_durations.items()
            ),
            t_vector_store - t_start,
        )

        if self._encode_events_phase_seconds is not None:
            for phase, duration in phase_durations.items():
                self._encode_events_phase_seconds.observe(
                    duration, labels={"phase": phase}
                )

    @classmethod
    def _build_derivative_record(
        cls,
        derivative: Derivative,
        derivative_embedding: Sequence[float],
    ) -> Record:
        """Build a vector record from a derivative and its embedding."""
        # Caller properties first: a reserved key cannot reach here (encode_events
        # validates), but building in this order means the invariant is enforced
        # by the merge itself rather than by a check in a different function.
        properties: dict[str, PropertyValue] = dict(derivative.properties)
        properties[cls._TIMESTAMP_FIELD_NAME] = derivative.timestamp

        return Record(
            uuid=derivative.uuid,
            vector=list(derivative_embedding),
            properties=properties,
        )

    async def query(
        self,
        query: str,
        *,
        vector_search_limit: int = 20,
        expand_context: int = 0,
        since: datetime.datetime | None = None,
        before: datetime.datetime | None = None,
        property_filter: FilterExpr | None = None,
    ) -> QueryResult:
        """
        Query event memory for segments relevant to the query.

        Args:
            query (str):
                The search query.
            vector_search_limit (int):
                The maximum number of seed segments
                to retrieve from the vector search
                (default: 20).
            expand_context (int):
                The number of additional segments to include
                around each matched segment for additional context
                (default: 0).
            since (datetime.datetime | None):
                Start of the time range, INCLUSIVE
                (default: None).
            before (datetime.datetime | None):
                End of the time range, EXCLUSIVE, so the range is
                `since <= timestamp < before` and consecutive ranges meet
                without overlapping
                (default: None).
            property_filter (FilterExpr | None):
                Filter over caller-supplied properties.
                Field names are used exactly as written; see
                `validate_property_key` for which keys are legal
                (default: None).

        Returns:
            QueryResult:
                The query result.

        """
        async with self._tracker("query"):
            return await self._query(
                query,
                vector_search_limit=vector_search_limit,
                expand_context=expand_context,
                since=since,
                before=before,
                property_filter=property_filter,
            )

    async def _query(
        self,
        query: str,
        *,
        vector_search_limit: int,
        expand_context: int,
        since: datetime.datetime | None,
        before: datetime.datetime | None,
        property_filter: FilterExpr | None,
    ) -> QueryResult:
        t_start = time.monotonic()
        query_embedding = (
            await self._embedder.search_embed(
                [query],
            )
        )[0]
        t_embedding = time.monotonic()

        # Reject caller filter fields that are not legal property keys, so an
        # unfilterable name fails loudly instead of silently matching nothing.
        if property_filter is not None:
            map_filter_fields(property_filter, EventMemory._validated_filter_field)

        # Caller filter fields reach the store exactly as written; only the
        # time range, which addresses a reserved key, is assembled here.
        collection_filter = EventMemory._with_time_range(
            property_filter, since, before, EventMemory._TIMESTAMP_FIELD_NAME
        )

        # Search derivative collection for matches.
        [query_result] = await self._vector_store_collection.query(
            query_vectors=[query_embedding],
            limit=vector_search_limit,
            property_filter=collection_filter,
        )
        t_vector_query = time.monotonic()

        # Map matched derivatives to their owning segments.
        segment_uuids_by_derivative_uuid = (
            await self._segment_store_partition.get_segment_uuids_by_derivative_uuids(
                match.record_uuid for match in query_result.matches
            )
        )

        # Extract seed segment UUIDs and their best embedding cosine similarities.
        # Deduplicate by first occurrence (multiple derivatives can map to the same segment).
        # First occurrence is the best since matches are ordered best-to-worst.
        seed_embedding_cosine_similarities: dict[UUID, float] = {}
        for match in query_result.matches:
            segment_uuid = segment_uuids_by_derivative_uuid.get(match.record_uuid)
            if segment_uuid is None:
                continue
            if segment_uuid not in seed_embedding_cosine_similarities:
                seed_embedding_cosine_similarities[segment_uuid] = (
                    match.cosine_similarity
                )

        seed_segment_uuids = list(seed_embedding_cosine_similarities)

        max_backward_segments = expand_context // 3
        max_forward_segments = expand_context - max_backward_segments

        segment_contexts_by_seed = (
            await self._segment_store_partition.get_segment_contexts(
                seed_segment_uuids=seed_segment_uuids,
                max_backward_segments=max_backward_segments,
                max_forward_segments=max_forward_segments,
                since=since,
                before=before,
                property_filter=property_filter,
            )
        )
        t_segment_query = time.monotonic()

        # Filter to seeds with results, preserving cosine similarity order.
        kept_seed_segment_uuids = [
            seed_segment_uuid
            for seed_segment_uuid in seed_segment_uuids
            if seed_segment_uuid in segment_contexts_by_seed
        ]
        segment_contexts: list[list[Segment]] = [
            segment_contexts_by_seed[seed_segment_uuid]
            for seed_segment_uuid in kept_seed_segment_uuids
        ]

        cosine_similarities = [
            seed_embedding_cosine_similarities[seed_uuid]
            for seed_uuid in kept_seed_segment_uuids
        ]
        t_scoring = time.monotonic()

        # Return matches ordered by cosine similarity, best first.
        matches = [
            SegmentContextMatch(
                cosine_similarity=cosine_similarity,
                seed_segment_uuid=seed_uuid,
                segments=context,
            )
            for cosine_similarity, seed_uuid, context in sorted(
                zip(
                    cosine_similarities,
                    kept_seed_segment_uuids,
                    segment_contexts,
                    strict=True,
                ),
                key=lambda triple: triple[0],
                reverse=True,
            )
        ]

        phase_durations = {
            "embedding": t_embedding - t_start,
            "vector_query": t_vector_query - t_embedding,
            "segment_query": t_segment_query - t_vector_query,
            "scoring": t_scoring - t_segment_query,
        }

        logger.debug(
            "query timing: %s total=%.3fs",
            " ".join(
                f"{phase}={duration:.3f}s"
                for phase, duration in phase_durations.items()
            ),
            time.monotonic() - t_start,
        )

        if self._query_phase_seconds is not None:
            for phase, duration in phase_durations.items():
                self._query_phase_seconds.observe(duration, labels={"phase": phase})

        return QueryResult(matches=matches)

    @staticmethod
    def string_from_segment_context(
        segment_context: Iterable[Segment],
        *,
        format_options: FormatOptions | None = None,
    ) -> str:
        """Format segment context as a string."""
        if format_options is None:
            format_options = FormatOptions(time_style="short")

        context_string = ""
        last_segment: Segment | None = None
        accumulated_text = ""
        first = True

        for segment in segment_context:
            is_continuation = (
                last_segment is not None
                and segment.event_uuid == last_segment.event_uuid
                and segment.index == last_segment.index
            )

            if not is_continuation:
                if not first:
                    context_string += (
                        json.dumps(accumulated_text, ensure_ascii=False) + "\n"
                    )
                first = False
                accumulated_text = ""
                context_string += EventMemory._segment_header(segment, format_options)

            text = EventMemory._extract_text(segment.block)
            if text is not None:
                accumulated_text += text
            elif not is_continuation:
                context_string += f"[{segment.block.block_type}]\n"

            last_segment = segment

        if not first:
            context_string += json.dumps(accumulated_text, ensure_ascii=False) + "\n"

        return context_string.strip()

    @staticmethod
    def string_from_segment_contexts(
        segment_contexts: Iterable[Iterable[Segment]],
        *,
        format_options: FormatOptions | None = None,
    ) -> str:
        """Format multiple segment contexts as a string, separating disconnected components."""
        segment_contexts = [list(context) for context in segment_contexts]

        # Deduplicate segments and build union-find over their UUIDs in one pass.
        segments_by_uuid: dict[UUID, Segment] = {}
        component_parent: dict[UUID, UUID] = {}

        def find(uuid: UUID) -> UUID:
            component_parent.setdefault(uuid, uuid)
            root = uuid
            while component_parent[root] != root:
                root = component_parent[root]
            while component_parent[uuid] != root:
                parent = component_parent[uuid]
                component_parent[uuid] = root
                uuid = parent
            return root

        for context in segment_contexts:
            first_segment_root: UUID | None = None
            for segment in context:
                segments_by_uuid.setdefault(segment.uuid, segment)
                if first_segment_root is None:
                    first_segment_root = find(segment.uuid)
                else:
                    segment_root = find(segment.uuid)
                    component_parent[segment_root] = first_segment_root

        # Group unique segments by component root.
        segments_by_root: dict[UUID, list[Segment]] = {}
        for segment_uuid, segment in segments_by_uuid.items():
            segments_by_root.setdefault(find(segment_uuid), []).append(segment)

        # Sort segments within each component, then order components chronologically.
        def segment_key(segment: Segment) -> tuple:
            return (
                segment.timestamp,
                segment.event_uuid,
                segment.index,
                segment.offset,
            )

        components = list(segments_by_root.values())
        for component in components:
            component.sort(key=segment_key)
        components.sort(key=lambda segments: segment_key(segments[0]))

        return "\n\n".join(
            EventMemory.string_from_segment_context(
                segments, format_options=format_options
            )
            for segments in components
        )

    @staticmethod
    def _segment_header(segment: Segment, format_options: FormatOptions) -> str:
        """Build the header emitted before a segment."""
        formatted_timestamp = format_timestamp(segment.timestamp, format_options)
        timestamp_prefix = f"[{formatted_timestamp}] " if formatted_timestamp else ""

        match segment.context:
            case ProducerContext(producer=producer):
                return f"{timestamp_prefix}{producer}: "
            case NullContext():
                return timestamp_prefix
            case _:
                raise NotImplementedError(
                    f"Unsupported context type: {type(segment.context).__name__}"
                )

    @staticmethod
    def _extract_text(block: Block) -> str | None:
        """Extract text from a block, if it contains text."""
        match block:
            case TextBlock(text=text):
                return text
            case _:
                return None

    async def forget_events(self, event_uuids: Iterable[UUID]) -> None:
        """Forget events by their UUIDs."""
        event_uuids = set(event_uuids)
        if not event_uuids:
            return

        async with self._tracker("forget_events"):
            await self._forget_events(event_uuids)

    async def _forget_events(self, event_uuids: set[UUID]) -> None:

        # Snapshot segment UUIDs for these events.
        segments_by_event = (
            await self._segment_store_partition.get_segment_uuids_by_event_uuids(
                event_uuids=event_uuids,
            )
        )
        segment_uuids = {
            segment_uuid
            for event_segment_uuids in segments_by_event.values()
            for segment_uuid in event_segment_uuids
        }
        if not segment_uuids:
            return

        # Get derivative UUIDs for those segments.
        derivatives_by_segment = (
            await self._segment_store_partition.get_derivative_uuids_by_segment_uuids(
                segment_uuids=segment_uuids,
            )
        )
        derivative_uuids = {
            derivative_uuid
            for segment_derivative_uuids in derivatives_by_segment.values()
            for derivative_uuid in segment_derivative_uuids
        }

        # Delete from vector DB first, then segment store.
        if derivative_uuids:
            await self._vector_store_collection.delete(record_uuids=derivative_uuids)

        await self._segment_store_partition.delete_segments(
            segment_uuids=segment_uuids,
        )

    @staticmethod
    def build_query_result_context(
        query_result: QueryResult,
        max_num_segments: int,
    ) -> list[Segment]:
        """
        Build a single segment context from the query result within the limit.

        Iterates contexts in cosine similarity order, accumulating segments until the limit is reached.
        When a context would exceed the limit, segments nearest the seed are prioritized.
        Deduplicates across segment contexts in the query result.

        Args:
            query_result (QueryResult):
                The query result with anchored segment context matches.
            max_num_segments (int):
                The maximum number of segments to return.

        Returns:
            list[Segment]:
                Deduplicated segments ordered chronologically.
        """
        unified: set[Segment] = set()

        for match in query_result.matches:
            context = match.segments

            if len(unified) >= max_num_segments:
                break
            if (len(unified) + len(context)) <= max_num_segments:
                unified.update(context)
            else:
                # Prioritize segments near the seed segment.
                seed_index = next(
                    index
                    for index, segment in enumerate(context)
                    if segment.uuid == match.seed_segment_uuid
                )

                for segment in sorted(
                    context,
                    key=lambda s: EventMemory._seed_proximity(s, context, seed_index),
                ):
                    if len(unified) >= max_num_segments:
                        break
                    unified.add(segment)

        return sorted(
            unified,
            key=lambda segment: (
                segment.timestamp,
                segment.event_uuid,
                segment.index,
                segment.offset,
            ),
        )

    @staticmethod
    def string_from_query_result(
        query_result: QueryResult,
        *,
        max_num_segments: int | None = None,
        format_options: FormatOptions | None = None,
    ) -> str:
        """Format a query result as a string with breaks between disconnected contexts."""
        contexts: list[list[Segment]] = [
            list(match.segments) for match in query_result.matches
        ]

        if max_num_segments is not None:
            included = {
                segment.uuid
                for segment in EventMemory.build_query_result_context(
                    query_result, max_num_segments
                )
            }
            contexts = [
                [segment for segment in context if segment.uuid in included]
                for context in contexts
            ]

        return EventMemory.string_from_segment_contexts(
            contexts, format_options=format_options
        )

    @staticmethod
    def _seed_proximity(
        segment: Segment,
        context: list[Segment],
        seed_index: int,
    ) -> float:
        """Score a segment by its proximity to the seed. Lower is closer."""
        offset = context.index(segment) - seed_index
        if offset >= 0:
            # Forward context is more useful than backward.
            return (offset - 0.5) / 2
        return -offset
