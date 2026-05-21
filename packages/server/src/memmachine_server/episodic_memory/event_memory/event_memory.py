"""Event memory system for storing and retrieving events."""

import asyncio
import datetime
import json
import logging
import math
import threading
import time
from collections.abc import Iterable, Sequence
from typing import ClassVar, cast
from uuid import UUID

import spacy
from babel.dates import format_date, format_time, get_datetime_format
from pydantic import BaseModel, Field, InstanceOf
from rank_bm25 import BM25Okapi

from memmachine_server.common.data_types import PropertyValue
from memmachine_server.common.embedder import Embedder
from memmachine_server.common.filter.filter_parser import (
    FilterExpr,
    demangle_user_metadata_key,
    map_filter_fields,
    normalize_filter_field,
)
from memmachine_server.common.metrics_factory import (
    MetricsFactory,
    OperationTracker,
)
from memmachine_server.common.reranker import Reranker
from memmachine_server.common.vector_store import (
    Record,
    VectorStoreCollection,
)

from .data_types import (
    Block,
    DateTimeStyle,
    DecoupledRetrievalContext,
    Derivative,
    Event,
    FormatOptions,
    NullContext,
    ProducerContext,
    QueryResult,
    RawSegmentEventContext,
    RewriteContext,
    ScoredSegmentContext,
    Segment,
    SurroundingEventsContext,
    TextBlock,
)
from .deriver import Deriver
from .segment_store import SegmentStorePartition
from .segmenter import Segmenter

logger = logging.getLogger(__name__)

# CLDR datetime style levels, ordered from compact to verbose.
_DATETIME_STYLE_LEVELS: tuple[DateTimeStyle, ...] = ("short", "medium", "long", "full")

# Supported BM25 fusion modes for EventMemory.query.
_BM25_FUSION_MODES: frozenset[str] = frozenset({"none", "additive", "rrf", "rsf", "noisy-or"})

# Reciprocal Rank Fusion smoothing constant.
_RRF_K: float = 60.0

# Lazy spaCy lookup-mode lemma pipeline for BM25 fusion tokenization.
# A blank English model + lookup lemmatizer (lemma table from
# spacy-lookups-data) is ~100x faster than `en_core_web_sm` because it
# skips tok2vec/tagger inference. POS-dependent lemma ambiguity (e.g.
# "meeting" noun vs verb) is hedged by also retaining the original `-ing`
# surface form, matching mem0's `lemmatize_for_bm25`.
_bm25_lemma_nlp: spacy.language.Language | None = None
_bm25_lemma_lock = threading.Lock()


def _get_bm25_lemma_nlp() -> spacy.language.Language:
    global _bm25_lemma_nlp
    if _bm25_lemma_nlp is not None:
        return _bm25_lemma_nlp
    with _bm25_lemma_lock:
        if _bm25_lemma_nlp is not None:
            return _bm25_lemma_nlp
        from spacy.lookups import load_lookups

        nlp = spacy.blank("en")
        lemmatizer = nlp.add_pipe("lemmatizer", config={"mode": "lookup"})
        lemmatizer.initialize(lookups=load_lookups("en", ["lemma_lookup"]))
        _bm25_lemma_nlp = nlp
        return _bm25_lemma_nlp


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
        reranker (Reranker | None):
            Reranker instance for scoring search results.
            If None, embedding similarity scores are used instead
            (default: None).
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
    reranker: InstanceOf[Reranker] | None = Field(
        None,
        description="Reranker instance for scoring search results. "
        "If None, embedding similarity scores are used instead",
    )
    metrics_factory: InstanceOf[MetricsFactory] | None = Field(
        None,
        description="An instance of MetricsFactory for collecting usage metrics",
    )


class EventMemory:
    """Event memory system."""

    # System-defined metadata field names. Reserved.
    _SEGMENT_UUID_FIELD_NAME = "_segment_uuid"
    _TIMESTAMP_FIELD_NAME = "_timestamp"

    _BASE_EVENT_MEMORY_FIELD_NAMES: ClassVar[frozenset[str]] = frozenset(
        {_SEGMENT_UUID_FIELD_NAME, _TIMESTAMP_FIELD_NAME}
    )

    @classmethod
    def expected_vector_store_collection_schema(cls) -> dict[str, type[PropertyValue]]:
        """
        Return the vector store collection schema expected by EventMemory.

        Callers should merge this with any user or external system-defined properties
        when creating the collection so that EventMemory's reserved fields are efficiently filterable.
        """
        return {
            cls._SEGMENT_UUID_FIELD_NAME: cast(type[PropertyValue], str),
            cls._TIMESTAMP_FIELD_NAME: cast(type[PropertyValue], datetime.datetime),
        }

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
        self._reranker = params.reranker

        self._tracker = OperationTracker(
            params.metrics_factory,
            prefix="event_memory",
        )

        self._schema_fields = frozenset(
            params.vector_store_collection.config.indexed_properties_schema
        )

        missing_base_fields = (
            EventMemory._BASE_EVENT_MEMORY_FIELD_NAMES - self._schema_fields
        )
        if missing_base_fields:
            raise ValueError(
                f"Collection schema missing fields required by EventMemory: "
                f"{', '.join(sorted(missing_base_fields))}"
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

        reserved_fields = {
            field
            for event in events
            for field in event.properties
            if field in EventMemory._BASE_EVENT_MEMORY_FIELD_NAMES
        }
        if reserved_fields:
            raise ValueError(
                f"Event properties must not contain reserved fields: "
                f"{', '.join(sorted(reserved_fields))}"
            )

    async def encode_events(
        self,
        events: Iterable[Event],
    ) -> None:
        """
        Encode events.

        Args:
            events (Iterable[Event]): The events to encode.

        Raises:
            ValueError:
                If any event supplies a reserved field name in its properties,
                or if the collection schema is missing fields required by any event's Context type.
        """
        async with self._tracker("encode_events"):
            await self._encode_events(events)

    async def _encode_events(
        self,
        events: Iterable[Event],
    ) -> None:
        t_start = time.monotonic()

        events = list(events)
        self._validate_events(events)

        segment_lists = await asyncio.gather(
            *(self._segmenter.segment(event) for event in events)
        )
        segments = [
            segment for segment_list in segment_lists for segment in segment_list
        ]
        t_segmentation = time.monotonic()

        derivative_lists = await asyncio.gather(
            *(self._deriver.derive(segment) for segment in segments)
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
        properties: dict[str, PropertyValue] = {}

        # System-defined metadata (underscore-prefixed).
        properties[cls._SEGMENT_UUID_FIELD_NAME] = str(derivative.segment_uuid)
        properties[cls._TIMESTAMP_FIELD_NAME] = derivative.timestamp

        # User-defined properties.
        properties.update(derivative.properties)

        return Record(
            uuid=derivative.uuid,
            vector=list(derivative_embedding),
            properties=properties,
        )

    @classmethod
    def _to_vector_record_property(cls, field: str) -> str:
        """
        Translates canonical filter field name to vector record property.

        Event memory base properties (`foo`) translate to `_foo`.
        User-defined properties (`m.foo` / `metadata.foo`) translate to `foo`.
        """
        internal_name, is_user_metadata = normalize_filter_field(field)
        if is_user_metadata:
            return demangle_user_metadata_key(internal_name)
        return f"_{field}"

    async def query(
        self,
        query: str,
        *,
        vector_search_limit: int = 20,
        expand_context: int = 0,
        property_filter: FilterExpr | None = None,
        format_options: FormatOptions | None = None,
        bm25_fusion: str = "none",
        bm25_fusion_weight: float = 0.5,
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
            property_filter (FilterExpr | None):
                Property fields and values
                to use for filtering segments
                (default: None).
            format_options (FormatOptions | None):
                Options for formatting.
                (default: None).
            bm25_fusion (str):
                BM25 fusion mode applied to the vector-retrieved candidate
                pool (each segment context is rendered to a single string
                and treated as one BM25 document). One of:

                - ``"none"``: skip fusion (default).
                - ``"additive"``: calibrated additive fusion.
                  ``combined = w_sem * semantic + w_bm25 * sigmoid(bm25)``
                  with a length-adaptive sigmoid; weights from
                  ``bm25_fusion_weight``. Requires higher-is-better
                  underlying scores.
                - ``"rrf"``: Reciprocal Rank Fusion (k=60).
                  ``combined = 1/(k + rank_sem) + 1/(k + rank_bm25)``.
                  Weight knob does not apply. Works with either
                  similarity metric direction.
                - ``"rsf"``: Weaviate-style Relative Score Fusion.
                  Max-normalize each channel within the result set then
                  weight-sum with ``bm25_fusion_weight``. Requires
                  higher-is-better underlying scores.

            bm25_fusion_weight (float):
                BM25 channel weight in ``[0.0, 1.0]`` for ``"additive"``
                and ``"rsf"`` modes; semantic weight is ``1 - weight``.
                ``0.5`` (default) is equal weighting. Ignored for
                ``"rrf"``.

        Returns:
            QueryResult:
                The query result.

        """
        async with self._tracker("query"):
            return await self._query(
                query,
                vector_search_limit=vector_search_limit,
                expand_context=expand_context,
                property_filter=property_filter,
                format_options=format_options,
                bm25_fusion=bm25_fusion,
                bm25_fusion_weight=bm25_fusion_weight,
            )

    async def _query(
        self,
        query: str,
        *,
        vector_search_limit: int,
        expand_context: int,
        property_filter: FilterExpr | None,
        format_options: FormatOptions | None,
        bm25_fusion: str,
        bm25_fusion_weight: float,
    ) -> QueryResult:
        if bm25_fusion not in _BM25_FUSION_MODES:
            raise ValueError(
                f"bm25_fusion={bm25_fusion!r} is not one of "
                f"{sorted(_BM25_FUSION_MODES)!r}."
            )
        if not 0.0 <= bm25_fusion_weight <= 1.0:
            raise ValueError(
                f"bm25_fusion_weight={bm25_fusion_weight!r} must be in [0.0, 1.0]."
            )
        if format_options is None:
            format_options = FormatOptions()

        t_start = time.monotonic()
        query_embedding = (
            await self._embedder.search_embed(
                [query],
            )
        )[0]
        t_embedding = time.monotonic()

        # Translate filter fields for vector store.
        collection_filter = (
            map_filter_fields(property_filter, EventMemory._to_vector_record_property)
            if property_filter is not None
            else None
        )

        # Search derivative collection for matches.
        [query_result] = await self._vector_store_collection.query(
            query_vectors=[query_embedding],
            limit=vector_search_limit,
            property_filter=collection_filter,
            return_vector=False,
            return_properties=True,
        )
        t_vector_query = time.monotonic()

        # Extract seed segment UUIDs and their best embedding scores.
        # Deduplicate by first occurrence (multiple derivatives can map to the same segment).
        # First occurrence has the best score since matches are ordered best-to-worst.
        seed_embedding_scores: dict[UUID, float] = {}
        for match in query_result.matches:
            segment_uuid = UUID(
                str(
                    cast(
                        dict[str, PropertyValue],
                        match.record.properties,
                    )[EventMemory._SEGMENT_UUID_FIELD_NAME]
                )
            )
            if segment_uuid not in seed_embedding_scores:
                seed_embedding_scores[segment_uuid] = match.score

        seed_segment_uuids = list(seed_embedding_scores)

        max_backward_segments = expand_context // 3
        max_forward_segments = expand_context - max_backward_segments

        segment_contexts_by_seed = (
            await self._segment_store_partition.get_segment_contexts(
                seed_segment_uuids=seed_segment_uuids,
                max_backward_segments=max_backward_segments,
                max_forward_segments=max_forward_segments,
                property_filter=property_filter,
            )
        )
        t_segment_query = time.monotonic()

        # Filter to seeds with results, preserving similarity order.
        kept_seed_segment_uuids = [
            seed_segment_uuid
            for seed_segment_uuid in seed_segment_uuids
            if seed_segment_uuid in segment_contexts_by_seed
        ]
        segment_contexts: list[list[Segment]] = [
            segment_contexts_by_seed[seed_segment_uuid]
            for seed_segment_uuid in kept_seed_segment_uuids
        ]

        # Use embedding scores if reranker is not available.
        if self._reranker is None:
            scores = [
                seed_embedding_scores[seed_uuid]
                for seed_uuid in kept_seed_segment_uuids
            ]
        else:
            scores = await self._score_segment_contexts(
                query, segment_contexts, format_options
            )
        t_scoring = time.monotonic()

        # Reranker scores are always higher-is-better.
        # Embedding scores depend on the similarity metric.
        higher_is_better = (
            self._reranker is not None
            or self._vector_store_collection.config.similarity_metric.higher_is_better
        )

        if bm25_fusion != "none":
            if bm25_fusion in {"additive", "rsf", "noisy-or"} and not higher_is_better:
                raise ValueError(
                    f"bm25_fusion={bm25_fusion!r} requires higher-is-better "
                    "underlying scores; reranker is not set and similarity "
                    "metric is lower-is-better."
                )
            scores = await EventMemory._fuse_bm25_scores(
                mode=bm25_fusion,
                query=query,
                segment_contexts=segment_contexts,
                semantic_scores=scores,
                higher_is_better=higher_is_better,
                bm25_weight=bm25_fusion_weight,
                format_options=format_options,
            )
            # All fusion modes produce higher-is-better outputs.
            higher_is_better = True
        t_bm25_fusion = time.monotonic()

        # Return scored contexts ordered by score.
        scored_segment_contexts = [
            ScoredSegmentContext(
                score=score, seed_segment_uuid=seed_uuid, segments=context
            )
            for score, seed_uuid, context in sorted(
                zip(
                    scores,
                    kept_seed_segment_uuids,
                    segment_contexts,
                    strict=True,
                ),
                key=lambda triple: triple[0],
                reverse=higher_is_better,
            )
        ]

        phase_durations = {
            "embedding": t_embedding - t_start,
            "vector_query": t_vector_query - t_embedding,
            "segment_query": t_segment_query - t_vector_query,
            "scoring": t_scoring - t_segment_query,
            "bm25_fusion": t_bm25_fusion - t_scoring,
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

        return QueryResult(scored_segment_contexts=scored_segment_contexts)

    async def _score_segment_contexts(
        self,
        query: str,
        segment_contexts: Iterable[Iterable[Segment]],
        format_options: FormatOptions,
    ) -> list[float]:
        """Score segment contexts using the reranker. Requires reranker."""
        assert self._reranker is not None
        context_strings = [
            EventMemory.string_from_segment_context(
                segment_context, format_options=format_options
            )
            for segment_context in segment_contexts
        ]
        return await self._reranker.score(query, context_strings)

    @staticmethod
    async def _fuse_bm25_scores(
        *,
        mode: str,
        query: str,
        segment_contexts: Sequence[Sequence[Segment]],
        semantic_scores: Sequence[float],
        higher_is_better: bool,
        bm25_weight: float,
        format_options: FormatOptions,
    ) -> list[float]:
        """
        Fuse BM25 over the candidate pool with the underlying scores.

        Dispatches on ``mode``:

        - ``"additive"``: mem0-style calibrated additive fusion with the
          length-adaptive sigmoid + max_possible divisor.
        - ``"rrf"``: Reciprocal Rank Fusion (k=60), ranks computed within
          the candidate pool and respecting the semantic metric direction.
        - ``"rsf"``: Weaviate-style Relative Score Fusion, max-normalize
          each channel within the pool then equally weight-average.

        Each segment context is rendered to one string and treated as one
        BM25 document. The output is always higher-is-better.
        """
        if not segment_contexts:
            return list(semantic_scores)

        context_strings = [
            EventMemory.string_from_segment_context(
                context, format_options=format_options, for_bm25=True
            )
            for context in segment_contexts
        ]

        raw_bm25, num_query_terms = await asyncio.to_thread(
            EventMemory._compute_raw_bm25, query, context_strings
        )

        if mode == "additive":
            return EventMemory._fuse_additive(
                semantic_scores=semantic_scores,
                raw_bm25=raw_bm25,
                num_query_terms=num_query_terms,
                bm25_weight=bm25_weight,
            )
        if mode == "rrf":
            # RRF combines ranks; weights don't apply.
            return EventMemory._fuse_rrf(
                semantic_scores=semantic_scores,
                raw_bm25=raw_bm25,
                higher_is_better=higher_is_better,
            )
        if mode == "rsf":
            return EventMemory._fuse_rsf(
                semantic_scores=semantic_scores,
                raw_bm25=raw_bm25,
                bm25_weight=bm25_weight,
            )
        if mode == "noisy-or":
            return EventMemory._fuse_noisy_or(
                semantic_scores=semantic_scores,
                raw_bm25=raw_bm25,
                num_query_terms=num_query_terms,
            )
        raise ValueError(f"Unsupported bm25_fusion mode: {mode!r}")

    @staticmethod
    def _compute_raw_bm25(
        query: str, context_strings: Sequence[str]
    ) -> tuple[list[float], int]:
        """
        Compute raw ``BM25Okapi`` scores for each context.

        Returns ``(raw_scores, num_query_terms)``. Negative raw scores are
        clamped to 0. ``num_query_terms`` is the count of query lemmas
        after stopword/punct removal, used to pick sigmoid params for the
        additive fusion mode.
        """
        # Batch query + corpus through one nlp.pipe() call so spaCy can
        # amortize tok2vec/tagger overhead across the batch.
        nlp = _get_bm25_lemma_nlp()
        texts = [query.lower()] + [s.lower() for s in context_strings]
        docs = list(nlp.pipe(texts, batch_size=64))
        tokenized = [EventMemory._lemmas_from_doc(doc) for doc in docs]
        tokenized_query = tokenized[0]
        tokenized_corpus = tokenized[1:]

        if not tokenized_query or not any(tokenized_corpus):
            return [0.0 for _ in context_strings], len(tokenized_query)

        bm25 = BM25Okapi(tokenized_corpus)
        raw_scores = bm25.get_scores(tokenized_query)
        return [max(0.0, float(s)) for s in raw_scores], len(tokenized_query)

    @staticmethod
    def _fuse_additive(
        *,
        semantic_scores: Sequence[float],
        raw_bm25: Sequence[float],
        num_query_terms: int,
        bm25_weight: float,
    ) -> list[float]:
        """
        Weighted calibrated additive fusion. Higher-is-better output.

        ``combined = (1 - bm25_weight) * semantic + bm25_weight * sigmoid(bm25)``,
        clamped to ``[0, 1]``. With ``bm25_weight = 0.5`` this is the mem0
        equal-weighting recipe.
        """
        midpoint, steepness = EventMemory._bm25_sigmoid_params(num_query_terms)
        normalized_bm25 = [
            EventMemory._bm25_sigmoid(float(raw), midpoint, steepness)
            if raw > 0.0
            else 0.0
            for raw in raw_bm25
        ]
        sem_weight = 1.0 - bm25_weight
        return [
            min(sem_weight * float(semantic) + bm25_weight * bm25_score, 1.0)
            for semantic, bm25_score in zip(
                semantic_scores, normalized_bm25, strict=True
            )
        ]

    @staticmethod
    def _fuse_noisy_or(
        *,
        semantic_scores: Sequence[float],
        raw_bm25: Sequence[float],
        num_query_terms: int,
    ) -> list[float]:
        """
        Noisy-OR / probabilistic-union fusion. Higher-is-better output.

        ``combined = 1 - (1 - semantic) * (1 - bm25_normalized)``
                  = ``semantic + bm25_normalized - semantic * bm25_normalized``
                  = ``semantic + (1 - semantic) * bm25_normalized``

        Treats each channel's score as P(relevant) and computes the
        independent-event union. Output in ``[0, 1]``. BM25 is normalized
        via the same length-adaptive sigmoid used by ``additive``, so the
        normalization methodology matches the standard ``additive`` recipe.
        Semantic scores are clamped to ``[0, 1]`` before combining.
        """
        midpoint, steepness = EventMemory._bm25_sigmoid_params(num_query_terms)
        normalized_bm25 = [
            EventMemory._bm25_sigmoid(float(raw), midpoint, steepness)
            if raw > 0.0
            else 0.0
            for raw in raw_bm25
        ]
        return [
            1.0 - (1.0 - max(0.0, min(1.0, float(semantic))))
                * (1.0 - bm25_score)
            for semantic, bm25_score in zip(
                semantic_scores, normalized_bm25, strict=True
            )
        ]

    @staticmethod
    def _fuse_rrf(
        *,
        semantic_scores: Sequence[float],
        raw_bm25: Sequence[float],
        higher_is_better: bool,
    ) -> list[float]:
        """
        Reciprocal Rank Fusion within the candidate pool. ``k=_RRF_K``.

        Combined score is ``1/(k + rank_sem) + 1/(k + rank_bm25)`` with
        ranks 0-indexed. Ties resolve to the position of first appearance
        (stable sort), so candidates with identical scores share the
        nearby rank and pick up the same RRF contribution within rounding.
        """
        sem_ranks = EventMemory._ranks(
            semantic_scores, higher_is_better=higher_is_better
        )
        bm25_ranks = EventMemory._ranks(raw_bm25, higher_is_better=True)
        return [
            1.0 / (_RRF_K + sem_ranks[i]) + 1.0 / (_RRF_K + bm25_ranks[i])
            for i in range(len(semantic_scores))
        ]

    @staticmethod
    def _fuse_rsf(
        *,
        semantic_scores: Sequence[float],
        raw_bm25: Sequence[float],
        bm25_weight: float,
    ) -> list[float]:
        """
        Weaviate-style Relative Score Fusion within the candidate pool.

        Each channel is max-normalized (score / max_in_pool) then
        weight-summed using ``bm25_weight`` (semantic weight is
        ``1 - bm25_weight``). When a channel's max is 0 (e.g. BM25 has
        no token overlap with any candidate), that channel contributes
        0. Output is in ``[0, 1]`` and higher-is-better.
        """
        sem = [float(s) for s in semantic_scores]
        bm = [float(s) for s in raw_bm25]

        max_sem = max(sem, default=0.0)
        max_bm = max(bm, default=0.0)

        def _norm(score: float, channel_max: float) -> float:
            return (score / channel_max) if channel_max > 0.0 else 0.0

        sem_weight = 1.0 - bm25_weight
        return [
            sem_weight * _norm(s, max_sem) + bm25_weight * _norm(b, max_bm)
            for s, b in zip(sem, bm, strict=True)
        ]

    @staticmethod
    def _ranks(scores: Sequence[float], *, higher_is_better: bool) -> list[int]:
        """0-indexed ranks via stable sort; ties keep insertion order."""
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda pair: pair[1], reverse=higher_is_better)
        ranks = [0] * len(scores)
        for rank, (original_index, _) in enumerate(indexed):
            ranks[original_index] = rank
        return ranks

    @staticmethod
    def _bm25_sigmoid_params(num_terms: int) -> tuple[float, float]:
        """Length-adaptive sigmoid (midpoint, steepness) from mem0."""
        if num_terms <= 3:
            return 5.0, 0.7
        if num_terms <= 6:
            return 7.0, 0.6
        if num_terms <= 9:
            return 9.0, 0.5
        if num_terms <= 15:
            return 10.0, 0.5
        return 12.0, 0.5

    @staticmethod
    def _bm25_sigmoid(raw: float, midpoint: float, steepness: float) -> float:
        """Logistic sigmoid mapping a raw BM25 score to [0, 1]."""
        return 1.0 / (1.0 + math.exp(-steepness * (raw - midpoint)))

    @staticmethod
    def _lemmas_from_doc(doc: spacy.tokens.Doc) -> list[str]:
        """
        Extract BM25 tokens from a spaCy ``Doc`` (verbatim port of mem0).

        Drops punctuation and stopwords, keeps alphanumeric lemmas, and
        additionally retains the original ``-ing`` surface form when it
        differs from the lemma to hedge against noun/verb ambiguity
        (e.g. "meeting" vs "meet").
        """
        tokens: list[str] = []
        for token in doc:
            if token.is_punct or token.is_stop:
                continue
            lemma = token.lemma_
            if lemma.isalnum():
                tokens.append(lemma)
            if (
                token.text.endswith("ing")
                and token.text != lemma
                and token.text.isalnum()
            ):
                tokens.append(token.text)
        return tokens

    @staticmethod
    def _segment_text_for(segment: Segment, *, for_bm25: bool) -> str | None:
        """Pick the text a segment contributes to a rendered context.

        Display (``for_bm25=False``) always uses ``block.text`` -- the
        verbatim text shown to the answering model. BM25 scoring
        (``for_bm25=True``) uses ``context.text_to_score_bm25`` when the
        segment carries a ``DecoupledRetrievalContext``, so lexical
        retrieval scores the clean reference-resolved rewrite instead of
        the raw conversational text (which false-matches vocatives /
        filler). Segments without that context fall back to ``block.text``.
        """
        if for_bm25:
            match segment.context:
                case DecoupledRetrievalContext(text_to_score_bm25=bm25_text):
                    return bm25_text
        return EventMemory._extract_text(segment.block)

    @staticmethod
    def string_from_segment_context(
        segment_context: Iterable[Segment],
        *,
        format_options: FormatOptions | None = None,
        for_bm25: bool = False,
    ) -> str:
        """Format segment context as a string.

        When ``for_bm25`` is set, segments carrying a
        ``DecoupledRetrievalContext`` contribute ``text_to_score_bm25``
        instead of their display ``block.text``; the per-context
        concatenation and headers are otherwise identical, so BM25 still
        scores one string per concatenated segment context.
        """
        if format_options is None:
            format_options = FormatOptions()

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

            text = EventMemory._segment_text_for(segment, for_bm25=for_bm25)
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
        formatted_timestamp = EventMemory._format_timestamp(
            segment.timestamp,
            date_style=format_options.date_style,
            time_style=format_options.time_style,
            locale=format_options.locale,
            timezone=format_options.timezone,
        )
        timestamp_prefix = f"[{formatted_timestamp}] " if formatted_timestamp else ""

        match segment.context:
            case ProducerContext(producer=producer):
                return f"{timestamp_prefix}{producer}: "
            case SurroundingEventsContext(producer=producer):
                return f"{timestamp_prefix}{producer}: "
            case RawSegmentEventContext(producer=producer):
                return f"{timestamp_prefix}{producer}: "
            case NullContext() | RewriteContext() | DecoupledRetrievalContext():
                # RewriteContext / DecoupledRetrievalContext display like
                # NullContext -- their retrieval texts are decoupled from
                # the rendered block, and the block text itself is a
                # self-contained statement that names its own subject.
                return timestamp_prefix
            case _:
                raise NotImplementedError(
                    f"Unsupported context type: {type(segment.context).__name__}"
                )

    @staticmethod
    def _format_timestamp(
        timestamp: datetime.datetime,
        *,
        date_style: DateTimeStyle | None,
        time_style: DateTimeStyle | None,
        locale: str,
        timezone: datetime.tzinfo | None,
    ) -> str:
        """Format a timestamp."""
        if date_style is None and time_style is None:
            return ""

        normalized_timestamp = (
            timestamp.astimezone(timezone) if timezone is not None else timestamp
        )

        date_string = ""
        time_string = ""

        if date_style is not None:
            date_string = format_date(
                normalized_timestamp, format=date_style, locale=locale
            )
        if time_style is not None:
            time_string = format_time(
                normalized_timestamp, format=time_style, locale=locale
            )

        if not time_string:
            return date_string
        if not date_string:
            return time_string

        connector_style = _DATETIME_STYLE_LEVELS[
            max(
                _DATETIME_STYLE_LEVELS.index(date_style),
                _DATETIME_STYLE_LEVELS.index(time_style),
            )
        ]

        template = str(get_datetime_format(connector_style, locale=locale))
        return template.replace("{1}", date_string).replace("{0}", time_string)

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

        Iterates contexts in score order, accumulating segments until the limit is reached.
        When a context would exceed the limit, segments nearest the seed are prioritized.
        Deduplicates across segment contexts in the query result.

        Args:
            query_result (QueryResult):
                The query result with scored anchored segment contexts.
            max_num_segments (int):
                The maximum number of segments to return.

        Returns:
            list[Segment]:
                Deduplicated segments ordered chronologically.
        """
        unified: set[Segment] = set()

        for scored_context in query_result.scored_segment_contexts:
            context = scored_context.segments

            if len(unified) >= max_num_segments:
                break
            if (len(unified) + len(context)) <= max_num_segments:
                unified.update(context)
            else:
                # Prioritize segments near the seed segment.
                seed_index = next(
                    index
                    for index, segment in enumerate(context)
                    if segment.uuid == scored_context.seed_segment_uuid
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
            list(scored_context.segments)
            for scored_context in query_result.scored_segment_contexts
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
