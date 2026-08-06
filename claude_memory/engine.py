"""The claude_memory engine: config, stores, embedder, and search/expand/ingest.

One cohesive module for the memory backend over EventMemory. Sections:

  1. config   — env-resolved paths, embedding choice, per-project partition
  2. sources  — event source taxonomy + the search-surface policy
  3. embedder — embeddinggemma-300m (the only model) + offline hash double (tests)
  4. deriver  — embed messages only (tool calls/files reached by expansion)
  5. stores   — SQLite vector + segment stores
  6. core     — MemoryCore (search/expand/ingest), stable ids, novelty, rendering

See DESIGN.md for the rationale behind every choice (stable ids, message-only
search surface, computed diminishing-returns, append-only/delta expansion).
"""

import datetime
import hashlib
import json
import math
import os
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast, override
from uuid import UUID
from zoneinfo import ZoneInfo

import numpy as np
from memmachine_server.common.data_types import PropertyValue, SimilarityMetric
from memmachine_server.common.embedder import Embedder
from memmachine_server.common.filter.filter_parser import (
    FilterExpr,
    FilterParseError,
    parse_filter,
)
from memmachine_server.common.vector_store import (
    VectorStore,
    VectorStoreCollection,
    VectorStoreCollectionConfig,
)
from memmachine_server.common.vector_store.data_types import Record
from memmachine_server.common.vector_store.sqlite_vec_vector_store import (
    SQLiteVecVectorStore,
    SQLiteVecVectorStoreParams,
)
from memmachine_server.common.vector_store.sqlite_vector_store import (
    SQLiteVectorStore,
    SQLiteVectorStoreParams,
)
from memmachine_server.common.vector_store.vector_search_engine.turbovec_engine import (
    TurboVecVectorSearchEngine,
)
from memmachine_server.episodic_memory.event_memory.data_types import (
    AnnotationContext,
    CompositeContext,
    Derivative,
    Event,
    FormatOptions,
    Segment,
    TextBlock,
)
from memmachine_server.episodic_memory.event_memory.deriver.deriver import Deriver
from memmachine_server.episodic_memory.event_memory.deriver.text_deriver import (
    WholeTextDeriver,
)
from memmachine_server.episodic_memory.event_memory.event_memory import (
    EventMemory,
    EventMemoryParams,
)
from memmachine_server.episodic_memory.event_memory.segment_store import (
    SegmentStorePartition,
    SegmentStorePartitionConfig,
)
from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import (
    SQLAlchemySegmentStore,
    SQLAlchemySegmentStoreParams,
)
from memmachine_server.episodic_memory.event_memory.segmenter.text_segmenter import (
    TextSegmenter,
)
from sqlalchemy import event
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import ConnectionPoolEntry

# Light client/wire surface lives in wire.py so thin clients can import it
# without pulling this module's heavy deps. engine uses these internally; other
# modules import them from wire directly.
from claude_memory.wire import (
    _SATURATED_MESSAGE,
    DemoteResult,
    ExpandResult,
    Hit,
    MemoryConfig,
    SearchResult,
    _demote_message,
    format_memory_line,
    is_searchable,
    kind_scope_filter,
    memory_id_for_segment_uuid,
    parse_memory_id,
    searchable_only,
    session_scope_filter,
)

# ===================================================================== embedders

# embeddinggemma-300m is the ONLY embedding model. No cloud / OpenAI model is
# imported or used anywhere (so no network and no API key). ``"hash"`` is a
# deterministic, offline double for tests only.
_EMBEDDINGGEMMA = "google/embeddinggemma-300m"


def build_embedder(model_name: str) -> Embedder:
    """Construct the embedder: only embeddinggemma-300m (or ``"hash"`` for tests).

    embeddinggemma loads weights per process (~5-6s), which is why the daemon
    loads it once. Asymmetric query/document prompts are handled below.
    """
    if model_name == "hash":
        return HashEmbedder()
    if model_name in ("embeddinggemma", _EMBEDDINGGEMMA):
        return _build_sentence_transformer(_EMBEDDINGGEMMA)
    raise ValueError(
        f"Unsupported embedding model {model_name!r}; this integration uses only "
        "embeddinggemma-300m (set CLAUDE_MEMORY_EMBEDDING=embeddinggemma, or 'hash' "
        "for offline tests)."
    )


def _build_sentence_transformer(hf_name: str) -> Embedder:
    """Build a local sentence-transformers embedder with correct asymmetric prompts."""
    from memmachine_server.common.embedder.sentence_transformer_embedder import (
        SentenceTransformerEmbedder,
        SentenceTransformerEmbedderParams,
    )
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(hf_name)
    # Asymmetric models (e.g. embeddinggemma) carry separate query/document
    # prompts. SentenceTransformerEmbedder hard-codes prompt_name="query" for
    # search; ingest passes no prompt, so the document prompt must be the default.
    if "document" in (model.prompts or {}):
        model.default_prompt_name = "document"
    return SentenceTransformerEmbedder(
        SentenceTransformerEmbedderParams(
            model_name=hf_name,
            sentence_transformer=model,
            batch_size=8,
        )
    )


class HashEmbedder(Embedder):
    """Deterministic, offline embedder for tests and smoke runs.

    Hashes whitespace tokens into a fixed-width bag-of-words vector and
    L2-normalizes. Enough to exercise wiring and overlap-based retrieval; not a
    substitute for a real semantic model.
    """

    def __init__(self, dimensions: int = 256) -> None:
        """Initialize a hash embedder of the given dimensionality."""
        super().__init__(batch_size=None)
        self._dimensions = dimensions

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        for token in text.lower().split():
            bucket = int(hashlib.sha1(token.encode()).hexdigest(), 16)
            vector[bucket % self._dimensions] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    async def _ingest_embed(
        self, inputs: list[Any], max_attempts: int = 1
    ) -> list[list[float]]:
        del max_attempts  # deterministic; retries are irrelevant
        return [self._embed_one(str(item)) for item in inputs]

    async def _search_embed(
        self, queries: list[Any], max_attempts: int = 1
    ) -> list[list[float]]:
        del max_attempts  # deterministic; retries are irrelevant
        return [self._embed_one(str(query)) for query in queries]

    @property
    def model_id(self) -> str:
        return f"hash-{self._dimensions}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def similarity_metric(self) -> SimilarityMetric:
        return SimilarityMetric.COSINE


# ======================================================================= deriver


class MessageOnlyDeriver(Deriver):
    """Emit derivatives (embeddings) only for searchable (message) segments.

    A suppressed segment is still stored on the timeline (reachable by expansion);
    it just never becomes a direct search hit and never bloats the vector index.
    """

    def __init__(self, inner: Deriver) -> None:
        """Wrap an inner deriver, suppressing it for non-message segments."""
        self._inner = inner

    @override
    async def derive(
        self,
        segment: Segment,
        *,
        format_options: FormatOptions | None = None,
    ) -> list[Derivative]:
        source = segment.properties.get("source")
        if isinstance(source, str) and not is_searchable(source):
            return []
        return await self._inner.derive(segment, format_options=format_options)


# ======================================================================== stores

# Indexed beyond EventMemory's reserved fields so the model can scope searches by
# speaker / source / session via metadata filters (e.g. `producer = "Caroline"`).
_EXTRA_INDEXED_PROPERTIES: dict[str, type[PropertyValue]] = {
    "source": cast(type[PropertyValue], str),
    "producer": cast(type[PropertyValue], str),
    "session_id": cast(type[PropertyValue], str),
    "project": cast(type[PropertyValue], str),
}


def _configure_sqlite(engine: AsyncEngine) -> None:
    @event.listens_for(engine.sync_engine, "connect")
    def _set_pragmas(
        dbapi_connection: DBAPIConnection,
        _record: ConnectionPoolEntry,
    ) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()


def _build_vector_store(
    config: MemoryConfig, vector_engine: AsyncEngine
) -> VectorStore:
    """Build the vector store backend (default: turbovec + SQLiteVectorStore).

    ``config.vector_backend`` (env ``CLAUDE_MEMORY_VECTOR_BACKEND`` >
    ``<home>/config.json`` ``vector_backend`` > default) selects ``turbovec``
    (TurboQuant-compressed in-RAM index: fast approximate search, ~8x smaller
    vectors at 4-bit),
    ``turbovecdisk`` (same quantized search, but the index is a memory-mapped
    file searched in place — near-zero charged RAM, identical warm latency), or
    ``sqlitevec`` (exact, float32 vectors in sqlite-vec). Both turbovec variants
    keep only compressed codes, so vector read-back is unavailable — fine here,
    since EventMemory only ever queries with return_vector=False. The compressed
    index lives in a separate ``vector_index/`` directory and is persisted on
    shutdown / every save batch. The two turbovec index file formats differ
    (.tvim in RAM vs .tvdm mmap); switching between them requires converting
    the files in vector_index/ via ``turbovec.DiskIndex.convert_id_map_file``.
    """
    backend = config.vector_backend
    if backend == "sqlitevec":
        return SQLiteVecVectorStore(SQLiteVecVectorStoreParams(engine=vector_engine))
    if backend in ("turbovec", "turbovecdisk"):
        bit_width = int(os.environ.get("CLAUDE_MEMORY_TURBOVEC_BITS", "4"))
        if backend == "turbovecdisk":
            # Lazy: DiskIndex needs the locally-built turbovec wheel; the
            # PyPI release (what `uv sync` installs) only covers the in-RAM
            # engine, so importing here keeps the default backend working.
            from memmachine_server.common.vector_store.vector_search_engine.turbovec_disk_engine import (
                TurboVecDiskVectorSearchEngine,
            )

            engine_class: type[TurboVecVectorSearchEngine] = (
                TurboVecDiskVectorSearchEngine
            )
        else:
            engine_class = TurboVecVectorSearchEngine
        return SQLiteVectorStore(
            SQLiteVectorStoreParams(
                sqlalchemy_engine=vector_engine,
                vector_search_engine_factory=(
                    lambda ndim, metric: engine_class(
                        num_dimensions=ndim,
                        similarity_metric=metric,
                        bit_width=bit_width,
                    )
                ),
                index_directory=str(config.home / "vector_index"),
            )
        )
    raise ValueError(
        f"unknown CLAUDE_MEMORY_VECTOR_BACKEND={backend!r} "
        f"(turbovec|turbovecdisk|sqlitevec)"
    )


@dataclass
class OpenedMemory:
    """An EventMemory and the per-partition handles that must be closed."""

    memory: EventMemory
    partition: SegmentStorePartition
    collection: VectorStoreCollection


@dataclass
class MemoryStores:
    """Shared SQLite stores and embedder for one process."""

    config: MemoryConfig
    embedder: Embedder
    vector_store: VectorStore
    segment_store: SQLAlchemySegmentStore
    vector_engine: AsyncEngine
    segment_engine: AsyncEngine

    @classmethod
    async def open(cls, config: MemoryConfig, embedder: Embedder) -> "MemoryStores":
        """Open the SQLite engines and start the stores."""
        config.ensure_dirs()

        segment_engine = create_async_engine(
            f"sqlite+aiosqlite:///{config.segment_db}",
            connect_args={"timeout": 30},
        )
        _configure_sqlite(segment_engine)
        segment_store = SQLAlchemySegmentStore(
            SQLAlchemySegmentStoreParams(engine=segment_engine)
        )
        await segment_store.startup()

        vector_engine = create_async_engine(
            f"sqlite+aiosqlite:///{config.vector_db}",
            connect_args={"timeout": 30},
        )
        _configure_sqlite(vector_engine)
        vector_store = _build_vector_store(config, vector_engine)
        await vector_store.startup()

        return cls(
            config=config,
            embedder=embedder,
            vector_store=vector_store,
            segment_store=segment_store,
            vector_engine=vector_engine,
            segment_engine=segment_engine,
        )

    def _collection_config(self) -> VectorStoreCollectionConfig:
        schema: dict[str, type[PropertyValue]] = dict(
            EventMemory.expected_vector_store_collection_schema()
        )
        schema.update(_EXTRA_INDEXED_PROPERTIES)
        return VectorStoreCollectionConfig(
            vector_dimensions=self.embedder.dimensions,
            similarity_metric=self.embedder.similarity_metric,
            indexed_properties_schema=schema,
        )

    async def open_memory(self, partition: str | None = None) -> OpenedMemory:
        """Open (or create) a collection/partition and build EventMemory.

        Defaults to ``config.partition``; the daemon passes an explicit partition
        so one set of stores (and one loaded embedder) can serve many projects.
        """
        partition_name = partition or self.config.partition
        collection = await self.vector_store.open_or_create_collection(
            namespace=self.config.namespace,
            name=partition_name,
            config=self._collection_config(),
        )
        segment_partition = await self.segment_store.open_or_create_partition(
            partition_name,
            SegmentStorePartitionConfig(),
        )
        memory = EventMemory(
            EventMemoryParams(
                segment_store_partition=segment_partition,
                vector_store_collection=collection,
                segmenter=TextSegmenter(),
                deriver=MessageOnlyDeriver(WholeTextDeriver()),
                embedder=self.embedder,
                reranker=None,
                eviction_similarity_threshold=self.config.eviction_threshold,
                eviction_target_size=self.config.eviction_target_size,
                eviction_search_limit=self.config.eviction_search_limit,
                # When eviction is on, serialize encodes so the per-batch
                # query/select/delete on the vector store can't interleave.
                serialize_encode=self.config.eviction_threshold is not None,
            )
        )
        return OpenedMemory(
            memory=memory, partition=segment_partition, collection=collection
        )

    async def close_memory(self, opened: OpenedMemory) -> None:
        """Release the per-partition handles."""
        await self.segment_store.close_partition(opened.partition)
        await self.vector_store.close_collection(collection=opened.collection)

    async def aclose(self) -> None:
        """Shut down the shared stores and dispose engines."""
        await self.segment_store.shutdown()
        await self.vector_store.shutdown()
        await self.segment_engine.dispose()
        await self.vector_engine.dispose()


def segment_sort_key(segment: Segment) -> tuple[datetime.datetime, UUID, int, int]:
    """Chronological ordering key for a segment within / among contexts."""
    return (segment.timestamp, segment.event_uuid, segment.index, segment.offset)


# ========================================================================== core


def _display_timezone() -> datetime.tzinfo:
    """The machine's local timezone for rendering stored UTC timestamps.

    Claude Code writes transcript timestamps in UTC and the segment store keeps
    them UTC; without converting at display time everything reads in UTC (e.g. a
    PDT user sees times 7h late). Prefer the OS IANA zone (DST-correct for each
    timestamp's own date); ``CLAUDE_MEMORY_TIMEZONE`` overrides; fall back to the
    current fixed local offset.
    """
    name = os.environ.get("CLAUDE_MEMORY_TIMEZONE")
    if name:
        with suppress(KeyError, ValueError):
            return ZoneInfo(name)
    with suppress(OSError, ValueError, KeyError):
        link = str(Path("/etc/localtime").readlink())
        if "zoneinfo/" in link:
            return ZoneInfo(link.split("zoneinfo/", 1)[1])
    return datetime.datetime.now().astimezone().tzinfo or datetime.UTC


# Dates carry signal and are cheap; wall-clock time of day rarely helps recall.
# time_style="medium" includes seconds so within-session order is visible in a
# tight expand window (short=minute collapsed consecutive turns to one instant).
# Stored timestamps are UTC; render them in the machine's local zone (no label).
DISPLAY_FORMAT = FormatOptions(
    date_style="medium", time_style="medium", timezone=_display_timezone()
)


# --- manual demotion: deprioritize a doc vector for a cue (Rocchio rotation) ---

_DEMOTIONS_FILE = "demotions.json"
_DEMOTE_MARGIN = 0.02
# Cap accumulated displacement so repeated demotes can't drift a vector into noise.
_DEMOTE_MAX_NORM = 1.5
# How many post-demote matches to echo back so the model can judge whether the
# cue is exhausted (a mostly-irrelevant tail => no better memory exists).
_DEMOTE_POOL_PREVIEW = 6
# Mirror EventMemory's reserved vector-record field names (a storage contract).
_SEGMENT_UUID_FIELD = "_segment_uuid"
_TIMESTAMP_FIELD = "_timestamp"


def _block_text(block: object) -> str | None:
    """Text of a derivative/segment block, or None for non-text blocks."""
    return block.text if isinstance(block, TextBlock) else None


def _l2norm(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


# --- per-session running context vector (TCM drift) for ambient recall ---
#
# The ambient cue is a hybrid: a decayed DOCUMENT-side EMA of the session's
# messages (the running context) blended with the current prompt's QUERY
# embedding. Validated in evaluation/event_memory/autosurface_cue_probe.py:
# rescues terse/anaphoric prompts, robust for prior weights 0.5-0.9.

CONTEXT_PRIOR_WEIGHT = 0.8


def fold_running_context(
    previous: np.ndarray | None, doc_vector: np.ndarray
) -> np.ndarray:
    """EMA-fold a message's document embedding into the running context."""
    folded = _l2norm(np.asarray(doc_vector, dtype=float))
    if previous is None:
        return folded
    return _l2norm(
        CONTEXT_PRIOR_WEIGHT * previous + (1 - CONTEXT_PRIOR_WEIGHT) * folded
    )


def blend_context_cue(
    previous: np.ndarray | None, query_vector: np.ndarray
) -> np.ndarray:
    """Hybrid ambient cue: decayed document-side prior + current query embedding."""
    current = _l2norm(np.asarray(query_vector, dtype=float))
    if previous is None:
        return current
    return _l2norm(
        CONTEXT_PRIOR_WEIGHT * previous + (1 - CONTEXT_PRIOR_WEIGHT) * current
    )


def _load_demotions(home: Path) -> dict[str, list[float]]:
    """Per-derivative accumulated demotion vectors (delta = sum of alpha*q), by uuid hex.

    Kept in a sidecar (the turbovec index can't be read back), so the demotion is
    durable, inspectable, and composable across cues; the live vector is always
    normalize(d_orig - delta) with d_orig recovered by re-embedding.
    """
    try:
        data = json.loads((home / _DEMOTIONS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_demotions(home: Path, data: dict[str, list[float]]) -> None:
    (home / _DEMOTIONS_FILE).write_text(json.dumps(data), encoding="utf-8")


def _solve_demotion_alpha(base: np.ndarray, query: np.ndarray, target: float) -> float:
    """Smallest alpha >= 0 with cos(query, normalize(base - alpha*query)) ~= target."""

    def cosine(alpha: float) -> float:
        shifted = base - alpha * query
        norm = float(np.linalg.norm(shifted))
        return float(query @ shifted / norm) if norm else -1.0

    high = 1.0
    while cosine(high) > target and high < 64.0:
        high *= 2.0
    low = 0.0
    for _ in range(50):
        mid = (low + high) / 2.0
        if cosine(mid) > target:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def _demote_target(c0: float, factor: float) -> float:
    """Target cosine for one demote step: geometric decay of the current cosine.

    Multiply by ``factor`` (e.g. 0.9), but always move at least the tie-break
    margin so a step is never a no-op for a meaningfully-matching doc.
    """
    return min(c0 * factor, c0 - _DEMOTE_MARGIN)


@dataclass
class MemoryCore:
    """Owns an open EventMemory and the per-session novelty set."""

    stores: MemoryStores
    opened: OpenedMemory
    seen_segment_uuids: set[str] = field(default_factory=set)

    @classmethod
    async def open(cls, config: MemoryConfig) -> "MemoryCore":
        """Build a standalone core (own stores + embedder) for one partition."""
        embedder = build_embedder(config.embedding_model)
        stores = await MemoryStores.open(config, embedder)
        opened = await stores.open_memory()
        return cls(stores=stores, opened=opened)

    @classmethod
    async def open_on(cls, stores: MemoryStores, partition: str) -> "MemoryCore":
        """Open one partition on already-built, shared stores (used by the daemon)."""
        opened = await stores.open_memory(partition)
        return cls(stores=stores, opened=opened)

    @property
    def memory(self) -> EventMemory:
        return self.opened.memory

    async def close_partition(self) -> None:
        """Release just this partition's handles, leaving shared stores open."""
        await self.stores.close_memory(self.opened)

    async def aclose(self) -> None:
        await self.close_partition()
        await self.stores.aclose()

    async def ingest(self, events: Iterable[Event]) -> int:
        """Encode events into memory, skipping any already ingested.

        Event uuids are derived from the source transcript record (see
        ``transcript``), so re-ingesting the same record — backfill/live overlap,
        or a forked/resumed session that copies earlier records — yields the same
        uuid, is recognized here, and is skipped. Ingest is therefore idempotent.
        Returns the number of newly-ingested events.
        """
        events = list(events)
        if not events:
            return 0
        already = await self.opened.partition.get_segment_uuids_by_event_uuids(
            [event.uuid for event in events]
        )
        new_events = [event for event in events if event.uuid not in already]
        if new_events:
            await self.memory.encode_events(new_events)
        return len(new_events)

    async def search(
        self,
        cue: str,
        *,
        limit: int = 8,
        filter_spec: str | None = None,
        seen: set[str] | None = None,
        commit_seen: bool = True,
        query_vector: list[float] | None = None,
    ) -> SearchResult:
        """Associative recall over the message search surface.

        ``seen`` is the novelty set to read/update (defaults to this core's own).
        The daemon passes a per-(partition, session) set so novelty is shared
        across the ambient hook and the MCP tools within a session.

        ``commit_seen`` controls whether new hits are added to ``seen``. The
        reflective Stop hook sets it False so it can first gate hits by relevance
        and only mark the ones it actually surfaces as seen.

        ``query_vector`` optionally replaces the cue's own embedding for the
        vector search (the ambient hook passes the running-context blend).
        """
        seen_set = seen if seen is not None else self.seen_segment_uuids
        try:
            property_filter = parse_filter(searchable_only(filter_spec))
        except FilterParseError as error:
            return SearchResult(
                hits=[], new_count=0, saturated=False, note=f"Invalid filter: {error}"
            )

        result = await self.memory.query(
            cue,
            vector_search_limit=limit,
            expand_context=0,
            property_filter=property_filter,
            format_options=DISPLAY_FORMAT,
            query_vector=query_vector,
        )

        hits: list[Hit] = []
        new_count = 0
        for scored_context in result.scored_segment_contexts:
            segment_uuid = scored_context.seed_segment_uuid
            text = EventMemory.string_from_segment_context(
                scored_context.segments, format_options=DISPLAY_FORMAT
            )
            is_new = segment_uuid.hex not in seen_set
            if is_new:
                new_count += 1
                if commit_seen:
                    seen_set.add(segment_uuid.hex)
            hits.append(
                Hit(
                    memory_id=memory_id_for_segment_uuid(segment_uuid),
                    score=scored_context.score,
                    text=text,
                    is_new=is_new,
                )
            )

        saturated = bool(hits) and new_count == 0
        return SearchResult(hits=hits, new_count=new_count, saturated=saturated)

    async def expand(
        self,
        seed_id: str,
        *,
        before: int = 5,
        after: int = 5,
        seen: set[str] | None = None,
        kinds: list[str] | None = None,
        blocklist: bool = False,
    ) -> ExpandResult:
        """Return the seed's session timeline window, same-event segments merged.

        ``kinds`` names sources (user_message, assistant_message, reasoning,
        tool_call, tool_result, injected), read as an allowlist or — with
        ``blocklist`` — as a blocklist. It is pushed into the store's window walk
        rather than applied to its result, so the budget is spent only on segments
        the caller asked for.
        """
        seen_set = seen if seen is not None else self.seen_segment_uuids
        seed_uuid = parse_memory_id(seed_id)
        if seed_uuid is None:
            return ExpandResult(
                seed_id=seed_id,
                found=False,
                note=f"'{seed_id}' is not a valid memory id (expected mem:<hex>).",
            )

        # Scope expansion to the seed's own session. In the shared search space
        # the timeline interleaves every session, so neighbours must be filtered
        # to the same session_id — otherwise expansion crosses conversations.
        seed_only = await self.opened.partition.get_segment_contexts(
            seed_segment_uuids=[seed_uuid],
            max_backward_segments=0,
            max_forward_segments=0,
            property_filter=None,
        )
        seed_segments = seed_only.get(seed_uuid)
        scope: list[str] = []
        if seed_segments:
            session = seed_segments[0].properties.get("session_id")
            if isinstance(session, str) and session:
                scope.append(session_scope_filter(session))
        kind_filter = kind_scope_filter(kinds, blocklist=blocklist)
        if kind_filter:
            scope.append(kind_filter)
        session_filter: FilterExpr | None = None
        if scope:
            with suppress(FilterParseError):
                session_filter = parse_filter(" AND ".join(scope))

        contexts = await self.opened.partition.get_segment_contexts(
            seed_segment_uuids=[seed_uuid],
            max_backward_segments=max(before, 0),
            max_forward_segments=max(after, 0),
            property_filter=session_filter,
        )
        window = list(contexts.get(seed_uuid) or [])
        # A kind filter can exclude the seed's own kind, and then the walk returns a
        # window that does not contain what the caller named. Re-attach the seed
        # segment itself — not its kind, which would widen the filter for every
        # other segment as well.
        if seed_segments and all(segment.uuid != seed_uuid for segment in window):
            window += [
                segment for segment in seed_segments if segment.uuid == seed_uuid
            ]
        if not window:
            return ExpandResult(
                seed_id=seed_id,
                found=False,
                note=f"No memory found for {seed_id} (it may not exist in memory).",
            )

        window = sorted(window, key=segment_sort_key)
        # Everything but the seed itself is newly surfaced.
        for segment in window:
            if segment.uuid != seed_uuid:
                seen_set.add(segment.uuid.hex)

        # Render the whole window in ONE pass: string_from_segment_context merges
        # consecutive same-event segment chunks back into whole events (one header
        # / timestamp per event), spanning the seed. Navigation ids are just the
        # window's earliest and latest segments.
        window_text = EventMemory.string_from_segment_context(
            window, format_options=DISPLAY_FORMAT
        )
        return ExpandResult(
            seed_id=seed_id,
            window_text=window_text,
            earliest_id=memory_id_for_segment_uuid(window[0].uuid),
            latest_id=memory_id_for_segment_uuid(window[-1].uuid),
        )

    async def _get_segment(self, seg_uuid: UUID) -> Segment | None:
        """Fetch one segment by uuid, or None if it does not exist."""
        contexts = await self.opened.partition.get_segment_contexts(
            seed_segment_uuids=[seg_uuid],
            max_backward_segments=0,
            max_forward_segments=0,
            property_filter=None,
        )
        return next(
            (s for s in (contexts.get(seg_uuid) or []) if s.uuid == seg_uuid), None
        )

    async def annotate(self, memory_id: str, note: str) -> str:
        """Append a one-line note to a memory's stored context (append-only).

        The note becomes part of how the segment renders wherever it surfaces
        (search, expand, ambient recall), labeled as a note; the embedding
        anchor is derived without annotations, so vectors and ranking are
        untouched.
        """
        note = " ".join(note.split())
        if not note:
            return "Cannot add an empty note."
        seg_uuid = parse_memory_id(memory_id)
        if seg_uuid is None:
            return f"'{memory_id}' is not a valid memory id (expected mem:<hex>)."
        segment = await self._get_segment(seg_uuid)
        if segment is None:
            return f"No memory found for {memory_id}."

        annotation = AnnotationContext(note=note)
        base = segment.context
        if isinstance(base, CompositeContext):
            new_context = CompositeContext(contexts=[*base.contexts, annotation])
        else:
            new_context = CompositeContext(contexts=[base, annotation])
        await self.opened.partition.update_segment_contexts({seg_uuid: new_context})

        segment.context = new_context
        rendered = EventMemory.string_from_segment_context(
            [segment], format_options=DISPLAY_FORMAT
        )
        return (
            f"Noted. This memory now reads:\n{format_memory_line(memory_id, rendered)}"
        )

    async def _demote_resolve(
        self, memory_id: str
    ) -> "tuple[UUID, Segment, list[UUID]] | DemoteResult":
        """Resolve a handle to (segment uuid, segment, derivatives) or a DemoteResult."""
        seg_uuid = parse_memory_id(memory_id)
        if seg_uuid is None:
            return DemoteResult(
                False,
                "invalid",
                f"'{memory_id}' is not a valid memory id (expected mem:<hex>).",
                memory_id,
            )
        segment = await self._get_segment(seg_uuid)
        if segment is None:
            return DemoteResult(
                False, "not_found", f"No memory found for {memory_id}.", memory_id
            )
        derivative_uuids = list(
            (
                await self.opened.partition.get_derivative_uuids_by_segment_uuids(
                    [seg_uuid]
                )
            ).get(seg_uuid, [])
        )
        if not derivative_uuids:
            return DemoteResult(
                False,
                "not_searchable",
                "That memory is on the timeline but not embedded (e.g. a tool "
                "call/result) - only message memories can be demoted.",
                memory_id,
            )
        return (seg_uuid, segment, derivative_uuids)

    async def demote(self, memory_id: str, cue: str) -> DemoteResult:
        """Deprioritize a memory for a cue (and similar future cues), score-free.

        Geometric decay: each call multiplies the memory's cosine to the cue by
        ``config.demote_decay`` (~0.9), via a directional Rocchio rotation away from
        the cue. Repeated demotes decay it further; there is no relevance floor,
        pool target, or per-call strength. The change accumulates as a bounded
        per-derivative delta and is applied by re-embedding the doc and upserting
        normalize(d_orig - delta) (turbovec can't read vectors back). The result
        echoes the cue's current top matches so the model can judge whether the cue
        is exhausted (a mostly-irrelevant tail => no better memory; stop demoting).
        """
        config = self.stores.config
        resolved = await self._demote_resolve(memory_id)
        if isinstance(resolved, DemoteResult):
            return resolved
        seg_uuid, segment, derivative_uuids = resolved

        derivatives = await MessageOnlyDeriver(WholeTextDeriver()).derive(
            segment, format_options=None
        )
        texts = [_block_text(d.block) for d in derivatives]
        if not texts or any(text is None for text in texts):
            return DemoteResult(
                False,
                "not_searchable",
                "Could not reconstruct the memory's embedding text.",
                memory_id,
                cue,
            )
        d_origs = [
            _l2norm(np.asarray(vec, dtype=float))
            for vec in await self.stores.embedder.ingest_embed(
                [text for text in texts if text is not None]
            )
        ]
        query = _l2norm(
            np.asarray((await self.stores.embedder.search_embed([cue]))[0], dtype=float)
        )
        # One geometric step (~0.9x); the model calls again to demote further.
        factor = config.demote_decay

        demotions = _load_demotions(config.home)
        dim = self.stores.embedder.dimensions
        records: list[Record] = []
        before_values: list[float] = []
        after_values: list[float] = []
        for derivative_uuid, d_orig in zip(derivative_uuids, d_origs, strict=False):
            delta = np.asarray(
                demotions.get(derivative_uuid.hex, [0.0] * dim), dtype=float
            )
            c0 = float(query @ _l2norm(d_orig - delta))
            before_values.append(c0)
            if c0 <= _DEMOTE_MARGIN:  # already negligible for this cue
                after_values.append(c0)
                continue
            target = _demote_target(c0, factor)
            new_delta = (
                delta + _solve_demotion_alpha(d_orig - delta, query, target) * query
            )
            if float(np.linalg.norm(new_delta)) > _DEMOTE_MAX_NORM:
                before = max(before_values, default=0.0)
                return DemoteResult(
                    False,
                    "saturated",
                    _SATURATED_MESSAGE,
                    memory_id,
                    cue,
                    before,
                    before,
                )
            d_prime = _l2norm(d_orig - new_delta)
            demotions[derivative_uuid.hex] = new_delta.tolist()
            records.append(
                Record(
                    uuid=derivative_uuid,
                    vector=d_prime.tolist(),
                    properties={
                        _SEGMENT_UUID_FIELD: str(seg_uuid),
                        _TIMESTAMP_FIELD: segment.timestamp,
                        **dict(segment.properties),
                    },
                )
            )
            after_values.append(float(query @ d_prime))

        if records:
            await self.opened.collection.upsert(records=records)
            _save_demotions(config.home, demotions)
        before = max(before_values, default=0.0)
        after = max(after_values, default=before)
        pool = await self._demote_pool_preview(cue)
        return DemoteResult(
            True,
            "demoted",
            _demote_message(pool),
            memory_id,
            cue,
            before,
            after,
        )

    async def _demote_pool_preview(self, cue: str) -> "list[Hit]":
        """The cue's current (post-demote) top matches, for the model to judge.

        Shows the real ranking — including the just-demoted memory's new position —
        so the model can see whether a better answer surfaced or the cue is dry.
        """
        result = await self.search(
            cue, limit=_DEMOTE_POOL_PREVIEW, seen=set(), commit_seen=False
        )
        return result.hits


# --- rendering for tool output / hook injection (shared by all entry points) ---
