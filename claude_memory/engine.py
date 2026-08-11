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
import logging
import math
import os
import time
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast, override
from uuid import UUID

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
from sqlalchemy import bindparam, event, text
from sqlalchemy.engine.interfaces import DBAPIConnection
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import ConnectionPoolEntry

# Light client/wire surface lives in wire.py so thin clients can import it
# without pulling this module's heavy deps. engine uses these internally; other
# modules import them from wire directly.
from claude_memory.wire import (
    _SATURATED_MESSAGE,
    ANCHOR_MARKER,
    ID_CANDIDATE_LIMIT,
    Beat,
    DemoteResult,
    ExpandResult,
    Hit,
    MemoryConfig,
    OutlineResult,
    SearchResult,
    Source,
    _demote_message,
    _display_timezone,
    abbreviation_length,
    ambiguous_id_note,
    format_memory_line,
    is_searchable,
    kind_scope_filter,
    memory_id_for_segment_uuid,
    observe,
    parse_memory_ref,
    scope_filter,
    score_shape,
    searchable_only,
    session_scope_filter,
)

logger = logging.getLogger("claude_memory.engine")

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


# --- claude_memory's own index on the segment table ---------------------------
#
# ``session_id`` is a USER property, so it lives inside the ``properties`` JSON
# rather than in a column, and memmachine's two indexes lead with
# ``partition_key`` then ``timestamp`` or ``event_uuid``. Nothing can seek by
# session: every session-scoped query walks the whole partition and evaluates the
# JSON per row. Measured on the deployed store (594,773 segments):
#
#     the outline's per-session timeline    1.23s -> 0.01s
#     the DISTINCT scan behind resolution   1.87s -> 0.04s   (covering)
#
# for 48MB, about 7% of the database, and ~0.9s to build once.
#
# This is claude_memory's index on a table whose schema memmachine owns, which is
# a layering choice worth naming rather than hiding: it exists because THIS
# application knows something the generic store does not — that of all the user
# properties, session_id is the one everything filters by. The name says who owns
# it. It is additive, survives independently of the ORM's create_all, and dropping
# it costs only the speed.
_SESSION_INDEX = "claude_memory_sg_session"
_SESSION_INDEX_SQL = (
    f"CREATE INDEX IF NOT EXISTS {_SESSION_INDEX} ON segment_store_sg "
    "(partition_key, json_extract(properties, '$.session_id.v'), timestamp)"
)


async def ensure_session_index(engine: AsyncEngine) -> None:
    """Create the session index if it is missing. Never fails startup.

    SQLite only: the expression indexes an ``json_extract`` call, which Postgres
    spells differently. claude_memory is SQLite-only, and a missing index costs
    speed rather than correctness, so anything unexpected here is logged past
    rather than raised.
    """
    if engine.dialect.name != "sqlite":
        return
    with suppress(Exception):
        async with engine.begin() as connection:
            existing = (
                await connection.execute(
                    text("SELECT 1 FROM sqlite_master WHERE type='index' AND name=:n"),
                    {"n": _SESSION_INDEX},
                )
            ).first()
            if existing is not None:
                return
            started = time.monotonic()
            await connection.execute(text(_SESSION_INDEX_SQL))
            logger.info("built %s in %.1fs", _SESSION_INDEX, time.monotonic() - started)


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
    # The partition's own name. The store keeps it privately, and the id
    # abbreviation queries below address ``segment_store_sg`` directly — they need
    # the leading key column to seek the primary-key index rather than scan.
    partition_name: str


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
        await ensure_session_index(segment_engine)

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
            memory=memory,
            partition=segment_partition,
            collection=collection,
            partition_name=partition_name,
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

# --- short handles: the queries behind abbreviation and prefix resolution ------
#
# These go straight at ``segment_store_sg`` rather than through the store's API,
# which has no prefix or neighbour operation. Each one leads with ``partition_key``
# and then ranges over ``uuid``, which is exactly the shape of the table's primary
# key, so every one of them is an index seek over a handful of rows — not a scan of
# the ~600k segments behind it.

_NEIGHBOUR_SQL = (
    "SELECT uuid FROM segment_store_sg "
    "WHERE partition_key = :partition AND uuid < :uuid "
    "ORDER BY uuid DESC LIMIT 1",
    "SELECT uuid FROM segment_store_sg "
    "WHERE partition_key = :partition AND uuid > :uuid "
    "ORDER BY uuid ASC LIMIT 1",
)
# Hex digits run 0-9a-f and 'g' sorts above all of them, so every uuid beginning
# with the prefix lies in [prefix, prefix + 'g') and nothing else does. A LIKE or
# GLOB would express the same set without being a range the index can seek.
_PREFIX_SQL = (
    "SELECT uuid FROM segment_store_sg "
    "WHERE partition_key = :partition AND uuid >= :low AND uuid < :high "
    "ORDER BY uuid LIMIT :limit"
)
# Exact size of an event without materializing it. LENGTH() over the extracted
# text counts code points — the same unit Python's len() reports — and SQLite does
# the aggregate in C. json_extract assumes the plaintext payload codec; under any
# other codec the sum is NULL and the caller reports segments instead.
_EVENT_EXTENT_SQL = (
    "SELECT event_uuid, COUNT(*), SUM(LENGTH(json_extract(block, '$.text'))) "
    "FROM segment_store_sg "
    "WHERE partition_key = :partition AND event_uuid IN :events "
    "GROUP BY event_uuid"
)
# The earliest segment of each named conversation, one row per session. Ordering
# inside the group is by the same key the timeline uses, so "first" here means the
# same thing it means everywhere else.
_FIRST_SEGMENT_SQL = (
    "SELECT session, uuid FROM ("
    "  SELECT json_extract(properties, '$.session_id.v') AS session, uuid,"
    "         ROW_NUMBER() OVER ("
    "           PARTITION BY json_extract(properties, '$.session_id.v')"
    '           ORDER BY timestamp, event_uuid, "index", "offset"'
    "         ) AS rank"
    "  FROM segment_store_sg WHERE partition_key = :partition"
    "  AND json_extract(properties, '$.session_id.v') IN :sessions"
    ") WHERE rank = 1"
)
_SESSION_PREFIX_SQL = (
    "SELECT DISTINCT json_extract(properties, '$.session_id.v') AS session "
    "FROM segment_store_sg WHERE partition_key = :partition "
    # The expression is repeated rather than referenced by its alias: a SELECT
    # alias is not in scope in WHERE, and it is the expression the index is on.
    "AND json_extract(properties, '$.session_id.v') >= :low "
    "AND json_extract(properties, '$.session_id.v') < :high "
    "ORDER BY session LIMIT :limit"
)
# One row per EVENT: block 0, chunk 0 is an event's opening and exists for every
# event (verified — no stored event has a hole in its block-index run), so this
# reads the segment table as a timeline without grouping over the text.
# The session's shape, WITHOUT its content: one row per event carrying only what
# the outline needs to count and place things. Deliberately no ``block``, because
# an outline renders twenty turns out of a session that may hold six thousand
# events, and fetching every event's text to decode twenty of them was most of the
# call's cost.
# ORDER BY is timestamp ALONE, and that is load-bearing. Adding event_uuid as a
# tiebreaker made the planner prefer the (partition_key, timestamp, event_uuid, …)
# index — which serves that order but cannot filter by session, so it scanned the
# whole partition: 179.9ms against 10.2ms. The session index already yields a
# session's rows in timestamp order, so ordering by timestamp alone lets one index
# do both, and the tiebreak happens in Python over a few thousand rows.
_SESSION_TIMELINE_SQL = (
    "SELECT uuid, event_uuid, timestamp, json_extract(properties, '$.source.v'), "
    "json_extract(properties, '$.project.v') "
    "FROM segment_store_sg "
    "WHERE partition_key = :partition "
    "AND json_extract(properties, '$.session_id.v') = :session "
    'AND "index" = 0 AND "offset" = 0 '
    "ORDER BY timestamp"
)
# ...and the opening words of just the turns that will actually be rendered.
_OPENING_TEXT_SQL = (
    "SELECT uuid, substr(json_extract(block, '$.text'), 1, :chars) "
    "FROM segment_store_sg "
    "WHERE partition_key = :partition AND uuid IN :uuids"
)

#: How much of a user turn an outline line shows. Long enough to recognise the
#: request, short enough that twenty of them stay a navigation aid.
_BEAT_CHARS = 90


def _parse_stamp(value: object) -> datetime.datetime:
    """A stored timestamp as an aware datetime, however the driver handed it back."""
    if isinstance(value, datetime.datetime):
        parsed = value
    else:
        try:
            parsed = datetime.datetime.fromisoformat(str(value))
        except ValueError:
            return datetime.datetime.now(datetime.UTC)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=datetime.UTC)


def _prefix_upper_bound(prefix: str) -> str:
    """The smallest string greater than every string starting with ``prefix``."""
    return prefix[:-1] + chr(ord(prefix[-1]) + 1)


def seat_time(when: datetime.datetime) -> datetime.datetime:
    """The stored timestamp as an aware instant, for comparing positions."""
    return _parse_stamp(when)


def _short_time(when: datetime.datetime) -> str:
    """A timestamp for an outline line: local, to the minute, with the year.

    The year is not optional here. An outline reaches back as far as the
    conversation does, sessions in this store span months, and a bare "Aug 10"
    silently invites the reader to assume the current one. ISO order, so the
    column lines up and sorts.
    """
    return (
        _parse_stamp(when)
        .astimezone(DISPLAY_FORMAT.timezone)
        .strftime("%Y-%m-%d %H:%M")
    )


def _clip(text_value: str, limit: int) -> str:
    """One line of at most ``limit`` characters, broken at a word where possible."""
    flat = " ".join(text_value.split())
    if len(flat) <= limit:
        return flat
    head = flat[:limit]
    space = head.rfind(" ")
    return (head[:space] if space > limit * 0.6 else head).rstrip() + "…"


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


# --- events too large to show whole ------------------------------------------
#
# The budget is what a person would write or read directly in one message, not a
# document they wrote elsewhere and pasted. Measured over the store, with the
# misfiled injected classes excluded, 4,000 characters shows ~95% of every kind of
# event whole: user messages median 108 / p99 5,340, assistant median 184 / p99
# 5,363.
#
# Over that, expand does NOT try to spend the budget. It shows the event's FIRST
# and LAST segment — the opening and the outcome, roughly 500 characters each —
# and the seed's own segment if it sits between them. Enough that the event is not
# a black box, little enough that it cannot crowd out the conversation, and the
# handles make it navigable at segment level rather than summarised.
#
# There is no exemption for the seed's own event. The rule is the same for every
# event in the window, which is what "read this in full" being a separate concern
# from "show me around here" buys.

_EVENT_BUDGET_CHARS = 4000


@dataclass(frozen=True)
class EventExtent:
    """How big an event really is, measured rather than inferred."""

    segments: int
    characters: int | None  # None when the payload codec hides the text from SQL


def _elision_marker(
    kept: list[Segment], handles: dict[UUID, str], extent: EventExtent
) -> str:
    """The mark between the pieces of an event too large to show whole.

    It names what is missing and hands back the surviving segments as expansion
    seeds, so the event is navigable rather than summarised: read on from its
    start, back from its end, or out from wherever the seed sat. The two ends are
    shown as CONTENT, not just as ids — that is what keeps this from being a black
    box while still refusing to spend the budget on the middle.

    **The size is measured, never predicted.** The window holds only what the walk
    fetched, so the event's real size comes from a SQL aggregate that reads no rows
    into Python. Characters are what SQLite's LENGTH() counts and what Python's
    len() reports: Unicode CODE POINTS, not bytes and not grapheme clusters. Where
    the payload codec hides the text from SQL, the marker says segments instead,
    which is always exact.
    """
    if extent.characters is not None:
        shown = sum(len(_block_text(segment.block) or "") for segment in kept)
        missing = f"{max(extent.characters - shown, 0):,} more characters"
    else:
        missing = f"{max(extent.segments - len(kept), 0):,} more segments"
    seeds = " ".join(
        handles[segment.uuid] for segment in kept if segment.uuid in handles
    )
    return f"[{missing} — memory_expand from {seeds}]"


def _fits_budget(extent: "EventExtent | None", group: list[Segment]) -> bool:
    """Whether an event is small enough to render whole."""
    measured = (
        extent.characters
        if extent is not None and extent.characters is not None
        else sum(len(_block_text(segment.block) or "") for segment in group)
    )
    return measured <= _EVENT_BUDGET_CHARS


def _sample_event(group: list[Segment], seed_uuid: UUID) -> list[Segment]:
    """The first and last segments of an event, plus the seed if it is between."""
    keep = {group[0].uuid, group[-1].uuid}
    keep.update(segment.uuid for segment in group if segment.uuid == seed_uuid)
    return [segment for segment in group if segment.uuid in keep]


# Grouping segments back into the events they were chunked from. The store returns
# a flat timeline; rendering, capping and the event-unit bookkeeping all work in
# events, and this is the one place that conversion happens.


def group_by_event(window: list[Segment]) -> list[list[Segment]]:
    """Consecutive segments of the same event, in timeline order."""
    groups: list[list[Segment]] = []
    for segment in window:
        if groups and groups[-1][0].event_uuid == segment.event_uuid:
            groups[-1].append(segment)
        else:
            groups.append([segment])
    return groups


def _seed_group_index(groups: list[list[Segment]], seed_uuid: UUID) -> int | None:
    """Which event group holds the seed, or None if a filter removed it."""
    return next(
        (
            index
            for index, group in enumerate(groups)
            if any(segment.uuid == seed_uuid for segment in group)
        ),
        None,
    )


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

    # --- short handles: abbreviate for display, resolve a prefix back ----------

    async def short_ids(self, segment_uuids: Iterable[UUID]) -> dict[UUID, str]:
        """Render each uuid as the shortest handle no other stored segment shares.

        The length comes from the uuid's immediate neighbours in sorted order:
        whatever prefix separates it from those two separates it from every other
        segment, because anything sharing more would have sorted between them.
        """
        wanted = list(dict.fromkeys(segment_uuids))
        if not wanted:
            return {}
        partition = self.opened.partition_name
        rendered: dict[UUID, str] = {}
        async with self.stores.segment_engine.connect() as connection:
            for segment_uuid in wanted:
                neighbours: list[str] = []
                for sql in _NEIGHBOUR_SQL:
                    row = (
                        await connection.execute(
                            text(sql),
                            {"partition": partition, "uuid": segment_uuid.hex},
                        )
                    ).first()
                    if row is not None:
                        neighbours.append(str(row[0]))
                rendered[segment_uuid] = memory_id_for_segment_uuid(
                    segment_uuid,
                    chars=abbreviation_length(segment_uuid.hex, neighbours),
                )
        return rendered

    async def event_extents(
        self, event_uuids: Iterable[UUID]
    ) -> dict[UUID, EventExtent]:
        """How big each event really is, without reading a row into Python.

        SQLite does the counting in C over the ``(partition_key, event_uuid)``
        index. Measured on the largest event in the store — 2,563 segments,
        842,014 characters — the aggregate takes 3.2ms, against 3.9ms merely to
        json-decode the same blocks after fetching them, and it agrees exactly.
        """
        wanted = list(dict.fromkeys(event_uuids))
        if not wanted:
            return {}
        async with self.stores.segment_engine.connect() as connection:
            rows = (
                await connection.execute(
                    text(_EVENT_EXTENT_SQL).bindparams(
                        bindparam("events", expanding=True)
                    ),
                    {
                        "partition": self.opened.partition_name,
                        "events": [uuid.hex for uuid in wanted],
                    },
                )
            ).fetchall()
        return {
            UUID(hex=str(row[0])): EventExtent(
                segments=int(row[1]),
                characters=int(row[2]) if row[2] is not None else None,
            )
            for row in rows
        }

    async def resolve_memory_id(self, memory_id: str) -> tuple[UUID | None, str]:
        """A handle's segment uuid, or the reason it did not name exactly one.

        One extra candidate over the reporting limit is fetched so the answer can
        say whether the list it shows is the whole of it.
        """
        reference = parse_memory_ref(memory_id)
        if reference is None:
            return None, f"'{memory_id}' is not a valid memory id (expected mem:<hex>)."
        async with self.stores.segment_engine.connect() as connection:
            rows = (
                await connection.execute(
                    text(_PREFIX_SQL),
                    {
                        "partition": self.opened.partition_name,
                        "low": reference,
                        "high": reference + "g",
                        "limit": ID_CANDIDATE_LIMIT + 1,
                    },
                )
            ).fetchall()
        matches = [str(row[0]) for row in rows]
        if not matches:
            return (
                None,
                f"No memory found for {memory_id} (it may not exist in memory).",
            )
        if len(matches) == 1:
            return UUID(hex=matches[0]), ""
        return None, ambiguous_id_note(memory_id, matches)

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
        within: str | None = None,
        kinds: list[str] | None = None,
        since: str | None = None,
        before: str | None = None,
        filter_spec: str | None = None,
        seen: set[str] | None = None,
        commit_seen: bool = True,
        query_vector: list[float] | None = None,
    ) -> SearchResult:
        """Associative recall over the message search surface.

        The scope is named parameters, not a filter string: ``within`` (a memory
        handle, whose conversation the search is confined to), ``kinds``, and the
        half-open time range ``since`` <= t < ``before``. Each is unrestricted when
        left None. A handle rather than a session id because a handle is the only
        kind of address these tools take.

        ``filter_spec`` is the raw-grammar escape hatch, and is NOT exposed to the
        model. The involuntary channels need one clause the parameters deliberately
        cannot express — "any session but this one, OR before this session's
        compaction" — and that disjunction is the whole of its remaining use.

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
        whole_session = None
        if within:
            anchor, problem = await self.resolve_memory_id(within)
            if anchor is None:
                return SearchResult(hits=[], new_count=0, saturated=False, note=problem)
            held = await self._get_segment_context(anchor)
            whole_session = (
                str(held[0].properties.get("session_id", "")) if held else ""
            )
        scope, problem = scope_filter(
            session=whole_session,
            kinds=kinds,
            since=since,
            before=before,
        )
        if problem:
            return SearchResult(hits=[], new_count=0, saturated=False, note=problem)
        if scope:
            filter_spec = f"({filter_spec}) AND {scope}" if filter_spec else scope
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

        handles = await self.short_ids(
            scored.seed_segment_uuid for scored in result.scored_segment_contexts
        )
        hits: list[Hit] = []
        new_count = 0
        for scored_context in result.scored_segment_contexts:
            segment_uuid = scored_context.seed_segment_uuid
            rendered = EventMemory.string_from_segment_context(
                scored_context.segments, format_options=DISPLAY_FORMAT
            )
            is_new = segment_uuid.hex not in seen_set
            if is_new:
                new_count += 1
                if commit_seen:
                    seen_set.add(segment_uuid.hex)
            hits.append(
                Hit(
                    memory_id=handles[segment_uuid],
                    score=scored_context.score,
                    text=rendered,
                    is_new=is_new,
                    segment_uuid=segment_uuid.hex,
                )
            )

        saturated = bool(hits) and new_count == 0
        # The score SHAPE, not just the top hit: a gate would need to tell "one
        # strong match" from "everything equally mediocre", and only the spread
        # carries that. Recorded so a threshold can be set from the distribution
        # rather than guessed.
        observe(
            self.stores.config,
            "search",
            cue_chars=len(cue),
            cue_words=len(cue.split()),
            limit=limit,
            filters=filter_spec or "",
            new_count=new_count,
            saturated=saturated,
            chars=sum(len(hit.text) for hit in hits),
            scores=score_shape([hit.score for hit in hits]),
        )
        return SearchResult(hits=hits, new_count=new_count, saturated=saturated)

    async def expand(
        self,
        seed_id: str,
        *,
        before: int = 5,
        after: int = 5,
        unit: str = "segments",
        seen: set[str] | None = None,
        kinds: list[str] | None = None,
        blocklist: bool = False,
    ) -> ExpandResult:
        """Return the seed's session timeline window, same-event segments merged.

        ``before`` and ``after`` count SEGMENTS — ~500-character chunks — which is a
        flat budget: every call costs about the same. ``unit="events"`` counts whole
        turns / tool calls / tool results instead, for when the question is "five
        turns either side" and the length of what is in the way should not decide
        how far the window reaches.

        ``kinds`` names sources (user_message, assistant_message, reasoning,
        tool_call, tool_result, injected), read as an allowlist or — with
        ``blocklist`` — as a blocklist. It is pushed into the store's window walk
        rather than applied to its result, so the budget is spent only on segments
        the caller asked for.
        """
        seen_set = seen if seen is not None else self.seen_segment_uuids
        seed_uuid, problem = await self.resolve_memory_id(seed_id)
        if seed_uuid is None:
            return ExpandResult(seed_id=seed_id, found=False, note=problem)

        # Scope expansion to the seed's own session. In the shared search space
        # the timeline interleaves every session, so neighbours must be filtered
        # to the same session_id — otherwise expansion crosses conversations.
        seed_segments = await self._get_segment_context(seed_uuid)
        scope: list[str] = []
        if seed_segments:
            session = seed_segments[0].properties.get("session_id")
            if isinstance(session, str) and session:
                scope.append(session_scope_filter(session))
        kind_filter = kind_scope_filter(kinds, blocklist=blocklist)
        if kind_filter:
            scope.append(kind_filter)
        neighbor_filter: FilterExpr | None = None
        if scope:
            with suppress(FilterParseError):
                neighbor_filter = parse_filter(" AND ".join(scope))

        window, at_start, at_end = await self._window_around(
            seed_uuid, before, after, unit, neighbor_filter
        )
        if not window:
            return ExpandResult(
                seed_id=seed_id,
                found=False,
                note=f"No memory found for {seed_id} (it may not exist in memory).",
            )

        window_text, shown, capped = await self._render_window(
            window, seed_uuid, unit == "events"
        )
        # Everything but the seed itself is newly surfaced — but only what was
        # actually rendered. A piece elided out of a bulky event was never put in
        # front of anyone, so recording it as seen would retire it unread.
        for segment in shown:
            if segment.uuid != seed_uuid:
                seen_set.add(segment.uuid.hex)

        # asked vs got is the yield signal, and both are in EVENTS so they compare:
        # a window that returns far fewer events than requested ran out of session
        # or was eaten by a filter, and telling those apart is the whole point.
        observe(
            self.stores.config,
            "expand",
            asked=max(before, 0) + max(after, 0) + 1,
            got=len({segment.event_uuid for segment in window}),
            segments=len(window),
            shown=len(shown),
            capped=capped,
            kinds=kinds or [],
            blocklist=blocklist,
            chars=len(window_text),
            sources=sorted(
                {
                    str(segment.properties.get("source", ""))
                    for segment in window
                    if segment.properties.get("source")
                }
            ),
        )
        edges = await self.short_ids([window[0].uuid, window[-1].uuid])
        return ExpandResult(
            seed_id=seed_id,
            window_text=window_text,
            earliest_id=edges[window[0].uuid],
            latest_id=edges[window[-1].uuid],
            session_id=str(window[0].properties.get("session_id", "")),
            events=len({segment.event_uuid for segment in window}),
            at_start=at_start,
            at_end=at_end,
        )

    async def first_handles(self, session_ids: list[str]) -> dict[str, str]:
        """The handle of each conversation's FIRST segment.

        A conversation's address. First rather than last because it does not move
        as the session grows — the same conversation keeps the same handle across
        renders, which is what lets the roster print one and a reader recognise it
        turn after turn.
        """
        if not session_ids:
            return {}
        async with self.stores.segment_engine.connect() as connection:
            rows = (
                await connection.execute(
                    text(_FIRST_SEGMENT_SQL).bindparams(
                        bindparam("sessions", expanding=True)
                    ),
                    {
                        "partition": self.opened.partition_name,
                        "sessions": list(session_ids),
                    },
                )
            ).fetchall()
        firsts = {str(row[0]): UUID(hex=str(row[1])) for row in rows}
        rendered = await self.short_ids(firsts.values())
        return {session: rendered[uuid] for session, uuid in firsts.items()}

    async def outline(
        self, memory_id: str, *, before: int = 20, after: int = 20
    ) -> OutlineResult:
        """A conversation's spine: its user turns, around the one you name.

        Addressed by a memory handle like everything else — any segment of a
        conversation names that conversation. A session's own address is its FIRST
        segment, which is stable as the session grows; the roster prints that.

        Search finds a moment and expand reads around it; neither shows the shape
        of the whole conversation, and asking for a huge expansion window to get it
        is the wrong tool. This is its own call for that reason: a fixed cost that
        does not grow with how long the session ran.
        """
        seed_uuid, problem = await self.resolve_memory_id(memory_id)
        if seed_uuid is None:
            return OutlineResult(session_id="", found=False, note=problem)
        seed = await self._get_segment_context(seed_uuid)
        session = str(seed[0].properties.get("session_id", "")) if seed else ""
        rows = await self._session_events(session) if session else []
        if not rows:
            return OutlineResult(
                session_id=session,
                found=False,
                note=f"Nothing captured for the conversation holding {memory_id}.",
            )

        # Every user turn, with the number of events that followed it before the
        # next one. That count is the density signal: a list of subjects says what
        # was discussed, and only this says where the work actually happened.
        marks = [
            index for index, row in enumerate(rows) if row[2] == Source.USER_MESSAGE
        ]
        beats = [
            (
                UUID(hex=rows[index][0]),
                rows[index][1],
                (marks[n + 1] if n + 1 < len(marks) else len(rows)) - index - 1,
            )
            for n, index in enumerate(marks)
        ]
        if not beats:
            return OutlineResult(
                session_id=session,
                project=str(rows[0][3] or ""),
                total_events=len(rows),
                span=f"{_short_time(rows[0][1])} to {_short_time(rows[-1][1])}",
            )

        # Where the named segment sits among the turns: the last turn at or before
        # it, so a handle pointing anywhere in a turn's aftermath finds that turn.
        seed_at = seed[0].timestamp if seed else rows[0][1]
        here = max(
            (i for i, beat in enumerate(beats) if beat[1] <= seat_time(seed_at)),
            default=0,
        )
        low = max(here - max(before, 0), 0)
        high = min(here + max(after, 0), len(beats) - 1)
        chosen = beats[low : high + 1]

        handles = await self.short_ids(
            [beat[0] for beat in chosen]
            + ([beats[low - 1][0]] if low > 0 else [])
            + ([beats[high + 1][0]] if high + 1 < len(beats) else [])
        )
        openings = await self._opening_words([beat[0] for beat in chosen])
        return OutlineResult(
            session_id=session,
            project=str(rows[0][3] or ""),
            total_events=len(rows),
            span=f"{_short_time(rows[0][1])} to {_short_time(rows[-1][1])}",
            beats=[
                Beat(
                    memory_id=handles[uuid],
                    when=_short_time(when),
                    text=_clip(openings.get(uuid, ""), _BEAT_CHARS),
                    events_after=events_after,
                )
                for uuid, when, events_after in chosen
            ],
            earlier_id=handles[beats[low - 1][0]] if low > 0 else None,
            later_id=handles[beats[high + 1][0]] if high + 1 < len(beats) else None,
        )

    async def _session_events(
        self, session_id: str
    ) -> list[tuple[str, datetime.datetime, str, str]]:
        """(uuid, timestamp, source, project) for each event of a session, in order.

        One row per event: taking only the first chunk of the first block turns the
        segment table into an event timeline without a GROUP BY. Index-backed on
        ``claude_memory_sg_session``. Carries no text — see ``_opening_words``.
        """
        async with self.stores.segment_engine.connect() as connection:
            rows = (
                await connection.execute(
                    text(_SESSION_TIMELINE_SQL),
                    {
                        "partition": self.opened.partition_name,
                        "session": session_id,
                    },
                )
            ).fetchall()
        timeline = [
            (
                str(uuid_hex),
                stamp if isinstance(stamp, datetime.datetime) else _parse_stamp(stamp),
                str(source or ""),
                project,
                str(event_uuid),
            )
            for uuid_hex, event_uuid, stamp, source, project in rows
        ]
        # Deterministic order for events sharing a timestamp, which a batch of tool
        # results does. Cheap here; ruinous as a SQL tiebreaker (see the query).
        timeline.sort(key=lambda row: (row[1], row[4]))
        return [row[:4] for row in timeline]

    async def _opening_words(self, uuids: list[UUID]) -> dict[UUID, str]:
        """The first words of each named segment, clipped in SQL.

        Only the turns that will be rendered. Fetching and decoding every event's
        text to show twenty of them was most of the cost of outlining a long
        session.
        """
        if not uuids:
            return {}
        async with self.stores.segment_engine.connect() as connection:
            rows = (
                await connection.execute(
                    text(_OPENING_TEXT_SQL).bindparams(
                        bindparam("uuids", expanding=True)
                    ),
                    {
                        "partition": self.opened.partition_name,
                        "uuids": [uuid.hex for uuid in uuids],
                        # Room for the clip to break on a word boundary.
                        "chars": _BEAT_CHARS * 3,
                    },
                )
            ).fetchall()
        return {UUID(hex=str(row[0])): str(row[1] or "") for row in rows}

    async def _window_around(
        self,
        seed_uuid: UUID,
        before: int,
        after: int,
        unit: str,
        neighbor_filter: FilterExpr | None,
    ) -> tuple[list[Segment], bool, bool]:
        """The seed's own segments plus its filtered neighbours, and the edges.

        The seed is an ADDRESS: the store locates it whether or not it passes the
        filter, and never returns it among the neighbours. What comes back here is
        the seed's own segments — always, because they are what the caller named —
        merged with whatever neighbours the filter admits. ``kinds`` therefore
        describes the surroundings, which is stated in the tool rather than left to
        be inferred from a window that sometimes contains the seed and sometimes
        does not.

        The two flags say whether the window reaches the start / end of the
        conversation, read off whether fewer neighbours came back than were asked
        for. Without them the navigation hints offer to walk further off a
        conversation that has already run out.
        """
        wanted_back, wanted_forward = max(before, 0), max(after, 0)
        partition = self.opened.partition
        if unit == "events":
            neighbors = (
                await partition.get_neighbor_events(
                    seed_segment_uuids=[seed_uuid],
                    max_backward_events=wanted_back,
                    max_forward_events=wanted_forward,
                    property_filter=neighbor_filter,
                )
            ).get(seed_uuid) or []
            seed_segments = await self._seed_event_segments(seed_uuid)
        else:
            neighbors = (
                await partition.get_neighbor_segments(
                    seed_segment_uuids=[seed_uuid],
                    max_backward_segments=wanted_back,
                    max_forward_segments=wanted_forward,
                    property_filter=neighbor_filter,
                )
            ).get(seed_uuid) or []
            seed_segments = [
                segment
                for segment in (await self._get_segment_context(seed_uuid))
                if segment.uuid == seed_uuid
            ]

        window = sorted([*neighbors, *seed_segments], key=segment_sort_key)
        seed_stamp = seed_segments[0].timestamp if seed_segments else None
        if seed_stamp is None:
            return window, False, False
        if unit == "events":
            seed_event = seed_segments[0].event_uuid
            groups = group_by_event(neighbors)
            got_back = sum(
                1
                for g in groups
                if (g[0].timestamp, g[0].event_uuid) < (seed_stamp, seed_event)
            )
            got_forward = len(groups) - got_back
        else:
            got_back = sum(
                1
                for s in neighbors
                if segment_sort_key(s) < segment_sort_key(seed_segments[0])
            )
            got_forward = len(neighbors) - got_back
        return window, got_back < wanted_back, got_forward < wanted_forward

    async def _get_segment_context(self, seed_uuid: UUID) -> list[Segment]:
        """The seed segment alone, unfiltered — it is an address, not a result."""
        contexts = await self.opened.partition.get_segment_contexts(
            seed_segment_uuids=[seed_uuid],
            max_backward_segments=0,
            max_forward_segments=0,
            property_filter=None,
        )
        return list(contexts.get(seed_uuid) or [])

    async def _seed_event_segments(self, seed_uuid: UUID) -> list[Segment]:
        """Every segment of the seed's own event, in order.

        In the event unit the neighbours are whole events and the seed's event is
        the anchor they are counted from, so the store excludes it entirely; the
        caller named a place inside it, so this puts it back.
        """
        seed = await self._get_segment_context(seed_uuid)
        if not seed:
            return []
        uuids = (
            await self.opened.partition.get_segment_uuids_by_event_uuids(
                [seed[0].event_uuid]
            )
        ).get(seed[0].event_uuid) or []
        contexts = await self.opened.partition.get_segment_contexts(
            seed_segment_uuids=uuids,
            max_backward_segments=0,
            max_forward_segments=0,
            property_filter=None,
        )
        return sorted(
            (segment for pieces in contexts.values() for segment in pieces),
            key=segment_sort_key,
        )

    async def _render_window(
        self, window: list[Segment], seed_uuid: UUID, whole_events: bool
    ) -> tuple[str, list[Segment], int]:
        """Render the window event by event, sampling the ones over budget.

        Rendering per event rather than in one pass is what makes room for a
        per-event marker; ``string_from_segment_context`` merges only within an
        event, so joining the groups reproduces the single-pass output exactly
        whenever nothing is sampled.

        ``whole_events`` says whether the window holds complete events. It does in
        the event unit, where the budget is what keeps one long output from
        crowding out the conversation. In the segment unit the caller has already
        chosen how many chunks to spend and is reading INSIDE something on purpose,
        so a second budget there would only take back what was asked for.
        """
        groups = group_by_event(window)
        extents = (
            await self.event_extents(group[0].event_uuid for group in groups)
            if whole_events
            else {}
        )
        kept_by_group = [
            group
            if not whole_events or _fits_budget(extents.get(group[0].event_uuid), group)
            else _sample_event(group, seed_uuid)
            for group in groups
        ]
        sampled = [
            group
            for group, kept in zip(groups, kept_by_group, strict=True)
            if len(kept) < len(group)
        ]
        # Every surviving segment of a sampled event is offered as a handle, so the
        # event can be read on from its start, back from its end, or out from where
        # the seed sat.
        handles = await self.short_ids(
            segment.uuid
            for group, kept in zip(groups, kept_by_group, strict=True)
            if len(kept) < len(group)
            for segment in kept
        )

        blocks: list[str] = []
        shown: list[Segment] = []
        for group, kept in zip(groups, kept_by_group, strict=True):
            shown += kept
            options = DISPLAY_FORMAT
            if len(kept) < len(group):
                options = DISPLAY_FORMAT.model_copy(
                    update={
                        "gap_marker": _elision_marker(
                            kept,
                            handles,
                            extents.get(
                                group[0].event_uuid, EventExtent(len(group), None)
                            ),
                        )
                    }
                )
            rendered = EventMemory.string_from_segment_context(
                kept, format_options=options
            )
            if any(segment.uuid == seed_uuid for segment in group):
                rendered += ANCHOR_MARKER
            blocks.append(rendered)
        return "\n".join(blocks), shown, len(sampled)

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
        seg_uuid, problem = await self.resolve_memory_id(memory_id)
        if seg_uuid is None:
            return problem
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
        handle = (await self.short_ids([seg_uuid]))[seg_uuid]
        return f"Noted. This memory now reads:\n{format_memory_line(handle, rendered)}"

    async def _demote_resolve(
        self, memory_id: str
    ) -> "tuple[UUID, Segment, list[UUID]] | DemoteResult":
        """Resolve a handle to (segment uuid, segment, derivatives) or a DemoteResult."""
        seg_uuid, problem = await self.resolve_memory_id(memory_id)
        if seg_uuid is None:
            # A handle can fail to name one memory two ways, and they call for
            # different things from the caller: fix the string, or lengthen it.
            malformed = parse_memory_ref(memory_id) is None
            verdict = "invalid" if malformed else "unresolved"
            return DemoteResult(False, verdict, problem, memory_id)
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
