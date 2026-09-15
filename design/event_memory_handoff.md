# EventMemory and SegmentStore: implementation requirements

A handoff for the agent implementing the accepted parts of the server
redesign that live in `EventMemory`, `SegmentStore` and their data
models. Everything else in `design/server_redesign.md` (tenants,
lifecycle, handles, the event store, the manager, settings, the HTTP
API) is out of scope here, and nothing is renamed: the class stays
`EventMemory`, the package stays `episodic_memory/event_memory/`.

Target: a branch off upstream `speedkick`, stacked on PR #1598 (cosine
as the only similarity, `min_cosine_similarity`, scores and uuids from
`query`; it replaced #1591 and #1593). Eviction needs a threshold
query and a single score scale, and #1598 provides both, so base on it
rather than re-deriving them. #1597 is the implementation of this
document on that base.

## References

Worktrees on this machine:

- `speedkick` code under this worktree:
  `packages/server/src/memmachine_server/episodic_memory/event_memory/`
  (`data_types.py`, `event_memory.py`, `segment_store/segment_store.py`,
  `segment_store/sqlalchemy_segment_store.py`, `deriver/text_deriver.py`,
  `segmenter/text_segmenter.py`), `common/filter/` (the string parser
  and `sql_filter_util.py`), `common/vector_store/`.
- The claude-memory branch, worktree
  `/Users/eyu/edwinyyyu/mmcc/agentic_expansion`, same package paths.
  What to take from it, by file and line at its tip `05902295`:
  - `segment_store/segment_store.py:73` `get_neighbor_segments` and
    `sqlalchemy_segment_store.py:384` its implementation (commit
    `0c19942a`): the neighbors-only read, which becomes
    `get_segment_neighborhoods`. Do not port `get_neighbor_events` (`:113`,
    `:454`); segments are the one unit.
  - `event_memory.py:429` `_compute_batch_predecessors`, `:480`
    `_select_eviction_targets`, and the eviction step inside
    `_encode_events` (`:267`, the block after the embedder call;
    commit `ed2c5702`).
  - `event_memory.py:762` `_immediately_follows`: adjacency of two
    segments of one event is `(index, offset)`, not `index` alone
    (commit `0c19942a`).
  - `common/filter/sql_filter_util.py` `_normalize_column_value`: a
    datetime bound is put in UTC before it is compared with a stored
    column. Check whether upstream #1462 already landed this on
    `speedkick`; port it if not.
- The `default` branch of `edwinyyyu/MemMachine`, worktree
  `/Users/eyu/edwinyyyu/mmcc/default`: commit `27b3279b` adds
  `common/property_keys.py` (the reserved key namespace and the key
  validator) and `822ccb6b` the closed filter union
  (`common/filter/filter_expression.py`). Take `property_keys.py` as
  it is; the filter union is optional here (below).
- Design, on branch `design/tenant-lifecycle` (this worktree):
  `design/components/episodic_memory.md` (API, eviction, expansion),
  `design/components/segment_store.md` (schema, the total order, the
  neighbors rule), `design/components/filters_and_properties.md`
  (reserved keys, system fields as typed parameters, the tree),
  `design/components/context.md` (parts), `design/components/blocks.md`
  (kinds). They describe the full redesign; take only what this
  document lists.
- MemMachine #1498 (closed): the rule for seeds versus anchors.

## Data models (`data_types.py`)

`Event`:

```python
class Event(BaseModel):
    uuid: UUID
    timestamp: AwareDatetime       # a naive value is rejected; keeps the offset it was given
    session_id: str                # the stream the event belongs to; required, non-empty
    source_id: str | None = None   # the responsible entity's id; None = null
    context: Context               # a mapping of parts; {} = no context
    blocks: list[Block]            # one or more, each of a registered kind
    properties: dict[str, PropertyValue] = Field(default_factory=dict)
```

- `session_id` and `source_id` are first-class fields of the model,
  not properties. The session is required: it is the unit the order is
  partitioned into, every event belongs to one stream, and a walk never
  leaves it. The source is nullable, and its null is encoded as a
  missing key on the vector record, so every backend's "missing"
  implements `IS NULL`. A typed id list (`session_ids`, `source_ids`)
  holds ids only, and a filter that names no session means every
  session. Property values are never `None`: absence is the one no-value
  state. One session name is reserved, `memmachine_default`: the stream
  of events the API ingests without a conversation id, a stop-gap until
  the API requires one (below, "Translation layers"). Any other name,
  `memmachine_`-prefixed or not, is a caller's to use.
- Stored events are immutable (`server_redesign.md`, "Propagation"):
  `encode_events` replaces an event's earlier encoding wholesale under
  new segment and derivative uuids, `forget_events` removes one, and no
  operation edits a stored segment or a vector record. The rule is stated
  on `Event`, on `EventMemory` and on the store contract, whose
  `add_segments` rejects a stored uuid, so the vector record's copy of
  its segment's fields is exact by construction and a future update
  operation has to argue with three docstrings and a test.
- Both are bounded strings; bound them by the same limit as a property
  string value, and reject longer ones where events are validated
  (`EventMemory._validate_events`).
- `Segment` and `Derivative` gain the same two fields, copied verbatim
  from the event by the segmenter and the deriver; that copy is a
  clause of both contracts and the segment store depends on it.
  `Derivative` carries the derived content and the segment's fields a
  vector record carries (timestamp, session, source, block kind) and
  nothing else: a deriver reads the segment's context while composing
  the text, and nothing reads a copy of the context or the properties
  on the derivative, so it has none.
- Timestamps are timezone-aware everywhere: the model rejects a naive
  value, and the typed bounds `since` and `until` reject one, since a
  naive datetime names no instant and a guessed zone would silently
  shift an event in the order. The timestamp round-trips with its offset: the segment store already
  keeps `timestamp_timezone_offset`; keep writing the UTC instant plus
  the offset and reapplying it on read.

`Block`:

```python
class Block(BaseModel, ABC):
    kind: str                                # a Literal on each subclass
    def render(self, datetime_format: DateTimeFormat) -> str | None: ...

class TextBlock(Block):
    kind: Literal["text"] = "text"
    text: str
```

- `block_type` becomes `kind`, the one discriminator name used for
  every registered family. `TextBlock.render` returns its text;
  `EventMemory._extract_text` goes and rendering calls `render`.
- The union may stay closed over `TextBlock` in this change. If the
  kind table and the `memmachine.block_kinds` entry-point group are
  added (`blocks.md`), the codec decodes through the table and an
  unregistered kind decodes to `UnknownBlock(kind, data)`; otherwise
  leave that for the next change. Either way the segment row records
  the kind name (below).

`Context`:

```python
class ContextPart(BaseModel, ABC):
    kind: ClassVar[str]
class Author(ContextPart):          # kind = "author"
    name: str
class UnknownPart(ContextPart):     # produced only by decode
    kind_name: str
    data: dict[str, JsonValue]
class Context:                               # at most one part of each kind
    def __init__(self, *parts: ContextPart)
    def get(self, kind: str) -> ContextPart | None
    def with_part(self, part: ContextPart) -> Context
    def encode(self) -> dict[str, JsonValue]
    @classmethod
    def decode(cls, encoded: Mapping[str, JsonValue]) -> Context
```

- Replaces `ProducerContext`, `NullContext` and the discriminated
  union. No context is `Context()`, never `None`; a context is built
  from parts and keyed by their kinds internally, so a caller never
  writes a key. A part is registered under its kind in a table (by
  import for `Author`; the entry-point group `memmachine.context_parts`
  may wait). `Context.encode` and `Context.decode` encode the parts as
  `{kind: fields}` and decode an unregistered kind to `UnknownPart`,
  which round-trips unchanged and renders nothing.
- The deriver reads `context.get("author")` where it read
  `ProducerContext.producer`; rendering prints the author's name.
- Do not port `AnnotationContext`, `CompositeContext` or
  `find_contexts` from the branch: annotate is out of scope, and keyed
  parts replace composition.

Results:

```python
class QueryHit(BaseModel):
    score: float                    # relevance of the seed to the query; cosine similarity from `query`
    seed: Segment                   # the segment the query matched
    neighborhood: Neighborhood      # the segments around it, in the store's order
    def window(self) -> list[Segment]   # before, seed, after

class Neighborhood(BaseModel):
    before: list[Segment]           # in order, ending just before the anchor
    after: list[Segment]            # in order, starting just after it

class EvictionOptions(BaseModel):
    cosine_similarity_threshold: float  # at or above it, eviction is considered
    search_limit: int               # stored derivatives at or above it fetched per new one
    target_size: int                # how many of a new derivative and those at or above it are kept
```

`QueryHit` replaces `ScoredSegmentContext` (`seed_segment_uuid` is
`seed.uuid`), and `EventMemory.query` returns `list[QueryHit]` in place
of `QueryResult`; a hit has the shape `expand` returns around a seed, so
walking further from a hit composes with `expand`.

## Reserved property keys and system fields

- Take `property_keys.py` from `default` (`27b3279b`):
  `RESERVED_PROPERTY_KEY_PREFIX = "memmachine_"`,
  `reserved_property_key(system, field)`, `validate_caller_property_key`.
  A caller key with the prefix is rejected at `_validate_events`; the
  current `_BASE_EVENT_MEMORY_FIELD_NAMES` check becomes the prefix
  check.
- Every system value written into a vector record uses a reserved key:
  `memmachine_em_timestamp`, `memmachine_em_session`,
  `memmachine_em_source`, `memmachine_em_block_kind`. The current
  `_segment_uuid` and `_timestamp` keys go: the segment store maps a
  derivative to its segment (#1598), so no uuid is written into a
  record. `expected_vector_store_collection_schema` declares the four
  with their types, so the vector store indexes them through the
  collection schema mechanism it has today, and a record carries the
  four and the vector, nothing else: the caller's properties stay with
  the segment, where `property_filter` reads them, so the collection
  schema a memory needs is exactly the four and a store that refuses a
  key it has not declared (`vector_store.md`) takes every record the
  memory writes. The server's adapter fields (`_episode_uid`,
  `_producer_id`, ...) are caller properties to the memory and follow
  the same path. No central list of reserved keys exists; the prefix
  is reserved as a whole and each service names its own under it.
- The prefix is reserved whether or not a caller is ever allowed to
  name a system field inside a filter, so both answers to the question
  below stay open.

Typed parameters. The memory takes the system filters as typed
parameters (`since`, `until`, `session_ids`, `source_ids`,
`block_kinds`) beside the caller's `property_filter`, and a caller
never names a system field in a tree the memory sees: the split
between the two is the API's, made at the boundary where a request is
read, and hoisted at least to `LongTermMemory` for the legacy API,
which carries the fields its ingestion maps onto the event in its
filter tree: it lifts every top-level conjunct on a mapped field back
into the typed parameter (`timestamp` and `created_at` bounds into
`since` and `until`, `producer_id =` and `IN` into `source_ids`); any
other predicate stays a post-filter.

- `EventMemory` owns `_system_predicates(since, until, session_ids,
  source_ids, block_kinds) -> FilterExpr | None`, which builds the tree
  the vector store gets, on reserved keys. The segment store never sees
  a tree for system fields: it gets the typed values and compares
  columns. `property_filter` is the segment store's alone.
- `EventMemory.query` and `expand` take the typed parameters.

Optional: the closed filter union from `822ccb6b` (`Equals`,
`NotEquals`, `Ordering`, `In`, `IsMissing`, `And`, `Or`, `Not`), which
removes the string parser. It is the design's target and touches
`common/filter` and every vector store's filter compiler; take it only
if the string parser is in the way, since the server still calls it.

## Segment store

Schema (`sqlalchemy_segment_store.py`), keeping the incarnation and
the partition tables exactly as shipped in #1548 since lifecycle is
out of scope:

- `segment_store_sg` gains `session_id VARCHAR(255) NOT NULL`,
  `source_id VARCHAR(255) NULL` and `block_kind VARCHAR(255) NOT NULL`:
  the type of `partition_key` and of the vector stores' key columns,
  for every string column the store compares on, so the database
  enforces the bound the model does and every filter column can be an
  index key on any SQL database (on PostgreSQL `varchar(n)` is `text`
  plus the length check and on SQLite both have text affinity; on
  MySQL, SQL Server and Oracle `text` is not an index key). All three
  are projections of
  the segment the store already holds, written in the same insert as
  the encoded block, the way `timestamp` and `properties` already are:
  the codec-encoded block is opaque to SQL, so what the store filters
  on is copied out beside it. The model does not change; `Segment`
  still carries one `block` with its `kind` inside. The store derives
  `block_kind` from `segment.block.kind` and never accepts it as a
  separate input, so the column cannot disagree with the block.
- Indexes: keep `(incarnation, event_uuid, index, offset)` for lookup
  by event; replace the timestamp ordering index with
  `segment_store_sg__in_se_ts_ev_ix_of (incarnation, session_id, timestamp,
  event_uuid, index, offset)`, which serves every walk since every walk
  pins a session; add `segment_store_sg__in_se_so_ts_ev_ix_of (incarnation,
  session_id, source_id, timestamp, event_uuid, index, offset)`, the same
  order with the source pinned, which the planner takes for every walk
  filtered by one source (measured on PostgreSQL; an index on the source
  alone served no read of the store's). Every index is one a read
  chooses; none for `block_kind` until a second kind exists and a
  kind-filtered walk is measured.
- The total order is `(timestamp, event_uuid, index, offset)` within
  an incarnation; a walk is confined to the seed's session by an
  equality predicate on the session id. The tie-break
  stays `event_uuid` here; the redesign's event position needs the
  event store, which is out of scope.
- Migration: none, per the 2026-09-10 decision that migrations wait
  for the lifecycle/DDL work; `startup()` keeps `create_all`, and a
  `speedkick` database is recreated. Rows written before the change
  would not decode anyway (their discriminators were `context_type` /
  `block_type`).

`SegmentStorePartition` (`segment_store/segment_store.py`):

```python
async def get_segments(self, segment_uuids: Iterable[UUID], *,
        since: datetime | None = None, until: datetime | None = None,
        session_ids: Iterable[str] | None = None,
        source_ids: Iterable[str] | None = None,
        block_kinds: Iterable[str] | None = None,
        property_filter: FilterExpr | None = None) -> dict[UUID, Segment]

async def get_segment_neighborhoods(self, seed_segment_uuids: Iterable[UUID], *,
        before: int = 0, after: int = 0,
        since: datetime | None = None, until: datetime | None = None,
        session_ids: Iterable[str] | None = None,
        source_ids: Iterable[str] | None = None,
        block_kinds: Iterable[str] | None = None,
        property_filter: FilterExpr | None = None) -> dict[UUID, Neighborhood]

async def delete_derivatives(self, derivative_uuids: Iterable[UUID]) -> None
```

- The two reads have different jobs, a filtered lookup and a walk.
  `before` and `after` count neighbors on each side of a given segment
  (the shipped `max_backward_segments` and `max_forward_segments`,
  renamed);
  `since` is inclusive and `until` exclusive on the `timestamp` column,
  so ranges meet without overlap (`until`, not `before`, so that
  `before` is a count everywhere); `source_ids`, `block_kinds` and
  `property_filter` select rows. A walk is confined to the given
  segment's session by an equality on its session id, or spans every
  session when it has none. The lateral and loop plans serve it, with
  the located seeds' keys bound as parameters.
- `get_segments` is the filtered lookup: the segments among the given
  uuids that the partition holds and that pass every filter; a uuid
  that fails has no entry. A search fetches its seeds with it.
- `get_segment_neighborhoods` is the walk. The seed is an address: it
  is located by uuid whether or not it passes any filter, the filters
  apply to the neighbors only, and the seed is never in the result. Each seed maps to two lists in the store's order, `before`
  ending just before the seed and `after` starting just after it, so
  the seed's place is between them and the caller needs nothing but
  the lists. A seed with no neighbors to show maps to two empty lists;
  an unknown seed is absent from the mapping. Port the branch's
  `get_neighbor_segments` and split its one list at the seed's position
  in the order. This is the rule of #1498: during a search a seed that
  fails is dropped before a window is built; after a search a
  neighborhood is kept even when its seed would fail, and then the
  seed is never returned.
- `delete_derivatives` removes link rows by derivative uuid and leaves
  the segments; eviction needs it.
- Datetime bounds are normalized to UTC before binding
  (`_normalize_column_value`), on both the column comparisons and the
  JSON property path.

## EventMemory

`EventMemoryParams`: `reranker` goes, and so does the per-call
`format_options` on `encode_events`; `FormatOptions` is `DateTimeFormat`,
date and time styles, locale and zone, and nothing else; `eviction:
EvictionOptions | None`
is added. The deriver owns the format of what it embeds
(`WholeTextDeriver(datetime_format)` here, the handler's `datetime_format`
and `parts` under the
tables of the second half), since a deriver decides the text it embeds
and one format per memory would assume every deriver wants the same
one; a display format is a call argument.

```python
class EventMemory:
    async def encode_events(self, events: Iterable[Event]) -> None
    async def forget_events(self, event_uuids: Iterable[UUID]) -> None
    async def query(self, query: str, *,
                    vector_search_limit: int, min_cosine_similarity: float | None,
                    expand_context: int,
                    since: datetime | None, until: datetime | None,
                    session_ids: Iterable[str] | None,
                    source_ids: Iterable[str] | None,
                    block_kinds: Iterable[str] | None,
                    property_filter: FilterExpr | None) -> list[QueryHit]
    async def expand(self, anchor: UUID, *, before: int, after: int,
                     since: datetime | None, until: datetime | None,
                     session_ids: Iterable[str] | None,
                     source_ids: Iterable[str] | None,
                     block_kinds: Iterable[str] | None,
                     property_filter: FilterExpr | None) -> Neighborhood
    @staticmethod
    def render(segments: Iterable[Segment], *,
               datetime_format: DateTimeFormat,
               parts: Iterable[str] = ("author",)) -> str
    @staticmethod
    async def rerank(query: str, hits: Sequence[QueryHit], *,
                     reranker: Reranker,
                     datetime_format: DateTimeFormat) -> list[QueryHit]
```

- `encode_events`: first `forget_events` for the batch's event uuids,
  so a repeated batch leaves one copy; then segment, derive, embed;
  then eviction (below); then `add_segments` with the surviving
  derivatives' links, `upsert` of the surviving records with the
  reserved keys only, and `delete` of the
  displaced records plus `delete_derivatives` of their links. The
  order segments then vectors is unchanged. Drop the branch's
  `serialize_encode` lock.
- `query` is the vector stage only: embed the query; `query` the
  collection with `vector_search_limit`, `min_cosine_similarity` and
  `system_predicates(...)` alone, since `property_filter` is the
  caller's, over properties the vector store does not index, and it
  never reaches the vector store; resolve seeds
  through the segment store's `get_segment_uuids_by_derivative_uuids`
  (#1598); `get_segments` with the same system values and
  `property_filter`, the post-filter, then `get_segment_neighborhoods` from the seeds
  it returned, `expand_context` split as today and no walk when it is
  zero; drop seeds the store did not return; return at most `vector_search_limit`
  hits in descending cosine similarity, each its seed with the
  neighborhood around it. Neighborhoods of different hits may overlap
  and each hit is returned whole. Every count is a maximum.
- `rerank` is the second stage, a static helper so a caller that has a
  reranker (the server's `LongTermMemory`, the claude-memory engine)
  runs it after `query` over `render(hit.window())`; it returns every
  hit rescored in descending score, and the caller cuts and thresholds,
  since the memory makes no use of either bound. It replaces the
  reranking that `_query` did inside. Call sites in the server change
  only as far as calling it; nothing else in the server is in scope.
- `expand`: an event uuid anchor resolves to its first segment via
  `get_segment_uuids_by_event_uuids`; then `get_segment_neighborhoods`
  from that uuid with the same filters a search takes, after
  `get_segments` with `session_ids` when sessions are named, so an
  anchor outside them is not found.
- `render` replaces `string_from_segment_context` and
  `string_from_segment_contexts` and uses `_immediately_follows` for
  the header decision: a new header when the segment is not the very
  next piece of the same event. No gap marker.
- `build_query_result_context` and `string_from_query_result`, already
  gone on the branch, go here too.

Eviction, from the branch, cosine only:

- `_compute_batch_predecessors(embeddings, threshold)`: normalize, one
  matrix product, keep only earlier indices (`j < i`) at or above the
  threshold, so a batch evicts exactly what serial ingestion would.
  The metric `match` goes.
- Stored neighbors: one collection `query` per derivative with the
  batch's embeddings, `min_cosine_similarity=threshold`,
  `limit=search_limit`, properties returned so the event timestamp is
  known.
- `_select_eviction_targets` as on the branch: the cluster is the
  stored neighbors not already displaced in this batch, the batch
  predecessors not already skipped, and the derivative itself; within
  `target_size` nothing happens; over it, sort by event timestamp and
  keep the earliest `target_size // 2` and the latest remainder; the
  middle's stored members are displaced and batch members skipped.
- Displaced derivatives lose their vector record and their link row;
  skipped ones are never written. Segments and events are untouched.
- `eviction=None` skips all of it, including the queries.

## Segmenter and deriver

`blocks.md`, "Processing": `Segmenter` and `Deriver` become tables
from block kind to handler, `BlockSegmenter[B]` and `BlockDeriver[B]`
the one-kind handler contracts, `Piece` what a segmenter handler
returns and `list[str]` what a deriver handler returns; the table
builds every envelope. `Derivative.block` becomes `text` plus
`block_kind`. `format_header` composes the embedded text and the
rendered header in one place, ordered by the composer's `parts`
(`context.md`, "Rendering"); the `Deriver` table composes each
derivative's text with its handler's `datetime_format` and `parts`.
`TextSegmenter`, `WholeTextDeriver` and `SentenceTextDeriver` become
`text` handlers; `PassthroughSegmenter` and the `passthrough`
configuration name go, an omitted `segmenter` meaning one segment per
block. The base-from-defaults and per-kind options of `blocks.md` are
not in this change. This is the third PR of the split; the first
carries session, source and expansion, the second eviction.

## Translation layers

Nothing in the server is rewired here, but the server's translation
from `Episode` to `Event` (`episodic_memory/long_term_memory/`) sets
`source_id = producer_id`, `timestamp = created_at` and
`session_id = memmachine_default`: the server's session is its own
grouping, not a conversation, and the API carries no conversation id,
so a partition's events are one stream under the reserved name; when
the API carries one, it goes here. Add no `Author` part, since the
server holds no readable name. The claude-memory engine already keeps
a session id and an author in properties; it moves them into the two
fields and keeps the rest of its properties as they are.

## Tests

- Port the branch's neighbor tests
  (`server_tests/.../segment_store/test_sqlalchemy_segment_store.py`
  on `agentic_expansion`) to the two-list shape, on both dialects, and
  add: the anchor is absent from both lists; an anchor that fails the
  filter still yields its neighbors; `since`/`until` meet
  without overlap on a boundary timestamp; a non-UTC bound compares as
  an instant on SQLite.
- Eviction tests from the branch (`test_event_memory.py`): cluster
  within target unchanged; over target trims the middle; batch
  predecessors mimic serial order; displaced links are gone;
  `eviction=None` issues no query.
- Do not test retrieval order with a fake embedder that ties every
  score; under cosine such a fake makes every ordering assertion
  vacuous. Use a fake whose vectors differ per text, and assert the
  contract (a neighbor ranks last, a threshold excludes) rather than
  an exact list.
- Run the new store tests against the unfixed store first, so each
  asserts something the change made true.

## Not in this change

Tenants, handles, the event store, positions, the manager and the
staged-search API, settings, the HTTP API, the `EpisodicMemory` rename,
block-kind registration beyond the rename, the `query_vector`
parameter, the gap marker, annotate and demote, and the vector store's
declared-index model. The design documents describe them; they land
separately.
