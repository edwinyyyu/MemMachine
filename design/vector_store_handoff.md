# Vector store: implementation requirements

A handoff for the agent implementing the accepted vector store changes
of the server redesign, cut so that none of it depends on the tenant
service, jobs, tombstones, handles or the key registry.

## The seam

The vector store changes fall on two sides of one line.

Records and queries: what a record carries, which keys are indexed,
what a filter may say, which nodes each backend evaluates during a
search, how a score is read back, how datetimes are stored, what a
client needs. None of this cares how a collection is addressed or how
it lives and dies. This document is that side.

Collections and keys: UUID keys minted once, one native container per
embedder with the tenant as a value inside it, registry rows in the key
registry or beside the data, strict create, logical delete, batched
purge, the check after each remote operation, stateless handles, the
SQLite stores on shared tables, containers provisioned by the schema
command, and the removal of incarnations, content-addressed names and
Qdrant shard keys. All of it rests on a key that is never reused, which
is what the tenant service's tombstone provides, so it lands with the
lifecycle work (`design/components/vector_store.md`, "Changes required",
and `design/server_redesign.md`, "Vector store"). Doing it now would
mean either incarnations, which the design then removes, or a
never-reused key nothing yet guarantees. Everything on this side of
the line is written against `VectorStoreCollection` as it is today, so
the later change replaces how a handle is obtained and nothing this
document adds.

## Base and order

- Branch off upstream `speedkick` after #1598 (cosine scores and uuids;
  replaces #1591 and #1593) and #1597 (the EventMemory handoff, which
  owns the reserved keys, `system_filters.py`, and the segment store's
  `session_id`, `source_id` and `block_kind` columns). #1599 (turbovec
  engine) is independent. #1588 (atomic index publish) is merged.
- One decision #1598 made must be reopened here, below under "Scoring
  by id".

## References

- `speedkick` in this worktree, `packages/server/src/memmachine_server/`:
  `common/vector_store/vector_store.py` (the ABCs: `VectorStoreCollection`
  from line 22, `VectorStore` from 147), `common/vector_store/data_types.py`
  (`VectorStoreCollectionConfig` at 18 with `indexed_properties_schema`,
  `Record`, `QueryMatch`, `QueryResult`), the four stores and their
  filter compilers (`qdrant_vector_store.py:88` `_build_qdrant_filter`,
  `milvus_vector_store.py:92`, `sqlite_vec_vector_store.py:254`,
  `sqlite_vector_store.py:414` `_build_key_filter`), the engines under
  `common/vector_store/vector_search_engine/`, `common/filter/` (the
  string parser and `sql_filter_util.py`).
- The `default` branch of `edwinyyyu/MemMachine`, worktree
  `/Users/eyu/edwinyyyu/mmcc/default`:
  - `822ccb6b`, the closed filter union
    (`common/filter/filter_expression.py`) and the per-backend
    compilers recompiled over it.
  - `2d5dc2b5`, score-only queries, `get_cosine_similarity` per engine,
    and the selectivity plan (a LIMIT probe; selective filters become
    an allowlist, broad ones widen with a cap).
  - `b17fd0a1`, the widening cap returns what survived.
  - `a0753d3c`, Qdrant `hnsw_config`, `optimizers_config` and
    `quantization_config` as plain mappings validated against the
    Qdrant models, with the `m = 0` rule.
  - `27b3279b`, `common/property_keys.py` (already taken by #1597).
- Design: `design/components/vector_store.md` (contract, backends
  table, schemas), `design/components/filters_and_properties.md`
  (union, provider support table, normalizations, `split_declared`),
  `design/server_redesign.md` under "Properties and filtering" (why
  declared indexes, routing, limits are maximums).

## Declared, not dynamic, indexes

Today each collection carries its own `indexed_properties_schema` in
`VectorStoreCollectionConfig`, and `EventMemory` tells the collection
what it expects. That is the dynamic index creation the redesign
prohibits: a filter index appears per collection, from a request's
configuration, with no owner and no migration.

- The store declares one schema for every collection it holds:
  `indexed_properties: Mapping[str, PropertyType]` in the store's
  params, from deployment configuration. `PropertyType` is the scalar
  set (`str`, `int`, `float`, `bool`, `datetime`).
- The system keys are always declared. `common/property_keys.py`
  gains `SYSTEM_PROPERTY_SCHEMA`, the four reserved keys #1597 writes
  (`memmachine_event_timestamp: datetime`, `memmachine_event_session:
  str`, `memmachine_event_source: str`, `memmachine_block_kind: str`),
  and every store adds it to its declared schema. Nothing in the
  vector store imports from `event_memory`.
- `indexed_properties_schema` leaves `VectorStoreCollectionConfig`;
  `create_collection` and `open_or_create_collection` keep their
  shape otherwise (their removal is the other side of the seam).
  `EventMemory.expected_vector_store_collection_schema` goes; instead
  `EventMemory` checks at construction that the collection's store
  declares the four system keys and raises `InvalidCollectionSchemaError`
  if not.
- The server's configuration gains the store-level
  `indexed_properties` setting and stops passing a schema per
  collection; that is the one wiring change, and it is a config field.

## Undeclared keys are rejected

- `upsert` raises `UndeclaredPropertyKeyError` before anything is sent
  when a record carries a key the store has not declared.
- `query` raises the same when a filter names an undeclared key. An
  undeclared key therefore never exists in the vector store: not stored
  write-only, not scanned for. The split between what the store filters
  and what the segment store filters is the caller's job (below), never
  something a store guesses at.
- Every store keeps storing declared properties and filtering on the
  stored copy; nothing returns them (#1598).

## The filter language

- Take the closed union from `822ccb6b`: `Equals`, `NotEquals`,
  `Ordering` (numbers and datetimes only), `In` (homogeneous,
  non-empty), `IsMissing`, n-ary `And` and `Or`, `Not`, in
  `common/filter/filter_expression.py`. `Comparison` and `IsNull` go.
  The string parser may stay as the server's translation into the new
  union for now, since the HTTP API still speaks it; `filter_from_json`
  from the same commit is the JSON form the redesign's API will use.
- Each store compiles the tree with an exhaustive `match`, so a node
  it does not handle is a type error, and declares
  `supported_filter_nodes: frozenset[type]`, the node classes it
  evaluates during the search, per the table in
  `filters_and_properties.md`: Qdrant, Milvus and the SQL stores
  evaluate everything; sqlite-vec evaluates `Equals`, `NotEquals`,
  `Ordering` and `And` only (its KNN takes comparisons joined by
  `AND`); the engine-backed store post-filters over its records table
  and reports the SQL set. A `query` whose tree uses a node outside the
  set raises `UnsupportedFilterError`.
- Two normalizations belong to the compilers that need them: `Not` is
  pushed to the leaves by De Morgan where a backend has no negation
  node, and `NotEquals` compiles to "not equal and exists" where a
  backend's inequality would match records lacking the key. Semantics
  are the design's: a predicate matches only a record holding a value
  of the compared type; `NotEquals` keeps records holding a differing
  value; `Not(Equals)` also keeps records holding none.

## Scoring by id

The selective plan (below) scores a small, known set of derivative
records against the query vector. #1598 removed `get` and added no
scoring-by-id entry point, on the reasoning that property filtering
stays inside the store so a candidate set never leaves it. Under the
declared-index model that is no longer true: an undeclared key is
filtered in the segment store, so the candidate set is assembled
outside the vector store and must be scored by id.

- Add `get_cosine_similarity(vector: Sequence[float], uuids:
  Iterable[UUID]) -> dict[UUID, float]` to `VectorStoreCollection`, as
  #1593 had it and `2d5dc2b5` implements it per backend: keyed vector
  access where an engine can return a stored vector (hnswlib, usearch,
  sqlite-vec via `vec_distance_cosine` per rowid, Qdrant and Milvus by
  id), and a filtered search with `limit = len(uuids)` on turbovec,
  which holds only codes. A similarity may come from a quantized stored
  vector and may differ from one computed on a fresh embedding; the
  docstring says so. A uuid with no record is absent from the result.
- This is the one place this document contradicts an open PR; if the
  reviewer of #1598 prefers an allowlist parameter on `query` instead,
  the plan below works with either, and the design chose the scoring
  call because the probe's result is bounded and small.

## Datetimes

Where a backend has no datetime type (sqlite-vec metadata columns, the
engine-backed store's records table, and S3 Vectors when it arrives),
a datetime property, system or user, is stored as an integer of
microseconds since the epoch, the same precision the SQL stores keep,
so a `since` or `until` bound evaluates identically in every store. A
bound is normalized to UTC before it is compared, as the segment store
already does.

## Clients

Every backend client params model has a required `request_timeout`,
passed to the client at construction (`AsyncQdrantClient(timeout=...)`,
the Milvus client's timeout). The design's tombstone retention rests
on every remote write being bounded by it, so it is required, not
defaulted.

## Qdrant collection options

From `a0753d3c`: `hnsw_config`, `optimizers_config` and
`quantization_config` on the Qdrant store's params as plain mappings,
validated against the `qdrant_client.models` types when the store is
built so the client stays an optional import, applied to data
collections only. In payload-partitioned mode `hnsw_config.m` must be 0
or unset, since a multi-tenant collection disables the global graph in
favor of per-tenant payload indexing, and `payload_m` is the knob; the
store rejects any other `m`.

## What EventMemory does with it

This is the consumer side of the contract and belongs with it; the
author of #1597 is the natural owner.

- `EventMemoryParams` gains `filter: FilterOptions` with
  `selective_limit: int` and `max_overfetch: int`.
- `split_declared(expr, declared) -> tuple[FilterExpr | None,
  FilterExpr | None]` (`filters_and_properties.md`): the part of a
  conjunction naming declared keys only, and the rest; a disjunction
  or negation that mixes the two is undeclared as a whole. Applied to
  the caller's `property_filter` after `system_predicates` has been
  conjoined, against the collection's declared schema.
- `SegmentStorePartition.find_segments(*, since, until, session_ids,
  source_ids, block_kinds, property_filter, limit) -> list[UUID]`:
  segments matching the system filters and a property filter, up to
  `limit + 1`, so the caller can tell selective from broad. Over the
  ordering index plus the JSON properties column.
- The plan in `query`. Selective: `find_segments` with the undeclared
  part up to `selective_limit`; if it fits, take those segments'
  derivative uuids (`get_derivative_uuids_by_segment_uuids`), score
  them with `get_cosine_similarity`, drop those below
  `min_cosine_similarity`, keep the best `limit`. Broad: `query` with
  the declared part and the system predicates, `limit` widened up to
  `max_overfetch` while the segment store rejects seeds against the
  undeclared part (`get_segment_contexts` with the same filters is the
  rejection), and cut to `limit`; at the cap return what survived
  (`b17fd0a1`). With no undeclared part there is one plan, `query` with
  the whole tree.
- Every count is a maximum: a filtered search returns fewer when the
  filter admits fewer, and nothing promises exactly `limit`.

## Tests

- Per backend: an undeclared key on `upsert` and in a filter raises;
  each node in `supported_filter_nodes` evaluates during a search and
  each node outside it raises `UnsupportedFilterError`; `NotEquals`
  excludes records lacking the key and `Not(Equals)` includes them;
  `IsMissing` matches absence, never null; a datetime bound at
  microsecond precision behaves the same on sqlite-vec, the engine
  store and pgvector; `get_cosine_similarity` returns nothing for an
  unknown uuid and a score within the engine's quantization error for
  a known one.
- EventMemory: the selective plan is chosen exactly when the probe
  returns at most `selective_limit` segments; the broad plan widens no
  further than `max_overfetch` and returns what survived; a fully
  declared filter takes the one-plan path.
- Do not test ranking with a fake embedder that ties every score under
  cosine; use one whose vectors differ per text and assert the
  contract, not an exact list. Run each new store test against the
  unfixed store first.

## Not in this change

UUID keys and never-reused keys, the key registry and registry rows,
strict create, logical delete and batched purge, the check after a
remote operation, stateless handles and the removal of `open_*`, one
container per embedder with tenants as values, the SQLite stores on
shared tables and the `vec0` partition key, container provisioning by
the schema command, content-addressed names and Qdrant shard keys
going, the six surveyed remote backends, and the engines' durability
contract (#1588, #1599). The interim registry stack (#1526 to #1533,
being resliced) and #1537 cover the collection side until the lifecycle
work lands; nothing here assumes or forbids them.
