# Split SemanticStorage into SemanticFeatureStore + VectorStoreCollection

## Context

`SemanticStorage` (ABC) currently conflates relational operations (features, citations, history tracking) with vector operations (embeddings, similarity search) in a single interface. This coupling prevents swapping vector backends independently and forces every implementation to handle both concerns.

The goal is to split it into two APIs:
1. **`SemanticFeatureStore`** (new ABC) — all relational/SQL operations
2. **`VectorStoreCollection`** (existing ABC from `common/vector_store/`) — embeddings and similarity search

This follows the pattern established by EventMemory's split of `SegmentStorePartition` + `VectorStoreCollection`.

Only define the ABCs. No implementation.

## Related semantic memory updates

Since this plan was first drafted, two relevant changes landed in semantic memory (commit `0ea3eab1`, #1240):

1. **Storage interface type modernization.** `SemanticStorage` now uses `Sequence`/`Mapping` for inputs and `AsyncIterator` for streaming methods (`get_feature_set`, `get_history_messages`, `get_history_set_ids`, `get_set_ids_starts_with`). The new `delete_history_set` method was also added. `SemanticFeatureStore` follows these conventions.

2. **Cluster engine added (`cluster_manager.py`, `cluster_splitter.py`, `cluster_store/`).** A new `ClusterStateStorage` Protocol persists per-set clustering state for semantic message grouping. This is **orthogonal to this plan** — see the [Out of scope](#out-of-scope-clusterstatestorage) section below.

## Key Design Decisions

### IDs: `FeatureIdT` is gone — features use `UUID` directly

A feature's primary key is a plain `uuid.UUID`. The same UUID is used as `Record.uuid` in `VectorStoreCollection` — there is no separate `vector_uuid` field, no link table, no two-id mapping. One UUID identifies the feature row *and* its embedding record.

- The orchestrator generates the UUID via `uuid4()` before either write.
- It passes the same UUID to `SemanticFeatureStore.add_feature(feature_id=...)` and to `VectorStoreCollection.upsert(Record(uuid=..., ...))`.
- The `FeatureIdT` type alias has been **deleted**. Type hints use `uuid.UUID` directly.
- The pgvector autoincrement int PK is replaced by a UUID column.
- `SemanticFeature.Metadata.id` is `UUID | None`. The API DTO at `api/spec.py` mirrors this — Pydantic auto-parses incoming JSON strings into UUIDs.

This is a **breaking API change** for `feature_id` in `api/spec.py` (`AddFeatureResponse`, `GetFeatureSpec`, `UpdateFeatureSpec`, `ListMemoriesSpec.semantic_memory_uids`). Acceptable because there are barely any users yet.

#### LLM compatibility

The LLM consumes feature IDs in two places (`semantic_llm.py`):
- Consolidation prompt input: `_features_to_consolidation_format` serializes `metadata.id` as a string for `json.dumps` — UUIDs work fine via `str(uuid)`.
- Consolidation response: `SemanticConsolidateMemoryRes.keep_memories: list[UUID] | None`. Pydantic parses the string-formatted UUIDs the LLM returns back into `UUID` objects automatically.

The LLM only uses these IDs as opaque references (copying them back), not as meaningful labels, so the change is invisible at the prompt level.

### Vector store properties schema

Three fields duplicated into vector store properties for filtering during similarity search:

```python
properties_schema = {
    "set_id": str,     # always filtered on during search
    "category": str,   # commonly filtered
    "tag": str,        # commonly filtered
}
```

Fields like `feature`, `value`, and user `metadata` stay SQL-only — rarely filtered during vector search and would bloat vector records.

### Where `get_feature_set` with vector search goes

The current `get_feature_set(vector_search_opts=...)` combines vector search + relational filtering in one call. After the split:

- `SemanticFeatureStore.get_feature_set()` — relational queries only (no `vector_search_opts` parameter)
- `VectorStoreCollection.query()` — vector similarity search with property filtering
- The orchestrator (`SemanticService`) combines them: query vector store → get UUIDs + scores → enrich from feature store via `get_features(uuids)`

## New ABC: SemanticFeatureStore

File: `semantic_memory/storage/feature_store.py`

### Methods

**Lifecycle** (unchanged from SemanticStorage):
- `startup()`
- `cleanup()`
- `delete_all()`
- `reset_set_ids(set_ids: Sequence[SetIdT])`

**Feature CRUD** (no `embedding` parameter; caller supplies the UUID):
- `add_feature(*, feature_id: UUID, set_id, category_name, feature, value, tag, metadata?) → None`
- `update_feature(feature_id: UUID, *, set_id?, category_name?, feature?, value?, tag?, metadata?)` — no embedding; caller updates vector store separately if value/properties changed
- `get_feature(feature_id: UUID, load_citations?) → SemanticFeature | None`
- `get_features(feature_ids: Sequence[UUID], load_citations?) → Mapping[UUID, SemanticFeature]` — bulk enrichment after vector search
- `get_feature_set(*, filter_expr?, page_size?, page_num?, tag_threshold?, load_citations?) → AsyncIterator[SemanticFeature]` — no `vector_search_opts`

**Delete:**
- `delete_features(feature_ids: Sequence[UUID]) → None` — caller already has the IDs; uses them directly to delete vector records
- `delete_feature_set(*, filter_expr?) → Sequence[UUID]` — returns deleted IDs so caller can clean up the vector store

**Citations** (unchanged):
- `add_citations(feature_id: UUID, history_ids: Sequence[EpisodeIdT])`

**History tracking** (unchanged, all purely relational):
- `add_history_to_set(set_id, history_id)`
- `get_history_messages(*, set_ids?, limit?, is_ingested?) → AsyncIterator[EpisodeIdT]`
- `get_history_messages_count(*, set_ids?, is_ingested?) → int`
- `mark_messages_ingested(*, set_id, history_ids)`
- `delete_history(history_ids)`
- `delete_history_set(set_ids)`
- `get_history_set_ids(*, min_uningested?, older_than?) → AsyncIterator[SetIdT]`
- `purge_ingested_rows(set_ids) → int`

**Set discovery** (unchanged):
- `get_set_ids_starts_with(prefix) → AsyncIterator[SetIdT]`

### Method-to-store mapping

| Operation | SemanticFeatureStore | VectorStoreCollection |
|-----------|---------------------|----------------------|
| Add feature | `add_feature(feature_id=uuid, ...)` | `upsert(Record(uuid=uuid, vector, properties))` |
| Update feature | `update_feature(...)` | `upsert(...)` if embedding/properties changed |
| Get by ID | `get_feature(...)` | not needed |
| Search by similarity | `get_features(uuids)` (enrichment) | `query(query_vectors, property_filter, limit)` |
| List/filter features | `get_feature_set(filter_expr, ...)` | not needed |
| Delete by IDs | `delete_features(uuids)` | `delete(uuids)` — same UUIDs |
| Delete by filter | `delete_feature_set(...)` → returns UUIDs | `delete(returned_uuids)` |
| All history/citation ops | yes | not involved |

## How VectorStoreCollection is used (no changes to its ABC)

The existing `VectorStoreCollection` ABC is used as-is. The orchestrator configures it with:

```python
VectorStoreCollectionConfig(
    vector_dimensions=embedder.dimensions,
    similarity_metric=SimilarityMetric.COSINE,
    properties_schema={"set_id": str, "category": str, "tag": str},
)
```

Records written during `add_feature`:
```python
Record(uuid=feature_id, vector=embedding.tolist(), properties={"set_id": ..., "category": ..., "tag": ...})
```

Search translates the `set_id`/`category`/`tag` parts of `FilterExpr` into `VectorStoreCollection.query(property_filter=...)`.

## Orchestration: `SemanticService` takes both stores directly

When implementation begins, `SemanticService.Params` and `IngestionService.Params` should accept the two stores as separate fields rather than hiding them behind a composite adapter:

```python
class Params(BaseModel):
    feature_store: InstanceOf[SemanticFeatureStore]
    vector_store_collection: InstanceOf[VectorStoreCollection]
    episode_storage: InstanceOf[EpisodeStorage]
    ...
```

Reasons to avoid a composite adapter as the end state:

- `SemanticService` already owns the embedder and decides when to call `search_embed()`. Hiding the two stores behind a `SemanticStorage`-shaped facade re-routes that orchestration through an interface that obscures which store failed, which one was slow, and what the intermediate scores were.
- `get_feature_set(vector_search_opts=...)` in the old interface hides the two-phase search. Direct access lets the caller access scores, decide on over-fetch factors, and short-circuit if the vector store returns nothing.
- `IngestionService._apply_commands` can batch all vector upserts into a single `vector_store_collection.upsert(records=[...])` call — impossible through a one-at-a-time `add_feature` facade.
- Error handling, retry logic, and write ordering should live in `SemanticService`, not be pushed into an adapter.

A composite adapter is only worth building as a short-lived migration aid if old and new code paths need to run side by side. For a clean end state, skip it.

## Backward Compatibility

**Assessment: not preserved. The old `SemanticStorage` ABC will be removed.**

The split is a replacement, not a parallel interface. Breaking aspects:

- `add_feature` requires a caller-supplied `feature_id: UUID`; returns `None` instead of an id
- `delete_features` returns `None`; `delete_feature_set` returns `Sequence[UUID]` for vector cleanup
- `get_feature_set` has no `vector_search_opts` parameter — vector search moves to `VectorStoreCollection.query()`
- `SemanticService` and `IngestionService` accept two stores instead of one `SemanticStorage`
- `FeatureIdT` type alias is **deleted**; all signatures use `uuid.UUID` directly
- `api/spec.py` `feature_id` fields and `semantic_memory_uids` are now `UUID` (was `str`)
- The `CompositeSemanticStorage` adapter **is not built** (see above)

Existing implementations (`SqlAlchemyPgVectorSemanticStorage`, `Neo4jSemanticStorage`) will be replaced by new implementations of `SemanticFeatureStore` paired with corresponding `VectorStoreCollection` backends. Data migration is required for the pgvector implementation: switch the `feature.id` PK from autoincrement int to UUID, drop the in-row `embedding` column in favor of a `VectorStoreCollection` record per feature.

## Out of scope: `ClusterStateStorage`

The cluster engine added in #1240 introduces its own storage abstraction (`ClusterStateStorage` Protocol at `semantic_memory/cluster_store/cluster_store.py`) with three methods: `get_state`, `save_state`, `delete_state`. It persists `ClusterState` objects containing cluster centroids (as `Sequence[float]`), event-to-cluster mappings, pending events, and split records.

This is **not affected by the SemanticFeatureStore split** for the following reasons:

- **No similarity search at the storage level.** `ClusterManager._select_cluster` loads the full state and iterates all clusters in Python, computing cosine similarity via numpy. The storage is pure key-value (load/save/delete per set_id).
- **Centroids are stored as JSON arrays, not vectors.** The SQLAlchemy implementation uses `JSON` column type for centroids. There's no pgvector integration, no vector index, no SQL-side distance computation.
- **Orthogonal responsibility.** Cluster state is scoped per set_id; features are scoped per set_id. They share the set_id concept but don't reference each other directly — the cluster engine groups *inbound messages*, while `SemanticFeatureStore` holds *extracted features*.
- **Not yet wired into the ingestion pipeline.** `semantic_memory.py` and `semantic_ingestion.py` do not reference the cluster engine. It exists as standalone infrastructure awaiting integration in a future change.

`ClusterStateStorage` should remain as-is. A future refactor could migrate cluster centroids into a dedicated `VectorStoreCollection` if the number of clusters per set grows large enough that in-memory similarity becomes a bottleneck — but that is a separate concern, not part of this plan.

## Out of scope (flagged): config store autoincrement IDs

Two more autoincrement int IDs surface to the API from a different storage layer (`semantic_memory/config_store/config_store_sqlalchemy.py`):

- `Category.id` (surfaced as `CategoryIdT = str`)
- `SetType.id` (surfaced as `SetTypeEntry.id: str | None`)

Both are minted by the config store, stringified for the API, and not LLM-meaningful (the LLM uses category/tag *names*). They qualify for the same UUID treatment but live outside the feature-store split. **Decision pending — extend scope or defer to a follow-up.**

## Files to create/modify

| File | Action |
|------|--------|
| `semantic_memory/storage/feature_store.py` | **Create** — new `SemanticFeatureStore` ABC (done) |
| `semantic_memory/semantic_model.py` | Drop `FeatureIdT` alias; `SemanticFeature.Metadata.id: UUID \| None` (done) |
| `common/api/spec.py` | Drop `FeatureIdT` alias; `feature_id: UUID` everywhere (done) |
| `semantic_memory/semantic_llm.py` | `keep_memories: list[UUID]`; `str()` conversion in consolidation format (done) |
| `semantic_memory/storage/__init__.py` | Export `SemanticFeatureStore` |
| `semantic_memory/storage/storage_base.py` | **Delete** when implementation phase completes (old ABC removed) |
| `semantic_memory/storage/sqlalchemy_pgvector_semantic.py` | **Delete** or rewrite as `SemanticFeatureStore` implementation |
| `semantic_memory/storage/neo4j_semantic_storage.py` | **Delete** or rewrite as `SemanticFeatureStore` implementation |
| `semantic_memory/semantic_memory.py` | Update `SemanticService.Params` to take both stores |
| `semantic_memory/semantic_ingestion.py` | Update `IngestionService.Params` to take both stores |
| `common/resource_manager/semantic_manager.py` | Wire both stores from configuration |
| `common/vector_store/vector_store.py` | No changes (used as-is) |
| `semantic_memory/cluster_store/*` | No changes (orthogonal, see Out of scope) |

## Verification

Since this is API-only (no implementation), verification is:
1. The new ABC imports cleanly and can be subclassed
2. All current `SemanticStorage` methods are accounted for in either `SemanticFeatureStore` or `VectorStoreCollection`
3. The `FeatureIdT` alias is gone from the entire codebase (`grep -r FeatureIdT` returns nothing in source)
4. The cluster engine (`ClusterStateStorage`) is confirmed orthogonal and not impacted
