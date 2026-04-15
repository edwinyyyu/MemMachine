# Split SemanticStorage into SemanticFeatureStore + VectorStoreCollection

## Context

`SemanticStorage` (ABC) currently conflates relational operations (features, citations, history tracking) with vector operations (embeddings, similarity search) in a single interface. This coupling prevents swapping vector backends independently and forces every implementation to handle both concerns.

The goal is to split it into two APIs:
1. **`SemanticFeatureStore`** (new ABC) — all relational/SQL operations
2. **`VectorStoreCollection`** (existing ABC from `common/vector_store/`) — embeddings and similarity search

This follows the pattern established by EventMemory's split of `SegmentStorePartition` + `VectorStoreCollection`.

Only define the ABCs. No implementation.

## Key Design Decisions

### ID linking: `vector_uuid` field on SemanticFeatureStore

Each feature gets a `vector_uuid: UUID` that identifies its corresponding `Record` in `VectorStoreCollection`. This is analogous to EventMemory's `DerivativeLinkRow` mapping `derivative_uuid → segment_uuid`.

- `FeatureIdT` stays as `str` (preserves existing callers, pgvector int IDs, Neo4j elementIds)
- `vector_uuid` is generated client-side (`uuid4()`) by the orchestrator before writing to either store
- The mapping is 1:1 (each feature has exactly one embedding)
- `SemanticFeatureStore` stores `vector_uuid` alongside the relational data
- `VectorStoreCollection` uses `vector_uuid` as `Record.uuid`

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
- The orchestrator (`SemanticService`) combines them: query vector store → get UUIDs + scores → enrich from feature store

## New ABC: SemanticFeatureStore

File: `semantic_memory/storage/feature_store.py`

### Methods

**Lifecycle** (unchanged from SemanticStorage):
- `startup()`
- `cleanup()`
- `delete_all()`
- `reset_set_ids(set_ids: Sequence[SetIdT])`

**Feature CRUD** (no `embedding` parameter, adds `vector_uuid`):
- `add_feature(*, vector_uuid: UUID, set_id, category_name, feature, value, tag, metadata?) → FeatureIdT`
- `update_feature(feature_id, *, set_id?, category_name?, feature?, value?, tag?, metadata?)` — no embedding; caller updates vector store separately if value changed
- `get_feature(feature_id, load_citations?) → SemanticFeature | None`
- `get_feature_set(*, filter_expr?, page_size?, page_num?, tag_threshold?, load_citations?) → AsyncIterator[SemanticFeature]` — no `vector_search_opts`

**New enrichment methods** (for post-vector-search loading):
- `get_features_by_vector_uuids(vector_uuids: Sequence[UUID], load_citations?) → Mapping[UUID, SemanticFeature]`

**New lookup methods** (for coordinating deletes/updates with vector store):
- `get_vector_uuid(feature_id) → UUID | None`
- `get_vector_uuids(feature_ids: Sequence[FeatureIdT]) → Mapping[FeatureIdT, UUID]`

**Delete** (returns vector UUIDs so caller can clean up vector store):
- `delete_features(feature_ids: Sequence[FeatureIdT]) → Sequence[UUID]` — returns deleted vector_uuids
- `delete_feature_set(*, filter_expr?) → Sequence[UUID]` — returns deleted vector_uuids

**Citations** (unchanged):
- `add_citations(feature_id, history_ids: Sequence[EpisodeIdT])`

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
| Add feature | `add_feature(vector_uuid, ...)` | `upsert(Record(uuid=vector_uuid, vector, properties))` |
| Update feature | `update_feature(...)` | `upsert(...)` if embedding/properties changed |
| Get by ID | `get_feature(...)` | not needed |
| Search by similarity | `get_features_by_vector_uuids(...)` (enrichment) | `query(query_vectors, property_filter, limit)` |
| List/filter features | `get_feature_set(filter_expr, ...)` | not needed |
| Delete by IDs | `delete_features(...)` → returns UUIDs | `delete(record_uuids)` |
| Delete by filter | `delete_feature_set(...)` → returns UUIDs | `delete(record_uuids)` |
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
Record(uuid=vector_uuid, vector=embedding.tolist(), properties={"set_id": ..., "category": ..., "tag": ...})
```

Search translates the `set_id`/`category`/`tag` parts of `FilterExpr` into `VectorStoreCollection.query(property_filter=...)`.

## Backward Compatibility

**Assessment: backward compatible via composite adapter.**

The old `SemanticStorage` ABC can be implemented as a `CompositeSemanticStorage` that wraps `SemanticFeatureStore` + `VectorStoreCollection`:

- `add_feature(embedding=...)` → generates `vector_uuid`, calls both stores
- `update_feature(embedding=...)` → looks up `vector_uuid`, calls both stores
- `get_feature_set(vector_search_opts=...)` → queries vector store, enriches from feature store
- `delete_features(ids)` → looks up vector UUIDs, deletes from both stores
- All history/citation methods → forwarded to feature store unchanged

This means:
- **Existing callers** (`SemanticService`, `IngestionService`) can continue using `SemanticStorage` unchanged during migration
- **Existing implementations** (`SqlAlchemyPgVectorSemanticStorage`, `Neo4jSemanticStorage`) remain valid `SemanticStorage` implementors
- **New implementations** only need to implement `SemanticFeatureStore`, paired with any `VectorStoreCollection` backend

**Breaking aspects** (in the new `SemanticFeatureStore` ABC only, not in old `SemanticStorage`):
- `add_feature` requires `vector_uuid: UUID` parameter (not present on old interface)
- `delete_features` / `delete_feature_set` return `Sequence[UUID]` (old returned `None`)
- `get_feature_set` has no `vector_search_opts` parameter

These are intentional interface changes for the new ABC. The old `SemanticStorage` ABC is preserved unchanged — backward compatibility is maintained by keeping it as a supported interface with the composite implementation.

## Files to create/modify

| File | Action |
|------|--------|
| `semantic_memory/storage/feature_store.py` | **Create** — new `SemanticFeatureStore` ABC |
| `semantic_memory/storage/__init__.py` | Update exports if needed |
| `semantic_memory/storage/storage_base.py` | No changes (old ABC kept as-is) |
| `common/vector_store/vector_store.py` | No changes (used as-is) |

## Verification

Since this is API-only (no implementation), verification is:
1. The new ABC imports cleanly and can be subclassed
2. All current `SemanticStorage` methods are accounted for in either `SemanticFeatureStore` or `VectorStoreCollection`
3. Type checking passes (`uv run mypy` or `uv run pyright` on the new file)
4. The existing `SemanticStorage` ABC is untouched — no regressions
