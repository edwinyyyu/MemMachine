# Handoff: SemanticStorage split

This file exists to seed a fresh Claude Code session with context from prior work. Read it first, then work from `PLAN.md` and the existing `feature_store.py`.

## What this work is

Split the monolithic `SemanticStorage` ABC into two separate storage interfaces:

1. **`SemanticFeatureStore`** — new ABC, all relational operations (features, citations, history tracking)
2. **`VectorStoreCollection`** — existing ABC at `common/vector_store/vector_store.py`, handles embeddings and similarity search

Reason: the old interface conflates vector and relational concerns, preventing backends from being swapped independently and forcing every implementation to do both.

Reference pattern: **EventMemory** at `packages/server/src/memmachine_server/episodic_memory/event_memory/` already splits storage into `SegmentStorePartition` + `VectorStoreCollection`. Study this before implementing — it has the best example of dual-store orchestration, UUID-based linking, and write ordering.

## Current state

**Done:**
- `SemanticFeatureStore` ABC written and committed — `packages/server/src/memmachine_server/semantic_memory/storage/feature_store.py`
- **All autoincrement-int IDs in semantic memory are now UUIDs.** Aliases `FeatureIdT`, `CategoryIdT`, `TagIdT` **deleted repo-wide**; type hints use `uuid.UUID` directly. `SetTypeEntry.id` is `UUID | None`.
  - `semantic_model.py`, `api/spec.py`: aliases removed; `feature_id`, `category_id`, `set_type_id`, `tag_id` fields are all `UUID` at the API boundary. `SemanticCategory.id`, `SemanticCategoryEntry.id`, `SemanticCategoryTemplateEntry.id`, `SemanticSetTypeEntry.id` are all `UUID | None` (or `UUID`).
  - `config_store.py` Protocol and `config_store_sqlalchemy.py` SQLAlchemy impl: `Category.id`, `Tag.id`, `SetType.id` columns are `Uuid` (client-generated via `uuid4` default); FK columns `SetIdSetType.set_type_id`, `Category.set_type_id`, `Tag.category_id` are also `Uuid`. All `int(...)` casts on IDs dropped. The caching wrapper mirrors.
  - `semantic_llm.py`: `SemanticConsolidateMemoryRes.keep_memories: list[UUID] | None`; consolidation format JSON-serializes via `str(uuid)` so the LLM still sees opaque string refs.
  - `semantic_memory.py`, `semantic_ingestion.py`, `semantic_session_manager.py`, `main/memmachine.py`, `server/api_v2/router.py`, `server/api_v2/mcp.py`: all signatures updated.
  - Client library: `packages/client/src/memmachine_client/memory.py` signatures updated to accept/return `UUID`.
  - Test suites updated and **all 1359 tests pass** (server + client unit tests).
- Plan document committed — `PLAN.md` in the repo root

**Not done:**
- No `SemanticFeatureStore` implementations yet (pgvector, neo4j, in-memory for tests) — the old `SemanticStorage`-based implementations still run, now with `Uuid` columns in the config store but unchanged embedding/feature semantics.
- `SemanticService` and `IngestionService` still take the old `SemanticStorage`
- `SemanticResourceManager` still instantiates the old implementations
- Alembic migrations:
  - Feature storage: migration to switch `feature.id` PK from autoincrement int to UUID and drop the `embedding` column is not written
  - Config store: migration to switch `Category.id`, `Tag.id`, `SetType.id` (and all FKs) from Integer to Uuid is not written. Current schema uses `BaseSemanticConfigStore.metadata.create_all` in tests which generates the new schema from scratch. For existing databases a migration is required.
- Old `SemanticStorage` ABC (`storage_base.py`) and its two implementations still exist (and remain runtime-broken in the int→UUID cast paths after the mechanical `FeatureIdT → UUID` rename)

## Key decisions made in the prior session

### 1. ID linking: features ARE UUIDs — no separate `vector_uuid`

A feature's primary key is a plain `uuid.UUID`. The same UUID is used as `Record.uuid` in `VectorStoreCollection`. **There is no separate `vector_uuid` field, no link table, and no two-id mapping** — one UUID identifies the feature row and its embedding record.

The orchestrator generates the UUID via `uuid4()` before either write, and passes it to `SemanticFeatureStore.add_feature(feature_id=...)` and `VectorStoreCollection.upsert(Record(uuid=..., ...))`.

The `FeatureIdT` type alias was **deleted entirely** in favor of `uuid.UUID`. This was reconsidered in this session and accepted as a clean break — the prior session's rejection (preserve relational ID space) was overruled because (a) we're already rewriting the pgvector schema and (b) there are barely any users.

This is much simpler than the EventMemory `DerivativeLinkRow` pattern because semantic features are 1:1 with embeddings, not 1:N.

### 2. Vector store properties schema is minimal

```python
properties_schema = {
    "set_id": str,    # always filtered during search
    "category": str,  # commonly filtered
    "tag": str,       # commonly filtered
}
```

Fields like `feature`, `value`, and arbitrary `metadata` stay SQL-only. They're rarely filtered during vector search and would bloat vector records. If a caller needs to filter by those fields during similarity search, they must either (a) post-filter after enrichment from `SemanticFeatureStore`, or (b) accept that the filter doesn't prune the candidate set.

### 3. No composite adapter — direct two-store orchestration

`SemanticService.Params` and `IngestionService.Params` take both stores as separate fields:

```python
class Params(BaseModel):
    feature_store: InstanceOf[SemanticFeatureStore]
    vector_store_collection: InstanceOf[VectorStoreCollection]
    episode_storage: InstanceOf[EpisodeStorage]
    ...
```

**Do not build a `CompositeSemanticStorage` adapter.** The user explicitly rejected this. Reasons:
- `SemanticService` already owns the embedder and the decision of when to call `search_embed()`. A facade hides orchestration that should be explicit.
- `get_feature_set(vector_search_opts=...)` in the old API hides the two-phase search. Direct access lets the caller see scores, control over-fetch factors, and short-circuit on empty results.
- `IngestionService._apply_commands` can batch vector upserts into a single call with direct access — impossible through a one-feature-at-a-time facade.
- Error handling, retry, and write ordering belong in `SemanticService`, not an adapter.

The old `SemanticStorage` ABC is slated for **deletion**, not preservation. This is a replacement, not a parallel interface.

### 4. Cluster engine is orthogonal

Commit `0ea3eab1` (#1240) added a cluster engine: `cluster_manager.py`, `cluster_splitter.py`, `cluster_store/`. It has its own `ClusterStateStorage` Protocol for per-set cluster state persistence.

**Do not touch it as part of this refactor.** Reasons:
- Centroids are stored as JSON arrays, not pgvector/VectorStoreCollection records
- Similarity is computed in-memory by `ClusterManager._select_cluster` via numpy
- Storage is pure key-value (load/save/delete per set_id), no vector search at the storage level
- Not yet wired into `semantic_memory.py` or `semantic_ingestion.py` — standalone infrastructure awaiting future integration

### 5. AsyncIterator / Sequence / Mapping conventions

Commit `0ea3eab1` modernized storage interface types. `SemanticStorage` now uses `AsyncIterator` for streaming methods (`get_feature_set`, `get_history_messages`, `get_history_set_ids`, `get_set_ids_starts_with`), `Sequence` for inputs, `Mapping` for metadata. `SemanticFeatureStore` already follows these conventions.

### 6. `get_feature_set` splits into two paths

The old `get_feature_set(vector_search_opts=...)` combined vector search with relational filtering. After the split:

- **Pure relational query** (no vector search): `SemanticFeatureStore.get_feature_set(filter_expr, ...)` — returns `AsyncIterator[SemanticFeature]`
- **Vector similarity search**: `VectorStoreCollection.query(query_vectors, property_filter, limit)` → get UUIDs + scores → `SemanticFeatureStore.get_features(uuids)` for enrichment

The orchestrator (`SemanticService.search`) stitches these together.

### 7. ABC method shape after the simplification

Compared to the original draft of `SemanticFeatureStore`, three vector-uuid coordination methods were removed and one was renamed:

- `add_feature(*, feature_id: UUID, ...)` — caller supplies the UUID; returns `None` (was: returned `FeatureIdT`)
- `get_features(uuids, load_citations?) → Mapping[UUID, SemanticFeature]` — was `get_features_by_vector_uuids`; now just bulk-by-id
- `get_vector_uuid` / `get_vector_uuids` — **deleted** (identity lookup, no longer needed)
- `delete_features(uuids) → None` — caller already has the UUIDs (was: returned `Sequence[UUID]`)
- `delete_feature_set(filter_expr) → Sequence[UUID]` — still returns IDs since the caller didn't know them

### 8. Config store IDs are also UUID

`Category.id`, `Tag.id`, `SetType.id` (in `semantic_memory/config_store/config_store_sqlalchemy.py`) are all `Uuid` columns with `default=uuid4`. The API exposes them as `UUID`. They were autoincrement ints previously and user-verified as opaque handles (never seen by the LLM, server-minted, never client-typed). The `CategoryIdT` and `TagIdT` aliases were **deleted entirely** along with `FeatureIdT`.

**Note:** The `TagIdT` alias was a misnomer — `Feature.tag_id` in the feature storage still stores the *tag name string* (the LLM-facing label). Only `Tag.id` in the config store's `semantic_config_tag` table is a UUID. The feature row's `tag_id` column remains `String` with the tag name.

**Note:** `SemanticSetEntry.id` (a set_id) stays `str` — set_ids are human-readable composites like `"mem_user_set_abc"` and are LLM/user-meaningful.

## Critical files

**Read first:**
- `PLAN.md` — authoritative plan document (committed)
- `packages/server/src/memmachine_server/semantic_memory/storage/feature_store.py` — the new ABC

**Reference / study:**
- `packages/server/src/memmachine_server/episodic_memory/event_memory/event_memory.py` — dual-store orchestration pattern
- `packages/server/src/memmachine_server/episodic_memory/event_memory/segment_store/segment_store.py` — SegmentStore ABC design
- `packages/server/src/memmachine_server/episodic_memory/event_memory/segment_store/sqlalchemy_segment_store.py` — three-table implementation with DerivativeLinkRow
- `packages/server/src/memmachine_server/common/vector_store/vector_store.py` — VectorStoreCollection ABC
- `packages/server/src/memmachine_server/common/vector_store/data_types.py` — Record, QueryMatch, QueryResult, VectorStoreCollectionConfig

**Will need modification:**
- `packages/server/src/memmachine_server/semantic_memory/storage/sqlalchemy_pgvector_semantic.py` — rewrite as `SemanticFeatureStore` implementation: `feature.id` becomes a UUID column (no autoincrement int), drop the `embedding` column, drop the citation table FK references to int IDs
- `packages/server/src/memmachine_server/semantic_memory/storage/neo4j_semantic_storage.py` — rewrite or replace
- `packages/server/src/memmachine_server/semantic_memory/semantic_memory.py` — `SemanticService.Params` takes both stores; `search()` becomes two-phase
- `packages/server/src/memmachine_server/semantic_memory/semantic_ingestion.py` — `IngestionService.Params` takes both stores; `_apply_commands` batches vector upserts
- `packages/server/src/memmachine_server/common/resource_manager/semantic_manager.py` — wire both stores from config
- `packages/server/server_tests/memmachine_server/semantic_memory/storage/in_memory_semantic_storage.py` — replace with `InMemorySemanticFeatureStore`
- `packages/server/server_tests/memmachine_server/conftest.py` — update fixtures
- Alembic migration in `semantic_memory/storage/alembic_pg/versions/` — switch `feature.id` from autoincrement int to UUID, backfill via `gen_random_uuid()`, update `citations.feature_id` FK type, drop `embedding` column
- Alembic migration for config store (`semantic_config_category.id`, `semantic_config_tag.id`, `set_type.id` and their FKs) — Integer → Uuid. SQLAlchemy model already points at `Uuid`; existing databases need a migration.

**Will need deletion:**
- `packages/server/src/memmachine_server/semantic_memory/storage/storage_base.py` (old `SemanticStorage` ABC) — after all callers migrated

## Known callers of old `SemanticStorage` (must be updated)

Roughly ~15 call sites in `semantic_memory.py` and ~8 in `semantic_ingestion.py`. The resource manager instantiates the old implementations in `common/resource_manager/semantic_manager.py`.

Test fixtures parametrize across pgvector, neo4j, and in-memory via `conftest.py`. These all need to switch to the new ABC.

## Suggested next-step ordering

1. **Rewrite the pgvector backend** as a `SemanticFeatureStore` implementation: change `feature.id` from autoincrement int to UUID PK, change `citations.feature_id` FK type, drop the `embedding` column, write the Alembic migration. Add a separate pgvector-backed `VectorStoreCollection` if one doesn't already exist; otherwise reuse the common one.
2. **Write `InMemorySemanticFeatureStore`** for tests — minimal, dict-backed.
3. **Update `SemanticService.Params`** to take `feature_store` + `vector_store_collection`. Rewrite `search()` as two-phase (vector query → enrichment via `get_features`).
4. **Update `IngestionService`** — batch vector upserts in `_apply_commands`, use direct access to both stores. The orchestrator generates `uuid4()` per feature before writing.
5. **Update `SemanticResourceManager`** — instantiate both stores from config, wire into service.
6. **Update test fixtures** and run the existing test suite.
7. **Rewrite or delete Neo4j implementation** — Neo4j currently stores embeddings as node properties with a per-set vector index. It could either (a) implement `SemanticFeatureStore` only with a separate `VectorStoreCollection` backend, or (b) be dropped if pgvector is the primary target.
8. **Delete `storage_base.py`** and verify nothing imports `SemanticStorage`.

## Useful shell commands

```bash
# Confirm FeatureIdT/CategoryIdT/TagIdT are gone everywhere (only docs should mention them)
grep -rn "FeatureIdT\|CategoryIdT\|TagIdT" packages/

# List all references to the old ABC
grep -rn "SemanticStorage" packages/server/

# Run the semantic memory test suite
uv run pytest packages/server/server_tests/memmachine_server/semantic_memory/

# AST-verify method coverage between old and new ABCs
uv run python -c "
import ast
def methods(path, cls):
    with open(path) as f: t = ast.parse(f.read())
    for n in ast.walk(t):
        if isinstance(n, ast.ClassDef) and n.name == cls:
            return {m.name for m in n.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}
old = methods('packages/server/src/memmachine_server/semantic_memory/storage/storage_base.py', 'SemanticStorage')
new = methods('packages/server/src/memmachine_server/semantic_memory/storage/feature_store.py', 'SemanticFeatureStore')
print('Missing from new:', old - new)
print('Added in new:', new - old)
"
```

## Things the prior session explored but ultimately didn't use

- **Composite adapter** — considered and rejected (see decision #3)
- **Keeping `FeatureIdT` as a type alias** — initially kept for grep-ability; user requested removal in favor of `uuid.UUID` directly
- **Separate `vector_uuid` field on the feature row** — initially planned, then collapsed: the feature's PK *is* the vector record UUID
- **One `VectorStoreCollection` per set_id** — rejected; use one per embedding dimension instead (grouped by embedder), with `set_id` as a property filter
- **Moving cluster centroids to `VectorStoreCollection`** — noted as a possible future optimization but explicitly out of scope
