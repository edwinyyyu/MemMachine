# Vector store

Existing component, `common/vector_store/`, reworked. Derived data: an
index of derivative embeddings, rebuildable from the event store.
Queries return ids and scores, never properties.

## Constructed with

- `<Backend>VectorStore(client, registry: KeyRegistry,
  settings: VectorStoreSettings)`, where `registry` is the scoped view
  from `key_registry.md` and the client is the backend's (an
  `AsyncQdrantClient`, a Milvus client, an engine for pgvector and the
  SQLite stores, for which the registry is a table beside the data).
- Settings: `containers: Mapping[str, ContainerSettings]`, one per
  embedder id, each with `dimensions`, `metric` (cosine) and any
  backend-specific option; `indexed_properties: Mapping[str,
  PropertyType]`, one schema for every container of the store, plus
  the system fields, which are always declared; `request_timeout`,
  required.

## Types

```python
class Record(BaseModel):
    uuid: UUID
    vector: list[float]
    properties: dict[str, PropertyValue]     # declared keys only

class QueryMatch(BaseModel):
    uuid: UUID
    score: float                             # cosine similarity

class QueryResult(BaseModel):
    matches: list[QueryMatch]                # descending score, at most `limit`
```

## API

```python
class VectorStore(ABC):
    async def provision_containers(self) -> None          # schema command only
    async def create_collection(self, key: UUID, container: str) -> None
    async def delete_collection(self, key: UUID) -> None
    async def reclaim_collection(self, key: UUID) -> Progress
    def for_container(self, container: str) -> VectorStore   # scoped view

    async def upsert(self, key: UUID, records: Iterable[Record]) -> None
    async def delete(self, key: UUID, uuids: Iterable[UUID]) -> None
    async def query(self, key: UUID, vectors: Iterable[Sequence[float]], *,
                    limit: int, min_score: float | None,
                    filter: FilterExpr | None,
                    allowed_uuids: Iterable[UUID] | None) -> list[QueryResult]
    async def get_cosine_similarity(self, key: UUID, vector: Sequence[float],
                                    uuids: Iterable[UUID]) -> dict[UUID, float]

    @property
    def concurrency_scope(self) -> ConcurrencyScope
```

Semantics:

- `provision_containers`: idempotently create every container the
  settings declare, with the store's one indexed-property schema. Run by
  `memmachine schema upgrade`, never by a request.
- `create_collection(key, container)`: strict. Where a tenant is a value
  inside the container: one registry row, `live`, address `{container}`.
  Where a tenant is a native object (Chroma collection, Weaviate
  tenant): row as `creating`, create the object, set `live` with its
  address (`{container, collection_id}`); a caller resuming a
  `creating` row creates the object if absent and sets `live`.
- `upsert`, `delete`: a record carrying an undeclared property key
  raises `UndeclaredPropertyKeyError` before anything is sent. Read the
  registry row (not `live`: `KeyNotLiveError`); perform the remote
  write, acknowledged as applied (Qdrant `wait=True`; Milvus strong
  consistency); read the row again and raise `KeyNotLiveError` if it is
  no longer `live`.
- `query`: `filter` names declared keys only and raises
  `UndeclaredPropertyKeyError` otherwise; evaluated during the search
  where the backend can. `allowed_uuids` restricts the search to those
  records. Read the row, query, read the row again. Returns at most
  `limit` matches per vector, fewer when the filter admits fewer.
- `get_cosine_similarity`: score given records against a vector,
  fenced the same way; the allowlist plan's scoring step.
- `delete_collection`: `set_state(key, DROPPING)`; O(1); idempotent.
- `reclaim_collection`: with a `dropping` row, delete records under the
  key in bounded steps (filter delete; a filtered-query loop and keyed
  delete on S3 Vectors; `delete_collection` on Chroma; remove the
  tenant on Weaviate); `MORE` while records remain; remove the row when
  none do and return `DONE`. With no row, delete by key in every
  container the store has; `DONE` when nothing is found. `reclaim` on
  a `live` row raises; the tenant service never calls it on one.
- `for_container(container)`: a view that raises `KeyNotLiveError` for
  any key whose registry row names another container, on every
  operation, and delegates otherwise. `EpisodicMemoryManager` builds
  each `EpisodicMemory` with the view for its embedder's container.

## Backends

| Backend | Tenant inside the container | Address | Rejects a dead key itself | Reclaim |
| --- | --- | --- | --- | --- |
| Qdrant | payload value | container | no | filter delete |
| Milvus | partition-key value | container, loaded once | no | filter delete |
| pgvector | column value | table | yes, in-statement | keyed delete |
| Pinecone | namespace parameter | index host | no | delete all in namespace |
| S3 Vectors | filterable metadata value | bucket, index | no | filtered query, keyed delete, repeat |
| Weaviate | native tenant | collection, tenant name | yes | remove tenant |
| Chroma | collection per tenant | collection UUID | yes | delete collection |
| sqlite-vec | partition-key value in one `vec0` table per container | table | yes, in-statement | keyed delete |
| usearch | shared records table, one index file per tenant | table, file path | yes, in-statement | keyed delete, unlink |

## Concurrency scope

The minimum of the backend's and the registry's: networked backends
`cluster`; local-mode Qdrant and Milvus `process`; sqlite-vec `host`;
the usearch store `process`.

## Changes required

- The registry leaves the backend (`qdrant_vector_store.py:609`,
  `milvus_vector_store.py:477`, the `_CollectionRow` tables in both
  SQLite stores) for the key registry, or, for pgvector and the SQLite
  stores, a table beside the data.
- `VectorStoreCollection` (`vector_store.py:22`) and `open_collection`,
  `open_or_create_collection`, `close_collection` (`:205`, `:235`,
  `:254`) go; operations take the key. `VectorStoreCollectionConfig`
  and the `config` parameter of `create_collection` (`:175`) go
  (#1573); the container's shape comes from `containers` settings.
- Keys are `UUID`; `validate_identifier` (`utils.py:31`) applies to
  property keys and container names only.
- Content-addressed native names (`_build_native_collection_name`,
  `qdrant_vector_store.py:619`) go; a container is named by its
  embedder id and provisioned by the schema command (#1572).
- Qdrant's shard key per collection and `_name_locks` go (#1564);
  payload partitioning by the key's hex is the one mode.
- `query` returns ids and scores only; `get` and `return_vector` go;
  `get_cosine_similarity` is added (reference branch, commit 2d5dc2b5).
- Undeclared property keys are rejected on write and query.
- Post-operation registry checks replace the absent fence (#1537,
  #1563); `reclaim_collection` and container retirement are added
  (#1565).
- The sqlite-vec store: one `vec0` table per container with the key as
  partition key, `chunk_size` a setting, instead of a table per
  collection (`sqlite_vec_vector_store.py:692`); the usearch store: a
  shared records table plus one index file per key, its `SQLiteVectorStore`
  per-collection tables (`sqlite_vector_store.py:1061`) go.
- Every client is constructed with `request_timeout`.
