# Vector store: Qdrant and Milvus layouts

Status: accepted 2026-09-29, in review in #1631. Part of [vector store
horizontal scaling](vector_store_horizontal_scaling.md).

How the Qdrant and Milvus stores lay out their collections, and why. Both keep
many logical collections, one per incarnation (see [collection
registry](vector_store_collection_registry.md)), in one native collection per
namespace and configuration.

## Shared native collections

A native collection holds every logical collection of one namespace whose
configuration is identical. Its name is derived from both:
`{namespace}__{sha256(config)}` on Qdrant and
`memmachine_{namespace}__{sha256(config)}` on Milvus, the digest taken over
the configuration's JSON. So:

- Admitting a collection is a registry row; nothing is created in the backend
  unless the configuration is new.
- A failed creation's empty native collection is the one the next creation of
  that configuration adopts.
- A purge round finds a dead incarnation's points through the namespace and
  configuration its tombstone keeps.

Measured on Milvus (2.6.24, 4 CPUs / 5 GB; 200 tenants: 1 of 50,000, 10 of
5,000, 50 of 1,000, 139 of 100; 163,900 vectors of 128 dimensions; HNSW_SQ
4-bit + FP16, ef 64, refine_k 2; one run), one shared collection against one
collection per tenant:

| | Shared (partition key + isolation) | One collection per tenant |
|---|---|---|
| Setup (create, insert, index, load) | 15 s wall, 28 CPU-s | 841 s wall, 155 CPU-s |
| Memory after load | 0.78 GB | 0.92 GB (about 0.7 MB more per collection) |
| Admitting a tenant | a registry row; nothing in Milvus | 2.0 s: create, index, load an empty collection |
| Recall@10 (50k / 5k / 1k / 100-row tenants) | 1.00 / 0.95 / 0.96 / 1.00 | 1.00 / 0.94 / 1.00 / 1.00 |
| Search p50 (same classes) | 2.1 / 1.2 / 1.1 / 1.2 ms | 3.4 / 1.3 / 1.5 / 1.4 ms |
| 4-client throughput | 2,794/s | 3,041/s |

Recall counts a hit when its score reaches the tenth exact score, within 0.001
for the FP16 refinement. The per-tenant collections' 1k-row recall of 1.00 is
an artifact: Milvus does not index a segment under 1,024 rows, so those
tenants were searched exactly. Milvus also caps a deployment's collection
count, so one collection per tenant does not reach the tenant counts
MemMachine targets.

## Qdrant

- **Vectors:** one unnamed dense vector, the configuration's dimensions and
  metric.
- **Graphs per tenant, not per collection:** HNSW `m=0`, `payload_m=16`. No
  collection-wide graph is built; each value of an indexed payload field gets
  its own graph once it has enough points in a segment, so a search filtered
  on one tenant walks that tenant's graph.
- **The incarnation:** payload field `sys-incarnation`, a keyword index with
  `is_tenant=true`, which also tells Qdrant to co-locate a tenant's points.
  Every operation of a handle filters on it.
- **Declared properties:** a payload index per property of the collection's
  schema, typed by the property's type.
- **Point ids:** the record UUIDs (see [record
  identity](vector_store_record_identity.md)).
- **Writes** pass `wait=True` (qdrant-client's default): an upsert or delete
  returns once it is applied.
- **Client:** `AsyncQdrantClient`, with `request_timeout_seconds` as its
  timeout. Custom sharding was removed in #1671.

**Filtered-search correctness (qdrant#10741).** With per-tenant graphs, a
search filtered on a tenant *and* a second condition can return nothing, or
unrelated points, when the tenant's graph entry point fails the second
condition. Measured on Qdrant 1.19.1 (12 tenants of 100k points, `m=0,
payload_m=16`, a boolean set on half the points; top 10 filtered on the tenant
and the boolean, against the same query run exactly): 26 of 60 searches
correct with the boolean indexed; 30 returned nothing while the exact search
returned 10; 4 returned 10 points sharing nothing with the exact top 10. No
result violated the filter. It first appears in 1.15.0 (0 of 60 wrong on
1.14.1; 15 on 1.15.0; 30 to 45 on later versions), and it is reachable for
tenants of a few thousand points and up at default settings. The upstream fix,
qdrant/qdrant#10741, is open. (Decided: the store is written as if that fix
has shipped; no graphless layout or exact-search workaround.)

**Purge:** one filter-delete of the whole incarnation per round (see
[purge](vector_store_purge.md)).

## Milvus

- **Fields:** `id` (VARCHAR primary key, `"{incarnation}:{record_uuid}"`),
  `record_uuid` (VARCHAR), `partition_key` (VARCHAR, the incarnation,
  `is_partition_key`), `vector` (FLOAT_VECTOR), `properties` (JSON), and one
  nullable typed field per declared property, `_p_<name>`, plus `_tz_<name>`
  for a datetime's UTC offset. Dynamic fields are off, so each property is
  stored once.
- **Tenancy:** partition-key multi-tenancy with `partitionkey.isolation` on:
  each segment builds its vector index per group of tenants, so a search
  filtered on one incarnation searches only its group. Milvus documents
  isolation for HNSW indexes; the store's HNSW_SQ is one.
- **Vector index:** HNSW_SQ with 4-bit codes and FP16 refinement, M=16,
  efConstruction=200; searched with ef=64 and refine_k=2. That is what
  AUTOINDEX resolves to on CPU from Milvus 2.6.10; naming it builds the same
  index on every deployment, whatever the server's `autoIndex.params.build`
  says. Measured on Milvus 3.0.2 at 600k vectors (768 dimensions; tenants of
  200k, 50k, 5k and 2,000 of 100; 4 CPUs; one run): isolation halved the
  index-build CPU (855 against 448 CPU-seconds) for about 0.2 GB more memory,
  with search throughput unchanged; under isolation, HNSW_SQ against fp32 HNSW
  took 0.8 GB less memory and 2.9 against 3.7 ms of CPU per search on the 200k
  tenant, at similar recall; the explicit search parameters raised recall over
  the previous AUTOINDEX, which set none (1.00 against 0.85 on 100-row
  tenants). (Decided: no index configurability for now.)
- **Declared properties:** each has a scalar index: VARCHAR with INVERTED;
  INT64 and DOUBLE with STL_SORT; BOOL with BITMAP; a datetime as TIMESTAMPTZ
  with STL_SORT, its offset kept beside it so it reads back in the timezone it
  was written in. Undeclared properties go in the JSON field, still filterable
  by path. Negation is the complement, as on Qdrant: a negated condition, `!=`
  included, holds where the property has no value, which Milvus's SQL-style
  null evaluation does not give on its own.
- **Scores:** the server's (cosine similarity, inner product, and the square
  root of Milvus's squared Euclidean distance), not recomputed from refetched
  vectors.
- **Server-configured limits are the server's.** A search `limit` reaches the
  server, which refuses one above `quotaAndLimits.limits.topK`. A declared
  string's VARCHAR length is `max_varchar_length` (65,535 unless configured,
  within `proxy.maxVarCharLength`), and a purge batch is `purge_batch_size`
  (10,000 unless configured, within
  `quotaAndLimits.limits.maxQueryResultWindow`). Both are `MilvusConf`
  settings and store parameters, not constants. (Decided: a limit the server's
  configuration bounds is not a Python constant.)
- **Creation converges:** the collection, its indexes (named by their fields)
  and its load are three steps, each run only when missing.
- **Client:** `AsyncMilvusClient`, every request carrying
  `request_timeout_seconds` (see [consistency](vector_store_consistency.md)).
- **Milvus Lite is not supported.** It is a separate embedded engine that
  scores, indexes and enforces collection properties differently; a URI with
  no scheme, which pymilvus reads as a Lite file, is refused. Every call and
  feature the store uses exists in Milvus 2.6.8 and later; CI tests against
  2.6.24, and the store's tests pass against 3.0.2.
- **Purge:** bounded batches listed by the incarnation field (see
  [purge](vector_store_purge.md)).

## Consequences

- An existing native Milvus collection created with the earlier schema is not
  usable by this store and has to be dropped. Existing Qdrant points are
  orphaned (see [collection registry](vector_store_collection_registry.md)).
- A Qdrant tenant's filtered searches can be wrong on Qdrant 1.15 and later
  until qdrant#10741 ships.
