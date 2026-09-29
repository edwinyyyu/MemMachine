# Qdrant vector store

Status: accepted and implemented 2026-09-29 in #1631, except conditional
upsert (proposed) and write ordering on replicated deployments (open). How the
Qdrant store meets the shared contracts: [collection
registry](vector_store_collection_registry.md),
[purge](vector_store_purge.md), [consistency](vector_store_consistency.md),
[isolation](vector_store_isolation.md).

## Layout

- **One native collection per namespace and configuration**, named
  `{namespace}__{sha256(config)}`, the digest over the configuration's JSON.
  It holds every logical collection of that namespace and configuration, one
  incarnation each. Admitting a collection creates nothing in Qdrant unless
  its configuration is new.
- **Vectors:** one unnamed dense vector, the configuration's dimensions and
  metric.
- **Graphs per tenant, not per collection:** HNSW `m=0`, `payload_m=16`. No
  collection-wide graph is built; each value of an indexed payload field gets
  its own graph once it has enough points in a segment, so a search filtered
  on one tenant walks that tenant's graph.
- **The incarnation:** payload field `sys-incarnation`, a keyword index with
  `is_tenant=true`, which also has Qdrant keep a tenant's points together.
  Every search, scroll and delete of a handle filters on it.
- **Declared properties:** a payload index per property of the collection's
  schema, typed by the property's type.
- **Creation converges:** the collection and each payload index are created
  under separate already-exists guards, so a creation that failed between them
  is completed by the next.
- **Client:** `AsyncQdrantClient`, with `request_timeout_seconds` as its
  timeout. Custom sharding was removed in #1671.

## Point ids

A point's id is its record's UUID, as it is. (Accepted: ids stay readable;
deriving them from the incarnation would scope them structurally but hide the
mapping from operators, and is kept for when it becomes necessary.)

The id space is the native collection's, shared by its logical collections,
and Qdrant's upsert replaces a whole point, vectors and payload alike
(`lib/shard/src/update/points/upsert.rs` at v1.19.1). So a plain upsert of a
UUID that another collection's point holds replaces that point and moves it to
the writer's incarnation: the first collection's record goes missing (its
reads filter it out), and neither collection reads the other's. The contract's
rule that record UUIDs are service-minted is what keeps that from happening
today.

**Proposed: conditional upsert.** Qdrant 1.16 added `update_filter`
(qdrant/qdrant#7006). An upsert with a filter on the writer's incarnation
inserts new ids as before, updates the writer's own points, and leaves a point
under another incarnation as it is, skipping the writer's point without an
error: first write wins. It would let the store meet the
[isolation](vector_store_isolation.md) guarantee whatever UUID a record
carries. Its cost stays with the writer (Qdrant 1.19.1, 4 CPUs / 4 GB, 300
tenants x 1,000 points in this layout):

| Request | p50 plain -> scoped | Server CPU per point |
|---|---|---|
| 1 new point | 1.53 -> 2.23 ms (+46%) | x1.85 |
| 1 point overwritten | 1.90 -> 2.00 ms (+6%) | x1.23 |
| 10 points | +12-14% | x1.10-1.20 |
| 100 points | +4% new, +39% overwrite (noisy) | x1.08 / x1.60 |
| 1,000 points | +1% new, +12% overwrite | x1.23-1.28 |
| 8 clients, 1 new point each | 4,384-4,684 -> 3,440-3,554 requests/s (-24%) | x1.10 per request |

With 4 clients searching tenants 0-149 while 4 others upserted into tenants
150-299, the searchers ran 2,653-2,964 per second beside conditional 1-point
upserts against 2,729-2,753 beside plain ones, and 2,300-2,367 against
2,307-2,381 with 5-point upserts; the writers' own throughput dropped 7-9%.

## Filtered-search correctness (qdrant#10741)

With per-tenant graphs, a search filtered on a tenant *and* a second condition
can return nothing, or unrelated points, when the tenant's graph entry point
fails the second condition. Measured on Qdrant 1.19.1 (12 tenants of 100k
points, `m=0, payload_m=16`, a boolean set on half the points; top 10 filtered
on the tenant and the boolean, against the same query run exactly): 26 of 60
searches correct with the boolean indexed; 30 returned nothing while the exact
search returned 10; 4 returned 10 points sharing nothing with the exact top
10; none violated the filter. It first appears in 1.15.0 (0 of 60 wrong on
1.14.1, 15 on 1.15.0, 30-45 on later versions) and is reachable for tenants of
a few thousand points and up at default settings. The upstream fix,
qdrant/qdrant#10741, is open. (Accepted: the store is written as if that fix
has shipped; no graphless layout or exact-search workaround.)

## Purge

One filter-delete of the whole incarnation per round: the round scrolls for
one point under the incarnation and, when it finds one, deletes by filter.
Measured (Qdrant 1.18.3, 1.19.0 and 1.19.1; 2.11M points in this layout, dead
incarnations of 10k to 1M among live tenants under search, upsert and scroll
traffic; 2 CPUs / 4 GB): the delete stalls writes and scrolls on the shard,
never searches, for its duration, about 1.3 s per 1M points on 1.19.1 at 3
segments per shard (7.3 s at 101), with no errors. Deleting in batches instead
collapsed once a vacuum rebuilt the incarnation's segments mid-purge: every
batched 1M purge (7 of 7 runs, on all three versions) slowed to minutes-long
deletes and 60-120 s stalls for every tenant.

## Consistency

**One node.** Upserts and deletes pass `wait=True`, qdrant-client's default,
and return once applied, so every later query, from any process, reflects
them; the store states no delay. Writes to a point apply in one order.

**Replicated.** Qdrant's consistency documentation: a write succeeds once
`write_consistency_factor` replicas (1 by default) apply it; a query reads one
replica by default and can miss a write another replica has; and with the
default `weak` write ordering "write operations can be freely reordered",
while `medium` and `strong` serialize them through a leader. The store sets
none of these, and creates collections with the server's default replication
factor. It states the replicated read delay as unbounded.

Measured on a three-node Qdrant 1.19.1 cluster in Docker (1.5 CPUs / 1.5 GB
per node; one shard replicated on all three nodes; write consistency factor 1;
each node's reads served by its local replica, so reading from each node
compares the replicas; 10 ms +- 10 ms of network jitter on every node for the
hazards; 8 threads, 400 points per case):

| Case | `weak` | `medium` | `strong` |
|---|---|---|---|
| Upsert through node 1, then, once it returned, delete through node 2: replicas still holding the point | 0 | 0 | 0 |
| Upsert v1 through node 1, then v2 through node 2: replicas holding v1 | 0 | 0 | 0 |
| Two clients upsert different values of one point at once, through nodes 1 and 2: points whose replicas disagree | 115 of 400 | 0 | 0 |

So sequential writes kept their order under the default, but overlapping
writes to one point left replicas disagreeing (compared 5 s after the writes),
against the contract's last clause. The cost of the stricter orderings falls
on writes, so this measures writes alone, no reads running: each phase on a
fresh collection, 8 gRPC clients for 10 s, three rounds with the orderings
rotated:

| Upserts | `weak` | `medium` | `strong` |
|---|---|---|---|
| cluster, 1 point: per second (p50) | 2,331-2,612 (2.8-3.1 ms) | 1,560-1,781 (4.1-4.7 ms) | 1,659-1,748 (4.2-4.6 ms) |
| cluster, 10 points: per second (p50) | 1,616-1,649 (3.3-3.9 ms) | 1,047-1,313 (5.6-6.9 ms) | 1,228-1,307 (5.6-5.9 ms) |
| single node, 1 point: per second (p50) | 5,570-5,727 (1.2 ms) | 3,533-4,425 (1.7-1.9 ms) | 3,684-4,321 (1.7-1.9 ms) |
| single node, 10 points: per second (p50) | 1,968-2,103 (1.9 ms) | 1,988-2,096 (2.6-2.8 ms) | 1,918-2,126 (2.5-2.7 ms) |

**Open.** Which ordering the store passes, and whether always or only on
replicated deployments; the options are in the
[consistency](vector_store_consistency.md) document.

## Consequences

- A Qdrant tenant's filtered searches can be wrong on Qdrant 1.15 and later
  until qdrant#10741 ships.
- Existing Qdrant points written before #1631 are orphaned (see [collection
  registry](vector_store_collection_registry.md)).
