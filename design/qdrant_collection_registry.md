# Qdrant on the collection registry

Contributor-facing design notes for QdrantVectorStore's collection metadata,
which lives in the SQL-backed collection registry (see
`collection_registry.md`) rather than in Qdrant itself.

## Why the registry moved out of Qdrant

Qdrant offers no conditional writes, transactions, or unique constraints. The
previous design kept collection metadata in per-namespace `<ns>__registry`
Qdrant collections (dummy-vector points keyed `uuid5(name)`), which made
`create_collection` / `open_collection` / `delete_collection` non-atomic
read-check-write sequences guarded only by a per-process asyncio lock. The
moment two processes manage the same name:

1. `VectorStoreCollectionAlreadyExistsError` stops firing -- both creates
   pass the absence check and last-writer-wins the registry point.
2. Two creations with different configs each create a different native
   collection (native names hash the config) while contending for one
   registry point; records written through the losing handle become
   permanently unreachable.

With the registry's unique-constraint insert as the commit point, both
guarantees hold across processes sharing the registry database. The
per-process `_name_locks` remain only as in-process serialization; they are
no longer correctness-bearing.

## Registry entries store resolved identity

Collections register a `CollectionRegistryEntry` =
`{config, native_collection_name, partition_key}` (see
`collection_registry.md` for the entry design and evolution policy).

Qdrant-specific use of the shared fields: the native naming scheme for new
collections is `f"{namespace}__{sha256(config)}"` (unchanged -- collections
with the same config share a native collection, separated by partition key),
generations are minted as `f"{name}#{uuid4().hex}"`, and `partition_key`
doubles as the shard key in distributed mode.

## Generations make deletion safe against held handles

`delete_collection` deletes that generation's data (partition filter delete,
or shard-key drop in distributed mode), then removes the registry entry. A
handle held in another process across the deletion keeps writing into the
dead generation: invisible to every reader, never resurrected, because a
re-creation of the same name mints a fresh generation -- even with an
identical config and therefore the identical shared native collection. No
locks, no per-write round trips; the residual cost is bounded garbage under
dead generations, sweepable by a future GC that deletes points whose
partition keys are absent from the registry.

## Ordering and crash windows

- Create: native collection first, registry entry last. The registry insert
  is the atomic commit point. A crash -- or a lost creation race -- before it
  leaves only an empty native collection (shared by config, adopted by the
  next same-config creation) and, in distributed mode, an unused shard key.
  Cleanup of the loser's native collection is deliberately not attempted:
  native collections are shared by config, and the registry intentionally has
  no list operation, so nothing can safely prove no other logical collection
  references one. The residue is bounded by the number of distinct configs
  attempted. Registry-first ordering would be worse: `open_collection` builds
  handles straight from the entry, so a crash between claim and native
  creation would hand out handles to a nonexistent native collection. The
  invariant is: registry entry implies native collection exists.
- Delete: data first, deregistration last. A crash in between leaves a
  registered-but-empty collection (documented on `delete_collection`);
  retrying the deletion completes it.

## Configuration and migration

`QdrantConf.registry_database` (required) names a configured relational
database entry; all processes sharing a Qdrant instance must use the same
registry database. There is deliberately no per-host SQLite fallback --
divergent per-host registries would look consistent while reproducing exactly
the races this design removes. One registry per Qdrant conf entry
(`qdrant_<conf name>`), so instances sharing a registry database get separate
tables.

Deployed `<ns>__registry` data is not migrated. Native collection names are
unchanged, so re-creation with an unchanged config re-registers the same
native collection and data is reachable again; the caveat is paths that
re-create with a currently-derived schema (the episodic event backend) orphan
old partitions if that schema drifted since original creation. Stale
`__registry` collections can be deleted manually.
