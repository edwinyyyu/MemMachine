# Strict create/open storage lifecycles

Contributor-facing design notes for the removal of `open_or_create_*` from
`VectorStore` and `SegmentStore`.

## The rule

Storage lifecycle APIs are strict: `create` fails on an existing resource,
`open` returns nothing for a missing one. Adopt-on-exists (ensure) semantics
are not a lifecycle primitive; where a call site genuinely needs them, it
composes them -- open, create suppressing `AlreadyExistsError` (a concurrent
creator winning is fine), open again.

## Why

Adopt-on-exists is the right primitive only where creation is a lazy side
effect of use by symmetric actors and there is nothing for adoption to get
wrong. Collections and partitions are not that: creation carries a config,
and adopting someone else's differently-configured resource is a hazard the
API was compensating for with `ConfigMismatchError` machinery. With strict
create/open, a caller that loses a creation race adopts the winner's resource
via `open`, and the handle carries the stored config for any consumer that
wants to enforce its own equality policy -- no current consumer does, because
every call site derives its config from code, not input.

An audit found exactly two consumers of the removed methods
(`semantic_manager.get_semantic_storage` and the episodic event backend's
`service_locator`), neither needing adoption as a primitive; both now use the
composed form. `EventMemory` itself never had ensure semantics -- it receives
already-opened handles.

The one place ensure remains is collection registry `startup()`,
deliberately: registry storage is materialized lazily at bootstrap by
symmetric processes, from names derived in code, and carries nothing for
adoption to mismatch (the entry format version is validated separately, by
the registry of registries). Bootstrap is where ensure semantics belongs;
everything above it is strict.

## Removed

- `VectorStore.open_or_create_collection` (all four implementations) and
  `VectorStoreCollectionConfigMismatchError` (the method was its only
  raiser).
- `SegmentStore.open_or_create_partition` and
  `SegmentStorePartitionConfigMismatchError`, with the implementation's
  mismatch-check helper.

Untouched, and out of scope: `EpisodicMemoryManager
.open_or_create_episodic_memory` is an in-process instance-cache accessor
(get-or-construct a session's memory object), not a durable-storage ensure.
