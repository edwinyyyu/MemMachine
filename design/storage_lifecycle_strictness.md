# Strict create/open storage lifecycles

Contributor-facing design notes for the removal of `open_or_create_*` from
`VectorStore` and `SegmentStore`.

## The rule

`create` fails on an existing resource; `open` returns nothing for a missing
one. Adopt-on-exists (ensure) is not a lifecycle primitive on a store. Where
a call site needs it, it composes it: open, create suppressing
`AlreadyExistsError` (a concurrent creator winning is fine), open again.

## Why: a store has no provenance

Not because strictness is a virtue. Because a store, handed a name, cannot
tell a retry of its own caller's create from a create by someone else who
chose the same name. Only a caller knows why an existing resource is
acceptable -- it is resuming work it started. Idempotency is a property of a
step that has provenance, and the store is precisely the layer without it.
An `open_or_create` on the store answers a question the store cannot
actually answer, and answers it silently.

What the removed methods bundled, beyond idempotency, was a config and a
returned handle: does this exist, what shape is it, and give me access, in
one call. The config half is what `ConfigMismatchError` existed to
compensate for -- adopting someone else's differently-configured resource --
and it is the half that is genuinely wrong, since no call site derives its
config from input. With strict create/open, a caller that loses a creation
race adopts the winner's resource via `open`, and the handle carries the
stored config for any consumer wanting its own equality policy.

An audit found exactly two consumers of the removed methods
(`semantic_manager.get_semantic_storage` and the episodic event backend's
`service_locator`), neither needing adoption as a primitive; both now use the
composed form, and both are callers that do know they are resuming their own
work. `EventMemory` itself never had ensure semantics -- it receives
already-opened handles.

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

## Relation to the server redesign

`design/server_redesign.md` reaches the same store interface by the same
argument, and places idempotency in a tenant layer's `ensure`, which has the
provenance a store lacks: it inserted the tenant row before any job ran, so
a live row is its own earlier attempt.

It also asks strict create to do something this change does not, and the
difference is worth not eliding. There, a key is a UUID minted once and
never reused, so a row existing under a key is evidence of a violated
invariant, and a create that raises on a row *in any state, including a
tombstone*, is the detector for it -- which is why tombstones are retained
rather than dropped after reclamation.

Here, names are reused by design: a collection can be deleted and another
created under the same name later. `create` raising means "this exists", not
"an invariant was violated", and separation between successive lives of a
name comes from incarnation-scoped resources (#1537, #1563) rather than from
the name being unique in time. The two read alike at the signature and prove
different things; do not carry the stronger reading across.
