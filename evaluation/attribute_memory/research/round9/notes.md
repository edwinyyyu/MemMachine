# Round 9 Research Notes

**Primary question**: Can a single append-only log with @entity mentions (AEN-1) track state as
effectively as an entity-partitioned architecture at scale?

## Plan

1. Build a synthetic conversation scenario with controlled ground truth for state-tracking
2. Build 3 architectures:
   - AEN-1: single log, each entry has `text`, `mentions[@Name]`, `refs[(uuid, relation)]`,
     `relation ∈ {clarify, refine, supersede, invalidate}`
   - AEN-1 + views: AEN-1 + materialized per-entity views
   - Round-7 baseline: entity-partitioned `<Entity>/<Category>` + `<Holder>/<Category>/<Role>` slots
3. Grade on 20 held-out state-tracking questions; deterministic where possible, LLM-judge where
   necessary.

## Budget tracking
- Hard cap: $5, target $2-3
- 400 LLM calls + 200 embedding calls ceiling

## Running log

### Phase 0: setup (2026-04-23)

Starting clean. Will reuse round 7/8 caches where possible.

Using a hand-designed scenario with embedded ground truth to avoid LLM scenario-gen cost and to
make the grader deterministic where possible.
