# Round 11 — Writer-stress study on AEN-1

Goal: determine (a) if one ref type can replace the four typed relations
(clarify/refine/supersede/invalidate) and (b) whether an LLM writer reliably
produces ref chains at 1000+ turn scale.

## Architecture variants

- `aen1_simple` — single ref type, `supersede_head` updated every time a new
  entry matches a `(@entity, predicate)` pair of a prior ref target.
- `aen1_typed_baseline` — four typed relations, equivalent to round10's
  `aen1_indexed.py`.
- `aen1_plain` — untyped, no indexes.

## Phase plan

- Phase 1: 110-turn round9 scenario, three arches, 20 state-tracking questions.
- Phase 2: deterministic ~1000-turn conversation, run simple writer, measure:
  - Ref-emission rate (did writer emit a ref on each ground-truth supersede?)
  - Ref accuracy (did ref point at the correct chain head?)
  - Chain integrity by turn-bucket (200-turn buckets)
  - @-tag drift (does the @-tag discipline hold at T=800+?)
  - End-to-end Q/A accuracy (~30 questions)
- Phase 3: same 1000-turn scenario, typed writer, same metrics + token/latency
  comparison.

## Budget

Hard cap $5 / 600 LLM / 200 embed. Target $2-3.
Phase 1 target: ~50 LLM + embeddings reused.
Phase 2 target: ~200 writer calls + 30 reader + 30 judge.
Phase 3 target: ~200 writer calls + 30 reader.

Total LLM budget plan: ~550 calls; stop hook at 500.

## Running log

(written live during experiments)
