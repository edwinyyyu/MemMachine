# Round 14 — High-density chain stress

Goal: stress-test round 11's simplified single-ref-type architecture
(`aen1_simple`) at >5x the supersede density of round 11 phase 2.

## Setup

- Scenario: `scenarios/dense_chains.py` (deterministic, seed=17).
- Architecture: round 11's `aen1_simple` writer + structural index, unchanged.
- Writer/reader/judge model: gpt-5-mini (low reasoning).
- Hard cap: 500 LLM, 50 embed, ~$1.50 (well under $4 cap).

### Scenario shape (deterministic)

- 743 turns, 14 chains, 100 total chain values.
- 86 non-first supersede transitions (vs 19 in round 11 phase 2 — 4.5x).
- Deep chains: @User.boss = 12, @User.location = 12, @User.title = 10,
  @User.employer = 10, @User.team = 8, @User.hobby = 8.
- Tail-bucket transitions: ~10-13 per 100-wide bucket from (100,200] through
  (700,800]. (Round 11 phase 2 had 0 in (800,1000] and ≤1 in (400,800].)

## Metrics

- `ref_emission_rate` per 100-wide bucket — overall durability claim.
- `ref_correctness_rate` — does the emitted ref point at the actual previous
  chain entry's uuid (not a random older entry)?
- `entry_emission_rate` — did writer recognize the transition at all?
- `atag_rate` — @-tag discipline.
- End-to-end Q/A on 32 chain-traversal questions (deterministic + LLM judge).

## Running log

(filled in as we run)
