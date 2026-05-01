# T_composition — multi-cue composition test

Tests whether the cue-gated modules (recency, open_ended_router, negation, causal) compose correctly when more than one fires per query. 25 queries (5 per type), 140 docs total. Each cluster has 1 GOLD that satisfies ALL constraints, 2-3 single-cue distractors that satisfy exactly one constraint, and 1-2 noise.

## R@1 by composition type, per strategy

| Type | n | rerank_only | S1 sequential | S2 multiplicative | S3 additive |
|---|---:|---:|---:|---:|---:|
| A: recency × absolute     | 5 | 0.200 (1/5)  | **0.200** (1/5) | 0.200 (1/5) | 0.200 (1/5) |
| B: negation × absolute    | 5 | 0.000 (0/5)  | **0.200** (1/5) | 0.200 (1/5) | 0.200 (1/5) |
| C: causal × recency       | 5 | 0.000 (0/5)  | **0.200** (1/5) | 0.400 (2/5) | 0.400 (2/5) |
| D: causal × absolute      | 5 | 0.000 (0/5)  | **0.200** (1/5) | 0.200 (1/5) | 0.200 (1/5) |
| E: open_ended × negation  | 5 | 0.000 (0/5)  | **0.200** (1/5) | 0.200 (1/5) | 0.200 (1/5) |
| **ALL**                   | 25 | 0.040 (1/25) | **0.200** (5/25) | 0.240 (6/25) | 0.240 (6/25) |

## R@5 by composition type, per strategy

| Type | n | rerank_only | S1 sequential | S2 multiplicative | S3 additive |
|---|---:|---:|---:|---:|---:|
| A | 5 | 1.000 | 0.800 | 1.000 | 1.000 |
| B | 5 | 0.600 | 0.600 | 0.600 | 0.600 |
| C | 5 | 0.600 | 0.800 | 0.600 | 0.600 |
| D | 5 | 0.600 | 0.600 | 0.600 | 0.600 |
| E | 5 | 0.400 | 0.400 | 0.400 | 0.400 |
| ALL | 25 | 0.640 | 0.640 | 0.640 | 0.640 |

## Where current sequential composition fails

S1 hits R@1 on 5/25 — basically one query per type. **All five types fail at >=80% R@1**, but the failure modes differ:

### Type A (recency × absolute)  — **double-counting failure**
- Only `rec_active` fires; absolute window (e.g. "Q4 2023") is left to the T-channel via lattice match.
- Distractor pattern: a recent doc that is OUTSIDE the absolute window. The recency_additive boost lifts that distractor over the gold (which is older but inside the window).
- Top-1 is consistently a `_d` recent-but-out-of-window distractor; gold sits at rank 2.
- Sequential's α=0.5 additive recency assumes "T already encodes the absolute window". When T_lblend's signal is weak relative to recency, recency wins and the window constraint is silently dropped.

### Type B (negation × absolute) — **negation strips both phrases**
- `neg_active` fires; `parse_negation_query` strips the cue + everything to the next sentence terminator. For "in 2024 not in summer", positive_query becomes "What client meetings did I have?" — the "in 2024" anchor is lost.
- Excluded phrase becomes "summer" → no temporal extraction (no year), so excl_ivs empty → mask fully passes through.
- Result: positive_composite (semantic only) ranks distractors high; absolute window is unenforced. This is a **regex-level bug** in `parse_negation_query`.

### Type C (causal × recency) — **causal mask kills recency lift**
- Both `rec_active` and `causal` fire. Anchor resolves correctly (we verified this by inspecting top-5).
- S1 ordering: `additive_with_recency` then `causal_signed`. The signed mask subtracts λ=0.5 from wrong-direction docs, but the anchor-doc itself gets `score - λ` which can still beat gold's combined score.
- Anchor doc embedding similarity is high (the anchor IS about the migration), so even after −0.5 it's competitive.
- S2 mult helps here (×0.0 on anchor → anchor reliably suppressed) — fixes 1/5 incremental.

### Type D (causal × absolute) — **causal cue does not fire**
- The `detect_causal()` function explicitly suppresses when `has_open_ended_cue()` is True. Queries like "What I did in Q3 2023 after the launch" trip the open_ended router (side-keyword "after" + year anchor "2023") and the causal module never fires.
- Result: open_ended router swaps to T_v5 but T_v5 has no concept of an event-anchor. Anchor is never resolved; wrong-direction distractors dominate.
- This is a **module-gating conflict**, not a composition bug. The open_ended gate's date-presence check is not sufficient: causal cues with absolute windows need the causal module too.

### Type E (open_ended × negation) — **same negation strip + no anchor**
- `neg_active` strips "but not in 2023" (or "outside of Q1 2023") — leaves the open-ended bound but parse_negation_query treats the whole tail as one excluded phrase.
- After strip: "What did I do after 2020?" — open_ended semantics lost in the positive query.
- Negation strategy bypasses the open-ended router (it replaces base with positive_composite which is semantic+T_lblend, not T_v5).

## Does multiplicative (S2) fix it?

| | S1 R@1 | S2 R@1 | S3 R@1 |
|---|---:|---:|---:|
| ALL | 0.200 | 0.240 | 0.240 |
| Δ vs S1 | — | +0.040 | +0.040 |
| Wins added | — | 1 (Type C) | 1 (Type C) |
| Wins broken | — | 0 | 0 |

S2 and S3 both fix exactly one Type C query (`comp_q_C_013`) by fully zeroing the anchor doc's score (mult: ×0.0; add: −0.5 lifts gold above it). They do **not** fix Types A, B, D, E because those failures are upstream of composition (cue gates wrong, negation parser too greedy, causal module suppressed).

## Recommended composition logic

The R@1 jump from S1 (0.20) to S2/S3 (0.24) is small because composition logic is **not the dominant failure mode** here. Four of five failure modes are upstream:

1. **Type A** — add an absolute-window module (or always run T_lblend's lattice match as a hard filter when the query has Month/Quarter/Year anchors AND a recency cue, with recency only resolving ties WITHIN the window).
2. **Type B** — fix `parse_negation_query` to handle "in YYYY not in PHRASE" — keep the FIRST temporal anchor in the positive query, treat only the post-cue phrase as excluded.
3. **Type D** — relax the open_ended gate: it should not pre-empt causal when the side-keyword's right-hand argument is a NOUN PHRASE (event reference) rather than a bare year. Concretely: prefer causal when the phrase resolves to a doc with high cosine; fall through to open_ended otherwise.
4. **Type E** — run negation on the residual after open_ended consumes its bound, not before.

For composition itself: **multiplicative (S2) is the right primitive** because each module has a natural [0,1] interpretation:
- recency → `1 + α*rec` (multiplicative lift)
- negation → `1 − excl_cont` (interval-overlap mask)
- causal → `1.0` right side / `0.1` wrong side / `0.0` anchor itself
- absolute window (when added) → `0.0` outside / `1.0` inside

Multipliers commute, don't double-count, and gracefully degrade when a module's score is uniform 1.0 (cue not fired). S1's sequential overrides actively destroy information from earlier modules (e.g. negation replaces base entirely, dropping recency's contribution).

**Recommendation**: refactor the modules so each exposes a `multiplier(query, doc)` taking values in [0, ∞), defaulting to 1.0 when its cue doesn't fire. Compose by product. Use S3-style additive form as a fallback when multipliers compress dynamic range too aggressively.

## Sample failure (Type A, double-counting)

- Query: `My latest budget review in Q2 2024` (gold: `comp_A_002_g0` = budget review on **June 26, 2024**)
- Cues: rec=True neg=False oe=False causal=False
- Distractors: same template, dates **March 14 2025** (`d0`, recent OUT of window), **January 9 2025** (`d1`, recent OUT), **April 8 2024** (`d2`, IN-window but earlier).
- S1 top-5: `[d2, g0, B_007_n0, C_014_g0, A_001_d0]` — gold ranked 2nd. Top-1 is `d2` (April 8 2024) which IS in Q2 2024 but earlier than gold; recency lifts a fresh-but-wrong-window doc (`d0` Mar 2025) too.
- The "in Q2 2024" constraint is NOT enforced — T_lblend gives `d2` (April) and `g0` (June) similar scores, and recency (with ref_time 2025-06-15) prefers `g0` slightly but the additive blend with semantic noise lets `d2` edge ahead.

The architectural fix: when a recency cue co-occurs with an absolute window, the absolute window must hard-gate the doc set BEFORE recency tie-breaks within it. Currently recency runs across the whole corpus.
