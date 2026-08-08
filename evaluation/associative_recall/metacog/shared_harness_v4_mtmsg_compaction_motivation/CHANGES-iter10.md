# Iteration 10 changes — perpetual harness

Build-only iteration. No tests run. Targets the three failure modes from
iteration 9 (n=1) on `perpetual_scenarios.json`:

1. Late-DP commitment abandonment (turns 26-39 dropped).
2. Compaction over-extraction (188-301 facts vs ~8 gold).
3. Motivation stuck in `satisfied` because `unresolved_goals=[]`.

## Fix 1 — Track open sub-decisions across compactions

- Added harness-level `open_sub_decisions: dict[label, info]` in
  `run_agent_loop`. `info` carries `state`, `summary`,
  `opened_at_compaction_turn`, optional `closed_at_turn`.
- Added `_refresh_open_sub_decisions()` to fold each compaction's
  `sub_decisions[*]` into the tracker:
  - `state="opened"` → add if not already present.
  - `state="active"` → keep / add (treated as still-open).
  - `state="closed"` → mark `closed` (kept in dict so re-mention stays
    closed; flagged via `state`).
- `maybe_compact()` now takes `open_sub_decisions=` and updates it.
- TurnLog gained `n_active_sub_decisions` (count of non-closed entries),
  recorded each turn.
- `agent_result` exposes `open_sub_decisions_final` and
  `n_open_sub_decisions_final`; per-scenario summary surfaces the count.

## Fix 2 — Pipe unresolved_goals into motivation generator

- Replaced `unresolved_goals=[]` in the `update_motivation(...)` call with
  the live list built from `open_sub_decisions`:
  `[f"{label}: {summary} (opened {age} turns ago)", ...]` (or
  `f"{label} (opened {age} turns ago)"` when summary missing). Closed
  entries skipped.
- `motivation_events[*]` records now carry `unresolved_goals` and
  `n_unresolved_goals` for tracing.

## Fix 3 — Compaction narrative includes commitment refresh

- Added `_format_open_commitments_block()` that renders
  `[OPEN COMMITMENTS: <label1> (opened turn N1), <label2> (opened turn
  N2)]` from the harness-level tracker (NOT just this compaction's
  `sub_decisions[*]`). Returns empty string when nothing is active.
- The `[COMPACTED MEMORY ...]` injection now appends the OPEN COMMITMENTS
  line below the main summary so commitments from earlier compactions —
  whose original turns have long since scrolled out — are continuously
  refreshed in-context.
- Active-sub-decisions section in the COMPACTED MEMORY line was relabeled
  `Active sub-decisions (this span):` to disambiguate from the global
  OPEN COMMITMENTS list.

## Fix 4 — Reduce compaction over-extraction

- `COMPACTION_SCHEMA.facts` gained `maxItems: 8` and an updated
  description requiring NEW + decision-constraining facts only, aiming
  for 3-7 with hard cap 8.
- `COMPACTION_PROMPT` rewrote the first two guideline bullets:
  - "Extract ONLY facts that are NEW (not already in EventMemory or
    recent context) AND that materially constrain a future decision.
    Skip facts that are mere paraphrases or background coloring."
  - "Aim for 3-7 facts per compaction event, not exhaustive coverage.
    The schema enforces a hard cap of 8."
- Defense-in-depth: `maybe_compact` truncates `facts` to
  `SH_COMPACTION_MAX_FACTS` before writing to EM, exposes
  `n_facts_dropped_by_cap` in the compaction record.

## Other config knobs

- `SH_MOTIVATION_PERIOD` default: `5` → `8` (less frequent updates so
  unresolved-goal signal accumulates between calls).
- New env var `SH_COMPACTION_MAX_FACTS` (default 8).
- Both knobs surfaced in per-scenario `agent_result.config` and the
  global `SUMMARY.json` config block.

## Untouched (per guardrails)

- `SYSTEM_PROMPT`, `USER_INITIAL_STREAM`, `USER_FOLLOWUP_PROBE_ONLY`,
  `USER_FOLLOWUP_WITH_STREAM`, `USER_FOLLOWUP_STREAM_CLOSED`,
  `USER_FOLLOWUP_PREMATURE_DONE` — bodies unchanged. The motivation
  prefix (`[CURRENT MOTIVATION ...]`) continues to be a prefix on the
  followup body (existing iter9 behavior).

## Verification

- `python3 -m py_compile` clean.
- `unresolved_goals=` is wired to `unresolved_goals_input` (built from
  `open_sub_decisions`) — non-empty whenever the tracker has active
  entries (line ~1805).
- `[COMPACTED MEMORY ...]` injection contains an `OPEN COMMITMENTS`
  block (added via `_format_open_commitments_block`, line ~1349).
- `COMPACTION_SCHEMA.facts.maxItems = 8` (line ~512).
