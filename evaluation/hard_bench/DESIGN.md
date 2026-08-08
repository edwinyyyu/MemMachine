# Hard memory-augmented benchmark — design

## Problem

Existing benchmarks each test ONE memory subsystem in isolation:

- LoCoMo / LongMemEval — episodic recall over dialog
- temporal_extraction / temporal_retrieval — anchored time queries
- attribute_memory R20-R25 — entity resolution over multi-mention dialog
- v9-v25 pattern_query thread — cross-domain pattern abstraction (V15)
- associative_recall — long-task spreading activation

Each thread optimizes its own substrate against its own corpus shape. None test how the subsystems interact under realistic agent workloads. The current production memory architecture mixes these subsystems but has not been benchmarked as a whole.

This benchmark exercises the FULL stack on a UNIFIED corpus across four task families chosen to require different memory capabilities.

## Constraints (from user)

- Use **real EventMemory** (`memmachine_server.episodic_memory.event_memory.EventMemory`), Qdrant + SQLite + OpenAI text-embedding-3-small. No strawmen.
- Use **best timestamp system**: `evaluation/temporal_retrieval` v5.1 (DNF planner + classifier + 2-pass extractor + hybrid pool + cross-encoder rerank) for queries with explicitly mentioned anchors.
- Use **best entity-resolution system**: R20-R25 prose-fact + DSU registry + working-memory + recursive-cognition (`semantic_memory/evaluation/attribute_memory/research/round24_recursive_cognition`).
- Use **best pattern-matching system**: V15 with soft judge weights `{YES:1.0, PARTIAL:0.85, NO:0.7}` (per `project_pattern_query_v15_softjudge_winning.md`).
- Use **only `gpt-5-mini`** for memory operations, answer generation, and judging.
- **Use Claude subagents** (not OpenAI) for benchmark data generation.

## Task families (data generation)

### Family A: Long autonomous tasks (`task_long.json`, n=20)

Agent gets a high-level task prompt; must probe memory iteratively to surface constraints, prior decisions, cross-domain priors. Tests planning-time memory access (DESIGN.md spreading activation pattern).

Each scenario: task_prompt + 15-40 memory_turns (plants/decoys/noise) + 3-7 gold_subdecisions + (optional) guidelines_present_in_memory. Each subdecision tagged with `memory_capability_required ∈ {episodic, temporal, cross_domain_pattern}` to enable per-capability scoring.

### Family B: Guideline-following (`task_guideline.json`, n=20)

Memory contains team-specific rules; user task innocently violates one. Agent must (a) recognize conflict, (b) cite guideline, (c) propose alternative. Tests obscure-rule recall + safety reasoning.

Each scenario: task_prompt + 5-15 memory_turns + gold_guideline (text + violation explanation + recommended_alternative) + obscurity ∈ {HIGH, MEDIUM, LOW}.

### Family C: Simple QA over dialog (`task_qa.json`, n=20)

Subagent-generated, internally consistent. Single-question single-answer over a self-contained dialog history. Tests episodic recall baseline.

Note: NOT pulled from LoCoMo or LongMemEval verbatim — mixing external dialog data with our synthetic team scenarios risks fact conflicts (e.g., persona attributes inconsistent across scenarios). All Family C data is generated from scratch with consistent personas.

Each scenario: task_prompt (the question) + 20-50 memory_turns + gold_answer + gold_evidence_plant_ids + difficulty ∈ {EASY, MEDIUM, HARD}.

### Family D: Temporal anchored QA (`task_temporal.json`, n=20)

Questions explicitly anchored in time ("after the Acme launch", "in March 2026", "last week"). Tests temporal_retrieval system specifically.

Subagent-generated, internally consistent. Each scenario contains plants in the anchor's window AND **out-of-window decoys** (events sharing topic but outside the time window) — the agent must distinguish by time, not by topic. Anchor types: calendar_pin, anaphoric_event, deictic_relative, recurring_period, date_range. current_time = 2026-04-30 (relative anchors resolve against this).

## System under test

### Substrate: Real EventMemory

Single Qdrant collection per scenario + SQLite segment store. Ingest converts each `memory_turn` (or LongMemEval session message) into an `Event` with `Content(items=[Text(...)], context=MessageContext(speaker=..., timestamp=...))` and `properties` for filterable scoping.

### Retrieval channels (RRF ensemble, fixed K=5 context budget)

| channel | implementation | strength |
|---|---|---|
| `em_cosine` | `EventMemory.query(text, vector_search_limit=K)` | surface vocab semantic match |
| `em_temporal` | `TemporalRetriever` over derivative texts indexed once per scenario | explicit anchor queries |
| `em_entity` | R25 retrieve() over DSU-registered facts (separate collection) | entity-tracking + coreference |
| `em_pattern_v15` | V15 tier emission + soft judge (NO=0.7) probes against EM | cross-domain abstract patterns |

Combined via Reciprocal Rank Fusion at K=5 (per `analyze_augmentation_k.py` finding: K≥2 augmentation strictly improves baseline).

### Agent loop (DESIGN.md spreading-activation, with multi-channel retrieval)

Phase 1 — planning probes:
- Agent gets `task_prompt` only.
- Iteratively probes memory (up to 6 rounds), receives RRF top-K hits with `[date, time] speaker:` prefixes.
- Multi-turn explicit thinking (mt_messages) preserves prior reasoning.
- STOP when no new turn_ids surfaced or agent emits STOP.

Phase 2 — plan-only:
- Agent receives accumulated context block, writes numbered plan with no execution.

Phase 3 — per-step execution:
- For each plan item: up to 2 mid-step probe rounds → write step content.
- Plan-context, exec-context-from-prior-steps preserved.

LLM: gpt-5-mini for all calls (planning, executing, judging). reasoning_effort="medium".

## Evaluation metrics

For each scenario:

1. **gold_subdecision_recall** — per gold subdecision, was it addressed in agent's transcript? (LLM judge: gpt-5-mini)
2. **plant_retrieval@K** — for each addressed subdecision, was the gold plant retrieved within top-K of the cue probe?
3. **guideline_compliance** (Family B): binary — did agent surface the guideline AND propose the recommended alternative?
4. **answer_correctness** (Family C/D): LLM judge against oracle.

Aggregate across families: per-family means + overall mean.

## Ablation matrix

To measure each channel's contribution as augmentation on `em_cosine`:

- `em_cosine` alone (baseline)
- `em_cosine + em_temporal` (RRF)
- `em_cosine + em_entity` (RRF)
- `em_cosine + em_pattern_v15` (RRF)
- `em_cosine + em_temporal + em_entity + em_pattern_v15` (full RRF)

Per-family delta vs baseline = each channel's contribution. Overlapping channels = redundant. Strict-positive disjoint deltas = end goal "any subset" condition.

## File layout

```
hard_bench/
  DESIGN.md             (this file)
  data/
    task_long.json      (subagent-generated, n=20)
    task_guideline.json (subagent-generated, n=20)
    task_qa.json        (LongMemEval subset, n=30)
    task_temporal.json  (temporal_extraction subset, n=20)
  cache/
    llm_cache.json      (gpt-5-mini call cache)
    em_*.sqlite3        (per-scenario EM segment stores)
  results/
    <run_id>_<family>_<channel_combo>.json
  system.py             (retrieval channel wiring)
  agent.py              (spreading-activation loop)
  runner.py             (per-scenario driver)
  judge.py              (LLM judges per metric)
  ingest.py             (bench data → EM events)
  analyze.py            (cross-channel ablation analysis)
```

## Non-goals

- Speed/throughput optimization
- Beating existing per-subsystem benchmarks (we expect to LOSE on simple-QA vs LongMemEval-tuned configs because we're optimizing for breadth)
- Production deployment readiness

## Open questions

- Can a single Qdrant collection support all 4 channels, or do entity/pattern need separate collections?
- How to feed temporal_retrieval's Doc representation (which expects ref_time per doc) without re-indexing? Likely: one parallel Doc list per scenario.
- V15 expects text corpus, not segments. Need an adapter that pulls segment texts from EM and feeds to V15 logic.
- Spreading-activation agent loop currently uses `mid_execution_eval_e2.py` patterns; we need to factor those into a reusable module.

## Phased delivery

Phase 1 (this session):
- DESIGN.md (this file)
- Subagent data generation (long + guideline)
- Skeleton `system.py` + `agent.py` + `runner.py`
- End-to-end smoke test on 2-3 scenarios (any family)

Phase 2 (next session):
- Wire all 4 channels to RRF ensemble
- Run ablation matrix on full 90-scenario corpus
- Analyze per-channel contribution

Phase 3:
- Address gaps revealed by ablation (likely: improve weakest channel, fix cross-channel interference)
- Re-run, document.
