# Unresolved Explorations — TODO list

Scanning back through the research conversation, experiments that were discussed/started but not completed:

## Killed early (before budget constraints were set — may be worth revisiting)

- **Cross-session synthesis data** (Outer Wilds-style) — agent started, killed during budget cleanup. Tests knowledge transfer across separate conversations.
- **Long conversation scaling test** — agent started, killed. Would map the "activation threshold" curve (100, 250, 500, 1000, 2000 turn conversations).
- **New adversarial categories** (causal, counterfactual, perspective swap, ellipsis, cross-topic bridging, retraction, commitment, hypothetical) — synthetic data generator agent killed during cleanup. User later said to hold off on new data until existing challenge categories improve.

## Scenario types mentioned but never operationalized

- **Drug interaction scenarios** — user mentioned as adversarial example. Requires external knowledge (interacting drugs) + memory (user's medications). No test data.
- **Brand guidelines / preemptive recall** — presentation prep needs brand guidelines that only surface at sub-subtask level. Proactive category is closest we have but isn't quite this pattern.
- **Procedural checklist completion** — exists in synthetic data but not as specific adversarial test.

## Methodology untested

- **Top-down prompt distillation** — user's suggestion. Start with verbose prompt (full research observations as phenomenon-based priors), cut iteratively. Supervisor_control uses compressed priors; phenom_supervisor uses phenomenon priors. Full top-down distillation from a v2f-replacement perspective not done.
- **Enumeration of all phenomena**: listed 20 failure modes in response to user. Could test whether a prompt enumerating them helps.

## Novel architectures not yet tested

- **Multi-model supervisor** — use stronger model (Opus/gpt-5) as supervisor over gpt-5-mini worker. Same-model supervision has been tested; different-model not yet.
- **Temperature-based cue diversity** — sample v2f cues at temp>0 multiple times, union results. Tests whether stochasticity beats prompt-engineered diversity.
- **Information-gain objective** — explicit reward for cues that find non-duplicate relevant segments. Hard to measure relevance though.
- **Combining constraint-type + v2f** — constraint-type cues got 100% at r@all on logic_constraint. Never tested as an ADDITIVE to v2f at K=50 with proper structure.
- **Post-hoc neighbor injection** — just launched (neighbor_priority agent). Tests error_analysis finding.
- **Higher K budget granularity** — tested K=20, 50, 100. Could test K=30, K=40 for finer curve.

## Hypotheses formed but not tested

- **Conditional activation** — skip cue generation for short conversations (<300 turns) since gains are small. Production config. Not tested as a gated system.
- **Retrieval log as STOP signal only** — supervisor_control is testing this (running). Distinct from reflection-for-diversification which was tested and failed.
- **LLM rerank trade-off** — LLM reranking of decompose_then_retrieve was 9W/0L but expensive. Reranker vs "just retrieve right K" showed cues do the heavy lifting. Unclear if reranking adds value ON TOP of v2f.
- **Extract-then-generalize cues** — entity_extract currently tests this but with one LLM call. Two-step (extract, then generate cues from entities) not tested.

## Ingestion-adjacent (out of scope per user, but listed for completeness)

- **Context-enriched embeddings** — embed "previous turn: X; current turn: Y" instead of just Y. Fixes anaphoric cases (8% of missed in error analysis).
- **Alias extraction at ingest** — run a pass extracting entity aliases/synonyms, embed those separately. Fixes evolving_terminology (and user flagged as hardest broad challenge).
- **Unresolved-question markers** — tag turns where the user asked a question but no answer appeared. Enables "known unknowns" retrieval.

## Questions deferred

- Can a reliable classifier route between v2f / CoT / constraint-type specialists at low cost?
- What's the actual K=50 "production default" story when all good specialists are available?
- Does combining ALL proven components (v2f + gencheck + constraint-type + neighbor priority) at K=50 beat v2f?
- Is there a single-prompt architecture that matches v2f on every dataset at every budget? (Seems no, but top-down distillation not fully tested)

## Currently running (will complete)

1. **entity_extract** — grounded cue generation via content analysis
2. **supervisor_control** — compressed-priors supervisor (stop/continue)
3. **two_call_dispatch** — separated classifier+specialist (fixes self_v3 mode-bleed)
4. **phenom_supervisor** — phenomenon-based priors (top-down methodology)
5. **neighbor_priority** — tests error_analysis finding that 50% misses are adjacent
6. **error_analysis** — DONE (ground-truth diagnosis; finding was huge)
7. **logic_constraint_deep_dive** — light analysis, probably done
8. **pareto_frontier** — light analysis, probably done
9. **cue_comparison** — light analysis, probably done

Need to check light-task completions.
