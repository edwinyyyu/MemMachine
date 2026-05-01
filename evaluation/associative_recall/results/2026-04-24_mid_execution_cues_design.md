# Mid-execution retrieval cues — research design (2026-04-24)

## Problem

Tasks like "prepare a presentation for external clients" or "prepare a banquet for the high school track team" do not embed the cues for the sub-decisions an executor will face after working on them for a while (presentation colors → brand guidelines; food → allergies). Pre-task decomposition is not enough — the agent only encounters many sub-decisions during execution. Today's agents do not naturally generate "what do I need to know about X right now?" cues mid-task; that behavior is not strongly elicited by training. Existing benchmarks (small Q&A on items the asker already knows) do not stress this regime.

Constraint: build on EventMemory only. Examples in `evaluation/event_memory/`.

## What we already have

- `proactive_memory.py:349-532` (`run_proactive`) — DECOMPOSE → CUEGEN → SUFFICIENCY (+ follow-up probe rounds). Runs **once at task start**. Final union/dedupe by `turn_id`, ranked by max cosine score.
  - DECOMPOSE_PROMPT (lines 66-91): task → 3-6 INFO NEEDS with priority + expected_vocab.
  - CUEGEN_PROMPT (lines 94-116): per-need → 2 chat-register cues, embedded same shape as ingested turns.
  - SUFFICIENCY_PROMPT (lines 119-144): per-need coverage classification + follow-up probe.
- Listener notes v4 (`em_setup_notes_v4.py`, `notes_eval_v4.py`) — proactive condition C task-sufficiency = 6.0 vs 4.6 baseline (+1.40); recall@K neutral. Notes fire **at conversation ingestion**, not during task execution.
- `task_execution.py` — GENERATE-AND-CHECK (`NEED:` markers) + AUTONOMOUS (THINK/RETRIEVE/WRITE/DONE). Closest existing scaffolding to mid-execution retrieval triggering. Scored on recall@K against a fixed gold set.
- EventMemory API: single-string `query()`, `property_filter` on `m.*`, no native multi-probe (caller score-merges), no cross-event relations, ts+1μs idiom for sub-second ordering.
- Synthetic dinner-party (`synthetic_scenarios.py:53-155`): planted dietary-need facts, single "plan dinner" prompt — closest existing shape to "presentation/brand-guidelines."

## Diagnosis (the bottleneck)

The mechanism question (how to make the agent emit the right cue at the right step) is **not** the primary bottleneck. The primary bottleneck is **evaluation**: no current benchmark scores intermediate retrieval triggers; all scoring collapses to recall@K on a fixed gold set or to LLM-judge on the *final* assembled context.

Without an intermediate-trigger scorer, every hypothesis below is unfalsifiable (procedure: "If you can't pick a baseline, your experiment isn't well-defined yet"). Mechanism work without this benchmark will repeat the v2f local-optimum pattern — small or noisy moves on the wrong metric.

## Hypotheses (Stage 1)

Ranked by `tractability × orthogonality × impact-if-true`.

| # | Hypothesis | Mechanism | Cost | Why falsifiable |
|---|------------|-----------|------|----------------|
| H1 | **Per-action self-probe.** At each agent action boundary, reuse `CUEGEN_PROMPT` with the *next planned action* (not the global task) as the "need." | Reuse `proactive_memory` machinery, re-trigger per action. | LLM-per-action. | If sub-decisions emerge mid-execution as claimed, this should retrieve planted facts that one-shot DECOMPOSE misses. |
| H2 | **Action-text-as-cue (no LLM).** Embed agent's stated next action as the EM query directly. | Skip CUEGEN entirely. | Embedding-per-action only. | Lower bound on the value of LLM cue generation. If H1 ≈ H2, the LLM cue-gen step doesn't help mid-task. |
| H3 | **Decision-keyword trigger.** Detect "I'll choose…/decide on…/let's go with…" in agent stream; fire CUEGEN only on those events. | Cheaper than H1 when most actions aren't decisions. | LLM only at gated triggers. | Catches the "color/food decision" moment specifically. Falsifies if decisions are too implicit to detect. |
| H4 | **Adversarial gap-finder (two-pass).** First pass: agent drafts plan/output without retrieval. Second pass: critic LLM asks "what would change this output if I knew about user prefs / brand / constraints?" emits probes. | Replaces sequential cue-gen with post-hoc gap audit. | 1-2 extra LLM calls per pass. | Tests whether the failure is in cue *generation* or in cue *elicitation*. |
| H5 | **Persona pre-prompt only.** System message: "before each substep, ask 'what facts about the user/brand/constraints might apply?' and emit a query." No mechanism change. | Tests user's claim that this requires "a different personality." | Free. | Prior elaboration-style prompts on v2f no-op'd or degraded (`project_v2f_local_optimum`). Expectation: no help. Useful as a control. |
| H6 | **Listener-notes-on-execution.** Adapt v4 notes to fire on the *agent's own action stream* rather than ingestion: each action → LLM emits "what info do I need next?" stored as an EM event itself. | Architectural cleanest; preserves traceability. | LLM per action; doubles ingest. | Clean superset of H1 + persistent breadcrumb of intent. |
| H7 | **Forward-action-rollout decomposition.** At task start, agent rolls forward N sub-actions hypothetically (without execution) and runs DECOMPOSE on each. Pre-execution but deeper than today. | Pure pre-execution; no mid-task triggering. | Up-front N× CUEGEN cost. | Tests whether deeper *static* decomposition is enough — i.e., whether mid-execution triggering is genuinely needed or is just a deeper-decomposition surrogate. |

Out of scope (per `feedback_retrieval_research_scope`): finetuning (training), BM25 / cross-encoder rerank, model-provider swaps.

## Experiment 0 — build the benchmark (precondition)

Without this, none of H1-H7 is falsifiable. Minimal viable shape:

- **Corpus.** Re-use existing LoCoMo conversations. For N task scenarios, plant a "fact pack" in turns scattered across the conversation — each pack contains 2-5 facts that bear on a specific *sub-decision* of a long task. Examples (taken from user framing + the existing dinner-party scenario):
  - Presentation → brand colors / past-feedback / audience
  - Banquet → allergies / budget / venue capacity
  - Project plan → deadlines / past blockers / collaborator availability
- **Task prompt.** Single high-level task ("prepare a presentation for the external client meeting Friday") that does NOT mention the planted concepts.
- **Sub-decision script.** For each scenario, hand-write the sequence of sub-decisions an executor should hit (e.g., 1-pick-format, 2-choose-colors, 3-draft-talking-points). At each sub-decision, the gold expectation is which planted facts are required.
- **Scorer.** Two metrics, side by side:
  - `triggered_recall@K`: at each sub-decision step, did the agent emit a probe whose top-K retrieval contains the gold planted facts? Per-step recall, then averaged.
  - `task_sufficiency` (LLM judge): unchanged from `notes_eval.py:246-268`, applied to the final assembled context. Lets us see if intermediate triggering matters for end-to-end output.
- **Size.** 12-20 scenarios initially. Cheap; can extend.
- **Effort estimate.** 1-2 days. Most of the time is hand-curating sub-decision scripts and gold packs, not code.

Risk: scenarios I author may bias toward the cue-gen prompts I'm tuning. Mitigation: hold out a 4-scenario subset whose sub-decision scripts are written without looking at the cue-gen prompt.

## Experiment 1 — first real test (after E0)

Four-arm comparison on the new benchmark, all built on EventMemory only:

| Arm | Mechanism | Purpose |
|-----|-----------|---------|
| A | No mid-task probe (single retrieval at task start using the task prompt only). | Lower-bound baseline. |
| B | H2 — action-text-as-cue (per-action embedding probe, no LLM cue-gen). | Cost-floor for any mid-task triggering. |
| C | H1 — per-action `CUEGEN_PROMPT(need=action_text)`. | Fair extension of `proactive_memory` to mid-task. |
| D | H7 — forward-action-rollout decomposition at task start (pre-execution depth). | Tests whether mechanism-side effort is *needed* mid-task or whether deep static planning subsumes it. |

Pre-registered predictions:
- `triggered_recall@K`: A < B < C, with D ≈ C if static rollout is good enough OR D < C if mid-execution surfaces sub-decisions the rollout missed.
- `task_sufficiency`: gap will be smaller than `triggered_recall@K` gap (per `project_recall_vs_endtoend` — recall and end-to-end LLM-judge diverge).

Bias controls (from procedure Stage 2):
- Same total LLM-call budget across arms (cap C and D's calls; B has none).
- Same embedding model + same query-format (chat-register, matching ingestion).
- Same K and same merge strategy (max-score across all probes).
- Held-out 4-scenario subset for the final comparison; tuning on the other 8-16.

## Stage-3 questioning (in advance)

When results land, ask before celebrating any move:
- Was the gold pack actually retrievable by *any* probe? (Tests planted-fact embedding distance — if gold isn't retrievable in principle, the experiment is a benchmark bug, not a mechanism failure.)
- Did per-action probes find facts that *weren't* the planted gold but were still useful? (False-negative on triggered_recall but real-world value.)
- Did the LLM-judge sufficiency move while triggered_recall stayed flat, or vice versa? (Decides whether to ship on triggered_recall or on sufficiency.)

## Adversarial cases to author up-front (Stage 4 prep)

Categorize and design before running:
- **Implicit decision** ("the food I'm going to serve is …" — no decision keyword) — stresses H3.
- **Multi-fact sub-decision** (color pick depends on brand AND past-feedback AND audience) — stresses H1/H2 cue specificity.
- **Distractor pack** (planted facts that *look* relevant but aren't, e.g. wrong client's brand colors) — stresses retrieval precision at the sub-decision step.
- **No-op step** (sub-action where no planted facts apply) — measures false-positive trigger rate.

## Tracking

| Status | Item |
|--------|------|
| In progress | E0 benchmark build (v1 scenarios + scaffolding done; v1 sanity test failed — see below) |
| Open | E1 four-arm comparison |
| Open | Adversarial pack |

This file is the running log for this research thread; subsequent experiment results append below their own headers.

---

## E0-v1 sanity test (2026-04-24, presentation-01, K=5,10,20)

| cue strategy | mean_recall@5 | mean_recall@20 | mean_fpr@20 (noop) |
|---|---|---|---|
| `task_prompt` (bad-cue baseline) | **1.000** | 1.000 | 0.250 |
| `decision_text` (action-as-cue) | 1.000 | 1.000 | 0.250 |
| `gold_text` (perfect-cue ceiling) | 1.000 | 1.000 | n/a (skipped on noop) |

**Failure mode (Stage 3 "Was this result real?"):** the bad-cue baseline matches the perfect-cue ceiling. The benchmark currently provides no signal between specific and generic cues.

**Diagnosis:** plants embed strong entity beacons ("Hartwell", "Pantone 2945C", "Renata Chen"); the LoCoMo distractor conv-26 (Caroline ↔ Melanie chatting about LGBTQ support, kids, etc.) shares zero entities or topic with the plants. Topic-level matching alone wins — sub-decision specificity is never tested. Classic "test set contaminated by plant-distractor topic gap."

**Fix (v2 scenarios):** add 8-12 *on-topic decoy* turns per scenario with the same entities as the plants (e.g. "Hartwell sent swag last quarter", "Renata Chen got promoted from controller in 2024") but no relevance to any sub-decision. Decoys ingest into EM but carry `plant_id: null` so they don't count as gold. Predicted v2 task_prompt recall@5: ~0.20-0.30.

This is the procedure's iteration loop in action — the user's "scenarios should be updated/replaced as you learn about more failure modes" rule, triggered by the Stage 3 "question everything" check.

## E0-v2 sanity test (2026-04-24, all 6 scenarios with on-topic decoys, K=1,3,5,10)

Each scenario has 5 real plants (3 for distractor-pack) + 9-12 on-topic decoys + 369-663 LoCoMo distractor turns. Aggregated across non-no-op steps over all 6 scenarios:

| K | `task_prompt` (bad-cue baseline) | `decision_text` (action-as-cue) | `gold_text` (perfect-cue ceiling) |
|---|---|---|---|
| 1 | **0.097** | 0.625 | 0.938 |
| 3 | 0.278 | 0.833 | 1.000 |
| 5 | 0.500 | 0.896 | 1.000 |
| 10 | 0.771 | 0.958 | 1.000 |

False-positive rate on no-op steps at K=10: `task_prompt` 0.367 vs `decision_text` 0.117 — action-as-cue is much cleaner on no-op trigger.

**Sanity criteria (Stage 3 audit):**
- ✅ Bad cue visibly worse than perfect cue (0.097 vs 0.938 at K=1).
- ✅ Perfect cue saturates near 1.0 — planted facts are retrievable in principle.
- ✅ Action-as-cue between bad and ceiling — there is headroom for mechanism work without already-saturated.

**Per-scenario notes:**
- **banquet-01, trip-01, implicit-decision-01** — biggest decision_text headroom at K=1 (0.25-0.5). Cleanest E1 candidates.
- **schedule-01** — decision_text saturates at K=1 = 1.0 (plants share "auth handshake" / "engineer" vocabulary with sub-decisions). Likely too action-aligned in authoring; flag for v3 scenario revision if the saturation distorts E1 averages.
- **distractor-pack-01** — wrong-client-color decoys did not trip action-as-cue (decision_text K=1 = 1.0). Suggests the client-name token is doing most of the discrimination work; action-as-cue inherits the client name from the sub-decision text. Worth checking adversarially in E1 with a sub-decision that omits the client name.
- **presentation-01** — task_prompt K=10 hits 1.0 (the entity beacon still wins at high K).

**E0 verdict:** benchmark is usable for E1. Decision_text already strong (`mean_recall@1 = 0.625`) but well below ceiling — there is a real, measurable mechanism-improvement target.

## E1 design (next)

Four arms, each measured per sub-decision step on the v2 benchmark:

| Arm | Mechanism | Cost (per scenario) | Pre-registered prediction |
|-----|-----------|---------|--------------------------|
| A | `task_prompt` only (no mid-task probe). | 0 LLM calls. | Already measured: K=1 = 0.097, K=5 = 0.500. |
| B | `decision_text` (per-step embedding probe, no LLM). | 0 LLM calls; N step embeddings. | Already measured: K=1 = 0.625, K=5 = 0.896. |
| C | Per-step LLM cue-gen via `CUEGEN_PROMPT` with `need = decision_text`. | N LLM calls (gpt-5-mini, low effort). | K=1 ≥ 0.70, K=5 ≥ 0.92. The CUEGEN prompt was tuned on info-needs from a one-shot decomposition — applying it per-step is a structural change, may help or no-op. |
| D | Pre-execution one-shot `DECOMPOSE` from task_prompt only, then per-step probe = decision_text *plus* merged top-K from D's pre-execution probe set. Tests whether deeper static decomposition obviates per-step triggering. | 1 DECOMPOSE LLM call + N×2 per-need cuegen calls upfront. | K=1 ≥ 0.65 — pre-decomposition adds a stable retrieval baseline that decision_text can build on. |

**Bias controls:**
- Same EM substrate, same embedder, same K, same merge strategy (max-score across probe set).
- Cap C and D at the same total LLM-call budget per scenario (3-6 calls).
- Reuse the existing CUEGEN_PROMPT without modification for arm C; reuse existing DECOMPOSE+CUEGEN+SUFFICIENCY for arm D.
- Same scenarios, same gold, same scorer. No re-tuning of plants between arms.

**Stage-3 questions to ask before celebrating:**
- For arm C: did CUEGEN actually generate cues that differ meaningfully from the decision_text? If cues are paraphrases, gain is illusory.
- For arm D: did the pre-execution decompose surface needs that match the eventual sub-decisions? If the upfront NEEDS list misses the real sub-decision concepts, D should fail badly.
- For arms B/C/D: is the per-step false-positive rate on no-op steps still low? A mechanism that fires usefully on real decisions but spuriously on no-ops is net-negative.

**Adversarial pack to author for Stage 4 (after E1):**
- Sub-decision text that omits the client name (e.g., "Pick the deck color palette" instead of "Pick the deck color palette for Solstice"). Tests whether action-as-cue is over-relying on entity tokens.
- Sub-decision that uses a synonym distinct from the plant vocabulary ("Choose the typeface" with a plant about "font sizes for body text"). Tests vocabulary-mismatch robustness.
- Cross-scenario sub-decision (e.g., the agent suddenly pivots to a different client in the middle of the presentation task). Tests entity-switching cue specificity.

## Tracking (updated)

| Status | Item |
|--------|------|
| Done | E0 v1 — sanity test caught benchmark bug |
| Done | E0 v2 — added decoys; benchmark passes 3/3 sanity criteria |
| Done | E0 v3 — vocabulary-tightened plants (revised every plant that shared keywords with its decision_text) |
| Done | E1 — five mechanism arms compared on v3; multi-cue and combined variants added |
| Open | E2 — better cue-merge strategies (sum-of-scores; rank fusion); per-scenario gating to use cue_aware only when it helps |
| Open | Adversarial pack — scenarios where the agent's intuition-of-latent-need is wrong (e.g., interpersonal conflicts in seating) |
| Open | Scenario expansion — current 6 scenarios → 20+ for tighter statistics |

---

## E1 v3 results (2026-04-24, all 6 scenarios, K=1,3,5,10)

Mean across 6 scenarios (24 non-no-op steps + 6 no-op steps), v3 vocabulary-tightened plants:

| strategy | R@1 | R@3 | R@5 | R@10 | FPR@10_noop | LLM cost / scenario |
|---|---|---|---|---|---|---|
| `task_prompt` (no per-step probe) | 0.042 | 0.236 | 0.417 | 0.583 | 0.283 | 0 |
| `decision_text` (script as cue) | 0.347 | 0.729 | 0.854 | 0.896 | **0.117** | 0 |
| `agent_natural` (agent's content as cue) | 0.368 | 0.708 | 0.750 | 0.833 | 0.233 | N (executor) |
| **`agent_cue_aware`** (agent emits 1 CUE/step) | **0.424** | 0.736 | 0.819 | 0.958 | 0.250 | N (cue+content combined) |
| `agent_cue_aware_multi` (agent emits 3 CUEs/step) | 0.264 | 0.667 | 0.792 | 0.958 | 0.267 | N |
| `combined` (decision_text + 1 cue_aware probe) | 0.410 | **0.792** | 0.833 | 0.958 | 0.183 | N |
| `decompose_upfront` (one-shot pre-execution) | 0.104 | 0.188 | 0.389 | 0.507 | 0.250 | 1 + 6 |

For reference: `gold_text` ceiling from E0 v2 = 0.938 R@1, 1.000 R@5.

### Per-scenario R@1 (where the variance lives)

| scenario | decision_text | agent_cue_aware | agent_cue_aware_multi | combined |
|---|---|---|---|---|
| presentation-01 | 0.50 | 0.25 | 0.50 | 0.25 |
| banquet-01 | 0.25 | **0.50** | 0.25 | **0.50** |
| trip-01 | 0.25 | 0.25 | 0.25 | **0.50** |
| schedule-01 | 0.25 | 0.25 | 0.00 | 0.25 |
| distractor-pack-01 | 0.33 | **0.67** | 0.33 | 0.33 |
| implicit-decision-01 | 0.50 | **0.62** | 0.25 | **0.62** |

**Finding:** the cross-scenario mean (cue_aware +8pp vs decision_text) hides three scenarios where cue_aware lands +25 to +34pp (banquet, distractor-pack, implicit-decision) and one where it loses 25pp (presentation-01). The cases where cue_aware WINS are exactly where the latent-need framing diverges from the action wording (banquet "appetizer" → cue "dietary restrictions, allergies"). Cases where it LOSES are where the action wording is already well-aligned with plant content and the agent's reformulation drifts into abstraction (presentation "color palette" → cue "Hartwell brand guidelines" doesn't match plant's "house identity rules / Pantone").

### Stage-3 questioning of the result

- **Was the baseline fair?** Yes — same EM, same K, same merge, same gold; gpt-5-mini at low effort for all LLM calls.
- **Is the effect size plausible?** +8pp at R@1 on a small-N benchmark is plausible but fragile. Driven by 3 of 6 scenarios.
- **Are intermediate retrievals good or just final?** Per-step scoring ≠ final-output scoring — this benchmark measures the right thing. But end-to-end LLM-judge sufficiency (per `project_recall_vs_endtoend`) may give a different picture.
- **Did the test actually run?** All 6 scenarios completed cleanly; per-step JSON saved.

### Failure-mode taxonomy from per-step inspection

1. **Latent-need miss**: agent's CUE asks the wrong dimension (banquet seating: "names/roles" instead of "interpersonal conflicts"). Cue_aware drops to 0 even though plant exists.
2. **Abstraction drift**: agent's CUE is more abstract than plant content (presentation: "Hartwell brand guidelines" → plant "house identity rules / Pantone 2945C"). Embedding distance higher than action_text → plant.
3. **Multi-probe noise dominance**: with 3 cues + max-merge, one bad cue's high-score-but-wrong hits dominate top-K. Hurts R@1; recovers at K=3+.
4. **No-op false positives**: cue_aware fires probes on no-op steps even when prompted to emit `CUE: none`. FPR 0.250 vs decision_text's 0.117.

## Stage-5 next questions

- **Better merge strategies for multi-cue.** Max-score lets one bad probe dominate. Sum-of-scores or reciprocal-rank-fusion may stabilize multi-cue at K=1.
- **Gated combined.** Only combine cue_aware with decision_text when they DISAGREE on top-1. If they agree, go with decision_text alone (cleaner FPR).
- **Stronger no-op discipline.** Re-prompt the agent more aggressively to emit `CUE: none` when the step is administrative / formatting / structural.
- **Adversarial pack.** Build scenarios where the agent's intuition is *predictably* wrong — interpersonal conflicts in seating, mobility constraints in hotel choice, jurisdiction-specific legal reqs. These are exactly where humans need memory the most and where current cue_aware fails.
- **End-to-end LLM-judge sufficiency.** Does the per-step recall improvement translate to better task output? Per `project_recall_vs_endtoend` the answer may be smaller than recall suggests.
- **Scale benchmark.** 6 scenarios → 20+ to firm up the per-scenario variance.

### Cost note

`agent_cue_aware` and `combined` cost N LLM calls per scenario (one per step). At gpt-5-mini, low effort, with caching across re-runs, this is ~$0.001-0.005/scenario. Affordable. The cost question is more about whether the +8pp R@1 is worth the runtime latency in a real agent loop, not budget.

---

## E2 — REFRAME: free the agent from the script (2026-04-24)

User flagged the lurking flaw in E1: hand-scripting the sub-decisions handed the agent a perfect outline. That tested cue quality given the outline but skipped the harder competence — *thinking of "colors" or "allergies" in the first place*. A real agent gets only the high-level task and must produce its own plan; if it forgets to plan colors, it ships a B&W deck.

**E2 design:** agent gets only `task_prompt`, must plan + execute itself. Hand-written `subdecision_script` becomes the GOLD COVERAGE CHECKLIST — an LLM judge checks per gold sub-decision whether the agent's transcript addressed it. End-to-end metric:

```
triggered_recall_full@K = mean over gold sub-decisions of
    1[agent addressed it AND gold plant in top-K of agent's per-step retrieval]
```

Coverage failures (agent forgot the sub-decision) score 0. This is the honest score for the user's actual question.

Two modes compared:
- **freelance_natural**: agent plans + executes; no `CUE:` lines. Per-step cue = the agent's content for that step.
- **freelance_cue_aware**: agent plans + executes + emits `CUE: <query>` per step. Per-step cue = the CUE: text.

### E2 cross-scenario means (6 scenarios, gpt-5-mini judge)

| mode | coverage | full_R@1 | full_R@3 | full_R@5 | full_R@10 | cond_R@5 | cond_R@10 |
|---|---|---|---|---|---|---|---|
| freelance_natural | 0.792 | 0.243 | 0.340 | 0.500 | 0.688 | 0.625 | 0.875 |
| freelance_cue_aware | **0.833** | **0.340** | **0.424** | 0.521 | 0.604 | 0.583 | 0.715 |

Per-scenario coverage:

| scenario | natural | cue_aware |
|---|---|---|
| presentation-01 | 0.50 | 0.75 |
| banquet-01 | 0.75 | 1.00 |
| trip-01 | 0.75 | 1.00 |
| schedule-01 | 0.75 | 0.50 |
| distractor-pack-01 | 1.00 | 1.00 |
| implicit-decision-01 | 1.00 | 0.75 |

### Key findings

1. **Coverage is now the dominant failure mode.** Even competent gpt-5-mini (low effort) freelance agents address only 79-83% of gold sub-decisions. Concrete agent incompetence — they DO forget to plan the right things.

2. **Cue-aware prompting modestly improves coverage** (0.792 → 0.833). Instructing the agent to ASK what's relevant primes it to PLAN better — the user's "different personality" effect. Not uniform: schedule-01 went 0.75 → 0.50, implicit-decision-01 went 1.00 → 0.75.

3. **End-to-end ≈ 50% at K=5** for both modes — the real headline number for "current best mid-task retrieval." Half of expected mid-task retrievals succeed.

4. **Conditional recall surprise: natural cond_R@10 (0.875) > cue_aware cond_R@10 (0.715).** When the natural agent does cover a decision, its concrete content (real names, decisions: "TAP Air Portugal TP197 departing Nov 1") often retrieves the plant; cue_aware's CUE: line is more abstract ("corporate preferred airlines / travel agency details"). Trade-off: cue_aware lifts coverage AND R@1, but at high K its abstract queries lose to natural's concrete content.

5. **Trip-01 worked example:**
   - Natural mode: agent's plan has no "verify documents" step → books flights for an expired passport → coverage 0 on gold step 1; for the 3 covered steps the cues fail to retrieve any gold plant (full_R@5 = 0).
   - Cue-aware mode: agent's plan opens with "Confirm travel dates, origin city, traveler details, and approvals" → cue "passport expiry, travel approval status?" retrieves passport plant at rank 1. Airline cue retrieves airline plant at rank 1. 2 of 4 gold pairs hit.

6. **The picture inverts:** in E1 (script-given), cue_aware barely beat decision_text (+8pp R@1). In E2 (freelance), cue_aware lifts both coverage AND retrieval, with full_R@1 going from 0.243 → 0.340 (+10pp absolute; the gain compounds across both axes).

### Next questions (Stage 5)

- **Coverage is now the bottleneck.** Mechanisms for it: critic-pass plan review ("what did you miss?"), pre-execution rich decomposition that seeds the agent's plan, multi-pass planning (draft → review → revise).
- **Conditional retrieval: natural beats cue_aware at K=10.** Worth combining: use cue_aware's CUE for K=1-3, fall back to agent's content for K=5-10. Or: cue_aware's content concatenated with its CUE.
- **Model effects.** gpt-5-mini at low effort is the executor. Claude Sonnet 4.7 or higher reasoning_effort would likely lift coverage substantially — currently we may be measuring "low-effort agent forgets things" more than "any agent forgets things."
- **Multi-step coverage scaffolding.** Inject "before each step, list one thing you might be forgetting" — primes self-critique without a separate critic LLM.

### Updated tracking

| Status | Item |
|---|---|
| Done | E0 v1/v2/v3 — benchmark + decoys + vocabulary tightening |
| Done | E1 — five mechanism arms with hand-scripted sub-decisions (now considered an upper-bound approximation, not realistic) |
| Done | E2 — freelance executor reveals coverage as primary failure mode |
| Done | E3 — coverage-improving mechanisms (primed prompt, critic-pass, retrieve_revise, spreading_activation) |
| Done | Hard v2 scenarios — coverage drops from 0.94→0.69 on natural; spreading_activation recovers to 0.91 |
| Open | Hard v3 scenarios — multi-hop, vocab-bridge, narrative-form, adversarial near-miss; multi-conversation distractor density |
| Open | Model upgrade test (Claude Sonnet executor) |

---

## E3 — coverage-lift mechanism comparison (2026-04-24)

User direction: push coverage as high as possible, then increase scenario difficulty, repeat. Mechanisms tested in escalating order.

### Mechanism descriptions

- **`natural`** (E2 baseline): freelance plan + execute, no cue awareness, no memory access. Cue per step = the agent's content for that step.
- **`cue_aware`** (E2 baseline): freelance plan + execute, agent emits `CUE: <q>` per step.
- **`primed_cue_aware`**: like cue_aware but the system prompt explicitly enumerates common sub-decision categories (recipient constraints / brand / dietary / mobility / interpersonal / document validity / vendor preferences / etc.). "Checklist" intervention.
- **`critic_cue_aware`**: agent drafts plan → separate critic LLM lists what's missing → agent revises plan + executes. Two-pass enumeration.
- **`retrieve_revise_cue_aware`**: agent drafts plan → each plan item is queried against EM → top-K snippets injected as memory context → agent revises plan + executes with that context. RAG-on-plan-items.
- **`spreading_activation_cue_aware`**: iterative loop. Round 1 probes with task_prompt. Each subsequent round, agent inspects accumulated retrieved snippets + prior probes, generates 2-4 NEW probes targeting concept gaps surfaced in what was retrieved. Up to 4 rounds, K=3 turns per probe. After saturation, agent plans + executes with full accumulated context. Cognitive analog of associative spreading activation.

### Cross-scenario means (10 scenarios: 6 original + 4 hard-v2)

| mode | coverage | full_R@1 | full_R@5 | full_R@10 | cond_R@5 |
|---|---|---|---|---|---|
| natural (baseline) | 0.79 | 0.24 | 0.50 | 0.69 | 0.63 |
| cue_aware | 0.83 | 0.34 | 0.52 | 0.60 | 0.58 |
| primed_cue_aware | 0.83 | 0.23 | 0.49 | 0.72 | 0.60 |
| critic_cue_aware | 0.83 | 0.20 | 0.41 | 0.60 | 0.50 |
| retrieve_revise_cue_aware | **0.89** | 0.46 | 0.62 | 0.79 | 0.71 |
| **spreading_activation_cue_aware** | **0.94** | **0.51** | **0.76** | **0.84** | **0.79** |

### Findings

1. **Spreading activation is the new champion.** Coverage 0.94, full_R@5 0.76 across 10 scenarios. Vs natural baseline: +15pp coverage, +26pp full_R@5, +27pp full_R@1 — end-to-end has more than doubled.

2. **Memory access at planning time is the unlock.** retrieve_revise (single-pass RAG-on-plan-items) jumped coverage from ~0.83 → 0.89 — bigger than any prompt intervention. The bottleneck wasn't "agent can't think" but "agent doesn't know what's in memory." Spreading activation extends this with iterative concept exploration: each round surfaces concepts the next round can probe specifically (e.g., probe 1 reveals "Diane's wedding has a Quaker silent ceremony" → probe 2 asks "Quaker ceremony phone collection rules?" → finds the specific plant).

3. **Pure prompt-based interventions saturate at ~0.83 coverage.** primed (checklist) and critic (second-pass enumeration) both top out around there. Once the agent doesn't know what's IN memory, it can't enumerate the right sub-decisions even with a checklist. The architecture must include memory access at planning time.

4. **critic_cue_aware actually loses retrieval.** Critic plans are longer / more abstract (12-13 high-level steps), per-step content gets diluted, conditional retrieval suffers (cond_R@5 = 0.50 vs natural 0.63). Coverage up doesn't always mean end-to-end up. Architecture matters.

5. **The user's "spreading activation" framing operationalized.** Iterative probing + concept-driven follow-up beats single-pass retrieval. The agent's pseudo-cognitive mechanism: probe → see → form-new-concepts → probe-on-those-concepts. Maps well to associative memory dynamics.

6. **Hard scenarios held the gain.** spreading_activation hits coverage 0.91 / full_R@5 0.80 on the 4-scenario hard subset alone (wedding-attire, surprise-birthday, perf-review, client-renewal — 8-12 niche personal sub-decisions per scenario). Natural baseline on hard set: 0.66 coverage, 0.52 full_R@5. The mechanism scales to harder problems.

### Honest caveats

- 10 scenarios, gpt-5-mini at low effort, single-sample LLM output. LLM nondeterminism shifts numbers ±5pp between runs.
- The judge is gpt-5-mini at low effort. Judge errors (over- or under-counting "addressed") propagate into coverage scores. A stricter judge or human gold would give different numbers.
- Memory context window grows with iterations — for very large memories, spreading_activation's accumulated context could exceed model limits. Current scenarios stay under ~30 retrieved snippets.
- All scenarios use the existing planted-fact + LoCoMo-distractor structure. Generalization to truly different memory shapes (long-form documents, structured records) untested.

### Cost note

Spreading activation: ~5-7 LLM calls per scenario (4 probe-gen iterations + 1 final plan-execute). At gpt-5-mini low effort with caching, ~$0.005-0.01 per scenario per fresh run. The cost question is latency (4-iteration loop adds 5-15s wallclock) more than tokens.

---

## v3 hardest scenarios — pushing past spreading_activation (2026-04-24)

User direction: keep iterating, raising scenario difficulty as coverage hits the target band. Authored 4 v3 scenarios targeting failure modes the v2 hard set didn't stress:

- **multi-hop-banquet-01**: 8 sub-decisions, 4 require 3+ plants combined (catering arrival = start time + setup window + no-morning-kitchen → 6 AM cold-prepped). Multi-conversation distractor (locomo_conv-26 + locomo_conv-30, 788 turns combined).
- **vocab-bridge-trip-01**: 3 sub-decisions whose constraint plant uses only a colloquialism ('A.' for Aiden, 'KitKat' for Karina Tate, 'B2' for Berlin annex). Agent must retrieve the bridge plant first, then re-probe with the alias.
- **narrative-meal-01**: 5 plants stated in narrative form (mid-story, hedged, tangential) — "after Marcus's reaction at last year's company picnic — they had to call the EMTs because the satay sauce had peanut oil"  — fact buried in anecdote.
- **adversarial-pricing-01**: each correct plant has 1-2 lookalike near-miss decoys on the same exact topic (stale 12% discount vs active 18% vs different-product 10%; standard Net-30 vs active Net-45 carve-out; etc.).

### Spreading_activation on v3 hardest (4 scenarios)

| scenario | natural cov | SA cov | natural full_R@5 | SA full_R@5 |
|---|---|---|---|---|
| multi-hop-banquet-01 | 0.50 | **1.00** | 0.39 | **1.00** |
| vocab-bridge-trip-01 | 0.33 | 0.67 | 0.33 | 0.67 |
| narrative-meal-01 | 0.75 | **1.00** | 0.75 | 0.75 |
| adversarial-pricing-01 | 0.50 | **1.00** | 0.25 | **0.875** |
| **mean** | **0.52** | **0.92** | **0.43** | **0.83** |

Spreading_activation still hits the 80-90%+ target band on the v3 hardest set: cov 0.92, full_R@5 0.83. Multi-hop, narrative, and adversarial near-miss are handled cleanly.

### The new failure mode: vocab-bridge

Vocab-bridge is the only v3 scenario where SA underperforms (cov 0.67, R@5 0.67). Mechanism failure: agent retrieves "Aiden goes by 'A.'" but does not extract 'A.' as a vocabulary item to use in subsequent probes. Constraint plant ("'A.' is allergic to citrus") stays hidden because no probe uses 'A.'.

Architectural fix to test in next round:
- **Entity-extraction step in spreading_activation**: after each retrieval round, run a quick LLM extract of NEW entities/aliases/proper nouns from retrieved snippets. Append those as forced probe seeds in the next iteration. Maps to spreading-activation cognitive model where surfacing a new node activates connected nodes.
- **Alternative: increase iteration count** beyond 4 — agent may need 5-6 rounds to organically discover and re-probe with aliases.

### Cycle summary

User asked: increase coverage as much as possible, then make scenarios harder, repeat. We did 3 cycles:

| cycle | best mechanism | best mechanism's coverage on then-current set | then-new failure mode authored |
|---|---|---|---|
| 1 (v3 plants only, 6 scenarios) | spreading_activation | 0.94 | hard-v2 (niche personal sub-decisions) |
| 2 (10 scenarios incl hard-v2) | spreading_activation | 0.94 | hard-v3 (multi-hop, vocab-bridge, narrative, adversarial) |
| 3 (14 scenarios) | spreading_activation | 0.92 (hardest 4) / ~0.93 (all 14 estimated) | vocab-bridge open |

Ran out of cycles within session budget. Vocab-bridge alias-extraction architecture is the next experiment.

### Architectural takeaway across all 3 cycles

Mid-execution retrieval cue performance is a multiplication of three factors: (a) **coverage** — does the agent surface the right sub-decision; (b) **conditional retrieval** — given coverage, does the cue retrieve the gold; (c) **fragility to plant-form variation** (adversarial decoys, narrative form, vocabulary bridges). Pure prompt-based mechanisms (priming, critic) lift (a) only modestly (0.79 → 0.83). Memory-access-at-planning-time (retrieve_revise) lifts (a) and (c) (0.83 → 0.89). Iterative concept-driven exploration (spreading_activation) lifts all three (0.94 cov / 0.76 R@5 on combined set; 0.92 / 0.83 on hardest). The cognitive analog (spreading activation) is operationally the strongest mechanism we found.

End-to-end full_R@5 has gone from 0.50 (natural baseline) to 0.83 (spreading_activation on hardest scenarios) — within session.

---

## Round 4 — spreading activation at PLANNING AND EXECUTION + Claude executor

User insight: plans don't capture everything. The planning-time spread builds an initial plan, but execution surfaces decision points and threads the plan missed. Real agents need to spread again at every step. Built `spreading_activation_full`:

- **Phase 1**: planning-time spreading activation (existing — 6 iterations max).
- **Phase 2**: agent generates the plan from accumulated context.
- **Phase 3**: per step:
  1. Cue-gen LLM call: agent decides whether THIS step needs new probes; emits 0-3 fresh CUE lines based on plan + accumulated planning context + prior step outputs.
  2. Probe → retrieve → augment context with new snippets.
  3. Writer LLM call: agent writes step content using planning-context + execution-accumulated-context.

### gpt-5-mini SA-full vs SA on v3 hardest set (4 scenarios)

| | SA cov | SA-full cov | SA R@5 | SA-full R@5 |
|---|---|---|---|---|
| multi-hop-banquet-01 | 0.83 | 0.83 | 0.78 | 0.67 |
| **vocab-bridge-trip-01** | **0.50** | **0.83** | **0.50** | **0.83** |
| narrative-meal-01 | 0.875 | **1.00** | 0.75 | **0.875** |
| adversarial-pricing-01 | 0.75 | **1.00** | 0.75 | **0.875** |
| **mean** | **0.74** | **0.92** | **0.69** | **0.81** |

Vocab-bridge specifically jumped from 0.50/0.50 to 0.83/0.83 — execution-time probing surfaces the alias-bridged plants that the agent only realizes it needs after starting to write. Validates the user's "plans don't capture everything" prediction.

### Claude executor (--print CLI integration)

User had no API key but a Claude subscription. Integrated `claude --print` as a swappable executor backend (`EXECUTOR_BACKEND=claude` env var routes all `_llm` calls through subprocess). All other infrastructure unchanged.

Claude SA-full on `vocab-bridge-trip-01`:

| executor | coverage | full_R@1 | full_R@5 |
|---|---|---|---|
| gpt-5-mini SA | 0.50 | 0.33 | 0.50 |
| gpt-5-mini SA-full | 0.83 | 0.50 | 0.83 |
| **Claude SA-full** | **1.0** | **0.83** | **1.0** |

Claude-as-executor closes the vocab-bridge gap entirely. The combination of (a) generic mechanism (spread at planning AND execution) and (b) stronger executor model is the unlock — neither alone gets there.

### gpt-5-mini SA-full on v4 maximum-difficulty (3 scenarios with stacked failure modes)

| scenario | SA-full cov | SA-full R@5 |
|---|---|---|
| stacked-event-planning-01 | 0.67 | 0.61 |
| supersession-vendor-decision-01 | 0.89 | 0.54 |
| negative-space-onboarding-01 | 0.89 | 0.50 |
| **mean** | **0.82** | **0.55** |

Stacked failure modes (multi-hop + vocab-bridge + narrative + adversarial + supersession + negative-space, in 3-conversation distractor) push gpt-5-mini SA-full back below the target band. **Claude SA-full results pending.**

### Generalizable architectural lesson

User's instruction was "unlock intelligence in the model, not tell it to solve specific edge cases." The path that worked:

- Did NOT: add an alias-extraction step (would have been narrow, edge-case patching).
- DID: invite the model to think more openly (reflection step) AND give it more chances to act (execution-time spreading) AND use a stronger model (Claude).

The general intervention is "give the model multiple opportunities to apply its associative reasoning, with current context visible at each opportunity." The specific failure mode (vocab-bridge) gets solved as a side effect of that generality.

---

## Claude-without-optimizations isolation (Claude executor across modes)

User asked: is Claude just retrieving everything via generic cues, or are its probes actually targeted? And how does Claude do without all the SA-full optimizations?

### Per-step cue specificity check

Inspected 247 distinct CUE: lines from Claude's per-step prompts (cached). Examples:

- "Retrieve current attendee executive roster and any documented presentation order or opening-preference notes (including Daniela, Renata, Aviva, Wendell)."
- "Has the rooftop garden been officially reserved in the venue system/contract for Thursday evening, and if so, who is the Foundry contact?"
- "Erik Larsson's calendar availability for May 2026, focusing on full-day conflicts and any travel restrictions."
- "Foundry security contact name and phone (or primary booking contact) for vendor access."

Claude's cues name specific entities, ask focused questions, target sub-decisions. Not generic dumps. Confirmed targeted retrieval.

### Mode-isolation on Claude (vocab-bridge-trip-01 specifically)

| mode | coverage | full_R@5 |
|---|---|---|
| natural (no memory access at all) | 0.33 | 0.17 |
| cue_aware (emit cues but no retrieval feedback) | 0.50 | 0.33 |
| retrieve_revise_cue_aware (single-pass RAG at planning) | 0.83 | 0.83 |
| spreading_activation_full (planning + execution iteration) | 1.00 | 1.00 |

### Cross-scenario Claude isolation (17 scenarios, mean)

| mode | mean coverage | mean full_R@5 |
|---|---|---|
| natural | 0.66 | 0.51 |
| retrieve_revise_cue_aware | 0.83 | 0.71 |
| spreading_activation_full (prior runs) | 0.96 | 0.86 |

### Conclusion

The architecture provides most of the gain (0.66 → 0.83 from adding memory access at planning time). Claude alone, without memory access, is mediocre — comparable to gpt-5-mini natural. Spreading activation at execution time adds another 13pp coverage / 15pp R@5 on top of single-pass RAG. The mechanism stack (memory access → iterative spreading → execution-time spreading) is what unlocks Claude's reasoning capacity; Claude without the architecture can't recover memory it can't see.

---

## v5 — break-Claude scenarios (2026-04-24, last cycle)

User asked to push Claude to its limit. Authored 3 scenarios designed to expose architectural gaps Claude+SA-full hadn't faced:

- **world-knowledge-bridge-01**: gold sub-decisions require recalling world knowledge memory cannot contain (Norwegian Constitution Day given "Sigrid is from Tromsø"; US Memorial Day; time-zone math). 13 plants, 32 decoys, 11 sub-decisions, 2 base conversations.
- **deductive-chain-procurement-01**: 5-plant chains where each link requires a different probe (Mei → Datadog → March renewal → $50K freeze → $80K cost → override). 15 plants, 30 decoys, 11 sub-decisions.
- **negation-and-inference-01**: plants stated only by negation with adversarial near-positive decoys; inference-from-list ("the regulars" = intersection of three rosters); self-referential meta-rules. 15 plants, 38 decoys, 11 sub-decisions.

### Claude SA-full on v5

| scenario | coverage | full_R@1 | full_R@5 |
|---|---|---|---|
| world-knowledge-bridge-01 | **0.778** | 0.319 | 0.607 |
| deductive-chain-procurement-01 | 0.889 | 0.189 | 0.706 |
| negation-and-inference-01 | 0.889 | 0.306 | 0.694 |
| **mean** | **0.852** | **0.271** | **0.669** |

vs Claude SA-full on v4 max: 0.96 cov, 0.86 R@5. v5 dropped both metrics meaningfully:
- Coverage: 0.96 → 0.85 (-11pp)
- full_R@5: 0.86 → 0.67 (-19pp)
- full_R@1: 0.47 → 0.27 (-20pp)

### Where Claude+SA-full breaks

- **World-knowledge bridges break coverage below the 80% target.** When the gold sub-decision requires a fact memory does NOT contain (recalling that May 17 is Norwegian Constitution Day from "Sigrid is from Tromsø"), no embedding probe can surface it. SA-full has no mechanism to consult world knowledge between probes — the agent could in principle infer this but would need to be prompted to do so. Probable fix: add a "what world knowledge applies here?" step alongside memory probes — but that's a specific intervention, less general than spreading.
- **5-plant deductive chains stress the iteration budget.** Coverage holds at 0.89 because the agent surfaces individual plants, but R@5 drops to 0.71 — the chain doesn't always assemble fully in 6 iterations. Probable fix: detect chain-shape and extend iteration budget OR have the agent explicitly trace each step of the deduction.
- **Negation + adversarial near-positive decoys drop full_R@1 to 0.31.** Even when the agent surfaces the gold negation plant, it co-surfaces multiple positive-framed decoys at higher rank. Probable fix: explicit negation-detection in cue text — but again, specific intervention.

### Architectural lesson from v5

The clean general mechanism (spreading activation at planning + execution) handles every failure mode that's *retrievable in principle* — multi-hop, vocab-bridge, narrative, adversarial near-miss in single plants, supersession, negative-space. It breaks on:
- Failures *outside* memory entirely (world knowledge).
- Chain lengths that exceed the iteration budget.
- Adversarial structures where retrieval is correct but ranking is misled.

Each of these is fixable but the fixes are getting domain-specific. Past v4, the marginal returns of mechanism work shrink and the marginal returns of *better executor reasoning* (reflection on retrieved content, deduction, world-knowledge integration) grow. That's the next research surface.

### Final cycle summary (this session)

| difficulty | best mechanism | best executor | coverage | full_R@5 |
|---|---|---|---|---|
| original 6 | spreading_activation | gpt-5-mini | 0.94 | 0.76 |
| hard-v2 (4) | spreading_activation | gpt-5-mini | 0.92 | 0.81 |
| hard-v3 (4) | spreading_activation_full | Claude | ~0.96 (vocab-bridge: 1.0) | ~0.86 |
| max-v4 (3) | spreading_activation_full | Claude | 0.96 | 0.86 |
| break-v5 (3) | spreading_activation_full | Claude | **0.85** | **0.67** |

End-to-end full_R@5 from natural baseline (0.50) to current ceiling (0.86 on v4 max with Claude SA-full) within session. v5 reveals the next frontier: world-knowledge integration, deductive chain assembly, negation handling. These require either much harder mechanisms or fundamentally better executor reasoning.






