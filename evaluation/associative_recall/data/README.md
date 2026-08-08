# Associative-Recall Evaluation Data

This directory holds scenario corpora for the associative-recall evaluation harness.

## `perpetual_scenarios.json`

Two scenarios (both kind `stream_perpetual`) designed for the perpetual-execution research loop:

1. `extended-project-stream-01` — solo indie gamedev (Hollowmark)
2. `consulting-engagement-stream-01` — solo strategy consultant engaged by a B2B SaaS startup (Lattice Health)

The second scenario was added as a structural sibling of the first, to test cross-domain generalization of the revisit (deterministic harness scheduler) mechanism that lifted DP coverage from 3.25 → 4.25 on Hollowmark at n=4. The question being asked: does the +1.0 lift hold on a different scenario, or is it Hollowmark-specific?

### Why a new file

Existing `streaming_scenarios.json` tops out at 15 messages. Even on its hardest case the system can hold the entire conversation in working memory without compaction, so it does not test the architectural pivot to perpetual execution where compaction-as-sleep (LLM-driven extraction-and-summarize when working memory approaches cap, with extracted facts written to EventMemory for long-term retrieval) is the load-bearing mechanism.

### Domain

Solo indie game development on a procedurally-generated roguelike codenamed **Hollowmark**, spanning roughly five months of pre-EA work. Domain was chosen to be disjoint from existing scenarios (banquet / presentation / wedding / dinner / trip / family / wedding-planning / personal-finance-app) and to support a believable extended timeline with naturally recurring constraints.

### Structure

- **40 user messages** (turn 1 → turn 40).
- **10 decision points**, distributed at turns 7, 10, 15, 20, 26, 28, 31, 34, 35, 39.
- **8 ground-truth facts**, of which 6 originate in turns 1-10 (the compaction-survival set).
- **8 distractor facts** spanning life-updates, hobby asides, and substantive-but-unbinding decisions (the cooldown-vs-stamina log).
- **Message-type mix**: ~40% substantive constraint, ~20% agent status updates from prior sprints, ~20% distractor / small-talk, ~25% decision-trigger (decision-trigger turns total 10/40 = 25%).

### Compaction stress test

The deliberately-engineered architectural pressure: 6 of the 10 decision points trigger after turn 25 and require at least one fact from the first 10 turns. Specifically:

- Turn 26 (Steam "About this game" copy) ← needs facts from turns 3, 5, 9
- Turn 28 (push back on snowy second biome) ← needs fact from turn 5
- Turn 31 (boss-art budget advice) ← needs facts from turns 8 and 25
- Turn 34 (accessibility QA checklist) ← needs fact from turn 9 (lag = 25)
- Turn 35 (Mira contract reply) ← needs fact from turn 6 (lag = 29, longest-lag canonical test)
- Turn 39 (pre-launch checklist) ← needs facts from turns 5, 6, 8, 9, 19 — five-fact composition spanning up to 34-message lag

A hard-truncate working-memory window much smaller than 35 turns will fail these by construction. A compaction-based system that extracts facts on sleep boundaries and writes them to EventMemory should retrieve them.

### Lag distribution

| Bucket | Lag range | Count |
| ------ | --------- | ----- |
| Easy | 1-3 | 2 |
| Medium | 5-10 | 4 |
| Hard | 15-30 | 10 |
| Extreme | 30+ | 3 |

Mean lag: 18.3 messages. The distribution intentionally weights toward hard / extreme because that is where the existing benchmark has zero coverage.

### Multi-fact composition

6 of 10 decision points compose two or more facts from different earlier turns. Two notable cases:

- **Turn 26 Steam copy**: visual identity (turn 3) + accessibility commitments (turn 9) + scope-of-one positioning (turn 5).
- **Turn 39 pre-launch checklist**: scope-of-one (turn 5) + audio contract terms (turn 6) + budget ceiling (turn 8) + accessibility (turn 9) + marketing window (turn 19). Five facts, max 34-message lag — the headline test of the scenario.

### Schema

Extends the `streaming_scenarios.json` schema with `kind: "stream_perpetual"` and is otherwise compatible (`messages`, `decision_points`, `ground_truth_facts`, `distractor_facts`).

### Open design questions

- The perpetual scenario assumes the harness exposes a clean compaction trigger; the trigger threshold is not encoded here.
- Distractor density (~20%) is inherited from `wedding-planning-stream-01`. It may be worth probing whether a higher distractor ratio further separates compaction-based from naive-truncation strategies.

## `consulting-engagement-stream-01`

Sibling scenario in a disjoint domain (B2B SaaS strategy consulting). Built to be structurally comparable to Hollowmark so the revisit mechanism's +1.0 lift can be tested for generalization rather than domain-overfit.

### Domain

A solo strategy consultant runs a ~6-month engagement with a healthcare-scheduling B2B SaaS startup ("Lattice Health"). Stakeholders are CEO Pat Lindgren, CTO Riley Okafor, CFO Casey Iwasaki, Head of Sales Jordan Mercier, Board Chair Avery Tan, and compliance officer Sam Petrov. Names are deliberately neutral per `feedback_prompt_examples_generic`.

Domain chosen because it naturally generates the same structural shape as Hollowmark:

- Phase milestones with hard deadlines (discovery → strategy → exec-ready proposal → board pitch).
- Multi-stakeholder constraint composition (CEO NDA + CTO technical no-go + CFO budget cap + compliance gate + confidentiality posture).
- Distractor messages from real-life consultant workflow (dental surgery, sailing, jackhammer neighbor, methodology decision logs).
- A late deliverable (the board-pitch checklist at turn 39) that has to reflect every early-locked constraint.

### Structure

- **40 user messages** (turn 1 → turn 40), spanning ~6 months.
- **10 decision points** at turns 7, 10, 15, 20, 26, 28, 31, 34, 35, 39 (same pacing as Hollowmark).
- **8 ground-truth facts**, of which 6 originate in turns 1-10 (the compaction-survival set).
- **8 distractor facts** spanning life-updates, hobby asides, and substantive-but-unbinding decisions.

### Compaction stress test

6 of the 10 decision points trigger after turn 25 and require at least one fact from the first 10 turns:

- Turn 26 (exec-summary proposal section) ← turns 2, 3, 9
- Turn 28 (engine-rewrite pushback) ← turn 5
- Turn 31 (interim CRO budget advice) ← turns 6, 25
- Turn 34 (compliance pre-review checklist) ← turn 8 (lag = 26)
- Turn 35 (Pat confidentiality reply) ← turn 9 (lag = 26)
- Turn 39 (pre-board-pitch checklist) ← turns 2, 3, 5, 6, 8, 9, 19 — seven-fact composition spanning up to 37-message lag

### Lag distribution

| Bucket | Lag range | Count |
| ------ | --------- | ----- |
| Easy | 1-3 | 2 |
| Medium | 5-10 | 2 |
| Hard | 15-30 | 9 |
| Extreme | 30+ | 5 |
| Boundary (lag=4) | 4 | 4 |

Mean lag: 20.0 messages (vs Hollowmark's 18.3). Max lag: 37 (vs Hollowmark's 34). Slightly heavier-tailed by design — the consulting domain's seven-fact final checklist drives more long-lag pairs.

### Multi-fact composition

6 of 10 decision points compose two or more facts from different earlier turns:

- **Turn 26 exec summary**: NDA (turn 2) + scope-lock (turn 3) + stakeholder confidentiality (turn 9).
- **Turn 39 pre-board-pitch checklist**: NDA + scope-lock + engine-no-rip-replace + budget ceiling + compliance gate + stakeholder confidentiality + board timeline. Seven facts, max 37-message lag — the headline test.

### Differences from Hollowmark (deliberate diversity)

- **Constraint nature**: Hollowmark's constraints are mostly self-imposed (the user is the maker); Lattice Health's constraints are externally-imposed by stakeholders (NDA, CTO veto, CFO ceiling, compliance officer). This stresses entity-binding (which stakeholder said what) on top of fact recall.
- **Final-deliverable fan-in**: Hollowmark's pre-launch checklist composes 5 facts; the consulting pre-board-pitch checklist composes 7. Wider fan-in tests retrieval recall under harder composition.
- **Stakeholder confidentiality fact**: turn 9 introduces a non-attribution rule that is structurally a constraint on how facts are surfaced (no `Casey said X` quoting), not just on what facts are retrieved. The turn 35 DP tests whether the agent both remembers the rule and applies it conversationally.
- **Naming convention**: neutral characters per project memory; Hollowmark had a single named external contractor (Mira), Lattice Health has 5 named stakeholders (Pat / Riley / Casey / Jordan / Avery / Sam), which raises entity-tracking pressure.

### Comparable structural properties (deliberate parallel)

- 40 messages, 10 DPs, same DP pacing.
- 6 compaction-stress DPs, 6 multi-fact-composition DPs.
- 8 ground-truth + 8 distractor facts.
- Boundary-lag canonical compaction test (turn 9 fact recalled at turn ≥34).
- Headline-test final deliverable (pre-launch / pre-board-pitch checklist) composing all major early constraints.
- ~40% constraint / ~20% status-update / ~20% distractor / ~25% decision-trigger message mix.

### Open design questions (sibling-scenario)

- The seven-fact composition at turn 39 may saturate near the upper bound of what any architecture can retrieve in one pass. If both Hollowmark and Lattice score similarly low here, the headline test isn't discriminating — may need to relax to 4-5 facts in a future sibling.
- The turn 35 confidentiality reply is a `meta-constraint` test (rule-about-rules) rather than a fact-recall test; it's structurally novel relative to Hollowmark's turn-35 Mira-contract reply. Could split into easier/harder variants if it turns out to be dominated by judge-style assessment rather than retrieval.
- Generalization conclusion will need ≥3 scenarios in disjoint domains; this is the second. A third (e.g. year-long apartment search, multi-month fitness program) would strengthen the inference.
