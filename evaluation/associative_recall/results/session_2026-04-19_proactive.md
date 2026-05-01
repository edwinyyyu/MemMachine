# Session 2026-04-19 — Proactive memory, generalization, and benchmark limits

Continuation of EventMemory-backed research. Focus: ingestion augmentation beyond speaker baking, model portability, reflective/iterative architectures, and proactive-memory for arbitrary tasks.

## Major findings

### 1. Cross-model portability achieved on EventMemory

Prior session found nano at 81% of mini. That was SegmentStore + specific spec + verify-repair.

On EventMemory with structural prompts (speakerformat, HyDE first-person):
- `nano + v2f` matches `mini + v2f` EXACTLY (0.742/0.883)
- `nano + hyde_first_person + speaker_filter` = 0.800/0.933 = **99.1% of mini's ceiling**
- 100% format compliance; zero retries

**Why it flipped**: prior failure was SS + complex spec; mini's speakerformat is a CLEANER structural constraint that nano can satisfy. EM's strong cosine baseline absorbs what free-form v2f would require.

**Implication**: production recipe is model-portable. gpt-5-nano at 1/6 cost-per-call of mini is viable for the cue-gen step.

### 2. Turn-summary dual-view indexing — major LoCoMo K=20 win, BUT corpus-dependent

At ingest, LLM generates 1-sentence summary per turn; both raw turn + summary are embedded as separate entries pointing to same turn_id.

- **LoCoMo K=20: 0.908** (prior ceiling 0.867) — +4.2pp new record
- **LoCoMo K=50 (with speaker_filter): 0.9417** — matches prior ceiling at cheaper cost
- **Zero-query-LLM variant at K=20: 0.850** — matches prior HyDE+filter K=20 with no query-time LLM
- 83-100% of gold credits go via summary view — summary is doing the work

**LME port FAILED**: LME turns are long-form (code, explanations). Summarization COMPRESSES useful detail. Slight regression (-0.5pp).

**Refined principle**: summary-view helps when turns are short-chat-register that queries don't cosine-match naturally. Fails on long-form informative turns where summaries lose specificity.

### 3. Topic-baking at ingest — modest LoCoMo win

- em_v2f_topic: 0.833/0.933 (+9.2pp R@20, +5pp R@50 vs plain em_v2f)
- em_topic_plus_speaker_filter: 0.867 (new K=20 record BEFORE turn_summary surpassed it)
- Does NOT stack with turn-summary — captures overlapping signal
- Retune principle FALSIFIED for topic prefix: high-variance prefixes hurt when cue format is forced to match

### 4. Reflective memory — works on non-saturated benchmarks only

- LoCoMo: reflmem ties ceiling (0.942 K=50) — round 2 contributes novel gold in only 3.3% of queries (LoCoMo saturates)
- LME: reflmem_3round = 0.876 (+1.3pp over best single-shot), round 2 novel in 22.2% of queries (LME has headroom)
- Temporal-reasoning: reflmem lifts to 0.807 (+1.1pp) where expand_context couldn't move it

**Durable principle**: architectural differences between single-shot and iterative are measurable only on non-saturated benchmarks. Saturated benchmarks collapse them to ceiling.

### 5. Proactive memory — ties on LoCoMo; needs better benchmark

Both proactive_flat (decompose task into parallel info-needs) and proactive_chained (DAG with entity discovery) tested on LoCoMo task-shape prompts with LLM-judge sufficiency metric.

- proactive_flat: ties single_shot at 4.5/10 sufficiency (~7× LLM cost for zero gain)
- proactive_chained: +0.27 TC over flat on entity-stress tasks, but **absolute scores are <3/10** across variants

**LoCoMo isn't the right benchmark for proactive memory.** Its coherent dialogue content doesn't have the separable entity structure ("team A roster" + "per-member allergies") that decomposition and chained retrieval are designed to exploit. Where LoCoMo DID have discoverable entities (t03_melanie_creative_projects), chain won 8 vs 4 — the architecture works when the corpus has what it needs.

## Session verdict on the generalization question

The "universally generalizable memory/reasoning system" recipe turned out to be more nuanced than expected:

**Universal components** (work regardless of corpus/input shape):
- EventMemory ingestion with speaker baking (massive substrate win)
- gpt-5-mini OR gpt-5-nano for cue gen (cross-model portable)

**Input-shape dependent**:
- Cue gen at all — helps on questions/imperatives, neutral on drafts, HURTS on meta/synthesis
- For general task systems, skipping cue gen on meta-queries might be the right default

**Corpus-structure dependent**:
- Turn-summary: works on short-chat corpora, fails on long-form
- Topic-baking: works when topics are extractable, adds noise otherwise
- Speaker_filter: only when queries name participants (LoCoMo-specific)
- Proactive decomposition: works only when info types are separable
- Expand_context: works on sparse gold (LME), hurts on dense gold (LoCoMo)

**Saturation-revealing** (architectural differences invisible at benchmark ceiling):
- LoCoMo-30 saturates at 0.942 K=50 — many architectures tie here
- LME-hard has headroom — architectural differences show up

## Open directions with clear next steps

1. **Build or find structured-entity benchmark** — synthetic meeting minutes, project logs, customer support corpora with explicit rosters/profiles. Test proactive_chained there.

2. **Task-sufficiency metric on LME** — we tested on LoCoMo (saturated). Would expect proactive architectures to show clearer wins on LME's less-saturated content.

3. **Adaptive cue-gen routing** — classifier decides whether to generate cues, based on input shape and corpus structure. Skip cue gen when META/synthesis; use it when question/imperative.

4. **Ingest-time decisions per corpus** — different corpus shapes need different ingest augmentations. Long-form corpus: don't summarize (loses detail). Short-chat: summarize. Schema-rich corpus: extract entities/relations. Auto-detect corpus shape and apply appropriate augmentations.

5. **Cross-session entity linking at ingest** — for multi-session LME: identify entities (user's job, their partner, their health issues) across sessions, build entity-profile embeddings.

## Final production recipes

### LoCoMo (named-speaker dialog)

```
K=20: em_v2f_summ (dual-view) = 0.908
K=50: em_v2f_summ_sf_spkfilter = 0.942
Zero-query-LLM K=20: em_cosine_baseline_summ = 0.850
```

### LongMemEval-hard (user/assistant diary)

```
Best single-shot: em_v2f_lme_mixed_7030 + expand_3 = 0.863
Best iterative: reflmemlme_3round = 0.876
(Turn-summary does NOT work on LME's long-form content)
```

### General task-completion (arbitrary tasks)

```
Benchmark limits currently binding (LoCoMo task-sufficiency ~2-3/10).
Recipe untested on proper benchmark.
Architecture (proactive_chained with entity discovery) is sound but needs structured-corpus evaluation.
```

## Session-total ceiling lift from raw cosine baseline

- LoCoMo K=20: 0.383 → 0.908 (+52.5pp)
- LoCoMo K=50: 0.508 → 0.942 (+43.4pp)
- LME K=50: raw baseline ~0.65 → 0.876 (+22.6pp)

Of this lift:
- ~35pp from EventMemory ingestion alone (speaker baking, universal)
- ~5-10pp from ingest-side additional augmentation (summary/topic, corpus-dependent)
- ~5pp from cue-gen architecture (HyDE+filter, partly corpus-specific)
- ~1-2pp from reflective iteration (on non-saturated benchmarks only)

**The single biggest generalizable lift remains EventMemory's speaker baking.**
