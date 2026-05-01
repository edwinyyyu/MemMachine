# Session 2026-04-18 — EventMemory re-evaluation

Continuation from 2026-04-17 session. Focus: port shipped session findings to EventMemory backend (which we should have been using all along), re-evaluate session wins, and extend with EventMemory-specific mechanisms.

## Headline

New LoCoMo K=50 ceiling: **0.9417** via `em_hyde_first_person + speaker_filter` (HyDE single-probe in `"{speaker}: <first-person chat>"` format + EventMemory property_filter on speaker).

Also ties: `em_v15_conditional_hop2_sf_filter` at 0.9417 K=50 (+2 LLM calls/q vs HyDE's +1, so HyDE wins on cost).

Prior SS ceiling was 0.892. New ceiling: **+5pp from session close**, **+8.4pp from em_v2f_speakerformat baseline**.

## Production recipes (EventMemory backend)

### LoCoMo (speaker-named conversations)

```
1. Speaker-match regex on query
   - Match → em_hyde_first_person + property_filter(context.source=<speaker>)
              → 0.850 R@20 / 0.942 R@50, 1 LLM call/q
2. No match → em_v2f_speakerformat
              → 0.817 / 0.892, 1 LLM call/q
```

### LongMemEval (User/Assistant format)

```
For peak R@50:
  em_v2f_lme_mixed_7030 + expand_context=3
  → 0.863 overall (0.848 multi-session, 0.945 single-session-pref, 0.796 temporal)

For peak temporal-reasoning (breaks prior 0.77 ceiling):
  em_ens_2_lme_retuned (v2f_lme_userformat + retuned type_enumerated) + expand_context=3
  → 0.850 overall, **0.806 temporal-reasoning** (new substrate ceiling)
```

## Key substrate insights added

1. **EventMemory speaker baking lifts cosine_baseline massively.** LoCoMo: 0.383 → 0.733 R@20 (+35pp), 0.508 → 0.883 R@50 (+37pp). Speaker is baked via `_format_text` which prepends `"{source}: "` when embedding; the content body stays raw text, speaker is in `MessageContext`.

2. **`expand_context` is corpus-shape dependent.**
   - LoCoMo (short, dense single conversation): `expand_context=0` is correct; nonzero REGRESSES (−8 to −12pp) because neighbors dilute top-K.
   - LME (long, multi-session, sparse gold): `expand_context=3` is decisive (+5pp). Gold spread across sessions → temporal neighbors rescue.
   - Set per-corpus, not globally.

3. **Multi-probe dispersion wins on WEAK cosine, single concentrated probe wins on STRONG cosine.** SS-era finding that "multi-probe > single probe" flipped on EM: HyDE `turn_format` (multi-probe) LOSES on EM; HyDE `first_person` (single probe) WINS. Geometry-dependent; applies generally when the base retrieval is strong.

4. **Speaker bake ≠ redundant for speaker filter.** em_cosine_baseline hits 0.733 on LoCoMo K=20; em_two_speaker_filter hits 0.842 (+11pp). Hard filter still adds value on top of speaker-baked embeddings.

5. **Cue generation value is K-budget dependent under EventMemory**:
   - K=20: cues earn their cost (+7-8pp over cosine_baseline)
   - K=50 without filter: cues are ~neutral (+0.9pp)
   - K=50 with strong filter: cues HURT (−3.3pp; em_two_speaker_query_only beats em_two_speaker_filter at K=50)
   - But HyDE first-person + filter reverses this! Single concentrated HyDE probe adds +0.8pp over query_only at K=50.

6. **LoCoMo-specific session wins DON'T all transfer to LME** — different corpus structure exposes different signals:
   - `two_speaker_filter`: fires 0/90 on LME (first-person "I", not named participants)
   - `critical_info_store`: 0.03% tag rate on LME (no medication/commitment-pattern facts)
   - v2f generalizes cleanly across both

7. **Specialist prompts need PER-CORPUS tuning.** `type_enumerated` regressed on LME (−4.5pp) due to LoCoMo-specific categories (ARRIVAL, PREFERENCE). Rewritten for LME diary register (USER_FACT/DECISION/PREFERENCE/TEMPORAL/QUESTION_ASKED), ensemble recovers and uniquely breaks LME temporal ceiling at 0.806.

8. **Prompt format should mirror embedded format.** When embeddings prepend speaker prefix, cues should too. The retuned `v2f_speakerformat` prompt requires `"{speaker}: "` prefix on cues; netted +9.2pp K=20 / +2.5pp K=50 over pristine v2f. Durable principle: any ingest-side transformation applied to turns should also be applied to cues.

9. **Multiple mechanisms converge at ~0.94 K=50 LoCoMo ceiling** (HyDE+filter, v15_conditional_hop2+filter, hypothesis_driven+filter, query_only). Strong evidence of substrate-saturation at that level given current EventMemory structure.

10. **Intent-parser's value is gated by ingest-schema richness.** LoCoMo with synthesized timestamps → no viable temporal filter. Real session_date_time ingested → filter works infrastructurally, but intent-parser's "when did X?" schema rarely treats as temporal_relation (treats as answer_form=date). Schema-bound: need different intent schema (retrieve-all-mentions-of-X + date rank) or reference-date resolver.

11. **SS-era iterative architectures don't port alone; need filter composition.** hypothesis_driven (+33pp SS), v15_conditional_hop2 (+31pp SS) tie or lose v2f_speakerformat on EM without filter. WITH speaker_filter, they tie HyDE at the 0.94 ceiling. SS lifts were "fighting raw cosine's 0.383 floor"; EM's strong baseline absorbs that work.

12. **`expand_context` is for context assembly, not recall@K.** Native EventMemory feature confirmed useful on LME (where sessions scatter gold) but harmful on LoCoMo (where gold clusters tightly). Use case-dependent.

## Architectures tested on EventMemory (new this wave)

### Ported from SS (shipped in prior session)
- em_v2f — 0.742 / 0.883 (parity with SS v2f)
- em_v2f_speakerformat (retuned) — **0.817 / 0.892** (+7.5pp K=20 over SS v2f via cue format matching)
- em_ens_2 — 0.783 / 0.867 (ens_2 on LME regressed; retuned version recovered)
- em_critical_info — LoCoMo: 0 flagged turns (corpus-specific)
- em_alias_expand_v2f — 0.825 / 0.883 (+13pp R@20 over SS version; speaker baking aligns alias probes)
- em_alias_expand_speakerformat — 0.833 K=20 (marginal tune)
- em_gated_no_speaker — 0.742 / 0.883 (speaker channel was carrying most of gated's SS value)
- em_two_speaker_filter — **0.842 / 0.900** (K=20 LoCoMo winner with cues)
- em_two_speaker_query_only — **0.800 / 0.933** (zero-cue K=50 winner)
- em_meta_router — identical to em_two_speaker_filter on LoCoMo (100% speaker-match coverage)

### New mechanisms this wave
- **em_hyde_first_person + speaker_filter** — **0.850 / 0.942** (new ceiling across both K)
- em_hyde_narrative — 0.750 / 0.833 (paragraph probes lose)
- em_hyde_turn_format — 0.706 / 0.750 (multi-probe dilutes — SS pattern flipped)
- em_orient_brief / em_orient_terminology — strictly worse than v2f_sf (binning was correct)
- em_hypothesis_driven_sf_filter — 0.808 / 0.933 (ties query_only K=50)
- **em_v15_conditional_hop2_sf_filter** — 0.817 / **0.942** (ties HyDE+filter K=50 at +1 LLM call)
- em_v15_rerank_sf_filter — 0.800 / 0.917
- em_working_memory_buffer_sf_filter — abandon
- intent_em (LLM-parsed filter) — 0.817 / 0.908 (less effective than regex-based two_speaker)
- intent_rts (real-timestamp temporal filter) — 0.783 / 0.908 (schema bind; temporal fires on 2/30)

### LME-specific
- em_v2f_expand_3 — 0.832 overall (prior LME ceiling)
- **em_v2f_lme_mixed_7030 + expand_3** — **0.863 overall, 0.945 single-session-pref**
- **em_ens_2_lme_retuned + expand_3** — 0.850, **0.806 temporal-reasoning** (new ceiling on the historically-hard category)

### Infrastructure
- LoCoMo real-timestamp re-ingest validated with parser `datetime.strptime("%I:%M %p on %d %B, %Y")`, 1451 turns, 0 errors. Temporal property_filter smoke-test passes with 0 violations.
- LoCoMo raw text embedded only (no speaker/timestamp in content body) — canonical separation preserved.
- EventMemory configured with `reranker=None`, `max_text_chunk_length=500`, `derive_sentences=False`.
- Collection prefixes: `arc_em_lc30_v1_<conv>` (synthesized ts), `arc_em_lc30_rts_v1_<conv>` (real ts), `arc_em_lmehard_v1_<question>` (LME hard).

## Abandoned this wave

- Orient-then-cue (both variants)
- HyDE narrative / turn_format (multi-probe on EM)
- Chain_with_scratchpad retuning on EM
- Working_memory_buffer
- v15_rerank on K=20 (didn't transfer)
- Short-cue / 5-cue / natural-turn prompt variants (all lose to em_v2f_speakerformat)
- Mixed-speakers / role-tag prompt variants
- Intent parser + temporal filter on LoCoMo (schema mismatch)

## Remaining open directions

- **Reference-date resolver for intent_parser temporal constraints** — map "4 years ago" to corpus-relative date before building window filter. Would unlock the temporal channel on LoCoMo.
- **Different intent schema for "when did X?" questions** — retrieve-all-mentions-of-X, rank by date. Not tested.
- **Cross-session entity linking for LME multi-session** — still at 0.865 K=50; a cross-conversation entity graph might lift further.
- **LME real-time-aware filters** — LME has real dates; haven't tested temporal filter on LME (only on LoCoMo).
- **Temporal-reasoning beyond 0.806** — structural mystery. Current substrate has a soft ceiling there.

## Session status

Converged. Ceilings:
- LoCoMo K=20: 0.850 (em_hyde_first_person + speaker_filter)
- LoCoMo K=50: 0.942 (em_hyde_first_person + speaker_filter)
- LME overall K=50: 0.863 (em_v2f_lme_mixed_7030 + expand_3)
- LME temporal-reasoning K=50: 0.806 (em_ens_2_lme_retuned + expand_3)

vs session-start baselines:
- LoCoMo K=20: 0.383 → 0.850 (+47pp)
- LoCoMo K=50: 0.508 → 0.942 (+43pp)

Of this lift: ~35pp from EventMemory ingestion alone (speaker baking); ~7-8pp from cue-gen + filter architectures on top.
