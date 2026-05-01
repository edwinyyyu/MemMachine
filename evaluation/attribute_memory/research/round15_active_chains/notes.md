# Round 15 - At-write-time fix for the writer ref-emission collapse

Round 14 result (baseline aen1_simple on dense_chains, seed=17):
- Overall ref_emission_rate: 0.465
- Overall ref_correctness_rate: 0.186
- Bucket curve (non-first transitions per 100-wide bucket):
    (0,100]   trans=8   emit=0.75 correct=0.50
    (100,200] trans=14  emit=0.71 correct=0.36
    (200,300] trans=10  emit=0.30 correct=0.20
    (300,400] trans=9   emit=0.33 correct=0.22
    (400,500] trans=10  emit=0.50 correct=0.20
    (500,600] trans=10  emit=0.50 correct=0.00
    (600,700] trans=13  emit=0.23 correct=0.08
    (700,800] trans=12  emit=0.42 correct=0.00
- QA: deterministic 16/32, judge 17/32

## Hypothesis

The collapse is a context-window problem: chain heads quietly fall out of
the writer's most-recent-12 prior log window. Inject the per-batch ACTIVE
STATE block (looked up from the structural index `supersede_head` for
entities mentioned in the batch). The writer then sees the chain heads
even when chains have been quiet for hundreds of turns.

## Architecture: aen1_active

- Reuses aen1_simple's data model, build_index, retrieve, answer_question.
- Overrides write_batch to inject "ACTIVE STATE OF ENTITIES IN THIS BATCH".
- Cheap regex-based entity extraction (TitleCased + always @User).
- max_active_state_size caps the block (50, 100, 200 tested).
- rebuild_index_every=4 batches (vs round 14's 40) so active-state stays
  fresh; this is structural-only, no extra LLM cost.

## Plan

- cap=100 primary run: full ingest + QA + judge (~197 LLM)
- cap=200 ablation: ingest only (~149 LLM)
- cap=50  ablation: ingest only (~149 LLM)
- Hard cap 400 LLM. If we hit it, save partial.

## Running log

(filled in as we run)
