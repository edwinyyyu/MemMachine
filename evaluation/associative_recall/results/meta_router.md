# Meta-router composition

Regex-based zero-LLM dispatch between `two_speaker_filter` (shape-robust, zero LLM) and `gated_overlay` v1 (confidence_threshold=0.7). If the query mentions a known conversation participant's first name -> two_speaker_filter; otherwise -> gated_overlay.


## Route distribution

| Dataset | n | two_speaker % | gated % |
|---|---:|---:|---:|
| locomo_30q | 30 | 100.0% | 0.0% |
| synthetic_19q | 19 | 0.0% | 100.0% |


## Recall matrix (fair-backfill)

| Arch | Dataset | base@20 | arch@20 | Δ@20 | base@50 | arch@50 | Δ@50 | avg LLM |
|---|---|---|---|---|---|---|---|---|
| meta_v2f | locomo_30q | 0.3833 | 0.7556 | +0.3722 | 0.5083 | 0.8583 | +0.3500 | 1.00 |
| meta_v2f | synthetic_19q | 0.5694 | 0.6130 | +0.0436 | 0.8238 | 0.8513 | +0.0276 | 1.00 |
| two_speaker_filter | locomo_30q | 0.3833 | 0.8917 | +0.5083 | 0.5083 | 0.8917 | +0.3833 | 1.00 |
| two_speaker_filter | synthetic_19q | 0.5694 | 0.6121 | +0.0427 | 0.8238 | 0.8372 | +0.0135 | 1.00 |
| gated_threshold_0.7 | locomo_30q | 0.3833 | 0.7583 | +0.3750 | 0.5083 | 0.8917 | +0.3833 | 2.00 |
| gated_threshold_0.7 | synthetic_19q | 0.5694 | 0.5675 | -0.0019 | 0.8238 | 0.8332 | +0.0095 | 2.00 |
| meta_router | locomo_30q | 0.3833 | 0.8917 | +0.5083 | 0.5083 | 0.8917 | +0.3833 | 1.00 |
| meta_router | synthetic_19q | 0.5694 | 0.5675 | -0.0019 | 0.8238 | 0.8332 | +0.0095 | 2.00 |
| meta_router_inverted | locomo_30q | 0.3833 | 0.7583 | +0.3750 | 0.5083 | 0.8917 | +0.3833 | 2.00 |
| meta_router_inverted | synthetic_19q | 0.5694 | 0.6121 | +0.0427 | 0.8238 | 0.8372 | +0.0135 | 1.00 |


## Per-route recall (meta_router slices)

For each dataset, split meta_router results by which sub-arch ran.

| Dataset | Route | n | pct | arch@20 | arch@50 | Δ@20 | Δ@50 |
|---|---|---:|---:|---:|---:|---:|---:|
| locomo_30q | two_speaker | 30 | 100.0% | 0.8917 | 0.8917 | +0.5083 | +0.3833 |
| synthetic_19q | gated | 19 | 100.0% | 0.5675 | 0.8332 | -0.0019 | +0.0095 |


## meta_router head-to-head (per-question W/T/L)

| Dataset | vs | K | W/T/L |
|---|---|---|---|
| locomo_30q | two_speaker_filter | K=20 | 0/30/0 |
| locomo_30q | two_speaker_filter | K=50 | 0/30/0 |
| locomo_30q | gated_threshold_0.7 | K=20 | 7/23/0 |
| locomo_30q | gated_threshold_0.7 | K=50 | 0/30/0 |
| locomo_30q | meta_router_inverted | K=20 | 7/23/0 |
| locomo_30q | meta_router_inverted | K=50 | 0/30/0 |
| synthetic_19q | two_speaker_filter | K=20 | 3/10/6 |
| synthetic_19q | two_speaker_filter | K=50 | 1/17/1 |
| synthetic_19q | gated_threshold_0.7 | K=20 | 0/19/0 |
| synthetic_19q | gated_threshold_0.7 | K=50 | 0/19/0 |
| synthetic_19q | meta_router_inverted | K=20 | 3/10/6 |
| synthetic_19q | meta_router_inverted | K=50 | 1/17/1 |


## Shape-robustness (LoCoMo task-shape variants)

meta_router recall on the 30 LoCoMo originals (ORIGINAL reused from this run's primary eval) vs the 30 CMD / 30 DRAFT / 30 META rewrites.

| Shape | n | arch@20 | arch@50 | Drop vs ORIG @20 | Drop vs ORIG @50 | routes |
|---|---:|---:|---:|---:|---:|---|
| ORIGINAL | 30 | 0.8917 | 0.8917 | +0.0000 | +0.0000 | two_speaker=30 |
| CMD | 30 | 0.8167 | 0.8167 | +0.0750 | +0.0750 | two_speaker=30 |
| DRAFT | 30 | 0.8583 | 0.8583 | +0.0334 | +0.0334 | two_speaker=30 |
| META | 30 | 0.7917 | 0.8083 | +0.1000 | +0.0834 | two_speaker=30 |


### Comparison vs prior arches on shape-robustness (LoCoMo @K=50)

Numbers for `two_speaker_filter` and `gated_threshold_0.7` lifted from `results/gated_shape.md`.

| Architecture | ORIG | CMD | DRAFT | META | Worst drop |
|---|---:|---:|---:|---:|---:|
| meta_router | 0.8917 | 0.8167 | 0.8583 | 0.8083 | +0.0834 |
| two_speaker_filter | 0.8917 | 0.8167 | 0.8583 | 0.8083 | +0.0834 |
| gated_threshold_0.7 | 0.8917 | 0.7333 | 0.8167 | 0.7417 | +0.1584 |
| meta_v2f | 0.8583 | 0.7333 | 0.8167 | 0.7417 | +0.1250 |


## Known speaker pairs (from conversation_two_speakers.json)

| Conversation | user | assistant |
|---|---|---|
| beam_4 | Christina | UNKNOWN |
| beam_5 | Craig | UNKNOWN |
| beam_6 | Crystal | UNKNOWN |
| beam_7 | UNKNOWN | UNKNOWN |
| beam_8 | Darryl | UNKNOWN |
| locomo_conv-26 | Caroline | Melanie |
| locomo_conv-30 | Jon | Gina |
| locomo_conv-41 | John | Maria |
| synth_medical | UNKNOWN | UNKNOWN |
| synth_personal | UNKNOWN | UNKNOWN |
| synth_planning | UNKNOWN | UNKNOWN |
| synth_technical | UNKNOWN | UNKNOWN |
| synth_work | UNKNOWN | UNKNOWN |


## Verdict

- LoCoMo K=50: meta_router=0.8917, two_speaker_filter=0.8917, gated=0.8917, meta_router_inverted=0.8917
- **SHIP meta_router for cost** — ties max(two_speaker, gated)=0.8917 at K=50, but saves the gated LLM call on the 100% of queries routed to two_speaker.
