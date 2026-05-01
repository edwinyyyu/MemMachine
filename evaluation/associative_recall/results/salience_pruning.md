# Salience Pruning Experiment

Pruning (or down-weighting) low-salience turns in the retrieval index to test whether a denser signal-per-turn pool improves v2f recall.

## Verdict: ABANDON

No variant improved v2f recall. The two variants that actually pruned anything (aggressive regex drop, downweight regex) both LOST recall on both datasets. Conservative regex and the LLM classifier ended up pruning zero segments, so they matched control trivially.

- `prune_regex_aggressive`: LoCoMo r@20 0.756 -> 0.733 (-0.023); synth r@20 0.613 -> 0.576 (-0.037). r@50 also drops on both.
- `downweight_regex`: identical to aggressive (dropping vs downweighting the same 51/43 segments had no differential effect; downweighted turns were never close enough to top-K to matter).
- `prune_regex_conservative`: word_count<=3 AND backchannel-first-token too narrow; matched 0 segments across both datasets.
- `prune_llm` (gpt-5-mini YES/NO): classifier said YES on 100% of 881 turns classified across both datasets. Even short acks look "askable" to the LLM. False-prune rate 0% by default, pool reduction 0%.

**False-prune diagnostic**: regex aggressive false-prunes 2.7% of LoCoMo gold (1/37) but 8.0% of synthetic gold (9/112), already near the 10% abandon threshold. The synth conversations embed factual info in short conversational acks more than LoCoMo does.

**Why it failed**: v2f's multi-cue retrieval already navigates around low-signal turns via vocabulary matching. Removing backchannel turns does not free up top-K slots for new gold turns because those slots were already going to real content. Removing them only risks dropping tokens/context that happened to share vocabulary with gold turns.

## Pool-size reduction & false-prune rate (by dataset)

| Dataset | Variant | Pool | Pruned | % Pruned | Gold total | Gold pruned | False-prune rate |
|---|---|---:|---:|---:|---:|---:|---:|
| locomo_30q | control | 419 | 0 | 0.0% | 44 | 0 | 0.0% |
| locomo_30q | prune_regex_aggressive | 419 | 51 | 12.2% | 37 | 1 | 2.7% |
| locomo_30q | prune_regex_conservative | 419 | 0 | 0.0% | 37 | 0 | 0.0% |
| locomo_30q | prune_llm | 419 | 0 | 0.0% | 37 | 0 | 0.0% |
| locomo_30q | downweight_regex | 419 | 51 | 12.2% | 37 | 1 | 2.7% |
| synthetic_19q | control | 462 | 0 | 0.0% | 154 | 0 | 0.0% |
| synthetic_19q | prune_regex_aggressive | 462 | 43 | 9.3% | 112 | 9 | 8.0% |
| synthetic_19q | prune_regex_conservative | 462 | 0 | 0.0% | 112 | 0 | 0.0% |
| synthetic_19q | prune_llm | 462 | 0 | 0.0% | 112 | 0 | 0.0% |
| synthetic_19q | downweight_regex | 462 | 43 | 9.3% | 112 | 9 | 8.0% |

## Recall results

| Dataset | Variant | r@20 | d@20 | W/T/L@20 | r@50 | d@50 | W/T/L@50 |
|---|---|---:|---:|:-:|---:|---:|:-:|
| locomo_30q | control | 0.756 | +0.372 | 13/17/0 | 0.858 | +0.350 | 13/17/0 |
| locomo_30q | prune_regex_aggressive | 0.733 | +0.350 | 12/18/0 | 0.825 | +0.317 | 12/18/0 |
| locomo_30q | prune_regex_conservative | 0.756 | +0.372 | 13/17/0 | 0.858 | +0.350 | 13/17/0 |
| locomo_30q | prune_llm | 0.756 | +0.372 | 13/17/0 | 0.858 | +0.350 | 13/17/0 |
| locomo_30q | downweight_regex | 0.733 | +0.350 | 12/18/0 | 0.825 | +0.317 | 12/18/0 |
| synthetic_19q | control | 0.613 | +0.044 | 8/7/4 | 0.851 | +0.028 | 4/14/1 |
| synthetic_19q | prune_regex_aggressive | 0.576 | +0.007 | 6/8/5 | 0.830 | +0.006 | 4/12/3 |
| synthetic_19q | prune_regex_conservative | 0.613 | +0.044 | 8/7/4 | 0.851 | +0.028 | 4/14/1 |
| synthetic_19q | prune_llm | 0.613 | +0.044 | 8/7/4 | 0.851 | +0.028 | 4/14/1 |
| synthetic_19q | downweight_regex | 0.576 | +0.007 | 6/8/5 | 0.830 | +0.006 | 4/12/3 |

## Per-category delta (LoCoMo-30, r@20 / r@50)


### locomo_multi_hop

| Variant | n | r@20 | d@20 | r@50 | d@50 |
|---|---:|---:|---:|---:|---:|
| control | 4 | 0.625 | +0.125 | 0.875 | +0.375 |
| prune_regex_aggressive | 4 | 0.625 | +0.125 | 0.875 | +0.375 |
| prune_regex_conservative | 4 | 0.625 | +0.125 | 0.875 | +0.375 |
| prune_llm | 4 | 0.625 | +0.125 | 0.875 | +0.375 |
| downweight_regex | 4 | 0.625 | +0.125 | 0.875 | +0.375 |

### locomo_single_hop

| Variant | n | r@20 | d@20 | r@50 | d@50 |
|---|---:|---:|---:|---:|---:|
| control | 10 | 0.617 | +0.567 | 0.825 | +0.700 |
| prune_regex_aggressive | 10 | 0.650 | +0.600 | 0.825 | +0.700 |
| prune_regex_conservative | 10 | 0.617 | +0.567 | 0.825 | +0.700 |
| prune_llm | 10 | 0.617 | +0.567 | 0.825 | +0.700 |
| downweight_regex | 10 | 0.650 | +0.600 | 0.825 | +0.700 |

### locomo_temporal

| Variant | n | r@20 | d@20 | r@50 | d@50 |
|---|---:|---:|---:|---:|---:|
| control | 16 | 0.875 | +0.312 | 0.875 | +0.125 |
| prune_regex_aggressive | 16 | 0.812 | +0.250 | 0.812 | +0.062 |
| prune_regex_conservative | 16 | 0.875 | +0.312 | 0.875 | +0.125 |
| prune_llm | 16 | 0.875 | +0.312 | 0.875 | +0.125 |
| downweight_regex | 16 | 0.812 | +0.250 | 0.812 | +0.062 |
