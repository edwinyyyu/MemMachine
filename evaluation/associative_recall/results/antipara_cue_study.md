# Anti-Paraphrase Cue Generation Study

Empirical test of whether adding explicit anti-paraphrase instructions and/or a verbatim-quote rule to the v2f cue-generation prompt improves retrieval recall.

## Constraints
- `text-embedding-3-small`, `gpt-5-mini` (fixed).
- No framework edits. Reuses hop0 top-10 + 2 cues top-10 structure.
- Fair-backfill eval to K=20 and K=50.

## Variants
- `v2f_anti_paraphrase` — v2f + two negative instructions: 'Do NOT restate or paraphrase the question.' and 'Do NOT guess the answer.'
- `v2f_verbatim_quote` — v2f + each cue must include a 2-5 word verbatim phrase from hop0 excerpts; 15-40 word length cap; post-filter drops cues that fail `verbatim_check`.
- `v2f_anti_paraphrase_verbatim` — both together.

## Overall recall (avg over 4 datasets)
| Arch | avg r@20 | avg r@50 |
|------|---------:|---------:|
| meta_v2f | 0.6105 | 0.8821 |
| v2f_anti_paraphrase | 0.5786 | 0.8558 |
| v2f_verbatim_quote | 0.3611 | 0.5528 |
| v2f_anti_paraphrase_verbatim | 0.3778 | 0.5278 |

## Per-dataset arch recall
| Dataset | K | meta_v2f | v2f_anti_paraphrase | v2f_verbatim_quote | v2f_anti_paraphrase_verbatim |
|---------|---|------:|------:|------:|------:|
| locomo_30q | 20 | 0.756 | 0.733 | 0.361 | 0.378 |
| locomo_30q | 50 | 0.858 | 0.800 | 0.553 | 0.528 |
| synthetic_19q | 20 | 0.613 | 0.538 | n/a | n/a |
| synthetic_19q | 50 | 0.851 | 0.820 | n/a | n/a |
| puzzle_16q | 20 | 0.480 | 0.505 | n/a | n/a |
| puzzle_16q | 50 | 0.917 | 0.904 | n/a | n/a |
| advanced_23q | 20 | 0.593 | 0.538 | n/a | n/a |
| advanced_23q | 50 | 0.902 | 0.900 | n/a | n/a |

## Deltas vs `meta_v2f` (arch_r@K)
| Dataset | K | v2f_anti_paraphrase | v2f_verbatim_quote | v2f_anti_paraphrase_verbatim |
|---------|---|------:|------:|------:|
| locomo_30q | 20 | -0.022 | -0.395 | -0.378 |
| locomo_30q | 50 | -0.058 | -0.305 | -0.330 |
| synthetic_19q | 20 | -0.075 | n/a | n/a |
| synthetic_19q | 50 | -0.032 | n/a | n/a |
| puzzle_16q | 20 | +0.025 | n/a | n/a |
| puzzle_16q | 50 | -0.013 | n/a | n/a |
| advanced_23q | 20 | -0.055 | n/a | n/a |
| advanced_23q | 50 | -0.002 | n/a | n/a |

## Target categories (paraphrase concentration)
These are the categories `per_cue_attribution` identified as paraphrase-loser hotspots.

### `locomo_temporal` (locomo_30q)
| Arch | n | r@20 | Δ@20 | r@50 | Δ@50 | W/T/L@50 |
|------|---|-----:|-----:|-----:|-----:|---------:|
| meta_v2f | 16 | 0.875 | +0.000 | 0.875 | +0.000 | 2/14/0 |
| v2f_anti_paraphrase | 16 | 0.812 | -0.062 | 0.875 | +0.000 | 2/14/0 |
| v2f_verbatim_quote | 16 | 0.438 | -0.438 | 0.750 | -0.125 | 0/16/0 |
| v2f_anti_paraphrase_verbatim | 16 | 0.500 | -0.375 | 0.750 | -0.125 | 0/16/0 |

### `locomo_single_hop` (locomo_30q)
| Arch | n | r@20 | Δ@20 | r@50 | Δ@50 | W/T/L@50 |
|------|---|-----:|-----:|-----:|-----:|---------:|
| meta_v2f | 10 | 0.617 | +0.000 | 0.825 | +0.000 | 8/2/0 |
| v2f_anti_paraphrase | 10 | 0.650 | +0.033 | 0.700 | -0.125 | 7/2/1 |
| v2f_verbatim_quote | 10 | 0.133 | -0.483 | 0.208 | -0.617 | 2/8/0 |
| v2f_anti_paraphrase_verbatim | 10 | 0.133 | -0.483 | 0.183 | -0.642 | 2/7/1 |

## Negative-check categories (don't regress here)

### `locomo_multi_hop` (locomo_30q)
| Arch | n | r@50 | Δ vs v2f |
|------|---|-----:|---------:|
| meta_v2f | 4 | 0.875 | +0.000 |
| v2f_anti_paraphrase | 4 | 0.750 | -0.125 |
| v2f_verbatim_quote | 4 | 0.625 | -0.250 |
| v2f_anti_paraphrase_verbatim | 4 | 0.500 | -0.375 |

## Qualitative: cue compliance
| Arch | total cues | kept | dropped (no verbatim) | empty |
|------|-----------:|-----:|----------------------:|------:|
| meta_v2f | 176 | 176 | 0 | 0 |
| v2f_anti_paraphrase | 174 | 174 | 0 | 0 |
| v2f_verbatim_quote | 54 | 53 | 1 | 0 |
| v2f_anti_paraphrase_verbatim | 44 | 44 | 0 | 0 |

### Heuristic paraphrase rate on locomo_30q cues
Proxy: cue starts with a question word OR ends with '?' OR shares ≥60% of non-stopword tokens with the question. Lower is better.
| Arch | paraphrase rate |
|------|----------------:|
| meta_v2f | 0.067 |
| v2f_anti_paraphrase | 0.117 |
| v2f_verbatim_quote | 0.000 |
| v2f_anti_paraphrase_verbatim | 0.000 |

## Sample cues on `locomo_temporal` (first 3 questions)

**meta_v2f**
- Q: _When did Caroline go to the LGBTQ support group?_
    - CUE: I went to a LGBTQ support group yesterday and it was so powerful.
    - CUE: I went to a support group yesterday
- Q: _When did Melanie paint a sunrise?_
    - CUE: Is this your own painting?
    - CUE: Yeah, I painted that lake sunrise last year! It's special to me.
- Q: _When did Melanie run a charity race?_
    - CUE: I ran a charity race for mental health last Saturday — it was really rewarding.
    - CUE: I ran a charity race last weekend for mental health.

**v2f_anti_paraphrase**
- Q: _When did Caroline go to the LGBTQ support group?_
    - CUE: I went to a LGBTQ support group yesterday and it was so powerful.
    - CUE: I went to the LGBTQ support group yesterday
- Q: _When did Melanie paint a sunrise?_
    - CUE: Yeah, I painted that lake sunrise last year! It's special to me.
    - CUE: I painted that lake sunrise last year
- Q: _When did Melanie run a charity race?_
    - CUE: I ran a charity race for mental health last Saturday – it was really rewarding.
    - CUE: That charity race sounds great, Mel! Making a difference & raising awareness for mental health is super rewarding - I'm really proud of you for taking part!

**v2f_verbatim_quote**
- Q: _When did Caroline go to the LGBTQ support group?_
    - CUE: Search the conversation for the message containing "LGBTQ support group" and extract the exact time phrasing (for example "yesterday") plus any nearby messages that clarify the date or day.
    - CUE: Locate all messages referencing "support group yesterday" and confirm whether "yesterday" consistently refers to the same calendar day or if other entries provide a specific date or alternative timing.
- Q: _When did Melanie paint a sunrise?_
    - CUE: Locate every occurrence of "I painted that lake sunrise" and capture the surrounding messages to extract the exact timing language and confirm who said they painted the sunrise.
    - CUE: Retrieve mentions of "last year" and nearby lines referencing "lake sunrise" or painting dates to confirm Melanie's statement about when she painted the sunrise.
- Q: _When did Melanie run a charity race?_
    - CUE: You said "I ran a charity race" and "last Saturday" — please provide the specific date, month, or year you meant so I can record the exact day of the event in the timeline.
    - CUE: Referencing your post about "charity race for mental health" and "last Saturday" please include the exact calendar date or year so the event can be unambiguously indexed in the conversation history.

**v2f_anti_paraphrase_verbatim**
- Q: _When did Caroline go to the LGBTQ support group?_
    - CUE: Retrieve the messages around 'support group yesterday' including any adjacent timestamps, replies, or system dates so we can determine the exact calendar date referenced by "yesterday".
    - CUE: Search for all occurrences of 'LGBTQ support group' and extract the full Caroline messages plus nearby messages or metadata that indicate when she attended (timestamps, explicit dates, or relative references).
- Q: _When did Melanie paint a sunrise?_
- Q: _When did Melanie run a charity race?_
    - CUE: Locate the earlier assistant message mentioning "charity race" and "last Saturday" to confirm the exact timing and any follow-up details about Melanie's participation in that event.
    - CUE: Search conversation for posts referencing "mental health" or the "charity race for mental health" phrase plus Caroline's reply to capture context and any calendar date tied to "last Saturday".

## Verdict

**All three variants abandoned.** None meet the ship criterion.

### `v2f_anti_paraphrase`: ABANDON (narrow negative)
- Loses to v2f on 6 of 8 (dataset, K) cells. Average recall: 0.579 vs v2f 0.611 (-0.032).
- Target category `locomo_single_hop` @K=20: +0.033 (tiny win), but @K=50: **-0.125** (loss).
- Target category `locomo_temporal` @K=20: **-0.062** (loss).
- Negative check `locomo_multi_hop` @K=50: **-0.125** (regresses).
- Qualitative: the 'Do NOT' instruction was partially obeyed — paraphrase rate on locomo actually rose (0.117 vs v2f's 0.067), because the LLM leans toward literal quote-style restatements that happen to share more question tokens. Cues are otherwise conversation-like.
- Per decision rule (b): loses to v2f on LoCoMo → abandon.

### `v2f_verbatim_quote`: ABANDON (catastrophic)
- LoCoMo r@20 collapses from 0.756 to **0.361** (-0.395), r@50 from 0.858 to 0.553 (-0.305).
- `locomo_single_hop` r@50: **-0.617**. Complete failure on target category.
- Post-filter dropped only 1 of 54 cues — the LLM did quote verbatim phrases, but wrapped them in search-command prose ("Search the conversation for ...", "Locate every occurrence of ..."). The 2-5-word literal quote is embedded in meta-instruction text that embeds nowhere useful.
- Per decision rule (c): the verbatim-quote rule causes degenerate instruction-style cues. The 15-40 word length cap already present didn't help. Not worth a second tuning pass.

### `v2f_anti_paraphrase_verbatim`: ABANDON
- Inherits `v2f_verbatim_quote`'s failure mode. LoCoMo r@20 = 0.378.
- Adding the anti-paraphrase instruction did not counteract the verbatim-induced degeneration.
- `locomo_single_hop` r@50: **-0.642**.

### Why the hypothesis didn't transfer
Per-cue attribution identified paraphrase-losers, but "not paraphrasing" is not the same as "being corpus-grounded." The winning cue property (entity density, conversation-style phrasing) is generated naturally by v2f when the LLM has useful hop0 context. Adding negative instructions either:
1. Leaves cues unchanged (LLM already avoided paraphrases in most cases), or
2. Pushes the LLM toward a _different_ off-distribution style (quote-wrapped search commands), which embeds worse.

The recommended follow-up is NOT another prompt tweak — it's exemplar-guided generation that _shows_ the LLM what a good long-entity-grounded cue looks like, without telling it what not to do. But the prior `fewshot_cue_eval` already tested that direction and also lost on LoCoMo, suggesting the v2f prompt is close to a local optimum within the single-call 2-cue architecture.

## Output paths
- Per-(arch, dataset) JSON: `results/antipara_{arch}_{dataset}.json`
- Consolidated JSON: `results/antipara_cue_study.json`
- This report: `results/antipara_cue_study.md`
- Source: `antipara_cue_gen.py`, `antipara_cue_eval.py`, `antipara_remaining.py`
