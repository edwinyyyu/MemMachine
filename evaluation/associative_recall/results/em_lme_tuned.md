# LME-hard EventMemory: LME-tuned cue variants

All variants run with `expand_context=3` (the LME winning recipe).

## References

- `em_v2f_userprefix` (expand=0, prior): R@50 = 0.780
- `em_v2f_expand_3` (prior leader): R@50 = 0.832
- `em_ens_2_userprefix` (regressed, prior): R@50 = 0.735
- SS v2f reference: R@50 = 0.817

## Per-variant summary

| Architecture | R@20 | R@50 | Δ vs em_v2f_expand_3 (0.832) | time (s) |
| --- | --- | --- | --- | --- |
| `em_v2f_lme_userformat` | 0.5976 | 0.8304 | -0.0016 | 201.1 |
| `em_v2f_lme_user_only` | 0.6067 | 0.8468 | +0.0148 | 235.9 |
| `em_v2f_lme_mixed_7030` | 0.6368 | 0.8631 | +0.0311 | 280.1 |
| `em_type_enumerated_lme_retuned` | 0.5681 | 0.8175 | -0.0145 | 288.3 |
| `em_ens_2_lme_retuned` | 0.5988 | 0.8499 | +0.0179 | 32.8 |

## Recall matrix (R@20)

| Architecture | multi-session | single-session-preference | temporal-reasoning |
| --- | --- | --- | --- |
| `em_v2f_lme_userformat` | 0.5628 | 0.7652 | 0.4647 |
| `em_v2f_lme_user_only` | 0.5781 | 0.7577 | 0.4844 |
| `em_v2f_lme_mixed_7030` | 0.5785 | 0.8239 | 0.5082 |
| `em_type_enumerated_lme_retuned` | 0.5623 | 0.6399 | 0.5020 |
| `em_ens_2_lme_retuned` | 0.5641 | 0.7190 | 0.5133 |

## Recall matrix (R@50)

| Architecture | multi-session | single-session-preference | temporal-reasoning |
| --- | --- | --- | --- |
| `em_v2f_lme_userformat` | 0.8384 | 0.8781 | 0.7748 |
| `em_v2f_lme_user_only` | 0.8509 | 0.9084 | 0.7812 |
| `em_v2f_lme_mixed_7030` | 0.8484 | 0.9449 | 0.7961 |
| `em_type_enumerated_lme_retuned` | 0.8472 | 0.8468 | 0.7584 |
| `em_ens_2_lme_retuned` | 0.8653 | 0.8785 | 0.8057 |

## Sample cues (first 2 questions)

### `em_v2f_lme_userformat`

- Q `ba358f49`: How many years will I be when my friend Rachel gets married?
  - CUE: `User: I'm looking for some advice on skincare routines for my age group`
  - CUE: `User: my friend Rachel's getting married next year`

- Q `5a7937c8`: How many days did I spend participating in faith-related activities in December?
  - CUE: `User: I helped out at the church's annual holiday food drive on December 10th, sorting donations and packing boxes for families`
  - CUE: `User: I just got back from a lovely midnight mass on Christmas Eve at St. Mary's Church, which was a really meaningful service`

### `em_v2f_lme_user_only`

- Q `ba358f49`: How many years will I be when my friend Rachel gets married?
  - CUE: `User: I already told you my age earlier in the chat, find the message where I said how old I am`
  - CUE: `User: I mentioned Rachel's wedding is next year and may have given a month or date, find that message`

- Q `5a7937c8`: How many days did I spend participating in faith-related activities in December?
  - CUE: `User: I helped out at the church's annual holiday food drive on December 10th, sorting donations and packing boxes for families`
  - CUE: `User: I just got back from a lovely midnight Mass on Christmas Eve at St. Mary's Church`

## Verdict

- Top variant: `em_v2f_lme_mixed_7030` R@50 = 0.8631 (Δ vs em_v2f_expand_3 0.832 = +0.0311)
- Ensemble recovery: `em_ens_2_lme_retuned` R@50 = 0.8499 (was 0.735, Δ = +0.1149)

## Outputs

- JSON: `results/em_lme_tuned.json`
- Sources: `em_lme_tuned_cues.py`, `em_lme_tuned_eval.py`
- Caches: `cache/lmetune_v2f_userformat_cache.json`, `cache/lmetune_v2f_useronly_cache.json`, `cache/lmetune_v2f_mixed7030_cache.json`, `cache/lmetune_te_retuned_cache.json`