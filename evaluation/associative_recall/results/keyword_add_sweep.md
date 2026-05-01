# Keyword/Granularity Add-only Steering Sweep

Single-round, additive-only probe update across granularities (keyword | short_phrase | sentence) and alpha scales. Unit-normalized phrase embeddings. Baseline = round-0 retrieval from v2f_lme_mixed_7030 speaker-format cue + expand_3.

Fixed: text-embedding-3-small, gpt-5-mini, LME-hard-30 POC, vector_search_limit=50, topk_for_llm=5.

## Recall matrix

| Variant | granularity | α | aggregation | R@20 | R@50 | time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| `baseline` | sentence | 0.0 | baseline | 0.6303 | 0.8169 | 25.9 |
| `keyadd_kw_a0.05` | keyword | 0.05 | arithmetic | 0.6454 | 0.8095 | 81.5 |
| `keyadd_kw_a0.1` | keyword | 0.1 | arithmetic | 0.6553 | 0.8140 | 2.5 |
| `keyadd_kw_a0.2` | keyword | 0.2 | arithmetic | 0.6587 | 0.8066 | 5.3 |
| `keyadd_short_a0.05` | short_phrase | 0.05 | arithmetic | 0.6461 | 0.8196 | 100.1 |
| `keyadd_short_a0.1` | short_phrase | 0.1 | arithmetic | 0.6459 | 0.8176 | 5.7 |
| `keyadd_sent_a0.05` | sentence | 0.05 | arithmetic | 0.6435 | 0.8191 | 67.3 |
| `keyadd_probe_union` | keyword | 0.0 | probe_union | 0.6511 | 0.8404 | 5.7 |

## Δ vs baseline

| Variant | R@20 Δ | R@50 Δ |
| --- | --- | --- |
| `keyadd_kw_a0.05` | +0.0151 | -0.0074 |
| `keyadd_kw_a0.1` | +0.0250 | -0.0029 |
| `keyadd_kw_a0.2` | +0.0284 | -0.0103 |
| `keyadd_short_a0.05` | +0.0158 | +0.0027 |
| `keyadd_short_a0.1` | +0.0156 | +0.0007 |
| `keyadd_sent_a0.05` | +0.0132 | +0.0022 |
| `keyadd_probe_union` | +0.0208 | +0.0235 |

## Per-category R@50

| Variant | multi-session | single-session-preference | temporal-reasoning |
| --- | --- | --- | --- |
| `baseline` | 0.7398 | 0.9021 | 0.8089 |
| `keyadd_kw_a0.05` | 0.7257 | 0.9021 | 0.8009 |
| `keyadd_kw_a0.1` | 0.7183 | 0.9187 | 0.8051 |
| `keyadd_kw_a0.2` | 0.6975 | 0.9187 | 0.8037 |
| `keyadd_short_a0.05` | 0.7298 | 0.9187 | 0.8102 |
| `keyadd_short_a0.1` | 0.7382 | 0.9187 | 0.7960 |
| `keyadd_sent_a0.05` | 0.7298 | 0.9187 | 0.8089 |
| `keyadd_probe_union` | 0.7860 | 0.9313 | 0.8041 |

## Sample add terms

### keyadd_kw_a0.05

- Q `ba358f49` (multi-session): `How many years will I be when my friend Rachel gets married?`
  - add_terms: ['Rachel', 'getting married', 'my age']
  - reasoning: Turns 1 and 2 mention Rachel's wedding next year and the user's age-related concerns, which are needed to answer how old they'll be at that event.
- Q `5a7937c8` (multi-session): `How many days did I spend participating in faith-related activities in December?`
  - add_terms: ['December 10th', 'church', 'holiday food drive', 'faith-related activities']
  - reasoning: Turn 1 contains the user's December 10th church holiday food drive participation, so those exact date and activity terms are relevant to counting faith-related days in December.
- Q `6c49646a` (multi-session): `What is the total distance I covered in my four road trips?`
  - add_terms: ['1,800 miles', 'Maroon Lake', '10 miles']
  - reasoning: Turn 1 states the three-trip total (1,800 miles) and turn 4 gives the Maroon Lake distance (~10 miles), which together allow computing the four-trip total.

### keyadd_kw_a0.1

- Q `ba358f49` (multi-session): `How many years will I be when my friend Rachel gets married?`
  - add_terms: ['Rachel', 'getting married', 'my age']
  - reasoning: Turns 1 and 2 mention Rachel's wedding next year and the user's age-related concerns, which are needed to answer how old they'll be at that event.
- Q `5a7937c8` (multi-session): `How many days did I spend participating in faith-related activities in December?`
  - add_terms: ['December 10th', 'church', 'holiday food drive', 'faith-related activities']
  - reasoning: Turn 1 contains the user's December 10th church holiday food drive participation, so those exact date and activity terms are relevant to counting faith-related days in December.
- Q `6c49646a` (multi-session): `What is the total distance I covered in my four road trips?`
  - add_terms: ['1,800 miles', 'Maroon Lake', '10 miles']
  - reasoning: Turn 1 states the three-trip total (1,800 miles) and turn 4 gives the Maroon Lake distance (~10 miles), which together allow computing the four-trip total.

### keyadd_kw_a0.2

- Q `ba358f49` (multi-session): `How many years will I be when my friend Rachel gets married?`
  - add_terms: ['Rachel', 'getting married', 'my age']
  - reasoning: Turns 1 and 2 mention Rachel's wedding next year and the user's age-related concerns, which are needed to answer how old they'll be at that event.
- Q `5a7937c8` (multi-session): `How many days did I spend participating in faith-related activities in December?`
  - add_terms: ['December 10th', 'church', 'holiday food drive', 'faith-related activities']
  - reasoning: Turn 1 contains the user's December 10th church holiday food drive participation, so those exact date and activity terms are relevant to counting faith-related days in December.
- Q `6c49646a` (multi-session): `What is the total distance I covered in my four road trips?`
  - add_terms: ['1,800 miles', 'Maroon Lake', '10 miles']
  - reasoning: Turn 1 states the three-trip total (1,800 miles) and turn 4 gives the Maroon Lake distance (~10 miles), which together allow computing the four-trip total.

### keyadd_short_a0.05

- Q `ba358f49` (multi-session): `How many years will I be when my friend Rachel gets married?`
  - add_terms: ['Rachel getting married next year', 'when my friend Rachel gets married', 'how many years will I be']
  - reasoning: Turns 1 and 2 mention Rachel's wedding next year and echo the user's question about their age at that time, so probes should focus on Rachel's wedding timing and the user's age.
- Q `5a7937c8` (multi-session): `How many days did I spend participating in faith-related activities in December?`
  - add_terms: ["church's annual holiday food drive", 'helped out on December 10th', 'sorting donations and packing boxes']
  - reasoning: Only the user's turn explicitly mentions a December 10th church volunteering activity, so I extracted exact phrases describing that event.
- Q `6c49646a` (multi-session): `What is the total distance I covered in my four road trips?`
  - add_terms: ['total of 1,800 miles', 'approximately 10 miles southwest of Aspen']
  - reasoning: Turn 1 gives the three-trip total (1,800 miles) and turn 4 supplies the Maroon Lake distance to use for the fourth trip calculation.

### keyadd_short_a0.1

- Q `ba358f49` (multi-session): `How many years will I be when my friend Rachel gets married?`
  - add_terms: ['Rachel getting married next year', 'when my friend Rachel gets married', 'how many years will I be']
  - reasoning: Turns 1 and 2 mention Rachel's wedding next year and echo the user's question about their age at that time, so probes should focus on Rachel's wedding timing and the user's age.
- Q `5a7937c8` (multi-session): `How many days did I spend participating in faith-related activities in December?`
  - add_terms: ["church's annual holiday food drive", 'helped out on December 10th', 'sorting donations and packing boxes']
  - reasoning: Only the user's turn explicitly mentions a December 10th church volunteering activity, so I extracted exact phrases describing that event.
- Q `6c49646a` (multi-session): `What is the total distance I covered in my four road trips?`
  - add_terms: ['total of 1,800 miles', 'approximately 10 miles southwest of Aspen']
  - reasoning: Turn 1 gives the three-trip total (1,800 miles) and turn 4 supplies the Maroon Lake distance to use for the fourth trip calculation.

### keyadd_sent_a0.05

- Q `ba358f49` (multi-session): `How many years will I be when my friend Rachel gets married?`
  - add_terms: ["Rachel's getting married next year", 'my friend Rachel', 'age when Rachel gets married']
  - reasoning: Turns 1 and 2 mention Rachel's upcoming wedding and its timing, which are needed to compute your age then.
- Q `5a7937c8` (multi-session): `How many days did I spend participating in faith-related activities in December?`
  - add_terms: ['December 10th', "church's annual holiday food drive", 'sorting donations and packing boxes']
  - reasoning: The user explicitly reported participating in a church holiday food drive on December 10th, which answers the date-related query.
- Q `6c49646a` (multi-session): `What is the total distance I covered in my four road trips?`
  - add_terms: ['1,800 miles on my recent three road trips', 'solo trip to Durango, weekend trip to Breckenridge, family trip to Santa Fe', 'fit in Maroon Lake']
  - reasoning: Turn 1 gives the 1,800-mile total and lists the three trips; include the Maroon Lake trip to compute the four-trip total.

### keyadd_probe_union

- Q `ba358f49` (multi-session): `How many years will I be when my friend Rachel gets married?`
  - add_terms: ['Rachel', 'getting married', 'my age']
  - reasoning: Turns 1 and 2 mention Rachel's wedding next year and the user's age-related concerns, which are needed to answer how old they'll be at that event.
- Q `5a7937c8` (multi-session): `How many days did I spend participating in faith-related activities in December?`
  - add_terms: ['December 10th', 'church', 'holiday food drive', 'faith-related activities']
  - reasoning: Turn 1 contains the user's December 10th church holiday food drive participation, so those exact date and activity terms are relevant to counting faith-related days in December.
- Q `6c49646a` (multi-session): `What is the total distance I covered in my four road trips?`
  - add_terms: ['1,800 miles', 'Maroon Lake', '10 miles']
  - reasoning: Turn 1 states the three-trip total (1,800 miles) and turn 4 gives the Maroon Lake distance (~10 miles), which together allow computing the four-trip total.

## Granularity verdict

- keyword: best variant `keyadd_kw_a0.1` R@50 = 0.8140 (Δ vs baseline = -0.0029)
- short_phrase: best variant `keyadd_short_a0.05` R@50 = 0.8196 (Δ vs baseline = +0.0027)
- sentence: best variant `keyadd_sent_a0.05` R@50 = 0.8191 (Δ vs baseline = +0.0022)

## Arithmetic vs probe_union

- best arithmetic: `keyadd_short_a0.05` R@50 = 0.8196
- probe_union: R@50 = 0.8404
- Δ (union - best_arith) = +0.0208
- Δ (union - baseline) = +0.0235

## Verdict

No meaningful lift from arithmetic add at any granularity (best Δ = +0.0027). Shorter is not better under arithmetic.
Score-merge (probe_union) outperforms best arithmetic by +0.0208 — multi-probe > arithmetic.

## Outputs

- JSON: `results/keyword_add_sweep.json`
- Source: `keyword_add_steering.py`
- Caches: `cache/keyadd_llm_cache.json`, `cache/keyadd_embedding_cache.json`
