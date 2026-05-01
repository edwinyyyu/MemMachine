# F11 — Temporal query rewriting + fusion

Compares three retrieval modes on five subsets. For each query we ask gpt-5-mini for up to 5 temporal paraphrases (temperature=0), extract each variant with the same base extractor, and fuse the resulting ranked lists.

## Per-subset metrics

| subset | mode | n | R@5 | R@10 | MRR | NDCG@10 |
|---|---|---:|---:|---:|---:|---:|
| base | baseline | 50 | 0.462 | 0.507 | 0.627 | 0.467 |
| base | rrf | 50 | 0.487 | 0.531 | 0.673 | 0.512 |
| base | max | 50 | 0.487 | 0.531 | 0.663 | 0.504 |
|  |  |  |  |  |  |  |
| axis | baseline | 20 | 0.050 | 0.050 | 0.038 | 0.025 |
| axis | rrf | 20 | 0.075 | 0.100 | 0.047 | 0.047 |
| axis | max | 20 | 0.075 | 0.100 | 0.047 | 0.047 |
|  |  |  |  |  |  |  |
| utterance | baseline | 10 | 0.200 | 0.220 | 0.222 | 0.210 |
| utterance | rrf | 10 | 0.267 | 0.287 | 0.270 | 0.263 |
| utterance | max | 10 | 0.267 | 0.287 | 0.270 | 0.263 |
|  |  |  |  |  |  |  |
| era | baseline | 20 | 0.000 | 0.000 | 0.014 | 0.000 |
| era | rrf | 20 | 0.000 | 0.000 | 0.014 | 0.000 |
| era | max | 20 | 0.000 | 0.000 | 0.014 | 0.000 |
|  |  |  |  |  |  |  |
| allen | baseline | 20 | 0.000 | 0.000 | 0.010 | 0.000 |
| allen | rrf | 20 | 0.000 | 0.000 | 0.010 | 0.000 |
| allen | max | 20 | 0.000 | 0.000 | 0.010 | 0.000 |
|  |  |  |  |  |  |  |
| all | baseline | 120 | 0.217 | 0.238 | 0.290 | 0.216 |
| all | rrf | 120 | 0.238 | 0.262 | 0.315 | 0.243 |
| all | max | 120 | 0.238 | 0.262 | 0.311 | 0.240 |
|  |  |  |  |  |  |  |

## Lift over baseline (R@5)

| subset | Δ RRF | Δ max |
|---|---:|---:|
| base | +0.025 | +0.025 |
| axis | +0.025 | +0.025 |
| utterance | +0.067 | +0.067 |
| era | +0.000 | +0.000 |
| allen | +0.000 | +0.000 |
| all | +0.020 | +0.020 |

## Cost

- Rewriter (gpt-5-mini): in=0, out=0 -> $0.0000
- Variant extraction (gpt-5-mini, cached vs base): in=115975, out=189930 -> $0.4089
- Doc extraction (amortised): in=0, out=0 -> $0.0000
- **New cost attributable to rewriting**: $0.4089 across 125 queries = $3.27 / 1000 queries

## Sample rewrites

- **q_rel_day_0** `What did I do yesterday?`
  - `What did I do on Thursday, April 22, 2026?`
  - `What did I do yesterday (April 22, 2026)?`
  - `What did I do the day before today?`
  - `Anything I did on 2026-04-22?`
  - `What was I up to yesterday?`
- **q_rel_day_1** `What did I do two weeks ago?`
  - `What did I do on Monday, April 6, 2026?`
  - `What was I up to two weeks before today?`
  - `Anything I did during the week beginning April 6, 2026?`
  - `What happened for me around April 6, 2026 (two weeks ago)?`
  - `What did I do a fortnight ago?`
- **q_rel_day_2** `What did I do last week?`
  - `What did I do the week of April 6–12, 2026?`
  - `What did I do in the week before this one?`
  - `What happened for me between April 6 and April 12, 2026?`
  - `Anything I did last week (Apr 6–12, 2026)?`
- **q_rel_day_3** `What did I do three days ago?`
  - `What did I do on April 11, 2026?`
  - `What was I doing three days ago?`
  - `Any activities from three days back (April 11, 2026)?`
  - `What happened 72 hours ago?`
- **q_rel_day_4** `What did I do a month ago?`
  - `What did I do on March 11, 2026?`
  - `What was I doing about a month ago, around March 11, 2026?`
  - `What did I do roughly four weeks ago?`
  - `What happened to me in early March 2026?`
- **q_rel_day_5** `What did I do last month?`
  - `What did I do in March 2026?`
  - `What did I do during March 2026?`
  - `What activities did I do last month (March 2026)?`
  - `What was I up to in the month of March 2026?`
  - `Anything I did during the previous calendar month, March 2026?`
- **q_rel_day_6** `What did I do a week ago?`
  - `What did I do on March 27, 2026?`
  - `What was I up to a week ago?`
  - `What did I do seven days ago?`
  - `What did I do the previous Saturday (March 27, 2026)?`
- **q_rel_day_7** `What did I do yesterday?`
  - `What did I do on March 28, 2026?`
  - `What did I do the day before today?`
  - `What was I up to yesterday?`
  - `Anything I did one day ago (March 28, 2026)?`
- **q_rel_day_8** `What did I do two weeks ago?`
  - `what did I do on Tuesday, March 10, 2026?`
  - `what was I doing two weeks before March 24, 2026?`
  - `anything I did the week of March 8–14, 2026?`
  - `what was I up to a fortnight ago?`
- **q_rel_day_9** `What did I do last year?`
  - `What did I do in 2025?`
  - `What was I up to last year?`
  - `What activities did I have during the year 2025?`
  - `Anything I did from Jan–Dec 2025?`
- **q_fuzzy_0** `What happened in 2015?`
  - `What happened in 2015?`
  - `What occurred during the year 2015?`
  - `What took place back in 2015?`
  - `What happened eleven years ago, in 2015?`
  - `Events from the year 2015?`
- **q_fuzzy_1** `What happened in the 90s?`
  - `1990 to 1999, what happened?`
  - `the 1990s — what occurred then?`
  - `the decade of the nineties, what events took place?`
  - `what happened during the nineties (1990s)?`
  - `early-to-late 1990s: what happened then?`

## Where rewriting helped most (top 10 by ΔR@5)

- `q_fuzzy_1` [base] `What happened in the 90s?` — Δrrf=+1.00, Δmax=+1.00
  - variant: `1990 to 1999, what happened?`
  - variant: `the 1990s — what occurred then?`
  - variant: `the decade of the nineties, what events took place?`
  - variant: `what happened during the nineties (1990s)?`
  - variant: `early-to-late 1990s: what happened then?`
- `q_utt_5` [utterance] `What happened in the 90s?` — Δrrf=+0.67, Δmax=+0.67
  - variant: `1990 to 1999, what happened?`
  - variant: `the 1990s — what occurred then?`
  - variant: `the decade of the nineties, what events took place?`
  - variant: `what happened during the nineties (1990s)?`
  - variant: `early-to-late 1990s: what happened then?`
- `axis_q_thu` [axis] `What do I do on Thursdays?` — Δrrf=+0.50, Δmax=+0.50
  - variant: `What do I do every Thursday?`
  - variant: `What are my Thursday tasks each week?`
  - variant: `What's scheduled for Thursdays?`
  - variant: `On Thursday each week, what should I be doing?`
  - variant: `What obligations do I have on Thursdays?`
- `q_interval_4` [base] `What happened during April 2026?` — Δrrf=+0.15, Δmax=+0.15
  - variant: `What happened in April 2026?`
  - variant: `What occurred during the month of April 2026?`
  - variant: `Anything notable this April (2026)?`
  - variant: `Events from April 2026`
  - variant: `What took place in April of 2026?`
- `q_interval_1` [base] `What happened between March 1 and March 15, 2026?` — Δrrf=+0.09, Δmax=+0.09
  - variant: `What happened from March 1 to March 15, 2026?`
  - variant: `What occurred during the first half of March 2026 (Mar 1–15, 2026)?`
  - variant: `Anything that took place between March 1–15, 2026?`
  - variant: `What events happened in early March 2026, specifically March 1 through March 15?`
  - variant: `What happened roughly six weeks ago, between Mar 1 and Mar 15, 2026?`

## Where rewriting hurt (top 10 by ΔR@5)


## Analysis

- RRF positive subsets: ['base', 'axis', 'utterance']; negative: none.
- max-of positive subsets: ['base', 'axis', 'utterance']; negative: none.
- Wall time total: 208.3s.
