# Extractor Improvement Plan

Current baseline (v1):
- Overall F1 = 0.860 (precision 0.870, recall 0.851)
- Docs F1 0.913, Queries F1 0.809 (query gap is notable)
- Resolution MAE median 0s, mean 5.6M s, p95 30.8M s (decade-bracket outliers)
- Known misses: 6/10 "last month"/"earlier this month" queries had 0 extracted times

The extractor is the ceiling on every downstream experiment. Improving it gates everything.

## Failure-mode analysis

1. **Named-relative calendar units** — "last month", "this month", "earlier this month", "last year", "this week" — frequently missed when appearing in short queries.
2. **Embedded references** — "I saw her last Tuesday" — the prepositional time phrase sometimes missed.
3. **Decade mid-points** — "the 90s" resolves to different canonical centers across runs (1995 vs 1990 vs some other date).
4. **Weekday alignment errors** — "next Thursday" sometimes resolves to a date that isn't actually Thursday.
5. **Month-length arithmetic** — "2 months ago" off by 1-3 days depending on which month boundaries the LLM chose.
6. **Precision-anchor detection** — "exactly 2 weeks ago" vs "2 weeks ago" — the LLM doesn't always distinguish.

## Techniques to test (stacked)

### v2 — Prompt engineering
- **Trigger gazetteer**: explicit list of temporal trigger words/phrases in Pass 1 prompt ("last/this/next + week/month/year/quarter", "N days/weeks/months/years/decades ago/from now/later/earlier", "morning/afternoon/evening/night/dawn/dusk", month names, day-of-week names, "recently/lately/soon/earlier/later", "ago/hence", era words).
- **Reference time full context**: "Today is Thursday, April 23, 2026. Yesterday = Wed Apr 22. Last week = Apr 13-19. Last month = March 2026. Last year = 2025." instead of just an ISO string.
- **Few-shot examples** of hard cases: named-relative, embedded reference, era, recurrence with cancellation.

### v3 = v2 + chain-of-thought
- Pass 1 instructed to first *list* every potentially-temporal phrase (even borderline), then classify each as real-time or not, then output final list.
- Reasoning helps catch "last month" type misses because the model sees it in its scan pass.

### v4 = v3 + recovery pass
- After v3's Pass 1 output, a second call: "Here's the text and the times you found. Did you miss any? Think carefully about implicit times, relative references, and phrases like 'last month'/'earlier this year'/'when I was in college'."
- Union the recovered spans with the original extraction.

### v5 = v4 + regex pre-pass
- Run regex over text to find candidate spans for known patterns (numbered-relative, named-relative, year-literals, month-day literals, day-of-week, ISO dates, time-of-day).
- Each candidate is presented to the LLM for resolution ("is this a real time reference? if so, resolve it").
- Ensures coverage of the patterns the LLM keeps missing.

### v6 = v5 + deterministic validation & retry
- After Pass 2 resolution, parse every ISO datetime. If `next Thursday` resolves to a non-Thursday, retry Pass 2 with the error message.
- Month-length check: if a -2 months arithmetic lands on a day that doesn't exist (e.g., Feb 30), correct to month-end and retry.
- Weekday alignment: if surface says "Monday" but resolved date is Tuesday, retry with correction hint.

### Upper bound — gpt-5 (not mini)
- Run v5's prompt with gpt-5 on the hardest subset (20 queries that v1 missed).
- Establishes ceiling: is the failure prompt shape or model capability?

## Evaluation

For each version, measure on (docs + queries + discriminator set, combined):
- Extraction F1 (match by surface-span overlap ≥ 50%)
- Precision, recall breakdown
- Resolution MAE (median, mean, P95) on matched pairs
- Failure-case-specific recall (did "last month" fire?)
- Downstream retrieval R@5 & MRR using each extractor's output through the current scorer

Output:
- `results/extractor_improvements.md` — ranked versions, technique ablation, failure-case breakdown
- `results/extractor_improvements.json` — raw per-version metrics
- New extractor files (`extractor_v2.py` ... `extractor_v6.py`) preserved for later comparison

## Decision criteria

Ship the version that maximizes downstream R@5 × precision, subject to:
- F1 doesn't regress on any subset
- Cost stays below 3× v1 (v2 adds prompt length; v4/v6 add calls)

If upper-bound gpt-5 doesn't beat v6 by more than 3%, the bottleneck is
prompt shape, not model capability; stay on gpt-5-mini. If gpt-5 beats by
more, flag for future migration.
