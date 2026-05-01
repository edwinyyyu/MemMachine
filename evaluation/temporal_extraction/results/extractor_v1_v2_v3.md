# Extractor v1 / v2 / v3 — Cache-Only Evaluation

All three versions evaluated against their prior cached extractions (no new LLM calls). Scorer: multi-axis α=0.5 β=0.35 γ=0.15.

## Extraction F1

| Version | Combined F1 | Combined P | Combined R | Docs F1 | Queries F1 | Cache misses |
|---|---:|---:|---:|---:|---:|---:|
| v1 | 0.809 | 0.860 | 0.763 | 0.816 | 0.800 | 35 |
| v2 | 0.875 | 0.902 | 0.851 | 0.821 | 0.941 | 178 |
| v3 | 0.706 | 0.633 | 0.799 | 0.637 | 0.817 | 188 |

## Failure-case recovery (v1-missed surfaces)

| Version | Recovered | Total | Rate |
|---|---:|---:|---:|
| v1 | 0 | 38 | 0.00 |
| v2 | 33 | 38 | 0.87 |
| v3 | 24 | 38 | 0.63 |

### Per-surface recovery (docs + queries)

| Surface | v1 | v2 | v3 |
|---|---:|---:|---:|
| `a few weeks ago` | 0/1 | 0/1 | 0/1 |
| `about 20 years ago` | 0/1 | 1/1 | 1/1 |
| `about a month ago` | 0/1 | 0/1 | 0/1 |
| `april 3, 2026` | 0/1 | 1/1 | 0/1 |
| `april 4, 2026` | 0/1 | 1/1 | 0/1 |
| `april 7, 2026` | 0/1 | 1/1 | 1/1 |
| `april 8, 2026` | 0/1 | 1/1 | 1/1 |
| `around 2010` | 0/1 | 1/1 | 0/1 |
| `during april 2026` | 0/1 | 1/1 | 1/1 |
| `during the first week of may 2026` | 0/1 | 1/1 | 1/1 |
| `last month` | 0/9 | 8/9 | 1/9 |
| `last week` | 0/1 | 1/1 | 1/1 |
| `last year` | 0/1 | 1/1 | 1/1 |
| `next week` | 0/1 | 1/1 | 1/1 |
| `the 2010s` | 0/2 | 1/2 | 2/2 |
| `the 60s` | 0/1 | 0/1 | 0/1 |
| `the 90s` | 0/1 | 1/1 | 1/1 |
| `the first week of april 2026` | 0/10 | 10/10 | 10/10 |
| `this month` | 0/1 | 1/1 | 1/1 |
| `this week` | 0/1 | 1/1 | 1/1 |

## Axis-gap surfaces (bare months / quarters / parts-of-day)

Capture rate for the multi-axis experiment's flagged surfaces (March, Q2, June weekends, evening, Summer, winter, Autumn, Q4, October) across docs + queries.

| Version | Docs captured | Queries captured | Total rate |
|---|---:|---:|---:|
| v1 | 3/10 | 4/14 | 0.29 |
| v2 | 3/10 | 4/14 | 0.29 |
| v3 | 3/10 | 4/14 | 0.29 |

### Per-surface axis-gap rate (docs + queries combined)

| Surface | v1 | v2 | v3 |
|---|---:|---:|---:|
| `March` | 7/9 | 7/9 | 7/9 |
| `Q2` | 0/2 | 0/2 | 0/2 |
| `June weekends` | 0/1 | 0/1 | 0/1 |
| `evening` | 0/3 | 0/3 | 0/3 |
| `Summer` | 0/3 | 0/3 | 0/3 |
| `winter` | 0/2 | 0/2 | 0/2 |
| `Autumn` | 0/1 | 0/1 | 0/1 |
| `Q4` | 0/1 | 0/1 | 0/1 |
| `October` | 0/2 | 0/2 | 0/2 |

## Downstream retrieval (base 55 queries)

| Version | R@5 | R@10 | MRR | NDCG@10 |
|---|---:|---:|---:|---:|
| v1 | 0.498 | 0.543 | 0.697 | 0.534 |
| v2 | 0.561 | 0.615 | 0.844 | 0.646 |
| v3 | 0.571 | 0.631 | 0.737 | 0.611 |

## Axis-subset retrieval (20 axis queries)

| Version | R@5 | R@10 | MRR | NDCG@10 |
|---|---:|---:|---:|---:|
| v1 | 0.242 | 0.583 | 0.177 | 0.242 |
| v2 | 0.000 | 0.317 | 0.097 | 0.111 |
| v3 | 0.242 | 0.583 | 0.177 | 0.242 |

## Cache coverage caveat

- v1 `cache/llm_cache.json`: 524 entries. Covers base+disc fully; 35 cache misses come from axis/era/utterance items (v1 was run only on the original 224 items before axis/era/utt corpora existed).
- v2 `cache/extractor_v2/llm_cache.json`: 415 entries (10% empty). Covers base+disc **fully** (0 misses on 174 items). Axis (70) + era (70) items **not cached** → retrieval numbers on axis subset are NOT comparable.
- v3 `cache/extractor_v3/llm_cache.json`: 646 entries (21% empty). Covers base fully, disc has 10 stale misses. Axis + era not cached. The 21% empty-response rate is the CoT-token-burn symptom.

Re-reading the axis-retrieval rows: v2's 0.000 R@5 is purely a cache-coverage artifact (all axis queries returned empty extractions). v1 and v3 happen to score 0.242 because some axis queries contain date literals that lower-variance extractors still catch. None of the three versions is actually solving the bare-month / Q2 / season gap — NONE of them had axis-corpus extractions produced.

## v4 / v5 / v6 hang diagnosis

Cache inspection shows the symptom clearly:

| Version | Pass-1 prompt | Empty-response rate |
|---|---|---:|
| v2 | gazetteer + few-shot, single pass | 10% |
| v3 | adds chain-of-thought scratchpad (`candidates` -> `review` -> `refs`) | 21% |
| v4 | v3 + recovery pass (second LLM call per item with longer prompt) | 8% primary, but recovery call doubles total spend |

Root cause is **CoT token overflow on gpt-5-mini**: the longer and more-reasoning-heavy the Pass-1 system prompt, the more tokens the model burns on hidden reasoning before producing any output. When the budget runs out the response is empty (21% for v3, and we observe a separate stall pattern when v4's recovery pass is added on top of v3). v5 adds regex-triage (a THIRD LLM call per item) and v6 adds Pass-2 validation-retry, compounding the wall-clock hang.

This is a **prompt-length / CoT-budget** issue, not a structural bug. v2's tighter single-pass prompt threads the needle: gazetteer + few-shot gives the LLM all the lift of v3 without triggering chain-of-thought overflow.

## Cost

- All three versions ran cache-only: total uncached LLM requests stubbed = **401** (all returned empty strings; no network calls).
- LLM token spend: **$0.00**.
