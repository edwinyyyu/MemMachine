# Cue Memoization (retrieve-and-reuse past cues)

Motivation: few-shot had the LLM imitate exemplars (and fabricate corpus-specific content); MMR/spreading activation/anchor stayed in the v2f basin. Memoization drops the LLM adaptation step and reuses exemplar cues verbatim as retrieval probes — probes come from an independent distribution (past successful runs on different questions), so their retrievals may be orthogonal to v2f.

## Exemplar bank

- Total exemplars: **49**
- Unique (dataset, conversation_id) pairs: **24**
- By dataset:
  - `advanced_23q`: 15
  - `locomo_30q`: 16
  - `puzzle_16q`: 10
  - `synthetic_19q`: 8

## Fair-backfill recall

| Arch | Dataset | base@20 | arch@20 | Δ@20 | base@50 | arch@50 | Δ@50 | W/T/L@50 | llm/q |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| meta_v2f | locomo_30q | 0.383 | 0.764 | +0.381 | 0.508 | 0.814 | +0.306 | 11/19/0 | 1.0 |
| meta_v2f | synthetic_19q | 0.569 | 0.608 | +0.039 | 0.824 | 0.831 | +0.007 | 1/18/0 | 1.0 |
| memoize_m2 | locomo_30q | 0.383 | 0.250 | -0.133 | 0.508 | 0.436 | -0.072 | 2/24/4 | 0.0 |
| memoize_m2 | synthetic_19q | 0.569 | 0.398 | -0.171 | 0.824 | 0.792 | -0.031 | 4/9/6 | 0.0 |
| memoize_m3 | locomo_30q | 0.383 | 0.250 | -0.133 | 0.508 | 0.369 | -0.139 | 3/20/7 | 0.0 |
| memoize_m3 | synthetic_19q | 0.569 | 0.392 | -0.178 | 0.824 | 0.718 | -0.105 | 4/8/7 | 0.0 |
| memoize_filtered | locomo_30q | 0.383 | 0.764 | +0.381 | 0.508 | 0.814 | +0.306 | 11/19/0 | 1.0 |
| memoize_filtered | synthetic_19q | 0.569 | 0.572 | +0.002 | 0.824 | 0.831 | +0.007 | 1/18/0 | 0.9 |
| memoize_plus_v2f | locomo_30q | 0.383 | 0.764 | +0.381 | 0.508 | 0.839 | +0.331 | 13/17/0 | 1.0 |
| memoize_plus_v2f | synthetic_19q | 0.569 | 0.622 | +0.053 | 0.824 | 0.829 | +0.005 | 5/10/4 | 1.0 |

## Query-to-nearest-exemplar cosine

Mean cosine between the new query embedding and its top-1 selected exemplar. Low values → weak match → memoization cannot cover query intent.

| Arch | Dataset | n | mean | min | max |
|---|---|---:|---:|---:|---:|
| memoize_m2 | locomo_30q | 30 | 0.2360 | 0.1562 | 0.3814 |
| memoize_m2 | synthetic_19q | 19 | 0.3812 | 0.2590 | 0.7309 |
| memoize_m3 | locomo_30q | 30 | 0.2360 | 0.1562 | 0.3814 |
| memoize_m3 | synthetic_19q | 19 | 0.3812 | 0.2590 | 0.7309 |
| memoize_filtered | locomo_30q | 0 | 0.0000 | 0.0000 | 0.0000 |
| memoize_filtered | synthetic_19q | 2 | 0.6337 | 0.5364 | 0.7309 |
| memoize_plus_v2f | locomo_30q | 30 | 0.2360 | 0.1562 | 0.3814 |
| memoize_plus_v2f | synthetic_19q | 19 | 0.3812 | 0.2590 | 0.7309 |

## Orthogonality vs v2f (K=50)

Fraction of gold turns the variant found that v2f did NOT find.

| Arch | Dataset | gold_found | novel_vs_v2f | frac_novel |
|---|---|---:|---:|---:|
| memoize_m2 | locomo_30q | 16 | 2 | 0.125 |
| memoize_m2 | synthetic_19q | 110 | 11 | 0.100 |
| memoize_m3 | locomo_30q | 14 | 3 | 0.214 |
| memoize_m3 | synthetic_19q | 107 | 16 | 0.149 |
| memoize_filtered | locomo_30q | 33 | 0 | 0.000 |
| memoize_filtered | synthetic_19q | 116 | 0 | 0.000 |
| memoize_plus_v2f | locomo_30q | 35 | 2 | 0.057 |
| memoize_plus_v2f | synthetic_19q | 116 | 11 | 0.095 |

## Per-cue gold hit rates (top-1 and any-top-K)

How often does a reused cue's top-1 retrieval equal a gold turn? Compared across memoize vs v2f cue sources.

| Dataset | Cue source | #cues | top1_hit | any_topK_hit |
|---|---|---:|---:|---:|
| locomo_30q | memoize_m2 | 120 | 0.008 | 0.092 |
| locomo_30q | memoize_m3 | 180 | 0.006 | 0.094 |
| locomo_30q | meta_v2f | 60 | 0.283 | 0.850 |
| synthetic_19q | memoize_m2 | 76 | 0.158 | 0.539 |
| synthetic_19q | memoize_m3 | 114 | 0.167 | 0.570 |
| synthetic_19q | meta_v2f | 38 | 0.263 | 0.974 |

## Top gain/loss categories for memoize_m2 (combined datasets)

Gaining:
- proactive (n=4): Δ@50=+0.063 W/T/L=2/1/1
- inference (n=3): Δ@50=+0.061 W/T/L=1/2/0
Losing:
- completeness (n=4): Δ@50=-0.138 W/T/L=0/2/2
- conjunction (n=3): Δ@50=-0.143 W/T/L=0/1/2

## Qualitative samples (memoize_m2)

Each row: new_q → nearest_exemplar_q (sim) → reused_cue → gold_found_turn

- **locomo_30q** `locomo_temporal`
  - new_q: When is Melanie planning on going camping?
  - nearest_exemplar: Where is the company retreat being held and when does it start? (sim=0.381, from `puzzle_16q`)
  - reused_cue: Diana's email says the retreat starts at 10am on Friday
  - gold_found_turn_id: 24
- **locomo_30q** `locomo_multi_hop` **(novel vs v2f)**
  - new_q: Would Caroline still want to pursue counseling as a career if she hadn't received support growing up?
  - nearest_exemplar: Based on the conversation, does the user follow any specific dietary pattern? What evidence supports your conclusion? (sim=0.241, from `puzzle_16q`)
  - reused_cue: I learned a lot about umami sources like miso, nutritional yeast, mushrooms, and soy sauce to create that deep savory flavor without needing any meat.
  - gold_found_turn_id: 39
- **locomo_30q** `locomo_single_hop` **(novel vs v2f)**
  - new_q: What activities does Melanie partake in?
  - nearest_exemplar: Based on all the patterns in our conversations, are there any health concerns about the user I should flag? (sim=0.261, from `puzzle_16q`)
  - reused_cue: I sleep later on weekends, drink more caffeine on Mondays, haven't changed my desk setup or taken meds, and I said I'd try stretches and see my doctor if it doesn't improve
  - gold_found_turn_id: 174
- **locomo_30q** `locomo_temporal`
  - new_q: When is Caroline going to the transgender conference?
  - nearest_exemplar: Where is the company retreat being held and when does it start? (sim=0.306, from `puzzle_16q`)
  - reused_cue: Diana's email says the retreat starts at 10am on Friday
  - gold_found_turn_id: 88
- **locomo_30q** `locomo_single_hop`
  - new_q: Where has Melanie camped?
  - nearest_exemplar: Where is the company retreat being held and when does it start? (sim=0.290, from `puzzle_16q`)
  - reused_cue: Lakewood Resort, April 18-20
  - gold_found_turn_id: 63

## Verdict

- memoize_m2 mean Δ@50 (vs cosine baseline): -0.052, arch@50=0.614
- memoize_m3 mean Δ@50 (vs cosine baseline): -0.122, arch@50=0.544
- memoize_filtered mean Δ@50 (vs cosine baseline): +0.157, arch@50=0.823
- memoize_plus_v2f mean Δ@50 (vs cosine baseline): +0.168, arch@50=0.834
- meta_v2f mean Δ@50 (vs cosine baseline): +0.157, arch@50=0.823

**Verdict: marginal — supplement only.** memoize_plus_v2f narrowly beats v2f (+0.011); weak signal, probes may still overlap v2f basin.
