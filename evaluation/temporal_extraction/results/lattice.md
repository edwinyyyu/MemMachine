# Lattice Inverted Index — Evaluation

Corpus: 137 unique docs, 135 queries. Wall: 757.9s. LLM cost: $0.5266.

## Tag cardinality

- Docs tagged: **134**
- Total tag rows: **1850**
- Unique tags: **1168**
- Avg tags/doc: **13.81** (min 1, max 130)
- Avg cells visited per query: **109.3** (max 3270)

Top-15 hub tags (most docs share these):

| tag | docs |
|---|---:|
| `weekend:no` | 59 |
| `season:spring` | 54 |
| `month_of_year:April` | 37 |
| `weekday:Thursday` | 24 |
| `month_of_year:March` | 17 |
| `weekend:yes` | 16 |
| `season:winter` | 15 |
| `weekday:Wednesday` | 12 |
| `weekday:Friday` | 11 |
| `weekday:Tuesday` | 11 |
| `decade:2010s` | 10 |
| `season:autumn` | 10 |
| `day_of_month:23` | 9 |
| `part_of_day:morning` | 9 |
| `season:summer` | 9 |

## Per-variant metrics

### Primary subsets

| Variant | lat R@5 | lat R@10 | lat MRR | lat NDCG | base R@5 | axis R@5 | adv R@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| V-LATTICE-ONLY | 0.662 | 0.672 | 0.768 | 0.652 | 0.307 | 0.900 | 0.576 |
| V7 | 0.785 | 0.845 | 0.864 | 0.809 | 0.428 | 0.858 | 0.890 |
| V7L | 0.785 | 0.835 | 0.864 | 0.802 | 0.433 | 0.858 | 0.848 |

### Lattice sub-subsets (R@5)

| Variant | narrow_query_broad_doc | broad_query_narrow_doc | same_precision | same_precision_s8 | cyclical | s8_crossdoc |
|---|---:|---:|---:|---:|---:|---:|
| V-LATTICE-ONLY | 0.550 | 0.400 | 1.000 | 1.000 | 0.900 | 0.500 |
| V7 | 0.900 | 0.640 | 1.000 | 1.000 | 0.800 | 0.500 |
| V7L | 0.900 | 0.640 | 1.000 | 1.000 | 0.800 | 0.500 |

### Adversarial by category (focus: A3, A6, S8) — R@5

| Variant | A3 | A6 | S8 | A1 | A2 | A4 | A5 | A7 | A8 | A9 | R1 | R2 | R3 | R4 | R5 | R6 | R7 | S1 | S2 | S3 | S4 | S5 | S6 | S7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V-LATTICE-ONLY | 0.000 | 1.000 | 0.500 | 1.000 | 0.000 | 0.000 | 0.333 | - | 1.000 | 1.000 | 0.500 | 1.000 | 1.000 | 0.000 | 0.667 | 1.000 | - | 0.667 | 0.000 | - | 1.000 | 0.750 | 1.000 | - |
| V7 | 0.500 | 1.000 | 0.500 | 1.000 | 1.000 | 0.750 | 1.000 | - | 1.000 | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - | 0.667 | 1.000 | - | 1.000 | 0.750 | 1.000 | - |
| V7L | 0.000 | 1.000 | 0.500 | 1.000 | 1.000 | 0.750 | 1.000 | - | 1.000 | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | - | 0.667 | 0.000 | - | 1.000 | 1.000 | 1.000 | - |

## Sample lattice tags (lattice-synth docs)

- **lat_doc_day_jan1_1999** (day): `On January 1, 1999 we hosted a millennium-eve countdown party with fri`
  tags: ['day:1999-01-01', 'day_of_month:1', 'day_of_month:31', 'minute:1999-12-31T18:00', 'minute:1999-12-31T18:01', 'minute:1999-12-31T18:02', 'minute:1999-12-31T18:03', 'minute:1999-12-31T18:04', 'minute:1999-12-31T18:05', 'minute:1999-12-31T18:06', 'minute:1999-12-31T18:07', 'minute:1999-12-31T18:08', 'minute:1999-12-31T18:09', 'minute:1999-12-31T18:10', 'minute:1999-12-31T18:11', 'minute:1999-12-31T18:12', 'minute:1999-12-31T18:13', 'minute:1999-12-31T18:14', 'minute:1999-12-31T18:15', 'minute:1999-12-31T18:16', 'minute:1999-12-31T18:17', 'minute:1999-12-31T18:18', 'minute:1999-12-31T18:19', 'minute:1999-12-31T18:20', 'minute:1999-12-31T18:21', 'minute:1999-12-31T18:22', 'minute:1999-12-31T18:23', 'minute:1999-12-31T18:24', 'minute:1999-12-31T18:25', 'minute:1999-12-31T18:26', 'minute:1999-12-31T18:27', 'minute:1999-12-31T18:28', 'minute:1999-12-31T18:29', 'minute:1999-12-31T18:30', 'minute:1999-12-31T18:31', 'minute:1999-12-31T18:32', 'minute:1999-12-31T18:33', 'minute:1999-12-31T18:34', 'minute:1999-12-31T18:35', 'minute:1999-12-31T18:36', 'minute:1999-12-31T18:37', 'minute:1999-12-31T18:38', 'minute:1999-12-31T18:39', 'minute:1999-12-31T18:40', 'minute:1999-12-31T18:41', 'minute:1999-12-31T18:42', 'minute:1999-12-31T18:43', 'minute:1999-12-31T18:44', 'minute:1999-12-31T18:45', 'minute:1999-12-31T18:46', 'minute:1999-12-31T18:47', 'minute:1999-12-31T18:48', 'minute:1999-12-31T18:49', 'month_of_year:December', 'month_of_year:January', 'season:winter', 'weekday:Friday', 'weekend:no']
- **lat_doc_day_mar15_2015** (day): `March 15, 2015 was the day I submitted the dissertation.`
  tags: ['day:2015-03-15', 'day_of_month:15', 'month_of_year:March', 'season:spring', 'weekday:Sunday', 'weekend:yes']
- **lat_doc_day_jul4_2020** (day): `On July 4, 2020 the neighborhood block party was cancelled due to rain`
  tags: ['day:2020-07-04', 'day_of_month:4', 'month_of_year:July', 'season:summer', 'weekday:Saturday', 'weekend:yes']
- **lat_doc_day_dec25_1995** (day): `December 25, 1995 was the Christmas we all drove to the lake house.`
  tags: ['day:1995-12-25', 'day:1995-12-26', 'day:2026-12-25', 'day_of_month:25', 'month_of_year:December', 'season:winter', 'weekday:Friday', 'weekday:Monday', 'weekend:no']
- **lat_doc_day_sep11_2001** (day): `On September 11, 2001 I was stuck in a train tunnel for an hour.`
  tags: ['day:2001-09-11', 'day_of_month:11', 'month_of_year:September', 'season:autumn', 'weekday:Tuesday', 'weekend:no']
- **lat_doc_mar2020** (month): `March 2020 was when everyone started working from home for the first t`
  tags: ['month:2020-03', 'month:2020-04', 'month_of_year:March', 'season:spring']
- **lat_doc_jun2018** (month): `June 2018 was the month I moved into the new apartment.`
  tags: ['minute:2018-06-01T00:00', 'minute:2018-06-01T00:01', 'minute:2018-06-01T00:02', 'minute:2018-06-01T00:03', 'minute:2018-06-01T00:04', 'minute:2018-06-01T00:05', 'minute:2018-06-01T00:06', 'minute:2018-06-01T00:07', 'minute:2018-06-01T00:08', 'minute:2018-06-01T00:09', 'minute:2018-06-01T00:10', 'minute:2018-06-01T00:11', 'minute:2018-06-01T00:12', 'minute:2018-06-01T00:13', 'minute:2018-06-01T00:14', 'minute:2018-06-01T00:15', 'minute:2018-06-01T00:16', 'minute:2018-06-01T00:17', 'minute:2018-06-01T00:18', 'minute:2018-06-01T00:19', 'minute:2018-06-01T00:20', 'minute:2018-06-01T00:21', 'minute:2018-06-01T00:22', 'minute:2018-06-01T00:23', 'minute:2018-06-01T00:24', 'minute:2018-06-01T00:25', 'minute:2018-06-01T00:26', 'minute:2018-06-01T00:27', 'minute:2018-06-01T00:28', 'minute:2018-06-01T00:29', 'minute:2018-06-01T00:30', 'minute:2018-06-01T00:31', 'minute:2018-06-01T00:32', 'minute:2018-06-01T00:33', 'minute:2018-06-01T00:34', 'minute:2018-06-01T00:35', 'minute:2018-06-01T00:36', 'minute:2018-06-01T00:37', 'minute:2018-06-01T00:38', 'minute:2018-06-01T00:39', 'minute:2018-06-01T00:40', 'minute:2018-06-01T00:41', 'minute:2018-06-01T00:42', 'minute:2018-06-01T00:43', 'minute:2018-06-01T00:44', 'minute:2018-06-01T00:45', 'minute:2018-06-01T00:46', 'minute:2018-06-01T00:47', 'minute:2018-06-01T00:48', 'minute:2018-06-01T00:49', 'month_of_year:June', 'season:summer']
- **lat_doc_nov1999** (month): `November 1999 was rainy and cold; I remember it well.`
  tags: ['month:1999-11', 'month:1999-12', 'month_of_year:November', 'season:autumn']
- **lat_doc_y2015_tough** (year): `2015 was a tough year — lots of transitions and moves.`
  tags: ['year:2015', 'year:2016']
- **lat_doc_y2012_wife_met** (year): `I met my wife at the 2012 retreat and we talked the whole weekend.`
  tags: ['day:2012-06-09', 'day:2012-06-10', 'day:2012-06-11', 'day:2012-06-15', 'day:2012-06-16', 'day:2012-06-17', 'day:2012-06-18', 'day_of_month:9', 'month_of_year:June', 'season:summer', 'weekday:Saturday', 'weekend:yes', 'year:2012', 'year:2013']

## Sample lattice query expansions

- **lat_q_mar15_2015** (narrow_query_broad_doc): `What happened on March 15, 2015?`  
  native_abs=['day:2015-03-15'], cyclical=['day_of_month:15', 'month_of_year:March', 'season:spring', 'weekday:Sunday', 'weekend:yes'], expanded=36, matched_docs=77
- **lat_q_jan1_1999** (narrow_query_broad_doc): `What happened on January 1, 1999?`  
  native_abs=['day:1999-01-01'], cyclical=['day_of_month:1', 'month_of_year:January', 'season:winter', 'weekday:Friday', 'weekend:no'], expanded=36, matched_docs=74
- **lat_q_jul4_2020** (narrow_query_broad_doc): `What did I do on July 4, 2020?`  
  native_abs=['day:2020-07-04'], cyclical=['day_of_month:4', 'month_of_year:July', 'season:summer', 'weekday:Saturday', 'weekend:yes'], expanded=36, matched_docs=33
- **lat_q_dec25_1995** (narrow_query_broad_doc): `What happened on December 25, 1995?`  
  native_abs=['day:1995-12-25'], cyclical=['day_of_month:25', 'month_of_year:December', 'season:winter', 'weekday:Monday', 'weekend:no'], expanded=36, matched_docs=74
- **lat_q_sep11_2001** (narrow_query_broad_doc): `What happened on September 11, 2001?`  
  native_abs=['day:2001-09-11'], cyclical=['day_of_month:11', 'month_of_year:September', 'season:autumn', 'weekday:Tuesday', 'weekend:no'], expanded=36, matched_docs=75
- **lat_q_the_90s** (broad_query_narrow_doc): `Anything from the 90s?`  
  native_abs=['decade:1990s', 'decade:2000s', 'decade:1990s', 'decade:2000s'], cyclical=[], expanded=48, matched_docs=26
- **lat_q_in_2015** (broad_query_narrow_doc): `Anything that happened in 2015?`  
  native_abs=['year:2015', 'year:2015', 'year:2016'], cyclical=[], expanded=19, matched_docs=28
- **lat_q_the_80s** (broad_query_narrow_doc): `What about the 1980s?`  
  native_abs=['decade:1980s', 'decade:1990s'], cyclical=[], expanded=23, matched_docs=11
- **lat_q_in_2020** (broad_query_narrow_doc): `Anything from 2020?`  
  native_abs=['year:2020', 'year:2020', 'year:2021'], cyclical=[], expanded=19, matched_docs=20
- **lat_q_the_2000s** (broad_query_narrow_doc): `What about the 2000s?`  
  native_abs=['minute:2000-01-01T00:00', 'minute:2000-01-01T00:01', 'minute:2000-01-01T00:02', 'minute:2000-01-01T00:03', 'minute:2000-01-01T00:04', 'minute:2000-01-01T00:05', 'minute:2000-01-01T00:06', 'minute:2000-01-01T00:07', 'minute:2000-01-01T00:08', 'minute:2000-01-01T00:09', 'minute:2000-01-01T00:10', 'minute:2000-01-01T00:11', 'minute:2000-01-01T00:12', 'minute:2000-01-01T00:13', 'minute:2000-01-01T00:14', 'minute:2000-01-01T00:15', 'minute:2000-01-01T00:16', 'minute:2000-01-01T00:17', 'minute:2000-01-01T00:18', 'minute:2000-01-01T00:19', 'minute:2000-01-01T00:20', 'minute:2000-01-01T00:21', 'minute:2000-01-01T00:22', 'minute:2000-01-01T00:23', 'minute:2000-01-01T00:24', 'minute:2000-01-01T00:25', 'minute:2000-01-01T00:26', 'minute:2000-01-01T00:27', 'minute:2000-01-01T00:28', 'minute:2000-01-01T00:29', 'minute:2000-01-01T00:30', 'minute:2000-01-01T00:31', 'minute:2000-01-01T00:32', 'minute:2000-01-01T00:33', 'minute:2000-01-01T00:34', 'minute:2000-01-01T00:35', 'minute:2000-01-01T00:36', 'minute:2000-01-01T00:37', 'minute:2000-01-01T00:38', 'minute:2000-01-01T00:39', 'minute:2000-01-01T00:40', 'minute:2000-01-01T00:41', 'minute:2000-01-01T00:42', 'minute:2000-01-01T00:43', 'minute:2000-01-01T00:44', 'minute:2000-01-01T00:45', 'minute:2000-01-01T00:46', 'minute:2000-01-01T00:47', 'minute:2000-01-01T00:48', 'minute:2000-01-01T00:49'], cyclical=[], expanded=58, matched_docs=11

## Cost & timing

- LLM tokens: input=320639, output=223199
- Estimated cost: $0.5266
- Wall clock: 757.9s
