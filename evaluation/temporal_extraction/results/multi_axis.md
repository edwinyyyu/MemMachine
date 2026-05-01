# Multi-Axis + Distributional Time Representation

Per-axis categorical distributions + cross-axis tags on top of interval brackets.

## Corpus

- Base: 39 docs, 55 queries.
- Axis (new): 15 docs, 20 queries.

## Per-variant metrics

| Variant | axis R@5 | axis R@10 | axis MRR | axis NDCG | base R@5 | base NDCG | all R@5 | all NDCG |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| INTERVAL-ONLY | 0.375 | 0.425 | 0.202 | 0.252 | 0.469 | 0.494 | 0.442 | 0.425 |
| TAGS-ONLY (hierarchical) | 0.367 | 0.467 | 0.359 | 0.375 | 0.335 | 0.420 | 0.344 | 0.407 |
| AXIS-DIST | 0.442 | 0.467 | 0.324 | 0.311 | 0.353 | 0.430 | 0.378 | 0.396 |
| MULTI-AXIS α=1.0 β=0.0 γ=0.0 | 0.375 | 0.425 | 0.202 | 0.252 | 0.469 | 0.494 | 0.442 | 0.425 |
| MULTI-AXIS α=0.0 β=1.0 γ=0.0 | 0.442 | 0.467 | 0.324 | 0.311 | 0.353 | 0.430 | 0.378 | 0.396 |
| MULTI-AXIS α=0.0 β=0.0 γ=1.0 | 0.267 | 0.383 | 0.204 | 0.227 | 0.273 | 0.315 | 0.271 | 0.290 |
| MULTI-AXIS α=0.5 β=0.35 γ=0.15 | 0.617 | 0.642 | 0.407 | 0.450 | 0.478 | 0.521 | 0.517 | 0.501 |
| MULTI-AXIS α=0.4 β=0.4 γ=0.2 | 0.617 | 0.642 | 0.445 | 0.468 | 0.411 | 0.515 | 0.470 | 0.502 |
| MULTI-AXIS α=0.3 β=0.5 γ=0.2 | 0.617 | 0.642 | 0.395 | 0.432 | 0.388 | 0.497 | 0.453 | 0.478 |
| HYBRID (MULTI-AXIS + semantic) | 0.633 | 0.658 | 0.675 | 0.619 | 0.360 | 0.429 | 0.438 | 0.484 |

## Best MULTI-AXIS blend

- MULTI-AXIS α=0.5 β=0.35 γ=0.15 (by axis-subset R@5)
- α=0.5, β=0.35, γ=0.15

## Per-axis ablation (skip one axis)

Removing each axis from the MULTI-AXIS best blend; axis-subset R@5:

- Baseline (all axes): R@5=0.617
- skip `weekday`: R@5=0.592 (Δ=-0.025)
- skip `part_of_day`: R@5=0.600 (Δ=-0.017)
- skip `year`: R@5=0.617 (Δ=-0.000)
- skip `month`: R@5=0.617 (Δ=-0.000)
- skip `day_of_month`: R@5=0.617 (Δ=-0.000)
- skip `quarter`: R@5=0.617 (Δ=-0.000)
- skip `decade`: R@5=0.617 (Δ=-0.000)
- skip `season`: R@5=0.617 (Δ=-0.000)
- skip `weekend`: R@5=0.617 (Δ=-0.000)
- skip `hour`: R@5=0.642 (Δ=+0.025)

## Extraction quality on axis queries (sample)

- **axis_q_thu** `What do I do on Thursdays?` ->
  - kind=recurrence, surface=`Thursdays`; rrule=FREQ=WEEKLY;BYDAY=TH
- **axis_q_mar** `What happens in March?` ->
- **axis_q_afternoon** `My afternoon activities?` ->
- **axis_q_weekend** `What weekend events do I have?` ->
  - kind=recurrence, surface=`weekend`; rrule=FREQ=WEEKLY;BYDAY=SA,SU
- **axis_q_q2** `Anything in Q2?` ->
- **axis_q_tue** `Tuesday specials?` ->
  - kind=recurrence, surface=`Tuesday`; rrule=FREQ=WEEKLY;BYDAY=TU
- **axis_q_thu_morning** `What do I do on Thursday mornings?` ->
  - kind=recurrence, surface=`Thursday mornings`; rrule=FREQ=WEEKLY;BYDAY=TH
- **axis_q_june_weekends** `June weekends?` ->

## Failure modes (axis queries)

- `axis_q_thu` ('What do I do on Thursdays?') Δ++0.50: gold=['axis_doc_thu_morning_standup', 'axis_doc_thu_run']; interval_top5=['doc_decade_1', 'doc_rec_simple_0', 'doc_multi_3', 'doc_rec_simple_2', 'axis_doc_thu_morning_standup']; multi_axis_top5=['doc_rec_simple_0', 'doc_multi_3', 'doc_decade_1', 'axis_doc_thu_run', 'axis_doc_thu_morning_standup']; hybrid_top5=['doc_rec_simple_0', 'axis_doc_thu_run', 'axis_doc_fri_off', 'doc_multi_3', 'axis_doc_tue_book']
- `axis_q_thu_morning` ('What do I do on Thursday mornings?') Δ++0.50: gold=['axis_doc_thu_morning_standup', 'axis_doc_thu_run']; interval_top5=['doc_decade_1', 'doc_rec_simple_0', 'doc_multi_3', 'axis_doc_thu_morning_standup', 'doc_rec_simple_2']; multi_axis_top5=['doc_rec_simple_0', 'doc_multi_3', 'doc_decade_1', 'axis_doc_thu_run', 'axis_doc_thu_morning_standup']; hybrid_top5=['axis_doc_thu_run', 'doc_rec_simple_0', 'doc_rec_simple_1', 'axis_doc_thu_morning_standup', 'doc_multi_3']
- `axis_q_morning` ('What are my morning activities?') Δ++0.33: gold=['axis_doc_sat_hike', 'axis_doc_thu_morning_standup', 'axis_doc_thu_run']; interval_top5=['doc_decade_1', 'doc_rec_simple_0', 'doc_multi_3', 'doc_rec_simple_2', 'doc_abs_recent_0']; multi_axis_top5=['doc_rec_simple_0', 'doc_multi_3', 'doc_decade_1', 'axis_doc_thu_morning_standup', 'doc_rec_simple_2']; hybrid_top5=['doc_rec_simple_1', 'axis_doc_thu_run', 'axis_doc_thu_morning_standup', 'doc_rec_simple_2', 'axis_doc_evening_reading']
- `axis_q_summer` ('Summer events?') Δ++0.50: gold=['axis_doc_jun_vac', 'axis_doc_summer_camp']; interval_top5=['doc_abs_recent_0', 'doc_abs_recent_1', 'doc_abs_recent_2', 'doc_abs_recent_3', 'doc_abs_recent_4']; multi_axis_top5=['doc_decade_1', 'doc_rel_distant_0', 'axis_doc_winter_sick', 'axis_doc_oct_harvest', 'axis_doc_summer_camp']; hybrid_top5=['axis_doc_summer_camp', 'axis_doc_oct_harvest', 'doc_abs_recent_4', 'doc_abs_recent_0', 'doc_abs_recent_3']
- `axis_q_winter` ('What happens in winter?') Δ++0.50: gold=['axis_doc_dec_holiday', 'axis_doc_winter_sick']; interval_top5=['doc_abs_recent_0', 'doc_abs_recent_1', 'doc_abs_recent_2', 'doc_abs_recent_3', 'doc_abs_recent_4']; multi_axis_top5=['doc_decade_1', 'doc_rel_distant_0', 'axis_doc_winter_sick', 'axis_doc_oct_harvest', 'axis_doc_summer_camp']; hybrid_top5=['axis_doc_winter_sick', 'axis_doc_summer_camp', 'axis_doc_oct_harvest', 'doc_abs_recent_1', 'doc_decade_1']
- `axis_q_autumn` ('Autumn activities?') Δ++1.00: gold=['axis_doc_oct_harvest']; interval_top5=['doc_abs_recent_0', 'doc_abs_recent_1', 'doc_abs_recent_2', 'doc_abs_recent_3', 'doc_abs_recent_4']; multi_axis_top5=['doc_decade_1', 'doc_rel_distant_0', 'axis_doc_winter_sick', 'axis_doc_oct_harvest', 'axis_doc_summer_camp']; hybrid_top5=['axis_doc_oct_harvest', 'axis_doc_summer_camp', 'doc_abs_recent_3', 'axis_doc_winter_sick', 'doc_abs_recent_1']
- `axis_q_q4` ('What do I do in Q4?') Δ++0.50: gold=['axis_doc_dec_holiday', 'axis_doc_oct_harvest']; interval_top5=['doc_abs_recent_0', 'doc_abs_recent_1', 'doc_abs_recent_2', 'doc_abs_recent_3', 'doc_abs_recent_4']; multi_axis_top5=['doc_decade_1', 'doc_rel_distant_0', 'axis_doc_winter_sick', 'axis_doc_oct_harvest', 'axis_doc_summer_camp']; hybrid_top5=['axis_doc_oct_harvest', 'doc_abs_recent_1', 'doc_rel_recent_4', 'doc_abs_recent_4', 'doc_abs_recent_0']
- `axis_q_october` ('Anything in October?') Δ++1.00: gold=['axis_doc_oct_harvest']; interval_top5=['doc_abs_recent_0', 'doc_abs_recent_1', 'doc_abs_recent_2', 'doc_abs_recent_3', 'doc_abs_recent_4']; multi_axis_top5=['doc_decade_1', 'doc_rel_distant_0', 'axis_doc_winter_sick', 'axis_doc_oct_harvest', 'axis_doc_summer_camp']; hybrid_top5=['axis_doc_oct_harvest', 'doc_abs_recent_3', 'doc_abs_recent_0', 'doc_abs_recent_4', 'doc_rel_recent_4']

## Cost

- New LLM tokens (axis corpus): input=43901, output=61889
- Estimated cost: $0.1348
