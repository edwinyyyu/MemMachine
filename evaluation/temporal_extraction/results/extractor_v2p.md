# v2' extractor — axis-aware single-pass (v2-prime)

Weights held fixed at α=0.5, β=0.35, γ=0.15 (previous best MULTI-AXIS blend).

## Axis-surface extraction rate (per category)

### Queries

| Category | v1 | v2 | v2' |
|---|---:|---:|---:|
| bare_month | 0/3 (0.00) | 2/3 (0.67) | 3/3 (1.00) |
| quarter | 0/2 (0.00) | 1/2 (0.50) | 2/2 (1.00) |
| season | 0/3 (0.00) | 1/3 (0.33) | 3/3 (1.00) |
| part_of_day | 3/5 (0.60) | 5/5 (1.00) | 5/5 (1.00) |
| weekend_weekday | 2/3 (0.67) | 2/3 (0.67) | 3/3 (1.00) |
| OVERALL | 5/16 (0.31) | 11/16 (0.69) | 16/16 (1.00) |

### Docs

| Category | v1 | v2 | v2' |
|---|---:|---:|---:|
| bare_month | 3/4 (0.75) | 4/4 (1.00) | 4/4 (1.00) |
| quarter | 1/1 (1.00) | 1/1 (1.00) | 1/1 (1.00) |
| season | 0/2 (0.00) | 0/2 (0.00) | 2/2 (1.00) |
| part_of_day | 5/5 (1.00) | 5/5 (1.00) | 5/5 (1.00) |
| weekend_weekday | - | - | - |
| OVERALL | 9/12 (0.75) | 10/12 (0.83) | 12/12 (1.00) |

## Retrieval metrics (axis subset, 20 queries)

| Variant | R@5 | R@10 | MRR | NDCG@10 |
|---|---:|---:|---:|---:|
| v1 + multi-axis | 0.617 | 0.642 | 0.407 | 0.450 |
| v2 + multi-axis | 0.683 | 0.750 | 0.557 | 0.578 |
| v2' + multi-axis | 0.958 | 1.000 | 0.804 | 0.840 |

### Interval-only (no axis scorer) for reference:

| Variant | R@5 | MRR | NDCG@10 |
|---|---:|---:|---:|
| v1 interval-only | 0.375 | 0.202 | 0.252 |
| v2 interval-only | 0.558 | 0.348 | 0.401 |
| v2' interval-only | 0.683 | 0.435 | 0.551 |

## Base regression check (10 sampled base queries, multi-axis blend)

| Variant | R@5 | R@10 | MRR | NDCG@10 |
|---|---:|---:|---:|---:|
| v1 base | 0.435 | 0.466 | 0.712 | 0.505 |
| v2 base | 0.388 | 0.466 | 0.654 | 0.475 |
| v2' base | 0.500 | 0.581 | 0.819 | 0.629 |

## Cost

- v2' new LLM usage: input=194376, output=58567
- v2' new LLM cost: $0.1657
- Wall time: 298.4 s

## Sample v2' axis-query extractions

- **axis_q_afternoon**: 'My afternoon activities?' -> rec['afternoon' rrule=FREQ=DAILY;BYHOUR=12,13,14,15,16,17]
- **axis_q_autumn**: 'Autumn activities?' -> rec['Autumn' rrule=FREQ=YEARLY;BYMONTH=9,10,11]
- **axis_q_evening**: 'What are my evening activities?' -> rec['evening' rrule=FREQ=DAILY;BYHOUR=18,19,20,21]
- **axis_q_fri**: 'What do I do on Fridays?' -> rec['Fridays' rrule=FREQ=WEEKLY;BYDAY=FR]
- **axis_q_june_weekends**: 'June weekends?' -> rec['June weekends?' rrule=FREQ=YEARLY;BYMONTH=6;BYDAY=SA,SU]
- **axis_q_mar**: 'What happens in March?' -> rec['March' rrule=FREQ=YEARLY;BYMONTH=3]
- **axis_q_morning**: 'What are my morning activities?' -> rec['morning' rrule=FREQ=DAILY;BYHOUR=6,7,8,9,10,11]
- **axis_q_october**: 'Anything in October?' -> rec['October' rrule=FREQ=YEARLY;BYMONTH=10]
- **axis_q_q2**: 'Anything in Q2?' -> rec['Q2' rrule=FREQ=YEARLY;BYMONTH=4,5,6]
- **axis_q_q4**: 'What do I do in Q4?' -> rec['Q4' rrule=FREQ=YEARLY;BYMONTH=10,11,12]
- **axis_q_saturday**: 'Saturday activities?' -> rec['Saturday' rrule=FREQ=WEEKLY;BYDAY=SA]
- **axis_q_summer**: 'Summer events?' -> rec['Summer' rrule=FREQ=YEARLY;BYMONTH=6,7,8]
- **axis_q_sunday**: 'Sunday plans?' -> rec['Sunday' rrule=FREQ=WEEKLY;BYDAY=SU]
- **axis_q_thu**: 'What do I do on Thursdays?' -> rec['Thursdays' rrule=FREQ=WEEKLY;BYDAY=TH]
- **axis_q_thu_morning**: 'What do I do on Thursday mornings?' -> rec['Thursday mornings' rrule=FREQ=WEEKLY;BYDAY=TH;BYHOUR=6,7,8,9,10,11]
- **axis_q_tue**: 'Tuesday specials?' -> rec['Tuesday' rrule=FREQ=WEEKLY;BYDAY=TU]
- **axis_q_wed**: 'What do I have on Wednesdays?' -> rec['Wednesdays' rrule=FREQ=WEEKLY;BYDAY=WE]
- **axis_q_weekday_morning**: 'Weekday morning events?' -> rec['Weekday morning' rrule=FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;BYHOUR=6,7,8,9,10,11]
- **axis_q_weekend**: 'What weekend events do I have?' -> rec['weekend' rrule=FREQ=WEEKLY;BYDAY=SA,SU]
- **axis_q_winter**: 'What happens in winter?' -> rec['winter' rrule=FREQ=YEARLY;BYMONTH=12,1,2]
