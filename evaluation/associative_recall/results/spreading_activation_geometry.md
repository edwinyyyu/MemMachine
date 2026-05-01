# Spreading Activation — Phase A Geometry
Question: do gold turns cluster tightly with each other relative to the query? (Inter-gold > gold-to-query => gold is off-center from the query, so spreading may help.)

## LoCoMo-30 (n multi-gold) = 11

| Stat | Value |
|---|---|
| fraction gold off-center (gap>0) | 0.909 |
| mean gap (inter-gold − gold-to-query) | +0.1586 |
| median gap | +0.0806 |
| mean inter-gold cos | 0.4444 |
| mean gold-to-query cos | 0.2857 |
| mean random-to-query cos (baseline) | 0.2325 |
| mean query lift over random | +0.0532 |

### Gap histogram (inter-gold − gold-to-query)

| bin | count |
|---|---|
| [-0.30,-0.25) | 0 |
| [-0.25,-0.20) | 0 |
| [-0.20,-0.15) | 0 |
| [-0.15,-0.10) | 0 |
| [-0.10,-0.05) | 0 |
| [-0.05,-0.00) | 1 |
| [-0.00,+0.05) | 1 |
| [+0.05,+0.10) | 4 |
| [+0.10,+0.15) | 0 |
| [+0.15,+0.20) | 0 |
| [+0.20,+0.25) | 2 |
| [+0.25,+0.30) | 0 |

## LoCoMo full 182Q (n multi-gold) = 85

| Stat | Value |
|---|---|
| fraction gold off-center (gap>0) | 0.906 |
| mean gap (inter-gold − gold-to-query) | +0.1417 |
| median gap | +0.1269 |
| mean inter-gold cos | 0.4481 |
| mean gold-to-query cos | 0.3064 |
| mean random-to-query cos (baseline) | 0.2142 |
| mean query lift over random | +0.0923 |

### Gap histogram (inter-gold − gold-to-query)

| bin | count |
|---|---|
| [-0.30,-0.25) | 0 |
| [-0.25,-0.20) | 0 |
| [-0.20,-0.15) | 0 |
| [-0.15,-0.10) | 2 |
| [-0.10,-0.05) | 3 |
| [-0.05,-0.00) | 3 |
| [-0.00,+0.05) | 10 |
| [+0.05,+0.10) | 15 |
| [+0.10,+0.15) | 16 |
| [+0.15,+0.20) | 8 |
| [+0.20,+0.25) | 10 |
| [+0.25,+0.30) | 7 |

## Missed turns vs retrieved-gold neighbors (LoCoMo-30)

n_pairs (missed-turn × its question's retrieved-gold set) = 5

| Stat | Value |
|---|---|
| mean max cos(missed, retrieved-gold) | 0.4925 |
| median max cos | 0.4511 |
| mean mean cos | 0.4845 |
| frac missed with max cos >= 0.5 | 0.400 |
| frac missed with max cos >= 0.6 | 0.400 |

## Decision gate

Fraction off-center = 0.909; mean gap = +0.1586.

=> PROCEED to Phase B (clear positive signal).

