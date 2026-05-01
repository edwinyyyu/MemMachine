# Spreading Activation — Phase B Eval (LoCoMo-30)

Params: α=0.5, kNN k=10, K0∈[10, 20].

## Summary (recall averaged over n=30)

| variant | r@20 | r@50 |
|---|---|---|
| cosine | 0.3833 | 0.5083 |
| v2f | 0.7556 | 0.8583 |
| spread_plain_K0=10 | 0.2167 | 0.3611 |
| spread_plain_K0=20 | 0.1500 | 0.3944 |
| spread_v2f | 0.5806 | 0.8250 |

## Per-category

| category | n | cosine r@20 | cosine r@50 | v2f r@20 | v2f r@50 | spread_plain_K0=10 r@20 | spread_plain_K0=10 r@50 | spread_plain_K0=20 r@20 | spread_plain_K0=20 r@50 | spread_v2f r@20 | spread_v2f r@50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| locomo_multi_hop | 4 | 0.500 | 0.500 | 0.625 | 0.875 | 0.125 | 0.250 | 0.125 | 0.500 | 0.750 | 0.875 |
| locomo_single_hop | 10 | 0.050 | 0.125 | 0.617 | 0.825 | 0.000 | 0.183 | 0.000 | 0.083 | 0.342 | 0.725 |
| locomo_temporal | 16 | 0.562 | 0.750 | 0.875 | 0.875 | 0.375 | 0.500 | 0.250 | 0.562 | 0.688 | 0.875 |

## Verdict

- spread_plain vs cosine: r@20 -0.1666, r@50 -0.1472
- spread_v2f vs v2f:     r@20 -0.1750, r@50 -0.0333

**Abandon: spreading activation does not help here.**

## Interpretation

Phase A said YES (gold clusters off-center: 91% of multi-gold questions,
mean gap +0.159), but Phase B said NO. Why?

Cosine r@20 is only 0.383 on LoCoMo-30 — i.e. only ~38% of the K0=10 seeds
are actually gold. When seeds are wrong, spreading amplifies wrongness:
a non-gold seed's kNN neighbors are semantically similar but also not gold,
and they now get nonzero activation and out-rank cosine's rank-11..50 turns
that WOULD have contained gold.

The geometry argument assumed seeds are gold. With unreliable seeds the
kNN graph leaks activation into the same wrong cluster the seeds live in.
Spreading only wins when the seed set already contains gold.

This matches the per-category result: on locomo_single_hop where cosine r@20
is 0.05, spread_plain r@20 = 0.0. With no gold seeds, spread has nothing
to propagate from.

`spread_v2f` underperforms v2f even though v2f seeds are much better. The
reason is ranking: v2f delivers gold turns at useful ranks inside the 2×cue
retrieval; re-ranking by spread activation shuffles them around and
deprioritizes gold turns that v2f ranked high but that don't have strong
kNN connectivity to other v2f seeds.

Takeaway: kNN-based spreading in embedding space is the wrong lever here.
The `inter_gold >> gold_to_query` geometry is real but does not imply the
kNN graph at each GOLD turn points at OTHER gold turns. It only says gold
turns share vocabulary with each other. That's a property of the gold set,
not of the graph-local neighborhood around a cosine-seed retrieval.

## Post-hoc diagnostic

For each gold turn in LoCoMo-30 multi-gold questions, we checked whether
another gold turn of the same question appears in its top-10 kNN
(same-conversation) neighborhood:

- 9 / 25 gold turns (36%) have ≥1 other gold turn in their top-10 kNN.

So even with inter_gold > gold_to_query in mean, fewer than half of gold
turns are each other's near neighbors. Spreading activation in embedding
space can only reach what's in the kNN of a seed; it cannot cross this
gap. This is the mechanism behind the negative Phase B result.
