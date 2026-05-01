# Inverse Query Generation Study

**Motivation.** LoCoMo gold sits +0.14 cosine off-center from queries and only
36% kNN-adjacent. Inverse query generation starts from retrieved content and
works backward ("what question would this turn answer?") — the generated
probe is anchored in corpus vocabulary, not user phrasing. Orthogonal in
concept to v2f, which imagines chat content forward from the query.

## Fair-backfill recall (LoCoMo-30, synthetic-19)

| Arch               | Dataset        | base@20 | arch@20 |  Δ@20  | base@50 | arch@50 |  Δ@50  | llm/q |
|--------------------|----------------|--------:|--------:|-------:|--------:|--------:|-------:|------:|
| meta_v2f           | locomo_30q     |  0.383  |  0.756  | +0.372 |  0.508  |  0.858  | +0.350 |  1.0  |
| meta_v2f           | synthetic_19q  |  0.569  |  0.613  | +0.044 |  0.824  |  0.851  | +0.028 |  1.0  |
| inverse_query      | locomo_30q     |  0.383  |  0.233  | -0.150 |  0.508  |  0.367  | -0.142 |  1.0  |
| inverse_query      | synthetic_19q  |  0.569  |  0.513  | -0.057 |  0.824  |  0.812  | -0.011 |  1.0  |
| inverse_query_top3 | locomo_30q     |  0.383  |  0.328  | -0.056 |  0.508  |  0.511  | +0.003 |  1.0  |
| inverse_query_top3 | synthetic_19q  |  0.569  |  0.566  | -0.004 |  0.824  |  0.831  | +0.007 |  1.0  |
| inverse_query_v2f  | locomo_30q     |  0.383  |  0.400  | +0.017 |  0.508  |  0.714  | +0.206 |  2.0  |
| inverse_query_v2f  | synthetic_19q  |  0.569  |  0.547  | -0.022 |  0.824  |  0.823  | -0.001 |  2.0  |

## Orthogonality vs v2f (K=50)

Fraction of gold turns found by the variant that v2f did NOT find.

| Arch               | Dataset        | inv_gold | novel_vs_v2f | frac_novel |
|--------------------|----------------|---------:|-------------:|-----------:|
| inverse_query      | locomo_30q     |   14     |      1       |   0.071    |
| inverse_query      | synthetic_19q  |  112     |      6       |   0.054    |
| inverse_query_top3 | locomo_30q     |   18     |      0       |   0.000    |
| inverse_query_top3 | synthetic_19q  |  117     |      4       |   0.034    |
| inverse_query_v2f  | locomo_30q     |   29     |      1       |   0.035    |
| inverse_query_v2f  | synthetic_19q  |  113     |      6       |   0.053    |

Across all variants and datasets, less than 7.5% of gold found by the inverse
family is novel vs v2f. The hypothesized orthogonality does not materialize
— inverse probes end up pointing back at the *same* gold turns v2f already
covers (and more commonly fail to retrieve anything useful at all).

## Qualitative example

Only one LoCoMo question surfaced a novel gold-turn across all inverse
variants:

- **Q:** Would Caroline pursue writing as a career option?
  (locomo_multi_hop)
  - Source turn (hop0): turn_id=115
  - Generated inverse question: *"What did the assistant mention reading
    last year that reminded them to pursue their dreams?"*
  - Novel gold turn_id found: 116 (adjacent — an N+1 follow-up turn)

Two additional illustrative failures (gold NOT retrieved by any variant):

- *"When did Melanie sign up for a pottery class?"* — inverse queries focus
  on what Melanie says she's building, never on the sign-up event.
- *"When did Caroline go to the LGBTQ conference?"* — inverse queries on the
  support-group hop0 ask about parade attendance, advocacy events, and
  assistant-side supportive language, all of which drift off the target
  date.

## Top categories gaining / losing (LoCoMo-30, inverse_query, Δr@50)

Gaining:
- `locomo_single_hop` (n=10): Δ=+0.025 (W/T/L=2/6/2)
- *(no other categories gain)*

Losing:
- `locomo_temporal` (n=16): Δ=-0.250 (W/T/L=0/12/4) — the biggest hit;
  inverse questions about "when X happened" mine adjacent content rather
  than dates.
- `locomo_multi_hop` (n=4): Δ=-0.125 (W/T/L=1/2/1).

`inverse_query_v2f` rescues `locomo_single_hop` (+0.517 @50) and
`locomo_multi_hop` (+0.250 @50) but remains at Δ=0 on `locomo_temporal`.
Its gains come from the v2f half of the union, NOT from the inverse half.

## Verdict: **ABANDON**

- `inverse_query` alone loses massively to v2f on LoCoMo
  (0.367 vs 0.858 @K=50). It retrieves only 14/44 gold turns vs v2f's 36/44.
- `inverse_query_top3` roughly ties cosine baseline — it does no useful work.
- `inverse_query_v2f` (0.714 @50) loses to v2f (0.858 @50): the inverse
  half dilutes v2f's probe set, and its novel-gold contribution is ~3.5%.
- Orthogonality is empirically near-zero: inverse probes hit back at the
  same gold v2f already retrieves. Starting from retrieved content does not
  escape v2f's semantic neighborhood.
- The max-cosine merge is the core failure mode: inverse-question probes
  are specific follow-ups that score *very* high against narrow
  conversational responses, so those segments rank above the raw-query
  top-K — evicting actual gold from the K-truncation.

## Outputs

- Markdown report: `results/inverse_query_study.md`
- Raw aggregate JSON: `results/inverse_query_study.json`
- Per-arch per-dataset JSON: `results/invq_{arch}_{dataset}.json`
- Source: `inverse_query.py`, `inverse_query_eval.py`
- Dedicated caches: `cache/inv_query_embedding_cache.json`,
  `cache/inv_query_llm_cache.json`
