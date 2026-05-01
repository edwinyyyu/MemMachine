# Pronoun-Resolution Ingest Index -- Empirical Test

At ingest, an LLM rewrites each turn with pronouns/deictics (it, they, this, that, these, those, here/there, he/she if ambiguous) substituted by their specific referents, using the 2-3 preceding turns as context. Turns already self-contained are SKIP. RESOLVED texts are embedded in a separate vector index. At query time v2f retrieves as usual; a disjoint cosine search over the resolved index is STACKED-MERGED (v2f fills top-K first; resolved hits only fill any remaining slots).

This differs from prior LLM alt-key tests (-7pp r@20) which generated alt-keys broadly (49% of turns) and used max-score merge (competed with v2f). Here the filter is narrow (only turns with unresolved pronouns) and the merge is stacked (validated by critical_info_store).

## 1. Resolution rate

| dataset | turns | resolved | SKIP rate |
|---|---:|---:|---:|
| locomo_30q | 419 | 105 | 74.9% |
| synthetic_19q | 462 | 132 | 71.4% |

## 2. Recall (fair-backfill)

| dataset | K | cosine | v2f | stacked | Δ vs v2f | with_bonus | Δ vs v2f |
|---|---:|---:|---:|---:|---:|---:|---:|
| locomo_30q | 20 | 0.3833 | 0.7556 | 0.7556 | +0.0000 | 0.7556 | +0.0000 |
| locomo_30q | 50 | 0.5083 | 0.8583 | 0.8583 | +0.0000 | 0.8583 | +0.0000 |
| synthetic_19q | 20 | 0.5694 | 0.6130 | 0.6130 | +0.0000 | 0.6130 | +0.0000 |
| synthetic_19q | 50 | 0.8238 | 0.8513 | 0.8513 | +0.0000 | 0.8566 | +0.0053 |

## 3. Per-category (LoCoMo-30)

| category | n | v2f @20 | stacked @20 | bonus @20 | v2f @50 | stacked @50 | bonus @50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| locomo_multi_hop | 4 | 0.625 | 0.625 | 0.625 | 0.875 | 0.875 | 0.875 |
| locomo_single_hop | 10 | 0.617 | 0.617 | 0.617 | 0.825 | 0.825 | 0.825 |
| locomo_temporal | 16 | 0.875 | 0.875 | 0.875 | 0.875 | 0.875 | 0.875 |

## 4. Orthogonality (gold surfaced via resolved-only)

| dataset | variant | K | frac gold via resolved | frac questions with resolved-gold |
|---|---|---:|---:|---:|
| locomo_30q | pronoun_resolve_stacked | 20 | 0.0% | 0.0% |
| locomo_30q | pronoun_resolve_stacked | 50 | 0.0% | 0.0% |
| locomo_30q | pronoun_resolve_with_bonus | 20 | 0.0% | 0.0% |
| locomo_30q | pronoun_resolve_with_bonus | 50 | 0.0% | 0.0% |
| synthetic_19q | pronoun_resolve_stacked | 20 | 0.0% | 0.0% |
| synthetic_19q | pronoun_resolve_stacked | 50 | 0.0% | 0.0% |
| synthetic_19q | pronoun_resolve_with_bonus | 20 | 0.0% | 0.0% |
| synthetic_19q | pronoun_resolve_with_bonus | 50 | 0.7% | 5.3% |

## 5. Cost

- LLM calls: uncached=881 cached=0
- Input tokens: 313543
- Output tokens: 500790
- Est. cost (gpt-5-mini): $1.080

## 6. Sample resolutions (first 6 RESOLVED turns)

- **locomo_30q turn 1 (assistant)**
  - original:  Hey Caroline! Good to see you! I'm swamped with the kids & work. What's up with you? Anything new?
  - resolved:  Hey Caroline! Good to see you! I'm swamped with my kids & work. What's up with you? Anything new?
- **locomo_30q turn 4 (user)**
  - original:  The transgender stories were so inspiring! I was so happy and thankful for all the support.
  - resolved:  The transgender stories were so inspiring! I was so happy and thankful for all the support from the LGBTQ support group.
- **locomo_30q turn 7 (assistant)**
  - original:  That's really cool. You've got guts. What now?
  - resolved:  The support group making you feel accepted and giving you courage to embrace yourself is really cool. You've got guts. What now?
- **locomo_30q turn 10 (user)**
  - original:  I'm keen on counseling or working in mental health - I'd love to support those with similar issues.
  - resolved:  I'm keen on counseling or working in mental health - I'd love to support people with similar mental health issues to me.
- **locomo_30q turn 13 (assistant)**
  - original:  Yeah, I painted that lake sunrise last year! It's special to me.
  - resolved:  Yeah, I painted the lake sunrise painting last year! The painting is special to me.
- **locomo_30q turn 19 (user)**
  - original:  That charity race sounds great, Mel! Making a difference & raising awareness for mental health is super rewarding - I'm really proud of you for taking part!
  - resolved:  The charity race you ran last Saturday sounds great, Mel! Making a difference & raising awareness for mental health is super rewarding - I'm really proud of you for taking part!
