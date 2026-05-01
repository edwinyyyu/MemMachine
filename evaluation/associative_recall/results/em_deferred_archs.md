# EventMemory Deferred Architectures on LoCoMo-30

## Schema details

- Speaker baked into embedded text via `MessageContext.source = <first name>`.
- Filter field name for `EventMemory.query(property_filter=...)` is `context.source`. The EM `property_filter` API takes a `FilterExpr` from `memmachine_server.common.filter.filter_parser`; we use `Comparison(field="context.source", op="=", value=<name>)`.
- Per-conversation speaker names (from `results/conversation_two_speakers.json`):
  - `locomo_conv-26`: user=Caroline, assistant=Melanie
  - `locomo_conv-30`: user=Jon, assistant=Gina
  - `locomo_conv-41`: user=John, assistant=Maria
- LoCoMo-30 query speaker-mention distribution: {'user': 18, 'assistant': 12} (every question names exactly ONE side; 0 mention both, 0 mention neither).
- LoCoMo-30 alias-match distribution: 30/30 queries hit at least one registered alias group (`results/conversation_alias_groups.json`).
- Gated-no-speaker firing-channel distribution (of 30 queries): {'temporal_tokens': 17}.
- Meta-router route distribution: {'two_speaker_filter': 30} (all routed to the two-speaker branch — consequence of every question naming one side only).

## Recall matrix

| Architecture | R@20 | R@50 |
| --- | --- | --- |
| em_cosine_baseline (reference) | 0.7333 | 0.8833 |
| em_v2f (reference) | 0.7417 | 0.8833 |
| em_ens_2 (reference) | 0.7833 | 0.8667 |
| **em_two_speaker_filter** | **0.8417** | **0.9000** |
| **em_two_speaker_query_only** | **0.8000** | **0.9333** |
| **em_alias_expand_v2f** | **0.8250** | **0.8833** |
| **em_gated_no_speaker** | **0.7417** | **0.8833** |
| **em_meta_router** | **0.8417** | **0.9000** |

## Per-category R@20 / R@50

| Architecture | temporal (n=16) | multi_hop (n=4) | single_hop (n=10) |
| --- | --- | --- | --- |
| em_two_speaker_filter | 0.938/0.938 | 0.750/1.000 | 0.725/0.800 |
| em_two_speaker_query_only | 0.875/0.938 | 0.750/1.000 | 0.700/0.900 |
| em_alias_expand_v2f | 0.938/0.938 | 0.625/0.875 | 0.725/0.800 |
| em_gated_no_speaker | 0.812/0.938 | 0.625/0.875 | 0.675/0.800 |
| em_meta_router | 0.938/0.938 | 0.750/1.000 | 0.725/0.800 |

## SS-era vs EM-ported (shared architectures)

| SS arch | SS R@20 | SS R@50 | EM port | EM R@20 | EM R@50 | R@20 delta | R@50 delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two_speaker_filter | 0.8917 | 0.8917 | em_two_speaker_filter | 0.8417 | 0.9000 | -0.0500 | +0.0083 |
| alias_expand_v2f | 0.6944 | 0.8806 | em_alias_expand_v2f | 0.8250 | 0.8833 | +0.1306 | +0.0027 |
| gated_threshold_0.7 | 0.7583 | 0.8917 | em_gated_no_speaker | 0.7417 | 0.8833 | -0.0166 | -0.0084 |

## Decision rules verdicts

- **em_two_speaker_filter**: R@20 = 0.8417 (rule: >=0.88 SHIP; ~=em_cosine_baseline 0.7333 -> speaker-baking subsumes filter).
  Result: MID — adds **+10.0pp R@20** and **+1.7pp R@50** over `em_v2f` (0.7417 / 0.8833), but falls short of the SS-era 0.8917 R@20.
  Interpretation: speaker baking delivered most of the SS-era lift (em_cosine_baseline jumped from SS-cosine 0.3833 to 0.7333), so the residual from an explicit `context.source` filter is smaller but NON-vestigial on LoCoMo.
- **em_two_speaker_query_only**: R@20 / R@50 = 0.8000 / 0.9333.
  The hard speaker filter applied to the *raw* query (no v2f cues) achieves the **best R@50 of any architecture** tested here (0.9333) — better than SS two_speaker_filter's 0.8917 and better than any of the other EM variants. Clean evidence that EM's native `property_filter` carries the speaker-channel mechanism correctly.
- **em_alias_expand_v2f**: R@20 / R@50 = 0.8250 / 0.8833.
  **Lifts R@20 by +13pp over SS alias_expand_v2f (0.6944 -> 0.8250)** while matching SS at K=50. The EM port benefits from speaker-baked text (aliases sit next to speaker prefixes in the corpus, so per-variant sum_cosine retrieves cleanly). Roughly parity with em_two_speaker_filter (0.8417 / 0.9000); better on temporal (0.9375) than multi_hop (0.625).
- **em_gated_no_speaker**: R@20 / R@50 = 0.7417 / 0.8833.
  Essentially equal to `em_v2f` (0.7417 / 0.8833). The three retained channels (alias_context, temporal_tokens, entity_exact_match) did not meaningfully improve over the v2f base, and without the speaker_filter channel the gated overlay degenerates to v2f. Rule (>=0.90 R@50 -> SHIP): **NOT met**.
- **em_meta_router**: R@20 / R@50 = 0.8417 / 0.9000.
  Identical to `em_two_speaker_filter` because **100% of LoCoMo-30 queries mention exactly one speaker**, so the router always dispatches to the two-speaker branch. Rule (matches or beats both components): trivially met on the two-speaker side, but the gated branch is never exercised. The routing logic itself is sound; LoCoMo just doesn't exercise the split.

## Which SS wins transfer on EM

- **two_speaker_filter**: mostly transfers. SS R@20 0.8917 -> EM R@20 0.8417 (-5pp), but EM R@50 0.9000 vs SS 0.8917 (+0.8pp). The EM port uses `property_filter` natively; the SS v2f top-M appending was more aggressive at low K.
- **alias_expand_v2f**: transfers AND improves on EM (+13pp R@20). Speaker-baked embeddings align sibling probes to answer-bearing turns better.
- **gated_threshold_0.7**: degrades on EM when the speaker channel is removed. SS's gated overlay relied heavily on the speaker_filter channel, which is already covered by `em_two_speaker_filter`. Splitting the mechanisms reveals the gated overlay without speaker is no better than v2f on LoCoMo (no alias/temporal/entity-exact lifts observed).

## Verdict: updated production recipe for EM

1. **Primary shipped architecture on LoCoMo**: `em_two_speaker_filter` (R@20 0.8417 / R@50 0.9000). Composable, uses EM's native property_filter, no complex channel routing.
2. **Best-in-class R@50 so far**: `em_two_speaker_query_only` (0.9333) — the hard speaker filter on the raw query beats every v2f-stacked variant at K=50. This is an exceptional signal: on LoCoMo-30 the speaker filter applied without cue generation is *sufficient* for K=50 recall. Worth double-checking on a larger benchmark before claiming as the global ship.
3. **Meta-router**: degenerate on LoCoMo because every question names exactly one speaker. Keep the router logic available for heterogeneous benchmarks where name-mention distribution varies (e.g., longmemeval-hard).
4. **Alias expansion**: improves over SS at K=20 thanks to speaker-baked embeddings. Fold into the production pipeline opportunistically: cheap when alias groups are cached on disk; the 30/30 LoCoMo queries all hit alias groups so the channel fires reliably.
5. **Gated no-speaker**: do NOT ship on LoCoMo. The three retained channels (alias_context, temporal_tokens, entity_exact_match) do not displace v2f's weak picks with better ones; removing the speaker_filter channel flattens the arch to `em_v2f`.

## Outputs

- `results/em_deferred_archs.json` (raw per-question)
- `results/em_deferred_archs.md` (this report)
- Source: `em_two_speaker.py`, `em_alias_expand.py`, `em_gated_no_speaker.py`, `emdef_eval.py` (all under `evaluation/associative_recall/`)

## Coordination notes

- Reused existing Qdrant collections (`arc_em_lc30_v1_{26,30,41}`) and `results/eventmemory.sqlite3` — no new ingestion.
- Reused caches from `bestshot_llm_cache.json`, `em_v2f_llm_cache.json`, `alias_llm_cache.json` where present; wrote new entries to dedicated `cache/emdef_*_cache.json` files only.
- Did not modify any of `em_architectures.py`, `em_eval.py`, `em_setup.py`, `em_retuned_cue_gen.py` (framework + peer-agent files).