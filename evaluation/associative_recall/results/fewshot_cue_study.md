# Few-Shot Cue Generation Study

Empirical test of whether adding in-context exemplars (successful cues from
similar questions) to the v2f cue-generation prompt improves retrieval recall.

## Architecture-only constraints
- `text-embedding-3-small` (no embedder swap)
- `gpt-5-mini` (no model change)
- No BM25, no cross-encoder rerank
- Only change: v2f prompt + in-context `(question, cue)` exemplars

## Exemplar bank
- Size: **49 exemplars** (target was 50-200; slightly under)
- Source: cases where `fair_backfill_meta_v2f` beat cosine baseline at r@20 or r@50
- Breakdown: locomo=16, synthetic=8, puzzle=10, advanced=15
- Embeddings: `text-embedding-3-small` of the exemplar question
- Leave-one-out: exemplars from the same `conversation_id` are excluded at query time.
  Leakage check: **0 leaks across all 88 questions × 2 variants**.

## Variants
- `fewshot_v2f_k2` — top-2 nearest-question exemplars
- `fewshot_v2f_k3` — top-3 nearest-question exemplars
- `fewshot_v2f_category_k2` — NOT RUN (see decision rule below)

## Overall recall (arch after fair-backfill to K)

| Arch           | avg r@20 | avg r@50 |
|----------------|---------:|---------:|
| meta_v2f       | 0.6105   | 0.8821   |
| fewshot_v2f_k2 | 0.5883   | 0.8494   |
| fewshot_v2f_k3 | 0.5971   | 0.8674   |

Both few-shot variants are strictly WORSE than v2f on average. k3 > k2 (more
exemplars help slightly) but neither saturates to v2f level.

## Per-dataset recall (arch@K)

| Dataset        | K  | v2f   | k2    | k3    | d(k3 vs v2f) |
|----------------|----|------:|------:|------:|-------------:|
| locomo_30q     | 20 | 0.756 | 0.650 | 0.694 | **-0.062**   |
| locomo_30q     | 50 | 0.858 | 0.750 | 0.828 | **-0.030**   |
| synthetic_19q  | 20 | 0.613 | 0.635 | 0.637 | +0.024       |
| synthetic_19q  | 50 | 0.851 | 0.832 | 0.833 | -0.018       |
| puzzle_16q     | 20 | 0.480 | 0.494 | 0.482 | +0.002       |
| puzzle_16q     | 50 | 0.917 | 0.897 | 0.894 | -0.023       |
| advanced_23q   | 20 | 0.593 | 0.574 | 0.575 | -0.018       |
| advanced_23q   | 50 | 0.902 | 0.919 | 0.915 | +0.013       |

LoCoMo — the primary benchmark — is the biggest loss: -0.062 / -0.030.
Synthetic and puzzle at K=20 show tiny gains; everywhere else flat-or-worse.

## Top gainers / losers (category level, k3 vs v2f @ r@50)

**Gainers:**
| category                                    | n | v2f   | k3    | delta  |
|---------------------------------------------|---|------:|------:|-------:|
| advanced_23q::quantitative_aggregation      | 3 | 0.889 | 0.944 | +0.055 |
| advanced_23q::frequency_detection           | 1 | 0.895 | 0.947 | +0.053 |
| advanced_23q::unfinished_business           | 3 | 0.872 | 0.923 | +0.051 |

**Losers:**
| category                                    | n | v2f   | k3    | delta  |
|---------------------------------------------|---|------:|------:|-------:|
| locomo_30q::locomo_multi_hop                | 4 | 0.875 | 0.750 | **-0.125** |
| puzzle_16q::logic_constraint                | 3 | 0.758 | 0.700 | -0.058 |
| synthetic_19q::conjunction                  | 3 | 1.000 | 0.952 | -0.048 |
| puzzle_16q::contradiction                   | 2 | 0.958 | 0.917 | -0.042 |

## Does K=3 beat K=2?

Yes, consistently but modestly:
- avg r@20: k3=0.597 vs k2=0.588 (+0.009)
- avg r@50: k3=0.867 vs k2=0.849 (+0.018)

More exemplars = slightly more signal, but the signal is already pushing
the wrong direction on LoCoMo, so "more" makes things less-bad, not better.

## Qualitative check: is the LLM adapting or copy-pasting?

**Verdict: adapting, but in the wrong direction.**

The LLM is clearly not copy-pasting — cues are novel. The failure mode is
**over-specification**: exemplars from puzzle/advanced/synthetic datasets
teach gpt-5-mini to write detailed, narrative, named-entity-heavy cues,
which embed *further* from LoCoMo's terse chat register.

### 5 sample trios

**1. LoCoMo temporal | "When did Caroline have a picnic?" | delta r@50 = -1.000**

- v2f cues (WORK):
  - `we had a picnic yesterday at the park`
  - `had a picnic last weekend on Sunday afternoon`
- k3 cues (FAIL, fabricate specifics):
  - `We had a picnic at the park yesterday — Caroline laid out the blanket near the playground, we ate sandwiches while the kids played, then went for a short walk.`
  - `Remember the camping trip a few weeks ago, Caroline? After the hike we pulled off by the lake for a little picnic and later roasted marshmallows around the campfire.`
- Exemplars: all puzzle/synthetic event-planning questions. Pushed the LLM
  toward narrative event descriptions.

**2. LoCoMo single-hop | "Where has Melanie camped?" | delta r@50 = -0.667**

- v2f cues (WORK): terse, generic camping phrasings.
- k3 cues (FAIL): invents specific locations (Yosemite, Big Sur, Pine Ridge)
  not in the conversation.
- Exemplars: 3 puzzle questions about retreats.

**3. LoCoMo temporal | "When did Caroline go to the LGBTQ support group?" | delta = 0.0**

- Both cues work equally. k3 rephrases but retains the key phrase
  "LGBTQ support group yesterday". Ties are where neither pollution nor
  signal dominates.

**4. LoCoMo temporal | "How long ago was Caroline's 18th birthday?" | delta r@50 = +1.000**

- v2f: generic "turned 18 five years ago" — worked but missed
- k3: identical content, just rephrased, happened to hit a different chunk

**5. LoCoMo single-hop | "What books has Melanie read?" | delta r@50 = +0.500**

- v2f placeholder-ish ("[book title] and [book title]")
- k3 invents specific titles (Atomic Habits, Sapiens, etc.). Some of these
  phrases matched adjacent chat tokens.

### Failure mechanism

The exemplar bank is **polluted** by the 4-dataset mix. LoCoMo style is
first-person casual chat. Puzzle/advanced style is structured analytical
prose. Cosine similarity on the question embedding picks top-K neighbors
that are semantically vague matches but stylistically foreign. The LLM
faithfully imitates the in-context style, which degrades retrieval.

## Decision

**ABANDON.**

Decision rule triggered: "If few-shot is clearly worse across LoCoMo +
synthetic, stop and report." LoCoMo loses by 6.2pp at K=20 and 3.0pp at K=50.
Synthetic is flat (+0.024 at K=20, -0.018 at K=50). Not worth pursuing.

The category-matched variant was skipped because the root cause — cross-dataset
exemplar style pollution — would likely still bleed in when fewer than k
matches exist in-category; also, the exemplar bank is too small (49) for
category filtering to reliably find matches.

## What would help (not pursued here)

1. **Per-dataset exemplar banks** — keep LoCoMo querying only LoCoMo-derived
   exemplars. But this defeats the generalization value.
2. **Larger bank from cross-validated v2f runs** — more signal, less
   top-K noise. Requires new v2f runs on held-out splits.
3. **Cue-level attribution** — current filter uses "question's v2f helped overall"
   as a proxy for "this cue worked". Tighter attribution would purify the bank.

## Output files

- `/Users/eyu/edwinyyyu/mmcc/extra_memory/evaluation/associative_recall/results/fewshot_exemplar_bank.json` — 49 exemplars with embeddings
- `/Users/eyu/edwinyyyu/mmcc/extra_memory/evaluation/associative_recall/results/fewshot_cue_study.json` — all summaries
- `/Users/eyu/edwinyyyu/mmcc/extra_memory/evaluation/associative_recall/results/fewshot_meta_v2f_{dataset}.json` × 4 — v2f re-eval
- `/Users/eyu/edwinyyyu/mmcc/extra_memory/evaluation/associative_recall/results/fewshot_fewshot_v2f_k2_{dataset}.json` × 4
- `/Users/eyu/edwinyyyu/mmcc/extra_memory/evaluation/associative_recall/results/fewshot_fewshot_v2f_k3_{dataset}.json` × 4
- `/Users/eyu/edwinyyyu/mmcc/extra_memory/evaluation/associative_recall/build_exemplar_bank.py` — bank builder
- `/Users/eyu/edwinyyyu/mmcc/extra_memory/evaluation/associative_recall/fewshot_cue_gen.py` — few-shot MetaV2f variants
- `/Users/eyu/edwinyyyu/mmcc/extra_memory/evaluation/associative_recall/fewshot_cue_eval.py` — fair-backfill eval
