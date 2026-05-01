# Anchor-Turn Expansion Retrieval

Motivation: v2f imagines corpus content forward from the question, so cues land near the query in embedding space. Anchor expansion starts from ACTUAL retrieved turns and imagines their continuations — cues are anchored in real corpus vocabulary. Per-cue attribution showed winning cues sit ~0.575 cosine from gold and are entity-rich; anchoring in retrieved content should push cues closer to winners.

## Fair-backfill recall

| Arch | Dataset | base@20 | arch@20 | Δ@20 | base@50 | arch@50 | Δ@50 | llm/q |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| meta_v2f | locomo_30q | 0.383 | 0.756 | +0.372 | 0.508 | 0.858 | +0.350 | 1.0 |
| meta_v2f | synthetic_19q | 0.569 | 0.613 | +0.044 | 0.824 | 0.851 | +0.028 | 1.0 |
| anchor_exp_3anchors | locomo_30q | 0.383 | 0.569 | +0.186 | 0.508 | 0.714 | +0.206 | 3.0 |
| anchor_exp_3anchors | synthetic_19q | 0.569 | 0.573 | +0.004 | 0.824 | 0.834 | +0.011 | 3.0 |
| anchor_exp_5anchors | locomo_30q | 0.383 | 0.619 | +0.236 | 0.508 | 0.764 | +0.256 | 5.0 |
| anchor_exp_5anchors | synthetic_19q | 0.569 | 0.631 | +0.061 | 0.824 | 0.844 | +0.020 | 5.0 |
| anchor_exp_plus_v2f | locomo_30q | 0.383 | 0.653 | +0.269 | 0.508 | 0.814 | +0.306 | 4.0 |
| anchor_exp_plus_v2f | synthetic_19q | 0.569 | 0.635 | +0.065 | 0.824 | 0.842 | +0.018 | 4.0 |

## Orthogonality vs v2f (K=50)

Fraction of gold turns found by the variant that v2f did NOT find.

| Arch | Dataset | gold_found | novel_vs_v2f | frac_novel |
|---|---|---:|---:|---:|
| anchor_exp_3anchors | locomo_30q | 28 | 0 | 0.000 |
| anchor_exp_3anchors | synthetic_19q | 117 | 4 | 0.034 |
| anchor_exp_5anchors | locomo_30q | 30 | 1 | 0.033 |
| anchor_exp_5anchors | synthetic_19q | 119 | 6 | 0.050 |
| anchor_exp_plus_v2f | locomo_30q | 33 | 0 | 0.000 |
| anchor_exp_plus_v2f | synthetic_19q | 118 | 4 | 0.034 |

## Qualitative trios (anchor_exp_3anchors, LoCoMo, K=50)

Each row: anchor turn the LLM imagined around → imagined cue → gold turn that cue retrieved.

- **Q:** When did Caroline go to the LGBTQ support group?
  - Anchor (assistant, turn_id=184): _Wow, Caroline! They must have felt so appreciated. It's awesome to see the difference we can make in each other's lives. Any other exciting LGBTQ advocacy stuff coming up?_
  - Imagined cue: _She actually went last Tuesday — March 12th — and led a sharing circle; everyone kept thanking her for making the space so welcoming._
  - Gold turn retrieved: 2

- **Q:** What fields would Caroline be likely to pursue in her educaton?
  - Anchor (assistant, turn_id=9): _Wow, Caroline! What kinda jobs are you thinkin' of? Anything that stands out?_
  - Imagined cue: _She's thinking about environmental science, veterinary tech, or UX design — something that mixes fieldwork with creative outreach._
  - Gold turn retrieved: 8

- **Q:** What did Caroline research?
  - Anchor (assistant, turn_id=9): _Wow, Caroline! What kinda jobs are you thinkin' of? Anything that stands out?_
  - Imagined cue: _She’s thinking about policy analyst roles at local government, environmental consulting, or joining a clean-air non-profit — maybe even grad school if the right PhD program shows up._
  - Gold turn retrieved: 25

- **Q:** When did Melanie run a charity race?
  - Anchor (user, turn_id=19): _That charity race sounds great, Mel! Making a difference & raising awareness for mental health is super rewarding - I'm really proud of you for taking part!_
  - Imagined cue: _Mel: I actually ran the mental health charity 5K on June 12th — it felt amazing to be part of it!_
  - Gold turn retrieved: 18

- **Q:** When is Melanie planning on going camping?
  - Anchor (user, turn_id=203): _Wow, Mel, that's awesome! What's your best camping memory?_
  - Imagined cue: _<She's planning to go the weekend after next — Memorial Day weekend — for a three-night trip up at Lake Wren.>_
  - Gold turn retrieved: 24

## Top categories by Δr@50 (anchor_exp_3anchors, LoCoMo-30)

Gaining:
  - locomo_single_hop (n=10): Δ=+0.317 W/T/L=4/6/0
  - locomo_multi_hop (n=4): Δ=+0.250 W/T/L=2/2/0
Losing:

## Drift audit (anchor_exp_3anchors, LoCoMo)

Over 180 imagined cues, mean token overlap with anchor = 0.120, mean overlap with question = 0.085. Higher anchor overlap than question overlap → cues are following anchor content; reverse → LLM is drifting back to the question.

## Verdict

**ABANDON**: neither variant beats v2f on LoCoMo-30 @K=50 (v2f=0.858, anchor_3=0.714, anchor+v2f=0.814).
