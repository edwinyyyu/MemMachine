# Pair-Level Q-A Embedding Study

Motivation: queries often ask questions; turns in the corpus are user asks and assistant replies separately. Embedding (user_turn, assistant_reply) as a single unit captures the Q-A structure. Pair-hits promote both constituent turns via an attenuated score.

Zero-LLM architecturally — the pair variants alone add no LLM calls beyond the main cosine retrieval. `pair_plus_v2f` layers v2f cues on top as the intended ship test.

## Pair index stats

| Dataset | num_pairs | use_role_prefix |
|---|---:|---|
| locomo_30q | 1325 | True (default) |
| synthetic_19q | 231 | True (default) |

## Fair-backfill recall

| Arch | Dataset | base@20 | arch@20 | d@20 | base@50 | arch@50 | d@50 | llm/q |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| meta_v2f | locomo_30q | 0.383 | 0.756 | +0.372 | 0.508 | 0.858 | +0.350 | 1.0 |
| meta_v2f | synthetic_19q | 0.569 | 0.613 | +0.044 | 0.824 | 0.851 | +0.028 | 1.0 |
| pair_roleprefix | locomo_30q | 0.383 | 0.383 | +0.000 | 0.508 | 0.539 | +0.031 | 0.0 |
| pair_roleprefix | synthetic_19q | 0.569 | 0.559 | -0.011 | 0.824 | 0.823 | -0.001 | 0.0 |
| pair_noprefix | locomo_30q | 0.383 | 0.383 | +0.000 | 0.508 | 0.539 | +0.031 | 0.0 |
| pair_noprefix | synthetic_19q | 0.569 | 0.548 | -0.021 | 0.824 | 0.811 | -0.012 | 0.0 |
| pair_plus_v2f | locomo_30q | 0.383 | 0.631 | +0.247 | 0.508 | 0.731 | +0.222 | 1.0 |
| pair_plus_v2f | synthetic_19q | 0.569 | 0.625 | +0.055 | 0.824 | 0.833 | +0.009 | 1.0 |

## Orthogonality vs meta_v2f (K=50)

% of arch's found gold turns NOT retrieved by meta_v2f at K=50. Higher = pair index covers different gold than v2f does.

| Dataset | Arch | pooled_arch_found | pooled_novel | pooled_frac_novel | mean_frac_novel/q |
|---|---|---:|---:|---:|---:|
| locomo_30q | pair_roleprefix | 20 | 0 | 0.000 | 0.000 |
| locomo_30q | pair_noprefix | 20 | 0 | 0.000 | 0.000 |
| locomo_30q | pair_plus_v2f | 29 | 0 | 0.000 | 0.000 |
| synthetic_19q | pair_roleprefix | 115 | 4 | 0.035 | 0.036 |
| synthetic_19q | pair_noprefix | 113 | 2 | 0.018 | 0.018 |
| synthetic_19q | pair_plus_v2f | 116 | 3 | 0.026 | 0.028 |

## Top categories by d_r@50 (on LoCoMo-30)


### pair_roleprefix

Gaining:
  - locomo_single_hop (n=10): delta=+0.092 W/T/L=2/7/1
Losing:

### pair_noprefix

Gaining:
  - locomo_single_hop (n=10): delta=+0.092 W/T/L=2/7/1
Losing:

### pair_plus_v2f

Gaining:
  - locomo_single_hop (n=10): delta=+0.367 W/T/L=4/6/0
  - locomo_multi_hop (n=4): delta=+0.250 W/T/L=2/2/0
Losing:

## Sample pair texts


### locomo_30q/pair_roleprefix

1. conv=beam_4 turns=(0/user, 1/assistant)
   `USER: I'm trying to figure out what day it will be in a week from March 3, 2024, can you help me with that, like what's the date gonna be
ASSISTANT: Certainly! If today is March 3, 2024, then one week from today would be March 10, 2024. Each week consists of seven days, so adding seven days to March`

2. conv=beam_4 turns=(2/user, 3/assistant)
   `USER: I'm Christina, a colour technologist, and I'm trying to understand triangle geometry for design precision, but I'm having trouble figuring out how to calculate the area of a triangle when I know the lengths of all three sides - can you help me with that?
ASSISTANT: Of course, Christina! To cal`

3. conv=beam_4 turns=(4/user, 5/assistant)
   `USER: That's really helpful! Could you show me how to apply it to a triangle with sides 8 cm, 10 cm, and 12 cm?
ASSISTANT: Sure, let's apply Heron's formula to a triangle with sides measuring 8 cm, 10 cm, and 12 cm.

First, we calculate the semi-perimeter (s):
\[
s = \frac{8 + 10 + 12}{2} = \frac{30`

### synthetic_19q/pair_roleprefix

1. conv=synth_personal turns=(0/user, 1/assistant)
   `USER: Hey, good morning! Can you remind me what I have going on today?
ASSISTANT: Good morning! Let me check. You have a dentist appointment at 10am and then lunch with Marcus at 12:30.`

2. conv=synth_personal turns=(2/user, 3/assistant)
   `USER: Right, Marcus. He's my college roommate - haven't seen him in years. We're going to that new Thai place on 5th street.
ASSISTANT: Sounds fun! Thai Orchid on 5th? I've heard great things about it.`

3. conv=synth_personal turns=(4/user, 5/assistant)
   `USER: Yeah that's the one. Oh by the way, remind me to pick up Bob's birthday present. His birthday is next Thursday.
ASSISTANT: Got it! Bob's birthday is next Thursday. Any gift ideas in mind?`

## Verdict

`pair_plus_v2f` LoCoMo@50=0.731 vs meta_v2f=0.858 — no gain over v2f.
Zero-LLM `pair_roleprefix` LoCoMo@50=0.539 beats cosine baseline=0.508.
Zero-LLM `pair_noprefix` LoCoMo@50=0.539 beats cosine baseline=0.508.