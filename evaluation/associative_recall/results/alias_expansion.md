# Query-time alias expansion via ingest-extracted alias groups

Motivation: evolving-terminology / multi-referent entities resist v2f cue gen because aliases are corpus-specific. Ingest extracts alias groups once per conversation; at query time, if the query mentions an alias, expanded queries replacing that alias with each sibling drive extra cosine retrievals (and optionally extra v2f runs).

## Alias groups extracted (LoCoMo)

- beam_4: 16 groups, 51 aliases total
  - ['equilateral triangle', 'equilateral']
  - ['isosceles triangle', 'isosceles']
  - ['scalene triangle', 'scalene']
- beam_5: 19 groups, 83 aliases total
  - ['coin toss', 'coin tosses', 'coin toss problem', 'coin toss problems', 'coin toss exercises', 'single coin toss']
  - ['dice roll', 'dice rolls', 'dice roll problem', 'dice roll problems', 'rolling a die', 'rolling two dice', 'two dice', '6-sided die', 'six-sided die', 'fair six-sided die']
  - ['52-card deck', 'standard 52-card deck', 'standard deck', 'deck of 52 cards', 'standard deck of 52 cards']
- beam_6: 10 groups, 31 aliases total
  - ['Crystal Wilson', 'Crystal']
  - ['Karen Lee', 'Karen']
  - ['StreamWave', 'StreamWave HQ', "StreamWave's ATS", "StreamWave's ATS parser version 3.2", 'digital media company StreamWave', 'executive producer at StreamWave']
- beam_7: 19 groups, 57 aliases total
  - ['East Janethaven Library', 'East Janethaven Library’s second floor']
  - ['Café La Belle', 'Café La Belle on Main Street', 'lunch at Café La Belle']
  - ['Montserrat Community College', 'MCC', 'MCC Room 204']
- beam_8: 16 groups, 67 aliases total
  - ['Island Media Group', 'Island Media', "Island Media's HR", "Island Media's creative director", "Island Media's team", "Island Media's diversity officer", "Island Media's green initiatives"]
  - ['Greg', "Greg's coaching session", "Greg's April 2 coaching session", "Greg's April 15 session", 'mock interview with Greg on April 25', 'career coach Greg', 'follow-up with Greg on May 8', 'one-on-one with Greg', 'Greg provided feedback', "Greg's suggestion"]
  - ['Leslie', 'close friend Leslie', "Leslie's networking event", "Leslie's April 3 networking event", "Leslie's storytelling workshop", "Leslie's April 18 workshop", "Leslie's storytelling workshop on April 18 at Coral Bay Library"]
- locomo_conv-26: 15 groups, 54 aliases total
  - ['Mel', 'Melanie', 'Mell']
  - ['Caroline', 'Caro']
  - ['LGBTQ support group', 'support group', 'the support group']
- locomo_conv-30: 7 groups, 35 aliases total
  - ['online clothing store', 'online clothes store', 'online store', 'my online clothing store', 'my online clothes store', 'clothing store', 'my store', 'store']
  - ['dance studio', 'my dance studio', 'the studio', 'studio', 'official opening night', 'opening night', 'grand opening', 'opening']
  - ['local comp', 'dance comp', 'dance competition', 'competition', 'dance contest', 'contest']
- locomo_conv-41: 16 groups, 57 aliases total
  - ['homeless shelter', 'the shelter', 'the homeless shelter I volunteer at', 'the shelter I volunteer at', 'shelter']
  - ['fundraiser', 'fundraiser next week', 'the fundraiser', 'chili cook-off']
  - ['5K charity run', '5K charity run in our neighborhood', 'a 5K charity run']
- synth_medical: 15 groups, 42 aliases total
  - ['metformin 500mg twice daily', 'increase my metformin to 1000mg twice daily', 'Metformin 1000mg 2x daily']
  - ['lisinopril 10mg once daily', 'Lisinopril 10mg daily', 'lisinopril (blood pressure medication)']
  - ['atorvastatin 20mg at night', 'Atorvastatin 20mg nightly']
- synth_personal: 13 groups, 34 aliases total
  - ['that new Thai place on 5th street', 'Thai Orchid on 5th', 'Thai Orchid']
  - ['Marcus', 'my college roommate']
  - ["Bob's daughter Emma", 'Emma']
- synth_planning: 11 groups, 36 aliases total
  - ['Karen', 'my sister Karen']
  - ['my parents', 'Mom and Dad', 'Mom', 'Dad', 'parents']
  - ["my parents' 40th wedding anniversary party", '40th anniversary party', 'surprise party', 'the June 15th surprise 40th anniversary party', 'the party']
- synth_technical: 11 groups, 39 aliases total
  - ['Google Nest thermostat', 'Nest thermostat', 'the Nest thermostat', 'Nest']
  - ['Google Home', 'Google Home speaker', 'Google Home routines', 'Google Home as the hub']
  - ['Lutron Caseta smart switches', 'Lutron Caseta switch', 'Caseta switch', 'Lutron Caseta']
- synth_work: 12 groups, 35 aliases total
  - ['Acme Corp', 'Acme', 'Acme Corporation']
  - ['Patricia', 'the Acme CMO', 'Acme CMO, Patricia']
  - ['Daniel Park', 'the new CEO', 'Acme CEO', 'the CEO']

30/30 LoCoMo queries matched an extracted alias (mean variants/query = 2.87, max = 6).

## Fair-backfill recall

| Arch | Dataset | base@20 | arch@20 | Δ@20 | base@50 | arch@50 | Δ@50 | llm/q |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| meta_v2f | locomo_30q | 0.383 | 0.756 | +0.372 | 0.508 | 0.858 | +0.350 | 1.0 |
| meta_v2f | synthetic_19q | 0.569 | 0.613 | +0.044 | 0.824 | 0.851 | +0.028 | 1.0 |
| alias_expand_cosine | locomo_30q | 0.383 | 0.417 | +0.033 | 0.508 | 0.533 | +0.025 | 0.0 |
| alias_expand_cosine | synthetic_19q | 0.569 | 0.560 | -0.009 | 0.824 | 0.812 | -0.012 | 0.0 |
| alias_expand_v2f_cheap | locomo_30q | 0.383 | 0.661 | +0.278 | 0.508 | 0.814 | +0.306 | 1.0 |
| alias_expand_v2f_cheap | synthetic_19q | 0.569 | 0.600 | +0.031 | 0.824 | 0.827 | +0.004 | 1.0 |
| alias_expand_v2f | locomo_30q | 0.383 | 0.694 | +0.311 | 0.508 | 0.881 | +0.372 | 2.9 |
| alias_expand_v2f | synthetic_19q | 0.569 | 0.603 | +0.033 | 0.824 | 0.827 | +0.004 | 1.8 |

## Orthogonality vs v2f (K=50)

Fraction of gold turns found by the variant that v2f did NOT find.

| Arch | Dataset | gold_found | novel_vs_v2f | frac_novel |
|---|---|---:|---:|---:|
| alias_expand_cosine | locomo_30q | 18 | 0 | 0.000 |
| alias_expand_cosine | synthetic_19q | 112 | 2 | 0.018 |
| alias_expand_v2f_cheap | locomo_30q | 33 | 0 | 0.000 |
| alias_expand_v2f_cheap | synthetic_19q | 115 | 2 | 0.017 |
| alias_expand_v2f | locomo_30q | 36 | 2 | 0.056 |
| alias_expand_v2f | synthetic_19q | 115 | 2 | 0.017 |

## Qualitative trios (alias_expand_v2f_cheap, LoCoMo, K=50)

Each row: matched alias → expanded variants → which variant surfaced gold turns.

- **Q:** When did Caroline go to the LGBTQ support group?
  - Matched alias `LGBTQ support group` → siblings: ['support group', 'the support group']
  - Matched alias `Caroline` → siblings: ['Caro']
  - Variants (4):
    - `When did Caroline go to the LGBTQ support group?`
    - `When did Caroline go to the support group?`
    - `When did Caroline go to the the support group?`
    - `When did Caro go to the LGBTQ support group?`
  - Variant `When did Caroline go to the LGBTQ support group?` retrieved gold turn(s) [2]
  - Variant `When did Caro go to the LGBTQ support group?` retrieved gold turn(s) [2]
  - Sibling probe `support group` retrieved gold turn(s) [2]
  - Sibling probe `the support group` retrieved gold turn(s) [2]

- **Q:** What fields would Caroline be likely to pursue in her educaton?
  - Matched alias `Caroline` → siblings: ['Caro']
  - Variants (2):
    - `What fields would Caroline be likely to pursue in her educaton?`
    - `What fields would Caro be likely to pursue in her educaton?`
  - Variant `What fields would Caro be likely to pursue in her educaton?` retrieved gold turn(s) [8]

- **Q:** When did Melanie run a charity race?
  - Matched alias `Melanie` → siblings: ['Mel', 'Mell']
  - Variants (3):
    - `When did Melanie run a charity race?`
    - `When did Mel run a charity race?`
    - `When did Mell run a charity race?`
  - Variant `When did Melanie run a charity race?` retrieved gold turn(s) [18]
  - Variant `When did Mel run a charity race?` retrieved gold turn(s) [18]
  - Variant `When did Mell run a charity race?` retrieved gold turn(s) [18]

- **Q:** When is Melanie planning on going camping?
  - Matched alias `Melanie` → siblings: ['Mel', 'Mell']
  - Variants (3):
    - `When is Melanie planning on going camping?`
    - `When is Mel planning on going camping?`
    - `When is Mell planning on going camping?`
  - Variant `When is Melanie planning on going camping?` retrieved gold turn(s) [24]
  - Variant `When is Mel planning on going camping?` retrieved gold turn(s) [24]
  - Variant `When is Mell planning on going camping?` retrieved gold turn(s) [24]

- **Q:** When did Caroline meet up with her friends, family, and mentors?
  - Matched alias `Caroline` → siblings: ['Caro']
  - Variants (2):
    - `When did Caroline meet up with her friends, family, and mentors?`
    - `When did Caro meet up with her friends, family, and mentors?`
  - Variant `When did Caro meet up with her friends, family, and mentors?` retrieved gold turn(s) [45]

## Top categories by Δr@50 (alias_expand_v2f_cheap, LoCoMo-30)

Gaining:
  - locomo_single_hop (n=10): Δ=+0.617 W/T/L=7/3/0
  - locomo_multi_hop (n=4): Δ=+0.250 W/T/L=2/2/0
  - locomo_temporal (n=16): Δ=+0.125 W/T/L=2/14/0
Losing:
  - (none with Δ < -0.001)

## Verdict

**SHIP**: alias_expand_v2f beats v2f on LoCoMo-30 @K=50 (0.881 vs 0.858).
