# Speaker-conditional cue generation

When a query mentions one of the two conversation participants, condition v2f's cue generation to produce cues AS IF that participant were speaking — first-person, casual chat register. Goal: cues embedded in the same register as gold (first-person) turns should cosine-match more tightly.

## Query coverage by side (speaker_cond_cue_only view)

| Dataset | n | user-only | assistant-only | both | none |
|---|---:|---:|---:|---:|---:|
| locomo_30q | 30 | 18 (60.0%) | 12 (40.0%) | 0 (0.0%) | 0 (0.0%) |

## Fair-backfill recall (LoCoMo-30)

| Arch | base@20 | arch@20 | Δ@20 | base@50 | arch@50 | Δ@50 | llm/q |
|---|---:|---:|---:|---:|---:|---:|---:|
| meta_v2f | 0.383 | 0.756 | +0.372 | 0.508 | 0.858 | +0.350 | 1.0 |
| two_speaker_filter | 0.383 | 0.892 | +0.508 | 0.508 | 0.892 | +0.383 | 1.0 |
| speaker_cond_cue_only | 0.383 | 0.689 | +0.306 | 0.508 | 0.789 | +0.281 | 1.0 |
| speaker_cond_plus_filter | 0.383 | 0.842 | +0.458 | 0.508 | 0.842 | +0.333 | 1.0 |
| v2f_mention_tag | 0.383 | 0.639 | +0.256 | 0.508 | 0.783 | +0.275 | 1.0 |

## Subset recall on fired queries

Each row restricts to queries where cue-conditioning fired (i.e. query mentions exactly one known speaker) and compares vs a reference arch.

| Arch | Subset key | n | ref@20 | arch@20 | Δ@20 | ref@50 | arch@50 | Δ@50 | W/T/L@50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| speaker_cond_cue_only | locomo_30q__conditioning_fired_vs_v2f | 30 | 0.756 | 0.689 | -0.067 | 0.858 | 0.789 | -0.069 | 2/22/6 |
| speaker_cond_cue_only | locomo_30q__conditioning_fired_vs_tsf | 30 | 0.892 | 0.689 | -0.203 | 0.892 | 0.789 | -0.103 | 1/23/6 |
| v2f_mention_tag | locomo_30q__conditioning_fired_vs_v2f | 30 | 0.756 | 0.639 | -0.117 | 0.858 | 0.783 | -0.075 | 2/22/6 |
| v2f_mention_tag | locomo_30q__conditioning_fired_vs_tsf | 30 | 0.892 | 0.639 | -0.253 | 0.892 | 0.783 | -0.108 | 1/23/6 |
| speaker_cond_plus_filter | locomo_30q__conditioning_fired_vs_v2f | 30 | 0.756 | 0.842 | +0.086 | 0.858 | 0.842 | -0.017 | 2/25/3 |
| speaker_cond_plus_filter | locomo_30q__conditioning_fired_vs_tsf | 30 | 0.892 | 0.842 | -0.050 | 0.892 | 0.842 | -0.050 | 1/26/3 |

## Per-category (speaker_cond_plus_filter)

### locomo_30q

| category | n | Δ@20 | Δ@50 | W/T/L@50 |
|---|---:|---:|---:|---:|
| locomo_multi_hop | 4 | +0.250 | +0.250 | 2/2/0 |
| locomo_single_hop | 10 | +0.675 | +0.600 | 7/3/0 |
| locomo_temporal | 16 | +0.375 | +0.188 | 3/13/0 |

## Per-category (speaker_cond_cue_only)

### locomo_30q

| category | n | Δ@20 | Δ@50 | W/T/L@50 |
|---|---:|---:|---:|---:|
| locomo_multi_hop | 4 | +0.125 | +0.250 | 2/2/0 |
| locomo_single_hop | 10 | +0.467 | +0.542 | 7/2/1 |
| locomo_temporal | 16 | +0.250 | +0.125 | 2/14/0 |

## Sample cues (v2f vs conditioned)

### speaker_cond_cue_only

**Q** (conv=locomo_conv-26, conditioned_on=Caroline): When did Caroline go to the LGBTQ support group?

- v2f cues:
  - I went to a LGBTQ support group yesterday and it was so powerful.
  - I went to a support group yesterday
- speaker_cond_cue_only cues:
  - I went to an LGBTQ support group yesterday and it was so powerful — I felt really seen and supported.
  - Yesterday I attended an LGBTQ support group; it moved me a lot and reminded me how important community is.
- r@20: v2f=1.0, speaker_cond_cue_only=1.0; r@50: v2f=1.0, speaker_cond_cue_only=1.0

**Q** (conv=locomo_conv-26, conditioned_on=Melanie): When did Melanie paint a sunrise?

- v2f cues:
  - Is this your own painting?
  - Yeah, I painted that lake sunrise last year! It's special to me.
- speaker_cond_cue_only cues:
  - I painted that lake sunrise last year; it's special to me.
  - I actually did it on a little weekend getaway last year — the sunrise colors stuck with me and I had to paint them.
- r@20: v2f=0.0, speaker_cond_cue_only=0.0; r@50: v2f=0.0, speaker_cond_cue_only=0.0

**Q** (conv=locomo_conv-26, conditioned_on=Caroline): What fields would Caroline be likely to pursue in her educaton?

- v2f cues:
  - I'm studying counseling and thinking about a master's in social work, counseling psychology, or becoming a therapist
  - I'm taking pottery classes and considering art therapy, studio art programs, or teaching community art classes
- speaker_cond_cue_only cues:
  - I'm leaning toward counseling or psychology — anything with therapy, mental health, or social work where I can actually help people one-on-one.
  - I'm also into creative stuff like pottery, so art therapy or occupational therapy that blends arts with mental-health work sounds perfect.
- r@20: v2f=1.0, speaker_cond_cue_only=0.5; r@50: v2f=1.0, speaker_cond_cue_only=1.0

### v2f_mention_tag

**Q** (conv=locomo_conv-26, conditioned_on=Caroline): When did Caroline go to the LGBTQ support group?

- v2f cues:
  - I went to a LGBTQ support group yesterday and it was so powerful.
  - I went to a support group yesterday
- v2f_mention_tag cues:
  - I went to a LGBTQ support group yesterday and it was so powerful.
  - I went to a LGBTQ support group last night and it was so powerful.
- r@20: v2f=1.0, v2f_mention_tag=1.0; r@50: v2f=1.0, v2f_mention_tag=1.0

**Q** (conv=locomo_conv-26, conditioned_on=Melanie): When did Melanie paint a sunrise?

- v2f cues:
  - Is this your own painting?
  - Yeah, I painted that lake sunrise last year! It's special to me.
- v2f_mention_tag cues:
  - Yeah, I painted that lake sunrise last year! It's special to me.
  - I painted the lake sunrise last year.
- r@20: v2f=0.0, v2f_mention_tag=0.0; r@50: v2f=0.0, v2f_mention_tag=0.0

**Q** (conv=locomo_conv-26, conditioned_on=Caroline): What fields would Caroline be likely to pursue in her educaton?

- v2f cues:
  - I'm studying counseling and thinking about a master's in social work, counseling psychology, or becoming a therapist
  - I'm taking pottery classes and considering art therapy, studio art programs, or teaching community art classes
- v2f_mention_tag cues:
  - I'm planning to study counseling, psychology, social work, or another mental-health field—possibly a master's and licensure to become a therapist/counselor.
  - I'm thinking about art, ceramics, or art therapy—taking pottery classes and maybe pursuing a studio art degree or certificate to combine creativity with helping others.
- r@20: v2f=1.0, v2f_mention_tag=1.0; r@50: v2f=1.0, v2f_mention_tag=1.0

### speaker_cond_plus_filter

**Q** (conv=locomo_conv-26, conditioned_on=Caroline): When did Caroline go to the LGBTQ support group?

- v2f cues:
  - I went to a LGBTQ support group yesterday and it was so powerful.
  - I went to a support group yesterday
- speaker_cond_plus_filter cues:
  - I went to an LGBTQ support group yesterday and it was so powerful — I felt really seen and supported.
  - Yesterday I attended an LGBTQ support group; it moved me a lot and reminded me how important community is.
- r@20: v2f=1.0, speaker_cond_plus_filter=1.0; r@50: v2f=1.0, speaker_cond_plus_filter=1.0

**Q** (conv=locomo_conv-26, conditioned_on=Melanie): When did Melanie paint a sunrise?

- v2f cues:
  - Is this your own painting?
  - Yeah, I painted that lake sunrise last year! It's special to me.
- speaker_cond_plus_filter cues:
  - I painted that lake sunrise last year; it's special to me.
  - I actually did it on a little weekend getaway last year — the sunrise colors stuck with me and I had to paint them.
- r@20: v2f=0.0, speaker_cond_plus_filter=0.0; r@50: v2f=0.0, speaker_cond_plus_filter=0.0

**Q** (conv=locomo_conv-26, conditioned_on=Caroline): What fields would Caroline be likely to pursue in her educaton?

- v2f cues:
  - I'm studying counseling and thinking about a master's in social work, counseling psychology, or becoming a therapist
  - I'm taking pottery classes and considering art therapy, studio art programs, or teaching community art classes
- speaker_cond_plus_filter cues:
  - I'm leaning toward counseling or psychology — anything with therapy, mental health, or social work where I can actually help people one-on-one.
  - I'm also into creative stuff like pottery, so art therapy or occupational therapy that blends arts with mental-health work sounds perfect.
- r@20: v2f=1.0, speaker_cond_plus_filter=1.0; r@50: v2f=1.0, speaker_cond_plus_filter=1.0

## Verdict

- speaker_cond_cue_only vs v2f: Δ@20=-0.067, Δ@50=-0.069
- speaker_cond_plus_filter vs v2f: Δ@20=+0.086, Δ@50=-0.017
- v2f_mention_tag vs v2f: Δ@20=-0.117, Δ@50=-0.075
- speaker_cond_plus_filter vs two_speaker_filter: Δ@20=-0.050, Δ@50=-0.050

**ABANDON conditioning+filter** — regresses vs two_speaker_filter (Δ@20=-0.050, Δ@50=-0.050). Conditioning hurts once filter already narrows candidates.
