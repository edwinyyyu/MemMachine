# Query-time alias expansion via dynamic alias tracker

Anaphora-resolution at INGEST fails (the first alias introduction is itself a cue). Instead, inject alias-sibling context at QUERY time into a SINGLE v2f call so the LLM picks whichever form matches the imagined conversation register. Tracker records first/last-seen turn indices for drift-aware variants.

## Alias tracker stats (LoCoMo-30 conversations)

- 1 conversations, 15 alias groups total, 54 aliases total
- 15/15 groups have turn-index coverage (first/last seen turn populated)
  - locomo_conv-26: 15 groups, 54 aliases, turns 0-418

## Fair-backfill recall (LoCoMo-30)

| Arch | base@20 | arch@20 | Δ@20 | base@50 | arch@50 | Δ@50 | llm/q |
|---|---:|---:|---:|---:|---:|---:|---:|
| meta_v2f | 0.383 | 0.756 | +0.372 | 0.508 | 0.858 | +0.350 | 1.0 |
| alias_trk_context | 0.383 | 0.572 | +0.189 | 0.508 | 0.756 | +0.247 | 1.0 |
| alias_trk_drift | 0.383 | 0.606 | +0.222 | 0.508 | 0.800 | +0.292 | 1.0 |
| alias_expand_v2f (ref, 3x cost) | 0.383 | 0.694 | +0.311 | 0.508 | 0.881 | +0.372 | 2.9 |

## Cost comparison (avg LLM calls/query)

**locomo_30q**
  - meta_v2f: 1.00
  - alias_trk_context: 1.00
  - alias_trk_drift: 1.00
  - alias_expand_v2f (reference): 2.90

## Sample alias injections


**Sample 1** (locomo_conv-26)
- **Q:** When did Caroline go to the LGBTQ support group?
- **Note:** Note: in the conversation, 'LGBTQ support group' is also known as: 'support group', 'the support group'. 'Caroline' is also known as: 'Caro'. Your cues may use any of these forms, whichever fits the imagined conversation register.
- Match `LGBTQ support group` (canonical: `support group`); siblings used: ['support group', 'the support group']; first/last seen at turns 2/6
- Match `Caroline` (canonical: `Caroline`); siblings used: ['Caro']; first/last seen at turns 1/416
- Generated cues:
  - `I went to a LGBTQ support group yesterday and it was so powerful.`
  - `Caroline said she went to the support group yesterday`
- Gold turns found @50: [2]

**Sample 2** (locomo_conv-26)
- **Q:** What fields would Caroline be likely to pursue in her educaton?
- **Note:** Note: in the conversation, 'Caroline' is also known as: 'Caro'. Your cues may use any of these forms, whichever fits the imagined conversation register.
- Match `Caroline` (canonical: `Caroline`); siblings used: ['Caro']; first/last seen at turns 1/416
- Generated cues:
  - `Caroline is thinking about counseling or mental health work, mentioning psychology, social work, counseling degrees (MSW, LPC, clinical counseling) and training to help others`
  - `Caroline (Caro) talks about pottery and the arts as a possible path, mentioning ceramics classes, studio practice, teaching pottery, or creative/art therapy programs`
- Gold turns found @50: [8, 10]

## Verdict

**ABANDON**: no tracker variant beats vanilla v2f (v2f=0.858). v2f-per-variant mechanism (the expensive version) was the actual value, not just seeing the aliases.
Details: alias_trk_context=0.756@1.0llm, alias_trk_drift=0.800@1.0llm; alias_expand_v2f ref=0.881.
