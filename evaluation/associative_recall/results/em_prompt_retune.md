# EventMemory Prompt Retune (speaker-baked embeddings)

## Setup

- n_questions = 30 (benchmark=locomo, first 30)
- EventMemory backend, `text-embedding-3-small`, `gpt-5-mini`
- Speaker-baked embedded format: `"{source}: {text}"` (from `event_memory.py::_format_text`)
- Caches: `cache/emretune_<variant>_cache.json` (dedicated)
- Architecture identical across variants (primer + 2 or 7 cues, merge by max_score per turn_id). Only the PROMPT varies.

## Speaker-baking format confirmation

`event_memory.py::_format_text` returns `f"{source}: {text}"` for MessageContext events. Per-conversation sources: conv-26 Caroline/Melanie, conv-30 Jon/Gina, conv-41 John/Maria.

## Recall comparison

| Variant | R@20 | R@50 | vs baseline R@20 | vs baseline R@50 | time (s) |
| --- | --- | --- | --- | --- | --- |
| `v2f_em_baseline` | 0.7250 | 0.8667 | +0.0000 | +0.0000 | 396.5 |
| `v2f_em_speakerformat` | 0.8167 | 0.8917 | +0.0917 | +0.0250 | 359.6 |
| `v2f_em_mixed_speakers` | 0.7556 | 0.8250 | +0.0306 | -0.0417 | 314.0 |
| `v2f_em_role_tag` | 0.7583 | 0.8667 | +0.0333 | +0.0000 | 302.2 |
| `type_enumerated_em_speakerformat` | 0.7139 | 0.8083 | -0.0111 | -0.0584 | 537.8 |

Prior `em_v2f` reference from `em_eval`: 0.7417 / 0.8833 at K=20 / K=50.

## W/T/L per variant vs v2f_em_baseline (re-run control)

| Variant | K=20 W/T/L | K=50 W/T/L |
| --- | --- | --- |
| `v2f_em_speakerformat` | 4/26/0 | 2/27/1 |
| `v2f_em_mixed_speakers` | 3/25/2 | 2/24/4 |
| `v2f_em_role_tag` | 2/28/0 | 1/28/1 |
| `type_enumerated_em_speakerformat` | 2/25/3 | 1/24/5 |

## Sample cues (3 questions)

Per question, shows cues from each variant side by side.

### Q0: `locomo_conv-26` -- 'When did Caroline go to the LGBTQ support group?'

- `v2f_em_baseline` (R@20=1.00, R@50=1.00):
  - `I went to a LGBTQ support group yesterday and it was so powerful.`
  - `The support group has made me feel accepted and given me courage to embrace myself.`
- `v2f_em_speakerformat` (R@20=1.00, R@50=1.00):
  - `Caroline: I went to an LGBTQ support group yesterday and it was so powerful.`
  - `Caroline: The support group I went to yesterday really made me feel accepted and gave me courage.`
- `v2f_em_mixed_speakers` (R@20=1.00, R@50=1.00):
  - `Caroline: I went to a LGBTQ support group yesterday and it was so powerful`
  - `Melanie: I'm so glad the LGBTQ support group was powerful for you and that it helped you feel accepted and gave you courage to embrace yourself`
- `v2f_em_role_tag` (R@20=1.00, R@50=1.00):
  - `[USER] I went to a LGBTQ support group yesterday and it was so powerful.`
  - `[USER] The support group has made me feel accepted and given me courage to embrace myself.`
- `type_enumerated_em_speakerformat` (R@20=1.00, R@50=1.00):
  - `Caroline: I showed up to the LGBTQ support group yesterday evening and it started right after I walked in.`
  - `Melanie: I really liked how they encouraged everyone to share; I prefer that open vibe to more structured meetings.`
  - `Caroline: There was a tense moment when someone interrupted and they and another attendee got into a disagreement about outing someone.`
  - `Melanie: Oh, and I forgot to mention earlier — I just got a message from the organizer confirming next week's session.`
  - `Caroline: We cleared the air afterwards and actually resolved the misunderstanding, so it's fine now.`
  - `Caroline: Wait, one more thing — I met a counselor there who might be able to help with the mentoring idea.`
  - `Melanie: She was sitting by the window in the community room, right next to the refreshments table.`

### Q15: `locomo_conv-26` -- 'What activities does Melanie partake in?'

- `v2f_em_baseline` (R@20=0.25, R@50=0.50):
  - `Melanie ran a charity race for mental health, goes running, reads, plays the violin, and signed up for a pottery class`
  - `Melanie bikes, goes camping and hiking with family, volunteers or participates in mental-health events, and mentions counseling or mental health services`
- `v2f_em_speakerformat` (R@20=0.50, R@50=0.75):
  - `Melanie: I ran a charity race for mental health last Saturday and it felt really rewarding, like taking care of our minds.`
  - `Melanie: I'm carving out me-time these days — running, reading, playing my violin, signed up for a pottery class, and we went camping and hiking with the kids (also been biking).`
- `v2f_em_mixed_speakers` (R@20=0.50, R@50=0.75):
  - `Caroline: reacting to Melanie's updates and responding to news about the charity race, pottery class, daily running/reading/violin practice, and recent camping, hiking and biking outings`
  - `Melanie: I ran a charity race for mental health; I carve out me-time each day — running, reading, and playing my violin; I just signed up for a pottery class; we went camping, hiking and biking with the kids`
- `v2f_em_role_tag` (R@20=0.25, R@50=0.50):
  - `[ASSISTANT] Melanie ran a charity race for mental health, signed up for a pottery class, and often goes running, reading, and playing violin to recharge`
  - `[USER] Melanie also likes biking, camping and hiking with the kids and has been involved in volunteering/support groups around mental health`
- `type_enumerated_em_speakerformat` (R@20=0.25, R@50=0.25):
  - `Melanie: Just arrived at the charity race start line — they were already warming up and the atmosphere is buzzing`
  - `Melanie: I prefer morning runs over evening gym sessions; they clear my head and set the tone for the day`
  - `Caroline: She said the rehearsal clashes with the charity race weekend and they're upset about the schedule conflict`
  - `Melanie: Oh btw, just got a confirmation — I signed up for that pottery class starting next week`
  - `Caroline: We sorted it out — she swapped shifts so they could still do the race and make the rehearsal`
  - `Melanie: Wait one more thing — she brought her violin on the camping trip and they ended up jamming around the fire`
  - `Melanie: I'm sitting at the picnic table by the trailhead, right next to the bike rack and the path up the hill`

### Q29: `locomo_conv-26` -- 'When did Melanie go to the pottery workshop?'

- `v2f_em_baseline` (R@20=1.00, R@50=1.00):
  - `Last Fri I finally took my kids to a pottery workshop.`
  - `Yeah, I made it in pottery class yesterday.`
- `v2f_em_speakerformat` (R@20=1.00, R@50=1.00):
  - `Melanie: Last Fri I finally took my kids to a pottery workshop.`
  - `Melanie: I finally took my kids to a pottery workshop last Friday; we all made our own pots.`
- `v2f_em_mixed_speakers` (R@20=1.00, R@50=1.00):
  - `Caroline: Can't wait to see your pottery project; asking when you went to the pottery workshop`
  - `Melanie: Last Fri I finally took my kids to a pottery workshop`
- `v2f_em_role_tag` (R@20=1.00, R@50=1.00):
  - `[ASSISTANT] Last Fri I finally took my kids to a pottery workshop.`
  - `[ASSISTANT] I made it in pottery class yesterday.`
- `type_enumerated_em_speakerformat` (R@20=1.00, R@50=1.00):
  - `Melanie: we arrived at the studio last Friday around noon and they were just setting up the wheels`
  - `Caroline: I really like that she finds pottery so calming — I prefer music, but they seem to love making things with clay`
  - `Melanie: there was a small mix-up with registration and they had to argue about who was on the list for a bit`
  - `Melanie: oh, forgot to mention I just signed us up for another session next Saturday after the workshop`
  - `Caroline: we cleared up the timing issue and now they're all set to go back next week`
  - `Melanie: btw one more thing — my youngest left their smock at the studio, so I'll swing by to grab it`
  - `Caroline: she sat at the far table by the window, hands buried in clay while they shaped little pots`

## Verdict

**Ship `v2f_em_speakerformat`** as new EM default. Lift vs current v2f: +0.0917 R@20, +0.0250 R@50.

## Outputs

- `results/em_prompt_retune.json`
- `results/em_prompt_retune.md`
- Source: `em_retuned_cue_gen.py`, `emretune_eval.py`
- Caches: `cache/emretune_<variant>_cache.json`
