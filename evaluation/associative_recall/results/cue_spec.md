# Model-agnostic cue-generation spec — results

Mechanism: structural constraints + verify-repair loop. Same prompt template for any competent model.

## Spec constraints (per cue)

1. Length 8-35 words.
2. Entity overlap: >= 1 non-stopword content token taken from the query OR from the retrieved hop-0 excerpts.
3. Anti-paraphrase: cue does not start with {what, when, how, why, who, which, where}.
4. Anti-duplication: Jaccard similarity with the query < 0.40.
5. Casual-chat register: at most 2 sentences.

## Spec constraints (set level)

6. Pairwise cue cosine < 0.85 (anti-redundant).
7. Each cue's cosine with the query > 0.30 (anti-random).

Prompt guidance (single template, used by every model): generate 1-2-sentence first-person chat-style text that would answer the question, pulling specific vocabulary from the hop-0 excerpts when relevant.

## Recall table

| Variant | Dataset | r@20 (arch) | r@50 (arch) | baseline r@20 | delta_r@20 | W/T/L@20 |
|---|---|---|---|---|---|---|
| meta_v2f (reference mini) | locomo_30q | 0.756 | 0.858 | 0.383 | +0.372 | 13/17/0 |
| meta_v2f (reference mini) | synthetic_19q | 0.613 | 0.851 | 0.569 | +0.044 | 8/7/4 |
| cuespec_mini | locomo_30q | 0.722 | 0.783 | 0.383 | +0.339 | 12/18/0 |
| cuespec_mini | synthetic_19q | 0.617 | 0.847 | 0.569 | +0.047 | 8/9/2 |
| cuespec_nano | locomo_30q | 0.511 | 0.589 | 0.383 | +0.128 | 7/21/2 |
| cuespec_nano | synthetic_19q | 0.604 | 0.824 | 0.569 | +0.034 | 6/11/2 |
| cuespec_nano_no_repair | locomo_30q | 0.544 | 0.622 | 0.383 | +0.161 | 8/20/2 |
| cuespec_nano_no_repair | synthetic_19q | 0.604 | 0.824 | 0.569 | +0.034 | 6/11/2 |
| v2f_nano | locomo_30q | 0.550 | 0.644 | 0.383 | +0.167 | 8/21/1 |
| v2f_nano | synthetic_19q | 0.592 | 0.809 | 0.569 | +0.023 | 5/9/5 |

## Repair-loop statistics (per dataset)

| Variant | Dataset | repair_rate | final_ok_rate | avg_repair_rounds | avg_llm_calls |
|---|---|---|---|---|---|
| cuespec_mini | locomo_30q | 0.00 | 0.83 | 0.00 | 1.00 |
| cuespec_mini | synthetic_19q | 0.00 | 1.00 | 0.00 | 1.00 |
| cuespec_nano | locomo_30q | 0.03 | 0.97 | 0.03 | 1.03 |
| cuespec_nano | synthetic_19q | 0.00 | 1.00 | 0.00 | 1.00 |
| cuespec_nano_no_repair | locomo_30q | 0.03 | 0.93 | 0.00 | 1.00 |
| cuespec_nano_no_repair | synthetic_19q | 0.00 | 1.00 | 0.00 | 1.00 |
| v2f_nano | locomo_30q | 0.00 | 0.37 | 0.00 | 1.00 |
| v2f_nano | synthetic_19q | 0.00 | 0.58 | 0.00 | 1.00 |

## First-attempt failure reasons (cuespec_nano, locomo_30q)

- `no_entity_overlap`: 1

## Sample cues (locomo_30q)

**Q:** When did Caroline go to the LGBTQ support group?
- mini + spec: ['I went to the LGBTQ support group yesterday evening after work and it was so powerful.', 'Caroline attended the LGBTQ support group last Saturday morning, right before she joined the parade.']
- nano + v2f : ['I went to a LGBTQ support group yesterday and it was so powerful.', 'LGBTQ support group yesterday']
- nano + spec (no repair): ['I went to the LGBTQ support group yesterday and it was incredibly empowering.', 'I left the LGBTQ support group yesterday feeling hopeful about building more community.']
- nano + spec attempt 0  : ['I went to the LGBTQ support group yesterday and it was incredibly empowering.', 'I left the LGBTQ support group yesterday feeling hopeful about building more community.']
- nano + spec final     : ['I went to the LGBTQ support group yesterday and it was incredibly empowering.', 'I left the LGBTQ support group yesterday feeling hopeful about building more community.'] (attempts=1)

**Q:** When did Melanie paint a sunrise?
- mini + spec: ['I painted the lake sunrise last year, it was during summer when I had a long weekend at the cabin.', 'I actually painted that beach sunrise just after my trip last week, inspired by watching the sun rise over the ocean.']
- nano + v2f : ["Yeah, I painted that lake sunrise last year! It's special to me.", 'I painted that lake sunrise last year.']
- nano + spec (no repair): ['I painted that lake sunrise last year; the colors still feel fresh to me.', 'That lake sunrise last year is special to me—it really captures the calm.']
- nano + spec attempt 0  : ['I painted that lake sunrise last year; the colors still feel fresh to me.', 'That lake sunrise last year is special to me—it really captures the calm.']
- nano + spec final     : ['I painted that lake sunrise last year; the colors still feel fresh to me.', 'That lake sunrise last year is special to me—it really captures the calm.'] (attempts=1)

**Q:** What fields would Caroline be likely to pursue in her educaton?
- mini + spec: ['I studied clinical psychology and counseling, plus community mental health internships and school counseling courses to prepare for jobs helping others.', 'I also took art therapy and ceramics classes plus educational psychology, planning to blend pottery into therapeutic programs and community arts education.']
- nano + v2f : ["I'm really eyeing psychology or counseling programs—helping people is what motivates me.", 'I’m also curious about social work or art therapy—combining care for others with creativity.']
- nano + spec (no repair): ['Caroline plans to study counseling after realizing she loves helping others.', "Caroline's plan includes psychology or social work courses to support clients."]
- nano + spec attempt 0  : ['Caroline plans to study counseling after realizing she loves helping others.', "Caroline's plan includes psychology or social work courses to support clients."]
- nano + spec final     : ['Caroline plans to study counseling after realizing she loves helping others.', "Caroline's plan includes psychology or social work courses to support clients."] (attempts=1)

**Q:** What did Caroline research?
- mini + spec: ['My research focused on work–family balance, studying how parents with young kids manage careers and childcare stress.', 'I researched job transition programs and vocational training that help people explore new jobs after a career or childcare break.']
- nano + v2f : ['Caroline researched', 'Caroline looked into']
- nano + spec (no repair): ['Caroline, my research on renewable energy startups looked at home solar adoption and financing options.', 'Caroline, I researched user onboarding for a tutoring app, focusing on engagement metrics and accessibility improvements.']
- nano + spec attempt 0  : ['Caroline, my research on renewable energy startups looked at home solar adoption and financing options.', 'Caroline, I researched user onboarding for a tutoring app, focusing on engagement metrics and accessibility improvements.']
- nano + spec final     : ['Caroline, my research on renewable energy startups looked at home solar adoption and financing options.', 'Caroline, I researched user onboarding for a tutoring app, focusing on engagement metrics and accessibility improvements.'] (attempts=1)

## Verdict

- Reference mini+v2f (meta_v2f) r@20 avg: **0.684**
- nano + vanilla v2f          r@20 avg: **0.571**  (delta vs ref = -0.113)
- nano + spec (no repair)     r@20 avg: **0.574**  (delta vs ref = -0.110)
- nano + spec + repair        r@20 avg: **0.557**  (delta vs ref = -0.127)
- mini + spec + repair        r@20 avg: **0.669**  (delta vs ref = -0.015)

nano+spec+repair reaches **81.4%** of the mini+v2f reference recall.
**FAIL.** Spec + repair did not rescue nano. Nano's language-model capacity is likely insufficient for this task.
Spec does not regress mini (delta vs v2f = -0.015).
Repair loop lift for nano: -0.017 r@20 (spec prompt alone vs spec+repair).

## Files

- `results/cue_spec.md` (this report)
- `results/cue_spec.json` (combined raw)
- `results/cuespec_cuespec_mini_locomo_30q.json`
- `results/cuespec_cuespec_mini_synthetic_19q.json`
- `results/cuespec_cuespec_nano_locomo_30q.json`
- `results/cuespec_cuespec_nano_synthetic_19q.json`
- `results/cuespec_cuespec_nano_no_repair_locomo_30q.json`
- `results/cuespec_cuespec_nano_no_repair_synthetic_19q.json`
- `results/cuespec_v2f_nano_locomo_30q.json`
- `results/cuespec_v2f_nano_synthetic_19q.json`
