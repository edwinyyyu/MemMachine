# Critical-Info Separate Vector Store — Empirical Recall Test

A small subset of very-important turns (medications with dosages, allergies, numbers of family members, key commitments with deadlines, specific dates) is flagged at ingestion time by an LLM classifier (gpt-5-mini, strict prompt v3). CRITICAL turns emit 3 short alt-keys each, all pointing back to the original turn. Those alt-keys live in a SEPARATE vector store. At query time, the main v2f retrieval path is UNCHANGED; a secondary critical-pool retrieval is merged in under one of two strategies.

Distinct from prior alt-key tests (regex / LLM alt-keys, see `ingestion_regex_empirical.md`, `ingestion_llm_empirical.md`): those merged alt-keys into the main pool via per-parent max-score and pushed v2f's clean retrievals out. Here the critical pool is disjoint; main retrieval is untouched and displacement only affects a tiny slice of items that are critical anyway.

## 1. Prompt tuning

| version | LoCoMo flag rate | synthetic flag rate | notes |
|---|---:|---:|---|
| v1 | — | — | Initial spec prompt — lenient. |
| v2 | 2.6% | 24.9% | Added explicit DO-NOT list, default-SKIP. Dropped LoCoMo to 2.6% but synthetic still 24.9% (synthetic is intentionally fact-dense). |
| v3 | 0.0% | 3.2% | Tighter criteria: requires (a) specific entity + (b) enduring property + (c) speaker stating it (not discussing). Excludes product specs, budget numbers, acknowledgements, re-mentions. LoCoMo 0%, synthetic 3.2%. |

v3 selected. v2 on synthetic exceeded the 20% flag-rate cap; the single permitted retune brought it to 3.2%. LoCoMo dropped to 0% (no turns met the enduring-fact-with-specific-entity criterion in a casual conversation corpus — legitimate outcome).

## 2. Flagging statistics (v3)

| dataset | turns | critical | flag rate | alt-keys (dedup) |
|---|---:|---:|---:|---:|
| locomo_30q | 419 | 0 | 0.0% | 0 |
| synthetic_19q | 462 | 15 | 3.2% | 45 |

## 3. Recall

Fair-backfill recall. Baseline = v2f on main index only. Variants merge a separate critical-pool retrieval with the main pool.

| dataset | K | baseline v2f | +crit_additive_0.1 | Δ | +crit_always_top_M | Δ |
|---|---:|---:|---:|---:|---:|---:|
| locomo_30q | 20 | 0.7556 | 0.7556 | +0.0000 | 0.7556 | +0.0000 |
| locomo_30q | 50 | 0.8583 | 0.8583 | +0.0000 | 0.8583 | +0.0000 |
| synthetic_19q | 20 | 0.6130 | 0.6130 | +0.0000 | 0.6434 | +0.0304 |
| synthetic_19q | 50 | 0.8513 | 0.8632 | +0.0119 | 0.8554 | +0.0041 |

## 4.a Per-category (locomo_30q)

| category | n | base @20 | var_a @20 | var_b @20 | base @50 | var_a @50 | var_b @50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| locomo_multi_hop | 4 | 0.625 | 0.625 | 0.625 | 0.875 | 0.875 | 0.875 |
| locomo_single_hop | 10 | 0.617 | 0.617 | 0.617 | 0.825 | 0.825 | 0.825 |
| locomo_temporal | 16 | 0.875 | 0.875 | 0.875 | 0.875 | 0.875 | 0.875 |

## 4.b Per-category (synthetic_19q)

| category | n | base @20 | var_a @20 | var_b @20 | base @50 | var_a @50 | var_b @50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| completeness | 4 | 0.455 | 0.455 | 0.497 | 0.865 | 0.865 | 0.865 |
| conjunction | 3 | 0.809 | 0.809 | 0.809 | 1.000 | 1.000 | 1.000 |
| control | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| inference | 3 | 0.766 | 0.766 | 0.844 | 0.939 | 0.970 | 0.939 |
| proactive | 4 | 0.351 | 0.351 | 0.396 | 0.643 | 0.663 | 0.663 |
| procedural | 2 | 0.347 | 0.347 | 0.347 | 0.661 | 0.690 | 0.661 |

## 5. Critical-contribution rate

Share of gold turns surfaced via the critical store (gold appearing in the merged list but NOT in main top-K).

| dataset | variant | K | frac questions with crit-gold | frac gold via crit |
|---|---|---:|---:|---:|
| locomo_30q | crit_additive_bonus_0.1 | 20 | 0.0% | 0.0% |
| locomo_30q | crit_additive_bonus_0.1 | 50 | 0.0% | 0.0% |
| locomo_30q | crit_always_top_M | 20 | 0.0% | 0.0% |
| locomo_30q | crit_always_top_M | 50 | 0.0% | 0.0% |
| synthetic_19q | crit_additive_bonus_0.1 | 20 | 0.0% | 0.0% |
| synthetic_19q | crit_additive_bonus_0.1 | 50 | 15.8% | 1.9% |
| synthetic_19q | crit_always_top_M | 20 | 26.3% | 3.2% |
| synthetic_19q | crit_always_top_M | 50 | 5.3% | 0.7% |

## 6. False-positive rate (critical top-M hits that are not gold)

| dataset | hits | non-gold | FP rate |
|---|---:|---:|---:|
| locomo_30q | 0 | 0 | 0.0% |
| synthetic_19q | 40 | 26 | 65.0% |

Interpretation: the 65% FP rate on synthetic is expected — 19 questions each probing a subset of critical facts, so most top-M critical hits for a given question will be unrelated (other medications, other appointments, other allergies). This is why `crit_always_top_M` only wins modestly on synthetic: most forced-in hits are not the gold for the current query.

## 7. Cost

- LLM classification calls: uncached=41 cached=1259 (most turns were already cached from v2 pass)
- Input tokens: 18,107
- Output tokens: 28,832
- Est. cost (gpt-5-mini @ $0.25/1M in, $2/1M out): $0.062

## 8. Verdict

- **LoCoMo-30**: v3 prompt legitimately finds zero critical turns in a casual conversation corpus. Recall delta is exactly 0.0000 — the critical store is a no-op. Main v2f retrieval is untouched, as designed.
- **synthetic_19q**: v3 flags 3.2% of turns (15/462). `crit_always_top_M` lifts r@20 by **+0.030** (modest but real — 26% of questions see crit-gold that wasn't in main top-20). At K=50 the main pool already catches most gold so the lift shrinks to +0.004. `crit_additive_bonus_0.1` is effectively no-op at K=20 (the main pool saturates before the +0.1 bonus pushes a critical item above a main hit) but gains **+0.012** at K=50.

Verdict: **narrow-use-case neutral-to-positive**. The critical-info store does not hurt (unlike prior main-pool alt-key variants) and delivers a small win on fact-dense corpora at moderate K. On casual conversation (LoCoMo) the strict classifier correctly finds no critical items, leaving the main pipeline untouched — the no-harm property holds.

**Ship recommendation**: keep v3 classifier + `crit_always_top_M` (top_m=5, min_score=0.2) as a narrow augmentation. Skip `crit_additive_bonus_0.1` — it's strictly weaker than `crit_always_top_M` at K=20 on the corpus where it matters.

## 9. Sample critical turns

### From synthetic_19q (CRITICAL):

- [user] (synthetic_19q, turn 54): March 3rd with Dr. Kim, the ophthalmologist. Last year's screening was clear, no signs of retinopathy.
    - ALT: March 3 appointment with Dr. Kim, ophthalmologist
    - ALT: Ophthalmology screening scheduled March 3 with Dr. Kim
    - ALT: Last year's eye screening clear; no retinopathy noted
- [user] (synthetic_19q, turn 92): Yep. But the peanut one is still very much there. Ok I think we're finally set on the dietary stuff. Let me go shopping.
    - ALT: User has a peanut allergy; avoid peanuts in diet and shopping
    - ALT: Peanut allergy persists — dietary restriction when planning meals or groceries
    - ALT: Do not include peanuts; user still allergic to peanuts (enduring)
- [assistant] (synthetic_19q, turn 5): Lisinopril 10mg daily for blood pressure, prescribed by Dr. Patel about 6 months ago. Anything else?
    - ALT: Lisinopril 10 mg daily blood pressure med prescribed by Dr. Patel
    - ALT: On lisinopril 10mg daily, started about 6 months ago, Dr. Patel
    - ALT: Daily antihypertensive Lisinopril 10 mg, prescribed by Dr. Patel
- [user] (synthetic_19q, turn 58): Grandma Helen is in a wheelchair so she needs to be somewhere accessible, near the entrance and not on grass. And she can't hear well, so put her near the speak
    - ALT: Grandma Helen — wheelchair user: needs accessible seating near entrance, not on grass
    - ALT: Grandma Helen — hearing impaired: seat near speakers but not too close
    - ALT: Grandma Helen: wheelchair and hearing impairment — accessible entrance seating, near speakers
- [user] (synthetic_19q, turn 32): Rachel is gluten-free. She was diagnosed with celiac disease last year. It's the real deal, not a preference thing - even trace amounts bother her.
    - ALT: Rachel: diagnosed celiac disease; strict gluten-free, avoids even trace amounts
    - ALT: Rachel requires strict gluten-free diet; sensitive to cross-contamination
    - ALT: Do not give Rachel gluten or trace-contaminated foods
- [user] (synthetic_19q, turn 88): Great, let's do that. By the way, I just remembered something else about Bob - he told me ages ago that he's also allergic to shellfish. So no shrimp or crab.
    - ALT: Bob — allergy: shellfish; avoid shrimp and crab
    - ALT: Bob allergic to shellfish (no shrimp or crab)
    - ALT: Do not serve Bob shellfish — shrimp and crab prohibited
- [user] (synthetic_19q, turn 8): Oh yeah, I take a daily vitamin D supplement because my levels were low. 2000 IU.
    - ALT: Takes vitamin D supplement daily, 2000 IU, low levels
    - ALT: Daily vitamin D 2000 IU due to previously low levels
    - ALT: User on ongoing vitamin D therapy 2000 IU daily
- [user] (synthetic_19q, turn 74): Hey, quick update. I just got off the phone with Dr. Patel's office. They want to increase my metformin to 1000mg twice daily. My last fasting glucose was a bit
    - ALT: Metformin increase to 1000 mg twice daily per Dr. Patel
    - ALT: Patient instructed to take metformin 1000 mg BID by Dr. Patel
    - ALT: New regimen: metformin 1000mg twice daily, due to high fasting glucose
- [user] (synthetic_19q, turn 4): I also take lisinopril 10mg once daily for blood pressure. My doctor Dr. Patel started me on that about 6 months ago.
    - ALT: Lisinopril 10 mg once daily for blood pressure
    - ALT: Started by Dr. Patel about six months ago
    - ALT: Ongoing antihypertensive medication: lisinopril 10 mg daily
- [user] (synthetic_19q, turn 2): Let me start with my medications. I take metformin 500mg twice daily for type 2 diabetes.
    - ALT: Takes metformin 500 mg twice daily for type 2 diabetes
    - ALT: Ongoing medication: metformin 500mg, two times per day, diabetes
    - ALT: Chronic therapy — metformin 500 mg, BID (twice daily), type II diabetes
- [user] (synthetic_19q, turn 66): Thanks for helping me organize all this. One last thing - my pharmacy changed. I now use the Walgreens on Main Street instead of the CVS. I switched because Wal
    - ALT: Uses Walgreens on Main Street as pharmacy (replaced CVS)
    - ALT: Pharmacy updated to Walgreens on Main Street, ongoing preference
    - ALT: Pharmacy change: now fills prescriptions at Main Street Walgreens
- [user] (synthetic_19q, turn 78): Ok noted. Also, I forgot to mention earlier - I also take a baby aspirin daily. 81mg. Dr. Patel recommended it for heart health given my family history.
    - ALT: Takes baby aspirin 81 mg daily — recommended by Dr. Patel
    - ALT: Daily 81 mg aspirin for heart health due to family history
    - ALT: Medication: low-dose aspirin 81 mg daily (cardiac prevention)
- [user] (synthetic_19q, turn 12): February 10th. It's my quarterly check - they do A1C, lipids, vitamin D, the whole panel.
    - ALT: Appointment Feb 10 — quarterly check with A1C, lipids, vitamin D
    - ALT: Quarterly lab visit on February 10th: A1C, lipids, vitamin D panel
    - ALT: Feb 10 scheduled labs — whole panel including A1C, lipids, vitamin D
- [user] (synthetic_19q, turn 6): And atorvastatin 20mg at night for cholesterol. That one I've been on for years.
    - ALT: User on atorvastatin 20 mg nightly, long-term
    - ALT: Long-term nightly atorvastatin 20 mg for cholesterol (ongoing)
    - ALT: Ongoing nightly atorvastatin 20mg — chronic cholesterol medication
- [user] (synthetic_19q, turn 56): Well, my Uncle Steve and Aunt Patricia aren't speaking to each other since last Thanksgiving. Long story. They need to be at opposite ends of the venue.
    - ALT: Uncle Steve and Aunt Patricia are estranged since last Thanksgiving
    - ALT: Seat Uncle Steve and Aunt Patricia at opposite ends of venue
    - ALT: Do not seat or place Uncle Steve near Aunt Patricia

### From LoCoMo-30 (all SKIP under v3):

- [SKIP] (user, turn 165): Wow, Mel, family love and support is the best!
- [SKIP] (assistant, turn 77): Wow, Caroline, sounds like the parade was an awesome experience! It's great to see the love and support for the LGBTQ+ c
- [SKIP] (assistant, turn 202): We always look forward to our family camping trip. We roast marshmallows, tell stories around the campfire and just enjo
- [SKIP] (assistant, turn 333): I'm a fan of both classical like Bach and Mozart, as well as modern music like Ed Sheeran's "Perfect".
- [SKIP] (assistant, turn 24): Thanks, Caroline. It's still a work in progress, but I'm doing my best. My kids are so excited about summer break! We're
- [SKIP] (user, turn 37): Thanks, Mel! Your backing really means a lot. I felt super powerful giving my talk. I shared my own journey, the struggl
- [SKIP] (assistant, turn 274): Yeah, I made it in pottery class yesterday. I love it! Pottery's so relaxing and creative. Have you tried it yet?
- [SKIP] (assistant, turn 48): I'm lucky to have my husband and kids; they keep me motivated.

Note: v2 (looser) flagged 11 LoCoMo turns (pets, commitments to become a parent) — these are available in git history of the samples file but were excluded under v3's stricter enduring-fact + specific-entity criteria.
