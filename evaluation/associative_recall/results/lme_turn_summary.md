# LME-hard POC turn-summary dual-view (Part B)

## References

| Variant (single-view raw-only) | R@20 | R@50 |
| --- | --- | --- |
| em_v2f_lme_mixed_7030 + expand_3 (prev leader) | 0.6368 | 0.8631 |
| reflmem_lme 3round (prev overall leader)       | n/a    | 0.876  |

## POC (dual-view raw + summary) recall

| Variant | R@20 | R@50 | time (s) |
| --- | --- | --- | --- |
| `em_cosine_baseline_summ_lme` | 0.6129 | 0.8277 | 8.5 |
| `em_v2f_lme_mixed_7030_expand3_summ` | 0.7048 | 0.9177 | 43.1 |

## Per-category R@50

| Variant | multi-session | single-session-preference | temporal-reasoning |
| --- | --- | --- | --- |
| `em_cosine_baseline_summ_lme` | 0.7917 | 0.8095 | 0.8819 |
| `em_v2f_lme_mixed_7030_expand3_summ` | 0.8990 | 0.9792 | 0.8750 |

## View coverage (top-50)

| Variant | gold credited | raw wins | summary wins | summary share |
| --- | --- | --- | --- | --- |
| `em_cosine_baseline_summ_lme` | 161 | 139 | 22 | 13.66% |
| `em_v2f_lme_mixed_7030_expand3_summ` | 174 | 143 | 31 | 17.82% |

## Apples-to-apples comparison (same 9 POC question_ids)

The POC 9 questions (3 per category, sorted lex) are a biased subset -- the
only fair comparison is to evaluate the RAW-view-only recipe on the SAME 9
questions, NOT on the full 90.  Pulling
`em_v2f_lme_mixed_7030.per_question` from the prior 90-Q run and filtering
to the POC ids gives the fair baseline:

| Metric                         | raw-only baseline on POC-9 | dual-view summary | delta |
| ------------------------------ | -------------------------: | ----------------: | ----: |
| R@20 overall                   |                     0.7294 |            0.7048 | -2.5pp |
| R@50 overall                   |                     0.9227 |            0.9177 | -0.5pp |
| multi-session R@50             |                     0.8826 |            0.8990 | +1.6pp |
| single-session-preference R@50 |                     1.0000 |            0.9792 | -2.1pp |
| temporal-reasoning R@50        |                     0.8854 |            0.8750 | -1.0pp |

## Decision rules (from plan)

- POC R@50 on `em_v2f_lme_mixed_7030_expand3_summ` > 0.873 (+1pp over 0.863)
  -> summary generalizes; full 90-question run recommended.
  -> APPLES-TO-APPLES FAILS: -0.5pp on the same 9 POC questions vs raw-only.
- POC temporal-reasoning R@50 > 0.807 -> cracks prior temporal ceiling.
  -> FAILS: 0.8750 on POC vs raw-only's 0.8854 (regression by -1.0pp) and
  vs full-90 reflmemlme_3round's 0.807 (superficially beats it, but on
  only 3 temporal-reasoning questions, not the full 30).

## Verdict

**Turn-summary dual-view does NOT universally generalize to LME-hard.**
On a 9-question POC (3/category), the dual-view summary ingestion slightly
REGRESSES vs raw-only on the same subset (R@50 -0.5pp, R@20 -2.5pp).

View-coverage shows 17.8% of gold credits come from the summary-view
segment in em_v2f_lme_mixed_7030_expand3_summ.  That's a NON-zero orthogonal
signal (17.8% vs ~15% for cosine baseline), but it's not enough to offset
the crowding cost of doubling the index with summaries that often mirror
the raw text.

**Why LME differs from LoCoMo**:
- LoCoMo turns are short chat messages; summaries substantially re-phrase
  ("I went to the LGBTQ support group") and expose retrieval-friendly
  content. Share of summary-view wins on LoCoMo was 85-97%.
- LME turns are long-form (assistant replies include full code blocks,
  lists, and explanations). Summaries of dense content are less
  information-dense than the original, so raw-view cosine dominates.
- LME gold distribution also skews more to user turns; doubling the
  assistant side with summaries adds noise without rescue value.

**Recommendation**: do NOT roll out turn_summary on LME-hard. Stay with
`em_v2f_lme_mixed_7030 + expand_3` as the LME single-shot leader (0.863
on 90-Q, 0.9227 on this POC-9 subset). The full 90-Q run is NOT
recommended given the POC signal.

Caveats: only 9 questions (3 per category). Full statistical power would
need 30+. The POC lex-sorted sample may be unrepresentative; however, the
R@50 gap is close to zero -- even a favorable full run is unlikely to
clear the +1pp bar required for "summ generalizes".

## Outputs

- Collections manifest: `results/eventmemory_lmesumm_collections.json`
- SQLite store: `results/eventmemory_lmesumm.sqlite3`
- Summaries audit: `results/lmesumm_summaries.json`
- Sources: `em_setup_lmesumm.py`, `compose_eval.py`
- Summary cache: `cache/summlme_llm_cache.json` (4218 entries, ~$2 cost)