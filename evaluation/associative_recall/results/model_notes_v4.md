# Model-note augmented EventMemory (v4)

## Architecture (v4)

- Same EM ingestion + V_combined retrieval pipeline as v3. Only difference: the note-generation prompt and its output shape.
- v4 prompt = "neutral A''" listener-observations prompt (RESOLVED/FACT/COUNT/UPDATE/LINK/NAME labels + PHATIC skip rule), with domain-neutral inline examples. Source: `notes_prompt_neutral.py` / `em_setup_notes_v4.py`.
- PHATIC turns produce no note event (v3 always produced a 3-line note).

## Ingest stats (v4)

- Turns ingested: 1451
- Notes ingested (non-PHATIC): 1034
- PHATIC turns (note skipped): 417  (28.7% of turns)
- Mean labels per non-PHATIC note: 3.00

| Label | Count | % of labels |
| --- | --- | --- |
| RESOLVED | 691 | 22.3% |
| FACT | 2283 | 73.6% |
| COUNT | 22 | 0.7% |
| UPDATE | 1 | 0.0% |
| LINK | 81 | 2.6% |
| NAME | 26 | 0.8% |

## Retrieval (LoCoMo-30)

v3 did not run the retrieval phase. Baseline reference is the standard `em_v2f_speakerformat` (no notes) reported in `model_notes.md`: R@20=0.8167, R@50=0.8917.

| Architecture | R@20 | delta vs em_v2f | R@50 | delta vs em_v2f | time (s) |
| --- | --- | --- | --- | --- | --- |
| `em_cosine_notes_v4` | 0.7583 | -0.0584 | 0.7833 | -0.1084 | 5.8 |
| `em_v2f_notes_v4` | 0.7472 | -0.0695 | 0.8583 | -0.0334 | 187.7 |
| `em_v2f_notes_msgs_only_v4` | 0.8167 | +0.0000 | 0.8917 | +0.0000 | 27.8 |
| `em_v2f_notes_only_v4` | 0.7333 | -0.0834 | 0.8583 | -0.0334 | 178.4 |

## Task-sufficiency (20 proactive tasks, LLM judge)

- A = single-shot v2f over STANDARD ingest (messages only).
- B = single-shot v2f over v4 NOTES ingest (messages + notes).
- C = proactive decomposition over v4 NOTES ingest.

### v4 results

| System | Suff | Cov | Depth | Noise | LLM calls | notes in retrieval |
| --- | --- | --- | --- | --- | --- | --- |
| A_std_singleshot | 4.6 | 4.75 | 4.3 | 6.35 | 1.0 | 0.0 |
| B_notes_singleshot | 5.5 | 5.35 | 4.7 | 7.0 | 1.0 | 29.45 |
| C_notes_proactive | 6.0 | 6.05 | 5.6 | 7.25 | 7.35 | 23.15 |

**v4 Delta vs baseline A**: B-A = +0.900, C-A = +1.400.

### v3 comparison (from `model_notes.md`)

| System | v3 Suff | v4 Suff | v4 - v3 |
| --- | --- | --- | --- |
| A_std_singleshot | 4.85 | 4.6 | -0.250 |
| B_notes_singleshot | 5.1 | 5.5 | +0.400 |
| C_notes_proactive | 5.45 | 6.0 | +0.550 |

### Per-task sufficiency (v4)

| task | conv | shape | A | B | C | notes@B | notes@C |
| --- | --- | --- | --- | --- | --- | --- | --- |
| t01 | 26 | draft | 6 | 7 | 8 | 39 | 17 |
| t02 | 26 | plan | 8 | 7 | 8 | 42 | 31 |
| t03 | 26 | analysis | 5 | 7 | 7 | 42 | 25 |
| t04 | 26 | brief | 7 | 9 | 8 | 30 | 17 |
| t05 | 26 | decision | 5 | 7 | 8 | 45 | 34 |
| t06 | 26 | synthesis | 8 | 7 | 8 | 11 | 16 |
| t07 | 30 | plan | 2 | 2 | 3 | 31 | 22 |
| t08 | 30 | brief | 3 | 6 | 7 | 37 | 37 |
| t09 | 30 | draft | 8 | 4 | 8 | 48 | 19 |
| t10 | 30 | analysis | 1 | 1 | 2 | 45 | 29 |
| t11 | 30 | decision | 6 | 4 | 6 | 13 | 38 |
| t12 | 30 | synthesis | 6 | 6 | 6 | 39 | 23 |
| t13 | 41 | analysis | 2 | 2 | 2 | 30 | 17 |
| t14 | 41 | plan | 5 | 7 | 7 | 14 | 18 |
| t15 | 41 | brief | 2 | 7 | 5 | 20 | 15 |
| t16 | 41 | draft | 1 | 2 | 2 | 9 | 28 |
| t17 | 41 | decision | 3 | 6 | 4 | 11 | 6 |
| t18 | 41 | synthesis | 8 | 7 | 7 | 3 | 9 |
| t19 | 26 | plan | 3 | 6 | 6 | 37 | 31 |
| t20 | 30 | synthesis | 3 | 6 | 8 | 43 | 31 |

## Verdict

- Retrieval: `em_v2f_notes_v4` R@50 = 0.8583 vs em_v2f baseline 0.8917 (delta -0.0334).
- Task-sufficiency: v4 best delta vs baseline A = +1.400 (v3 best was +0.600). B-A=+0.900, C-A=+1.400.
- **Verdict: SHIP v4 (sufficiency-only; retrieval neutral)**

## Outputs

- `results/model_notes_v4.json`
- `results/model_notes_v4_retrieval.json`
- `results/model_notes_v4_sufficiency.json`
- `results/model_notes_v4.md`
- Notes collections manifest: `results/eventmemory_notes_v4_collections.json`
- Source: `em_setup_notes_v4.py`, `notes_eval_v4.py`