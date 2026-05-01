# Model-note augmented EventMemory (v3)

## Architecture (EM-format context, v3)

- Per turn: ingest message normally (speaker baked via `MessageContext(source=speaker)`). At `turn_ts + 1us`, generate a note and ingest it as a second event with `MessageContext(source="ModelNote")`, `event_type="model_note"`.
- Note-generator (gpt-5-mini, `reasoning_effort="low"`) sees context formatted as EM-canonical strings `"<source>: <content>"`: prior notes appear as `ModelNote: ...`, recent and retrieved messages as `<speaker>: ...`. No JSON, no bracket labels at note-gen time.
- Similarity retrieval at note time: V_combined — latest prior note + last 3 turns, joined as EM-format strings, top-K=4.
- Output prose is three labeled lines `current_understanding / open_questions / recent_realization` concatenated into one paragraph and ingested as `"ModelNote: <prose>"`.

## Sample notes (conv-26)

### late (turn 418, Caroline)

**Turn**: Yeah, that's true! It's so freeing to just be yourself and live honestly. We can really accept who we are and be content.

**Note**: current_understanding: Caroline is increasingly comfortable with her identity and emphasizes the freedom of living honestly; Melanie remains a warm, affirming friend who continues to reinforce Caroline’s confidence and plans to help others. open_questions: expected timeline for placement, whether Caroline needs practical or financial help, who else is in her support network recent_realization: This turn reiterates Caroline’s emotional readiness and contentment rather than adding new logistical details, reinforcing that the conversation is focused on mutual emotional support and affirmation.

## Task-sufficiency (20 proactive tasks, LLM judge)

- A = single-shot v2f over STANDARD ingest (messages only).
- B = single-shot v2f over NOTES ingest (messages + notes).
- C = proactive decomposition over NOTES ingest.

| System | Suff | Cov | Depth | Noise | LLM calls | notes in retrieval |
| --- | --- | --- | --- | --- | --- | --- |
| A_std_singleshot | 4.85 | 4.95 | 4.55 | 7.0 | 1.0 | 0.0 |
| B_notes_singleshot | 5.1 | 5.6 | 4.45 | 7.85 | 1.0 | 48.75 |
| C_notes_proactive | 5.45 | 5.85 | 4.75 | 7.5 | 7.35 | 42.95 |

**Delta vs baseline A**: B-A = +0.250, C-A = +0.600.

### Per-task sufficiency

| task | conv | shape | A | B | C | notes@B | notes@C |
| --- | --- | --- | --- | --- | --- | --- | --- |
| t01 | 26 | draft | 7 | 8 | 8 | 50 | 47 |
| t02 | 26 | plan | 8 | 7 | 7 | 49 | 48 |
| t03 | 26 | analysis | 5 | 6 | 8 | 50 | 45 |
| t04 | 26 | brief | 8 | 8 | 8 | 50 | 43 |
| t05 | 26 | decision | 7 | 4 | 4 | 49 | 46 |
| t06 | 26 | synthesis | 8 | 7 | 7 | 50 | 42 |
| t07 | 30 | plan | 2 | 3 | 4 | 46 | 48 |
| t08 | 30 | brief | 5 | 8 | 6 | 50 | 49 |
| t09 | 30 | draft | 7 | 5 | 4 | 50 | 40 |
| t10 | 30 | analysis | 1 | 2 | 1 | 50 | 44 |
| t11 | 30 | decision | 7 | 4 | 7 | 50 | 46 |
| t12 | 30 | synthesis | 6 | 4 | 5 | 50 | 50 |
| t13 | 41 | analysis | 2 | 2 | 2 | 49 | 35 |
| t14 | 41 | plan | 4 | 6 | 5 | 50 | 29 |
| t15 | 41 | brief | 2 | 6 | 6 | 48 | 48 |
| t16 | 41 | draft | 1 | 2 | 2 | 43 | 40 |
| t17 | 41 | decision | 4 | 6 | 7 | 44 | 28 |
| t18 | 41 | synthesis | 6 | 8 | 8 | 47 | 33 |
| t19 | 26 | plan | 2 | 3 | 3 | 50 | 49 |
| t20 | 30 | synthesis | 5 | 3 | 7 | 50 | 49 |

## Verdict

- Task-sufficiency: best delta over standard-ingest baseline = +0.60. Verdict: **Tie**.

## Outputs

- `results/model_notes.json`
- `results/model_notes.md`
- Notes collections manifest: `results/eventmemory_notes_v3_collections.json`
- Source: `em_setup_notes_v3.py`, `notes_eval.py`