# Logic-Constraint Failure Analysis: why v15/v2f cues hurt below cosine baseline

## Headline numbers (puzzle_16q, fair-backfill r@20)

| Category         | n | baseline r@20 | v15 r@20 | v2f r@20 | delta |
| ---------------- | - | ------------- | -------- | -------- | ----- |
| logic_constraint | 3 | 0.242         | 0.166    | 0.166    | -0.076|
| all other cats   |13 | 0.474         | 0.579    | 0.579    | +0.10+|

Logic-constraint is the only category where cue generation is a net loss at r@20 (W/T/L 0/0/3). At r@50 both are tied with baseline, meaning the missed turns are still out there, cosine just needs more budget to find them.

## Why v15/v2f lose at r@20

All three LC questions have the same structural property: **the answer is assembled from 11-19 scattered, LOW-similarity source turns** (avg cos(q, source) ≈ 0.30, with many sources below 0.25). Cosine top-10 only reaches 2-4 of them.

V15/V2F's hop-0 uses cosine top-10 (a very small seed), then produces 2 cues that spend 10 slots each. With `exclude_indices`, those 20 cue slots cannot be used to backfill the cosine neighborhood. The 3 hop-0 cosine hits that would have appeared at cosine ranks 11-20 never get recovered.

Concretely:
- Cosine top-20 on Q0 finds turns {6, 24, 28, 29} (4 hits).
- V15/V2F get the same 10-hop0 (which contains {6, 28, 29}), then burn 20 slots on cues that find 0-1 new source turns. Turn 24 (which is cosine rank 11-20) is displaced and only shows up via backfill if < 20 unique cue results came back.
- Net: both v15 and v2f land at 3 hits vs baseline 4. Same pattern on Q1 (lost turns 24, 28) and Q2 (lost turns 16, 70).

## Per-question breakdown

### Q0 — "Final valid desk arrangement for 6 desks" (puzzle_logic_1)

- 12 source turns; cosine top-20 finds 4, v15/v2f find 3 (lost turn 24).
- V2F cues:
  - `CUE 0` (cos to question = **0.783**): `"Final valid desk arrangement for the six desks listing who sits at desks 1–6, respecting 'you in desk 1', 'Bob desk 6', and 'Alice and Dave need at least one desk between them'"`
  - `CUE 1` (cos = 0.541): `"Henderson project argument that 'Alice and Dave cannot be adjacent' plus any lines assigning Alice, Carol, Dave, Eve to specific desk numbers"`
- V15 cues: similar "final valid desk arrangement desks 1-6..." construction (cos 0.75 / 0.63).

Missed source turns (examples):
- Turn 12 (cos to q = 0.19): *"Alice says she absolutely cannot sit next to Dave. They had that whole argument about the Henderson project..."* — MAX cue sim: 0.479.
- Turn 72 (cos = 0.19): *"I just got a text from Dave. He says 'we actually cleared the air about the Henderson thing'"* — cue sim 0.41.
- Turn 44 (cos = 0.29): *"Eve just responded. She says she doesn't really care where she sits as long as she's not at either end"* — cue sim 0.36.
- Turn 50 (cos = 0.25): *"She and Eve work on the same sub-team and they pair-program a lot, so it'd be nice if they sat next to each other"* — cue sim 0.38.
- Turn 74 (cos = 0.35): *"So we're back to the arrangement: me at 1, then Alice and Dave in 2 and 3..."* — cue sim 0.61. This is almost literally the answer; even the best cue does not retrieve it because many higher-scoring non-source turns rank ahead at top-10.

### Q1 — "All constraints including those resolved or irrelevant" (puzzle_logic_1)

- 11 source turns; cosine top-20 finds 2, v15/v2f find 1 (lost both turns 24 and 28, gained turn 6 — net -1).
- V2F cue 0 (cos = **0.781**): `"Summarize all desk-arrangement constraints mentioned (Alice and Dave not adjacent; desk 1 preferred for the whiteboard; all desks by the windows; Bob may move)..."` — this is a literal summary/paraphrase of the question.

Missed source turns:
- Turn 78 (cos = 0.25): *"Eve just sent another message - she says she also wants to be near the office plants. Where are the plants?"* — max cue sim 0.44.
- Turn 80 (cos = 0.35): *"The big fiddle leaf fig is between desks 3 and 4..."* — cue sim 0.34.
- Turn 82 (cos = 0.39): *"Bob said something else...he's bringing a standing desk converter and the IT closet with the extra monitors is near desk 6"* — cue sim 0.46.
- Turn 12 (Alice-Dave conflict), Turn 72 (conflict resolved) — same as Q0.

### Q2 — "Final conference room schedule including changes" (puzzle_logic_2)

- 19 source turns; cosine top-20 finds 4, v15/v2f find 3 (lost 16, 70; gained 8).
- V2F cue 0 (cos = 0.698): `"Final Room A schedule for next week including all changes: Monday 9-11am Marketing strategy, Monday 1-3pm Engineering sprint planning, Tuesday 10-12pm HR all-hands, Wednesday 2-4pm Marketing brainstorm (moved), Thursday 10am-3pm Sales client presentation..."` — essentially a paraphrase of the expected answer, not the conversation.

Missed source turns:
- Turn 10 (cos = 0.28): *"Room B only fits 8 comfortably. So Marketing needs Room A."* — cue sim 0.51.
- Turn 52 (cos = 0.41): *"Sales presentation - Kevin told me the clients rescheduled. They want to come Thursday now"* — this is a critical UPDATE turn, cue sim 0.47.
- Turn 58 (cos = 0.32): *"If Sales isn't on Wednesday anymore, then Marketing could go back to Wednesday afternoon."* — cue sim 0.52.
- Turn 62 (cos = 0.36): *"She says yes, Wednesday 2-4 works for her and the team. Lock it in."* — cue sim 0.44.
- Turn 44 (cos = 0.25): *"Just 5 people, so Room B would actually be fine for that."* — cue sim 0.41.

## Hypothesis confirmed: cues are paraphrases of the question

Cue-to-question cosine similarity across all 12 LC cues:

| arch | n  | mean  | min   | max   | frac > 0.6 | frac > 0.7 |
| ---- | -- | ----- | ----- | ----- | ---------- | ---------- |
| v15  | 6  | 0.603 | 0.499 | 0.746 | 4/6        | 1/6        |
| v2f  | 6  | 0.645 | 0.453 | 0.782 | 4/6        | 2/6        |

**V2F cues are even more paraphrase-like than V15** (mean 0.645 vs 0.603). The anti-question instruction ("don't write questions") does not stop the LLM from emitting declarative paraphrases of the expected *answer structure*.

Also notable: for 16/35 missed source turns across the 3 questions, the best v2f cue's similarity to the missed turn is **lower** than the *next higher-ranked non-source turn* pulled by that same cue — which is exactly what causes the cue's top-10 to be non-source-turn noise.

## What the conversation vocabulary looks like

The missed turns share a distinctive **informal, update-with-context** register:

| vocabulary pattern            | example |
| ----------------------------- | ------- |
| arrival of new info           | "I just got a message from", "just got a text from", "Lisa just mentioned" |
| preference statement          | "she says she doesn't really care where she sits as long as..." |
| conflict/constraint           | "absolutely cannot sit next to", "non-negotiable", "set in stone" |
| resolution/update             | "we actually cleared the air", "rescheduled", "clients rescheduled", "forgot", "actually, " |
| local physical reference      | "between desks 3 and 4", "server room door on the right side", "IT closet near desk 6" |
| afterthought                  | "Oh wait, one more thing", "Oh crud, I just remembered", "Oh right!" |
| deictic pronouns instead of named entities | "she said", "he says", "they rescheduled" — which tank cosine against a question that names no one |

None of the v15/v2f cues use any of this vocabulary. The cues are written in *answer-document register* ("Final Room A schedule for next week including all changes:...") which embeds near the question, not near the conversation.

## Comparison with the approach that got 100% r@all

`constraint_retrieval.py`'s iterative constraint collection (cache file `constraint_llm_cache.json`) achieved 100% recall on all 3 LC questions. Its prompt succeeds because:

1. It explicitly names **constraint types** ("location/proximity preferences", "interpersonal conflicts", "updates/overrides", "pair/group requirements").
2. Round 2+ explicitly asks *"what KINDS of constraints are probably missing — any UPDATES, resolutions, overrides? Any constraints from OTHER people?"*.
3. This shifts the LLM from paraphrasing the question into generating conversational snippets with vocabulary that matches the long tail.

Example cues generated by the constraint-aware prompt (from cache, same questions):
- `"I really need to sit near [location] because..."` — matches turn 36.
- `"actually we resolved that, you can put us wherever"` — matches turn 72.
- `"oh I forgot, she also wants..."` — matches turn 78.

These cues have cos(q, cue) in the 0.3-0.5 range instead of 0.6-0.8, which is why they reach disjoint regions of the embedding space.

## Proposed fix (one concrete change)

Replace the single "generate 2 cues" LLM call in `MetaV2f`/`V15Control` with a **constraint-type enumeration cue generator** that emits cues in conversational register, keyed by constraint type. Concretely, change `V2F_PROMPT` to:

```text
You are generating search cues for semantic retrieval over a casual chat
conversation. The target conversation is planning/scheduling/constraint
discussion, and the answer is scattered across 10-20 turns in informal
chat register.

Question: {question}
{context_section}

Emit cues that mimic how participants ACTUALLY write chat messages
about each constraint type. Use one cue per type below, filled with
concrete details from the retrieved context when possible:

1. [ARRIVAL]        "I just got a message from <person>, she says <constraint>"
2. [PREFERENCE]     "he said he needs to be near <location> because <reason>"
3. [CONFLICT]       "<A> absolutely cannot <verb> <B>, they had that argument..."
4. [UPDATE]         "actually, they rescheduled — <old> is off, now it's <new>"
5. [RESOLUTION]     "we cleared that up, the <old_constraint> no longer applies"
6. [AFTERTHOUGHT]   "oh wait, one more thing — <person> also wants <detail>"
7. [PHYSICAL]       "the <object> is between <location A> and <location B>"

Rules:
- Write as if typing in the actual chat (casual, with deictic pronouns).
- Do NOT summarize what the final arrangement is. Do NOT enumerate
  items answer-style. Write ONE plausible chat utterance per type.
- Skip a type if you are confident it cannot apply to this question.

Format:
ASSESSMENT: <1-2 sentences>
CUE: [ARRIVAL] <text>
CUE: [PREFERENCE] <text>
... etc.
```

Why this specific change:
- **Constraint-type scaffolding** breaks the "paraphrase the question" attractor — the LLM is forced to write vocabulary from a different register (chat, not answer-document).
- **Named prefixes** like `[UPDATE]` and `[RESOLUTION]` push the model to emit the exact informal-update phrases (*"they rescheduled"*, *"we cleared that up"*) that match the deictic/update turns cosine cannot reach. From the data: turn 52 (*"clients rescheduled, they want to come Thursday now"*) and turn 72 (*"we actually cleared the air"*) have cosine to question of 0.19-0.41 and were missed by every method except the iterative constraint approach.
- **One cue per type** naturally produces 5-7 cues instead of 2, which compensates for the small per-cue top-10 budget — the current 2-cue limit is itself a contributor to the loss (both v2f cues frequently retrieve the same noise turns).
- Keeps the single LLM-call structure of v15/v2f (no iteration, no reranker), so it's a drop-in replacement at the same cost envelope.

Expected impact on LC r@20: based on the 100% r@all performance of the more aggressive iterative variant which uses the same scaffolding (and the 76.8% r@all of the single-call `constraint_aware` variant with fewer cues), this should recover at least ~0.55-0.70 LC r@20 — an improvement of +0.30 to +0.45 over current v2f.

Risk for other categories: the prefix scaffolding is domain-neutral (arrival, preference, update, resolution) and would not distort sequential_chain or state_change questions, where these types also apply. For categories like contradiction/absence_inference (already doing well) the "skip a type" instruction prevents forcing irrelevant cues. A sanity A/B on `locomo_30q` is warranted before rolling it out.

## Files

- Raw analysis (cue text, missed turn text, all cosine metrics):
  `/Users/eyu/edwinyyyu/mmcc/extra_memory/evaluation/associative_recall/results/lc_failure_analysis_raw.json`
- Prior investigation confirming iterative = 100% r@all:
  `/Users/eyu/edwinyyyu/mmcc/extra_memory/evaluation/associative_recall/results/constraint_investigation.json`
- Existing implementations to reference for the fix:
  - `CONSTRAINT_AWARE_PROMPT` in `/Users/eyu/edwinyyyu/mmcc/extra_memory/evaluation/associative_recall/constraint_retrieval.py` (lines 466-496)
  - `V2F_PROMPT` in `/Users/eyu/edwinyyyu/mmcc/extra_memory/evaluation/associative_recall/best_shot.py` (lines 171-196) — replacement target
