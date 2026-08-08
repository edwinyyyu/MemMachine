---
name: project-rewind-handoff-20260522
description: "LoCoMo: rewind investigation of unreproducible boverb=91.28 result. Session handoff written 2026-05-22 ~21:30. Read FIRST on rewound sessions to recover state."
metadata:
  node_type: memory
  type: project
  originSessionId: dd9b889d-0eda-4b43-8516-60e10b3665eb
---

# LoCoMo rewind investigation — HANDOFF

Written 2026-05-22 ~21:30 before potential rewind. This memory is in
the rewind-immune `~/.claude/.../memory/` dir. A mirror copy lives at
`evaluation/event_memory/locomo/REWIND-HANDOFF.md` (rewind-vulnerable
inside project). If they disagree, this one is authoritative.

## 0. WHY (the investigation)

Standing goal: Pareto-beat Mem0 on LoCoMo at <350 tok/q. The leading
shippable today is **A — slim_v3 + BM25-only date aliases + RESOLVE
prompt** at n=6 mem0-bench = **91.45 mini / 89.39 gpt-5** (n=6, sd
0.39 / 0.17, mean ~5210 segments).

Mid-session the user asked to try a "verbatim dates" simplification.
That produced ingest case `terse-decoupled-slim-v3-bo-verbatim` and a
test (`bww6uf6gh`) at 16:28 May 22, **n=3 = 91.28 mini / 89.65 gpt-5
@ 5381 seg** — close enough to A to look like a clean win at lower
prompt complexity. Call this **boverb**.

Later I added `terse-decoupled-slim-v3-bo-verbatim-ve` (regex code path
removed entirely) and ingested it at 17:17, n=3 = **90.28 mini / 88.61
gpt-5 @ 5456 seg**. Call this **boverbve**.

The 1pp accuracy gap and 75-seg systematic gap between boverb and
boverbve **cannot be noise** (boverb sd 6, boverbve sd 11; gap 75 ≈
7σ). Something changed in the probe between the two ingests.

Probe file (`evaluation/event_memory/longmemeval/llm_pipeline_probe/
probe_terse_decoupled_slim_v3.py`) was last modified at **16:32 May 22**
— between boverb (16:28) and boverbve (17:17). That edit shifted
verbatim-mode segment counts.

The current code state **cannot reproduce boverb's 91.28**. Re-running
the current `bo-verbatim` case would give boverbve-like 90.28. The
question: was the boverb-time prompt actually better, or just an
intermediate artifact?

## 1. CONFIRMED NUMBERS (all HANDOFF iteration config:
   gpt-5-mini answerer + judge, mem0-bench, K=10, BM25 add 0.5,
   vec-search-limit 28, no-reranker, ts-short, nb8 both, 54n@low)

| variant                            | n | seg/rep mean (range)     | mini-bench     | gpt-5-bench    |
|------------------------------------|---|--------------------------|----------------|----------------|
| A = tslimv3bo (RESOLVE)            | 6 | 5210 (range 64)          | **91.45** sd.39 | 89.39 sd.17    |
| A verify (current code, RESOLVE)   | 1 | 5270                     | 90.97           | (not run)      |
| boverb (mystery 16:28 prompt)      | 3 | 5381 (range 12)          | 91.28           | 89.65          |
| bo-natural (RESOLVE_NATURAL hybrid)| 1 | 5346                     | 90.71           | (not run)      |
| boverbve (VERBATIM + verbose-event)| 3 | 5456 (range 34)          | 90.28           | 88.61          |
| boverbve same — mem0-CLASSIC judge | 6 | 5456 (range 34)          | 83.54 sd.51    | 81.16 sd.53    |
| raw-seg + whole-deriver + emb-only K=10, mini-CLASSIC | 1 | n/a    | 77.27           | n/a            |

n=3 expansion **IN FLIGHT now** as task `bovpdskxg`:
  rep2+rep3 of A verify (tslimv3datealias-bm25only) + bo-natural.
  Started ~21:13. Should land around 21:30. Will tell us:
  - A verify n=3 — confirms whether current RESOLVE matches A n=6 (91.45)
  - bo-natural n=3 — confirms whether RESOLVE_NATURAL ties A or loses to it

After it lands, decision matrix:
  - bo-natural ≈ A within noise → ship bo-natural (simpler prompt, source-faithful, no ISO bias). NO rewind needed.
  - bo-natural < A by >0.5pp → bo-natural is a real regression. Then either ship A as-is (already validated n=6), OR pursue the boverb rewind to try recovering 91.28.

## 2. REWIND PLAN (only if bo-natural fails and you want boverb back)

The user's checkpoint list (oldest first):
  1. "don't need to add long variants…" — 3 files +155 — added _DATE_BLOCK_RESOLVE/VERBATIM constants + bo-verbatim case + ran boverb test (this is where boverb's prompt was set)
  2. "it's close enough to try block.text = raw span anyway" — no code
  3. "to be clear, you only replaced the raw event part of the embedding with the raw span?" — run_rawspan_phase2.py +104
  4. "span-last means generating span last right? no reordering of embedding components?" — 2 files +121
  5. "but verbose alias generator only works on iso dates?" — no code
  6. "and yes, check how useful the regex is" — 3 files +132 — likely the 16:32 probe edit; added verbose-event/cldr alias modes + probably tweaked verbatim handling
  7. "B if possible -- get rid of regex…" — 2 files +125 — added bo-verbatim-ve ship-candidate case

Best first rewind: **checkpoint 5** ("but verbose alias generator only works on iso dates?"). State preserved: bo-verbatim case + rawspan probe. State lost: alias modes from 6, bo-verbatim-ve from 7, bo-natural added this session (post-7).

**Validation protocol after each rewind step:**
  1. Re-ingest 1 rep of bo-verbatim with current code state:
     ```bash
     cd evaluation/event_memory/locomo
     rm -f locomo-tslimv3boverb-CHECK.sqlite* locomo-tslimv3boverb-CHECK.vec.sqlite*
     uv run python locomo_ingest.py \
       --data-path ../../data/locomo10.json \
       --segment-db locomo-tslimv3boverb-CHECK.sqlite \
       --vector-db locomo-tslimv3boverb-CHECK.vec.sqlite \
       --segmenter terse-decoupled-slim-v3-bo-verbatim \
       --segmenter-model gpt-5.4-nano --segmenter-reasoning low \
       --neighbor-window 8 --neighbor-direction both
     uv run python -c "import sqlite3; print(sqlite3.connect('locomo-tslimv3boverb-CHECK.sqlite').execute('SELECT COUNT(*) FROM segment_store_sg').fetchone()[0])"
     ```
  2. Read the seg count:
     - **≈5381 (within ±15)** → FOUND boverb's prompt! Run search + eval to confirm ~91.28 accuracy, then ship.
     - **≈5456 (within ±20)** → still current-code's verbatim. Rewind further (checkpoint 4 → 3 → 2 → 1).
     - **≈5270 (within ±20)** → looks like RESOLVE (not verbatim). bo-verbatim case at this checkpoint doesn't pass date_handling="verbatim". Not boverb's state.
     - Other → unknown intermediate; investigate by inspecting the bo-verbatim case args and the verbatim prompt text.
  3. If rewind 5 doesn't recover, try rewinds 4, 3, 2, 1 in order. If none give 5381 segments, the boverb prompt is unreproducible from any listed checkpoint and the trail is dead.

## 3. CODE TO RE-ADD AFTER REWIND (because bo-natural was added post-checkpoint-7)

Two edits if you want bo-natural back:

### Edit 1 — `evaluation/event_memory/longmemeval/llm_pipeline_probe/probe_terse_decoupled_slim_v3.py`

Add after `_DATE_BLOCK_VERBATIM` definition (~line 119):

```python
# Hybrid: resolve the relative phrase to an absolute date (same as RESOLVE)
# but match the source's register on output format instead of prescribing
# ISO. Chat / prose -> "March 15, 2024", "March 2024", "2024". ISO source
# ("2024-03-15") -> stay ISO. Precision stays at what the speaker stated.
_DATE_BLOCK_RESOLVE_NATURAL = """Dates in the statement: the message's own date ({date}) is attached automatically when this memory is surfaced, so the statement text must never contain {date}. Resolve every relative time reference -- "yesterday", "last week", "three years ago", "next Friday", "the weekend", "today", "recently", "now", "just" -- to an absolute date anchored at {date}.
  - If the resolved date EQUALS {date}, the statement carries no date and no relative phrase.
  - If it DIFFERS from {date}, delete the relative phrase and weave the absolute date into the prose. Match the source's register: use ISO-like dates ("2024-03-15") only if the source itself uses ISO; for chat or prose, use natural language ("on March 15, 2024", "in March 2024", "in 2024"). Match the precision the speaker stated (don't invent a day if they only said a month). Never leave a relative phrase beside the resolved date, and never write a date as a bracketed, parenthetical, or sentence-prefixed tag."""

PROMPT_SLIM_V3_NATURAL_DATES = PROMPT_SLIM_V3.replace(
    _DATE_BLOCK_RESOLVE, _DATE_BLOCK_RESOLVE_NATURAL
)
assert _DATE_BLOCK_RESOLVE_NATURAL in PROMPT_SLIM_V3_NATURAL_DATES, (
    "Date-block swap failed: _DATE_BLOCK_RESOLVE not found in PROMPT_SLIM_V3"
)
```

And update the constructor's date_handling validation (around line 300):

```python
if date_handling not in ("resolve", "verbatim", "natural"):
    raise ValueError(
        f"date_handling={date_handling!r} not in ('resolve', 'verbatim', 'natural')"
    )
# If the caller didn't override prompt_template, dispatch on
# date_handling. Explicit prompt_template wins.
if prompt_template is PROMPT_SLIM_V3 and date_handling == "verbatim":
    prompt_template = PROMPT_SLIM_V3_VERBATIM_DATES
elif prompt_template is PROMPT_SLIM_V3 and date_handling == "natural":
    prompt_template = PROMPT_SLIM_V3_NATURAL_DATES
```

### Edit 2 — `evaluation/event_memory/locomo/locomo_ingest.py`

Add the bo-natural ingest case (insert after the bo-verbatim-ve case if present, or after bo-verbatim if checkpoint 7 was rewound away):

```python
case "terse-decoupled-slim-v3-bo-natural":
    # Hybrid: resolve relative dates to absolute (like RESOLVE) but
    # match the source's register on format (chat -> natural, ISO
    # source -> ISO). BM25-only date aliases.
    from probe_terse_decoupled_slim_v3 import (
        TerseDecoupledSegmenter as TerseDecoupledSegmenterBONatural,
    )

    lm = OpenAIResponsesLanguageModel(
        OpenAIResponsesLanguageModelParams(
            client=openai_client,
            model=args.segmenter_model,
            reasoning_effort=args.segmenter_reasoning,
        )
    )
    return TerseDecoupledSegmenterBONatural(
        language_model=lm,
        date_aliases_in_embed=False,
        date_aliases_in_bm25=True,
        date_handling="natural",
    )
```

And add `"terse-decoupled-slim-v3-bo-natural",` to the `--segmenter`
choices list (around line 1779).

## 4. PRESERVED DATA THAT WON'T BE LOST ON REWIND

These are SQLite DB files in `evaluation/event_memory/locomo/` (rewind
doesn't touch ingest outputs):
- boverb DBs (3 reps): `locomo-tslimv3boverb-54n-l-nb8-rep{1,2,3}.sqlite[.vec.sqlite]` — the MYSTERY 16:28 state. Preserve these — if you ship boverb, search + eval can re-run from these without re-ingesting.
- boverbve DBs (6 reps): `locomo-tslimv3boverbve-54n-l-nb8-rep{1..6}.sqlite[.vec.sqlite]`
- A DBs (6 reps): `locomo-tslimv3bo-54n-l-nb8-rep{1..6}.sqlite[.vec.sqlite]`
- bo-natural DBs: `locomo-tslimv3bonatural-54n-l-nb8-rep{1,2,3}.sqlite[.vec.sqlite]` (rep1 only, rep2-3 ingesting now)
- A verify rep1: `locomo-tslimv3datealias-bm25only-54n-l-nb8-verify.sqlite[.vec.sqlite]`
- raw-seg + whole-deriver: `locomo-text-whole.sqlite[.vec.sqlite]`

Search files: `search-*.json` for each.
Eval files: `eval-*-mb-c14.json` (mem0-bench) and `eval-*-mc-c14.json`
(mem0-classic).

## 5. METHODOLOGY (always use these for iteration; matches HANDOFF.md)

- Segmenter: gpt-5.4-nano @ low reasoning
- Neighbor window: 8 both
- Search: vec-limit 28, expand 0, K=10, no-reranker, BM25 additive 0.5, ts-short
- Answerer: gpt-5-mini (search.py default)
- Iteration judge: gpt-5-mini + **mem0-bench** + skip-category-5
- Mem0-comparable judge: gpt-5 + **mem0-classic** (corrected this session — bench is the looser newer rubric, classic is what Mem0's published 87.3% used)
- Default eval scope: 1 rep × 1 judge unless explicitly asked for more (corrected feedback this session — don't auto-batch n=6×2)

## 6. USER PROMPTS VERBATIM — THIS SESSION

Captured so on rewind the intent is recoverable. Newest at bottom.

1. (loop fire) "/loop Do your best to do better than the v22 baseline. Pareto improvement on LoCoMo at low tokens vs Mem0. Diagnose failures and successes; generalizable changes; validate with evidence; methodology rules."
2. "what is the k = 10 score with raw text segments, whole deriver, embedding only?"
3. (after machine sleep) "computer went to sleep -- please resume"
4. "actually not mem0-bench -- use classic"
5. "wtf? where's the eval on k = 10 score with raw text segments, whole deriver, embedding only? with mem0 classic prompt, single run, gpt-5-mini?"
6. "not that many evals wth"
7. "it's not meant to be comparable"
8. "wtf? I want mem0-bench on the runs for the potential ship, and a single classic run with the particular config I just provided"
9. "wtf why 12 evals? I thought we already had several evals on option B"
10. "can we just match the n for A?"
11. (interrupted) "n = 3 x 2 first -- see if it's actually comparable"
12. "where did the initial B number 91.28 come from if we need to run 12 evals?"
13. "are prompts verbatim?"
14. (interrupted) "are prompts verbatim between B and B'?"
15. "how many of these share the same ingestion?"
16. "number of segments points to prompt diff"
17. "so which of these to rewind to?" [pasted: list of 7 checkpoints with edit deltas]
18. [pasted: long dump showing the original "don't need to add long variants" checkpoint that introduced the verbatim mechanism + ran the boverb test (bww6uf6gh)]
19. "not rewinding -- switch it to A and try a run to see if your guess is correct / explain the difference between RESOLVE and VERBATIM?"
20. "ok now make another version -- can we do something in between that and verbatim, where dates are resolved, but without the examples that push it to ISO format or natural language -- like a prompt that has the same benefits, but pushes to ISO format only if the original text looks like it would use iso, natural language if it's a chat, etc. ?"
21. "confirm now evals follow the typical methodology as in HANDOFF, not some weird one-off"
22. "n = 3"
23. "then, what's the best candidate for rewind, if needed?"
24. "save the problem and current status (progressm, ongoing tasks, etc.) to a handoff-like file so that when I rewind, you can understand the situation / I think it's better to rewind piece by piece until we hit the right one, so you should record all the relevant prompts verbatim in the handoff-like file (also so we don't lose our current stuff) right now, as well as between rewinds if needed"
25. "can I ensure that the file is immune to rewind?"

## 7. NEXT ACTION ON RE-ENTRY

If you've just been rewound: re-read this file. Look at the seg count
from the post-rewind validation re-ingest. Decide ship/rewind-further
per §2. If shipping, run search + mini-bench eval on the recovered
boverb-state DBs and report.

If you haven't been rewound and the n=3 batch `bovpdskxg` just landed:
report the bo-natural vs A comparison and let user decide ship vs
rewind. If bo-natural ties → ship bo-natural, no rewind. If
bo-natural loses meaningfully → present rewind plan from §2.

Update this file (and its mirror) between rewinds if the situation
shifts. Important: update the **memory copy** because that's the
rewind-immune one; the locomo/REWIND-HANDOFF.md mirror gets reverted
on each rewind.

Related: [[feedback-eval-stack-purpose]] (judge variant correction),
[[feedback-eval-scope-minimal]] (1×1 default for diagnostics).
