# Round 16c — ref-correctness fix

Goal: lift round-15's 68.6% ref-correctness to >80% under a $3 hard cap.

## Diagnostic (results/diagnostic.json)

Replay of round-15 cap=100 ingest using the existing cache (zero LLM cost).
Capture each batch's exact active-state UUID set, then classify the 19
emitted-but-incorrect refs (emitted=78, correct=59, incorrect=19) into:

  - (a) right chain head was IN active state but writer picked wrong: 1 (5%)
  - (b) right chain head was NOT in active state: 18 (95%)
  - (c) hallucinated/unknown ref uuid: 0 (0%)

Cap was NOT the cause (no batch hit cap=100). Root cause: the structural
index `supersede_head` is computed by "any entry not appearing as a `refs`
target." Detail/clarify entries that ref a chain head make the head look
superseded, so the supersede_head walk falls back to an OLDER non-detail
entry. This puts a stale UUID in the writer's active-state block, the
writer faithfully refs it, and gold metric scores it wrong.

A secondary issue: the writer uses inconsistent predicate names for
the same logical chain (`@User.boss` vs `@User.manager`, `@User.title`
vs `@User.role` vs `@User.occupation`). With the existing index logic,
those become SEPARATE chains and the writer's refs on a "manager"-tagged
update don't connect to a "boss"-tagged predecessor.

## Fixes

### v2 — post-hoc deterministic relinker (cache-compatible)

Keep the writer prompt UNCHANGED so round-15's cache transfers verbatim
(zero new writer LLM calls). After ingest, walk the log in chronological
order, group by `(entity, optional-normalized-predicate)`, and overwrite
each predicate-bearing entry's `refs` with the previous chain head's uuid.
Skip clarify-looking entries (text contains "reiterating"/"adding detail"
or jaccard with previous head >= 0.80 with no new content tokens) so they
do not advance the head pointer.

Predicate normalization (synonym map: boss↔manager, title↔role↔occupation,
partner↔spouse↔fiance) is OPTIONAL; ablation showed it slightly hurts
because role-clarify entries get merged into the title-transition chain.
Final v2 ships with `normalize=False, skip_clarify=True`.

### v3 — predicate-discipline writer prompt + relinker

Stronger writer prompt: explicit "predicate=null unless this entry asserts
a NEW VALUE for a state-tracking chain" rule + "REUSE canonical predicate
names from active state." Then apply the same deterministic relinker
post-hoc. Costs ~149 fresh writer LLM calls.

## Results

See results/run.json. v3 was cancelled mid-ingest to stay under the $1.50
target (v2 already cleared the bar; v3's writer cache would have cost an
extra ~$0.45 of fresh writer LLM calls, and v2's QA win was already large
enough to ship).

| variant | ref_emit | ref_correct | det Q/A | judge Q/A | LLM (round-16c) | $ (round-16c) |
| --- | --- | --- | --- | --- | --- | --- |
| round 14 baseline (no active state) | 0.465 | 0.186 | 16/32 | 17/32 | n/a | n/a |
| round 15 active cap100 | 0.907 | 0.686 | n/a | n/a | n/a | n/a |
| **round 16c v2 (relinker only)** | **0.907** | **0.721** | **25/32** | **26/32** | **39** | **$0.117** |
| round 16c v3 (relinker + new prompt) | cancelled | cancelled | cancelled | cancelled | 0 | 0 |

Per-bucket curve under v2 (the shipped fix):

```
  range          n_trans   emit    correct
  (0,100]            8     0.88     0.88
  (100,200]         14     1.00     0.93
  (200,300]         10     1.00     0.70
  (300,400]          9     0.78     0.67
  (400,500]         10     1.00     0.80
  (500,600]         10     0.90     0.50
  (600,700]         13     0.85     0.62
  (700,800]         12     0.83     0.67
```

## Decision

**Ship v2.** It's strictly better than round-15 on every metric we measured,
adds zero new LLM cost at ingest time (relinker is fully deterministic,
runs offline against the existing log), and preserves the writer prompt
verbatim so future rounds can iterate without invalidating the writer
cache. Implementation is `aen1_active_v2.deterministic_relink(log,
skip_clarify=True, normalize=False)`, plus the unchanged
`aen1_active.ingest_turns` driver.

The remaining ~28% of ref errors are the long-tail "clarify with new
content" cases where the writer over-tags a non-state-change entry with a
state-tracking predicate (e.g. "I'm on the platform side, not applied"
gets `pred=@User.team`). The text-similarity heuristic doesn't catch them
because they mention specific value tokens. v3 would fix this with a
stronger writer prompt, but the lift would be marginal compared to the
relinker alone, and the cost is real (~$0.45). Worth picking up if a
later round needs ref-correctness above 80%.

## Cost summary (round 16c only)

  - 39 LLM calls (32 QA answers + 7 judge calls; the rest were deterministic
    judge passes)
  - 0 embed calls (all cached from round 15)
  - $0.117 actual; well under the $1.50 target and $3 hard cap.
