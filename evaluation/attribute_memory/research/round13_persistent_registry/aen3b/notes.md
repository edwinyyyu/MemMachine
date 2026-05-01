# aen3b — round-13 follow-up: descriptor recovery on LRU stress

## Problem (round 13 baseline)

Round 13's `aen3_persistent` was designed to fix round 12's bug where evicting an
entity from the LRU caused later descriptor mentions to create a fresh duplicate.
The fix: split persistence (full registry, never forgets) from LRU (active
context window) and add an embedding-search fallback when descriptor lookups
miss the alias index.

The implementation ran 59 embedding searches on S3 LRU stress, but accuracy
came in at 74.29 % — identical to aen2 — and descriptor accuracy was 0/8.

## Root cause(s)

Investigation of the round-13 result file showed two bugs, only one of which
was the LLM prompt the original task description called out.

### Bug A (the actually load-bearing one): grader surface-form mismatch

The grader keys decisions by exact-match surface. Ground-truth surfaces in
`coref_stress.scenario_s3` are lowercase (`"the dog walker"`,
`"the accountant"`). The coref LLM emits surfaces with sentence casing
(`"The dog walker"`, `"The accountant"`). 7 of the 8 descriptor failures were
case-only mismatches; 1 was a span mismatch
(GT `"the barista"` vs coref `"The barista at the cafe by my apartment"`).
For all 8, the coref decision **already pointed at the correct entity id**.

This is also why the failure file shows `arch_id: null, mapped_gt: null` for
every descriptor failure: the grader couldn't find any decision under the GT
surface, so it never even consulted the (correct) resolution.

aen3b's `run_s3.py` adds a normalised-surface index in `coref_log_decisions`
(lowercase + collapse whitespace; also indexes 2-to-6 token noun-phrase
prefixes of the coref surface, with and without leading article). This change
alone, applied to aen3's existing decisions, lifts aen3 from 74 % → 94.29 %
(descriptor 0 % → 87.5 %).

### Bug B (the prompt the task description called out): DESCRIPTOR_PICK_PROMPT

The old prompt insisted on literal feature alignment and warned against
"topical overlap." That's the right rule for a NAMED mention coming through
embedding search ("Quinn" should not match Carla just because both are
recruiters), but is too strict for genuine descriptor mentions ("the dog
walker" should match the entity aliased "dog walker"). It also doesn't
matter on S3 in practice — by turn 35-47, every descriptor target is already
back in the LRU due to repeated interleaved cache touches in this scenario,
so descriptor mentions resolve via single-candidate alias-resolve, not
embedding pick. So Bug B is real but its blast radius is mostly S6 / scenarios
with longer LRU eviction.

aen3b softens the prompt with calibration examples (a positive
recruiter-at-Anthropic case, a negative recruiter-vs-boss case, and a
dog-walker case) and a default-MATCH lean.

### Fix 2: accumulating descriptions

aen3 only updated an entity's description when the per-turn coref LLM
explicitly emitted a `description` field — which it usually does at intro
time and rarely afterward. After the intro snippet, the description blob
stayed stale, so embedding search never improved with use.

aen3b appends a turn-text snippet (cap 240 chars per snippet, 600 chars total,
6 most-recent retained) to every non-User entity touch. The cheap
concat option in the spec — no per-turn LLM call. Description-embed text is
rebuilt from `aliases :: snippet1 :: snippet2 ...`, so the embedding picks up
later wording variants automatically.

### Fix 3 (defensive): named-vs-descriptor split in the pick prompt

After Fix 1, the softened prompt over-matched **named** mentions: "Nora"
(introduced fresh at turn 16) was picked as Elena/OFFSITE_FACILITATOR because
both are described as event-hostish. "Quinn" (turn 20) was picked as Carla.

Fix 3 plumbs `mention_kind` into `descriptor_pick`. Named mentions get a
strict rule ("only MATCH if the proposed name is actually in the candidate's
aliases"; default to create_new); descriptors get the soft rule. There's also
a defensive Python-side check: if the LLM picks a candidate for a named
lookup but the proposed name doesn't substring-match any of the candidate's
aliases, the architecture rejects the pick and creates a fresh entity.

## Results

| arch       | total acc | named  | descriptor | embed searches | LLM/turn | embed/turn |
|------------|-----------|--------|------------|----------------|----------|------------|
| baseline   | 97.14 %   | 100 %  | 100 %      | -              | 0        | 0          |
| aen3       | 74.29 %   | 100 %  | 0 %        | 59             | 1.6      | 1.3        |
| aen3 (regraded with normalised surface) | 94.29 % | 100 % | 87.5 % | 59 | -    | -          |
| aen3b      | 97.14 %   | 100 %  | 100 %      | 60             | 2.15     | 1.90       |

aen3b matches the baseline. The single remaining failure is the same one the
baseline has — turn 1 pronoun "I" expected to map to USER but coref returns
surface "I've" (verb-contraction), which is a separate mention-extraction
issue, not a descriptor recovery one.

## Cost

- aen3b S3 incremental cost over warm aen3 cache: ~$0.293 (97 LLM + 95
  embed calls). All Coref / alias-disambig prompts hit the warm cache; only
  the new descriptor-pick wording and the new fuller-snippet embedding-text
  miss.
- Independent of the LLM/embed cost, the cheap-concat description
  accumulator runs at zero LLM cost — no per-turn description-update LLM
  call needed.

## Recommendation

Ship aen3b's pattern as the canonical descriptor-recovery architecture:

1. Fix the grading bug (normalised-surface lookup) — it accounts for ~20 of
   the 23 missing accuracy points.
2. Keep the prompt softening behind a `mention_kind` switch — descriptors
   default to MATCH on the highest-similarity-with-consistent-role candidate;
   named mentions stay strict.
3. Always accumulate per-turn snippets into the description blob; rebuild
   the embedding text from the most-recent N snippets.
4. Defensive Python-side alias check on named-mention picks is cheap
   insurance.

The diagnosis in the original task description was partially correct (prompt
was indeed too strict) but understated the load-bearing issue (the 0 %
descriptor accuracy on S3 was mostly a grader artifact, not a
resolution-quality artifact). Both fixes were needed for end-to-end recall:
without (1) the grader miscredits aen3b too; without (2) and (3),
embedding-search-driven mentions in S6-style long-eviction scenarios will
either reject the right candidate (aen3) or pick the wrong same-role entity
(soft-prompt without name-strictness).
