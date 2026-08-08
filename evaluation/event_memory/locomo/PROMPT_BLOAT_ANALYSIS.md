# Segmenter prompt bloat analysis (PROMPT_TERSE_DECOUPLED_V2)

The prompt is ~75 lines / ~700 words after many iterations. Redundancy
and over-specification found:

## Redundancy — the same rule stated in multiple places

1. **The "specific particular" list is stated THREE times.**
   - Intro KEEP rule: "names, places, dates, numbers, decisions, plans,
     preferences, opinions, described events, attached media"
   - (A) memory: "names, numbers, identifiers, distinctive phrasing"
   - (B) terse: "every name, place, date, number, identifier, decision,
     plan, preference, opinion, quoted phrase, attached-media detail"
   → Define a SPECIFIC PARTICULAR once; reference it everywhere else.

2. **"Preserve every concrete particular verbatim"** appears in both
   (A) memory and (B) terse. Both fields must keep all particulars —
   state it once as a shared principle.

3. **The retrieval objective** is framed three times: the intro
   ("a future user querying any specific content should find..."), the
   KEEP rule, and the (C) queries rule. One statement suffices.

## Over-specification — enumerations that a principle covers

4. **DATES section is ~19 lines.** Two enumerations bloat it:
   - 10 relative-reference examples (yesterday/last week/.../just) —
     a 3-4 example sample + "every relative time reference" covers it.
   - 4 "Forbidden date forms" (sentence-prefix / parenthetical /
     square-bracket / as-of) — one principle ("never as a bracketed,
     parenthetical, or sentence-prefix tag") covers all four.
   Load-bearing behaviour to KEEP: don't repeat the message date;
   resolve relative→absolute; replace the relative phrase; weave in
   prose (on D / in M / in Y); omit if == message date.

5. **"FAILURE" appears 7 times.** It is an attention-forcing device;
   7 occurrences dilute it. Keep it on the 2-3 highest-stakes rules.

6. **(B) terse re-derives its whole spec** instead of "field (A) with
   the fewest words." terse = memory minus filler; it needs one
   sentence, not a paragraph.

## Result
Redundancy-removed slim keeps every load-bearing rule, ~33 lines
(≈55% shorter). Whether fields (B) terse / (C) queries survive at all
is decided by the decoupling ablation, not by this analysis.
