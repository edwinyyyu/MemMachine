---
name: episodic-recall
description: Procedures for retrieving from the claude-memory episodic store via the memory_search and memory_expand MCP tools. Use when recalling past sessions or decisions, when the user references earlier work or asks "do you remember", before re-deriving something likely discussed before, or when a memory_search returned poor results.
---

# Episodic recall (claude-memory)

Findings below were measured by controlled experiments (same known target, varied cue style, rank scored) on 2026-06-10.

## Cue construction

- **Event descriptions win.** "User instructed that all prompts in the claude-memory system should contain clean operating principles" → target at rank 1. The same target was *absent* from top-8 for the bare-keyword cue "clean principles". Describe who did what, about which system, with what intent.
- **Verbatim quotes work but bind to surface text.** A quoted line ranks the textually closest variant (often an earlier draft, not the operative version), and every incidental word in the quote becomes a retrieval term — "give a score" inside a quote dragged in unrelated scoring discussions. Trim quotes to the load-bearing clause.
- **Bare keywords retrieve lexical neighbors across all time.** Never sufficient for an episodic target.
- **Match the register of the producer.** Conceptual/technique cues retrieve assistant analyses; imperative cues retrieve user directives. Combine with a producer filter to sharpen.

## Search mechanics

- **Snippets are truncated.** Always `memory_expand` a hit before quoting or relying on it — expanded content can be several times longer than the search snippet.
- **There is no relevance floor.** Top-k always returns something, confidently formatted, even for cues about events that never happened. Stop rule: if results are off-register for the cue, conclude "no memory exists" — do not keep reformulating past 2 attempts.
- **Filters**: single-quoted strings; combine with AND/OR.
  - `m.producer = 'user'` / `'assistant'` — caveat: when the user pastes assistant output back as input (a frequent habit), the quoted text exists under both roles, and such memories can match *both* producer filters (observed: mem:60d5cd585… passes both, rendering double-escaped under the non-native filter). Treat producer filters as a ranking aid, not a guarantee; double-escaped content (`\\\"…\\\"`) marks a quotation inside another event.
  - `timestamp >= date('2026-06-09')` — date filters compose with producer filters.
- **Ingestion is near-real-time.** Same-day turns (including the current session's) are already searchable. Anything said aloud is encoded — stating an inference explicitly is writing it to memory.

## Multi-hop algorithm

1. Search with an event-description cue (+ filters if producer/date are known).
2. Expand the best hit with asymmetric `before`/`after` matched to whether the target precedes or follows it.
3. Hop using the continuation handles each expand returns (`memory_expand mem:<first> before=N` / `mem:<last> after=N`).
4. If the target won't surface directly, search its *surrounding context* (what was being discussed, roughly when) and expand from a neighbor.
5. Cap at ~2 cue reformulations; absence of on-register results means the memory probably doesn't exist.

## Duplicates

The user interrupts-and-resubmits to steer, so instructions appear as 2–3 near-identical memories ranked adjacently. The **latest timestamp is the operative version**. Check for `[note: superseded by …]` annotations before acting on an early draft.

## Curation verbs (use sparingly)

- **annotate**: attach supersession pointers and outcomes ("superseded by mem:X", "this approach was abandoned"). Notes render inline on every future retrieval. Append-only — write facts that stay true, not instructions.
- **demote**: only when a memory itself misled for a cue you will search again. Sharpen the cue first; understand why a memory exists before judging it junk; never demote to clear a path to a lower-ranked answer.
