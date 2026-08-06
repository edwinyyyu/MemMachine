---
name: episodic-recall
description: Procedures for retrieving from the claude-memory episodic store via the memory_search and memory_expand MCP tools. Use when recalling past sessions or decisions, when the user references earlier work or asks "do you remember", before re-deriving something likely discussed before, before restating any fact or number that came out of memory, or when a memory_search returned poor results.
---

# Episodic recall (claude-memory)

The store holds every turn of every past session, reached through `memory_search`,
`memory_expand`, `memory_annotate` and `memory_demote`.

## What a hit is

- `mem:<id>` is a **segment**: a chunk of a message, not a message. One message holds one or
  more segments. Most are a single segment; a long message can run to dozens, so a hit may be a
  small fraction of what was said there.
- Chunk boundaries fall at sentence ends and carry no truncation marker, so a partial hit reads
  as a complete thought.
- `memory_expand(seed, before, after)` counts **segments, not messages**, and returns only the
  seed's own session. It cannot reach another conversation.
- Ingestion is near-real-time. Same-day turns, including the current session's, are already
  searchable — anything stated aloud is written to memory.
- **Search and expand see different things.** Only messages are embedded, so a search can never
  return a tool call, its output, or injected text. Expansion walks the stored timeline and
  reaches all of it. "Not in search" therefore means "not embedded", never "not recorded" — if
  a search cannot find how something was done, expand around a message near it.

## Choosing what a window spends itself on

`memory_expand(seed, before, after, kinds, blocklist)` — `kinds` is an allowlist by default, a
blocklist with `blocklist=True`. It is applied while the window is gathered, so the budget buys
only what was asked for.

| goal | call |
|---|---|
| read the argument in a session full of long command output | `kinds=["tool_result"], blocklist=True` |
| replay just the procedure | `kinds=["tool_call","tool_result"]` |
| see exactly what the session was handed | `kinds=["injected"]` |
| block nothing at all | `kinds=[], blocklist=True` |

Kinds are `user_message`, `assistant_message`, `reasoning`, `tool_call`, `tool_result`,
`injected`. With no `kinds`, `injected` is blocked: hook context, skill bodies, system
reminders, slash-command echoes and the session's own compaction summary. That text is on the
timeline because it is genuinely what the session saw, but it arrives in runs, so a default
window landing in one would be entirely boilerplate. Naming kinds replaces the default outright.

**A thin window is a reason to name kinds, not to raise `before`/`after`.** A window that comes
back short of what was asked for has usually been eaten by one long event — a single tool result
can run to dozens of segments. Widening spends more budget on the same blob; excluding its kind
buys turns.

## Before relying on a hit

Expand until the window is **sandwiched**: a distinct timestamp/speaker header on each side of
the seed's block. A single undivided block means the window is still inside one message and
part of it is unread — raise `after` and repeat.

Expand asymmetrically, toward what is needed:

- **Backward** for framing — the task and the constraints in force. A figure measured against
  one target means something different against another.
- **Forward** for consequences — qualifications, pushback and retractions arrive in later
  turns. When checking a claim, `before=2, after=15` is more useful than a symmetric window.

A hit is a fragment of a conversation in motion: it may be a draft, a quotation of an outside
source, or a claim revised shortly afterwards. Formatting confers no authority — emphasis marks
the sentence someone cared about, which is as often the one later corrected. Before restating a
number, an attribution or a decision, establish three things:

1. **Whose claim it is.** Quotation markers survive into the chunk; check for them before
   writing "we measured".
2. **Whether it was revised.** Expand forward.
3. **Whether it was disputed elsewhere.** Expansion is session-scoped, so this needs a second
   search — cued on the *dispute*, not the claim. A cue carrying the claim's own wording returns
   the passages that elaborate it; "was X disputed, was that resolved" returns the ones that
   overturn it.

## Two thresholds

Searching and surfacing are separate decisions with opposite economics.

**Search on plausible bearing.** A search costs one call, and its payoff cannot be judged
before reading the result. Search when the answer depends on decisions already taken, when a
request builds on earlier work, or when a recommendation would change given what was already
tried, rejected or measured.

**Surface only what changes the answer** — the recommendation, the assumptions, or how the
question should be read. Citing a past conversation to demonstrate continuity adds nothing.
Searching and then not mentioning the result is a normal outcome.

**Ask only when the referent is ambiguous** — "continue the plan" where several plans are live.
Even then, prefer retrieving, naming the candidates, and proceeding on the most likely with the
assumption stated; a concrete list the user can correct beats a question they must answer.
Reserve a question for when proceeding on the wrong reading would be costly.

## Cue construction

- **Event descriptions win.** "User instructed that all prompts in the claude-memory system
  should contain clean operating principles" → target at rank 1; the bare-keyword cue "clean
  principles" missed it entirely. Describe who did what, about which system, with what intent.
- **Verbatim quotes bind to surface text.** A quoted line ranks the textually closest variant,
  often an earlier draft rather than the operative version, and every incidental word in the
  quote becomes a retrieval term. Trim quotes to the load-bearing clause.
- **Bare keywords** retrieve lexical neighbours across all time. Never sufficient alone.
- **Match the register of the producer.** Conceptual cues retrieve assistant analyses;
  imperative cues retrieve user directives.

## Search mechanics

- **There is no relevance floor.** Top-k always returns something, confidently formatted, even
  for cues about events that never happened. If results are off-register for the cue, conclude
  the memory does not exist rather than reformulating indefinitely.
- **Filters** take single-quoted strings and combine with AND/OR:
  - `m.producer = 'user'` / `'assistant'` — a ranking aid, not a guarantee. When a user pastes
    assistant output back as input, the text exists under both roles and can match either
    filter; doubly-escaped content marks a quotation inside another event.
  - `m.session_id = '<uuid>'` scopes a search to one conversation. The id must be **whole** —
    the grammar has equality but no prefix match, so a shortened id matches nothing and returns
    an empty result that reads like "no such memory" rather than a malformed filter.
  - `timestamp >= date('2026-06-09')` composes with the others.

## Multi-hop

1. Search with an event-description cue, plus filters where producer or date are known.
2. Expand the best hit. If its content will be restated, expand to the sandwich, forward first.
3. Hop with the continuation handles each expand returns.
4. If the target will not surface directly, search its *surrounding context* — what was being
   discussed, roughly when — and expand from a neighbour.
5. For a claim that will be acted on, add one search on its status.
6. Cap at about two cue reformulations.

## Duplicates

Interrupted and resubmitted instructions appear as two or three near-identical memories ranked
adjacently. The **latest timestamp is the operative version**. Check for `[note: …]`
annotations before acting on an early draft.

## Curation

- **annotate** attaches supersession pointers and outcomes ("superseded by mem:X", "this
  approach was abandoned"). Notes render inline on every future retrieval. Append-only, so
  write facts that stay true rather than instructions.
- **Expand to the sandwich before writing a note.** An annotation drawn from an unread fragment
  propagates to every future retrieval and cannot be edited or removed, only appended to.
- **demote** only when a memory itself misled for a cue that will be searched again. Sharpen
  the cue first, understand why the memory exists before judging it junk, and never demote to
  clear a path to a lower-ranked answer.
