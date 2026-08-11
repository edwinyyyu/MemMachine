---
name: episodic-recall
description: Procedures for retrieving from the claude-memory episodic store via the memory_search, memory_expand and memory_outline MCP tools. Use when recalling past sessions or decisions, when the user references earlier work or asks "do you remember", before re-deriving something likely discussed before, before restating any fact or number that came out of memory, or when a memory_search returned poor results.
---

# Episodic recall (claude-memory)

The store holds every turn of every past session, reached through five tools:

| tool | answers |
|---|---|
| `memory_search(cue, …)` | *which* memory — associative recall over messages |
| `memory_expand(id, …)` | *what surrounded it* — the timeline around one memory |
| `memory_outline(id, …)` | *what shape a conversation had* — its user turns, in order |
| `memory_annotate(id, note)` | records what was later learned about a memory |
| `memory_demote(id, cue)` | stops a memory answering a cue it misleads on |

**One kind of address.** Every id is a `mem:` handle naming a segment — abbreviated to a
prefix long enough to be unambiguous, with whole uuids still accepted and an ambiguous prefix
answering with candidates rather than guessing. A handle also names its own *conversation*, so
`memory_outline(<any handle from it>)` outlines that conversation and `memory_search(within=…)`
confines a search to it; the session roster prints each conversation's FIRST handle, which is
stable as it grows. Every tool takes the same `id`, and there is no separate session id to pass
anywhere.

## What a hit is

- `mem:<id>` is a **segment**: a chunk of a message, not a message. One message holds one or
  more segments. Most are a single segment; a long message can run to dozens, so a hit may be a
  small fraction of what was said there.
- Chunk boundaries fall at sentence ends and carry no truncation marker, so a partial hit reads
  as a complete thought.
- `memory_expand(id, before, after, unit)` returns only that memory's own session; it cannot
  reach another conversation. `unit="segments"` (default) counts chunks — a flat budget, and
  the way to read *inside* one long event. `unit="events"` counts whole turns, for when the
  length of what is in the way should not decide how far the window reaches.
- Ingestion is near-real-time. Same-day turns, including the current session's, are already
  searchable — anything stated aloud is written to memory.
- **Search and expand see different things.** Only messages are embedded, so a search can never
  return a tool call, its output, or injected text. Expansion walks the stored timeline and
  reaches all of it. "Not in search" therefore means "not embedded", never "not recorded" — if
  a search cannot find how something was done, expand around a message near it.

## Choosing what a window spends itself on

`kinds` selects what to show **around** the seed — an allowlist by default, a blocklist with
`blocklist=True`. It never hides the memory you named: that one is always shown, marked
`← expanded from here` so you can see which turn you anchored on.

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

**A thin window is a reason to name kinds or switch unit, not to raise `before`/`after`.** A
window short of what was asked for has usually been eaten by one long event — a single tool
result can run to a thousand segments. Widening spends more budget on the same blob; excluding
its kind, or counting in events, buys turns.

**Events too long to show whole** — over ~4,000 characters, a pasted document rather than a
written message — appear as their first and last segments with a marker between them:

```
tool: "{\"type\": \"image\", \"source\": {\"type\": \"base64\", \"data\":"
      [564,992 more characters — memory_expand from mem:440e3a mem:e6d5c4]
      "\"image/png\"}}"
```

Nothing is silently truncated: the count is measured, and the handles are seeds. To read
inward, expand from one of them with `unit="segments"`. Often the two ends are enough to see
that reading further is pointless, as above.

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

Searching and surfacing **to the user** are separate decisions with opposite economics.

**Search on plausible bearing.** A search costs one call, and its payoff cannot be judged
before reading the result. Search when the answer depends on decisions already taken, when a
request builds on earlier work, or when a recommendation would change given what was already
tried, rejected or measured.

**Surface to the user only what changes the answer** — the recommendation, the assumptions, or how the
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
- **Narrowing is named parameters**, each unrestricted when omitted, and they combine:
  - `kinds=["user_message"]` / `["assistant_message"]` — a ranking aid, not a guarantee. When a
    user pastes assistant output back as input the text exists under both roles; doubly-escaped
    content marks a quotation inside another event.
  - `within=<mem:id>` confines the search to the conversation that handle belongs to — any
    handle from it will do. Use it to search inside one conversation instead of outlining it
    and reading turn by turn; use `memory_outline` when the question is *where* rather than
    *what*.
  - `since=` / `before=` bound the time range, half-open: `since <= when < before`, so one day
    is `since="2026-08-08", before="2026-08-09"`. ISO 8601; no zone means your local one.
  - `limit=` caps how many memories come back (default 8).

## Multi-hop

1. Search with an event-description cue, narrowed by `kinds`, `within` or `since`/`before`
   where the producer, conversation or date are known.
2. Expand the best hit. If its content will be restated, expand to the sandwich, forward first.
3. Hop with the continuation handles each expand returns.
   `memory_outline(id)` is the cheaper move when the question is *where* in a conversation
   rather than *what*: one line per user turn, with how many events followed each — a turn
   followed by sixty is where the work happened. `before`/`after` count turns, as in expand.
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
