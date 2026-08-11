---
name: episodic-recall
description: Procedures for retrieving from the claude-memory episodic store via the memory_search, memory_expand and memory_outline MCP tools. Use when recalling past sessions or decisions, when the user references earlier work or asks "do you remember", before re-deriving something likely discussed before, before restating any fact or number that came out of memory, or when a memory_search returned poor results.
---

# Episodic recall (claude-memory)

Every turn of every past session is stored and searchable, including the one happening now.

## What is stored

Four nested things. Most confusion with these tools comes from mistaking one for another:

| | |
|---|---|
| **conversation** | one continuous exchange between a user and the assistant. Compaction does not start a new one. Expansion and outline never cross from one into another |
| **turn** | one contribution to it — a user message, or one assistant reply together with the tool work inside it |
| **event** | one piece of a turn: the message text, a tool call, a tool result, or text the host injected. A user turn is usually one event; an assistant turn is often many |
| **segment** | a chunk of one event's text, at most 500 characters |

Every event has a **kind**, and the `kinds` argument selects on it:
`user_message`, `assistant_message`, `reasoning`, `tool_call`, `tool_result`, `injected`.

## How memories are addressed

A **handle** looks like `mem:a3f8c1`. It always names one **segment** — never a message, a turn
or a whole conversation. It is the shortest prefix of that segment's id that nothing else
answers to; the full id works too, and an ambiguous prefix returns candidates rather than
guessing.

One handle is every address you need, because it names two things at once:

- **its segment** — pass it to `memory_expand` to read around that exact point
- **its conversation** — pass the same value to `memory_outline`, or to
  `memory_search(within=…)`, to work inside the conversation it came from

No tool takes a session id. The list of recent conversations shown at session start gives each
one's first handle.

The handle passed to `memory_expand` is the **seed**, and what comes back is a **window**: that
segment plus a stretch of the conversation on either side of it.

## The tools

| tool | answers |
|---|---|
| `memory_search(cue, …)` | *which* memory — associative recall from a **cue**, the text it matches against ([how to write one](#writing-cues)) |
| `memory_expand(id, …)` | *what surrounded it* — the timeline around one segment |
| `memory_outline(id, …)` | *what shape a conversation had* — its user turns, in order |
| `memory_annotate(id, note)` | records what was later learned about a memory |
| `memory_demote(id, cue)` | stops a memory answering a cue it misleads on |

**Search and expand see different things.** Only messages are embedded, so a search can never
return a tool call, its output, or injected text. Expansion reads the stored timeline and
reaches all of it. "Not in search" therefore means "not embedded", never "not recorded" — when
a search cannot find how something was done, expand around a message near it.

**A search result is one segment, not the whole message.** A long message runs to dozens of
them, and nothing marks a result as partial. Assume there is more until an expansion shows
otherwise.

## Reading a window

**`unit` sets what `before`/`after` count.**

- `unit="segments"` (default) counts chunks — the only way to read *inside* one long event.
- `unit="events"` counts whole events, so one enormous event costs the same as one short one.

**`kinds` selects what is shown around the seed** — an allowlist by default, a blocklist with
`blocklist=True`. It never hides the seed itself: that one is always shown, marked
`← expanded from here`.

| goal | call |
|---|---|
| read the argument in a session full of long command output | `kinds=["tool_result"], blocklist=True` |
| replay just the procedure | `kinds=["tool_call","tool_result"]` |
| see exactly what the session was handed | `kinds=["injected"]` |
| exclude nothing at all | `kinds=[], blocklist=True` |

Naming any `kinds` replaces the default outright. The default excludes `injected` — hook
context, skill bodies, system reminders, slash-command echoes, compaction summaries — because
it arrives in long runs, and a window landing in one would be all boilerplate.

### When the window comes back short

Two causes, opposite remedies. The window itself shows which, so read it before changing any
argument:

- **Something bulky filled it** — an elision marker, or a wall of one kind; a single tool result
  can run to a thousand segments. Raising `before`/`after` buys more of the same material.
  Exclude that kind, or switch to `unit="events"`.
- **One message is longer than the window** — a single unbroken stretch of one speaker, no
  marker, no one else. Raise `after`. Nothing under ~4,000 characters is elided, so an ordinary
  long message can fill a segment window with no marker to signal it.

### Events too large to show whole

Over ~4,000 characters, an event is shown as its first and last segments with an **elision
marker** between them:

```
tool: "{\"type\": \"image\", \"source\": {\"type\": \"base64\", \"data\":"
      [564,992 more characters — memory_expand from mem:440e3a mem:e6d5c4]
      "\"image/png\"}}"
```

The handles in the marker are usable seeds: to read the middle, expand from one with
`unit="segments"`. Often the two ends are enough to show that reading further is pointless, as
above.

## Before relying on a result

Expand until the window is **sandwiched**: a timestamp/speaker header on *both* sides of the
seed. Anything less means part of that message is still unread — apply the short-window test
above.

Expand asymmetrically, toward what is needed:

- **Backward** for framing — the task and the constraints in force. A figure measured against
  one target means something different against another.
- **Forward** for consequences — qualifications, pushback and retractions arrive in later
  turns. When checking a claim, `before=2, after=15` beats a symmetric window.

A result is a fragment of a conversation in motion: it may be a draft, a quotation of an
outside source, or a claim revised a few turns later. Emphasis confers no authority — it marks
the sentence someone cared about, which is as often the one later corrected. Before restating a
number, an attribution or a decision, establish three things:

1. **Whose claim it is.** Quotation marks survive into the segment; check for them before
   writing "we measured".
2. **Whether it was revised.** Expand forward.
3. **Whether it was disputed elsewhere.** Expansion stays inside one conversation, so this
   needs a second search — cued on the *dispute*, not the claim. A cue carrying the claim's own
   wording returns the passages that elaborate it; "was X disputed, was that resolved" returns
   the ones that overturn it.

## When to search, and when to say so

**Search on plausible bearing.** Whether a search was worth it cannot be known before reading
the result. Search when the answer depends on decisions already taken, when a request builds on
earlier work, or when a recommendation would change given what was already tried, rejected or
measured.

**Tell the user only what changes the answer** — the recommendation, the assumptions, or how
the question should be read. Citing a past conversation to demonstrate continuity adds nothing.
Searching and then not mentioning the result is a normal outcome.

**Ask only when the referent is ambiguous** — "continue the plan" where several plans are live.
Even then, prefer retrieving, naming the candidates, and proceeding on the most likely with the
assumption stated. Reserve a question for when proceeding on the wrong reading would be costly.

## Writing cues

A **cue** is the text a search is matched against. It works by re-evoking the context a memory
was formed in, not by naming the thing — so it needs enough surrounding detail to pin one
episode.

- **Describe the event.** "User instructed that all prompts in the claude-memory system should
  contain clean operating principles" put the target at rank 1; the bare cue "clean principles"
  missed it entirely. Say who did what, about which system, with what intent.
- **Trim quotes to the load-bearing clause.** A quoted line matches the textually closest
  variant, often an earlier draft rather than the version that stuck, and every incidental word
  in the quote becomes part of what is matched.
- **Bare keywords** pull in lexical neighbours from any time. Never enough on their own.
- **Write in the voice you are looking for.** Conceptual phrasing finds assistant analysis;
  imperative phrasing finds user instructions.

**There is no relevance floor.** A search always returns its top few, confidently formatted,
even for something that never happened. If the results are about a different kind of thing than
the cue was, conclude the memory does not exist rather than reformulating indefinitely. Stop
after about two rewrites.

## Search arguments

Each is unrestricted when omitted, and they combine.

- `kinds=["user_message"]` / `["assistant_message"]` — helps ranking, guarantees nothing. When
  a user pastes assistant output back in, that text exists under both speakers.
- `within=<mem:id>` — confine the search to the conversation that handle came from. Any handle
  from it will do.
- `since=` / `before=` — bound the time range, half-open: `since <= when < before`, so a single
  day is `since="2026-08-08", before="2026-08-09"`. ISO 8601; no zone means your local one.
- `limit=` — how many results come back (default 8).

## Working through several hops

1. Search with an event-describing cue, narrowed where the speaker, conversation or date is
   known.
2. Expand the best result. If its content will be restated, expand to the sandwich, forward
   first.
3. Continue from the edge handles each expansion returns.
4. When the question is *where* in a conversation rather than *what*, `memory_outline` is
   cheaper than a wide window: one line per user turn, with how many events followed each — a
   turn followed by sixty is where the work happened.
5. If the target will not surface directly, search its *surroundings* — what was being
   discussed, roughly when — and expand from a neighbour.
6. For a claim that will be acted on, add one search on its status.

## Duplicates

Interrupted and resubmitted instructions appear as two or three near-identical memories ranked
next to each other. The **latest timestamp is the operative version**. Check for `[note: …]`
annotations before acting on an early draft.

## Curation

- **annotate** attaches pointers and outcomes ("superseded by mem:X", "this approach was
  abandoned"). Notes appear inline on every future retrieval. Append-only, so write facts that
  stay true rather than instructions.
- **Expand to the sandwich before writing a note.** A note drawn from an unread fragment
  reaches every future retrieval and cannot be edited or removed, only added to.
- **demote** only when a memory itself misled for a cue that will be searched again. Sharpen
  the cue first, understand why the memory exists before judging it junk, and never demote to
  clear a path to a lower-ranked answer.
