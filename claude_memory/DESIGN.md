# claude_memory — design & rationale

A persistent associative memory for Claude Code, backed by **EventMemory** (the
repo's SQLite VectorStore + SegmentStore). This document is meant to let someone
(or some agent) understand the whole system cold. Read it before changing
anything.

---

## 1. The mental model

Human memory has two modes, and so does this:

- **Ambient recall** — things surface *unbidden* while you read, without
  deciding to remember. Here: a `UserPromptSubmit` hook searches memory with the
  user's prompt every turn and injects the top hits. Involuntary, cheap, loose
  precision. This is what answers *"the agent should use memory without being
  told."* You don't make the model decide; baseline recall is automatic.

- **Deliberate recall** — effortful, directed lookup and follow-the-thread. Here:
  MCP tools (`memory_search`, `memory_expand`) that Claude calls itself, the way
  it calls Grep/Read. This is where multi-hop reasoning happens (search → read a
  result → search the lead it revealed).

Plus **verbatim capture**: a `Stop` hook ingests the full turn (messages, tool
calls, tool results, reasoning where present) into the timeline after every turn.

Why split it this way: the deliberate decisions (when to dig, how to phrase a
cue, when to stop) belong to a model *trained for agentic tool loops* — that is
Claude, and it is good at them. The things a model cannot reliably self-govern
(noticing recall has gone dry, keeping a worklist across turns) belong to the
harness, computed deterministically. That division is the whole philosophy.

---

## 2. The daemon and its thin clients

Because the default embedder is local (`embeddinggemma`, ~5-6s to load), the
model must load **once**, not per call. So a long-lived **daemon** owns the
loaded embedder, the EventMemory stores, the per-(partition, session) novelty
state, and the ingest high-water mark. Three thin clients talk to it over a Unix
socket (`$CLAUDE_MEMORY_HOME/daemon.sock`). All clients are subcommands of one
`cli.py`:

- `cli mcp` — the `memory_search` / `memory_expand` MCP tools
- `cli ambient` — UserPromptSubmit auto-recall
- `cli ingest` — Stop-time verbatim capture

A fourth, `cli warm` (SessionStart), spawns + warms the daemon so the first
prompt's recall is instant. Lifecycle: single instance enforced by an
exclusive `flock`; auto-spawned by the first client that needs it (with a spawn
marker for backoff so callers don't pile up waits — the daemon clears that marker
once it is listening); idle-exits after `CLAUDE_MEMORY_DAEMON_IDLE` (default
1800s); accepts a `shutdown` op.

`cli daemon {status,start,stop,restart}` controls it explicitly (e.g. to pick up
edited engine code — `daemon restart`). Crucially, these **address the daemon by
its home-keyed primitives, never by process name**: `stop` asks the home's socket
to `shutdown` (graceful), and only if that fails does it signal the PID the
daemon wrote into *this home's* lock file — after confirming via the lock that
that process is still the live daemon (the OS holds the `flock` for the daemon's
whole life) and via `ps` that it is a `claude_memory.daemon`. So a stop/restart
can only ever touch this home's daemon, not some unrelated process a `pkill -f`
pattern might match. Liveness itself is read off the lock (`daemon_alive`: if the
lock can't be acquired, a daemon holds it).

Two things keep this from getting fragile:

1. **Deterministic ids** — a memory's handle is `mem:<segment-uuid-hex>`, a pure
   function of the segment. `expand` resolves a handle back to a segment via the
   store, so there is no id registry to keep in sync even across the daemon's
   per-partition cores.
2. **Clients hold no state** — all of it (model, cores, novelty, hwm) lives in
   the daemon. A client is just connect → one JSON request → one JSON reply.

Novelty is keyed by (partition, session). The hooks know the session id; MCP
calls do not, so they attach to the partition's most recently active session —
which, mid-conversation, is the current one. Good enough; documented limitation.

Partitioning: each project gets its own collection/partition, derived from the
working directory (`config.partition`). Clients compute it from their cwd and
send it; the daemon opens that partition's core on demand and serves any number
of projects from the one loaded model. Start Claude in the project (or pin
`CLAUDE_MEMORY_PARTITION`) so clients agree on the partition.

---

## 3. Storage: one timeline, one search surface

Two concerns that look like one but are not:

- **Reconstruction substrate** — a *single ordered timeline* per project holding
  everything: user/assistant messages, reasoning, tool calls, tool results, and
  text injected into the session rather than typed in it. This is what `expand`
  walks. It must be unified, because replaying "what happened" means pulling a
  *contiguous slice* — e.g. a trip-planning request *and the tool calls that
  fulfilled it*.

- **Search surface** — only **message** events are embedded (see `engine.py`:
  `MessageOnlyDeriver` emits no derivative for non-message segments). So the
  message stream is what vector search ranks over. Tool calls,
  their results, and file contents are **reached by expansion**, never searched
  directly.

Why: raw command strings and file blobs live in a different register and would
out-rank or drown natural-language messages; and embedding large low-value blobs
you would never search for by content is pure waste. You still get them back —
by timeline adjacency to a message seed — which is exactly the replay path.

The same reasoning puts `injected` outside the search surface. A user turn carries
both what the user wrote and what was loaded in around them (hook context, skill
bodies, system reminders, slash-command echoes, the session's own compaction
summary); role cannot tell them apart, so `wire.user_text_source` classifies on the
text at ingest. Measured on one real corpus it was 37% of user-role segments. The
compaction summary is the sharpest case: compaction does not fork a session, so the
summary sits among the very turns it paraphrases, and embedding it only lets a
description outrank its own source. It stays on the timeline, where it is the one
record of where the session lost its context.

Search composes its own filter with the searchable sources rather than trusting the
index's contents, since anything embedded before a source became non-searchable
would otherwise need a full re-index to remove.

Metadata: every event carries `source`, `producer` (speaker), `session_id`, and
tool calls also carry `tool_name` / `path`. `producer`/`source`/`session_id` are
indexed, which is what lets a search be scoped by `kinds` or `within` without a
scan; `session_id` and `timestamp` are indexed together so a conversation's spine
can be read in one pass.

---

## 4. The tool surface (and what was deliberately cut)

```
memory_search(cue, limit=8, within=None, kinds=None,
              since=None, before=None)                     -> str
memory_expand(id, before=5, after=5, unit="segments",
              kinds=None, blocklist=False)                 -> str
memory_outline(id, before=20, after=20)                    -> str
memory_demote(id, cue)                                     -> str
memory_annotate(id, note)                                  -> str
```

- **One address space.** Every `id` is a `mem:` handle naming a segment, given as
  the shortest unambiguous prefix. A handle also names its conversation, so the
  same value scopes an outline or a `within=` search; there is no session id in
  the surface. Narrowing is named parameters rather than a filter language, so
  the schema states what may be narrowed and the model cannot write an expression
  that parses but matches nothing.

- **`memory_expand`'s `kinds`** — which sources a window may spend its budget on,
  read as an allowlist or (with `blocklist`) a blocklist. It is pushed into the
  store's window walk rather than applied to its result, which is the whole point:
  the walk is LIMIT-bounded, so filtering afterwards returns fewer segments than
  were asked for, with the budget already spent on what was dropped. Naming kinds
  replaces the default outright; the default blocks only `injected`.

- **`memory_outline`** — a conversation's spine: one line per user turn, with how
  many events followed it before the next. It answers *where*, which neither of
  the other read tools does — search finds a moment and expand reads around one,
  and getting structure out of them means a huge window spent on text nobody
  wanted. The event count is the signal: a turn followed by sixty is where the
  work happened, a run of turns followed by two each is a session that kept
  changing direction.

- **`memory_demote`** — score-free negative feedback: "this memory was wrong for
  this cue." Each call decays the memory's similarity to the cue geometrically
  (and to similar cues, via cosine); the result echoes the cue's current top
  matches so the model judges from the ranking whether to continue or stop.

- **`memory_annotate`** — append-only recontextualization: attach a one-line
  note to a segment, rendered after its content as `[note: ...]` wherever it
  surfaces (search, expand, ambient). The note never enters the embedding
  anchor (the deriver ignores `AnnotationContext`), so vectors and ranking are
  untouched — it changes what is known when a memory resurfaces, not when. Use
  case: a memory that was later corrected carries its correction to every
  future retrieval that hits the same segment.

- **No `follow_lead` tool.** Following a lead *is* a search with a new cue. A
  separate tool would fragment the action space and imply a mechanism that isn't
  there. Multi-hop = call `memory_search` again. (Caroline "moved from her home
  country" → search "Caroline home country" → "Sweden".)

- **`memory_expand`'s `unit`** — what `before`/`after` count. Segments by default,
  which is a flat budget and the only way to read *inside* one long event. The
  alternative is whole events, for when the walk keeps starving: a single tool
  result can run to a thousand segments, so a segment-counted window can spend
  itself entirely inside one blob and reach no other turn. Counting events buys
  turns instead, at the cost of an unbounded amount of text per step — which is
  why it is not the default and why oversized events are sampled rather than
  shown whole.

- **No `memory_check_sufficiency` tool.** See §6 — it was overpromising.

Cue construction (Temporal Context Model): a cue re-evokes the *context* a memory
was encoded in, not just the item's name — so `memory_search`'s description tells
the model to give cues enough context to pin a specific episode (an event
description, or the verbatim line with speaker, with the *why* when available),
**not a bare entity** ("cat" is too diffuse). A question or statement both work;
the user's original wording is often a fine cue (no reflexive HyDE rewriting,
which done badly is worse); trying both a literal and a sharper cue can help. When
the target can't be pinned directly, the model searches for its *surrounding
context* and `memory_expand`s to the target — like method of loci: traverse the
remembered path, using the place as the cue, not the item — and may ask the user
for that context (how long ago, what was being discussed). Guidance lives on the
tool, not a self-policed system prompt.

---

## 5. Diminishing returns is computed, not self-judged

The daemon keeps a per-(partition, session) set of segment uuids surfaced so far,
shared across ambient recall and the MCP tools. Every search reports *new vs
already-seen*. Zero-new is the honest **"recall is saturating"** signal —
surfaced as data ("0 new — try a different angle or stop"), never asked of the
model as "are you done?". This is the legitimate give-up cue: the
diminishing-returns mechanism humans actually use, computed externally.
(`MemoryCore` also carries its own set for the daemon-less standalone path.)

---

## 6. On "sufficiency" — the honest version

**Groundedness is verifiable; completeness is not.**

- *Groundedness* ("does my evidence actually say what I'm claiming?") is a real
  entailment check and a fine future tool. It is the analogue of "the test
  passed."
- *Completeness* ("have I found everything?") cannot be verified in general — it
  requires knowing the complete set you're checking against, which is the thing
  you don't have. A tool claiming to check it would lie.

So this system does **not** try to make the model self-judge sufficiency, and it
does **not** rely on a visible budget counter (telling a model "turn 3 of 10" is
a *self-monitoring* task models calibrate poorly; availability ≠ activation).
Instead, stopping rests on: (a) the computed diminishing-returns signal of §5,
and (b) the model being Claude, which has trained satisficing calibration. This
mirrors how Claude Code itself stops: not on a harness budget (there is none),
but on verifiable task-state plus trained judgment.

A future **groundedness** check (separate actor/verifier call) is the right way
to harden commitment — see §9. It checks *support*, not *completeness*.

---

## 7. Cacheability (a first-class constraint)

The KV cache is preserved by **appends** and destroyed by **prefix mutations**.
Every injection in this system is an append:

- **Deliberate recall** enters as MCP `tool_result` blocks — appended at the
  current conversation position. Pure append.
- **Ambient recall** enters as `additionalContext` attached to the *new* user
  turn — after all previously cached context. Pure append.
- **Expansion sends the window once.** The neighbours are what the call is for;
  the seed comes back with them, marked `← expanded from here`, because a window
  whose centre is missing cannot be read as a timeline and the model would have to
  reconstruct where it anchored. That is one segment of overlap against a window
  of many, and it is still an append — nothing earlier in context is rewritten.
- **There is no mutable "memory panel."** A maintained/edited memory region would
  rewrite a prefix every turn and blow the cache — explicitly avoided.
- **No mid-session discard.** Dropping items mid-stream = prefix rewrite = cache
  kill. If pruning is ever wanted, do it in a `PreCompact` hook, where the cache
  is already being rebuilt and the drop is free.

---

## 8. Performance

- **The embedder is local `embeddinggemma-300m`** — the only model; no cloud /
  OpenAI embedder is imported or used (no network, no API key). 768-dim, cosine,
  asymmetric query/document prompts handled in `engine.py`. It loads its weights
  once, in the **daemon** (~5-6s; measured ~7s cold incl. first ingest).
- **The daemon makes that a one-time cost.** Warm round trips are ~0.09s
  (measured), so ambient recall, MCP tools, and ingest are all fast after warm-up.
  The SessionStart hook warms it before the first prompt, so the user never waits.
- **Ambient recall never blocks**: its client call uses `wait_for_start=0` — if
  the daemon isn't warm yet it simply skips that turn rather than stalling the
  prompt. Ingest waits (end of turn, tolerable); MCP warms on first call.
- **Ingest is once per turn (Stop)**, batched — at most one writer, no per-tool
  write contention.
- `hash` (a deterministic offline double) is the only other accepted value, for
  tests; the daemon-less standalone path is otherwise unchanged.
- **Vector backend = turbovec by default** (`SQLiteVectorStore` + TurboQuant ANN;
  `_build_vector_store` in `engine.py`, switchable via
  `CLAUDE_MEMORY_VECTOR_BACKEND`). SQLite stores only `uuid`+`properties`; the
  vectors are a compressed in-memory index (~384 B/vec at 4-bit, ~10× smaller than
  float32) persisted to `$CLAUDE_MEMORY_HOME/vector_index/*.idx` on shutdown and
  every ~1000 ops, reloaded on restart (pending-op replay covers a hard kill).
  A save is turbovec's own `IdMapIndex.sync` (turbovec ≥ 1.0.0, container **v7**),
  which appends what changed since the last checkpoint rather than restating the
  whole index and commits it durably — one fsync, and a crash at any byte of it
  leaves the previous commit intact. That closes the gap this store used to carry:
  publication was atomic but not durable, so a **power** failure could revert the
  last publication after the pending log — the only other copy of those vectors —
  had been trimmed behind it, leaving memories that still expand from `segment.db`
  but stop being *searchable*, undetected. Re-running ingest never fixed that:
  event uuids are derived from the transcript record, so `ingest` skips anything
  already in `segment.db` rather than re-embedding it, and recovery meant
  rebuilding a fresh home from the transcripts, which reaches only as far back as
  `cleanupPeriodDays` retention. The v3 index written by turbovec 0.7.0 is not
  convertible to v7 (v3 predates the v5 rotation change), so the one-time move is
  a rebuild from the segment text — `claude_memory.migrate_index_v7`, which
  preserves row ids and reapplies the demotion deltas.
  Chosen for **search speed at scale** (≈flat vs sqlite-vec's linear float32 scan),
  not disk — total disk drops only ~15-20% since `segment.db` and the property
  records dominate. Approximate (quantized cosine; near-exact recall on real
  embeddings), cosine/dot only, no vector read-back — fine because EventMemory
  only queries `return_vector=False`. `sqlitevec` reverts to exact float32.

---

## 9. Known limitations & what's still deferred

Limitations (all documented at their site):

- **Reasoning capture is best-effort.** Extended-thinking blocks are not reliably
  written to the transcript, and hooks cannot see hidden reasoning. We capture
  everything else verbatim (messages, tool calls + args, tool results, files).
- **MCP novelty attaches to the latest session.** Novelty is shared via the
  daemon, keyed by (partition, session). MCP calls carry no session id, so they
  attach to the partition's most recently active session — almost always the
  current one, but a corner case if two sessions share a partition concurrently.
- **Cross-batch sub-second ordering.** Timestamps are made strictly increasing
  within an ingest batch; across batches we rely on real timestamps being
  non-decreasing. Pathological same-second boundaries could disorder by
  microseconds. Negligible for reconstruction.
- **Partition depends on consistent cwd** across the clients (see §2).
- **Forced continuation is not implemented.** The "revisit" scheduler (a
  deterministic Stop-hook that detects stalls / open leads and forces another
  round) hinges on whether a `Stop` hook can reliably block-and-continue in
  current Claude Code — *unverified*. Verify that with a probe before building
  it. Everything here works regardless.
- **Contextualized embeddings (future, not now).** Today each message segment is
  embedded in isolation. A TCM-flavoured upgrade: embed a message *with its
  neighbourhood* — e.g. a weighted average of `emb(message)` and `emb(its ~5-10
  preceding neighbours + message)` — so each stored vector carries conversational
  context and recall of context-dependent targets improves. Implement via custom
  Context / Segmenter / Deriver types that carry neighbour info. Deliberately
  deferred: ship the simple non-contextualized embeddings first, measure, then try
  this (don't pre-optimize).

The daemon (§2) is **built**: the local model loads once, novelty is shared, and
it is the natural home for the two things still deferred — a **groundedness
verifier** (§6: checks support, not completeness) and the **revisit scheduler**
(pending the Stop-hook probe). Stateless ids keep the daemon-less standalone
`MemoryCore` path (used by `smoke.py`'s core suite) correct too.

### Ambient recall is off by default (measured 2026-08)

`cmd_ambient` (the `UserPromptSubmit` hook) retrieves on the user's raw prompt
and injects the hits before the model answers. It ships **disabled**
(`ambient_enabled`, default false). The reasoning, from the channel's own
observability log and a replay over real transcripts:

- **The gate does not discriminate.** Across 560 logged injections the top
  cosine ran from 0.549 at the 10th percentile to 0.685 at the 90th, and a
  one-word reply retrieved at a median 0.626 against 0.656 for prompts of fifty
  words or more. No threshold on that score separates turns that need memory
  from turns that do not, which is why the channel fired on 92% of prompts.
- **Roughly a quarter of injections change the answer.** Lexical overlap
  between an injected memory and the reply that followed it ran 23.7%. A blind
  test, in which a judge had to pick the real injection out of a decoy drawn
  from the same conversation minutes away, implied about 25% independently.
- **About 1.3% of injections produced a claim the user then had to correct.**
  Measured by tracing memory text into the reply and then into the user's next
  message, with every quoted span verified as a verbatim substring of its
  source. The blind decoy arm scored 7.9% where the real arm scored 50.4%, so
  the trace is signal rather than the judge confabulating.
- **No benefit figure exists to weigh against that, and this data cannot
  produce one.** A memory that helps draws no reaction and so leaves no label,
  while one that misleads produces a visible correction. The measurement is
  one-sided by construction, which means "leave it on" was winning by absence
  of evidence rather than by evidence.

What remains is the deliberate path — `memory_search`, `memory_expand`,
`memory_outline` — where the cue states what is being looked for, so a retrieved
fragment can be tested against a purpose that existed before it arrived. An
ambient injection has no such target, so a fragment that does not apply produces
no mismatch signal at all.

To re-enable for an experiment, with no reinstall: set `ambient_enabled: true` in
`<home>/config.json`, or `CLAUDE_MEMORY_AMBIENT=1`. The hook stays registered
either way. Transcript capture, the MCP tools, and `roster.py` (cross-session
activity) are unaffected — none of them were part of this channel.

---

## 10. File map

| File | Role |
|---|---|
| `engine.py` | The memory engine: config, sources, embedder, deriver, stores, `MemoryCore` (search/expand/ingest), ids, novelty, rendering, wire serde |
| `transcript.py` | Claude transcript JSONL → timeline events |
| `daemon.py` | The single memory process + its Unix-socket IPC: server (`MemoryService`) + client (`call`) + per-session novelty + ingest hwm + lifecycle control (`daemon_alive` / `stop_daemon` / `daemon_status`, addressed by socket + lock, not process name) |
| `cli.py` | One entry point: `mcp` / `warm` / `ambient` / `ingest` / `daemon` (status/start/stop/restart) / `install` (hooks + MCP server are thin daemon clients; `install` writes the config) |
| `smoke.py` | Dependency-free tests: `core` (engine) + `daemon` (socket) suites |
