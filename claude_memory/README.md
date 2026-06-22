# claude_memory

Persistent associative memory for Claude Code, backed by EventMemory. It (1)
captures every turn verbatim, (2) surfaces relevant memories automatically before
each prompt, and (3) gives Claude `memory_search` / `memory_expand` tools to dig
deliberately. See `DESIGN.md` for how and why; this file is how to run it.

## How it fits together

```
                          ┌──────────────── memory daemon ─────────────────┐
                          │  embeddinggemma loaded once · EventMemory ·     │
                          │  shared novelty · ingest high-water mark        │
                          └───────────────▲──────────▲─────────▲───────────┘
        SessionStart ─ cli warm ──────────┘          │         │  (Unix socket)
        UserPromptSubmit ─ cli ambient ──────────────┘         │
        Claude ─ MCP tools memory_search / memory_expand ──────┤
        Stop ─ cli ingest (verbatim capture) ──────────────────┘
```

`cli mcp` and the three hook subcommands are thin clients of the one daemon,
which holds the loaded model and all state — so the ~5-6s model load happens
**once**. Everything lives under `$CLAUDE_MEMORY_HOME` (default
`~/.claude/claude_memory`).

## Fresh-system install

From a clean clone, in `agentic_expansion/`:

```bash
uv sync                                       # 1. package + server deps (embedder, sqlitevec — all locked)
uv pip install turbovec                       # 2. default ANN backend (see note); or skip + use sqlitevec
uv run python -m claude_memory.smoke          # 3. optional sanity check (no network, no model)
uv run python -m claude_memory.cli install    # 4. wire hooks + MCP + skill (--dry-run to preview)
# 5. restart Claude Code
```

Step 4 is one idempotent installer that does everything; step 5 makes it live.

- **Run everything from this repo** so `claude_memory` imports and the partition
  is stable. Start Claude Code in `agentic_expansion/`.
- **Embedder** — local **embeddinggemma-300m**, the only model. `sentence-transformers`
  and `torch` are declared deps, so `uv sync` installs them; the model **weights**
  download from HuggingFace into the HF cache on the first real turn (one-time,
  network — set `HF_TOKEN` if rate-limited). **No cloud/OpenAI model, no API key.**
- **Vector backend** — the default **turbovec** (TurboQuant-compressed ANN) is a
  local wheel **not** in the lockfile, so `uv sync` neither installs it nor keeps
  it: run `uv pip install turbovec` (step 2) and re-run after any later `uv sync`.
  To avoid it entirely, set `CLAUDE_MEMORY_VECTOR_BACKEND=sqlitevec` (exact
  float32, no extra dep — `sqlite-vec` is already locked) and skip step 2.
- The SessionStart hook warms the daemon so the first prompt's recall is instant
  (~0.1s warm calls).

## 1. Smoke test (no key, no network)

```bash
uv run python -m claude_memory.smoke            # core + daemon suites
uv run python -m claude_memory.smoke --daemon   # one suite
```

Uses a deterministic hash embedder. Expect `SMOKE PASSED`.

## 2. Enable globally for all projects (recommended)

One idempotent installer wires it into every future session, using the repo's
venv Python + `PYTHONPATH` (so it resolves from any project's cwd) and pinning
the database location:

```bash
uv run python -m claude_memory.cli install --dry-run   # preview
uv run python -m claude_memory.cli install             # apply
uv run python -m claude_memory.cli install --disable   # uninstall (keeps your data)
uv run python -m claude_memory.cli install --disable --purge   # uninstall + delete databases
```

It merges the three hooks into `~/.claude/settings.json` (backed up to
`settings.json.bak`), registers the MCP server at user scope via `claude mcp add`,
and copies the **`episodic-recall` skill** into `~/.claude/skills/`. Restart
Claude Code afterward. Databases default to `~/.claude/claude_memory` (override
with `--db-home PATH`). Flags: `--skip-mcp` / `--skip-skill` to omit a piece.

The hooks register `SessionStart`→`cli warm` (spawns/warms the daemon),
`UserPromptSubmit`→`cli ambient` (non-blocking), `Stop`→`cli ingest` (capture).
All best-effort: they never fail a turn, and the first turn of a fresh store
recalls nothing (there is nothing yet).

## 2b. Or enable for this project only

```bash
uv run python -m claude_memory.cli install --scope project
```

Writes `./.claude/settings.json` and registers the MCP server at project scope.

## Uninstall

```bash
uv run python -m claude_memory.cli install --disable           # keep data
uv run python -m claude_memory.cli install --disable --purge   # also delete databases
```

`--disable` reverses everything install did: it removes the three hooks from
`settings.json` (leaving your other hooks/settings untouched, with a fresh
`.bak`), runs `claude mcp remove`, removes the installed `episodic-recall` skill,
and shuts down the running daemon. It keeps the databases unless you add
`--purge`. Match the scope you installed with (`--scope project` if you enabled
per-project). After it, restart Claude Code. (There is nothing else to clean up —
no system services, no global env changes.)

## Configuration (environment)

| Variable | Default | Meaning |
|---|---|---|
| `CLAUDE_MEMORY_HOME` | `~/.claude/claude_memory` | Where the DBs + state live |
| `CLAUDE_MEMORY_EMBEDDING` | `embeddinggemma` | `embeddinggemma` (the only model) or `hash` (offline tests). Nothing else is accepted. |
| `CLAUDE_MEMORY_NAMESPACE` | `claude` | Vector-store namespace |
| `CLAUDE_MEMORY_PARTITION` | derived from cwd | Pin to share one memory across checkouts |
| `CLAUDE_MEMORY_VECTOR_BACKEND` | `turbovec` | `turbovec` (compressed ANN, fast at scale, ~10× smaller vectors, approximate) or `sqlitevec` (exact float32). |
| `CLAUDE_MEMORY_TURBOVEC_BITS` | `4` | turbovec quantization bits/dim (2/3/4). `2` halves the index at some recall cost. |
| `CLAUDE_MEMORY_EVICTION_THRESHOLD` | unset (off) | Cosine similarity at/above which derivatives form one cluster. Set (e.g. `0.9`) to enable eviction; unset disables it. `0.90` ≈ OpenAI text-embedding-3-small `0.85` (calibrated); `0.95` is much stricter (≈ text-embedding-3-small `0.93`). |
| `CLAUDE_MEMORY_EVICTION_TARGET` | `15` | Max cluster size kept; over this, temporally middle members are evicted (oldest/newest kept). |
| `CLAUDE_MEMORY_EVICTION_SEARCH_LIMIT` | `20` | Max stored neighbours fetched per new derivative when evaluating eviction. |

## Eviction (semantic compaction) — optional, off by default

When `CLAUDE_MEMORY_EVICTION_THRESHOLD` is set, ingest caps each cluster of
near-duplicate memories at `…_TARGET`, deleting the temporal middle and keeping
the earliest + latest. This is implemented in `EventMemory` itself
(`eviction_similarity_threshold=None` ⇒ disabled), so it works for any backend,
not just this agent. Unset ⇒ identical to before (no eviction).

The setting is read when the **daemon starts**, so it's fixed for that daemon.
To A/B *eviction off vs on* cleanly, give each its own home so both memories
coexist:

```bash
# baseline (no eviction)
CLAUDE_MEMORY_HOME=~/.claude/cm_off \
  uv run python -m claude_memory.cli daemon restart

# eviction on
CLAUDE_MEMORY_HOME=~/.claude/cm_on CLAUDE_MEMORY_EVICTION_THRESHOLD=0.9 \
  uv run python -m claude_memory.cli daemon restart
```

(Or point the installed hooks at one home and restart the daemon with the env
var set to flip the live agent between modes.)

## Vector backend (turbovec, default)

By default the vector store is `SQLiteVectorStore` + the **turbovec** search engine:
SQLite holds only `uuid` + `properties` (no vectors), and the vectors live as a
TurboQuant-compressed ANN index in `$CLAUDE_MEMORY_HOME/vector_index/*.idx`.

- **Why:** fast approximate search that stays ~flat as the corpus grows (sqlite-vec
  scans float32 and scales linearly), and a small in-memory index (~384 B/vector
  at 4-bit ≈ ~10× smaller than float32). Storage savings on *total* disk are modest
  (the `properties` records and `segment.db` dominate); the win is search speed.
- **Trade-offs:** approximate (quantized cosine, near-exact recall on real
  embeddings, not bit-exact); cosine/dot metrics only; **no vector read-back**
  (fine here — EventMemory only queries with `return_vector=False`).
- **Persistence:** the index is in RAM while the daemon runs, saved to `.idx` on
  shutdown and every ~1000 ops, and rebuilt on restart via load + pending-op
  replay (so a hard kill loses nothing — unsaved ops are replayed from SQLite).
- **Revert / tune:** `CLAUDE_MEMORY_VECTOR_BACKEND=sqlitevec` for exact float32
  (no turbovec dep, vectors in sqlite-vec); `CLAUDE_MEMORY_TURBOVEC_BITS=2|3|4`.

Switching backends re-reads the env at daemon start, so `daemon restart` after
changing it. The two backends use the same `vector.db` records but different
vector storage, so for a clean A/B give each its own `CLAUDE_MEMORY_HOME`.

## Daemon control

The daemon is a single long-lived process (one per `CLAUDE_MEMORY_HOME`, shared by
every session and project). Control it by subcommand — these address it by its
own home-keyed socket and lock, never by process-name matching, so they can only
ever touch *this* home's daemon (unlike `pkill -f`, which can match unrelated
processes):

```bash
uv run python -m claude_memory.cli daemon status    # running? pid, home, socket
uv run python -m claude_memory.cli daemon stop       # graceful shutdown (verified-pid fallback)
uv run python -m claude_memory.cli daemon start      # spawn + warm
uv run python -m claude_memory.cli daemon restart    # stop, then start fresh
```

(It also idle-exits on its own after 30 min — `CLAUDE_MEMORY_DAEMON_IDLE`.)

### When to restart (picking up your changes)

Code runs in **two** long-lived places, so two refresh actions:

- **Memory engine changes** (`engine.py` / `daemon.py` / `transcript.py` — search,
  expand, ingest, embedder) run in the **daemon**. Pick them up with
  `daemon restart`. A Claude Code restart alone does *not* refresh the daemon (the
  SessionStart hook reuses an already-running one).
- **Tool docstrings / usage instructions** (`memory_search` / `memory_expand`
  help) run in the **MCP child**, one per Claude session. Pick them up by
  **restarting Claude Code**. A daemon restart does *not* refresh these.

To get both at once: `daemon restart`, then restart Claude Code.

## Inspect / reset

```bash
H="${CLAUDE_MEMORY_HOME:-$HOME/.claude/claude_memory}"
ls -la "$H"                 # vector.db, vector_index/, segment.db, state/, daemon.sock, daemon.lock, daemon.log
tail -f "$H/daemon.log"     # daemon startup + errors
rm -rf "$H"                 # wipe all memory
rm "$H"/state/*             # re-ingest from scratch next turn
```

## Caveats (see DESIGN.md §9)

- Reasoning/CoT is captured only if present in the transcript (best-effort).
- Embeddings are 100% local (embeddinggemma-300m); no OpenAI/cloud access at all.
- A groundedness verifier and the "force another round" revisit scheduler are
  still deferred (the latter pending a Stop-hook continuation probe).
