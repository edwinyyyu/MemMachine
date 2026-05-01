"""AEN-1 SLIDING — sliding-window writer with active-chain injection.

Replaces aen1_active's batch-boundary writer (every batch_size turns the writer
sees only that 5-turn block + recent log + active state) with a sliding-window
writer:

  Every K turns, the writer fires with a window of the most recent W turns.
  The OUTPUT is constrained to memory entries about the LAST K turns; the older
  W-K turns are CONTEXT for coreference / state inference only.

Why: batch boundaries fragment cross-turn coreference (anonymous->named pairs
that straddle a batch boundary lose context) and force the writer to re-derive
state at each boundary. Sliding window dissolves those boundaries.

Reuses aen1_active's:
  - active-chain injection from supersede_head (structural index)
  - LogEntry / IndexedLog / build_index / retrieve / answer_question

Overrides:
  - write_window: window-aware prompt; output keyed on LAST K turns.
  - ingest_turns: sliding window driver; per-K-turn writer fire.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND16A = HERE.parent
RESEARCH = ROUND16A.parent
ROUND15 = RESEARCH / "round15_active_chains"
ROUND11 = RESEARCH / "round11_writer_stress"
ROUND7 = RESEARCH / "round7"
sys.path.insert(0, str(ROUND15 / "architectures"))
sys.path.insert(0, str(ROUND11 / "architectures"))
sys.path.insert(0, str(ROUND7 / "experiments"))

import aen1_active  # noqa: E402
import aen1_simple  # noqa: E402
from _common import Budget, Cache, extract_json, llm  # noqa: E402

# Re-export for callers
LogEntry = aen1_simple.LogEntry
IndexedLog = aen1_simple.IndexedLog
build_index = aen1_simple.build_index
retrieve = aen1_simple.retrieve
answer_question = aen1_simple.answer_question
extract_batch_entities = aen1_active.extract_batch_entities
gather_active_state = aen1_active.gather_active_state
render_active_state = aen1_active.render_active_state


# ---------------------------------------------------------------------------
# Sliding-window writer prompt
# ---------------------------------------------------------------------------

WRITE_PROMPT = """You are a semantic-memory writer using a SINGLE APPEND-ONLY LOG.

You are processing a CONVERSATION WINDOW. The window contains:
  - CONTEXT TURNS: earlier turns shown for coreference / state inference. Do
    NOT emit memory entries about these. They are already accounted for.
  - TARGET TURNS: the NEW turns (last {k} turns of the window). You MUST emit
    memory entries (or skip) for these turns ONLY.

Each entry you write is an atomic natural-language fact. Use @Name tags to
mention entities. Use `refs` to point at prior entries when this entry relates
to, updates, or corrects a prior fact.

REFS: a list of prior entry UUIDs - there is only ONE kind of ref. If this
entry updates/corrects/refines/clarifies a prior fact, include that prior entry
in `refs`. The prose text carries the nuance.

PREDICATE (optional but recommended for state-tracking facts):
  Format: "@Entity.predicate_name" - e.g. "@User.employer", "@Jamie.job".

KNOWN ENTITIES so far: {known_entities}

ACTIVE STATE OF ENTITIES IN THIS WINDOW (each line is the CURRENT chain head
for that (entity, predicate) - if a TARGET turn updates/corrects/replaces one
of these states, you MUST include its uuid in `refs`. Do NOT emit a ref if the
new entry only mentions the entity casually):
{active_state}

PRIOR LOG SAMPLE (most recent committed entries; cite older ones by uuid):
{prior_log}

CONVERSATION WINDOW
-------------------
{window_block}

Emit JSON describing memory entries for the TARGET turns ONLY:
{{
  "entries": [
    {{
      "turn": <integer turn number, MUST be from the TARGET turns>,
      "text": "<atomic fact in one sentence>",
      "mentions": ["@Name", ...],
      "refs": ["<prior-uuid>", ...],
      "predicate": "@Entity.pred" or null
    }}
  ]
}}

RULES
- ONLY emit entries whose `turn` is in the TARGET turn range. CONTEXT turns are
  read-only.
- Use @Name for every named entity. ALWAYS @User when the speaker is the subject.
- If a TARGET turn is pure filler (weather, chitchat, jokes), skip it silently.
- If a TARGET turn UPDATES or CORRECTS a prior fact, emit a new entry with
  `refs` pointing at the matching ACTIVE STATE entry's uuid. Make the prose
  carry the nuance.
- For state-tracking facts (job, location, boss, employer, role, relationship,
  partner, hobby, commute, car, gym, team, title, etc.), include the
  `predicate` ("@Entity.pred_name", lowercase). REUSE the predicate names shown
  in the ACTIVE STATE block when updating those chains.
- A coreferential update from a CONTEXT turn (e.g. CONTEXT had "my new boss",
  TARGET says "his name is Marcus") MUST resolve in the new entry: write the
  fact with the resolved name AND ref the entry that introduced the descriptor
  if it was already committed.
- Prefer ONE entry per turn. If TARGET turns have no memory-worthy content,
  output {{"entries": []}}.
- Do NOT invent entities. Do NOT add @User to facts where User isn't mentioned.
- Output JSON ONLY.
"""


def _render_prior_log(
    prior_entries: list[LogEntry],
    target_turn_lo: int,
    max_recent: int = 12,
) -> str:
    """Render the most recent committed entries (those with ts < target_turn_lo).

    We avoid showing entries that are already in the conversation-window
    rendering (the window itself shows the raw turns; prior_log shows the
    derived memory state from earlier in the log).
    """
    eligible = [e for e in prior_entries if e.ts < target_turn_lo]
    recent = list(reversed(eligible[-max_recent:]))
    if not recent:
        return "(empty)"
    lines = []
    for e in recent:
        ref_str = f" refs=[{','.join(e.refs)}]" if e.refs else ""
        pred_str = f" pred={e.predicate}" if e.predicate else ""
        lines.append(
            f"[{e.uuid}] t{e.ts} mentions={','.join(e.mentions)} "
            f":: {e.text}{ref_str}{pred_str}"
        )
    return "\n".join(lines)


def _render_window(
    window_turns: list[tuple[int, str]],
    target_turn_lo: int,
) -> str:
    """Render the window with explicit CONTEXT / TARGET separation."""
    lines = []
    in_target = False
    for tidx, text in window_turns:
        if not in_target and tidx >= target_turn_lo:
            lines.append("--- TARGET TURNS (emit entries for these) ---")
            in_target = True
        if not in_target:
            lines.append(f"  CONTEXT TURN {tidx}: {text}")
        else:
            lines.append(f"  TARGET  TURN {tidx}: {text}")
    if not in_target:
        # All target (window_size <= K)
        lines.insert(0, "--- TARGET TURNS (emit entries for these) ---")
    return "\n".join(lines)


def write_window(
    window_turns: list[tuple[int, str]],
    target_turn_lo: int,
    target_turns: list[tuple[int, str]],
    prior_entries: list[LogEntry],
    idx: IndexedLog | None,
    known_entities: set[str],
    cache: Cache,
    budget: Budget,
    max_active_state_size: int = 100,
) -> tuple[list[LogEntry], dict]:
    """Write entries for the TARGET turns using the full WINDOW as context.

    Returns (new_entries, telemetry_dict).
    """
    # Active-state lookup uses entities from the *target* turns (those are the
    # ones whose state is potentially updating). Context turns inform coref but
    # don't drive active-state injection.
    target_entities = aen1_active.extract_batch_entities(target_turns)
    heads = aen1_active.gather_active_state(idx, target_entities, max_active_state_size)
    active_state_str = aen1_active.render_active_state(heads)

    prior_log = _render_prior_log(prior_entries, target_turn_lo)
    window_block = _render_window(window_turns, target_turn_lo)
    k = len(target_turns)

    prompt = WRITE_PROMPT.format(
        k=k,
        known_entities=", ".join(sorted(known_entities))
        if known_entities
        else "(none)",
        active_state=active_state_str,
        prior_log=prior_log,
        window_block=window_block,
    )
    raw = llm(prompt, cache, budget)
    obj = extract_json(raw)
    telemetry = {
        "n_active_state_heads": len(heads),
        "active_state_chars": len(active_state_str),
        "prompt_chars": len(prompt),
        "window_size": len(window_turns),
        "k": k,
        "target_turns": [t for t, _ in target_turns],
        "context_turns_n": len(window_turns) - k,
    }
    if not isinstance(obj, dict):
        return [], telemetry

    target_turn_set = {t for t, _ in target_turns}
    entries_raw = obj.get("entries", []) or []
    entries: list[LogEntry] = []
    counter = 0
    for e in entries_raw:
        if not isinstance(e, dict):
            continue
        text = (e.get("text") or "").strip()
        if not text:
            continue
        ts_raw = e.get("turn")
        try:
            ts = int(ts_raw) if ts_raw is not None else target_turns[-1][0]
        except (TypeError, ValueError):
            ts = target_turns[-1][0]
        # Coerce ts to be inside target range; if writer hallucinates a turn
        # outside, snap to last target turn.
        if ts not in target_turn_set:
            ts = target_turns[-1][0]
        mentions = [m for m in (e.get("mentions") or []) if isinstance(m, str)]
        refs_raw = e.get("refs") or []
        refs = [r for r in refs_raw if isinstance(r, str)]
        predicate = e.get("predicate")
        if predicate is not None and not isinstance(predicate, str):
            predicate = None
        uuid = f"e{ts:04d}_{counter}"
        counter += 1
        entries.append(
            LogEntry(
                uuid=uuid,
                ts=ts,
                text=text,
                mentions=mentions,
                refs=refs,
                predicate=predicate,
            )
        )
    return entries, telemetry


def ingest_turns(
    turns: list[tuple[int, str]],
    cache: Cache,
    budget: Budget,
    *,
    window_size: int = 15,
    k: int = 3,
    rebuild_index_every: int = 4,
    max_active_state_size: int = 100,
) -> tuple[list[LogEntry], IndexedLog, list[dict]]:
    """Ingest with a sliding window of `window_size` turns; the writer fires
    every `k` turns and emits entries for the last `k` turns of the window
    only. The earlier (window_size - k) turns are coref/state context.

    Returns (log, idx, telemetry_per_window).
    """
    log: list[LogEntry] = []
    known: set[str] = {"User"}
    idx: IndexedLog | None = None
    telemetry: list[dict] = []

    n_turns = len(turns)
    fire_no = 0
    # Fire at indices k, 2k, 3k, ... (and finally at n_turns if not aligned).
    # Each fire: target = turns[fire_end-k : fire_end]; window = turns[max(0, fire_end-window_size) : fire_end]
    fire_end = k
    while True:
        fire_end = min(fire_end, n_turns)
        if fire_end <= 0:
            break
        target_lo_idx = max(0, fire_end - k)
        win_lo_idx = max(0, fire_end - window_size)
        window_turns = turns[win_lo_idx:fire_end]
        target_turns = turns[target_lo_idx:fire_end]
        if not target_turns:
            break
        target_turn_lo = target_turns[0][0]

        new_entries, tele = write_window(
            window_turns,
            target_turn_lo,
            target_turns,
            log,
            idx,
            known,
            cache,
            budget,
            max_active_state_size=max_active_state_size,
        )
        for e in new_entries:
            for m in e.mentions:
                if m.startswith("@"):
                    known.add(m[1:])
        log.extend(new_entries)
        tele["fire_no"] = fire_no
        tele["last_turn"] = target_turns[-1][0] if target_turns else None
        tele["n_emitted"] = len(new_entries)
        telemetry.append(tele)

        if fire_no % rebuild_index_every == 0:
            idx = build_index(log, cache, budget)
        fire_no += 1
        if fire_end >= n_turns:
            break
        fire_end += k

    idx = build_index(log, cache, budget)
    return log, idx, telemetry
