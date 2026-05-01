"""AEN-1 CENTERED — K-block writer with a CENTERED sliding window.

Design:
  window = w_past (past context) + K (TARGET block) + w_future (future context)
  Each fire emits entries for K turns at the MIDDLE of the window. Slide by K
  each fire. Cost = N/K LLM calls.

  Example (w_past=7, K=3, w_future=7, total window=17):
    fire 1: window=[1..17],  target={8,9,10}
    fire 2: window=[4..20],  target={11,12,13}
    fire 3: window=[7..23],  target={14,15,16}
    ...

  K=1 is the per-turn extreme (turn t at middle of window [t-w_past..t+w_future]).
  K=window_size + w_future=0 reduces to round16a's last-window sliding writer.

Why centered + asymmetric:
  Natural human conversation lookahead (working memory + conversational repair)
  is ~5-10 turns. A 2x fudge factor on w_future captures gap=10-20 anonymous->
  named pairs, which exceed the usual 7±2 working memory but still fit
  comfortably under the windowed writer's prompt-character budget.

  w_past doesn't need a fudge factor: past is fully observed and committed; the
  only constraint is prompt size.

Reuses round16a's WRITE_PROMPT and write_window — only ingest_turns changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND18 = HERE.parent
RESEARCH = ROUND18.parent
ROUND16A = RESEARCH / "round16a_sliding_window"
ROUND15 = RESEARCH / "round15_active_chains"
ROUND11 = RESEARCH / "round11_writer_stress"
ROUND7 = RESEARCH / "round7"
sys.path.insert(0, str(ROUND16A / "architectures"))
sys.path.insert(0, str(ROUND15 / "architectures"))
sys.path.insert(0, str(ROUND11 / "architectures"))
sys.path.insert(0, str(ROUND7 / "experiments"))

import aen1_simple  # noqa: E402
import aen1_sliding  # noqa: E402
from _common import Budget, Cache  # noqa: E402

# Re-export for callers
LogEntry = aen1_simple.LogEntry
IndexedLog = aen1_simple.IndexedLog
build_index = aen1_simple.build_index
retrieve = aen1_simple.retrieve
answer_question = aen1_simple.answer_question


def ingest_turns(
    turns: list[tuple[int, str]],
    cache: Cache,
    budget: Budget,
    *,
    w_past: int = 7,
    w_future: int = 7,
    k: int = 3,
    rebuild_index_every: int = 4,
    max_active_state_size: int = 100,
) -> tuple[list[LogEntry], IndexedLog, list[dict]]:
    """K-block writer with centered window of past + K + future turns.

    Each fire: target = K turns at the middle; window = w_past turns before +
    K target + w_future turns after. Slide by K. Cost = ceil(N/K) fires.
    Edges are clamped to scenario bounds.

    Returns (log, idx, telemetry_per_fire).
    """
    log: list[LogEntry] = []
    known: set[str] = {"User"}
    idx: IndexedLog | None = None
    telemetry: list[dict] = []

    n_turns = len(turns)

    fire_no = 0
    target_lo = 0  # 0-indexed start of target block
    while target_lo < n_turns:
        target_hi = min(n_turns, target_lo + k)  # exclusive end of target block
        win_lo = max(0, target_lo - w_past)
        win_hi = min(n_turns, target_hi + w_future)
        window_turns = turns[win_lo:win_hi]
        target_turns = turns[target_lo:target_hi]
        if not target_turns:
            break
        target_turn_lo = target_turns[0][0]

        new_entries, tele = aen1_sliding.write_window(
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
        tele["last_turn"] = target_turns[-1][0]
        tele["n_emitted"] = len(new_entries)
        tele["w_past_actual"] = target_lo - win_lo
        tele["w_future_actual"] = win_hi - target_hi
        tele["k_actual"] = target_hi - target_lo
        telemetry.append(tele)

        if fire_no % rebuild_index_every == 0:
            idx = build_index(log, cache, budget)
        fire_no += 1
        target_lo = target_hi

    idx = build_index(log, cache, budget)
    return log, idx, telemetry
