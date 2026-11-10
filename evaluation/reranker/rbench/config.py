"""What the evaluation reports at: the values it varies or summarizes over. Change them here,
or per run with command-line flags (cli.py: --budgets, --limits, --readings); report code
reads them from EVAL rather than hard-coding them.

  budgets      reader budgets (tokens) the charts drawn at fixed budgets and the tables' focused
               columns (deltas, settings, query types) use; 500-4,000 is the realistic range
  columns      reader budgets of the coverage tables' wider sweep
  limits       vector search limits shown, besides the deepest scored (EventMemory's default
               is 20)
  readings     coverage readings the single-reading charts are drawn for: every, first,
               every_touch, first_touch (metrics.py), share, share_touch (the share of a
               question's evidence turns read: report/readings.py). Every is the headline (all
               of a question's evidence is needed)
  expansions   expand_context values replayed by the history simulation (EventMemory's default
               is 0). Rerank-then-expand is free at all of them; expand-then-rerank exists only
               where a run scored the windows (grid.py)
  token_top    where the reader-token axes end, and the budget every pool is deep enough to fill
               (depth.py: reading a pool does not run out before it). The user's rule: as high
               as the price allows, not below 4,000 without asking. 16,000 priced the 8B at
               about $8.35 for core, sizes and cut together (2026-11-09: 4,000 $1.97, 8,000
               $3.87, 12,000 $5.64; pools from 8 LongMemEval-M histories)
  cost_chunking, cost_expand  the setting the cost comparison holds fixed (cost_saving.png and
               report section 7): both sides at this chunking (None: EventMemory's default
               segment length) and expand_context, so only the vector search limit varies, as
               a service would run (the user's decision, 2026-11-09). Other values draw other
               charts; re-picking among every configuration per target made the chart jagged
  hide         rerankers left out of reports and charts though they have scores. The raw-pairs
               format test (q06-local-raw) is settled (the model card's prompt won: METHODOLOGY
               section 3), so it is hidden; `--hide ""` shows it (e.g. to redraw format-test)
  size_tolerance  how far apart (relative) two measured unit sizes may be and still be reported
               as matched, where a comparison pairs units by size (the expansion comparison);
               beyond it the report flags the pair as not size-matched
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Evaluation:
    budgets: tuple[int, ...] = (500, 1000, 2000, 4000)
    columns: tuple[int, ...] = (250, 500, 1000, 2000, 4000, 8000)
    limits: tuple[int, ...] = (10, 20, 50)
    readings: tuple[str, ...] = ("every",)
    expansions: tuple[int, ...] = (1, 2, 3, 6)
    token_top: int = 16000
    size_tolerance: float = 0.25
    cost_chunking: str | None = None
    cost_expand: int = 0
    hide: tuple[str, ...] = ("q06-local-raw",)


EVAL = Evaluation()

# The simple chunker's length fitted to TextSegmenter's measured mean segment (sizes.py:
# `python -m rbench fit-cut`, 2026-11-09, over every question). seg500 averages 299.2
# characters per segment on LongMemEval-S (cut354: 299.1) and 299.1 on LongMemEval-M (cut355:
# 299.4); TextSegmenter splits at natural boundaries under its limit, so the cut is shorter.
CUT = {("longmemeval_s", "seg500"): 354, ("longmemeval_m", "seg500"): 355}


def cut_for(dataset: str, chunking: str) -> str:
    """The simple chunker size-matched to a TextSegmenter chunking (seg<N>): the fitted cut<M>,
    or cut<N> where no fit is recorded."""
    return f"cut{CUT.get((dataset, chunking), chunking[3:])}"


READINGS = ("first", "every", "first_touch", "every_touch", "share", "share_touch")  # report/readings.py


def configure(**changes) -> None:
    """Override EVAL's values for this process (None leaves one as it is)."""
    global EVAL
    changes = {k: v for k, v in changes.items() if v is not None}
    unknown = set(changes.get("readings", ())) - set(READINGS)
    if unknown:
        raise SystemExit(f"unknown reading(s) {sorted(unknown)}; choose from {list(READINGS)}")
    EVAL = replace(EVAL, **changes)
