"""What the evaluation reports at: the values it varies or summarizes over. Change them here,
or per run with command-line flags (cli.py: --budgets, --limits, --readings); report code
reads them from EVAL rather than hard-coding them.

  budgets      reader budgets (tokens) of the tables and of the charts drawn at fixed budgets;
               500-4,000 is the realistic range
  limits       vector search limits shown, besides the deepest scored (EventMemory's default
               is 20)
  readings     coverage readings the single-reading charts are drawn for: every, first,
               every_touch, first_touch (metrics.py). Every is the headline (all of a
               question's evidence is needed)
  expansions   expand_context values replayed by the history simulation (EventMemory's default
               is 0). Rerank-then-expand is free at all of them; expand-then-rerank exists only
               where a run scored the windows (grid.py)
  token_top    where the reader-token axes end (the data runs to metrics.CURVE_BUDGETS' last)
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Evaluation:
    budgets: tuple[int, ...] = (500, 1000, 2000, 4000)
    limits: tuple[int, ...] = (10, 20, 50)
    readings: tuple[str, ...] = ("every",)
    expansions: tuple[int, ...] = (1, 2, 3, 6)
    token_top: int = 8000


EVAL = Evaluation()
READINGS = ("first", "every", "first_touch", "every_touch")


def configure(**changes) -> None:
    """Override EVAL's values for this process (None leaves one as it is)."""
    global EVAL
    changes = {k: v for k, v in changes.items() if v is not None}
    unknown = set(changes.get("readings", ())) - set(READINGS)
    if unknown:
        raise SystemExit(f"unknown reading(s) {sorted(unknown)}; choose from {list(READINGS)}")
    EVAL = replace(EVAL, **changes)
