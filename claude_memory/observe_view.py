#!/usr/bin/env python3
"""Read `<home>/observability.jsonl` and report what retrieval actually did.

The log exists to answer questions that cannot be answered by reasoning, so this
prints distributions rather than averages — a mean hides exactly the thing a
threshold would be set from.

Four questions it is built to settle:

  1. How often does ambient recall fire at all? A gate is only worth building if
     the silent fraction is small enough to be worth changing.
  2. Does it fire on turns with nothing to retrieve for? Cue length is the
     available proxy for "conversational", and the split by it is the whole point.
  3. Is any score signal discriminative? The top score is reported beside the
     SPREAD, because the first sample suggested magnitude is not the signal
     ("ok" scored 0.5462 against a real question's 0.5359) while flatness might be.
  4. Does expansion return what it was asked for? asked-vs-got catches a window
     eaten by one long event, which is the failure kind filtering exists to fix.

    python3 -m claude_memory.observe_view              # everything
    python3 -m claude_memory.observe_view --event ambient
    python3 -m claude_memory.observe_view --since 2026-08-07
"""

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from claude_memory.wire import MemoryConfig

# Below this many content words a prompt is treated as conversational ("ok",
# "status?", "yes do that"). A proxy, not a classifier — but the split only has to
# be good enough to show whether the two populations behave differently at all.
_CONVERSATIONAL_WORDS = 4
_MIN_ROWS = 20


def _read(path: Path, event: str | None, since: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a partial trailing write; skip it
                if event and row.get("event") != event:
                    continue
                if since and str(row.get("ts", "")) < since:
                    continue
                rows.append(row)
    except OSError:
        return []
    return rows


def _pct(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(fraction * len(ordered)))
    return ordered[index]


def _distribution(label: str, values: list[float], places: int = 4) -> str:
    if not values:
        return f"  {label:22s} (none)"
    return (
        f"  {label:22s} n={len(values):<5d} "
        f"p10={_pct(values, 0.10):.{places}f}  "
        f"median={statistics.median(values):.{places}f}  "
        f"p90={_pct(values, 0.90):.{places}f}  "
        f"max={max(values):.{places}f}"
    )


def report_ambient(rows: list[dict[str, Any]]) -> None:
    """Fire rate and score shape, split by whether the cue looked conversational."""
    if not rows:
        return
    print(f"\n=== ambient ({len(rows)} turns) ===")
    fired = [r for r in rows if r.get("injected")]
    print(
        f"  injected on {len(fired)}/{len(rows)} turns ({100 * len(fired) / len(rows):.0f}%)"
    )
    if len(rows) < _MIN_ROWS:
        print(f"  (fewer than {_MIN_ROWS} rows — treat everything below as anecdote)")

    short = [r for r in rows if r.get("cue_words", 0) < _CONVERSATIONAL_WORDS]
    long_ = [r for r in rows if r.get("cue_words", 0) >= _CONVERSATIONAL_WORDS]
    for label, group in (("conversational cues", short), ("contentful cues", long_)):
        if not group:
            continue
        hit = sum(1 for r in group if r.get("injected"))
        print(
            f"  {label:20s} {len(group):4d} turns, injected {hit:4d} "
            f"({100 * hit / len(group):3.0f}%)"
        )
    # The question a gate would need answered: do the two populations separate on
    # ANY recorded quantity? If these lines overlap, no threshold on them exists.
    print("  score shape, by cue class:")
    for label, group in (("conversational top", short), ("contentful top", long_)):
        print(
            _distribution(
                label,
                [r["scores"]["top"] for r in group if r.get("scores", {}).get("n")],
            )
        )
    for label, group in (
        ("conversational spread", short),
        ("contentful spread", long_),
    ):
        print(
            _distribution(
                label,
                [r["scores"]["spread"] for r in group if r.get("scores", {}).get("n")],
            )
        )
    print(
        _distribution(
            "injected chars", [float(r.get("chars", 0)) for r in fired], places=0
        )
    )


def report_search(rows: list[dict[str, Any]]) -> None:
    """Deliberate searches: how many were novel, and how saturated the store is."""
    if not rows:
        return
    print(f"\n=== search ({len(rows)} calls) ===")
    saturated = sum(1 for r in rows if r.get("saturated"))
    print(f"  saturated (hits but none new): {saturated}/{len(rows)}")
    print(
        _distribution(
            "top score",
            [r["scores"]["top"] for r in rows if r.get("scores", {}).get("n")],
        )
    )
    print(
        _distribution(
            "spread",
            [r["scores"]["spread"] for r in rows if r.get("scores", {}).get("n")],
        )
    )
    print(
        _distribution(
            "new hits", [float(r.get("new_count", 0)) for r in rows], places=1
        )
    )
    filtered = sum(1 for r in rows if r.get("filters"))
    print(f"  used a filter: {filtered}/{len(rows)}")


def report_expand(rows: list[dict[str, Any]]) -> None:
    """Yield: a window returning far fewer segments than asked was eaten."""
    if not rows:
        return
    print(f"\n=== expand ({len(rows)} calls) ===")
    yields = [
        r["got"] / r["asked"]
        for r in rows
        if r.get("asked") and r.get("got") is not None
    ]
    print(_distribution("got/asked", yields, places=2))
    thin = sum(1 for y in yields if y < 0.5)
    print(f"  returned under half of what was asked: {thin}/{len(yields)}")
    print(
        _distribution(
            "events per window", [float(r.get("events", 0)) for r in rows], places=1
        )
    )
    kinds: Counter[str] = Counter()
    for row in rows:
        for source in row.get("sources") or []:
            kinds[source] += 1
    if kinds:
        print(
            "  sources actually returned: "
            + ", ".join(f"{k}={v}" for k, v in kinds.most_common())
        )
    used = sum(1 for r in rows if r.get("kinds"))
    print(f"  called with an explicit kinds list: {used}/{len(rows)}")


def report_ingest(rows: list[dict[str, Any]]) -> None:
    """Capture volume, and whether any session is falling behind."""
    if not rows:
        return
    print(f"\n=== ingest ({len(rows)} calls) ===")
    print(
        _distribution(
            "lines per call",
            [float(r.get("to_line", 0) - r.get("from_line", 0)) for r in rows],
            places=0,
        )
    )
    print(
        _distribution(
            "events per call", [float(r.get("events", 0)) for r in rows], places=1
        )
    )
    sessions = {r.get("session") for r in rows}
    print(f"  distinct sessions: {len(sessions)}")


def main(argv: list[str] | None = None) -> int:
    """Print the report for the configured home."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument(
        "--event", default=None, help="ambient | search | expand | ingest"
    )
    parser.add_argument("--since", default=None, help="ISO date, e.g. 2026-08-07")
    args = parser.parse_args(argv)

    config = MemoryConfig.load()
    path = args.log or (config.home / "observability.jsonl")
    if not path.exists():
        print(f"No log at {path}.", file=sys.stderr)
        print(
            'Enable it with {"observe": true} in the home config, then take some turns.',
            file=sys.stderr,
        )
        return 1

    rows = _read(path, args.event, args.since)
    if not rows:
        print(f"{path} has no matching records.", file=sys.stderr)
        return 1

    by_event: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_event.setdefault(str(row.get("event", "?")), []).append(row)
    span = f"{rows[0].get('ts', '?')} .. {rows[-1].get('ts', '?')}"
    print(f"{path}\n{len(rows)} records   {span}")
    print("  " + "  ".join(f"{k}={len(v)}" for k, v in sorted(by_event.items())))

    report_ambient(by_event.get("ambient", []))
    report_search(by_event.get("search", []))
    report_expand(by_event.get("expand", []))
    report_ingest(by_event.get("ingest", []))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
