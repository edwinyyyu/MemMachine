#!/usr/bin/env python3
"""Read `<home>/observability.jsonl` and report what retrieval actually did.

The log exists to answer questions that cannot be answered by reasoning, so this
prints distributions rather than averages — a mean hides exactly the thing a
threshold would be set from.

Five questions it is built to settle:

  0. Is each tool used at all, and does a search lead anywhere? Reach (distinct
     conversations) rather than call count, because one enthusiastic session
     otherwise speaks for the population; and expansions seeded by a memory the
     system surfaced, which is the closest available signal that a search
     returned something worth reading more of.
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


def report_usefulness(by_event: dict[str, list[dict[str, Any]]]) -> None:
    """Whether each tool is earning its place, and whether searches lead anywhere.

    Two things are reported and they are not the same. REACH — how many distinct
    conversations a tool was used in — is the honest measure of whether a tool is
    used at all; raw call counts let one enthusiastic session speak for the
    population. FOLLOW-THROUGH is the closest thing to a usefulness signal
    available without asking the model: an expansion seeded by a memory this
    system surfaced means a search returned something worth reading more of.

    Records written before a field existed simply lack it, so every rate here is
    reported over the rows that carry the field, with that denominator shown. A
    percentage over a mixed population would be the more useful-looking number and
    the wrong one.
    """
    print("\n=== tool usage ===")
    # "attributed" is the denominator for the reach column, not decoration: records
    # written before a field existed carry no session, and reporting reach against
    # the full call count would read as "603 calls in 1 conversation".
    print(f"  {'tool':10s} {'calls':>7s} {'attributed':>11s} {'conversations':>14s}")
    for tool in ("search", "expand", "outline", "annotate", "demote", "ambient"):
        rows = by_event.get(tool, [])
        attributed = [r for r in rows if r.get("session")]
        reach = len({r.get("session") for r in attributed})
        note = "   never called" if not rows else ""
        print(f"  {tool:10s} {len(rows):7,d} {len(attributed):11,d} {reach:14,d}{note}")

    expands = [r for r in by_event.get("expand", []) if "from_surfaced" in r]
    if expands:
        followed = sum(1 for r in expands if r.get("from_surfaced"))
        print(
            f"\n  expansions seeded by a memory this system surfaced: "
            f"{followed}/{len(expands)} ({100 * followed / len(expands):.0f}%)"
        )
        print("    the rest were seeded from a roster handle or the user")

    # A cue searched twice in one conversation means the first search did not
    # settle it. This is the one negative signal the log can see directly.
    seen: set[tuple[str, str]] = set()
    repeats = considered = 0
    for row in by_event.get("search", []):
        key = (str(row.get("session", "")), str(row.get("cue", "")))
        if not key[0] or not key[1]:
            continue
        considered += 1
        if key in seen:
            repeats += 1
        seen.add(key)
    if considered:
        print(
            f"  searches repeating a cue already used in that conversation: "
            f"{repeats}/{considered} ({100 * repeats / considered:.0f}%)"
        )

    outlines = by_event.get("outline", [])
    if outlines:
        own = sum(1 for r in outlines if r.get("own_conversation"))
        print(
            f"  outlines of the CURRENT conversation: {own}/{len(outlines)}"
            " (the rest looked at another)"
        )

    annotates = by_event.get("annotate", [])
    stacked = sum(1 for r in annotates if int(r.get("existing_notes", 0) or 0) > 0)
    if annotates:
        print(
            f"  annotations added to a memory that already had one: "
            f"{stacked}/{len(annotates)}"
        )

    demotes = [r for r in by_event.get("demote", []) if r.get("cue")]
    if demotes:
        by_cue = Counter(str(r.get("cue")) for r in demotes)
        worst = by_cue.most_common(1)[0]
        print(
            f"  demotes: {len(demotes)} over {len(by_cue)} distinct cues"
            f" (most-demoted cue: {worst[1]}x)"
        )


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

    report_usefulness(by_event)
    report_ambient(by_event.get("ambient", []))
    report_search(by_event.get("search", []))
    report_expand(by_event.get("expand", []))
    report_ingest(by_event.get("ingest", []))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
