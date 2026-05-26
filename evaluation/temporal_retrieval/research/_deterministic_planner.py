"""Prototype deterministic planner: rule-based parser that handles the
common temporal-query patterns without an LLM call.

Goal: see what fraction of the 214 cached queries this planner can match
exactly to the LLM planner's output. Whatever it covers cleanly is
LLM-free at query time.

This is an offline analysis tool, not production code. Production would
need more polishing (better tokenization, fuzzy match, larger gazetteer,
etc.).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Direct calendar phrases (with year). Order matters: more specific first.
_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
CALENDAR_PATTERNS = [
    # Month Day, Year   ("April 5, 2015", "March 15, 2024")
    (re.compile(
        rf"\b(?:early |late |mid-?)?({_MONTHS})\s+(\d{{1,2}}),?\s+(\d{{4}})\b",
        re.I,
    ), lambda m: f"{m.group(1).capitalize()} {m.group(2)}, {m.group(3)}"),
    # Q1 2023, Q4 2024
    (re.compile(r"\b(Q[1-4])\s+(\d{4})\b", re.I),
     lambda m: f"{m.group(1).upper()} {m.group(2)}"),
    # H1 2024, H2 2024
    (re.compile(r"\b(H[12])\s+(\d{4})\b", re.I),
     lambda m: f"{m.group(1).upper()} {m.group(2)}"),
    # March 2024, early/late/mid Month YYYY
    (re.compile(
        rf"\b(?:early |late |mid-?)?({_MONTHS})\s+(\d{{4}})\b",
        re.I,
    ), lambda m: f"{m.group(1).capitalize()} {m.group(2)}"),
    # spring 2024, summer 2024
    (re.compile(r"\b(spring|summer|fall|autumn|winter)\s+(\d{4})\b", re.I),
     lambda m: f"{m.group(1).lower()} {m.group(2)}"),
    # Decades: "the 2010s", "2010s"
    (re.compile(r"\b(?:the\s+)?(19\d0|20\d0)s\b", re.I),
     lambda m: f"{m.group(1)}s"),
    # 2024 alone (4-digit year)
    (re.compile(r"\b(19\d{2}|20\d{2})\b"), lambda m: m.group(1)),
]

# Weekday deictic ("next Tuesday", "last Friday", "this Monday")
WEEKDAY_DEICTIC = re.compile(
    r"\b(last|this|next|coming)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.I,
)

# Personal era / life-stage scopes ("in college", "in high school", "as a kid",
# "during my PhD"). These the LLM treats as deictic-era leaves.
PERSONAL_ERA = re.compile(
    r"\b(in (?:college|high school|elementary school|grad school|middle school)"
    r"|during (?:my (?:phd|undergrad|degree|childhood)|college)"
    r"|as a (?:kid|child|teenager|teen)"
    r"|back in (?:college|high school|the day|my (?:teens|twenties|thirties)))\b",
    re.I,
)

# Recurrence triggers ("every X", "every other X", "weekly", "monthly")
RECURRENCE_TRIGGER = re.compile(
    r"\b(every (?:other )?\w+(?:\s+\w+)?"
    r"|(?:every|each) (?:day|week|month|year|quarter|monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"|(?:weekly|monthly|yearly|annually|daily|quarterly))\b",
    re.I,
)

# Hyphenated anaphoric: post-X, pre-X
HYPHEN_ANAPHORIC = re.compile(r"\b(post|pre)[- ](\w+(?:[- ]\w+)?)\b", re.I)

# Framing words that mean "not scoping" → emit expr=[].
SKIP_FRAMING = re.compile(
    r"\b("
    r"notes from|lessons from|outcomes of|aftermath of|review of|recap of|"
    r"look back at|looking back at|thinking back to|story behind|story of|"
    r"how did .* go|what was .* like|when did .* happen"
    r")\b",
    re.I,
)

# Topical references (no direction cue) → skip.
TOPICAL_SHAPE = re.compile(r"\b(what did .* say about|who attended|what's? the .* of|tell me about)\b", re.I)

# Generic time vocab non-deictic → skip.
GENERIC_TIME = re.compile(
    r"\bduring the day\b|\bmorning routine\b|\bfuture of\b|\bpast and future\b",
    re.I,
)

# Direction cues to NOT_IN
NOT_IN_CUES = re.compile(r"\b(not in|outside of|outside|excluding|except)\b", re.I)

# Direction cues for OPEN-ENDED after/before/since/until
AFTER_CUES = re.compile(r"\b(after|since|post)\b", re.I)
BEFORE_CUES = re.compile(r"\b(before|until|prior to)\b", re.I)

# Extremum cues
LATEST_CUES = re.compile(r"\b(latest|most recent|recently|recent)\b", re.I)
EARLIEST_CUES = re.compile(r"\b(earliest|first ever)\b", re.I)

# Deictic relative phrases that resolve via downstream extractor (KEEP AS-IS).
DEICTIC_PHRASES = re.compile(
    r"\b("
    r"yesterday|today|tomorrow|"
    r"last (?:week|month|year|quarter)|this (?:week|month|year|quarter|morning|afternoon|evening)|"
    r"next (?:week|month|year|quarter)|"
    r"(?:a|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(?:days?|weeks?|months?|years?|quarters?)\s+ago|"
    r"the (?:last|past|next|coming)\s+\d+\s+(?:days?|weeks?|months?|years?|hours?)|"
    r"the (?:last|past|next|coming)\s+(?:day|week|month|year|hour|24 hours|few days|few weeks)"
    r")\b",
    re.I,
)

# Anaphoric event pattern: direction cue + "the X" where X is a noun phrase
# without a year/calendar phrase. e.g. "after the launch", "since the redesign shipped"
ANAPHORIC_PATTERN = re.compile(
    r"\b(after|since|before|until|during)\s+(the\s+\w+(?:\s+\w+)?)\b", re.I
)


def has_calendar_phrase(query: str) -> bool:
    for pat, _ in CALENDAR_PATTERNS:
        if pat.search(query):
            return True
    return False


def find_calendar_phrases(query: str) -> list[tuple[str, int, int]]:
    """Return list of (phrase, start, end) for all calendar matches, dedup."""
    out = []
    seen_spans: set[tuple[int, int]] = set()
    for pat, normalize in CALENDAR_PATTERNS:
        for m in pat.finditer(query):
            span = (m.start(), m.end())
            # Avoid double-counting if a year matched inside a month+year span
            overlaps = any(s <= span[0] < e or s < span[1] <= e for s, e in seen_spans)
            if overlaps:
                continue
            seen_spans.add(span)
            out.append((normalize(m), span[0], span[1]))
    return sorted(out, key=lambda t: t[1])


def deterministic_plan(query: str) -> dict:
    """Best-effort rule-based planner. Returns {"expr": [...], "extremum": ...}."""
    # 1. Framing words → skip
    if SKIP_FRAMING.search(query) or TOPICAL_SHAPE.search(query) or GENERIC_TIME.search(query):
        return {"expr": [], "extremum": None}

    # 2. Extremum
    extremum = None
    if LATEST_CUES.search(query):
        extremum = "latest"
    elif EARLIEST_CUES.search(query):
        extremum = "earliest"

    # 3. "recently" / "recent" alone (no calendar) → empty + extremum=latest
    if extremum == "latest" and not has_calendar_phrase(query) and not DEICTIC_PHRASES.search(query):
        # Pure recency intent
        return {"expr": [], "extremum": "latest"}

    # 4. Calendar phrases
    cal = find_calendar_phrases(query)
    leaves = []
    if cal:
        # Detect direction in surrounding window per phrase
        for phrase, start, end in cal:
            # Look at the 30 chars before the phrase for direction cues
            ctx = query[max(0, start - 30):start].lower()
            # Check not_in (priority)
            if NOT_IN_CUES.search(ctx) or NOT_IN_CUES.search(query[max(0, start - 30):end].lower()):
                direction = "not_in"
            elif AFTER_CUES.search(ctx):
                direction = "after"
            elif BEFORE_CUES.search(ctx):
                direction = "before"
            else:
                direction = "in"
            leaves.append({"phrase": phrase, "direction": direction})

    # 5. Deictic phrases
    for m in DEICTIC_PHRASES.finditer(query):
        phrase = m.group(0).lower()
        ctx = query[max(0, m.start() - 30):m.start()].lower()
        if NOT_IN_CUES.search(ctx):
            direction = "not_in"
        elif AFTER_CUES.search(ctx):
            direction = "after"
        elif BEFORE_CUES.search(ctx):
            direction = "before"
        else:
            direction = "in"
        # Dedup: if a calendar phrase already covers this region, skip
        if any(start <= m.start() < end for _, start, end in cal):
            continue
        leaves.append({"phrase": phrase, "direction": direction})

    # 6. Anaphoric event references
    for m in ANAPHORIC_PATTERN.finditer(query):
        cue = m.group(1).lower()
        target = m.group(2).strip().lower()
        # Skip if this overlaps a calendar phrase
        if any(start <= m.start(2) < end for _, start, end in cal):
            continue
        direction = {
            "after": "after",
            "since": "after",
            "before": "before",
            "until": "before",
            "during": "in",
        }[cue]
        leaves.append({"phrase": target, "direction": direction})

    if not leaves:
        return {"expr": [], "extremum": extremum}

    # Default: single clause (AND). OR detection skipped in prototype.
    return {"expr": [leaves], "extremum": extremum}


def plans_match(det: dict, llm: dict) -> str:
    """Return EXACT / SHAPE / DIFF."""
    if det == llm:
        return "EXACT"
    det_expr = det.get("expr", [])
    llm_expr = llm.get("expr", [])
    if not det_expr and not llm_expr:
        # Both empty; check extremum
        if det.get("extremum") == llm.get("extremum"):
            return "EXACT"
        return "DIFF_EXTREMUM"
    if (not det_expr) != (not llm_expr):
        return "DIFF_SHAPE"
    # Both non-empty; compare directions sets
    det_dirs = sorted([l["direction"] for c in det_expr for l in c])
    llm_dirs = sorted([l["direction"] for c in llm_expr for l in c])
    if det_dirs == llm_dirs:
        det_phrases = sorted([l["phrase"].lower() for c in det_expr for l in c])
        llm_phrases = sorted([l["phrase"].lower() for c in llm_expr for l in c])
        if det_phrases == llm_phrases:
            return "EXACT"
        return "DIFF_PHRASE"
    return "DIFF_DIR"


def main() -> None:
    # Same hash logic as planner.py
    import hashlib
    MODEL = "gpt-5-mini"
    PROMPT_VERSION = "v4.1"

    def cache_key(query: str, ref_time: str) -> str:
        h = hashlib.sha256()
        h.update(MODEL.encode())
        h.update(b"|")
        h.update(PROMPT_VERSION.encode())
        h.update(b"|")
        h.update(query.encode())
        h.update(b"|")
        h.update(ref_time.encode())
        return h.hexdigest()

    cache = json.loads(
        Path("temporal_retrieval/cache/planner/llm_plan_cache.json").read_text()
    )
    data_dir = Path("temporal_extraction/data")
    all_queries = []
    for qf in sorted(data_dir.glob("*_queries.jsonl")):
        bench = qf.name.replace("_queries.jsonl", "")
        for line in qf.read_text().splitlines():
            if not line.strip():
                continue
            q = json.loads(line)
            all_queries.append((bench, q["text"], q.get("ref_time", "")))

    matched = []
    for bench, qt, rt in all_queries:
        k = cache_key(qt, rt)
        if k in cache:
            matched.append((bench, qt, rt, cache[k]))

    print(f"Comparing {len(matched)} cached queries to deterministic planner")
    print()

    verdict_counts: Counter[str] = Counter()
    per_bench_verdict: dict[str, Counter[str]] = {}
    diffs: list[tuple] = []
    for bench, qt, _rt, llm_plan in matched:
        det_plan = deterministic_plan(qt)
        verdict = plans_match(det_plan, llm_plan)
        verdict_counts[verdict] += 1
        per_bench_verdict.setdefault(bench, Counter())[verdict] += 1
        if verdict != "EXACT" and len(diffs) < 25:
            diffs.append((bench, qt, det_plan, llm_plan, verdict))

    print(f"{'verdict':<20s} count   pct")
    total = sum(verdict_counts.values())
    for v, n in verdict_counts.most_common():
        print(f"{v:<20s} {n:5d}  {100 * n / total:5.1f}%")

    print()
    print("Per-bench breakdown:")
    for bench, counts in sorted(per_bench_verdict.items()):
        total_b = sum(counts.values())
        exact = counts.get("EXACT", 0)
        print(f"  {bench:25s}  EXACT {exact:3d}/{total_b:<3d}  ({100*exact/total_b:5.1f}%)")

    print()
    print("Sample non-EXACT cases (up to 25):")
    for bench, qt, det, llm, v in diffs:
        print(f"  [{v}] [{bench}] {qt!r}")
        print(f"      det:  {det}")
        print(f"      llm:  {llm}")


if __name__ == "__main__":
    main()
