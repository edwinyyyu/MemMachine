"""Generate synthetic docs + queries + gold for temporal-extraction eval.

Writes:
- data/docs.jsonl
- data/queries.jsonl
- data/gold.jsonl  (per-query {query_id, relevant_doc_ids})

Each doc/query has a ``ref_time`` and a list of gold ``TimeExpression``s
encoded as JSON.  Because we generate deterministically, we can author
ground truth directly.

Critical-case pairs (N days from now at T1 <-> N days ago at T1+2N) are
included per DESIGN.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dateutil.relativedelta import relativedelta
from schema import (
    FuzzyInstant,
    FuzzyInterval,
    Recurrence,
    TimeExpression,
    time_expression_to_dict,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)

NOW = datetime(2026, 4, 23, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers to bracket well-known expression shapes deterministically
# ---------------------------------------------------------------------------
def bracket_day(dt: datetime) -> FuzzyInstant:
    d0 = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return FuzzyInstant(
        earliest=d0,
        latest=d0 + timedelta(days=1),
        best=d0 + timedelta(hours=12),
        granularity="day",
    )


def bracket_month(year: int, month: int) -> FuzzyInstant:
    d0 = datetime(year, month, 1, tzinfo=timezone.utc)
    end = datetime(year + (month == 12), (month % 12) + 1, 1, tzinfo=timezone.utc)
    mid = d0 + (end - d0) / 2
    return FuzzyInstant(earliest=d0, latest=end, best=mid, granularity="month")


def bracket_year(year: int) -> FuzzyInstant:
    d0 = datetime(year, 1, 1, tzinfo=timezone.utc)
    end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    mid = datetime(year, 7, 1, tzinfo=timezone.utc)
    return FuzzyInstant(earliest=d0, latest=end, best=mid, granularity="year")


def bracket_decade(decade_start: int) -> FuzzyInstant:
    d0 = datetime(decade_start, 1, 1, tzinfo=timezone.utc)
    end = datetime(decade_start + 10, 1, 1, tzinfo=timezone.utc)
    mid = datetime(decade_start + 5, 1, 1, tzinfo=timezone.utc)
    return FuzzyInstant(earliest=d0, latest=end, best=mid, granularity="decade")


def bracket_century(century_start: int) -> FuzzyInstant:
    d0 = datetime(century_start, 1, 1, tzinfo=timezone.utc)
    end = datetime(century_start + 100, 1, 1, tzinfo=timezone.utc)
    mid = datetime(century_start + 50, 1, 1, tzinfo=timezone.utc)
    return FuzzyInstant(earliest=d0, latest=end, best=mid, granularity="century")


def bracket_week(day: datetime) -> FuzzyInstant:
    # Week containing day, Mon-Sun
    start = (day - timedelta(days=day.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    end = start + timedelta(days=7)
    mid = start + timedelta(days=3, hours=12)
    return FuzzyInstant(earliest=start, latest=end, best=mid, granularity="week")


# ---------------------------------------------------------------------------
# Gold expression helpers
# ---------------------------------------------------------------------------
def expr_instant(
    surface: str,
    ref_time: datetime,
    instant: FuzzyInstant,
    text: str | None = None,
) -> TimeExpression:
    te = TimeExpression(
        kind="instant",
        surface=surface,
        reference_time=ref_time,
        instant=instant,
    )
    if text is not None:
        idx = text.find(surface)
        if idx >= 0:
            te.span_start = idx
            te.span_end = idx + len(surface)
    return te


def expr_interval(
    surface: str,
    ref_time: datetime,
    start: FuzzyInstant,
    end: FuzzyInstant,
    text: str | None = None,
) -> TimeExpression:
    te = TimeExpression(
        kind="interval",
        surface=surface,
        reference_time=ref_time,
        interval=FuzzyInterval(start=start, end=end),
    )
    if text is not None:
        idx = text.find(surface)
        if idx >= 0:
            te.span_start = idx
            te.span_end = idx + len(surface)
    return te


def expr_recurrence(
    surface: str,
    ref_time: datetime,
    rec: Recurrence,
    text: str | None = None,
) -> TimeExpression:
    te = TimeExpression(
        kind="recurrence",
        surface=surface,
        reference_time=ref_time,
        recurrence=rec,
    )
    if text is not None:
        idx = text.find(surface)
        if idx >= 0:
            te.span_start = idx
            te.span_end = idx + len(surface)
    return te


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
@dataclass
class Doc:
    doc_id: str
    text: str
    ref_time: datetime
    gold_expressions: list[TimeExpression] = field(default_factory=list)


@dataclass
class Query:
    query_id: str
    text: str
    ref_time: datetime
    gold_expressions: list[TimeExpression] = field(default_factory=list)


def build_docs_and_queries() -> tuple[list[Doc], list[Query], list[tuple[str, str]]]:
    """Return (docs, queries, critical_pairs)."""
    docs: list[Doc] = []
    queries: list[Query] = []
    # Doc ref_times distributed around NOW
    # ------------------------------------------------------------------
    # 1. Absolute date, recent (5 docs)
    # ------------------------------------------------------------------
    for i, (y, m, d, story) in enumerate(
        [
            (2026, 3, 15, "I visited the aquarium"),
            (2025, 12, 25, "we opened presents by the tree"),
            (2026, 1, 7, "the team shipped the beta"),
            (2025, 9, 1, "school started for the kids"),
            (2026, 4, 1, "we hosted an open house"),
        ]
    ):
        surface_date = datetime(y, m, d).strftime("%B %-d, %Y")
        text = f"On {surface_date} {story}."
        ref = NOW - relativedelta(days=5 * (i + 1))
        doc = Doc(f"doc_abs_recent_{i}", text, ref)
        doc.gold_expressions.append(
            expr_instant(
                surface_date,
                ref,
                bracket_day(datetime(y, m, d, tzinfo=timezone.utc)),
                text,
            )
        )
        docs.append(doc)

    # 2. Absolute date, distant (3 docs)
    for i, (y, story) in enumerate(
        [
            (1987, "grandma moved to Chicago"),
            (1969, "humans landed on the moon"),
            (1776, "the colonies declared independence"),
        ]
    ):
        surface = f"In {y}"
        text = f"{surface}, {story}."
        ref = NOW - relativedelta(months=i)
        doc = Doc(f"doc_abs_distant_{i}", text, ref)
        doc.gold_expressions.append(expr_instant(str(y), ref, bracket_year(y), text))
        docs.append(doc)

    # 3. Fuzzy decade (3 docs)
    for i, (surf, dec_start, story) in enumerate(
        [
            ("the 90s", 1990, "we used to record songs onto cassette tapes"),
            ("the 2010s", 2010, "smartphones reshaped daily routines"),
            ("the 60s", 1960, "my dad rode freight trains across the country"),
        ]
    ):
        text = f"Back in {surf} {story}."
        ref = NOW - relativedelta(weeks=i)
        doc = Doc(f"doc_decade_{i}", text, ref)
        doc.gold_expressions.append(
            expr_instant(surf, ref, bracket_decade(dec_start), text)
        )
        docs.append(doc)

    # 4. Relative recent (5 docs) - uses each doc's ref_time
    rel_cases = [
        ("Yesterday", -1, "I grabbed coffee with Priya"),
        ("Last week", -7, "we finished the renovation on the kitchen"),
        ("Two weeks ago", -14, "we signed the lease on the new office"),
        ("Three days ago", -3, "the puppy finally slept through the night"),
        ("A month ago", -30, "I started the couch-to-5k program"),
    ]
    for i, (surf, delta_days, story) in enumerate(rel_cases):
        ref = NOW - relativedelta(days=3 * (i + 1))
        target = ref + timedelta(days=delta_days)
        text = f"{surf} {story}."
        doc = Doc(f"doc_rel_recent_{i}", text, ref)
        # Bracket shape depends on the surface
        if surf.lower() == "yesterday":
            gi = bracket_day(target)
        elif surf.lower() == "last week":
            gi = bracket_week(target)
        elif surf.lower() == "a month ago":
            # bracket_month is awkward; just a day bracket of target
            d0 = target.replace(hour=0, minute=0, second=0, microsecond=0)
            gi = FuzzyInstant(
                earliest=d0 - timedelta(days=1),
                latest=d0 + timedelta(days=2),
                best=d0 + timedelta(hours=12),
                granularity="day",
            )
        else:
            # N <days|weeks> ago => day granularity centered on target
            d0 = target.replace(hour=0, minute=0, second=0, microsecond=0)
            gi = FuzzyInstant(
                earliest=d0,
                latest=d0 + timedelta(days=1),
                best=d0 + timedelta(hours=12),
                granularity="day",
            )
        doc.gold_expressions.append(expr_instant(surf, ref, gi, text))
        docs.append(doc)

    # 5. Relative distant (3 docs)
    rel_distant_cases = [
        ("about 20 years ago", 20, "decade", "we bought our first house"),
        ("a few years ago", 3, "year", "I visited Tokyo"),
        ("a couple decades back", 20, "decade", "the town had no stoplights"),
    ]
    for i, (surf, n, gran, story) in enumerate(rel_distant_cases):
        ref = NOW - relativedelta(weeks=2 * (i + 1))
        if gran == "decade":
            target = ref - relativedelta(years=n)
            span_lo = ref - relativedelta(years=n + 5)
            span_hi = ref - relativedelta(years=max(n - 5, 1))
            gi = FuzzyInstant(
                earliest=span_lo,
                latest=span_hi,
                best=target,
                granularity="decade",
            )
        else:
            # "a few years ago" => [ref-5y, ref-2y]
            span_lo = ref - relativedelta(years=5)
            span_hi = ref - relativedelta(years=2)
            gi = FuzzyInstant(
                earliest=span_lo,
                latest=span_hi,
                best=ref - relativedelta(years=3),
                granularity="year",
            )
        text = f"It was {surf} that {story}."
        doc = Doc(f"doc_rel_distant_{i}", text, ref)
        doc.gold_expressions.append(expr_instant(surf, ref, gi, text))
        docs.append(doc)

    # 6. Explicit interval (3 docs)
    interval_cases = [
        (
            datetime(2026, 3, 5, tzinfo=timezone.utc),
            datetime(2026, 3, 12, tzinfo=timezone.utc),
            "I was in Lisbon",
        ),
        (
            datetime(2025, 11, 1, tzinfo=timezone.utc),
            datetime(2025, 11, 15, tzinfo=timezone.utc),
            "we took a long road trip through Utah",
        ),
        (
            datetime(2026, 2, 10, tzinfo=timezone.utc),
            datetime(2026, 2, 20, tzinfo=timezone.utc),
            "I was on call every night",
        ),
    ]
    for i, (start, end, story) in enumerate(interval_cases):
        sname = start.strftime("%B %-d")
        ename = end.strftime("%B %-d, %Y")
        surface = f"from {sname} to {ename}"
        text = f"{surface} {story}."
        ref = NOW - relativedelta(days=10 * (i + 1))
        doc = Doc(f"doc_interval_{i}", text, ref)
        doc.gold_expressions.append(
            expr_interval(
                surface,
                ref,
                bracket_day(start),
                bracket_day(end),
                text,
            )
        )
        docs.append(doc)

    # 7. Recurrence, simple (3 docs)
    recs = [
        (
            "every Thursday",
            "FREQ=WEEKLY;BYDAY=TH",
            datetime(2024, 1, 4, 19, 0, tzinfo=timezone.utc),
            "I have book club",
        ),
        (
            "every Monday at 7am",
            "FREQ=WEEKLY;BYDAY=MO;BYHOUR=7;BYMINUTE=0",
            datetime(2024, 1, 1, 7, 0, tzinfo=timezone.utc),
            "I run with the club",
        ),
        (
            "every day at noon",
            "FREQ=DAILY;BYHOUR=12;BYMINUTE=0",
            datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
            "I check in with the team",
        ),
    ]
    for i, (surf, rrule, dtstart, story) in enumerate(recs):
        text = f"{story.capitalize()} {surf}."
        ref = NOW - relativedelta(days=i + 1)
        rec = Recurrence(
            rrule=rrule, dtstart=bracket_day(dtstart), until=None, exdates=[]
        )
        doc = Doc(f"doc_rec_simple_{i}", text, ref)
        doc.gold_expressions.append(expr_recurrence(surf, ref, rec, text))
        docs.append(doc)

    # 8. Recurrence with start (2 docs)
    rec_start_cases = [
        (
            "Starting in June 2026, we'll meet monthly on the first Monday",
            "FREQ=MONTHLY;BYDAY=1MO",
            datetime(2026, 6, 1, 9, 0, tzinfo=timezone.utc),
            "Starting in June 2026, we'll meet monthly on the first Monday",
        ),
        (
            "Starting next September we'll do yoga every Sunday morning",
            "FREQ=WEEKLY;BYDAY=SU;BYHOUR=9",
            datetime(2026, 9, 6, 9, 0, tzinfo=timezone.utc),
            "Starting next September we'll do yoga every Sunday morning",
        ),
    ]
    for i, (surf, rrule, dtstart, full) in enumerate(rec_start_cases):
        text = full + "."
        ref = NOW - relativedelta(days=5 + i)
        rec = Recurrence(
            rrule=rrule, dtstart=bracket_day(dtstart), until=None, exdates=[]
        )
        doc = Doc(f"doc_rec_start_{i}", text, ref)
        doc.gold_expressions.append(expr_recurrence(surf, ref, rec, text))
        docs.append(doc)

    # 9. Recurrence with cancellation (2 docs)
    rec_cancel_cases = [
        (
            "every Monday except Jan 15 and Feb 5",
            "FREQ=WEEKLY;BYDAY=MO",
            datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc),
            [
                datetime(2026, 1, 15, tzinfo=timezone.utc),
                datetime(2026, 2, 5, tzinfo=timezone.utc),
            ],
            "I have standup",
        ),
        (
            "every Friday except December 25",
            "FREQ=WEEKLY;BYDAY=FR",
            datetime(2026, 1, 2, 15, 0, tzinfo=timezone.utc),
            [datetime(2026, 12, 25, tzinfo=timezone.utc)],
            "we have pizza night",
        ),
    ]
    for i, (surf, rrule, dtstart, exdates, story) in enumerate(rec_cancel_cases):
        text = f"{story.capitalize()} {surf}."
        ref = NOW - relativedelta(days=15 + i)
        rec = Recurrence(
            rrule=rrule,
            dtstart=bracket_day(dtstart),
            until=None,
            exdates=[bracket_day(d) for d in exdates],
        )
        doc = Doc(f"doc_rec_cancel_{i}", text, ref)
        doc.gold_expressions.append(expr_recurrence(surf, ref, rec, text))
        docs.append(doc)

    # 10. Multiple times per doc (5 docs)
    multi_cases: list[tuple[str, datetime, list[TimeExpression]]] = []
    for i in range(5):
        ref = NOW - relativedelta(days=4 * (i + 1))
        if i == 0:
            text = (
                "Yesterday I met Sam for lunch; then on March 10, 2026 "
                "we visited the museum."
            )
            exprs = [
                expr_instant(
                    "Yesterday",
                    ref,
                    bracket_day(ref - timedelta(days=1)),
                    text,
                ),
                expr_instant(
                    "March 10, 2026",
                    ref,
                    bracket_day(datetime(2026, 3, 10, tzinfo=timezone.utc)),
                    text,
                ),
            ]
        elif i == 1:
            text = "Last month we moved; two weeks ago we finally unpacked."
            one_month_ago = ref - relativedelta(months=1)
            two_weeks_ago = ref - timedelta(days=14)
            exprs = [
                expr_instant(
                    "Last month",
                    ref,
                    bracket_month(one_month_ago.year, one_month_ago.month),
                    text,
                ),
                expr_instant(
                    "two weeks ago",
                    ref,
                    bracket_day(two_weeks_ago),
                    text,
                ),
            ]
        elif i == 2:
            text = (
                "From April 1 to April 15, 2026 we'll be abroad, and we "
                "return on April 16, 2026."
            )
            exprs = [
                expr_interval(
                    "From April 1 to April 15, 2026",
                    ref,
                    bracket_day(datetime(2026, 4, 1, tzinfo=timezone.utc)),
                    bracket_day(datetime(2026, 4, 15, tzinfo=timezone.utc)),
                    text,
                ),
                expr_instant(
                    "April 16, 2026",
                    ref,
                    bracket_day(datetime(2026, 4, 16, tzinfo=timezone.utc)),
                    text,
                ),
            ]
        elif i == 3:
            text = (
                "In 2005 I started teaching, and every Thursday since "
                "then I've hosted office hours."
            )
            rec = Recurrence(
                rrule="FREQ=WEEKLY;BYDAY=TH",
                dtstart=bracket_day(datetime(2005, 9, 1, tzinfo=timezone.utc)),
                until=None,
                exdates=[],
            )
            exprs = [
                expr_instant("2005", ref, bracket_year(2005), text),
                expr_recurrence("every Thursday", ref, rec, text),
            ]
        else:
            text = (
                "Around 2010 I learned to sail, and last year I finally bought a boat."
            )
            last_year = ref.year - 1
            exprs = [
                expr_instant(
                    "Around 2010",
                    ref,
                    FuzzyInstant(
                        earliest=datetime(2008, 1, 1, tzinfo=timezone.utc),
                        latest=datetime(2013, 1, 1, tzinfo=timezone.utc),
                        best=datetime(2010, 1, 1, tzinfo=timezone.utc),
                        granularity="year",
                    ),
                    text,
                ),
                expr_instant("last year", ref, bracket_year(last_year), text),
            ]
        doc = Doc(f"doc_multi_{i}", text, ref, exprs)
        docs.append(doc)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    # 1. Specific-day queries (10)
    specific_days = [
        (2026, 3, 15),
        (2026, 4, 1),
        (2026, 1, 7),
        (2025, 12, 25),
        (2026, 3, 10),
        (2026, 4, 16),
        (2025, 11, 5),
        (2026, 2, 14),
        (2026, 3, 5),
        (2026, 2, 15),
    ]
    for i, (y, m, d) in enumerate(specific_days):
        surface = datetime(y, m, d).strftime("%B %-d, %Y")
        text = f"What happened on {surface}?"
        ref = NOW
        q = Query(f"q_spec_day_{i}", text, ref)
        q.gold_expressions.append(
            expr_instant(
                surface,
                ref,
                bracket_day(datetime(y, m, d, tzinfo=timezone.utc)),
                text,
            )
        )
        queries.append(q)

    # 2. Relative-day queries (10): at varied ref_times so they line up
    # with some doc ref_times.
    rel_q_cases = [
        ("yesterday", -1, "day"),
        ("two weeks ago", -14, "day"),
        ("last week", -7, "week"),
        ("three days ago", -3, "day"),
        ("a month ago", -30, "day"),
        ("last month", -30, "month"),
        ("a week ago", -7, "day"),
        ("yesterday", -1, "day"),
        ("two weeks ago", -14, "day"),
        ("last year", None, "year"),
    ]
    # distinct ref_times chosen to intersect with docs
    rel_q_refs = [
        NOW,
        NOW - relativedelta(days=3),
        NOW - relativedelta(days=6),
        NOW - relativedelta(days=9),
        NOW - relativedelta(days=12),
        NOW - relativedelta(days=15),
        NOW - relativedelta(days=20),
        NOW - relativedelta(days=25),
        NOW - relativedelta(days=30),
        NOW,
    ]
    for i, ((surf, delta_days, gran), ref) in enumerate(zip(rel_q_cases, rel_q_refs)):
        text = f"What did I do {surf}?"
        if surf == "last year":
            gi = bracket_year(ref.year - 1)
        elif surf == "last month":
            prev_month_date = ref - relativedelta(months=1)
            gi = bracket_month(prev_month_date.year, prev_month_date.month)
        elif surf == "last week":
            gi = bracket_week(ref - timedelta(days=7))
        else:
            target = ref + timedelta(days=delta_days)
            d0 = target.replace(hour=0, minute=0, second=0, microsecond=0)
            gi = FuzzyInstant(
                earliest=d0,
                latest=d0 + timedelta(days=1),
                best=d0 + timedelta(hours=12),
                granularity="day",
            )
        q = Query(f"q_rel_day_{i}", text, ref)
        q.gold_expressions.append(expr_instant(surf, ref, gi, text))
        queries.append(q)

    # 3. Fuzzy-period queries (10)
    fuzzy_cases = [
        ("in 2015", "year", 2015, None),
        ("in the 90s", "decade", 1990, None),
        ("around 1998", "fuzzy_year", 1998, None),
        ("in 1987", "year", 1987, None),
        ("in 1969", "year", 1969, None),
        ("in the 60s", "decade", 1960, None),
        ("in the 2010s", "decade", 2010, None),
        ("around 2010", "fuzzy_year", 2010, None),
        ("in 2005", "year", 2005, None),
        ("in 1776", "year", 1776, None),
    ]
    for i, (surf, kind, yr, _) in enumerate(fuzzy_cases):
        text = f"What happened {surf}?"
        ref = NOW
        if kind == "year":
            gi = bracket_year(yr)
        elif kind == "decade":
            gi = bracket_decade(yr)
        elif kind == "fuzzy_year":
            gi = FuzzyInstant(
                earliest=datetime(yr - 2, 1, 1, tzinfo=timezone.utc),
                latest=datetime(yr + 3, 1, 1, tzinfo=timezone.utc),
                best=datetime(yr, 1, 1, tzinfo=timezone.utc),
                granularity="year",
            )
        else:
            gi = bracket_year(yr)
        q = Query(f"q_fuzzy_{i}", text, ref)
        # surface for q_fuzzy strips "in " prefix
        surface = surf.removeprefix("in ")
        q.gold_expressions.append(expr_instant(surface, ref, gi, text))
        queries.append(q)

    # 4. Recurrence-probe queries (10)
    rec_q_cases = [
        ("next Thursday", "day", 4),  # upcoming TH
        ("this Monday", "day", 0),
        ("next Monday", "day", 0),
        ("this month", "month", None),
        ("next week", "week", None),
        ("this Friday", "day", 4),
        ("next Sunday", "day", 6),
        ("today", "day", None),
        ("this week", "week", None),
        ("next Tuesday", "day", 1),
    ]

    def next_weekday(ref: datetime, weekday: int) -> datetime:
        days_ahead = (weekday - ref.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        return ref + timedelta(days=days_ahead)

    for i, (surf, kind, wkd) in enumerate(rec_q_cases):
        text = f"What's on for {surf}?"
        ref = NOW
        if surf == "today":
            gi = bracket_day(ref)
        elif surf == "this week":
            gi = bracket_week(ref)
        elif surf == "next week":
            gi = bracket_week(ref + timedelta(days=7))
        elif surf == "this month":
            gi = bracket_month(ref.year, ref.month)
        elif surf.startswith("next"):
            target = next_weekday(ref, wkd)
            gi = bracket_day(target)
        elif surf.startswith("this"):
            # closest weekday in same week
            days_ahead = (wkd - ref.weekday()) % 7
            target = ref + timedelta(days=days_ahead)
            gi = bracket_day(target)
        else:
            gi = bracket_day(ref)
        q = Query(f"q_rec_{i}", text, ref)
        q.gold_expressions.append(expr_instant(surf, ref, gi, text))
        queries.append(q)

    # 5. Interval-probe queries (5)
    interval_q_cases = [
        (
            "during the first week of May 2026",
            datetime(2026, 5, 4, tzinfo=timezone.utc),
            datetime(2026, 5, 11, tzinfo=timezone.utc),
        ),
        (
            "between March 1 and March 15, 2026",
            datetime(2026, 3, 1, tzinfo=timezone.utc),
            datetime(2026, 3, 15, tzinfo=timezone.utc),
        ),
        (
            "during November 2025",
            datetime(2025, 11, 1, tzinfo=timezone.utc),
            datetime(2025, 12, 1, tzinfo=timezone.utc),
        ),
        (
            "from February 10 to February 20, 2026",
            datetime(2026, 2, 10, tzinfo=timezone.utc),
            datetime(2026, 2, 20, tzinfo=timezone.utc),
        ),
        (
            "during April 2026",
            datetime(2026, 4, 1, tzinfo=timezone.utc),
            datetime(2026, 5, 1, tzinfo=timezone.utc),
        ),
    ]
    for i, (surface, start, end) in enumerate(interval_q_cases):
        text = f"What happened {surface}?"
        ref = NOW
        q = Query(f"q_interval_{i}", text, ref)
        q.gold_expressions.append(
            expr_interval(
                surface,
                ref,
                bracket_day(start),
                bracket_day(end),
                text,
            )
        )
        queries.append(q)

    # 6. No-time queries (5) - semantic only
    no_time_cases = [
        "What is my favorite hobby?",
        "Who is my dentist?",
        "What did I name the puppy?",
        "Who is on the renovation crew?",
        "What is my wife's coffee order?",
    ]
    for i, text in enumerate(no_time_cases):
        q = Query(f"q_notime_{i}", text, NOW)
        queries.append(q)

    # 7. Critical "N days from now" paired queries (5)
    critical_pairs: list[tuple[str, str]] = []
    for i, n in enumerate([7, 14, 21, 30, 10]):
        # Doc: authored at T1, says "N days from now"
        t1 = NOW - relativedelta(days=40 * (i + 1))
        doc_target = t1 + timedelta(days=n)
        surface_d = f"{n} days from now"
        doc_text = f"{surface_d.capitalize()} I have a dentist appointment."
        doc = Doc(f"doc_crit_{i}", doc_text, t1)
        doc.gold_expressions.append(
            expr_instant(surface_d, t1, bracket_day(doc_target), doc_text)
        )
        docs.append(doc)
        # Query: at T1+2N, says "N days ago"
        t2 = t1 + timedelta(days=2 * n)
        query_target = t2 - timedelta(days=n)
        surface_q = f"{n} days ago"
        q_text = f"What did I have scheduled {surface_q}?"
        q = Query(f"q_crit_{i}", q_text, t2)
        q.gold_expressions.append(
            expr_instant(surface_q, t2, bracket_day(query_target), q_text)
        )
        queries.append(q)
        critical_pairs.append((doc.doc_id, q.query_id))

    return docs, queries, critical_pairs


# ---------------------------------------------------------------------------
# Gold relevance (doc-query pairs where intervals overlap)
# ---------------------------------------------------------------------------
from schema import GRANULARITY_ORDER as _GORDER


def _flatten_intervals_to_us(
    te: TimeExpression,
) -> list[tuple[int, int, int | None, str]]:
    from datetime import datetime, timedelta, timezone

    from expander import expand
    from schema import to_us

    out: list[tuple[int, int, int | None, str]] = []
    if te.kind == "instant" and te.instant:
        out.append(
            (
                to_us(te.instant.earliest),
                to_us(te.instant.latest),
                to_us(te.instant.best) if te.instant.best else None,
                te.instant.granularity,
            )
        )
    elif te.kind == "interval" and te.interval:
        g = (
            te.interval.start.granularity
            if _GORDER[te.interval.start.granularity]
            >= _GORDER[te.interval.end.granularity]
            else te.interval.end.granularity
        )
        best = te.interval.start.best or te.interval.start.earliest
        out.append(
            (
                to_us(te.interval.start.earliest),
                to_us(te.interval.end.latest),
                to_us(best),
                g,
            )
        )
    elif te.kind == "recurrence" and te.recurrence:
        now = datetime.now(tz=timezone.utc)
        anchor = te.recurrence.dtstart.best or te.recurrence.dtstart.earliest
        start = min(now - timedelta(days=365 * 10), anchor - timedelta(days=365))
        end = now + timedelta(days=365 * 2)
        if te.recurrence.until is not None:
            end = min(
                end,
                te.recurrence.until.latest or te.recurrence.until.earliest,
            )
        for inst in expand(te.recurrence, start, end):
            out.append(
                (
                    to_us(inst.earliest),
                    to_us(inst.latest),
                    to_us(inst.best) if inst.best else None,
                    inst.granularity,
                )
            )
    return out


def compute_gold_relevance(
    docs: list[Doc], queries: list[Query]
) -> dict[str, list[str]]:
    """A doc is relevant if ANY doc-interval overlaps ANY query-interval
    with Jaccard >= 0.05 OR both bests are within the union granularity.
    """
    # Pre-flatten doc intervals.
    doc_ivals: dict[str, list[tuple[int, int, int | None, str]]] = {}
    for d in docs:
        ivs: list[tuple[int, int, int | None, str]] = []
        for te in d.gold_expressions:
            ivs.extend(_flatten_intervals_to_us(te))
        doc_ivals[d.doc_id] = ivs

    out: dict[str, list[str]] = {}
    for q in queries:
        q_ivs: list[tuple[int, int, int | None, str]] = []
        for te in q.gold_expressions:
            q_ivs.extend(_flatten_intervals_to_us(te))
        rel: list[str] = []
        if not q_ivs:
            out[q.query_id] = rel
            continue
        for d_id, d_ivs in doc_ivals.items():
            matched = False
            for qe, ql, qb, qg in q_ivs:
                for de, dl, db, dg in d_ivs:
                    if qe >= dl or de >= ql:
                        continue
                    overlap = min(ql, dl) - max(qe, de)
                    union = max(ql, dl) - min(qe, de)
                    jaccard = overlap / union if union > 0 else 0.0
                    both_best_ok = False
                    if qb is not None and db is not None:
                        union_span = max(union, 1_000_000)
                        if abs(qb - db) <= union_span:
                            both_best_ok = True
                    if jaccard >= 0.05 or both_best_ok:
                        matched = True
                        break
                if matched:
                    break
            if matched:
                rel.append(d_id)
        out[q.query_id] = rel
    return out


# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------
def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    docs, queries, critical_pairs = build_docs_and_queries()
    print(f"Generated {len(docs)} docs and {len(queries)} queries.")
    print(f"Critical pairs: {critical_pairs}")

    from schema import iso

    doc_rows = [
        {
            "doc_id": d.doc_id,
            "text": d.text,
            "ref_time": iso(d.ref_time),
            "gold_expressions": [
                time_expression_to_dict(te) for te in d.gold_expressions
            ],
        }
        for d in docs
    ]
    query_rows = [
        {
            "query_id": q.query_id,
            "text": q.text,
            "ref_time": iso(q.ref_time),
            "gold_expressions": [
                time_expression_to_dict(te) for te in q.gold_expressions
            ],
        }
        for q in queries
    ]
    gold = compute_gold_relevance(docs, queries)
    gold_rows = [
        {"query_id": qid, "relevant_doc_ids": sorted(rel)} for qid, rel in gold.items()
    ]

    write_jsonl(DATA_DIR / "docs.jsonl", doc_rows)
    write_jsonl(DATA_DIR / "queries.jsonl", query_rows)
    write_jsonl(DATA_DIR / "gold.jsonl", gold_rows)
    with (DATA_DIR / "critical_pairs.json").open("w") as f:
        json.dump(critical_pairs, f, indent=2)
    print("Wrote docs.jsonl, queries.jsonl, gold.jsonl, critical_pairs.json")

    # Summary
    has_gold = sum(1 for r in gold_rows if r["relevant_doc_ids"])
    print(f"Queries with >=1 relevant doc: {has_gold}/{len(queries)}")


if __name__ == "__main__":
    main()
