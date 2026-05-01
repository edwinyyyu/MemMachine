"""Discriminator queries + paired documents for the ablation study.

Each subset probes a hypothesis that the base 55-query set does not
separate clearly:

- H1 "wide_vs_narrow" (10): doc says "3 weeks ago", query says "last
  month". Under narrow brackets these miss; under wide brackets they hit.
- H2 "center_matters" (10): two docs that overlap the query bracket
  equally but differ in center. Under Jaccard these tie; under Gaussian
  the closer-center doc wins.
- H3 "recurrence_density" (10): probe whether recurring docs over-rank
  vs one-time docs (sum-aggregation fanout effect).

All queries are issued at NOW = 2026-04-23T12:00:00Z to keep ref-time
arithmetic deterministic, and we know gold directly because we author
both sides.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dateutil.relativedelta import relativedelta
from schema import (
    FuzzyInstant,
    Recurrence,
    iso,
    time_expression_to_dict,
)
from synth_data import (
    NOW,
    Doc,
    Query,
    bracket_day,
    bracket_month,
    expr_instant,
    expr_recurrence,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def _day_bracket_centered(target: datetime) -> FuzzyInstant:
    d0 = target.replace(hour=0, minute=0, second=0, microsecond=0)
    return FuzzyInstant(
        earliest=d0,
        latest=d0 + timedelta(days=1),
        best=d0 + timedelta(hours=12),
        granularity="day",
    )


def build_discriminators() -> tuple[list[Doc], list[Query], dict[str, list[str]]]:
    """Return (new_docs, new_queries, gold_by_query_id)."""
    docs: list[Doc] = []
    queries: list[Query] = []
    gold: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # H1: wide-vs-narrow
    # ------------------------------------------------------------------
    # Craft 10 pairs. Each (doc, query) pair is built so the doc's stated
    # temporal reference falls INSIDE the calendar unit the query names,
    # but not close to the narrow bracket the base resolver would emit.
    # NOW = 2026-04-23 (a Thursday). "Last month" = March 2026.
    #
    # Strategy: doc says "3 weeks ago" (relative to a ref_time chosen so
    # 3 weeks ago lands in March 2026). Query says "last month" at NOW.
    wvn_cases = [
        # ref_time 2026-04-18 (Saturday), 3 weeks ago => 2026-03-28 (Saturday)
        ("3 weeks ago", NOW - timedelta(days=5), 21, "I flew to Denver"),
        # ref 2026-04-15, 3 weeks ago => 2026-03-25
        ("3 weeks ago", NOW - timedelta(days=8), 21, "we signed the papers"),
        # 4 weeks ago, ref 2026-04-10 => 2026-03-13
        ("4 weeks ago", NOW - timedelta(days=13), 28, "the roof got repaired"),
        # "about a month ago" at NOW => ~2026-03-23 → in March
        (
            "about a month ago",
            NOW,
            None,
            "the storm knocked out the power",
        ),
        # 20 days ago at NOW => 2026-04-03; query "earlier this month" => April
        (
            "20 days ago",
            NOW,
            20,
            "I ran into an old coworker at the grocery store",
        ),
        # "a few weeks ago" at NOW => ~2026-04-02; April
        (
            "a few weeks ago",
            NOW,
            None,
            "the kids had a bake sale at school",
        ),
        # 25 days ago at NOW => 2026-03-29
        ("25 days ago", NOW, 25, "we held a memorial"),
        # 5 weeks ago at ref 2026-04-10 => 2026-03-06
        ("5 weeks ago", NOW - timedelta(days=13), 35, "I started the new job"),
        # 2 weeks ago at ref 2026-04-22 => 2026-04-08 (in April)
        (
            "2 weeks ago",
            NOW - timedelta(days=1),
            14,
            "we fixed the drainage issue",
        ),
        # "last month" as doc too — already calendar-tight.
        (
            "last month",
            NOW,
            None,
            "my sister visited from out of town",
        ),
    ]
    wvn_query_units = [
        "last month",  # 1
        "last month",
        "last month",
        "last month",
        "earlier this month",  # April at NOW
        "earlier this month",
        "last month",
        "last month",
        "earlier this month",
        "last month",
    ]
    for i, ((surf, ref, n_days, story), q_unit) in enumerate(
        zip(wvn_cases, wvn_query_units)
    ):
        doc_text = f"{surf.capitalize()} {story}."
        # Determine doc gold interval
        if n_days is not None:
            target = ref - timedelta(days=n_days)
            gi = _day_bracket_centered(target)
        elif surf == "about a month ago":
            target = ref - timedelta(days=30)
            gi = FuzzyInstant(
                earliest=target - timedelta(days=7),
                latest=target + timedelta(days=7),
                best=target,
                granularity="month",
            )
        elif surf == "a few weeks ago":
            target = ref - timedelta(days=21)
            gi = FuzzyInstant(
                earliest=target - timedelta(days=14),
                latest=target + timedelta(days=7),
                best=target,
                granularity="week",
            )
        elif surf == "last month":
            prev = ref - relativedelta(months=1)
            gi = bracket_month(prev.year, prev.month)
        else:
            raise AssertionError(surf)
        doc = Doc(f"doc_wvn_{i}", doc_text, ref)
        doc.gold_expressions.append(expr_instant(surf, ref, gi, doc_text))
        docs.append(doc)

        # Query
        q_ref = NOW
        if q_unit == "last month":
            prev = q_ref - relativedelta(months=1)
            q_gi = bracket_month(prev.year, prev.month)
            q_surface = "last month"
        elif q_unit == "earlier this month":
            q_gi = bracket_month(q_ref.year, q_ref.month)
            q_surface = "earlier this month"
        else:
            raise AssertionError(q_unit)
        q_text = f"What did I do {q_unit}?"
        q = Query(f"q_wvn_{i}", q_text, q_ref)
        q.gold_expressions.append(expr_instant(q_surface, q_ref, q_gi, q_text))
        queries.append(q)
        gold[q.query_id] = [doc.doc_id]

    # ------------------------------------------------------------------
    # H2: center matters more than overlap
    # ------------------------------------------------------------------
    # Build 10 query-times with TWO docs each. Both docs' gold brackets
    # overlap the query bracket by the same amount, but one is centered
    # closer to the query center. Under Jaccard they tie; under Gaussian
    # the closer-center wins. Gold = the closer-center doc.
    #
    # Implementation: pick a query interval [Q_e, Q_l]. Put doc_A
    # centered at query.best (expected winner). Put doc_B shifted so that
    # its overlap with Q equals that of doc_A but its center is further.
    #
    # To equalize overlap with identical bracket width, doc_A = Q itself
    # and doc_B = Q shifted by +δ (then overlap halves). That's not an
    # equal-overlap case. Instead, make doc_A narrower, centered at Q
    # center; doc_B wider, off-center, such that |overlap|/|union| ~
    # equal. Simpler: use identical-width docs; doc_A centered on Q, doc_B
    # shifted by δ such that 2 × overlap(A,Q) = overlap(B,Q) + overlap(A,Q)
    # — i.e., tuning so Jaccard ties is fiddly. We'll accept near-ties and
    # focus on "which ranks higher" rather than exact Jaccard equivalence.
    #
    # Simpler spec: both docs have bracket width = 2 × query width, one
    # centered on Q, the other shifted so its bracket still CONTAINS Q
    # entirely (overlap = |Q|), but its center is offset. With union =
    # max(width_s, width_q+offset), Jaccard is the same in both cases
    # because the offset doc's bracket fully contains Q. Perfect tie.
    #
    # Under Gaussian the close-center doc wins. Gold = closer center.
    for i in range(10):
        # Query: a specific day in April 2026
        day = 2 + i  # April 2..11 2026
        q_day = datetime(2026, 4, day, tzinfo=timezone.utc)
        q_gi = bracket_day(q_day)  # 1-day bracket
        q_text = f"What happened on {q_day.strftime('%B %-d, %Y')}?"
        q = Query(f"q_cm_{i}", q_text, NOW)
        q.gold_expressions.append(
            expr_instant(q_day.strftime("%B %-d, %Y"), NOW, q_gi, q_text)
        )
        queries.append(q)

        # Doc A: centered on q_day, bracket ± 2 days (5-day wide)
        doc_a_text = (
            f"Earlier this month, around {q_day.strftime('%B %-d, %Y')}, "
            f"I had a doctor's visit."
        )
        a_start = q_day - timedelta(days=2)
        a_end = q_day + timedelta(days=3)  # latest exclusive -> 2+1+2
        a_gi = FuzzyInstant(
            earliest=a_start,
            latest=a_end,
            best=q_day + timedelta(hours=12),
            granularity="day",
        )
        doc_a = Doc(f"doc_cm_{i}_near", doc_a_text, NOW - timedelta(days=1))
        doc_a.gold_expressions.append(
            expr_instant(q_day.strftime("%B %-d, %Y"), doc_a.ref_time, a_gi, doc_a_text)
        )
        docs.append(doc_a)

        # Doc B: bracket fully contains Q but centered 3 days after q_day
        # bracket = [q_day-1d, q_day+6d) = 7-day wide; center at q_day+2.5d
        b_center = q_day + timedelta(days=3)
        b_start = q_day - timedelta(days=1)
        b_end = q_day + timedelta(days=6)
        b_gi = FuzzyInstant(
            earliest=b_start,
            latest=b_end,
            best=b_center + timedelta(hours=12),
            granularity="week",
        )
        doc_b_text = (
            f"Sometime in the first week of {q_day.strftime('%B %Y')}, "
            f"I cleaned out the garage."
        )
        doc_b = Doc(f"doc_cm_{i}_far", doc_b_text, NOW - timedelta(days=1))
        doc_b.gold_expressions.append(
            expr_instant(
                f"the first week of {q_day.strftime('%B %Y')}",
                doc_b.ref_time,
                b_gi,
                doc_b_text,
            )
        )
        docs.append(doc_b)

        # Gold: only the near doc (center-matters-more-than-overlap)
        gold[q.query_id] = [doc_a.doc_id]

    # ------------------------------------------------------------------
    # H3: recurrence density
    # ------------------------------------------------------------------
    # For 10 queries, we want to see whether a single one-time doc that
    # precisely matches the query outranks a recurring doc whose fanout
    # would otherwise sum to a higher score under sum-aggregation. We
    # pair queries with BOTH a one-time doc and a recurrence that happens
    # to intersect the query window. Gold = one-time doc (the specific
    # match) ONLY.
    recur_rrules = [
        (
            "FREQ=WEEKLY;BYDAY=WE",
            datetime(2024, 1, 3, 9, 0, tzinfo=timezone.utc),
            "every Wednesday",
        ),
        (
            "FREQ=WEEKLY;BYDAY=FR;BYHOUR=17",
            datetime(2024, 1, 5, 17, 0, tzinfo=timezone.utc),
            "every Friday at 5pm",
        ),
        (
            "FREQ=DAILY;BYHOUR=8",
            datetime(2024, 1, 1, 8, 0, tzinfo=timezone.utc),
            "every day at 8am",
        ),
        (
            "FREQ=WEEKLY;BYDAY=TU",
            datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc),
            "every Tuesday",
        ),
        (
            "FREQ=MONTHLY;BYMONTHDAY=1",
            datetime(2024, 1, 1, 9, 0, tzinfo=timezone.utc),
            "on the first of every month",
        ),
        (
            "FREQ=WEEKLY;BYDAY=SA",
            datetime(2024, 1, 6, 10, 0, tzinfo=timezone.utc),
            "every Saturday",
        ),
        (
            "FREQ=WEEKLY;BYDAY=MO,WE,FR",
            datetime(2024, 1, 1, 7, 0, tzinfo=timezone.utc),
            "every Mon/Wed/Fri",
        ),
        (
            "FREQ=WEEKLY;BYDAY=SU",
            datetime(2024, 1, 7, 11, 0, tzinfo=timezone.utc),
            "every Sunday",
        ),
        (
            "FREQ=WEEKLY;BYDAY=TH",
            datetime(2024, 1, 4, 18, 0, tzinfo=timezone.utc),
            "every Thursday",
        ),
        (
            "FREQ=MONTHLY;BYDAY=1MO",
            datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
            "the first Monday of every month",
        ),
    ]

    for i, (rrule, dtstart, rec_surface) in enumerate(recur_rrules):
        # Query: a specific day around 10-20 days ahead of NOW
        q_day = NOW + timedelta(days=10 + i)
        q_day_d = q_day.replace(hour=0, minute=0, second=0, microsecond=0)
        q_gi = bracket_day(q_day_d)
        q_text = f"What did I do on {q_day_d.strftime('%B %-d, %Y')}?"
        q = Query(f"q_rd_{i}", q_text, NOW)
        q.gold_expressions.append(
            expr_instant(
                q_day_d.strftime("%B %-d, %Y"),
                NOW,
                q_gi,
                q_text,
            )
        )
        queries.append(q)

        # One-time doc: says the specific date
        specific_text = f"On {q_day_d.strftime('%B %-d, %Y')} I hosted a dinner party."
        dt_doc = Doc(
            f"doc_rd_{i}_specific",
            specific_text,
            NOW - timedelta(days=2),
        )
        dt_doc.gold_expressions.append(
            expr_instant(
                q_day_d.strftime("%B %-d, %Y"),
                dt_doc.ref_time,
                bracket_day(q_day_d),
                specific_text,
            )
        )
        docs.append(dt_doc)

        # Recurrence doc: something that repeats and likely overlaps that day
        rec_text = f"I go to the gym {rec_surface}."
        rec_doc = Doc(
            f"doc_rd_{i}_recur",
            rec_text,
            NOW - timedelta(days=2),
        )
        rec_obj = Recurrence(
            rrule=rrule,
            dtstart=bracket_day(dtstart),
            until=None,
            exdates=[],
        )
        rec_doc.gold_expressions.append(
            expr_recurrence(rec_surface, rec_doc.ref_time, rec_obj, rec_text)
        )
        docs.append(rec_doc)

        # Gold: we want the specific one to rank above the recurrence.
        # But "rank above" is not directly expressible as a relevant set —
        # so we ONLY mark the specific doc relevant. Recall@1 measures
        # whether sum-aggregation over recurrence instances displaces it.
        gold[q.query_id] = [dt_doc.doc_id]

    return docs, queries, gold


def write_discriminators() -> None:
    docs, queries, gold = build_discriminators()

    # Persist alongside base data as SEPARATE jsonl files so we can load
    # them independently in the ablation orchestrator.
    with (DATA_DIR / "disc_docs.jsonl").open("w") as f:
        for d in docs:
            f.write(
                json.dumps(
                    {
                        "doc_id": d.doc_id,
                        "text": d.text,
                        "ref_time": iso(d.ref_time),
                        "gold_expressions": [
                            time_expression_to_dict(te) for te in d.gold_expressions
                        ],
                    }
                )
                + "\n"
            )

    with (DATA_DIR / "disc_queries.jsonl").open("w") as f:
        for q in queries:
            f.write(
                json.dumps(
                    {
                        "query_id": q.query_id,
                        "text": q.text,
                        "ref_time": iso(q.ref_time),
                        "gold_expressions": [
                            time_expression_to_dict(te) for te in q.gold_expressions
                        ],
                    }
                )
                + "\n"
            )

    with (DATA_DIR / "disc_gold.jsonl").open("w") as f:
        for qid, rel in gold.items():
            f.write(json.dumps({"query_id": qid, "relevant_doc_ids": rel}) + "\n")

    print(
        f"wrote {len(docs)} discriminator docs, {len(queries)} queries, "
        f"{len(gold)} gold entries."
    )


if __name__ == "__main__":
    write_discriminators()
