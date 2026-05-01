"""Hard temporal-stress synthetic benchmark.

Goal: produce a benchmark on which semantic-only R@5 < 0.95 but a usable
temporal pipeline can make discriminative gains. Stresses both axes:

- Entity collisions: many entities share first OR last names so semantic
  retrieval cannot uniquely identify the right doc by name alone.
- Topic collisions: the same topical phrase ("quarterly review", "moved
  apartments", "started a new role at") is reused across many docs.
- Time spread: docs span Jan 2022 – Dec 2024 (3 years) so temporal can
  meaningfully discriminate between a 2022 event and a 2024 event for the
  same entity.
- Per-entity multiplicity: each persona has ~5-10 events at different
  times. Queries ask about a specific persona's event in a specific time
  window — gold is the unique doc matching BOTH the entity AND the time.

Domain: fictional newsletter-style "team activity log" entries — natural
language but template-instantiated for control.

Writes:
    data/hard_bench_docs.jsonl
    data/hard_bench_queries.jsonl
    data/hard_bench_gold.jsonl
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

REF_TIME = "2025-01-15T00:00:00Z"  # benchmark "now"

# ---------------------------------------------------------------------------
# Entity pool — designed to MAXIMIZE NAME COLLISIONS
# ---------------------------------------------------------------------------
# 4 first names × 6 last names = 24 base personas. We add a handful of
# extras to push to 30 personas, several of which deliberately share at
# least one name component with another persona, so a query like "Sarah's
# project launch" or "Kim's promotion" is ambiguous.
FIRST_NAMES = ["Sarah", "Kim", "Marcus", "Priya"]
LAST_NAMES = ["Chen", "Park", "Johnson", "Patel", "Nguyen", "Davis"]

# Build base personas: every (first, last) combination.
BASE_PERSONAS = [(f, l) for f in FIRST_NAMES for l in LAST_NAMES]

# A few extra personas using the same names to amplify collisions.
EXTRA_PERSONAS = [
    ("Sarah", "Lee"),
    ("Marcus", "Park"),
    ("Priya", "Johnson"),
    ("Kim", "Patel"),
    ("Daniel", "Chen"),
    ("Daniel", "Park"),
]
ALL_PERSONAS = BASE_PERSONAS + EXTRA_PERSONAS  # 30 personas

# ---------------------------------------------------------------------------
# Event templates — designed to MAXIMIZE TOPIC COLLISIONS
# ---------------------------------------------------------------------------
# Each template has a topic phrase that will appear in many docs.
EVENT_TEMPLATES = [
    "{name} was promoted to {role} on {date}.",
    "{name} joined the {team} team on {date}.",
    "{name} left the company on {date}.",
    "{name} led the {project} project kickoff on {date}.",
    "{name} closed the {client} account on {date}.",
    "{name} presented at the {conf} conference on {date}.",
    "{name} attended the company offsite in {city} on {date}.",
    "{name} completed the {cert} certification on {date}.",
    "{name} moved to the {city} office on {date}.",
    "{name} hosted a workshop on {topic} on {date}.",
    "{name} hit a five-year work anniversary on {date}.",
    "{name} delivered the quarterly review to leadership on {date}.",
    "{name} mentored a new hire starting on {date}.",
    "{name} completed onboarding on {date}.",
    "{name} was awarded employee of the month on {date}.",
]

ROLES = [
    "senior engineer",
    "staff engineer",
    "engineering manager",
    "director",
    "principal",
    "VP of engineering",
    "team lead",
    "product manager",
    "senior PM",
]
TEAMS = [
    "platform",
    "growth",
    "infrastructure",
    "billing",
    "data science",
    "frontend",
    "search",
    "ranking",
    "checkout",
    "payments",
    "ML",
]
PROJECTS = [
    "Aurora",
    "Polaris",
    "Northstar",
    "Compass",
    "Helios",
    "Nimbus",
    "Atlas",
    "Phoenix",
    "Tempo",
    "Orion",
]
CLIENTS = [
    "Globex",
    "Initech",
    "Wayne Enterprises",
    "Stark Industries",
    "Acme",
    "Hooli",
    "Pied Piper",
    "Soylent",
    "Umbrella",
]
CONFS = [
    "Strata",
    "ICML",
    "QCon",
    "PyCon",
    "KubeCon",
    "RailsConf",
    "AWS re:Invent",
    "Google I/O",
]
CITIES = [
    "Seattle",
    "Austin",
    "New York",
    "London",
    "Berlin",
    "Singapore",
    "Tokyo",
    "Mexico City",
    "Toronto",
    "Dublin",
]
CERTS = [
    "AWS Solutions Architect",
    "GCP Professional",
    "Kubernetes CKA",
    "Scrum Master",
    "PMP",
    "Six Sigma",
    "Azure Architect",
]
TOPICS = [
    "data pipelines",
    "incident response",
    "code review",
    "performance tuning",
    "design systems",
    "leadership",
    "user research",
    "feature flags",
]


def fmt_date(dt: datetime) -> str:
    """Render as 'Mar 14, 2023' — natural language with month-day-year."""
    return dt.strftime("%b %d, %Y").replace(" 0", " ")


def random_date(rng: random.Random) -> datetime:
    """Uniform sample in [Jan 1 2022, Dec 31 2024]."""
    start = datetime(2022, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 12, 31, tzinfo=timezone.utc)
    span = (end - start).days
    return start + timedelta(days=rng.randrange(span))


def generate_doc(rng: random.Random, persona: tuple[str, str]) -> dict:
    """Generate one doc for a persona using a random template."""
    first, last = persona
    name = f"{first} {last}"
    tmpl = rng.choice(EVENT_TEMPLATES)
    date = random_date(rng)
    text = tmpl.format(
        name=name,
        date=fmt_date(date),
        role=rng.choice(ROLES),
        team=rng.choice(TEAMS),
        project=rng.choice(PROJECTS),
        client=rng.choice(CLIENTS),
        conf=rng.choice(CONFS),
        city=rng.choice(CITIES),
        cert=rng.choice(CERTS),
        topic=rng.choice(TOPICS),
    )
    return {
        "text": text,
        "_first": first,
        "_last": last,
        "_template": tmpl,
        "_date": date,
    }


def build_docs(per_persona_min: int = 4, per_persona_max: int = 8) -> list[dict]:
    """Generate ~30 personas × ~6 events = ~180 docs, padded with topical
    distractors not tied to any persona to push to ~600 total."""
    rng = random.Random(20260424)
    out: list[dict] = []

    # Per-persona events
    for persona in ALL_PERSONAS:
        n = rng.randint(per_persona_min, per_persona_max)
        for _ in range(n):
            doc = generate_doc(rng, persona)
            out.append(doc)

    # Pad with broader-topic distractors using random non-collision names so
    # the corpus has natural-name diversity. Same templates → same topic
    # collisions.
    OTHER_FIRSTS = [
        "Alex",
        "Jordan",
        "Morgan",
        "Taylor",
        "Casey",
        "Robin",
        "Sam",
        "Avery",
        "Riley",
        "Quinn",
        "Hannah",
        "Eric",
        "Maya",
        "Liam",
        "Noah",
        "Olivia",
        "Ava",
        "Mia",
        "Ethan",
        "Lucas",
    ]
    OTHER_LASTS = [
        "Smith",
        "Brown",
        "Wilson",
        "Garcia",
        "Martinez",
        "Anderson",
        "Thomas",
        "Jackson",
        "White",
        "Harris",
        "Roberts",
        "Walker",
        "Hall",
        "Young",
        "King",
    ]
    n_pad = 600 - len(out)
    for _ in range(n_pad):
        f = rng.choice(OTHER_FIRSTS)
        l = rng.choice(OTHER_LASTS)
        doc = generate_doc(rng, (f, l))
        out.append(doc)

    rng.shuffle(out)
    docs: list[dict] = []
    for i, d in enumerate(out):
        docs.append(
            {
                "doc_id": f"hd_{i:04d}",
                "text": d["text"],
                "ref_time": REF_TIME,
                "_first": d["_first"],
                "_last": d["_last"],
                "_template": d["_template"],
                "_date_iso": d["_date"].isoformat(),
            }
        )
    return docs


# ---------------------------------------------------------------------------
# Query generation
# ---------------------------------------------------------------------------
# We generate three difficulty tiers:
#   easy:   query names a unique persona (full name) AND a unique year.
#   medium: query gives only a first OR last name + a quarter-level window.
#   hard:   query gives a TOPIC + a quarter-level window, no name.
# Gold = exactly ONE doc that matches the query's stated constraints.
#
# To ensure exactly-one gold per query, we walk the doc list and:
# - Pick a doc d.
# - Try to phrase a query whose constraints (entity hint, topic hint, time
#   window) are matched only by d among the corpus.
# - If 1 match, accept; if not, try a different formulation.

QUERY_TEMPLATES = {
    # Each entry is (parameterized_query_text, gold_filter_lambda)
    "promotion": "When was {entity_hint} promoted{time_hint}?",
    "joined": "When did {entity_hint} join the team{time_hint}?",
    "left": "When did {entity_hint} leave the company{time_hint}?",
    "kickoff": "When did {entity_hint} lead the project kickoff{time_hint}?",
    "closed_account": "When did {entity_hint} close the client account{time_hint}?",
    "conference": "When did {entity_hint} present at a conference{time_hint}?",
    "offsite": "When did {entity_hint} attend the company offsite{time_hint}?",
    "cert": "When did {entity_hint} complete a certification{time_hint}?",
    "moved": "When did {entity_hint} move to a new office{time_hint}?",
    "workshop": "When did {entity_hint} host a workshop{time_hint}?",
    "anniversary": "When did {entity_hint} hit a work anniversary{time_hint}?",
    "review": "When did {entity_hint} deliver the quarterly review{time_hint}?",
    "mentored": "When did {entity_hint} mentor a new hire{time_hint}?",
    "onboarding": "When did {entity_hint} complete onboarding{time_hint}?",
    "employee": "When was {entity_hint} awarded employee of the month{time_hint}?",
}

# Map template-text → query-template-key (for matching docs to queries)
TEMPLATE_TO_KEY = {
    "{name} was promoted to {role} on {date}.": "promotion",
    "{name} joined the {team} team on {date}.": "joined",
    "{name} left the company on {date}.": "left",
    "{name} led the {project} project kickoff on {date}.": "kickoff",
    "{name} closed the {client} account on {date}.": "closed_account",
    "{name} presented at the {conf} conference on {date}.": "conference",
    "{name} attended the company offsite in {city} on {date}.": "offsite",
    "{name} completed the {cert} certification on {date}.": "cert",
    "{name} moved to the {city} office on {date}.": "moved",
    "{name} hosted a workshop on {topic} on {date}.": "workshop",
    "{name} hit a five-year work anniversary on {date}.": "anniversary",
    "{name} delivered the quarterly review to leadership on {date}.": "review",
    "{name} mentored a new hire starting on {date}.": "mentored",
    "{name} completed onboarding on {date}.": "onboarding",
    "{name} was awarded employee of the month on {date}.": "employee",
}


def quarter_of(dt: datetime) -> tuple[int, int]:
    """Return (year, quarter)."""
    return dt.year, (dt.month - 1) // 3 + 1


def quarter_str(year: int, quarter: int) -> str:
    return f"Q{quarter} {year}"


def year_str(year: int) -> str:
    return str(year)


def parse_doc_date(d: dict) -> datetime:
    return datetime.fromisoformat(d["_date_iso"].replace("Z", "+00:00"))


def docs_matching(
    docs: list[dict],
    *,
    template_key: str,
    first: str | None = None,
    last: str | None = None,
    full_name: str | None = None,
    year: int | None = None,
    quarter: tuple[int, int] | None = None,
) -> list[dict]:
    """Return all docs matching the given filters."""
    out = []
    for d in docs:
        tk = TEMPLATE_TO_KEY.get(d["_template"])
        if tk != template_key:
            continue
        if full_name is not None:
            if f"{d['_first']} {d['_last']}" != full_name:
                continue
        if first is not None and d["_first"] != first:
            continue
        if last is not None and d["_last"] != last:
            continue
        dt = parse_doc_date(d)
        if year is not None and dt.year != year:
            continue
        if quarter is not None and quarter_of(dt) != quarter:
            continue
        out.append(d)
    return out


def build_queries(docs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Build queries with exactly-one gold per query, mixing difficulties."""
    rng = random.Random(20260425)
    queries: list[dict] = []
    gold: list[dict] = []
    used_doc_ids: set[str] = set()

    target_per_tier = 30  # 30 each = 90 total

    def try_add(qtext: str, gold_doc_id: str, tier: str, ref_time: str) -> bool:
        if gold_doc_id in used_doc_ids:
            return False
        qid = f"q_{tier}_{len([q for q in queries if q['subset'] == tier]):03d}"
        queries.append(
            {
                "query_id": qid,
                "text": qtext,
                "ref_time": ref_time,
                "subset": tier,
            }
        )
        gold.append({"query_id": qid, "relevant_doc_ids": [gold_doc_id]})
        used_doc_ids.add(gold_doc_id)
        return True

    # ---- EASY: full name + year ----
    # Should be uniquely matched in most cases (each persona has 1-2 docs/year of any given template)
    candidates = list(docs)
    rng.shuffle(candidates)
    for d in candidates:
        if len([q for q in queries if q["subset"] == "easy"]) >= target_per_tier:
            break
        tk = TEMPLATE_TO_KEY.get(d["_template"])
        if tk is None:
            continue
        full_name = f"{d['_first']} {d['_last']}"
        dt = parse_doc_date(d)
        matches = docs_matching(
            docs, template_key=tk, full_name=full_name, year=dt.year
        )
        if len(matches) != 1:
            continue
        # Skip if the persona doesn't appear in main personas (these are
        # padding personas; their full names are too unique — too easy).
        if (d["_first"], d["_last"]) not in ALL_PERSONAS:
            continue
        qtext = QUERY_TEMPLATES[tk].format(
            entity_hint=full_name,
            time_hint=f" in {dt.year}",
        )
        # Use a ref_time AFTER the event (Jan 15 2025) — so the query is
        # retrospective.
        try_add(qtext, d["doc_id"], "easy", REF_TIME)

    # ---- MEDIUM: only first OR last name + quarter window ----
    # Disambiguation is needed because multiple personas share the name.
    candidates = list(docs)
    rng.shuffle(candidates)
    for d in candidates:
        if len([q for q in queries if q["subset"] == "medium"]) >= target_per_tier:
            break
        tk = TEMPLATE_TO_KEY.get(d["_template"])
        if tk is None:
            continue
        if (d["_first"], d["_last"]) not in ALL_PERSONAS:
            continue
        dt = parse_doc_date(d)
        q = quarter_of(dt)
        # Try first-name only.
        first_matches = docs_matching(
            docs, template_key=tk, first=d["_first"], quarter=q
        )
        if len(first_matches) == 1:
            qtext = QUERY_TEMPLATES[tk].format(
                entity_hint=d["_first"],
                time_hint=f" in {quarter_str(*q)}",
            )
            if try_add(qtext, d["doc_id"], "medium", REF_TIME):
                continue
        # Try last-name only.
        last_matches = docs_matching(docs, template_key=tk, last=d["_last"], quarter=q)
        if len(last_matches) == 1:
            qtext = QUERY_TEMPLATES[tk].format(
                entity_hint=d["_last"],
                time_hint=f" in {quarter_str(*q)}",
            )
            try_add(qtext, d["doc_id"], "medium", REF_TIME)

    # ---- HARD: NO name, only template + tight window ----
    # Many docs share the template; a quarter window discriminates.
    candidates = list(docs)
    rng.shuffle(candidates)
    for d in candidates:
        if len([q for q in queries if q["subset"] == "hard"]) >= target_per_tier:
            break
        tk = TEMPLATE_TO_KEY.get(d["_template"])
        if tk is None:
            continue
        dt = parse_doc_date(d)
        q = quarter_of(dt)
        # Match by (template, quarter) only — no name filter. Must be
        # exactly one match.
        matches = docs_matching(docs, template_key=tk, quarter=q)
        if len(matches) != 1:
            continue
        # Use a partial entity hint from the doc to keep query natural-language —
        # but mask it with a generic phrase so semantic alone can't trivially
        # match. For HARD we use NO name (generic "someone") + a topic-y form.
        qtext = QUERY_TEMPLATES[tk].format(
            entity_hint="someone on the team",
            time_hint=f" in {quarter_str(*q)}",
        )
        try_add(qtext, d["doc_id"], "hard", REF_TIME)

    return queries, gold


def main() -> None:
    docs = build_docs(per_persona_min=4, per_persona_max=8)

    # Stats
    print(f"Total docs: {len(docs)}")
    from collections import Counter

    yr_count = Counter(parse_doc_date(d).year for d in docs)
    print(f"Year distribution: {dict(sorted(yr_count.items()))}")
    template_count = Counter(d["_template"] for d in docs)
    print(f"Distinct templates: {len(template_count)}")
    persona_count = Counter((d["_first"], d["_last"]) for d in docs)
    print(f"Distinct personas: {len(persona_count)}")
    first_count = Counter(d["_first"] for d in docs)
    print(f"Distinct first names: {len(first_count)}")
    last_count = Counter(d["_last"] for d in docs)
    print(f"Distinct last names: {len(last_count)}")
    avg_words = sum(len(d["text"].split()) for d in docs) / len(docs)
    print(f"Avg words per doc: {avg_words:.1f}")

    # Strip private fields before writing
    public_docs = [
        {"doc_id": d["doc_id"], "text": d["text"], "ref_time": d["ref_time"]}
        for d in docs
    ]

    queries, gold = build_queries(docs)
    print(f"\nQueries: {len(queries)}")
    by_subset = Counter(q["subset"] for q in queries)
    print(f"  By subset: {dict(by_subset)}")

    # Write
    docs_path = DATA_DIR / "hard_bench_docs.jsonl"
    q_path = DATA_DIR / "hard_bench_queries.jsonl"
    g_path = DATA_DIR / "hard_bench_gold.jsonl"
    docs_path.write_text("\n".join(json.dumps(d) for d in public_docs) + "\n")
    q_path.write_text("\n".join(json.dumps(q) for q in queries) + "\n")
    g_path.write_text("\n".join(json.dumps(g) for g in gold) + "\n")
    print(f"\nWrote {docs_path}")
    print(f"Wrote {q_path}")
    print(f"Wrote {g_path}")


if __name__ == "__main__":
    main()
