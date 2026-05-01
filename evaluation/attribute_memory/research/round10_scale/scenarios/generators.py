"""Deterministic scenario generators for round 10.

Each scenario produces:
  - `entries`: list[Entry] with {uuid, ts, text, mentions, refs, meta}
  - `questions`: list[Question] with ground-truth

No LLM calls. Entries are the post-write log as AEN-1 would store them,
had its writer emitted "correctly". We are testing retrieval+reader, not
the writer — the writer was already benchmarked in round 9.

`meta` carries scenario-level signals (predicate, cardinal, is_current, etc.)
used only for index construction / grading; it's NOT shown to the reader.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

Relation = Literal["clarify", "refine", "supersede", "invalidate"]


@dataclass
class Ref:
    uuid: str
    relation: Relation


@dataclass
class Entry:
    uuid: str
    ts: int
    text: str
    mentions: list[str] = field(default_factory=list)
    refs: list[Ref] = field(default_factory=list)
    # meta for ground-truth + indexing
    predicate: str | None = None  # e.g. "User.employer", "@Marcus.title"
    value: str | None = None  # the value stated ("Anthropic")
    is_current: bool = True  # whether still the current value after log


@dataclass
class Question:
    qid: str
    kind: str  # "current" | "history" | "supersede" | "entity" | "multi"
    question: str
    expected_contains: list[str]
    expected_absent: list[str] = field(default_factory=list)
    # Scenario-tag, useful for splitting
    scenario_tag: str = ""


# ---------------------------------------------------------------------------
# Entity / event libraries
# ---------------------------------------------------------------------------

ENTITY_POOL = [
    # humans
    "Jamie",
    "Marcus",
    "Alice",
    "Priya",
    "Diego",
    "Theo",
    "Nadia",
    "Ben",
    "Sara",
    "Liam",
    "Noah",
    "Maya",
    "Zane",
    "Aria",
    "Leo",
    "Nina",
    "Quinn",
    "Tess",
    "Emil",
    "Yuki",
    "Ivy",
    "Otto",
    "Hana",
    "Kai",
    "Rhea",
    "Silas",
    "Uma",
    "Wes",
    "Xochi",
    "Yves",
    "Zara",
    "Bram",
    "Cleo",
    "Deren",
    "Emi",
    "Finn",
    "Gia",
    "Hugo",
    "Ilse",
    "Jago",
    "Kira",
    "Lena",
    "Milo",
    "Nora",
    "Oren",
    "Pax",
    "Rafi",
    "Sena",
    "Tavi",
    "Ulla",
    "Vik",
    "Wren",
    "Xia",
    "Yair",
    "Zed",
    "Ana",
    "Bodhi",
    "Cass",
    "Dov",
    "Esa",
    "Fia",
    "Gus",
    "Hal",
    "Iris",
    "Juno",
    "Kato",
    "Lore",
    "Mona",
    "Nate",
    "Osa",
    "Pia",
    "Quint",
    "Reed",
    "Sana",
    "Tobi",
    "Uma2",
    "Veda",
    "Win",
    "Yao",
    "Zen",
]

PET_POOL = [
    "Luna",
    "Miso",
    "Rex",
    "Pepper",
    "Ziggy",
    "Taro",
    "Olive",
    "Biscuit",
    "Mango",
    "Pixel",
    "Waffle",
    "Nori",
    "Cleo2",
    "Rhubarb",
    "Doodle",
]

PLACE_POOL = [
    "Seattle",
    "New York",
    "San Francisco",
    "Austin",
    "Chicago",
    "Boston",
    "Denver",
    "Portland",
    "Miami",
    "Atlanta",
    "Toronto",
    "Vancouver",
]

EMPLOYER_POOL = [
    "Anthropic",
    "Google",
    "Stripe",
    "Meta",
    "Apple",
    "OpenAI",
    "Adobe",
    "Figma",
    "Pentagram",
    "Ramp",
    "Databricks",
    "Notion",
    "Vercel",
    "Cursor",
    "Replit",
    "Modal",
    "LinkedIn",
    "Microsoft",
]

TITLE_POOL = [
    "software engineer",
    "product manager",
    "designer",
    "tech lead",
    "staff engineer",
    "director",
    "researcher",
    "senior engineer",
    "data scientist",
    "marketing lead",
    "nurse",
    "teacher",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uuid(prefix: str, i: int) -> str:
    return f"{prefix}{i:05d}"


def _mentions(*names: str) -> list[str]:
    out = []
    for n in names:
        tag = f"@{n}"
        if tag not in out:
            out.append(tag)
    return out


# ---------------------------------------------------------------------------
# S-dense: one entity (User) dominates; ~70% of entries about User's employer/job
# ---------------------------------------------------------------------------


def gen_dense(n_entries: int, seed: int = 1) -> tuple[list[Entry], list[Question]]:
    """User has N job transitions plus filler. N ~ n_entries/6."""
    rng = random.Random(seed)
    entries: list[Entry] = []
    ts = 0

    # Pool of employers, shuffle for variety. We'll cycle through for
    # supersede chain.
    employers = EMPLOYER_POOL[:]
    rng.shuffle(employers)
    # Cap chain length; 20 transitions already a deep chain stress.
    # At 1000 entries this gives 1 transition per 50 entries of filler.
    n_trans = min(20, max(5, n_entries // 20))
    # pick companies (no cycling, cap at len(employers))
    n_trans = min(n_trans, len(employers))
    job_chain = employers[:n_trans]

    prev_employer_uuid: str | None = None
    job_history: list[tuple[str, str]] = []  # (employer, uuid)

    # Open with a few setup entries (entity intros)
    setup_entities = ["Jamie", "Marcus", "Priya", "Diego"]
    for name in setup_entities:
        ts += 1
        u = _uuid("e_", ts)
        text = f"@User's friend @{name} is mentioned."
        e = Entry(
            u,
            ts,
            text,
            _mentions("User", name),
            [],
            predicate=f"@{name}.relation_to_user",
            value="friend",
        )
        entries.append(e)

    # Interleave job transitions with filler
    trans_idx = 0
    while len(entries) < n_entries:
        if trans_idx < len(job_chain) and rng.random() < 0.35:
            employer = job_chain[trans_idx]
            ts += 1
            u = _uuid("e_", ts)
            refs = []
            if prev_employer_uuid is not None:
                refs = [Ref(prev_employer_uuid, "supersede")]
            text = f"@User started a new job at {employer}."
            e = Entry(
                u,
                ts,
                text,
                _mentions("User"),
                refs,
                predicate="@User.employer",
                value=employer,
                is_current=(trans_idx == len(job_chain) - 1),
            )
            # Mark previous as not current
            for h_emp, h_uuid in job_history:
                pass  # we'll compute current by supersede chain later
            entries.append(e)
            job_history.append((employer, u))
            prev_employer_uuid = u
            trans_idx += 1
        else:
            # filler — various activities about User
            ts += 1
            u = _uuid("e_", ts)
            templates = [
                f"@User had coffee with @{rng.choice(setup_entities)} today.",
                "@User is reading a book about productivity.",
                "@User went for a walk this morning.",
                "@User tried a new recipe for dinner.",
                "@User finished a big project at work.",
            ]
            t = rng.choice(templates)
            mentioned = ["User"]
            for ent in setup_entities:
                if f"@{ent}" in t:
                    mentioned.append(ent)
            e = Entry(u, ts, t, _mentions(*mentioned), [], predicate=None, value=None)
            entries.append(e)

    # Mark job-history is_current correctly: only last one is current.
    final_employer = job_chain[-1]

    # Questions
    qs: list[Question] = [
        Question(
            "D_Q01",
            "current",
            "What is User's current employer?",
            expected_contains=[final_employer],
            expected_absent=[],
            scenario_tag="dense",
        ),
        Question(
            "D_Q02",
            "history",
            "What is User's full job history, in order, from earliest to most recent?",
            expected_contains=job_chain,  # all names must appear
            scenario_tag="dense",
        ),
        Question(
            "D_Q03",
            "supersede",
            f"Did User ever work at {job_chain[0]}?",
            expected_contains=["yes"],
            scenario_tag="dense",
        ),
        Question(
            "D_Q04",
            "supersede",
            f"Does User currently work at {job_chain[0]}?",
            expected_contains=["no"],
            expected_absent=[],
            scenario_tag="dense",
        ),
    ]
    # Add a middle-of-chain question if chain long enough
    if len(job_chain) >= 4:
        mid_i = len(job_chain) // 2
        before = job_chain[mid_i - 1]
        after = job_chain[mid_i]
        qs.append(
            Question(
                "D_Q05",
                "supersede",
                f"What employer did User work at immediately before {after}?",
                expected_contains=[before],
                scenario_tag="dense",
            )
        )
    return entries, qs


# ---------------------------------------------------------------------------
# S-distractors: many entities, most mentioned once or twice
# ---------------------------------------------------------------------------


def gen_distractors(
    n_entries: int, seed: int = 2
) -> tuple[list[Entry], list[Question]]:
    rng = random.Random(seed)
    entries: list[Entry] = []
    ts = 0

    # Reserve a few "focus" entities with 5+ mentions; the rest get 1-2.
    # N_entities ~ n_entries * 0.6 to ensure density 1.5 mentions/entity.
    focus_names = ["Jamie", "Marcus", "Priya"]
    focus_facts = {
        "Jamie": (
            "@Jamie",
            "spouse",
            "Jamie is User's spouse, they got married last year.",
        ),
        "Marcus": (
            "@Marcus",
            "employer",
            "Marcus works at Anthropic as a staff engineer.",
        ),
        "Priya": ("@Priya", "employer", "Priya works at Google as a data scientist."),
    }
    # Build a pool of distractor names not in focus
    pool = [n for n in ENTITY_POOL if n not in focus_names]
    rng.shuffle(pool)

    # Record the ground truth for focus entities
    for name, (tag, pred, text) in focus_facts.items():
        ts += 1
        e = Entry(
            _uuid("e_", ts),
            ts,
            text,
            _mentions(name),
            [],
            predicate=f"{tag}.{pred}",
            value={"Jamie": "spouse", "Marcus": "Anthropic", "Priya": "Google"}[name],
        )
        entries.append(e)

    # For focus entities, sprinkle ~8 clarify/mention entries across the log
    focus_extra = {
        "Jamie": [
            "@Jamie cooked a great dinner last night.",
            "@Jamie took the dog for a walk.",
            "@Jamie is reading a mystery novel this month.",
            "@Jamie hates the rain in autumn.",
            "@Jamie plays tennis on weekends.",
            "@Jamie made coffee this morning.",
            "@Jamie is planning a weekend trip.",
            "@Jamie started a new hobby — pottery.",
        ],
        "Marcus": [
            "@Marcus wrote a proposal doc at work.",
            "@Marcus is mentoring a new hire.",
            "@Marcus gave a talk at the all-hands.",
            "@Marcus joined the book club.",
            "@Marcus took a PTO day on Monday.",
            "@Marcus switched desks to the quiet side.",
            "@Marcus recommended a podcast.",
            "@Marcus is running a half-marathon.",
        ],
        "Priya": [
            "@Priya shared a paper she's been reading.",
            "@Priya is on-call this week.",
            "@Priya started learning violin.",
            "@Priya moved apartments last month.",
            "@Priya made a great curry for the potluck.",
            "@Priya is training for a triathlon.",
            "@Priya refactored the pipeline.",
            "@Priya is taking a vacation next week.",
        ],
    }

    # Interleave distractors. Each distractor gets 1-2 entries.
    distractor_queue = pool[:]
    extra_queue = {k: v[:] for k, v in focus_extra.items()}
    focus_cycle = ["Jamie", "Marcus", "Priya"]
    focus_i = 0
    while len(entries) < n_entries:
        # 40% chance focus, 60% distractor
        if rng.random() < 0.4 and any(extra_queue.values()):
            # pick a focus entity with remaining extras
            for _ in range(3):
                name = focus_cycle[focus_i % 3]
                focus_i += 1
                if extra_queue[name]:
                    ts += 1
                    text = extra_queue[name].pop(0)
                    e = Entry(_uuid("e_", ts), ts, text, _mentions(name), [])
                    entries.append(e)
                    break
            else:
                # all extras done; drop through to distractor
                pass
        if len(entries) >= n_entries:
            break
        # distractor
        if not distractor_queue:
            distractor_queue = pool[:]
            rng.shuffle(distractor_queue)
        name = distractor_queue.pop()
        ts += 1
        templates = [
            f"@User saw @{name} at a coffee shop downtown.",
            f"@{name} sent @User a birthday card.",
            f"@{name} recommended a book to @User.",
            f"@User ran into @{name} at the park.",
            f"@{name} is starting a podcast soon.",
        ]
        t = rng.choice(templates)
        e = Entry(_uuid("e_", ts), ts, t, _mentions("User", name), [])
        entries.append(e)

    # Questions: entity-centric against focus (many mentions) and distractor
    qs = [
        Question(
            "X_Q01",
            "entity",
            "Tell me about Jamie — who are they to User and what do they do?",
            expected_contains=["spouse"],
            scenario_tag="distractors",
        ),
        Question(
            "X_Q02",
            "entity",
            "Where does Marcus work?",
            expected_contains=["Anthropic"],
            scenario_tag="distractors",
        ),
        Question(
            "X_Q03",
            "entity",
            "Where does Priya work?",
            expected_contains=["Google"],
            scenario_tag="distractors",
        ),
        Question(
            "X_Q04",
            "current",
            "What is Jamie's relationship to User?",
            expected_contains=["spouse"],
            scenario_tag="distractors",
        ),
    ]
    return entries, qs


# ---------------------------------------------------------------------------
# S-interleaved: realistic chat — many entities, scattered supersedes
# ---------------------------------------------------------------------------


def gen_interleaved(
    n_entries: int, seed: int = 3
) -> tuple[list[Entry], list[Question]]:
    rng = random.Random(seed)
    entries: list[Entry] = []
    ts = 0

    # Track predicate state: (entity, predicate) -> (current_value, current_uuid)
    state: dict[tuple[str, str], tuple[str, str]] = {}

    # Ground-truth state we care about
    tracked: list[tuple[str, str, list[str]]] = [
        ("@User", "employer", ["Stripe", "Anthropic", "Meta", "OpenAI"]),  # chain
        ("@User", "location", ["Seattle", "New York", "San Francisco"]),
        ("@User", "boss", ["Marcus", "Alice", "Theo", "Nadia"]),
        ("@Jamie", "job", ["freelance designer", "Pentagram designer"]),
        ("@Priya", "employer", ["Google", "Anthropic"]),
        ("@Priya", "role", ["senior engineer", "tech lead"]),
        ("@Diego", "role_to_user", ["gym trainer", "friend"]),
    ]

    # For supersede-chain traversal to work, we must fire each transition
    # in order, scattered across the log. We keep a pool of "pending transitions":
    pending: list[
        tuple[str, str, str, int]
    ] = []  # (entity, predicate, new_value, chain_pos)
    for ent, pred, chain in tracked:
        for pos, val in enumerate(chain):
            pending.append((ent, pred, val, pos))
    rng.shuffle(pending)
    # Re-order so that within each (entity, pred) the chain-order is preserved
    # but across different (entity, pred) they're interleaved.
    per_key: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for ent, pred, val, pos in pending:
        per_key.setdefault((ent, pred), []).append((pos, val))
    for k in per_key:
        per_key[k].sort()

    # Now schedule: pick a random (entity, pred) with remaining transitions and
    # fire the next one.
    keys = list(per_key.keys())
    # We want ~40% of entries to be supersede transitions, rest are filler.
    n_transitions = sum(len(v) for v in per_key.values())
    n_filler = max(0, n_entries - n_transitions)
    # Interleave: produce a schedule of 'transition' vs 'filler' slots
    schedule = ["T"] * n_transitions + ["F"] * n_filler
    rng.shuffle(schedule)

    # fill transitions in round-robin from per_key, preserving chain order
    cursors = dict.fromkeys(keys, 0)
    # But we want random selection among keys-that-still-have-more; with fixed ordering within key.

    # We'll iterate schedule:
    distractor_names = ENTITY_POOL[:40]

    for slot in schedule:
        ts += 1
        if slot == "T":
            # pick a random key with more transitions
            avail = [k for k in keys if cursors[k] < len(per_key[k])]
            if not avail:
                slot = "F"  # fall through
            else:
                k = rng.choice(avail)
                _, val = per_key[k][cursors[k]]
                cursors[k] += 1
                ent, pred = k
                name = ent.lstrip("@")
                refs = []
                prev = state.get(k)
                if prev is not None:
                    refs = [Ref(prev[1], "supersede")]
                # Render
                if pred == "employer":
                    text = f"{ent} is now working at {val}."
                elif pred == "location":
                    text = f"{ent} has relocated to {val}."
                elif pred == "boss":
                    text = f"{ent}'s new manager is @{val}."
                elif pred == "job":
                    text = f"{ent} is now a {val}."
                elif pred == "role":
                    text = f"{ent} got promoted to {val}."
                elif pred == "role_to_user":
                    text = f"{ent} is now @User's {val}."
                else:
                    text = f"{ent} {pred} changed to {val}."
                u = _uuid("e_", ts)
                mentioned = [name]
                # also mention User if applicable
                if name != "User":
                    pass  # keep only the subject entity
                if pred == "boss":
                    mentioned += [val]
                if pred == "role_to_user":
                    mentioned += ["User"]
                e = Entry(
                    u,
                    ts,
                    text,
                    _mentions(*mentioned),
                    refs,
                    predicate=f"{ent}.{pred}",
                    value=val,
                    is_current=(cursors[k] == len(per_key[k])),
                )
                entries.append(e)
                state[k] = (val, u)
                continue
        # Filler
        name = rng.choice(distractor_names)
        templates = [
            f"@User and @{name} chatted about weekend plans.",
            f"@{name} started a new hobby recently.",
            "@User read a good article today.",
            f"@{name} recommended a restaurant to @User.",
            f"@User went running with @{name}.",
        ]
        text = rng.choice(templates)
        mentioned = ["User", name] if f"@{name}" in text else []
        if not mentioned:
            mentioned = ["User"]
        e = Entry(_uuid("e_", ts), ts, text, _mentions(*mentioned), [])
        entries.append(e)

        if len(entries) >= n_entries:
            break

    # Truncate to n_entries
    entries = entries[:n_entries]

    # Questions — current-state for each tracked key, plus history
    qs = [
        Question(
            "I_Q01",
            "current",
            "What is User's current employer?",
            expected_contains=[tracked[0][2][-1]],
            scenario_tag="interleaved",
        ),
        Question(
            "I_Q02",
            "current",
            "Where does User currently live?",
            expected_contains=[tracked[1][2][-1]],
            scenario_tag="interleaved",
        ),
        Question(
            "I_Q03",
            "current",
            "Who is User's current manager/boss?",
            expected_contains=[tracked[2][2][-1]],
            scenario_tag="interleaved",
        ),
        Question(
            "I_Q04",
            "current",
            "What is Jamie's current job?",
            expected_contains=["Pentagram"],
            scenario_tag="interleaved",
        ),
        Question(
            "I_Q05",
            "current",
            "Where does Priya work now?",
            expected_contains=["Anthropic"],
            scenario_tag="interleaved",
        ),
        Question(
            "I_Q06",
            "current",
            "What is Priya's current role?",
            expected_contains=["tech lead"],
            scenario_tag="interleaved",
        ),
        Question(
            "I_Q07",
            "current",
            "What is Diego's role to User now?",
            expected_contains=["friend"],
            scenario_tag="interleaved",
        ),
        Question(
            "I_Q08",
            "history",
            "List Priya's employers in chronological order.",
            expected_contains=["Google", "Anthropic"],
            scenario_tag="interleaved",
        ),
        Question(
            "I_Q09",
            "history",
            "List User's bosses in chronological order.",
            expected_contains=["Marcus", "Alice", "Theo", "Nadia"],
            scenario_tag="interleaved",
        ),
        Question(
            "I_Q10",
            "supersede",
            "Is User still working at Stripe?",
            expected_contains=["no"],
            scenario_tag="interleaved",
        ),
    ]
    return entries, qs


# ---------------------------------------------------------------------------
# S-deep-chain: one predicate, 10-15 transitions
# ---------------------------------------------------------------------------


def gen_deep_chain(
    n_entries: int,
    seed: int = 4,
    chain_len: int = 12,
) -> tuple[list[Entry], list[Question]]:
    rng = random.Random(seed)
    entries: list[Entry] = []
    ts = 0

    # Deep chain on User.boss with N transitions
    boss_chain = [rng.choice(ENTITY_POOL) for _ in range(chain_len)]
    # Dedupe preserving order
    seen = set()
    deduped = []
    for b in boss_chain:
        if b not in seen:
            seen.add(b)
            deduped.append(b)
    boss_chain = deduped[:chain_len]
    # Ensure length
    while len(boss_chain) < chain_len:
        candidate = rng.choice(ENTITY_POOL)
        if candidate not in boss_chain:
            boss_chain.append(candidate)

    # Schedule: chain_len transitions scattered among n_entries with fillers.
    n_filler = max(0, n_entries - chain_len)
    schedule = ["T"] * chain_len + ["F"] * n_filler
    rng.shuffle(schedule)

    prev_uuid = None
    chain_cursor = 0
    distractors = ENTITY_POOL[:40]

    for slot in schedule:
        ts += 1
        if slot == "T" and chain_cursor < len(boss_chain):
            boss = boss_chain[chain_cursor]
            chain_cursor += 1
            u = _uuid("e_", ts)
            refs = []
            if prev_uuid is not None:
                refs = [Ref(prev_uuid, "supersede")]
            text = f"@User's new manager is @{boss}."
            e = Entry(
                u,
                ts,
                text,
                _mentions("User", boss),
                refs,
                predicate="@User.boss",
                value=boss,
                is_current=(chain_cursor == len(boss_chain)),
            )
            entries.append(e)
            prev_uuid = u
        else:
            # filler
            name = rng.choice(distractors)
            templates = [
                "@User went to the gym today.",
                f"@User had lunch with @{name}.",
                f"@{name} called @User about a project.",
                "@User is trying a new coffee shop.",
                "@User is traveling next week.",
            ]
            t = rng.choice(templates)
            mentioned = ["User"]
            if f"@{name}" in t:
                mentioned.append(name)
            e = Entry(_uuid("e_", ts), ts, t, _mentions(*mentioned), [])
            entries.append(e)
        if len(entries) >= n_entries:
            break

    qs = [
        Question(
            "C_Q01",
            "current",
            "Who is User's current manager?",
            expected_contains=[boss_chain[-1]],
            scenario_tag="deep_chain",
        ),
        Question(
            "C_Q02",
            "history",
            "List User's managers in order, from earliest to most recent.",
            expected_contains=boss_chain,  # all must appear
            scenario_tag="deep_chain",
        ),
        Question(
            "C_Q03",
            "supersede",
            f"Is {boss_chain[0]} still User's manager?",
            expected_contains=["no"],
            scenario_tag="deep_chain",
        ),
        Question(
            "C_Q04",
            "supersede",
            f"Was {boss_chain[len(boss_chain) // 2]} ever User's manager?",
            expected_contains=["yes"],
            scenario_tag="deep_chain",
        ),
    ]
    return entries, qs


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS = {
    "dense": gen_dense,
    "distractors": gen_distractors,
    "interleaved": gen_interleaved,
    "deep_chain": gen_deep_chain,
}
