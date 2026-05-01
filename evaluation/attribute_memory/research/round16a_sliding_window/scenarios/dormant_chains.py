"""Dormant-chain scenario: long quiet periods between updates on the same chain.

User has 5 chains (boss, location, partner, hobby, employer). Each chain has
5-8 transitions, but transitions on the same chain are separated by 50-100
turns of unrelated content (other chains' transitions, filler, detail).

This stresses the writer's ability to look up the chain head AFTER a long quiet
period — exactly the scenario where batch-boundary writers lose context but
sliding windows + active-chain injection should hold up.

Target: ~600 turns.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

TurnKind = Literal["intro", "state", "clarify", "filler"]


@dataclass
class Turn:
    idx: int
    text: str
    kind: TurnKind
    mentions: list[str] = field(default_factory=list)
    predicate_updates: dict[tuple[str, str], str] = field(default_factory=dict)
    intro_entity: str | None = None


# ---------------------------------------------------------------------
# Pools
# ---------------------------------------------------------------------
BOSSES = ["Marcus", "Alice", "Theo", "Nadia", "Priya", "Quentin", "Rhea", "Sana"]
CITIES = [
    "Seattle",
    "Brooklyn",
    "Austin",
    "Chicago",
    "Denver",
    "Portland",
    "Boston",
    "Miami",
]
PARTNERS = ["Jamie", "Robin", "Sam", "Casey", "Morgan", "River"]  # progression sequence
HOBBIES = [
    "climbing",
    "running",
    "pottery",
    "cycling",
    "guitar",
    "photography",
    "woodworking",
    "chess",
]
EMPLOYERS = [
    "Stripe",
    "Anthropic",
    "Google",
    "OpenAI",
    "Notion",
    "Figma",
    "Datadog",
    "Vercel",
]


FILLER = [
    "Coffee was good this morning.",
    "Weather is gross again.",
    "Traffic was awful.",
    "Long day, slow afternoon.",
    "Working from a cafe today.",
    "Just got back from a run.",
    "Tired. Need a nap.",
    "Going to grab lunch soon.",
    "Been on calls all morning.",
    "Stomach hurts, too much espresso.",
    "Garbage truck woke me up.",
    "Rainy again.",
    "The office printer is broken.",
    "Cold brew is hitting different.",
    "Slack is laggy.",
    "Email avalanche this morning.",
    "Slow news day.",
    "Watching a movie tonight.",
    "Listening to a new album. Mid.",
    "Should probably hydrate more.",
    "It's freezing in here.",
    "Got a paper cut, very dramatic.",
    "OK gonna log off, ttyl.",
    "Inbox is at 412 unread.",
    "Pretty mellow morning.",
]


def generate(
    seed: int = 23,
    target_turns: int = 600,
    chain_length: int = 6,
    min_gap: int = 50,
    max_gap: int = 100,
) -> list[Turn]:
    rng = random.Random(seed)
    turns: list[Turn] = []
    entities_introduced: set[str] = set()

    # Plans: each chain has chain_length values to walk through.
    plans: dict[tuple[str, str], list[str]] = {
        ("@User", "boss"): BOSSES[:chain_length],
        ("@User", "location"): CITIES[:chain_length],
        ("@User", "partner"): PARTNERS[:chain_length],
        ("@User", "hobby"): HOBBIES[:chain_length],
        ("@User", "employer"): EMPLOYERS[:chain_length],
    }
    cursors: dict[tuple[str, str], int] = dict.fromkeys(plans, 0)

    def add(
        text: str,
        kind: TurnKind,
        *,
        mentions: list[str] | None = None,
        predicate_updates: dict[tuple[str, str], str] | None = None,
        intro_entity: str | None = None,
    ) -> None:
        turns.append(
            Turn(
                idx=len(turns) + 1,
                text=text,
                kind=kind,
                mentions=mentions or [],
                predicate_updates=predicate_updates or {},
                intro_entity=intro_entity,
            )
        )

    def filler():
        add(rng.choice(FILLER), "filler")

    def emit(key: tuple[str, str]) -> bool:
        ent, pred = key
        plan = plans[key]
        i = cursors[key]
        if i >= len(plan):
            return False
        new = plan[i]
        cursors[key] += 1
        old = plan[i - 1] if i >= 1 else None
        ent_name = ent.lstrip("@")

        if pred == "boss":
            entities_introduced.add(new)
            if old is None:
                text = f"My new manager started today — {new}."
                mentions = ["User", new]
            else:
                text = f"{new} is my new manager now — {old} switched teams."
                mentions = ["User", new, old]
            add(text, "state", mentions=mentions, predicate_updates={key: new})
        elif pred == "location":
            if old is None:
                text = f"We've been living in {new} for a while."
            else:
                text = f"We're moving — packing up for {new}."
            add(text, "state", mentions=["User"], predicate_updates={key: new})
        elif pred == "partner":
            entities_introduced.add(new)
            if old is None:
                text = f"My partner is {new} — we've been dating about a year."
                mentions = ["User", new]
                add(
                    text,
                    "state",
                    mentions=mentions,
                    predicate_updates={key: new},
                    intro_entity=new,
                )
            else:
                text = f"{old} and I broke up. I'm dating {new} now."
                mentions = ["User", new, old]
                add(text, "state", mentions=mentions, predicate_updates={key: new})
        elif pred == "hobby":
            if old is None:
                text = f"Picked up a new hobby — {new}."
            else:
                text = f"Dropped {old}, my new thing is {new}."
            add(text, "state", mentions=["User"], predicate_updates={key: new})
        elif pred == "employer":
            if old is None:
                text = f"I work at {new} these days."
            else:
                text = f"Big news — I'm starting at {new} next week."
            add(text, "state", mentions=["User"], predicate_updates={key: new})
        else:
            text = f"{ent_name}'s {pred} is now {new}."
            add(text, "state", mentions=["User"], predicate_updates={key: new})
        return True

    # ---- Strategy: schedule transitions across the conversation such that
    # consecutive updates on the SAME chain are min_gap..max_gap turns apart.
    # ----
    chain_keys = list(plans.keys())
    # last_emit_turn[key] = turn idx of last emit on that chain (or -inf)
    last_emit_turn: dict[tuple[str, str], int] = dict.fromkeys(plans, -(10**9))
    pending_keys = list(chain_keys)

    # Seed each chain with its first transition spread early
    rng.shuffle(pending_keys)
    cursor_pos = 0
    for k in pending_keys:
        # space initial intros across first ~120 turns
        target = cursor_pos + rng.randint(8, 25)
        while len(turns) < target:
            filler()
        emit(k)
        last_emit_turn[k] = len(turns)
        cursor_pos = len(turns)

    # Now walk through remaining transitions, scheduling each so the gap on
    # its chain is in [min_gap, max_gap].
    while True:
        eligible = [k for k in chain_keys if cursors[k] < len(plans[k])]
        if not eligible:
            break
        # Pick the chain whose next-eligible-time (last_emit_turn + min_gap)
        # is earliest -> drives the conversation forward.
        eligible.sort(key=lambda k: last_emit_turn[k] + min_gap)
        target_key = eligible[0]
        # Determine when to emit for this chain
        earliest = last_emit_turn[target_key] + min_gap
        latest = last_emit_turn[target_key] + max_gap
        target_turn = max(
            len(turns) + 1,
            rng.randint(max(earliest, len(turns) + 1), max(latest, len(turns) + 2)),
        )
        # Don't go past target_turns - leave a tail
        if target_turn > target_turns - 5:
            break
        while len(turns) < target_turn - 1:
            filler()
        emit(target_key)
        last_emit_turn[target_key] = len(turns)

    # Tail filler to reach target_turns
    while len(turns) < target_turns:
        filler()

    # Renumber
    for i, t in enumerate(turns, start=1):
        t.idx = i
    return turns


# ---------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------


@dataclass
class GroundTruth:
    chains: dict[tuple[str, str], list[tuple[int, str]]] = field(default_factory=dict)
    intros: dict[str, int] = field(default_factory=dict)

    def current_value(self, key):
        chain = self.chains.get(key)
        return chain[-1][1] if chain else None

    def chain_values(self, key):
        return [v for _, v in self.chains.get(key, [])]


def ground_truth(turns: list[Turn]) -> GroundTruth:
    gt = GroundTruth()
    for t in turns:
        if t.intro_entity and t.intro_entity not in gt.intros:
            gt.intros[t.intro_entity] = t.idx
        for k, v in t.predicate_updates.items():
            gt.chains.setdefault(k, []).append((t.idx, v))
    return gt


# ---------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------


@dataclass
class Question:
    qid: str
    kind: str
    question: str
    expected_contains: list[str]
    expected_absent: list[str] = field(default_factory=list)


def build_questions(gt: GroundTruth) -> list[Question]:
    qs: list[Question] = []
    boss = gt.chain_values(("@User", "boss"))
    city = gt.chain_values(("@User", "location"))
    partner = gt.chain_values(("@User", "partner"))
    hobby = gt.chain_values(("@User", "hobby"))
    emp = gt.chain_values(("@User", "employer"))

    def add(qid, kind, q, contains, absent=None):
        qs.append(Question(qid, kind, q, contains, absent or []))

    # Current values (5)
    if boss:
        add("Q01", "current", "Who is User's current manager?", [boss[-1]])
    if city:
        add("Q02", "current", "Where does User currently live?", [city[-1]])
    if partner:
        add("Q03", "current", "Who is User's current partner?", [partner[-1]])
    if hobby:
        add("Q04", "current", "What hobby is User into now?", [hobby[-1]])
    if emp:
        add("Q05", "current", "Where does User work now?", [emp[-1]])

    # Supersede (no longer / first)
    if len(boss) >= 2:
        add("Q06", "supersede", f"Is {boss[0]} still User's manager?", ["no"])
        add("Q07", "supersede", f"Was {boss[0]} ever User's manager?", ["yes"])
    if len(city) >= 2:
        add("Q08", "supersede", f"Does User live in {city[0]} now?", ["no"])
    if len(partner) >= 2:
        add("Q09", "supersede", f"Is {partner[0]} User's current partner?", ["no"])
        add("Q10", "supersede", f"Did User ever date {partner[0]}?", ["yes"])

    # History (chain depth)
    if len(boss) >= 3:
        add("Q11", "history", "Who was User's first manager?", [boss[0]])
    if len(boss) >= 4:
        add("Q12", "history", "Who was User's 3rd manager?", [boss[2]])
    if len(city) >= 3:
        add(
            "Q13",
            "history",
            "What's the first city User mentioned living in?",
            [city[0]],
        )
    if len(emp) >= 3:
        add("Q14", "history", "How many employers has User had?", [str(len(emp))])
    if len(partner) >= 3:
        add("Q15", "history", "List User's partners in order.", partner)
    if len(hobby) >= 3:
        add(
            "Q16",
            "history",
            "How many different hobbies has User had?",
            [str(len(hobby))],
        )

    # Mid-chain "ever" questions (specifically test dormant lookup)
    if len(boss) >= 4:
        mid = boss[len(boss) // 2]
        add("Q17", "supersede", f"Was {mid} ever User's manager?", ["yes"])
    if len(city) >= 4:
        mid_c = city[len(city) // 2]
        add("Q18", "supersede", f"Did User ever live in {mid_c}?", ["yes"])
    if len(emp) >= 4:
        mid_e = emp[len(emp) // 2]
        add("Q19", "supersede", f"Did User ever work at {mid_e}?", ["yes"])

    # First/last comparisons
    if len(boss) >= 3:
        add(
            "Q20",
            "history",
            "Who was User's manager right before the current one?",
            [boss[-2]],
        )

    return qs


if __name__ == "__main__":
    turns = generate()
    gt = ground_truth(turns)
    qs = build_questions(gt)
    kinds = {}
    for t in turns:
        kinds[t.kind] = kinds.get(t.kind, 0) + 1
    print(f"turns: {len(turns)}")
    print(f"kind distribution: {kinds}")
    for k, vs in gt.chains.items():
        ts = [t for t, _ in vs]
        gaps = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
        print(f"  {k[0]}.{k[1]}: {len(vs)} -> turns={ts} gaps={gaps}")
    print(f"# questions: {len(qs)}")
