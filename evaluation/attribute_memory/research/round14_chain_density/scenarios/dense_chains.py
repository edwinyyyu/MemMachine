"""High-density chain scenario generator (round 14).

Produces ~700 turns with ~110 ground-truth supersede transitions distributed
across deep chains. Several predicates have 10+ transitions on the same entity
so we can stress-test the writer's chain-integrity at the tail.

Mix targets:
  - State updates (chain transitions): ~50% (heavy)
  - Filler/phatic: ~30%
  - Detail/clarification: ~15%
  - Multi-entity events: ~5%

Deep chains (single (entity, predicate) pair):
  - @User.boss     : 12 transitions
  - @User.location : 12 transitions
  - @User.title    : 10 transitions
  - @User.employer : 10 transitions
  - @User.team     :  8 transitions
  - @User.hobby    :  8 transitions  (start/stop cycles)
  - @User.commute  :  6 transitions
  - @Jamie.job     :  8 transitions
  - @Jamie.hobby   :  6 transitions
  - @Marcus.role   :  4 transitions  (Marcus moved teams a few times)
  - @User.partner_state : 3 transitions (partner -> fiance -> spouse)
  Total non-first transitions: ~75-80, total ground-truth values: ~85+

We bump it further with @User.car (5), @Sam.role (4), @User.gym (4) for
~100+ non-first transitions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

TurnKind = Literal["intro", "state", "clarify", "multi", "filler"]


@dataclass
class Turn:
    idx: int
    text: str
    kind: TurnKind
    mentions: list[str] = field(default_factory=list)
    predicate_updates: dict[tuple[str, str], str] = field(default_factory=dict)
    intro_entity: str | None = None


# ---------------------------------------------------------------------
# Pools (large enough for deep chains)
# ---------------------------------------------------------------------
BOSSES = [
    "Marcus",
    "Alice",
    "Theo",
    "Nadia",
    "Priya",
    "Quentin",
    "Rhea",
    "Sana",
    "Tobias",
    "Uma",
    "Vik",
    "Wren",
]
CITIES = [
    "Seattle",
    "Brooklyn",
    "Austin",
    "Chicago",
    "Denver",
    "Portland",
    "Boston",
    "Miami",
    "Phoenix",
    "Atlanta",
    "Pittsburgh",
    "Nashville",
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
    "Replit",
    "Linear",
]
TITLES = [
    "engineer",
    "senior engineer",
    "staff engineer",
    "principal engineer",
    "tech lead",
    "engineering manager",
    "director",
    "VP engineering",
    "CTO",
    "founder",
]
TEAMS = [
    "platform",
    "infra",
    "applied",
    "research",
    "product",
    "growth",
    "internal-tools",
    "security",
]
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
COMMUTES = ["bike", "subway", "walking", "driving", "scooter", "remote"]
JAMIE_JOBS = [
    "barista",
    "graphic designer",
    "UX researcher",
    "freelance writer",
    "product manager",
    "art director",
    "creative director",
    "consultant",
]
JAMIE_HOBBIES = ["pottery", "yoga", "knitting", "watercolor", "improv", "rock-climbing"]
MARCUS_ROLES = [
    "my manager",
    "my skip-level",
    "VP of engineering",
    "advisor (he moved companies)",
]
CARS = [
    "used Civic",
    "leased Tesla Model 3",
    "minivan",
    "Subaru Outback",
    "electric Hyundai",
]
SAM_ROLES = [
    "intern on my team",
    "my new hire",
    "my tech lead",
    "manager of platform team",
]
GYMS = [
    "the corner CrossFit",
    "Equinox",
    "neighborhood YMCA",
    "the climbing gym downtown",
]
PARTNER_STATES = ["partner", "fiance", "spouse"]


FILLER = [
    "Coffee was good this morning.",
    "Weather is gross again.",
    "Traffic was awful.",
    "Long day, slow afternoon.",
    "Working from a cafe today.",
    "Just got back from a run.",
    "Tired. Need a nap.",
    "That meme made me laugh.",
    "Going to grab lunch soon.",
    "Been on calls all morning.",
    "Stomach hurts, too much espresso.",
    "Garbage truck woke me up.",
    "Rainy again.",
    "The office printer is broken.",
    "Pretty mellow morning so far.",
    "Cold brew is hitting different today.",
    "My standing desk is making weird noises.",
    "Slack is laggy.",
    "Email avalanche this morning.",
    "Slow news day.",
    "Random thought: croissants are underrated.",
    "Watching a movie tonight, undecided which.",
    "Listening to a new album. Mid.",
    "Weekend plans: nothing, glorious nothing.",
    "Should probably hydrate more.",
    "It's freezing in here.",
    "Got a paper cut, very dramatic.",
    "Cleaning my desk. Found a granola bar from 2023.",
    "OK gonna log off, ttyl.",
    "Inbox at 412 unread, ignoring.",
]

DETAIL_BOSS = [
    "{boss} runs really tight 1:1s, I appreciate that.",
    "{boss} just sent over reading material for the team.",
    "{boss} is into endurance sports apparently.",
    "{boss} has weirdly good taste in restaurants.",
]

DETAIL_JAMIE = [
    "Jamie made an incredible dinner last night.",
    "Jamie's been collecting weird old camera lenses.",
    "Jamie binged a new show in two days, called it 'mid'.",
    "Jamie won a small art contest last weekend.",
]

DETAIL_LUNA = [
    "Luna knocked over a plant. Again.",
    "Luna is sleeping on my keyboard.",
    "Luna got super spooked by a delivery person.",
    "Luna is being clingy today.",
]

CLARIFY_USER = [
    "I should add — most of my work is async these days.",
    "Half my role is mentoring, honestly.",
    "I'm on the platform side, not applied.",
    "Lots of my time is spent in design review meetings.",
]


def generate(seed: int = 17) -> list[Turn]:
    rng = random.Random(seed)
    turns: list[Turn] = []

    # ----- planned chains (entity, predicate) -> list of values -----
    plans: dict[tuple[str, str], list[str]] = {
        ("@User", "boss"): BOSSES[:12],  # 12
        ("@User", "location"): CITIES[:12],  # 12
        ("@User", "title"): TITLES[:10],  # 10
        ("@User", "employer"): EMPLOYERS[:10],  # 10
        ("@User", "team"): TEAMS[:8],  # 8
        ("@User", "hobby"): HOBBIES[:8],  # 8
        ("@User", "commute"): COMMUTES[:6],  # 6
        ("@Jamie", "job"): JAMIE_JOBS[:8],  # 8
        ("@Jamie", "hobby"): JAMIE_HOBBIES[:6],  # 6
        ("@Marcus", "role"): MARCUS_ROLES[:4],  # 4
        ("@User", "car"): CARS[:5],  # 5
        ("@Sam", "role"): SAM_ROLES[:4],  # 4
        ("@User", "gym"): GYMS[:4],  # 4
        ("@User", "partner_state"): PARTNER_STATES[:3],  # 3
    }

    # entity intros done lazily
    entities_introduced: set[str] = set()

    # cursors
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

    def filler_turn():
        add(rng.choice(FILLER), "filler")

    def boss_detail(boss):
        add(
            rng.choice(DETAIL_BOSS).format(boss=boss),
            "clarify",
            mentions=["User", boss],
        )

    def jamie_detail():
        add(rng.choice(DETAIL_JAMIE), "clarify", mentions=["User", "Jamie"])

    def luna_detail():
        add(rng.choice(DETAIL_LUNA), "clarify", mentions=["User", "Luna"])

    def user_clarify():
        add(rng.choice(CLARIFY_USER), "clarify", mentions=["User"])

    def multi_event():
        if "Jamie" not in entities_introduced:
            filler_turn()
            return
        opts = []
        if "Luna" in entities_introduced:
            opts += [
                "Jamie and Luna are having a standoff over the new chair.",
                "Jamie took Luna to the vet, all clear.",
                "Jamie and I took Luna to a friend's place.",
            ]
            mentions = ["User", "Jamie", "Luna"]
        else:
            opts += [
                "Jamie and I tried a new restaurant last night.",
                "Jamie and I went on a hike on Saturday.",
            ]
            mentions = ["User", "Jamie"]
        add(rng.choice(opts), "multi", mentions=mentions)

    # ---- transition emitters ----
    def emit_transition(key: tuple[str, str]) -> bool:
        ent, pred = key
        plan = plans[key]
        i = cursors[key]
        if i >= len(plan):
            return False
        new = plan[i]
        cursors[key] += 1
        old_idx = i - 1
        old = plan[old_idx] if old_idx >= 0 else None

        ent_name = ent.lstrip("@")
        intro_marker = None
        # Intro entity if needed (Jamie, Marcus, Sam, Luna)
        if ent_name not in entities_introduced and ent_name != "User":
            entities_introduced.add(ent_name)
            intro_marker = ent_name

        if pred == "boss":
            entities_introduced.add(new)  # the boss themselves
            if old is None:
                text = f"My new manager started today — {new}."
                mentions = ["User", new]
            else:
                text = f"{new} is my new manager now — {old} moved to a different team."
                mentions = ["User", new, old]
            add(text, "state", mentions=mentions, predicate_updates={key: new})
        elif pred == "location":
            if old is None:
                text = f"We've been living in {new} for a while now."
            else:
                text = f"We're moving again — packing up for {new}."
            add(text, "state", mentions=["User"], predicate_updates={key: new})
        elif pred == "title":
            if old is None:
                text = f"I'm a {new} — that's my role."
            else:
                text = f"Got promoted — I'm a {new} now."
            add(text, "state", mentions=["User"], predicate_updates={key: new})
        elif pred == "employer":
            if old is None:
                text = f"I work at {new} these days."
            else:
                text = f"Big change — I'm starting at {new} next week."
            add(text, "state", mentions=["User"], predicate_updates={key: new})
        elif pred == "team":
            if old is None:
                text = f"I'm on the {new} team."
            else:
                text = f"Switched teams — I'm on the {new} team now."
            add(text, "state", mentions=["User"], predicate_updates={key: new})
        elif pred == "hobby":
            if old is None:
                text = f"Picked up a new hobby — {new}."
            else:
                text = f"Dropped {old}, my new thing is {new}."
            add(text, "state", mentions=["User"], predicate_updates={key: new})
        elif pred == "commute":
            if old is None:
                text = f"I commute by {new}."
            else:
                text = (
                    f"Switched up my commute — taking the {new} now instead of {old}."
                )
            add(text, "state", mentions=["User"], predicate_updates={key: new})
        elif pred == "car":
            if old is None:
                text = f"Just got a {new} as our main car."
            else:
                text = f"Traded in the {old}; we got a {new} now."
            add(text, "state", mentions=["User"], predicate_updates={key: new})
        elif pred == "gym":
            if old is None:
                text = f"Started going to {new} this month."
            else:
                text = f"Quit {old}, switched over to {new}."
            add(text, "state", mentions=["User"], predicate_updates={key: new})
        elif pred == "partner_state":
            if new == "partner":
                text = "Jamie is my partner — we've been dating about a year."
                mentions = ["User", "Jamie"]
                entities_introduced.add("Jamie")
            elif new == "fiance":
                text = "We got engaged last weekend! Jamie said yes."
                mentions = ["User", "Jamie"]
            else:
                text = "Jamie and I got married last month — courthouse ceremony."
                mentions = ["User", "Jamie"]
            add(
                text,
                "state",
                mentions=mentions,
                predicate_updates={key: new},
                intro_entity=("Jamie" if intro_marker == "Jamie" else None),
            )
        elif ent_name == "Jamie" and pred == "job":
            if "Jamie" not in entities_introduced:
                entities_introduced.add("Jamie")
            if old is None:
                text = f"Jamie just took a job as a {new}."
            else:
                text = f"Jamie quit being a {old} — they're a {new} now."
            add(text, "state", mentions=["User", "Jamie"], predicate_updates={key: new})
        elif ent_name == "Jamie" and pred == "hobby":
            if old is None:
                text = f"Jamie has been getting into {new}."
            else:
                text = f"Jamie is bored of {old} — moved on to {new}."
            add(text, "state", mentions=["User", "Jamie"], predicate_updates={key: new})
        elif ent_name == "Marcus" and pred == "role":
            entities_introduced.add("Marcus")
            text = f"Marcus is now {new}."
            add(
                text, "state", mentions=["User", "Marcus"], predicate_updates={key: new}
            )
        elif ent_name == "Sam" and pred == "role":
            if "Sam" not in entities_introduced:
                entities_introduced.add("Sam")
                text = f"Sam joined — they're {new}."
            else:
                text = f"Sam moved up — now {new}."
            add(text, "state", mentions=["User", "Sam"], predicate_updates={key: new})
        else:
            # Generic fallback
            text = f"{ent_name}'s {pred} is now {new}."
            add(
                text, "state", mentions=["User", ent_name], predicate_updates={key: new}
            )
        return True

    # ---- gap filler: insert a few non-state turns between transitions ----
    def gap(min_turns: int, max_turns: int) -> None:
        n = rng.randint(min_turns, max_turns)
        for _ in range(n):
            r = rng.random()
            if r < 0.55:
                filler_turn()
            elif r < 0.7:
                # boss detail (use current boss if known)
                cur_boss_idx = cursors[("@User", "boss")] - 1
                if cur_boss_idx >= 0:
                    boss_detail(plans[("@User", "boss")][cur_boss_idx])
                else:
                    filler_turn()
            elif r < 0.82:
                jamie_detail() if "Jamie" in entities_introduced else filler_turn()
            elif r < 0.9:
                luna_detail() if "Luna" in entities_introduced else filler_turn()
            elif r < 0.96:
                user_clarify()
            else:
                multi_event()

    # ---- intro Jamie + Luna early so they're available for chains ----
    add(
        "Hey — just finished another day at work. I'm a software engineer.",
        "state",
        mentions=["User"],
        predicate_updates={("@User", "title"): "engineer"},
    )
    cursors[("@User", "title")] = 1  # consumed "engineer"
    # Introduce Jamie via partner_state
    emit_transition(("@User", "partner_state"))  # partner

    # Introduce Luna early
    add(
        "We adopted a cat today — her name is Luna, a tortoiseshell.",
        "intro",
        mentions=["User", "Luna"],
        intro_entity="Luna",
    )
    entities_introduced.add("Luna")

    # Initial location, employer, boss, team, hobby, commute, car, gym
    emit_transition(("@User", "location"))  # Seattle
    emit_transition(("@User", "employer"))  # Stripe
    emit_transition(("@User", "boss"))  # Marcus
    emit_transition(("@Marcus", "role"))  # Marcus's first role
    emit_transition(("@User", "team"))  # platform
    emit_transition(("@User", "hobby"))  # climbing
    emit_transition(("@User", "commute"))  # bike
    emit_transition(("@User", "car"))  # used Civic
    emit_transition(("@User", "gym"))  # corner CrossFit
    emit_transition(("@Jamie", "job"))  # barista
    emit_transition(("@Jamie", "hobby"))  # pottery

    # ----------------------------------------------------------------
    # Now we have ~13 turns done. We schedule ~95 more transitions
    # spread across ~700 turns total. We'll cycle through a transition
    # plan that picks the predicate with most remaining values, plus
    # some random churn for realism.
    # ----------------------------------------------------------------

    def remaining_keys() -> list[tuple[str, str]]:
        return [k for k in plans if cursors[k] < len(plans[k])]

    target_total = 700
    while len(turns) < target_total:
        # Bias: with prob 0.55, do a transition; else gap.
        if rng.random() < 0.55 and remaining_keys():
            # Pick: 70% chance pick the chain with the most values left
            # (drives deep chains forward), 30% random.
            keys = remaining_keys()
            if rng.random() < 0.7:
                keys.sort(key=lambda k: -(len(plans[k]) - cursors[k]))
                key = keys[0]
            else:
                key = rng.choice(keys)
            emit_transition(key)
            # immediate gap of 2-6 filler/detail turns
            gap(2, 6)
        else:
            gap(3, 8)

    # ---- tail stress: ensure we burn through *all* remaining transitions
    # in the last ~100 turns so late-bucket chains exist ----
    while remaining_keys() and len(turns) < target_total + 100:
        keys = remaining_keys()
        # Pick longest remaining chain
        keys.sort(key=lambda k: -(len(plans[k]) - cursors[k]))
        key = keys[0]
        emit_transition(key)
        gap(1, 4)

    # If we still have fewer than ~750 turns and unexhausted, keep going
    while remaining_keys():
        emit_transition(remaining_keys()[0])
        if rng.random() < 0.5:
            filler_turn()

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
# Questions  (~30 chain-traversal questions)
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
    emp = gt.chain_values(("@User", "employer"))
    title = gt.chain_values(("@User", "title"))
    team = gt.chain_values(("@User", "team"))
    hobby = gt.chain_values(("@User", "hobby"))
    commute = gt.chain_values(("@User", "commute"))
    jamie_job = gt.chain_values(("@Jamie", "job"))
    car = gt.chain_values(("@User", "car"))
    gym = gt.chain_values(("@User", "gym"))

    def add(qid, kind, q, contains, absent=None):
        qs.append(Question(qid, kind, q, contains, absent or []))

    # Current values (8)
    if boss:
        add("Q01", "current", "Who is User's current manager?", [boss[-1]])
    if city:
        add("Q02", "current", "Where does User currently live?", [city[-1]])
    if emp:
        add("Q03", "current", "Where does User work now?", [emp[-1]])
    if title:
        add("Q04", "current", "What is User's current job title?", [title[-1]])
    if team:
        add("Q05", "current", "Which team is User on right now?", [team[-1]])
    if hobby:
        add("Q06", "current", "What hobby is User into these days?", [hobby[-1]])
    if commute:
        add("Q07", "current", "How does User commute now?", [commute[-1]])
    if jamie_job:
        add("Q08", "current", "What does Jamie do for work right now?", [jamie_job[-1]])

    # Deep-chain "Nth value" questions (chain traversal at depth)
    if len(boss) >= 5:
        add("Q09", "history", "Who was User's 5th manager?", [boss[4]])
    if len(boss) >= 8:
        add("Q10", "history", "Who was User's 8th manager?", [boss[7]])
    if len(city) >= 5:
        add("Q11", "history", "What was the 5th city User lived in?", [city[4]])
    if len(city) >= 8:
        add("Q12", "history", "What was the 8th city User lived in?", [city[7]])
    if len(emp) >= 5:
        add("Q13", "history", "What was User's 5th employer?", [emp[4]])
    if len(emp) >= 8:
        add("Q14", "history", "What was User's 8th employer?", [emp[7]])
    if len(title) >= 5:
        add("Q15", "history", "What was User's 5th job title?", [title[4]])

    # Counts ("how many X has User had")
    if len(boss) >= 5:
        add(
            "Q16",
            "history",
            "How many different managers has User had over the conversation?",
            [str(len(boss))],
        )
    if len(city) >= 5:
        add(
            "Q17",
            "history",
            "How many different cities has User lived in?",
            [str(len(city))],
        )
    if len(emp) >= 5:
        add("Q18", "history", "How many employers has User worked at?", [str(len(emp))])

    # Supersede / "ever" (no longer the case)
    if len(boss) >= 3:
        add("Q19", "supersede", f"Is {boss[0]} still User's manager?", ["no"])
        add(
            "Q20",
            "supersede",
            f"Was {boss[len(boss) // 2]} ever User's manager?",
            ["yes"],
        )
    if len(city) >= 3:
        add("Q21", "supersede", f"Does User currently live in {city[0]}?", ["no"])
        add(
            "Q22",
            "supersede",
            f"Did User ever live in {city[len(city) // 2]}?",
            ["yes"],
        )
    if len(emp) >= 3:
        add("Q23", "supersede", f"Did User ever work at {emp[1]}?", ["yes"])

    # Sequential ordering (after X)
    if len(boss) >= 4:
        # Pick boss[3]; the answer should be boss[4]
        if len(boss) >= 5:
            add(
                "Q24",
                "history",
                f"Who was User's manager right after {boss[3]}?",
                [boss[4]],
            )
    if len(boss) >= 2:
        # First boss question - "User's first manager"
        add("Q25", "history", "Who was User's very first manager?", [boss[0]])

    # Marcus-specific (he had multiple roles)
    marcus = gt.chain_values(("@Marcus", "role"))
    if len(marcus) >= 2:
        add("Q26", "current", "What is Marcus's current role?", [marcus[-1]])

    # Hobby cycling
    if len(hobby) >= 3:
        add(
            "Q27",
            "history",
            f"Has User ever done {hobby[0]} as a hobby?",
            ["yes"],
        )

    # Counts on Jamie
    if len(jamie_job) >= 3:
        add(
            "Q28",
            "history",
            "How many different jobs has Jamie had during the conversation?",
            [str(len(jamie_job))],
        )

    # Car
    if len(car) >= 2:
        add("Q29", "current", "What car does User drive now?", [car[-1]])

    # Gym
    if len(gym) >= 2:
        add("Q30", "history", f"Did User ever go to {gym[0]}?", ["yes"])

    # Cat
    if "Luna" in gt.intros:
        add(
            "Q31",
            "entity",
            "Does User have a cat? If yes, what's the cat's name?",
            ["Luna"],
        )

    # Long-history chronological dump
    if len(boss) >= 6:
        add("Q32", "history", "List User's managers in chronological order.", boss)

    return qs


if __name__ == "__main__":
    turns = generate()
    gt = ground_truth(turns)
    qs = build_questions(gt)

    kinds = {}
    for t in turns:
        kinds[t.kind] = kinds.get(t.kind, 0) + 1

    n_trans_total = sum(len(v) for v in gt.chains.values())
    n_trans_non_first = sum(max(0, len(v) - 1) for v in gt.chains.values())

    print(f"turns: {len(turns)}")
    print(f"kind distribution: {kinds}")
    print(f"total chain values (incl. first): {n_trans_total}")
    print(f"non-first transitions:            {n_trans_non_first}")
    for k, vs in gt.chains.items():
        print(f"  {k[0]}.{k[1]}: {len(vs)} -> {[v for _, v in vs]}")
    print(f"# questions: {len(qs)}")
