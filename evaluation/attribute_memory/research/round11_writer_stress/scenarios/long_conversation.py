"""Deterministic ~1000-turn conversation generator for writer stress.

Produces a list of user-utterance strings (no LLM cost) plus ground-truth
annotations describing which turns should produce memory-worthy entries,
which create supersede chains on which predicates, entity intros, etc.

Mix targets:
  - Identity/intro: ~3%
  - State updates (new value for a predicate): ~13%
  - Casual/phatic/filler (should be noop): ~42%
  - Detail/clarification: ~10%
  - Multi-entity events: ~5%
  - Long supersede-chain segments (role-churn): ~27% (concentrated)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

TurnKind = Literal[
    "intro",
    "state",
    "clarify",
    "multi",
    "filler",
    "invalidate",
]


@dataclass
class Turn:
    idx: int
    text: str
    kind: TurnKind
    # Ground-truth for this turn
    mentions: list[str] = field(default_factory=list)
    # (@Entity, predicate) -> new_value. Indicates what state this turn updates.
    predicate_updates: dict[tuple[str, str], str] = field(default_factory=dict)
    # For invalidate turns: list of (entity, predicate) pairs being retracted
    invalidates: list[tuple[str, str]] = field(default_factory=list)
    # For intro turns: entity name introduced
    intro_entity: str | None = None


# -----------------------------------------------------
# Entity pools
# -----------------------------------------------------
BOSS_NAMES = [
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
]
# We'll pick ~10 bosses in sequence for a deep chain across the convo.

CITY_NAMES = [
    "Seattle",
    "Brooklyn",
    "Austin",
    "Chicago",
    "Denver",
    "Portland",
]

EMPLOYER_NAMES = [
    "Stripe",
    "Anthropic",
    "Google",
    "OpenAI",
    "Notion",
    "Figma",
]

TITLE_NAMES = [
    "junior engineer",
    "engineer",
    "senior engineer",
    "staff engineer",
    "tech lead",
    "engineering manager",
]

PARTNER_STATES = ["partner", "fiance", "spouse"]  # three-step chain

FILLER_TEMPLATES = [
    "Coffee was really good this morning.",
    "The weather is awful today.",
    "Traffic was brutal on the way home.",
    "Pretty slow day today, mostly catching up on emails.",
    "Working from a cafe right now.",
    "Just got back from a run.",
    "Tired. Long day.",
    "Haha that meme is so funny.",
    "Ok, gonna go grab lunch.",
    "Thinking about what to cook tonight.",
    "The cat is being dramatic again.",
    "Been reading a lot lately, feels good.",
    "Ugh, Mondays.",
    "Listening to a new podcast, decent.",
    "Cool — ttyl.",
    "Totally agree with that article you sent.",
    "Nice weather for a walk.",
    "Another rainy day in the neighborhood.",
    "Garbage truck woke me up at 6am.",
    "Watching a movie tonight.",
    "Didn't sleep great.",
    "Feeling pretty good today.",
    "Stomach hurts, probably too much coffee.",
    "Let me know when you're free this week.",
    "The office espresso machine is broken again.",
]

DETAIL_TEMPLATES_BOSS = [
    "My boss {boss} is really into endurance sports.",
    "{boss} always runs 1:1s on Tuesday mornings.",
    "{boss} is a surprisingly good writer, reads widely.",
    "{boss} likes to debrief via Slack, not in-person.",
    "{boss} grew up in the Midwest, moved out west after college.",
]

DETAIL_TEMPLATES_JAMIE = [
    "Jamie and I watched a new series last night.",
    "Jamie finished their pottery piece this weekend.",
    "Jamie made pasta from scratch, very impressive.",
    "Jamie has been collecting vintage posters.",
    "Jamie is dragging me to a yoga class tomorrow.",
]

DETAIL_TEMPLATES_LUNA = [
    "Luna knocked a vase off the counter, again.",
    "Luna has claimed the new rug as her territory.",
    "Luna's been especially chatty this week.",
    "Luna caught a moth and was extremely proud.",
    "Luna and I have a standing 7am breakfast date.",
]

CLARIFY_USER_TEMPLATES = [
    "I should mention — my job technically isn't just coding, there's a lot of design review too.",
    "To be clear I'm on the platform team, not applied.",
    "My role has a lot of mentoring in it these days.",
    "I should add — my work involves a lot of on-call rotations.",
]


def _tag(name: str) -> str:
    return f"@{name}"


def generate(n_target_turns: int = 1000, seed: int = 42) -> list[Turn]:
    rng = random.Random(seed)
    turns: list[Turn] = []
    idx_counter = 0

    def add(
        text: str,
        kind: TurnKind,
        *,
        mentions: list[str] | None = None,
        predicate_updates: dict[tuple[str, str], str] | None = None,
        invalidates: list[tuple[str, str]] | None = None,
        intro_entity: str | None = None,
    ) -> None:
        nonlocal idx_counter
        idx_counter += 1
        turns.append(
            Turn(
                idx=idx_counter,
                text=text,
                kind=kind,
                mentions=mentions or [],
                predicate_updates=predicate_updates or {},
                invalidates=invalidates or [],
                intro_entity=intro_entity,
            )
        )

    # --- State we track as ground truth ---
    # (all for @User unless noted)
    current_boss: str | None = None
    current_city: str | None = None
    current_employer: str | None = None
    current_title: str | None = None
    current_partner_state: str | None = None  # partner / fiance / spouse
    jamie_introduced = False
    luna_introduced = False

    # Plan the chain of boss transitions. We want ~10 transitions across the
    # conversation, but the main bursts happen in role-churn segments.
    planned_bosses = BOSS_NAMES[:10]
    # City chain: 4 relocations
    planned_cities = CITY_NAMES[:4]
    # Employer chain: 3 employers
    planned_employers = EMPLOYER_NAMES[:3]
    # Title chain: 4
    planned_titles = TITLE_NAMES[:4]

    # Progress pointers
    boss_idx = 0
    city_idx = 0
    emp_idx = 0
    title_idx = 0
    partner_idx = 0

    # Main loop — plan by "segments"
    # Each segment is ~50-150 turns with a theme. Total ~= n_target_turns.
    # We schedule transitions across the first ~900 turns and leave ~100 tail.

    # Helper transition generators:
    def boss_transition() -> None:
        nonlocal boss_idx, current_boss
        if boss_idx >= len(planned_bosses):
            return
        new = planned_bosses[boss_idx]
        boss_idx += 1
        old = current_boss
        if old is None:
            text = f"I just started at my job — my manager is {new}."
            mentions = ["User", new]
        else:
            text = f"{new} is my new manager now — {old} moved to a different team."
            mentions = ["User", new, old]
        add(
            text, "state", mentions=mentions, predicate_updates={("@User", "boss"): new}
        )
        current_boss = new

    def city_transition() -> None:
        nonlocal city_idx, current_city
        if city_idx >= len(planned_cities):
            return
        new = planned_cities[city_idx]
        city_idx += 1
        if current_city is None:
            text = f"We've been living in {new} for a while now."
        else:
            text = f"We're moving! Packing up the apartment — relocating to {new}."
        add(
            text,
            "state",
            mentions=["User"],
            predicate_updates={("@User", "location"): new},
        )
        current_city = new

    def employer_transition() -> None:
        nonlocal emp_idx, current_employer
        if emp_idx >= len(planned_employers):
            return
        new = planned_employers[emp_idx]
        emp_idx += 1
        text = f"I'm starting at {new} next week. Exciting change."
        add(
            text,
            "state",
            mentions=["User"],
            predicate_updates={("@User", "employer"): new},
        )
        current_employer = new

    def title_transition() -> None:
        nonlocal title_idx, current_title
        if title_idx >= len(planned_titles):
            return
        new = planned_titles[title_idx]
        title_idx += 1
        text = f"I got promoted — I'm a {new} now."
        add(
            text,
            "state",
            mentions=["User"],
            predicate_updates={("@User", "title"): new},
        )
        current_title = new

    def partner_transition() -> None:
        nonlocal partner_idx, current_partner_state
        if partner_idx >= len(PARTNER_STATES):
            return
        new = PARTNER_STATES[partner_idx]
        partner_idx += 1
        if new == "partner":
            text = "Jamie is my partner — we've been dating about a year."
            mentions = ["User", "Jamie"]
        elif new == "fiance":
            text = "We got engaged last weekend! Jamie said yes."
            mentions = ["User", "Jamie"]
        else:
            text = "Jamie and I got married last month — just a courthouse ceremony."
            mentions = ["User", "Jamie"]
        add(
            text,
            "state",
            mentions=mentions,
            predicate_updates={("@User", "partner_state"): new},
        )
        current_partner_state = new

    def introduce_jamie() -> None:
        nonlocal jamie_introduced
        if jamie_introduced:
            return
        add(
            "Jamie is my partner — we've been together a while.",
            "intro",
            mentions=["User", "Jamie"],
            intro_entity="Jamie",
            predicate_updates={("@User", "partner_state"): "partner"},
        )
        jamie_introduced = True
        # increment partner_idx to 1 since partner is the first state
        # (we want partner → fiance → spouse)

    def introduce_luna() -> None:
        nonlocal luna_introduced
        if luna_introduced:
            return
        add(
            "We adopted a cat today — her name is Luna, a tortoiseshell.",
            "intro",
            mentions=["User", "Luna"],
            intro_entity="Luna",
        )
        luna_introduced = True

    def filler() -> None:
        add(rng.choice(FILLER_TEMPLATES), "filler")

    def boss_detail() -> None:
        if current_boss is None:
            filler()
            return
        t = rng.choice(DETAIL_TEMPLATES_BOSS).format(boss=current_boss)
        add(t, "clarify", mentions=["User", current_boss])

    def jamie_detail() -> None:
        if not jamie_introduced:
            filler()
            return
        t = rng.choice(DETAIL_TEMPLATES_JAMIE)
        add(t, "clarify", mentions=["User", "Jamie"])

    def luna_detail() -> None:
        if not luna_introduced:
            filler()
            return
        t = rng.choice(DETAIL_TEMPLATES_LUNA)
        add(t, "clarify", mentions=["User", "Luna"])

    def user_clarify() -> None:
        t = rng.choice(CLARIFY_USER_TEMPLATES)
        add(t, "clarify", mentions=["User"])

    def multi_event() -> None:
        """Multi-entity event: e.g. Jamie + Luna both feature."""
        if not jamie_introduced:
            filler()
            return
        if luna_introduced:
            options = [
                "Jamie and Luna are having a standoff over the new armchair.",
                "Jamie took Luna to the vet — all clear.",
                "Jamie and I took Luna on a picnic.",
            ]
            mentions = ["User", "Jamie", "Luna"]
        else:
            options = [
                "Jamie and I went hiking over the weekend.",
                "Jamie and I had dinner with friends.",
            ]
            mentions = ["User", "Jamie"]
        add(rng.choice(options), "multi", mentions=mentions)

    # -----------------------------------------------------
    # Scripted segments
    # -----------------------------------------------------
    # Segment 1 (turns 1-60): opening — introduce user job, Jamie, first boss
    add(
        "Hey — just finished another day at work. I'm a software engineer.",
        "state",
        mentions=["User"],
        predicate_updates={("@User", "title"): "engineer"},
    )
    current_title = "engineer"
    title_idx = 2  # skip "junior engineer", start at "engineer"
    introduce_jamie()
    partner_idx = 1  # we've used "partner" state
    employer_transition()  # Stripe
    city_transition()  # Seattle
    boss_transition()  # Marcus
    # Fill to turn 60 with filler/detail
    while len(turns) < 60:
        r = rng.random()
        if r < 0.6:
            filler()
        elif r < 0.8:
            jamie_detail()
        else:
            boss_detail()

    # Segment 2 (turns 61-120): early churn on boss (2 transitions) + cat intro
    while len(turns) < 75:
        filler() if rng.random() < 0.7 else jamie_detail()
    introduce_luna()
    while len(turns) < 90:
        r = rng.random()
        if r < 0.4:
            filler()
        elif r < 0.7:
            luna_detail()
        else:
            jamie_detail()
    boss_transition()  # Alice
    while len(turns) < 110:
        r = rng.random()
        if r < 0.5:
            filler()
        elif r < 0.7:
            boss_detail()
        elif r < 0.85:
            luna_detail()
        else:
            jamie_detail()
    boss_transition()  # Theo
    while len(turns) < 120:
        filler()

    # Segment 3 (turns 121-180): employer + city transition + title promo
    employer_transition()  # Anthropic
    while len(turns) < 140:
        r = rng.random()
        if r < 0.5:
            filler()
        elif r < 0.7:
            boss_detail()
        elif r < 0.85:
            jamie_detail()
        else:
            luna_detail()
    city_transition()  # Brooklyn
    while len(turns) < 155:
        filler() if rng.random() < 0.7 else luna_detail()
    title_transition()  # senior engineer
    while len(turns) < 170:
        r = rng.random()
        if r < 0.6:
            filler()
        elif r < 0.8:
            user_clarify()
        else:
            jamie_detail()
    partner_transition()  # fiance
    while len(turns) < 180:
        filler()

    # Segment 4 (turns 181-260): long rapid boss-churn burst (6 transitions in ~80 turns)
    for i in range(6):
        # Space transitions 8-12 turns apart with filler/details
        target = len(turns) + rng.randint(8, 14)
        while len(turns) < target:
            r = rng.random()
            if r < 0.6:
                filler()
            elif r < 0.8:
                boss_detail()
            else:
                luna_detail()
        boss_transition()
    while len(turns) < 260:
        filler() if rng.random() < 0.6 else jamie_detail()

    # Segment 5 (turns 261-360): employer swap + invalidate/correction stretch
    employer_transition()  # Google
    while len(turns) < 280:
        r = rng.random()
        if r < 0.6:
            filler()
        elif r < 0.8:
            user_clarify()
        else:
            jamie_detail()
    # Explicit invalidate: user "corrects" which city
    # Say they moved Brooklyn -> Austin actually: we supersede, not invalidate.
    # But we also do a genuine invalidate: the previous title they mentioned
    # was wrong.
    add(
        "Actually — scratch what I said about being a staff engineer; I was wrong. I'm still senior engineer.",
        "invalidate",
        mentions=["User"],
        invalidates=[("@User", "title")],
        predicate_updates={("@User", "title"): "senior engineer"},
    )
    # (title was last set via title_transition() to "senior engineer"; this
    # "invalidate" exercises the language but doesn't actually change
    # ground-truth state)
    while len(turns) < 310:
        r = rng.random()
        if r < 0.6:
            filler()
        elif r < 0.8:
            boss_detail()
        else:
            luna_detail()
    title_transition()  # staff engineer
    while len(turns) < 330:
        filler() if rng.random() < 0.7 else jamie_detail()
    city_transition()  # Austin
    while len(turns) < 360:
        r = rng.random()
        if r < 0.6:
            filler()
        elif r < 0.8:
            user_clarify()
        else:
            luna_detail()

    # Segment 6 (turns 361-500): stability + multi-events + detail
    partner_transition()  # spouse
    while len(turns) < 420:
        r = rng.random()
        if r < 0.5:
            filler()
        elif r < 0.65:
            jamie_detail()
        elif r < 0.8:
            luna_detail()
        elif r < 0.9:
            boss_detail()
        else:
            multi_event()
    # Another boss churn burst: 2 more (now ~8 bosses total)
    for _ in range(2):
        target = len(turns) + rng.randint(10, 18)
        while len(turns) < target:
            r = rng.random()
            if r < 0.55:
                filler()
            elif r < 0.75:
                boss_detail()
            else:
                jamie_detail()
        boss_transition()
    while len(turns) < 500:
        filler() if rng.random() < 0.6 else luna_detail()

    # Segment 7 (turns 501-650): title promotion + big employer change + churn
    title_transition()  # tech lead
    while len(turns) < 550:
        r = rng.random()
        if r < 0.5:
            filler()
        elif r < 0.7:
            user_clarify()
        elif r < 0.85:
            boss_detail()
        else:
            multi_event()
    employer_transition()  # OpenAI
    while len(turns) < 600:
        filler() if rng.random() < 0.7 else jamie_detail()
    # Rapid boss churn: 2 more
    for _ in range(2):
        target = len(turns) + rng.randint(12, 18)
        while len(turns) < target:
            r = rng.random()
            if r < 0.6:
                filler()
            elif r < 0.8:
                boss_detail()
            else:
                luna_detail()
        boss_transition()
    while len(turns) < 650:
        filler() if rng.random() < 0.65 else multi_event()

    # Segment 8 (turns 651-800): city change + title + detail-heavy
    city_transition()  # Chicago
    while len(turns) < 700:
        r = rng.random()
        if r < 0.5:
            filler()
        elif r < 0.7:
            jamie_detail()
        elif r < 0.85:
            luna_detail()
        else:
            user_clarify()
    title_transition()  # engineering manager
    while len(turns) < 760:
        r = rng.random()
        if r < 0.55:
            filler()
        elif r < 0.75:
            boss_detail()
        elif r < 0.9:
            user_clarify()
        else:
            multi_event()
    while len(turns) < 800:
        filler() if rng.random() < 0.7 else jamie_detail()

    # Segment 9 (turns 801-900): late-conversation rapid churn stress
    # The point: make sure writer still emits refs at T=800+.
    # We'll do 3 rapid boss transitions (to consume any remaining bosses)
    remaining = max(0, len(planned_bosses) - boss_idx)
    for _ in range(min(3, remaining) + 1):
        target = len(turns) + rng.randint(15, 25)
        while len(turns) < target:
            r = rng.random()
            if r < 0.6:
                filler()
            elif r < 0.8:
                boss_detail()
            else:
                luna_detail()
        if boss_idx < len(planned_bosses):
            boss_transition()
        else:
            break
    # Pad filler
    while len(turns) < 900:
        r = rng.random()
        if r < 0.6:
            filler()
        elif r < 0.8:
            user_clarify()
        else:
            jamie_detail()

    # Segment 10 (turns 901-1000): tail — mostly filler with a final city move
    city_transition()  # Denver (if available)
    while len(turns) < 980:
        r = rng.random()
        if r < 0.7:
            filler()
        elif r < 0.85:
            luna_detail()
        else:
            jamie_detail()
    # Final update: one more boss transition
    if boss_idx < len(planned_bosses):
        boss_transition()
    while len(turns) < n_target_turns:
        filler()

    # Cap at exactly n_target_turns
    turns = turns[:n_target_turns]
    # Renumber indices to guarantee contiguous 1..N
    for i, t in enumerate(turns, start=1):
        t.idx = i
    return turns


# -----------------------------------------------------
# Ground-truth summary
# -----------------------------------------------------


@dataclass
class GroundTruth:
    # (@entity, pred) -> list of (turn_idx, value)
    chains: dict[tuple[str, str], list[tuple[int, str]]] = field(default_factory=dict)
    # intro events
    intros: dict[str, int] = field(default_factory=dict)  # entity -> first turn

    def current_value(self, key: tuple[str, str]) -> str | None:
        chain = self.chains.get(key)
        return chain[-1][1] if chain else None

    def chain_values(self, key: tuple[str, str]) -> list[str]:
        return [v for _, v in self.chains.get(key, [])]


def ground_truth(turns: list[Turn]) -> GroundTruth:
    gt = GroundTruth()
    for t in turns:
        if t.intro_entity and t.intro_entity not in gt.intros:
            gt.intros[t.intro_entity] = t.idx
        for key, value in t.predicate_updates.items():
            gt.chains.setdefault(key, []).append((t.idx, value))
    return gt


# -----------------------------------------------------
# Questions for end-to-end Q/A
# -----------------------------------------------------


@dataclass
class Question:
    qid: str
    kind: str  # "current", "history", "supersede", "entity", "multi"
    question: str
    expected_contains: list[str]
    expected_absent: list[str] = field(default_factory=list)


def build_questions(gt: GroundTruth) -> list[Question]:
    """Build ~30 state-tracking questions from the ground truth."""
    qs: list[Question] = []

    # Current values
    cur_boss = gt.current_value(("@User", "boss"))
    cur_city = gt.current_value(("@User", "location"))
    cur_emp = gt.current_value(("@User", "employer"))
    cur_title = gt.current_value(("@User", "title"))
    cur_partner = gt.current_value(("@User", "partner_state"))

    if cur_boss:
        qs.append(
            Question("Q01", "current", "Who is User's current manager?", [cur_boss])
        )
    if cur_city:
        qs.append(
            Question("Q02", "current", "Where does User currently live?", [cur_city])
        )
    if cur_emp:
        qs.append(Question("Q03", "current", "Where does User work now?", [cur_emp]))
    if cur_title:
        qs.append(
            Question("Q04", "current", "What is User's current job title?", [cur_title])
        )
    if cur_partner:
        # Map internal state to user-facing word
        label = {"partner": "partner", "fiance": "fiance", "spouse": "spouse"}[
            cur_partner
        ]
        qs.append(
            Question(
                "Q05",
                "current",
                "What is User's relationship status with Jamie?",
                [label],
            )
        )

    # Supersede (was X ever true / is X still true)
    boss_chain = gt.chain_values(("@User", "boss"))
    if len(boss_chain) >= 3:
        first = boss_chain[0]
        middle = boss_chain[len(boss_chain) // 2]
        qs.append(
            Question("Q06", "supersede", f"Is {first} still User's manager?", ["no"])
        )
        qs.append(
            Question("Q07", "supersede", f"Was {middle} ever User's manager?", ["yes"])
        )

    city_chain = gt.chain_values(("@User", "location"))
    if city_chain:
        first_city = city_chain[0]
        qs.append(
            Question(
                "Q08", "supersede", f"Does User currently live in {first_city}?", ["no"]
            )
        )

    emp_chain = gt.chain_values(("@User", "employer"))
    if len(emp_chain) >= 2:
        first_emp = emp_chain[0]
        qs.append(
            Question(
                "Q09", "supersede", f"Is User still working at {first_emp}?", ["no"]
            )
        )
        qs.append(
            Question("Q10", "supersede", f"Did User ever work at {first_emp}?", ["yes"])
        )

    # History (list all values in order)
    if len(boss_chain) >= 2:
        qs.append(
            Question(
                "Q11",
                "history",
                "List User's managers in chronological order.",
                boss_chain,
            )
        )
    if len(city_chain) >= 2:
        qs.append(
            Question(
                "Q12",
                "history",
                "List the cities User has lived in, in order.",
                city_chain,
            )
        )
    if len(emp_chain) >= 2:
        qs.append(
            Question("Q13", "history", "List User's employers in order.", emp_chain)
        )

    # Title history
    title_chain = gt.chain_values(("@User", "title"))
    if len(title_chain) >= 2:
        # We skipped "junior engineer" above; chain starts at "engineer".
        # Just check that the current title appears
        qs.append(
            Question(
                "Q14",
                "current",
                "What's User's job title right now?",
                [title_chain[-1]],
            )
        )

    # Middle-state (was X before Y)
    if len(boss_chain) >= 4:
        before = boss_chain[-2]
        qs.append(
            Question(
                "Q15",
                "history",
                "Who was User's manager right before the current one?",
                [before],
            )
        )

    # Entity intros
    if "Luna" in gt.intros:
        qs.append(
            Question(
                "Q16",
                "entity",
                "Does User have a cat? If yes, what's their name?",
                ["Luna"],
            )
        )
    if "Jamie" in gt.intros:
        qs.append(
            Question("Q17", "entity", "Who is Jamie to User?", ["partner"])
        )  # intro says partner — even after spouse, partner is not false

    # Multi-entity
    if "Luna" in gt.intros and "Jamie" in gt.intros:
        qs.append(
            Question("Q18", "multi", "Who or what lives with User?", ["Jamie", "Luna"])
        )

    # Supersede-heavy: ever lived in each city
    if len(city_chain) >= 2:
        mid_city = city_chain[len(city_chain) // 2]
        qs.append(
            Question("Q19", "supersede", f"Did User ever live in {mid_city}?", ["yes"])
        )

    # Current partner name
    qs.append(Question("Q20", "current", "Who is User's partner or spouse?", ["Jamie"]))

    # A few extras / deep chain
    if len(boss_chain) >= 5:
        # "Who was User's second manager?"
        second = boss_chain[1]
        qs.append(
            Question(
                "Q21",
                "history",
                "Who was User's second manager (the one right after the first)?",
                [second],
            )
        )

    if len(boss_chain) >= 6:
        # deep chain stress
        qs.append(
            Question(
                "Q22",
                "history",
                "How many managers has User had over the course of the conversation? List them.",
                boss_chain,
            )
        )

    # invalidate/correction scenario: "Was User ever a staff engineer?"
    if "staff engineer" in title_chain:
        qs.append(
            Question("Q23", "supersede", "Was User ever a staff engineer?", ["yes"])
        )

    # Current snapshot — catch-all
    if cur_emp and cur_title:
        qs.append(
            Question(
                "Q24",
                "current",
                "What's User's current job? Include employer and title.",
                [cur_emp, cur_title],
            )
        )

    # partner chain history
    partner_chain = gt.chain_values(("@User", "partner_state"))
    if len(partner_chain) >= 2:
        qs.append(
            Question(
                "Q25",
                "history",
                "What's the progression of User and Jamie's relationship?",
                ["partner", "spouse"] if "spouse" in partner_chain else partner_chain,
            )
        )

    # employer — specific: "Did User ever work at {emp}?" for the middle one
    if len(emp_chain) >= 3:
        mid_emp = emp_chain[1]
        qs.append(
            Question("Q26", "supersede", f"Did User ever work at {mid_emp}?", ["yes"])
        )

    # Location: ever in each
    if len(city_chain) >= 3:
        qs.append(
            Question(
                "Q27", "supersede", f"Did User ever live in {city_chain[1]}?", ["yes"]
            )
        )

    # First boss
    if boss_chain:
        qs.append(
            Question(
                "Q28", "history", "Who was User's very first manager?", [boss_chain[0]]
            )
        )

    # First employer
    if emp_chain:
        qs.append(
            Question(
                "Q29",
                "history",
                "Where did User work first in this conversation?",
                [emp_chain[0]],
            )
        )

    # First city
    if city_chain:
        qs.append(
            Question(
                "Q30",
                "history",
                "What's the first city User mentioned living in?",
                [city_chain[0]],
            )
        )

    return qs


if __name__ == "__main__":
    # Quick sanity check
    turns = generate(1000)
    gt = ground_truth(turns)
    qs = build_questions(gt)
    kinds = {}
    for t in turns:
        kinds[t.kind] = kinds.get(t.kind, 0) + 1
    print(f"turns: {len(turns)}")
    print(f"kind distribution: {kinds}")
    print(f"boss chain: {gt.chain_values(('@User', 'boss'))}")
    print(f"city chain: {gt.chain_values(('@User', 'location'))}")
    print(f"employer chain: {gt.chain_values(('@User', 'employer'))}")
    print(f"title chain: {gt.chain_values(('@User', 'title'))}")
    print(f"partner chain: {gt.chain_values(('@User', 'partner_state'))}")
    print(f"n questions: {len(qs)}")
