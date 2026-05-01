"""Deterministic scenario generators for entity-coref stress tests.

Each scenario produces a list of Turn objects with ground-truth annotations
indicating which entity-id (semantic, scenario-internal) each named/descriptor
mention should resolve to.

Ground-truth `entity_id` is a stable scenario-side label like "BOSS_ALICE",
"HS_ALICE", "MARCUS" — distinct semantic identities. The architecture under
test gets to assign its own internal IDs (e.g. "ent_00007"); the grader maps
the architecture's IDs onto the GT IDs by majority surface-form alignment
across all mentions in the scenario.

Mentions that should NOT resolve to any tracked entity (e.g. distractor names
in filler) carry entity_id=None.

Five scenarios:
  S1 same-name different-entity (boss-Alice vs HS-Alice)
  S2 different-name same-entity (Marcus -> Mark)
  S3 LRU stress (25+ entities, then descriptor-only mentions to early ones)
  S4 pronoun chains (he/she/they refer to recently-introduced entities)
  S5 silent-context default (multiple "Alice" with no clarifying context)

All deterministic. No LLM cost.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

TurnKind = Literal["intro", "state", "filler", "alias", "ambiguous", "descriptor"]


@dataclass
class Mention:
    """A single name/descriptor occurrence inside a turn."""

    surface: str  # the literal text used in the turn ("Alice", "she", "the manager")
    entity_id: str | None  # ground-truth entity id (scenario-internal label) or None
    kind: str = "named"  # "named" | "pronoun" | "descriptor"


@dataclass
class Turn:
    idx: int
    text: str
    kind: TurnKind
    # Ordered list of mentions to evaluate in this turn
    mentions: list[Mention] = field(default_factory=list)
    # Entity introductions / alias-add events
    intro_entity_id: str | None = None
    alias_event: tuple[str, str] | None = None  # (entity_id, new_alias_added)
    notes: str = ""


@dataclass
class Scenario:
    name: str
    description: str
    turns: list[Turn]
    # ground-truth: entity_id -> list of all aliases that should map to it
    aliases: dict[str, set[str]] = field(default_factory=dict)


# ----------------------------------------------------------------------
# S1 — same-name different-entity (Alice-Alice)
# ----------------------------------------------------------------------


def scenario_s1(seed: int = 1) -> Scenario:
    rng = random.Random(seed)
    turns: list[Turn] = []

    def add(
        text: str,
        kind: TurnKind,
        mentions: list[Mention] | None = None,
        intro: str | None = None,
        alias_event: tuple[str, str] | None = None,
        notes: str = "",
    ) -> None:
        turns.append(
            Turn(
                idx=len(turns) + 1,
                text=text,
                kind=kind,
                mentions=mentions or [],
                intro_entity_id=intro,
                alias_event=alias_event,
                notes=notes,
            )
        )

    fillers = [
        "Coffee was good this morning.",
        "Long day. Tired.",
        "Watching a movie tonight.",
        "Traffic was awful.",
        "Pretty quiet day.",
        "Nice weather for a walk.",
        "Just finished a run.",
    ]

    # User identity (always present)
    USER = "USER"

    # --- Pre-Alice setup: introduce User and a colleague Bob (distractor) ---
    add(
        "Hey, I just started a new job at a fintech startup as a backend engineer.",
        "intro",
        mentions=[Mention("I", USER, "pronoun")],
        intro=USER,
        notes="User self-intro",
    )
    for _ in range(3):
        add(rng.choice(fillers), "filler")

    # --- Turn 5: Alice introduced as boss with employer/role detail ---
    add(
        "My new manager Alice is fantastic — she leads the platform team and used to work at Stripe.",
        "intro",
        mentions=[
            Mention("My", USER, "pronoun"),
            Mention("Alice", "BOSS_ALICE", "named"),
            Mention("she", "BOSS_ALICE", "pronoun"),
        ],
        intro="BOSS_ALICE",
        notes="Alice = boss. Strong context: manager, platform, Stripe.",
    )

    # --- Turns 6-30: various boss-Alice mentions ---
    boss_alice_turns = [
        "Alice scheduled our 1:1 for tomorrow morning.",
        "Alice asked me to lead the migration sprint.",
        "Alice and I had a great discussion about Q3 roadmap.",
        "Alice gave me really useful feedback on my design doc.",
        "I'm presenting to Alice and her director next week.",
        "Alice approved my promotion to senior — finally!",
        "Alice mentioned she's been at the company for 5 years.",
        "Alice is going on vacation next week so I'm covering for her.",
    ]
    for t in boss_alice_turns:
        # all of these are unambiguous boss-Alice
        ms = [Mention("Alice", "BOSS_ALICE", "named")]
        if "she" in t.lower():
            ms.append(Mention("she", "BOSS_ALICE", "pronoun"))
        elif "her" in t.lower():
            ms.append(Mention("her", "BOSS_ALICE", "pronoun"))
        add(t, "state", mentions=ms)
        # interleave a filler
        if rng.random() < 0.5:
            add(rng.choice(fillers), "filler")

    # Ensure we have ~25 turns before the high-school-Alice intro
    while len(turns) < 30:
        add(rng.choice(fillers), "filler")

    # --- Turn 31: high-school Alice introduced ---
    add(
        "So I ran into Alice from high school at the grocery store last night — Alice Foster, haven't seen her in like 10 years.",
        "intro",
        mentions=[
            Mention("Alice", "HS_ALICE", "named"),
            Mention("Alice Foster", "HS_ALICE", "named"),
            Mention("her", "HS_ALICE", "pronoun"),
        ],
        intro="HS_ALICE",
        notes="HS Alice = different entity, intro with 'from high school' marker and last name Foster",
    )

    # --- Turns 32-50: ambiguous Alice mentions disambiguated by context ---
    after_intro = [
        (
            "My boss Alice is doing performance reviews this week.",
            "BOSS_ALICE",
            "boss explicit",
        ),
        (
            "Alice (the high school one) and I are getting coffee Saturday.",
            "HS_ALICE",
            "explicit (high school one)",
        ),
        (
            "Alice from work just gave me really hard feedback.",
            "BOSS_ALICE",
            "from work",
        ),
        (
            "Alice Foster is apparently a vet now, that was a surprise.",
            "HS_ALICE",
            "Alice Foster surface form",
        ),
        (
            "Alice ran the whole sprint review today.",
            "BOSS_ALICE",
            "sprint review = work context",
        ),
        (
            "I'll text Alice — the old friend — about brunch.",
            "HS_ALICE",
            "old friend cue",
        ),
        ("Alice approved the prod deploy at 3pm.", "BOSS_ALICE", "prod deploy = work"),
    ]
    for text, ent, _note in after_intro:
        add(
            text,
            "ambiguous",
            mentions=[Mention("Alice", ent, "named")]
            + (
                [Mention("Alice Foster", ent, "named")]
                if "Alice Foster" in text
                else []
            ),
            notes=_note,
        )
        # filler
        if rng.random() < 0.5:
            add(rng.choice(fillers), "filler")

    return Scenario(
        name="S1_same_name_different_entity",
        description="Two different Alices (boss vs high school friend); disambiguation must be context-driven.",
        turns=turns,
        aliases={
            USER: {"User", "I", "me", "my"},
            "BOSS_ALICE": {"Alice"},
            "HS_ALICE": {"Alice", "Alice Foster"},
        },
    )


# ----------------------------------------------------------------------
# S2 — different-name same-entity (Marcus -> Mark)
# ----------------------------------------------------------------------


def scenario_s2(seed: int = 2) -> Scenario:
    rng = random.Random(seed)
    turns: list[Turn] = []

    def add(
        text: str,
        kind: TurnKind,
        mentions: list[Mention] | None = None,
        intro: str | None = None,
        alias_event: tuple[str, str] | None = None,
        notes: str = "",
    ) -> None:
        turns.append(
            Turn(
                idx=len(turns) + 1,
                text=text,
                kind=kind,
                mentions=mentions or [],
                intro_entity_id=intro,
                alias_event=alias_event,
                notes=notes,
            )
        )

    fillers = [
        "Coffee was good this morning.",
        "Long day. Tired.",
        "Watching a movie tonight.",
        "Traffic was awful.",
        "Just finished a run.",
        "Nice weather today.",
    ]

    USER = "USER"

    add(
        "I started a new role as a staff engineer at the fintech startup.",
        "intro",
        mentions=[Mention("I", USER, "pronoun")],
        intro=USER,
    )
    for _ in range(2):
        add(rng.choice(fillers), "filler")

    # --- Turn 4: Marcus introduced ---
    add(
        "My manager is Marcus — he's been at the company about 3 years and runs the platform org.",
        "intro",
        mentions=[
            Mention("Marcus", "MARCUS", "named"),
            Mention("he", "MARCUS", "pronoun"),
        ],
        intro="MARCUS",
        notes="Marcus = manager, platform org",
    )

    marcus_turns = [
        "Marcus scheduled our weekly 1:1.",
        "Marcus has been a great mentor so far.",
        "Marcus suggested I take the lead on the migration project.",
        "Marcus is going to a conference next week.",
        "Marcus gave me really useful design-doc feedback.",
        "Marcus and I aligned on Q3 priorities today.",
        "Marcus said he's pleased with my ramp-up.",
        "Marcus asked me to mentor the new hire.",
        "Marcus approved my promotion to senior staff.",
        "Marcus has the calmest meeting style I've ever seen.",
    ]
    for t in marcus_turns:
        ms = [Mention("Marcus", "MARCUS", "named")]
        if " he " in t or " he's" in t:
            ms.append(Mention("he", "MARCUS", "pronoun"))
        add(t, "state", mentions=ms)
        if rng.random() < 0.5:
            add(rng.choice(fillers), "filler")

    # Ensure we have ~25 turns before the alias-add event
    while len(turns) < 30:
        add(rng.choice(fillers), "filler")

    # --- Alias-add event ---
    add(
        "Oh, btw — Marcus mentioned today that he goes by 'Mark' professionally now. He prefers Mark.",
        "alias",
        mentions=[
            Mention("Marcus", "MARCUS", "named"),
            Mention("he", "MARCUS", "pronoun"),
            Mention("Mark", "MARCUS", "named"),
        ],
        alias_event=("MARCUS", "Mark"),
        notes="Alias added: Mark = Marcus",
    )

    # --- Turns 32-50: 'Mark' surface form, must resolve to MARCUS ---
    mark_turns = [
        "Mark and I had a really productive 1:1 today.",
        "Mark suggested I lead the platform redesign.",
        "Mark is on PTO next week so I'm covering escalations.",
        "Mark gave me the green light on the rewrite.",
        "I CC'd Mark on the design doc this morning.",
        "Mark mentioned he wants me to present to the VP.",
        "Mark approved the promo nomination.",
        "Mark is back from his trip — caught up over coffee.",
    ]
    for t in mark_turns:
        ms = [Mention("Mark", "MARCUS", "named")]
        if " he " in t or " he's" in t or " his " in t:
            for token in ["he", "he's", "his"]:
                if f" {token} " in t.lower() or t.lower().endswith(f" {token}"):
                    ms.append(Mention(token, "MARCUS", "pronoun"))
                    break
        add(t, "state", mentions=ms)
        if rng.random() < 0.5:
            add(rng.choice(fillers), "filler")

    return Scenario(
        name="S2_different_name_same_entity",
        description="Marcus = Mark (alias added at turn ~30). All Mark mentions should resolve to MARCUS.",
        turns=turns,
        aliases={
            USER: {"User", "I", "me", "my"},
            "MARCUS": {"Marcus", "Mark"},
        },
    )


# ----------------------------------------------------------------------
# S3 — LRU stress: 25+ named entities, then descriptor-only mentions of early ones
# ----------------------------------------------------------------------


def scenario_s3(seed: int = 3) -> Scenario:
    rng = random.Random(seed)
    turns: list[Turn] = []

    def add(
        text: str,
        kind: TurnKind,
        mentions: list[Mention] | None = None,
        intro: str | None = None,
        alias_event: tuple[str, str] | None = None,
        notes: str = "",
    ) -> None:
        turns.append(
            Turn(
                idx=len(turns) + 1,
                text=text,
                kind=kind,
                mentions=mentions or [],
                intro_entity_id=intro,
                alias_event=alias_event,
                notes=notes,
            )
        )

    USER = "USER"
    fillers = [
        "Quick coffee break.",
        "Phone call ran long.",
        "Email triage day.",
        "Heading out for lunch.",
        "Slow afternoon.",
    ]

    add(
        "I've been doing a lot of recruiting lately for the platform team.",
        "intro",
        mentions=[Mention("I", USER, "pronoun")],
        intro=USER,
    )

    # 26 distinct entities, each introduced with a clear identifying descriptor
    # so we can later refer to them by descriptor.
    candidates = [
        (
            "Carla",
            "Q1_RECRUITER",
            "Talked to Carla today — she was the recruiter that placed me back in Q1.",
        ),
        (
            "David",
            "DESIGNER_UMA_TEAM",
            "David is the new product designer on Uma's team.",
        ),
        (
            "Elena",
            "OFFSITE_FACILITATOR",
            "Elena facilitated the offsite last month — really sharp.",
        ),
        (
            "Felix",
            "VP_NEW_HIRE",
            "Felix joined as the new VP of engineering, started this week.",
        ),
        ("Greta", "DOG_WALKER", "Greta is our dog walker, she's been great with Luna."),
        (
            "Hassan",
            "CONFERENCE_SPEAKER",
            "Hassan gave the keynote at the security conference I attended.",
        ),
        (
            "Ines",
            "FORMER_ROOMMATE",
            "Ines is my former roommate from college — visited last weekend.",
        ),
        (
            "Julian",
            "PLATFORM_ARCHITECT",
            "Julian is the principal architect on the platform team.",
        ),
        (
            "Kai",
            "INTERN_FROM_BERKELEY",
            "Kai is the new intern from Berkeley, joined this summer.",
        ),
        (
            "Lara",
            "FAMILY_PRACTICE_DOCTOR",
            "Lara is my new primary-care doctor at the clinic on 4th.",
        ),
        (
            "Miguel",
            "GUITAR_TEACHER",
            "Miguel is my guitar teacher, we have weekly lessons.",
        ),
        (
            "Nora",
            "BOOK_CLUB_HOST",
            "Nora hosts the monthly book club — really thoughtful person.",
        ),
        (
            "Oscar",
            "CLIENT_FROM_CHICAGO",
            "Oscar is the client from Chicago I had a call with last quarter.",
        ),
        ("Petra", "YOGA_INSTRUCTOR", "Petra is the new yoga instructor at the studio."),
        (
            "Quinn",
            "Q3_RECRUITER",
            "Quinn placed two new engineers on our team this quarter.",
        ),
        (
            "Raj",
            "MARKETING_LEAD",
            "Raj leads marketing — we collaborated on the launch.",
        ),
        (
            "Sasha",
            "VENDOR_REP",
            "Sasha is the vendor rep from Datadog who handles our account.",
        ),
        (
            "Tomas",
            "OLD_FRIEND_FROM_LISBON",
            "Tomas is an old friend from when I lived in Lisbon.",
        ),
        (
            "Una",
            "ML_RESEARCHER",
            "Una is the ML researcher I've been collaborating with.",
        ),
        (
            "Vince",
            "NEW_NEIGHBOR",
            "Vince is our new neighbor, just moved in next door.",
        ),
        (
            "Wren",
            "NEW_GRAD_DESIGNER",
            "Wren is a new-grad product designer that joined Greta's team.",
        ),
        (
            "Xavi",
            "BARISTA_AT_LOCAL_CAFE",
            "Xavi is the barista at the cafe by my apartment.",
        ),
        (
            "Yara",
            "DENTIST",
            "Yara is my new dentist, the office is just two blocks away.",
        ),
        ("Zane", "PERSONAL_TRAINER", "Zane is my personal trainer at the gym."),
        ("Bella", "VET", "Bella is the new vet at the clinic where we take Luna."),
        ("Cyrus", "ACCOUNTANT", "Cyrus is my accountant — handles tax season for me."),
    ]
    # introduce all 26
    for name, ent_id, intro_text in candidates:
        add(
            intro_text, "intro", mentions=[Mention(name, ent_id, "named")], intro=ent_id
        )
        if rng.random() < 0.3:
            add(rng.choice(fillers), "filler")

    # Now (post turn ~50), refer to early entities by descriptor only
    # These should resolve to the entities introduced WAY earlier (LRU has evicted them).
    # The architecture should still match the descriptor against the registry.
    descriptor_lines = [
        (
            "The recruiter from Q1 reached out about another role.",
            "Q1_RECRUITER",
            "the recruiter from Q1",
        ),
        (
            "The dog walker said she might be moving — need to find a backup.",
            "DOG_WALKER",
            "the dog walker",
        ),
        (
            "The former roommate from college is in town again next weekend.",
            "FORMER_ROOMMATE",
            "the former roommate from college",
        ),
        (
            "The guitar teacher canceled this week's lesson.",
            "GUITAR_TEACHER",
            "the guitar teacher",
        ),
        (
            "The client from Chicago wants to renew the contract.",
            "CLIENT_FROM_CHICAGO",
            "the client from Chicago",
        ),
        (
            "The barista at the cafe by my apartment remembered my order today.",
            "BARISTA_AT_LOCAL_CAFE",
            "the barista",
        ),
        (
            "The personal trainer is on vacation, so the gym is feeling weird.",
            "PERSONAL_TRAINER",
            "the personal trainer",
        ),
        (
            "The accountant emailed about the Q1 estimated taxes.",
            "ACCOUNTANT",
            "the accountant",
        ),
    ]
    for text, ent_id, desc in descriptor_lines:
        add(
            text,
            "descriptor",
            mentions=[Mention(desc, ent_id, "descriptor")],
            notes=f"Should resolve to {ent_id} via descriptor-match against registry (LRU likely evicted)",
        )
        if rng.random() < 0.5:
            add(rng.choice(fillers), "filler")

    return Scenario(
        name="S3_lru_stress",
        description="26 named entities introduced rapidly; later descriptor-only mentions test descriptor-match against the long-tail of the registry (out of LRU).",
        turns=turns,
        aliases={USER: {"User", "I", "me", "my"}, **{e[1]: {e[0]} for e in candidates}},
    )


# ----------------------------------------------------------------------
# S4 — pronoun chains
# ----------------------------------------------------------------------


def scenario_s4(seed: int = 4) -> Scenario:
    rng = random.Random(seed)
    turns: list[Turn] = []

    def add(
        text: str,
        kind: TurnKind,
        mentions: list[Mention] | None = None,
        intro: str | None = None,
        notes: str = "",
    ) -> None:
        turns.append(
            Turn(
                idx=len(turns) + 1,
                text=text,
                kind=kind,
                mentions=mentions or [],
                intro_entity_id=intro,
                notes=notes,
            )
        )

    USER = "USER"
    fillers = ["Quick break.", "Coffee run.", "Phone rang.", "Lunch."]

    add(
        "Setting up — busy week ahead.",
        "intro",
        mentions=[Mention("I", USER, "pronoun")],
        intro=USER,
    )

    # Introduce Sara (she)
    add(
        "My friend Sara just moved to Berlin for grad school.",
        "intro",
        mentions=[Mention("Sara", "SARA", "named")],
        intro="SARA",
    )
    add(
        "She's studying ML at TU Berlin.",
        "state",
        mentions=[Mention("She", "SARA", "pronoun")],
    )
    add(
        "She told me the program is intense but she's loving it.",
        "state",
        mentions=[
            Mention("She", "SARA", "pronoun"),
            Mention("she's", "SARA", "pronoun"),
        ],
    )

    add(rng.choice(fillers), "filler")

    # Introduce Tom (he)
    add(
        "My brother Tom is starting a brewery with two friends.",
        "intro",
        mentions=[Mention("Tom", "TOM", "named")],
        intro="TOM",
    )
    add(
        "He's been working on the business plan for months.",
        "state",
        mentions=[Mention("He", "TOM", "pronoun")],
    )
    add(
        "His wife is supportive but worried about the finances.",
        "state",
        mentions=[Mention("His", "TOM", "pronoun")],
    )

    add(rng.choice(fillers), "filler")

    # Introduce Pat (they/them)
    add(
        "My coworker Pat got promoted to senior staff this week.",
        "intro",
        mentions=[Mention("Pat", "PAT", "named")],
        intro="PAT",
    )
    add(
        "They've been pushing for it for two years.",
        "state",
        mentions=[Mention("They", "PAT", "pronoun")],
    )
    add(
        "Their manager finally signed off.",
        "state",
        mentions=[Mention("Their", "PAT", "pronoun")],
    )

    add(rng.choice(fillers), "filler")

    # Mixed pronouns over a few turns
    add(
        "Sara called from Berlin yesterday.",
        "state",
        mentions=[Mention("Sara", "SARA", "named")],
    )
    add(
        "She's dealing with a brutal thesis advisor.",
        "state",
        mentions=[Mention("She", "SARA", "pronoun")],
    )

    add(
        "Tom dropped by Saturday with samples of his stout.",
        "state",
        mentions=[Mention("Tom", "TOM", "named"), Mention("his", "TOM", "pronoun")],
    )
    add(
        "He's getting really good at the brewing process.",
        "state",
        mentions=[Mention("He", "TOM", "pronoun")],
    )

    add(
        "Pat hosted dinner last night.",
        "state",
        mentions=[Mention("Pat", "PAT", "named")],
    )
    add(
        "They made an excellent pasta from scratch.",
        "state",
        mentions=[Mention("They", "PAT", "pronoun")],
    )

    # Pronoun chains where the antecedent is the most-recent named entity
    # but a different earlier entity is also in the cache
    add(
        "Pat and Sara are actually old college friends.",
        "state",
        mentions=[Mention("Pat", "PAT", "named"), Mention("Sara", "SARA", "named")],
    )
    add(
        "They've known each other since freshman year.",
        "state",
        mentions=[Mention("They", "PAT_AND_SARA_GROUP", "pronoun")],
        notes="'They' here is plural, refers to both; for grading we accept either resolution to PAT or SARA or to a group.",
    )

    return Scenario(
        name="S4_pronoun_chains",
        description="3 entities (Sara, Tom, Pat) introduced and referenced via mixed pronouns within a small window.",
        turns=turns,
        aliases={
            USER: {"User", "I", "me", "my"},
            "SARA": {"Sara"},
            "TOM": {"Tom"},
            "PAT": {"Pat"},
        },
    )


# ----------------------------------------------------------------------
# S5 — silent context: default to LRU-most-recent
# ----------------------------------------------------------------------


def scenario_s5(seed: int = 5) -> Scenario:
    rng = random.Random(seed)
    turns: list[Turn] = []

    def add(
        text: str,
        kind: TurnKind,
        mentions: list[Mention] | None = None,
        intro: str | None = None,
        notes: str = "",
    ) -> None:
        turns.append(
            Turn(
                idx=len(turns) + 1,
                text=text,
                kind=kind,
                mentions=mentions or [],
                intro_entity_id=intro,
                notes=notes,
            )
        )

    USER = "USER"

    add(
        "Catching up — long week.",
        "intro",
        mentions=[Mention("I", USER, "pronoun")],
        intro=USER,
    )

    # ONE Alice in this scenario; multiple silent-context mentions should all
    # default to her. (No second Alice intro.)
    add(
        "My new manager Alice is incredibly hands-off, which I love.",
        "intro",
        mentions=[Mention("Alice", "ONLY_ALICE", "named")],
        intro="ONLY_ALICE",
    )

    # Ten subsequent neutral mentions of "Alice" with no disambiguating context.
    bland = [
        "Alice texted me earlier.",
        "Alice was on the call this morning.",
        "Just heard back from Alice.",
        "Alice and I caught up briefly.",
        "Alice approved the budget request.",
        "Met Alice for coffee.",
        "Alice sent over the doc.",
        "Alice had a question about timelines.",
        "Alice is on PTO Thursday.",
        "Alice mentioned the all-hands is moving.",
    ]
    for t in bland:
        # interleave fillers
        add(
            t,
            "ambiguous",
            mentions=[Mention("Alice", "ONLY_ALICE", "named")],
            notes="Default behavior: most-recent Alice in cache, NOT a new entity",
        )
        if rng.random() < 0.4:
            add("Coffee break.", "filler")

    return Scenario(
        name="S5_silent_context",
        description="Many bland 'Alice' mentions; system should keep them all on a single entity (default to LRU-most-recent), NOT split.",
        turns=turns,
        aliases={USER: {"User", "I", "me", "my"}, "ONLY_ALICE": {"Alice"}},
    )


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


def all_scenarios() -> list[Scenario]:
    return [
        scenario_s1(),
        scenario_s2(),
        scenario_s3(),
        scenario_s4(),
        scenario_s5(),
    ]


if __name__ == "__main__":
    for s in all_scenarios():
        print(f"=== {s.name} ({len(s.turns)} turns) ===")
        print(f"  {s.description}")
        n_mentions = sum(len(t.mentions) for t in s.turns)
        print(f"  total mentions w/ ground truth: {n_mentions}")
        ent_counts: dict[str | None, int] = {}
        for t in s.turns:
            for m in t.mentions:
                ent_counts[m.entity_id] = ent_counts.get(m.entity_id, 0) + 1
        print(f"  per-entity mention counts: {ent_counts}")
