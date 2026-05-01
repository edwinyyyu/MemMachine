"""mixed: ~50 turns combining all three world types within one conversation,
testing whether the world-classifier handles transitions cleanly. Includes
sarcasm/jokes, a brief D&D session, a hypothetical, and real life.
"""

from __future__ import annotations

from ._types import FactCheck, QACheck, Scenario, Turn


def generate() -> Scenario:
    raw: list[tuple[str, str]] = [
        ("real", "Morning, working on the migration with Marcus today."),
        ("real", "I'm at home in Brooklyn, Jamie is making coffee."),
        ("real", "Luna is barking at squirrels again."),
        ("real", "Stand-up at 10 with the platform team."),
        (
            "joke",
            "I'm definitely going to quit and become a goat farmer in Vermont, mark my words.",
        ),
        ("real", "OK seriously, big migration push this sprint."),
        ("real", "Marcus said the platform team is moving to a new office."),
        # D&D session begins
        ("game:dnd", "Tonight's D&D session — I'm playing Thalia, a halfling rogue."),
        ("game:dnd", "Thalia has +6 stealth and a magical dagger called Whisperedge."),
        ("game:dnd", "The party is in the Underdark hunting a beholder."),
        ("game:dnd", "Our DM is Priya. She's brutal."),
        ("game:dnd", "Thalia stole a healing potion from the cleric mid-combat."),
        ("game:dnd", "We met an NPC: a drow merchant named Xeraz."),
        ("game:dnd", "Thalia invested 200 gold in a magical lockpick set."),
        ("game:dnd", "Our cleric is named Brom, played by my friend Eli."),
        ("game:dnd", "Thalia successfully backstabbed the beholder for 24 damage."),
        # Back to real
        ("real", "OK in real life, just had a meeting with Marcus about Q2 OKRs."),
        ("real", "Jamie and I are getting groceries tonight."),
        ("real", "Luna's vet appointment is next Tuesday."),
        # Hypothetical
        ("hypothetical", "What if Jamie and I moved to Vermont for real?"),
        (
            "hypothetical",
            "If we moved to Vermont I'd work fully remote and Jamie would freelance.",
        ),
        ("hypothetical", "Hypothetically, Luna would love the snow."),
        # Real
        ("real", "Jamie said no to Vermont. Brooklyn it is."),
        ("real", "Marcus approved my migration plan today."),
        # Sarcasm / joke
        ("joke", "Yeah I'm a billionaire and Marcus is my butler."),
        ("real", "Anyway, Marcus is just my regular manager."),
        # Novel / fiction (different world from D&D)
        (
            "fiction:novel",
            "Quick note for my novel: the protagonist Linnea finds a hidden cave.",
        ),
        ("fiction:novel", "Linnea's mentor in the novel is named Old Tobias."),
        ("fiction:novel", "Their village is called Dunmere."),
        # Real
        ("real", "Jamie made dinner — green curry, fantastic as usual."),
        ("real", "Marcus circulated the Q2 OKR doc."),
        # D&D again — should reuse game:dnd
        ("game:dnd", "Back to D&D — Thalia is sneaking past the beholder's guards."),
        ("game:dnd", "Brom casts bless on the party. Eli always picks bless."),
        (
            "game:dnd",
            "Xeraz the drow merchant offers Thalia a magical bag of holding for 800 gold.",
        ),
        # Real
        ("real", "Walked Luna in Prospect Park, beautiful afternoon."),
        ("real", "Marcus mentioned a possible re-org but it's just rumors."),
        # Joke
        ("joke", "I'm the best engineer who ever lived, no contest."),
        # Real
        ("real", "Lol obviously I'm not, but Stripe gives me good leveling feedback."),
        ("real", "Working from home tomorrow, Jamie is too."),
        # Hypothetical
        ("hypothetical", "Imagine if I had gone to law school instead of CS."),
        ("hypothetical", "If I were a lawyer I'd be miserable but well-paid."),
        # Real
        (
            "real",
            "OK, sticking with engineering. Marcus said the staff promo is final.",
        ),
        # Novel
        (
            "fiction:novel",
            "In the novel, Linnea returns to Dunmere with a stolen relic.",
        ),
        ("fiction:novel", "Old Tobias warns Linnea that the relic is cursed."),
        # Real
        ("real", "Jamie and I are watching a movie tonight."),
        ("real", "Luna is sound asleep, peaceful."),
        # Joke
        (
            "joke",
            "I will single-handedly write the Linux kernel from scratch tomorrow.",
        ),
        ("real", "OK signing off, big real-world day tomorrow."),
        ("real", "Marcus, Jamie, Luna, Brooklyn — that's my life."),
    ]

    turns = [Turn(idx=i, text=t, expected_world=w) for i, (w, t) in enumerate(raw)]

    fact_checks = [
        FactCheck(
            "Real boss Marcus",
            "real",
            ["Marcus", "manager"],
            ["game:dnd", "fiction:novel"],
        ),
        FactCheck("Real partner Jamie", "real", ["Jamie"], []),
        FactCheck("Real dog Luna", "real", ["Luna"], ["game:dnd", "fiction:novel"]),
        FactCheck("Real location Brooklyn", "real", ["Brooklyn"], ["fiction:novel"]),
        FactCheck(
            "D&D character Thalia",
            "game:dnd",
            ["Thalia", "halfling", "rogue"],
            ["real"],
        ),
        FactCheck("D&D NPC Xeraz", "game:dnd", ["Xeraz"], ["real"]),
        FactCheck(
            "Novel protagonist Linnea",
            "fiction:novel",
            ["Linnea"],
            ["real", "game:dnd"],
        ),
        FactCheck(
            "Novel mentor Tobias", "fiction:novel", ["Tobias"], ["real", "game:dnd"]
        ),
    ]

    qa_checks = [
        QACheck(
            "real_location",
            "Where does User actually live?",
            "real",
            ["Brooklyn"],
            ["Vermont", "Underdark", "Dunmere"],
        ),
        QACheck(
            "real_boss",
            "Who is User's actual boss?",
            "real",
            ["Marcus"],
            ["Priya", "DM", "Old Tobias"],
        ),
        QACheck(
            "real_pet",
            "What pet does User have in real life?",
            "real",
            ["Luna"],
            ["Whisperedge", "Thalia"],
        ),
        QACheck(
            "dnd_character",
            "Who does User play in the D&D game?",
            "game:dnd",
            ["Thalia", "rogue"],
            ["Marcus", "Stripe"],
        ),
        QACheck(
            "novel_characters",
            "Who is the protagonist in User's novel?",
            "fiction:novel",
            ["Linnea"],
            ["Marcus", "Thalia"],
        ),
    ]

    return Scenario(
        name="mixed",
        turns=turns,
        fact_checks=fact_checks,
        qa_checks=qa_checks,
    )
