"""fantasy_roleplay: ~40 turns. User does a fantasy role-play as a dragon for
~20 turns then transitions to real life for ~20 turns.

Real-life entities (boss Marcus, partner Jamie, dog Luna) should be world="real".
Role-play entities (the user-dragon's hoard, fellow dragons) should be world
"fiction:fantasy_rp".
"""

from __future__ import annotations

from ._types import FactCheck, QACheck, Scenario, Turn


def generate() -> Scenario:
    raw: list[tuple[str, str]] = [
        # ---- entry: real life baseline ----
        ("real", "Hey, work was rough today."),
        ("real", "My boss Marcus assigned me a big migration project."),
        ("real", "Jamie and I are watching that show again tonight."),
        # ---- enter fantasy role-play ----
        (
            "fiction:fantasy_rp",
            "Let's role-play. I'm a red dragon named Pyrrhus living in the Cinderpeak mountains.",
        ),
        (
            "fiction:fantasy_rp",
            "My hoard contains 4000 gold coins and a magical sword called Embershard.",
        ),
        ("fiction:fantasy_rp", "I have a rival dragon, a frost wyrm named Glacius."),
        (
            "fiction:fantasy_rp",
            "Pyrrhus is 800 years old and has scales the color of molten lava.",
        ),
        (
            "fiction:fantasy_rp",
            "I just torched a band of orcs that tried to climb to my lair.",
        ),
        (
            "fiction:fantasy_rp",
            "Glacius sent a messenger raven challenging me to combat at the frozen lake.",
        ),
        ("fiction:fantasy_rp", "I accept the duel. My fire breath against his ice."),
        (
            "fiction:fantasy_rp",
            "I've added a goblet of unicorn horn to my hoard, taken from a fallen knight.",
        ),
        ("fiction:fantasy_rp", "My familiar is a phoenix chick named Cinder."),
        (
            "fiction:fantasy_rp",
            "I fly down to a village to demand tribute. They offer me 12 sheep.",
        ),
        (
            "fiction:fantasy_rp",
            "I make Cinderpeak my permanent lair. The mountain glows red at night.",
        ),
        (
            "fiction:fantasy_rp",
            "Glacius's blizzard nearly froze my left wing in the duel.",
        ),
        ("fiction:fantasy_rp", "I won the duel. Glacius is now my reluctant ally."),
        (
            "fiction:fantasy_rp",
            "Pyrrhus has begun teaching Cinder how to breathe fire.",
        ),
        (
            "fiction:fantasy_rp",
            "A wandering wizard named Erevan visits seeking a riddle game.",
        ),
        (
            "fiction:fantasy_rp",
            "Erevan loses the riddle game and surrenders his enchanted staff.",
        ),
        (
            "fiction:fantasy_rp",
            "My hoard now has the staff, the goblet, Embershard, and 4500 gold.",
        ),
        # ---- transition back to reality ----
        ("real", "OK that was fun, anyway, in real life I should reply to Marcus."),
        ("real", "Marcus wants the migration plan by Friday."),
        (
            "real",
            "Jamie made dinner — pasta with peas. Real life is calmer than dragon life.",
        ),
        ("real", "Luna chewed up another shoe. She's still a puppy though."),
        ("real", "I live in Brooklyn — been here three years now."),
        ("real", "Marcus said he'll loop in Priya from platform team."),
        ("real", "I switched my title to staff engineer last month."),
        ("real", "Jamie and I are looking at apartments closer to Prospect Park."),
        ("real", "Luna's vet appointment is Saturday. She needs shots."),
        ("real", "Coffee's good this morning, Brooklyn fall weather is finally here."),
        ("real", "Quick standup with Marcus and the team — migration moving."),
        ("real", "Honestly, my real life pets are just Luna, my golden retriever."),
        ("real", "Jamie and I had pancakes for brunch."),
        ("real", "Marcus is going on PTO next week so I'm covering the team."),
        ("real", "Picked up new running shoes — going for a run with Jamie tomorrow."),
        ("real", "Luna tracked mud through the apartment again."),
        ("real", "We're thinking about a beach trip this summer."),
        ("real", "I love my actual quiet life in Brooklyn."),
        ("real", "Marcus approved my migration plan."),
        ("real", "Big day done, going to log off."),
    ]

    turns = [Turn(idx=i, text=t, expected_world=w) for i, (w, t) in enumerate(raw)]

    fact_checks = [
        FactCheck(
            "Marcus is User's boss (real)",
            "real",
            ["Marcus", "boss"],
            ["fiction:fantasy_rp"],
        ),
        FactCheck(
            "User lives in Brooklyn (real)",
            "real",
            ["Brooklyn"],
            ["fiction:fantasy_rp"],
        ),
        FactCheck(
            "Luna is User's dog (real)",
            "real",
            ["Luna", "dog", "puppy", "retriever"],
            ["fiction:fantasy_rp"],
        ),
        FactCheck("Jamie is User's partner (real)", "real", ["Jamie"], []),
        FactCheck(
            "Pyrrhus is a red dragon (fiction)",
            "fiction:fantasy_rp",
            ["Pyrrhus", "dragon"],
            ["real"],
        ),
        FactCheck(
            "User's role-play hoard (fiction)",
            "fiction:fantasy_rp",
            ["hoard", "gold", "Embershard", "staff"],
            ["real"],
        ),
        FactCheck(
            "Glacius the rival dragon (fiction)",
            "fiction:fantasy_rp",
            ["Glacius"],
            ["real"],
        ),
        FactCheck(
            "Cinder the phoenix familiar (fiction)",
            "fiction:fantasy_rp",
            ["Cinder", "phoenix"],
            ["real"],
        ),
    ]

    qa_checks = [
        QACheck(
            "rl_pets",
            "Tell me about User's pets in real life.",
            "real",
            ["Luna"],
            ["Pyrrhus", "dragon", "phoenix", "Cinder", "Glacius"],
        ),
        QACheck(
            "rl_location",
            "Where does User actually live?",
            "real",
            ["Brooklyn"],
            ["Cinderpeak", "mountain", "lair"],
        ),
        QACheck(
            "rl_boss",
            "Who is User's real boss?",
            "real",
            ["Marcus"],
            ["Glacius", "dragon"],
        ),
        QACheck(
            "rp_identity",
            "In the role-play, who is User?",
            "fiction:fantasy_rp",
            ["Pyrrhus", "dragon"],
            ["Marcus", "boss", "engineer"],
        ),
        QACheck(
            "rp_hoard",
            "What's in the dragon's hoard in the role-play?",
            "fiction:fantasy_rp",
            ["gold", "Embershard", "staff"],
            ["Brooklyn", "Marcus"],
        ),
    ]

    return Scenario(
        name="fantasy_roleplay",
        turns=turns,
        fact_checks=fact_checks,
        qa_checks=qa_checks,
    )
