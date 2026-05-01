"""hypothetical: ~30 turns. User explores hypothetical scenarios — "what if I
moved to Paris" / "if I were a vegan" / "imagine Jamie and I broke up".

Real-world chains should be untouched: User actually lives in Brooklyn, eats
omnivore, is together with Jamie.
"""

from __future__ import annotations

from ._types import FactCheck, QACheck, Scenario, Turn


def generate() -> Scenario:
    raw: list[tuple[str, str]] = [
        # Real baseline
        ("real", "I live in Brooklyn with my partner Jamie."),
        ("real", "I work as an engineer at Stripe."),
        ("real", "Jamie and I have been together for five years."),
        ("real", "I eat pretty much everything — big steak person."),
        ("real", "Marcus is my manager at Stripe."),
        # Hypothetical: Paris move
        ("hypothetical", "What if I moved to Paris though?"),
        ("hypothetical", "Imagine I lived in the 11th arrondissement."),
        ("hypothetical", "If I lived in Paris I'd commute by metro every day."),
        ("hypothetical", "Hypothetically, I'd become fluent in French in two years."),
        (
            "hypothetical",
            "If I were in Paris I'd work for a tech startup near Sentier.",
        ),
        # Back to real
        ("real", "Anyway, in real life I'm still in Brooklyn and Stripe."),
        ("real", "Jamie made dinner — coq au vin from the new cookbook."),
        # Hypothetical: vegan
        ("hypothetical", "What if I went fully vegan?"),
        ("hypothetical", "Imagine never eating cheese again — I couldn't do it."),
        (
            "hypothetical",
            "If I were vegan, breakfast would be tofu scramble every day.",
        ),
        # Real
        ("real", "Had a steak burrito for lunch — definitely not vegan."),
        ("real", "Marcus pulled me into a planning meeting this afternoon."),
        # Hypothetical: relationship breakup
        (
            "hypothetical",
            "Imagine if Jamie and I broke up — I'd probably move to Chicago.",
        ),
        ("hypothetical", "Hypothetically, if I were single I'd adopt three more dogs."),
        ("hypothetical", "If we broke up I'd take Luna with me obviously."),
        # Real
        ("real", "Jamie and I are doing great honestly. Anniversary next month."),
        ("real", "Luna is sound asleep on the rug."),
        # Hypothetical career
        ("hypothetical", "What if I quit Stripe and became a barista?"),
        ("hypothetical", "If I were a barista I'd open a tiny shop in Fort Greene."),
        ("hypothetical", "Hypothetically, my latte art would be unbeatable."),
        # Real
        ("real", "OK, in reality I just got promoted to staff engineer at Stripe."),
        ("real", "Jamie is proud — celebrated with sushi."),
        ("real", "Marcus said the staff promo was overdue."),
        ("real", "I'm sticking with Stripe and Brooklyn for the foreseeable future."),
        ("real", "Luna chewed up Jamie's slipper. Real life problems."),
    ]

    turns = [Turn(idx=i, text=t, expected_world=w) for i, (w, t) in enumerate(raw)]

    fact_checks = [
        FactCheck(
            "User actually lives in Brooklyn", "real", ["Brooklyn"], ["hypothetical"]
        ),
        FactCheck(
            "User actually works at Stripe", "real", ["Stripe"], ["hypothetical"]
        ),
        FactCheck("User and Jamie are together (real)", "real", ["Jamie"], []),
        FactCheck(
            "User is omnivore (real)",
            "real",
            ["steak", "burrito", "everything"],
            ["hypothetical"],
        ),
        FactCheck(
            "Marcus is real manager", "real", ["Marcus", "manager"], ["hypothetical"]
        ),
        FactCheck(
            "Hypothetical Paris move",
            "hypothetical",
            ["Paris", "11th", "metro"],
            ["real"],
        ),
        FactCheck(
            "Hypothetical vegan diet", "hypothetical", ["vegan", "tofu"], ["real"]
        ),
        FactCheck(
            "Hypothetical breakup",
            "hypothetical",
            ["broke up", "single", "Chicago"],
            ["real"],
        ),
    ]

    qa_checks = [
        QACheck(
            "actual_location",
            "Where does User actually live in real life?",
            "real",
            ["Brooklyn"],
            ["Paris"],
        ),
        QACheck(
            "actual_employer",
            "Where does User actually work?",
            "real",
            ["Stripe"],
            ["barista", "Sentier"],
        ),
        QACheck("actual_diet", "Is User a vegan?", "real", ["No"], []),
        QACheck(
            "actual_relationship",
            "Are Jamie and User actually together?",
            "real",
            ["Yes", "Jamie"],
            ["broke up"],
        ),
        QACheck(
            "hypothetical_paris",
            "If User hypothetically moved to Paris, where would they live?",
            "hypothetical",
            ["11th", "Paris"],
            [],
        ),
    ]

    return Scenario(
        name="hypothetical",
        turns=turns,
        fact_checks=fact_checks,
        qa_checks=qa_checks,
    )
