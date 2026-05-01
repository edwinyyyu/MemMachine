"""novel_writing: ~40 turns. User is writing a novel. Talks about characters,
plot, mixed with real-life mentions ("my editor", "my therapist").

Critical disambiguation: in this scenario the protagonist is named Marcus,
which collides with a real-world boss named Marcus from the user's job. The
two should NOT be conflated.
"""

from __future__ import annotations

from ._types import FactCheck, QACheck, Scenario, Turn


def generate() -> Scenario:
    raw: list[tuple[str, str]] = [
        ("real", "I'm finally drafting that novel I've been talking about."),
        ("real", "My editor Diane wants a synopsis by next month."),
        ("real", "My boss Marcus at work approved my reduced hours so I can write."),
        ("real", "Working from a cafe in Brooklyn today."),
        (
            "fiction:novel",
            "In my novel, the protagonist is named Marcus, an exiled cartographer.",
        ),
        (
            "fiction:novel",
            "Marcus the cartographer lives in a fictional city called Vauris.",
        ),
        ("fiction:novel", "The antagonist is a guild magistrate named Halberd."),
        ("fiction:novel", "Marcus has a sister named Vela who is searching for him."),
        ("fiction:novel", "Vauris is famous for its glass towers and tide-clocks."),
        (
            "fiction:novel",
            "Marcus carries a stolen map showing the route to a forbidden archipelago.",
        ),
        ("real", "Diane sent feedback — she loves the protagonist's voice."),
        (
            "real",
            "I told my therapist about the protagonist Marcus and how he resembles my dad.",
        ),
        (
            "fiction:novel",
            "Halberd is hunting Marcus across the inland sea on a steam-galleon.",
        ),
        (
            "fiction:novel",
            "Vela meets a ferryman who claims to have seen Marcus three months ago.",
        ),
        (
            "fiction:novel",
            "Marcus reaches the first island of the archipelago, a place called Coraline.",
        ),
        ("fiction:novel", "Coraline is inhabited by reclusive cartographer-monks."),
        ("real", "Took a break for lunch — Jamie and I split a pizza."),
        ("fiction:novel", "Marcus negotiates passage with the cartographer-monks."),
        ("fiction:novel", "Halberd captures Vela and uses her to bait Marcus."),
        ("fiction:novel", "Marcus surrenders the map to save Vela."),
        ("real", "Diane wants me to expand the cartographer-monk subplot."),
        ("real", "My boss Marcus signed off on me using PTO next Friday."),
        ("real", "Just walked Luna in the park, gorgeous fall day."),
        (
            "fiction:novel",
            "Halberd's steam-galleon catches fire during the betrayal scene.",
        ),
        ("fiction:novel", "Vela escapes during the chaos and steals back the map."),
        ("fiction:novel", "Marcus and Vela flee to a second island called Brassmoor."),
        (
            "fiction:novel",
            "Brassmoor is a clockwork city built into a volcanic crater.",
        ),
        ("real", "Editor Diane sent another set of margin notes — 80 pages worth."),
        ("real", "Therapist suggested I'm projecting my brother into Vela."),
        (
            "fiction:novel",
            "Marcus repairs the stolen map using monk-glass at Brassmoor.",
        ),
        ("fiction:novel", "Halberd, scarred and furious, follows them to Brassmoor."),
        (
            "fiction:novel",
            "Final showdown: Marcus stabs Halberd with a tide-clock fragment.",
        ),
        ("fiction:novel", "Vauris welcomes Marcus and Vela home as heroes."),
        ("real", "Drafted the climax today, sent it to Diane."),
        ("real", "Boss Marcus approved my book-launch sabbatical for next year."),
        ("real", "Jamie made tea while I edited the epilogue."),
        ("real", "Brooklyn rain is finally here."),
        ("real", "Diane called — she wants to pitch the novel to publishers."),
        ("real", "Therapist appointment Tuesday."),
        ("real", "Going to bed, big writing day tomorrow."),
    ]

    turns = [Turn(idx=i, text=t, expected_world=w) for i, (w, t) in enumerate(raw)]

    fact_checks = [
        FactCheck(
            "Real boss Marcus exists at work",
            "real",
            ["Marcus", "boss"],
            ["fiction:novel"],
        ),
        FactCheck(
            "Editor Diane is real", "real", ["Diane", "editor"], ["fiction:novel"]
        ),
        FactCheck("Jamie is real partner", "real", ["Jamie"], []),
        FactCheck(
            "Novel-Marcus is a cartographer",
            "fiction:novel",
            ["Marcus", "cartographer", "exiled"],
            ["real"],
        ),
        FactCheck("Vela is novel-Marcus's sister", "fiction:novel", ["Vela"], ["real"]),
        FactCheck(
            "Halberd is the antagonist",
            "fiction:novel",
            ["Halberd", "magistrate"],
            ["real"],
        ),
        FactCheck("Vauris is a fictional city", "fiction:novel", ["Vauris"], ["real"]),
    ]

    qa_checks = [
        QACheck(
            "real_boss",
            "Who is User's real boss at work?",
            "real",
            ["Marcus"],
            ["cartographer", "Vauris", "Vela", "Halberd"],
        ),
        QACheck(
            "real_editor",
            "Who is User's editor in real life?",
            "real",
            ["Diane"],
            ["Halberd"],
        ),
        QACheck(
            "novel_protagonist",
            "Who is the protagonist in the novel?",
            "fiction:novel",
            ["Marcus", "cartographer"],
            ["boss"],
        ),
        QACheck(
            "novel_antagonist",
            "Who is the antagonist in the novel?",
            "fiction:novel",
            ["Halberd"],
            ["boss"],
        ),
        QACheck(
            "disambiguate_marcus",
            "Are there two different people named Marcus in this conversation? "
            "Distinguish them.",
            "real",
            ["Marcus"],
            [],
        ),
    ]

    return Scenario(
        name="novel_writing",
        turns=turns,
        fact_checks=fact_checks,
        qa_checks=qa_checks,
    )
