"""S6 — very-long descriptor delay.

Entity introduced at turn ~10 with a detailed description ("the recruiter
who got me started at Anthropic"). Then 100+ turns of unrelated content
fills up the LRU and evicts it. Then turn ~130: descriptor-only reference.

The aen3_persistent architecture should retrieve the entity via embedding
search over the full registry's description embeddings.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROUND12 = Path(__file__).resolve().parent.parent.parent / "round12_entity_registry"
sys.path.insert(0, str(ROUND12 / "scenarios"))

from coref_stress import Mention, Scenario, Turn  # noqa: E402


def scenario_s6(seed: int = 6) -> Scenario:
    turns: list[Turn] = []

    def add(
        text: str,
        kind: str,
        mentions: list[Mention] | None = None,
        intro: str | None = None,
        notes: str = "",
    ) -> None:
        turns.append(
            Turn(
                idx=len(turns) + 1,
                text=text,
                kind=kind,  # type: ignore[arg-type]
                mentions=mentions or [],
                intro_entity_id=intro,
                notes=notes,
            )
        )

    USER = "USER"

    # turns 1-9: setup
    add(
        "I'm starting a new chapter at work — joined a research lab.",
        "intro",
        mentions=[Mention("I", USER, "pronoun")],
        intro=USER,
    )
    setup_fillers = [
        "Coffee was good this morning.",
        "Long walk after lunch.",
        "Got groceries.",
        "Cleaned out the garage.",
        "Watched a movie tonight.",
        "Slow start to the day.",
        "Ran into an old neighbor.",
        "Read a chapter of the book.",
    ]
    for line in setup_fillers:
        add(line, "filler")

    # turn 10: introduction with detailed description
    add(
        "Sophie is the recruiter who got me started at Anthropic — she "
        "ran the whole loop and convinced me to take the offer two years ago.",
        "intro",
        mentions=[Mention("Sophie", "RECRUITER_SOPHIE", "named")],
        intro="RECRUITER_SOPHIE",
        notes=(
            "Long-delay target. Description: recruiter at Anthropic, "
            "ran the loop two years ago."
        ),
    )

    # turns 11-15: a couple more Sophie mentions to ensure description is
    # rich, then we deliberately go SILENT on her.
    add(
        "Sophie sent me a thank-you note when I accepted.",
        "state",
        mentions=[Mention("Sophie", "RECRUITER_SOPHIE", "named")],
    )
    add(
        "Sophie checked in a month after I started, super thoughtful.",
        "state",
        mentions=[Mention("Sophie", "RECRUITER_SOPHIE", "named")],
    )

    # ~115 turns of unrelated content to flood the LRU. We introduce many
    # OTHER entities so the LRU(20) definitely evicts Sophie.
    flood_entities = [
        ("Alex", "ALEX_DENTIST", "Alex is my new dentist; office is on Main St."),
        ("Bree", "BREE_NEIGHBOR", "Bree is my neighbor in 4B — has a corgi."),
        ("Cleo", "CLEO_BARISTA", "Cleo is the barista at the cafe by my place."),
        ("Dora", "DORA_VET", "Dora is the vet at the local animal hospital."),
        ("Eli", "ELI_PT", "Eli is my physical therapist for the running injury."),
        (
            "Faye",
            "FAYE_LIBRARIAN",
            "Faye is the librarian who runs the local poetry night.",
        ),
        ("Gabe", "GABE_CHEF", "Gabe is the chef at the new bistro on 7th."),
        ("Hana", "HANA_MUSIC_TEACHER", "Hana is my piano teacher, weekly lessons."),
        ("Iris", "IRIS_TRAINER", "Iris is my personal trainer at the gym."),
        ("Jay", "JAY_LANDLORD", "Jay is my landlord — mostly absent, which is good."),
        ("Kit", "KIT_ROOMMATE", "Kit is my new roommate, just moved in."),
        ("Lin", "LIN_ACCOUNTANT", "Lin is my accountant for tax season."),
        ("Mae", "MAE_HAIRDRESSER", "Mae is my hairdresser at the salon downtown."),
        ("Nate", "NATE_MECHANIC", "Nate is the mechanic who fixes my car."),
        ("Oren", "OREN_THERAPIST", "Oren is my therapist, weekly sessions."),
        ("Pia", "PIA_NUTRITIONIST", "Pia is my nutritionist."),
        ("Quinn", "QUINN_PEDIATRICIAN", "Quinn is the pediatrician for my niece."),
        ("Rae", "RAE_PASTOR", "Rae is the pastor at the church I sometimes visit."),
        ("Sol", "SOL_FLORIST", "Sol owns the flower shop I buy from."),
        ("Tate", "TATE_TAXI", "Tate is the cab driver I always seem to get."),
        ("Uma", "UMA_GARDENER", "Uma is my landscaper — does the yard."),
        ("Van", "VAN_PLUMBER", "Van is the plumber who fixed the kitchen sink."),
        ("Wes", "WES_OPTOMETRIST", "Wes is my optometrist, new glasses last month."),
        ("Xan", "XAN_DRY_CLEANER", "Xan owns the dry cleaner on the corner."),
        ("Yuna", "YUNA_PILATES", "Yuna teaches the pilates class on Saturdays."),
        ("Zev", "ZEV_LAWYER", "Zev is my lawyer for the lease review."),
        ("Bea", "BEA_BUTCHER", "Bea is the butcher at the market."),
        ("Cam", "CAM_TRAVEL_AGENT", "Cam is my travel agent — books our trips."),
        ("Dex", "DEX_BIKE_REPAIR", "Dex runs the bike repair shop."),
        ("Em", "EM_TAILOR", "Em is my tailor — alterations done well."),
    ]
    fillers = [
        "Quiet morning.",
        "Weather was nice.",
        "Long phone call.",
        "Slow afternoon.",
        "Made dinner at home.",
        "Watched a podcast.",
        "Went for a run.",
        "Coffee break.",
    ]
    fi = 0
    for name, ent_id, intro in flood_entities:
        add(intro, "intro", mentions=[Mention(name, ent_id, "named")], intro=ent_id)
        # Two filler turns between each flood entity to spread them out
        for _ in range(2):
            add(fillers[fi % len(fillers)], "filler")
            fi += 1
        # Plus one bland mention to keep them in cache too — bland follow-up
        add(
            f"{name} is great, by the way.",
            "state",
            mentions=[Mention(name, ent_id, "named")],
        )

    # By here we have 13 (setup) + 30 entities * (intro + 2 filler + state)
    # = 13 + 120 = ~133 turns. Sophie is definitely out of the 20-slot LRU.

    # The descriptor target turn — turn ~134.
    add(
        "By the way, the recruiter who got me started at Anthropic just "
        "messaged me out of the blue about another candidate.",
        "descriptor",
        mentions=[
            Mention(
                "the recruiter who got me started at Anthropic",
                "RECRUITER_SOPHIE",
                "descriptor",
            )
        ],
        notes="Long-delay descriptor: must be retrieved via embedding search.",
    )
    # A second descriptor-style mention with slightly different wording
    add(
        "Yeah, the person who recruited me into Anthropic two years ago "
        "wants to chat about a referral.",
        "descriptor",
        mentions=[
            Mention(
                "the person who recruited me into Anthropic two years ago",
                "RECRUITER_SOPHIE",
                "descriptor",
            )
        ],
        notes="Second long-delay descriptor — paraphrased.",
    )

    return Scenario(
        name="S6_long_delay",
        description=(
            "Entity introduced at turn 10 with rich description, then ~120 "
            "unrelated turns evict it from LRU, then descriptor-only "
            "references. Tests embedding-search recovery in aen3."
        ),
        turns=turns,
        aliases={
            USER: {"User", "I", "me", "my"},
            "RECRUITER_SOPHIE": {"Sophie"},
            **{e[1]: {e[0]} for e in flood_entities},
        },
    )


if __name__ == "__main__":
    s = scenario_s6()
    print(f"{s.name}: {len(s.turns)} turns")
    print(f"  description: {s.description}")
    target_mentions = [
        (t.idx, m)
        for t in s.turns
        for m in t.mentions
        if m.entity_id == "RECRUITER_SOPHIE"
    ]
    print(f"  RECRUITER_SOPHIE mentions: {target_mentions}")
