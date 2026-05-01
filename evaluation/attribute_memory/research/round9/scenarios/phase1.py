"""Phase-1 synthetic scenario: ~100 turns with known state-tracking ground truth.

Entities:
- User (the speaker)
- Jamie (partner)
- Marcus (first boss)
- Alice (second boss, replaces Marcus)
- Priya (coworker/friend)
- Diego (gym trainer, then friend)
- Luna (cat)

Major state transitions the turns cover:
- User job: software engineer at Stripe -> (invalidate) never was at Stripe, at Anthropic
- User boss: Marcus -> Alice (supersede; Alice is new boss after Marcus leaves)
- User/Jamie relationship: dating -> engaged -> married
- Priya workplace: Google -> Anthropic (joins User's team)
- Diego role: trainer -> (refine) then became friend after the gym closed
- Luna: new entity, has distinctive traits
- User's workplace: Stripe (correction: was always Anthropic) -> later moves to NY

The scenario includes a mix of:
- State-changing turns (role, employer, relationship, location)
- Clarifying turns (adds detail)
- Invalidation (correction) turns
- Multi-entity turns (two entities state-changing together)
- Entity-history questions
- Null/filler turns (chitchat to not write)

Each turn in `TURNS` is a tuple (turn_idx, text, annotation).
`annotation` is a dict with:
- kind: one of 'state', 'clarify', 'supersede', 'invalidate', 'multi', 'filler'
- entities: list of primary entities mentioned (the @mentions)
- note: human-readable expected effect on state
This lets us deterministically grade retrieval / reasoning at the end.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Turn:
    idx: int
    text: str
    kind: str
    entities: list[str] = field(default_factory=list)
    note: str = ""


# ---------------------------------------------------------------------------
# The conversation: ~100 turns. Ground-truth is embedded in annotations.
# ---------------------------------------------------------------------------

TURNS: list[Turn] = [
    # --- Arc 1: setup (turns 1-15) ---
    Turn(
        1,
        "Hey, just got back from my shift at the hospital",
        "state",
        ["User"],
        note="User is a nurse",
    ),
    Turn(2, "It's been a long day, three back-to-back patients", "filler", []),
    Turn(
        3,
        "My partner Jamie is making dinner tonight",
        "state",
        ["User", "Jamie"],
        note="Jamie = User's partner",
    ),
    Turn(
        4,
        "Jamie works as a graphic designer — pretty much freelance now",
        "state",
        ["Jamie"],
        note="Jamie job = freelance graphic designer",
    ),
    Turn(
        5,
        "We've been together five years, feels like forever",
        "clarify",
        ["User", "Jamie"],
        note="Jamie/User 5 years",
    ),
    Turn(6, "Haha, the weather here is crazy lately", "filler", []),
    Turn(
        7,
        "My boss Marcus wants me to lead the Q3 initiative",
        "state",
        ["User", "Marcus"],
        note="boss role: @Marcus, Marcus is User's boss",
    ),
    Turn(
        8,
        "Marcus is intense but fair, he's been running the unit for years",
        "clarify",
        ["Marcus"],
        note="Marcus personality detail",
    ),
    Turn(
        9,
        "Actually wait — I realize I keep saying nurse but I'm technically a nurse practitioner",
        "clarify",
        ["User"],
        note="User role refined: nurse -> nurse practitioner",
    ),
    Turn(
        10,
        "We live in Seattle for now, though Jamie hates the rain",
        "state",
        ["User", "Jamie"],
        note="Location: Seattle",
    ),
    Turn(
        11,
        "Priya, my old college friend, just messaged me out of the blue",
        "state",
        ["User", "Priya"],
        note="Priya = User's college friend",
    ),
    Turn(
        12,
        "She's been working at Google for like seven years",
        "state",
        ["Priya"],
        note="Priya works at Google",
    ),
    Turn(13, "We used to study together in the chem lab", "clarify", ["User", "Priya"]),
    Turn(
        14,
        "Jamie is also thinking about moving to contract work full-time",
        "clarify",
        ["Jamie"],
    ),
    Turn(15, "Long day. Going to bed early.", "filler", []),
    # --- Arc 2: development & first transitions (turns 16-35) ---
    Turn(16, "Started my morning with coffee at the usual spot", "filler", []),
    Turn(
        17,
        "Marcus said he's pushing me to present at the conference next month",
        "clarify",
        ["Marcus"],
        note="Marcus still boss",
    ),
    Turn(
        18,
        "Priya's considering switching jobs — she's interviewing at Anthropic",
        "state",
        ["Priya"],
        note="Priya interviewing at Anthropic",
    ),
    Turn(
        19,
        "Jamie and I had a great brunch, talked about the future",
        "filler",
        ["User", "Jamie"],
    ),
    Turn(
        20,
        "Actually Jamie is now officially my fiancé — we got engaged!",
        "supersede",
        ["User", "Jamie"],
        note="Jamie/User relationship: partner -> engaged. Supersede the earlier 'partner' claim.",
    ),
    Turn(
        21,
        "The engagement was on the beach, very classic",
        "clarify",
        ["User", "Jamie"],
    ),
    Turn(
        22,
        "Priya accepted the offer at Anthropic!",
        "supersede",
        ["Priya"],
        note="Priya workplace: Google -> Anthropic",
    ),
    Turn(
        23,
        "We adopted a cat named Luna this afternoon",
        "state",
        ["Luna", "User"],
        note="Luna = User/Jamie's cat; new entity",
    ),
    Turn(
        24,
        "Luna is a tortoiseshell, about 2 years old, very chatty",
        "clarify",
        ["Luna"],
    ),
    Turn(
        25,
        "Marcus left the company today — I'm getting a new boss soon",
        "state",
        ["Marcus", "User"],
        note="Marcus no longer User's boss",
    ),
    Turn(
        26,
        "They brought in Alice from the outside team to replace Marcus",
        "supersede",
        ["Alice", "Marcus", "User"],
        note="boss role: @Marcus -> @Alice",
    ),
    Turn(27, "Alice has a different style — more hands-off", "clarify", ["Alice"]),
    Turn(
        28,
        "Jamie's graphic-design contract got renewed for another year",
        "clarify",
        ["Jamie"],
    ),
    Turn(
        29,
        "Priya started at Anthropic last week, she's on the safety team",
        "clarify",
        ["Priya"],
        note="Priya at Anthropic, safety team",
    ),
    Turn(30, "Luna knocked over the water bowl again", "filler", ["Luna"]),
    Turn(
        31,
        "Wait — actually scratch all that earlier stuff. I'm NOT a nurse practitioner. I'm a registered nurse.",
        "invalidate",
        ["User"],
        note="INVALIDATE User role: reverts 'nurse practitioner'. Current role: registered nurse",
    ),
    Turn(
        32,
        "I keep saying 'registered nurse' because that's what's on the license.",
        "clarify",
        ["User"],
    ),
    Turn(
        33,
        "Alice is also a Scorpio — randomly came up",
        "clarify",
        ["Alice"],
        note="Alice Scorpio",
    ),
    Turn(
        34,
        "Diego, my gym trainer, is pushing me to add cardio",
        "state",
        ["Diego", "User"],
        note="Diego = User's gym trainer",
    ),
    Turn(35, "He trains three days a week — very strict", "clarify", ["Diego"]),
    # --- Arc 3: multi-entity + history (turns 36-60) ---
    Turn(
        36,
        "Priya and I actually met at a conference in Portland, not college",
        "invalidate",
        ["Priya", "User"],
        note="INVALIDATE earlier 'college friend' claim. Replaces with: met at Portland conference",
    ),
    Turn(37, "The kitchen is a mess, Luna climbed the fridge", "filler", ["Luna"]),
    Turn(
        38,
        "Jamie mentioned they're flying to Paris for a client in June",
        "state",
        ["Jamie"],
    ),
    Turn(
        39,
        "Alice handed me a new project — redesign of the intake system",
        "clarify",
        ["Alice", "User"],
    ),
    Turn(
        40,
        "Marcus messaged me, wished me luck. He moved to a startup.",
        "clarify",
        ["Marcus"],
        note="Marcus at startup now",
    ),
    Turn(41, "Pretty slow day at work", "filler", []),
    Turn(
        42,
        "Diego said he's closing the gym — his last client next month",
        "state",
        ["Diego"],
        note="Diego gym closing",
    ),
    Turn(43, "Luna vet appointment went well, she's healthy", "filler", ["Luna"]),
    Turn(
        44, "Priya is loving Anthropic — says the team is great", "clarify", ["Priya"]
    ),
    Turn(
        45,
        "Jamie and I set the wedding date — next September",
        "clarify",
        ["User", "Jamie"],
        note="wedding date = next September",
    ),
    Turn(46, "Marcus texted — he wants to catch up next week", "filler", ["Marcus"]),
    Turn(
        47,
        "Diego and I have been hanging out more since the gym closed",
        "refine",
        ["Diego", "User"],
        note="Diego role refined: trainer -> friend",
    ),
    Turn(
        48,
        "Alice scheduled a 1:1 with me about my Q4 goals",
        "clarify",
        ["Alice", "User"],
    ),
    Turn(
        49,
        "Priya joined my team at Anthropic! She transferred from safety.",
        "state",
        ["Priya"],
        note="Priya's team changed: safety team -> User's team",
    ),
    Turn(
        50,
        "Wait. I've been saying I work at the hospital but that's in my past. I'm actually a software engineer now, at Anthropic.",
        "invalidate",
        ["User"],
        note="MAJOR INVALIDATE: User is NOT a nurse. User is software engineer at Anthropic. Everything before about nursing is wrong.",
    ),
    Turn(
        51,
        "Yeah I just realized I mixed things up in my head. Nursing was years ago.",
        "clarify",
        ["User"],
    ),
    Turn(
        52,
        "I'm on the same team as Priya now, actually — she was my recruiter.",
        "clarify",
        ["Priya", "User"],
        note="Priya is User's teammate, was recruiter",
    ),
    Turn(
        53,
        "Alice is not my nurse manager obviously — Alice is my engineering manager.",
        "clarify",
        ["Alice", "User"],
        note="Alice role refined: engineering manager",
    ),
    Turn(
        54,
        "Jamie still freelances graphic design, that part was right",
        "clarify",
        ["Jamie"],
    ),
    Turn(
        55,
        "Marcus was my manager at Anthropic before Alice — same role",
        "clarify",
        ["Marcus", "User"],
        note="Marcus also an engineering manager at Anthropic",
    ),
    Turn(
        56, "Luna is only interested in one thing: the tuna treats", "filler", ["Luna"]
    ),
    Turn(
        57,
        "Diego is teaching programming now actually, switched careers",
        "clarify",
        ["Diego"],
    ),
    Turn(
        58,
        "Priya remembers when I interviewed, was terrified",
        "filler",
        ["Priya", "User"],
    ),
    Turn(
        59,
        "Jamie is joining me in NY for a trip next week",
        "filler",
        ["Jamie", "User"],
    ),
    Turn(60, "Alice said I should apply for a promotion", "clarify", ["Alice", "User"]),
    # --- Arc 4: more state changes + potential traps (61-85) ---
    Turn(
        61,
        "We moved! Just relocated to New York — Jamie got a big client there",
        "supersede",
        ["User", "Jamie"],
        note="Location supersede: Seattle -> New York",
    ),
    Turn(62, "Luna does not like the car ride", "filler", ["Luna"]),
    Turn(
        63,
        "Priya is still in Seattle at Anthropic, will be remote for me",
        "clarify",
        ["Priya"],
    ),
    Turn(
        64,
        "Diego visited New York last week, we caught up",
        "clarify",
        ["Diego", "User"],
    ),
    Turn(
        65,
        "Alice announced she's leaving Anthropic to start her own company",
        "state",
        ["Alice"],
        note="Alice leaves Anthropic",
    ),
    Turn(
        66,
        "So now I'll have a new manager again — a guy named Theo",
        "supersede",
        ["Alice", "Theo", "User"],
        note="boss role: @Alice -> @Theo",
    ),
    Turn(
        67,
        "Theo was previously on the infra team, moved into management",
        "clarify",
        ["Theo"],
    ),
    Turn(
        68,
        "Jamie is freelancing still but thinking about going back in-house",
        "clarify",
        ["Jamie"],
    ),
    Turn(69, "Luna has claimed the new apartment's sunny window", "filler", ["Luna"]),
    Turn(
        70,
        "Priya got promoted to senior engineer at Anthropic!",
        "clarify",
        ["Priya"],
        note="Priya senior engineer",
    ),
    Turn(
        71,
        "Marcus is doing well at his startup — raised a seed round",
        "clarify",
        ["Marcus"],
    ),
    Turn(72, "Diego sent me a book on Python metaprogramming", "filler", ["Diego"]),
    Turn(
        73,
        "Theo's first 1:1 went well, he's direct but reasonable",
        "clarify",
        ["Theo"],
    ),
    Turn(
        74,
        "Jamie and I hit our 2-year anniversary of being engaged",
        "filler",
        ["Jamie", "User"],
    ),
    Turn(
        75,
        "Wait, that's wrong — we got married three months ago, it's our first wedding anniversary coming up",
        "supersede",
        ["Jamie", "User"],
        note="SUPERSEDE: Jamie/User relationship: engaged -> married",
    ),
    Turn(76, "Priya's team at Anthropic has been growing fast", "filler", ["Priya"]),
    Turn(
        77,
        "Alice's startup has a weird name — 'Paradox Labs'",
        "clarify",
        ["Alice"],
        note="Alice's startup = Paradox Labs",
    ),
    Turn(
        78,
        "Theo is fine but I miss Alice's management style",
        "filler",
        ["Theo", "Alice"],
    ),
    Turn(
        79,
        "Jamie is closing the freelance shop and going full-time at Pentagram",
        "supersede",
        ["Jamie"],
        note="Jamie job: freelance -> full-time at Pentagram",
    ),
    Turn(80, "Luna is now 4 years old, we've had her two years", "clarify", ["Luna"]),
    Turn(
        81,
        "Diego started a coding bootcamp, teaching junior devs",
        "clarify",
        ["Diego"],
    ),
    Turn(
        82, "Marcus is hiring engineers — asked if I know anyone", "clarify", ["Marcus"]
    ),
    Turn(
        83, "Priya sent me a puzzle that completely broke my brain", "filler", ["Priya"]
    ),
    Turn(
        84,
        "Theo changed my review cadence — now monthly instead of quarterly",
        "clarify",
        ["Theo", "User"],
    ),
    Turn(
        85,
        "Jamie started at Pentagram this week, loves it so far",
        "clarify",
        ["Jamie"],
    ),
    # --- Arc 5: deep-history questions + supersede-chain stress (86-105) ---
    Turn(
        86,
        "Priya's now technical lead on her team",
        "supersede",
        ["Priya"],
        note="Priya role: senior engineer -> technical lead",
    ),
    Turn(87, "Luna caught a fly, very proud of herself", "filler", ["Luna"]),
    Turn(
        88,
        "Got coffee with Marcus today. Still at his startup.",
        "filler",
        ["Marcus", "User"],
    ),
    Turn(
        89, "Alice's Paradox Labs just hired their first engineer", "clarify", ["Alice"]
    ),
    Turn(90, "Theo mentioned doing 360 feedback this quarter", "filler", ["Theo"]),
    Turn(91, "Diego's bootcamp got funded by Y Combinator", "clarify", ["Diego"]),
    Turn(
        92, "Jamie is loving Pentagram — says the culture is great", "filler", ["Jamie"]
    ),
    Turn(
        93,
        "Priya and I had a long debate about async patterns",
        "filler",
        ["Priya", "User"],
    ),
    Turn(
        94,
        "Marcus and Alice both know each other, they worked together before I joined",
        "state",
        ["Marcus", "Alice"],
        note="Marcus and Alice worked together previously",
    ),
    Turn(
        95,
        "Theo asked about my past managers — said I'd had three: Marcus, Alice, and him",
        "clarify",
        ["Theo", "Marcus", "Alice"],
        note="boss history: Marcus, Alice, Theo (current)",
    ),
    Turn(
        96,
        "Scratch that — there was actually a very short-lived interim manager named Ben between Marcus and Alice",
        "invalidate",
        ["Theo", "Ben", "Marcus", "Alice"],
        note="INVALIDATE the 'three managers' claim. Adds Ben as interim between Marcus and Alice.",
    ),
    Turn(
        97, "Ben was only there for like three weeks, barely counts", "clarify", ["Ben"]
    ),
    Turn(
        98,
        "So the correct sequence is: Marcus -> Ben -> Alice -> Theo",
        "clarify",
        ["Marcus", "Ben", "Alice", "Theo"],
        note="CANONICAL boss history: Marcus, Ben, Alice, Theo",
    ),
    Turn(99, "Priya just got her green card approved", "filler", ["Priya"]),
    Turn(
        100,
        "Jamie and I are planning a vacation to Italy next summer",
        "filler",
        ["Jamie", "User"],
    ),
    Turn(101, "Alice's Paradox Labs raised a Series A", "clarify", ["Alice"]),
    Turn(
        102,
        "Luna and the new cat, Miso, get along surprisingly well",
        "state",
        ["Luna", "Miso"],
        note="Miso = second cat, new entity",
    ),
    Turn(103, "Miso is a black Maine coon, big personality", "clarify", ["Miso"]),
    Turn(104, "Diego's bootcamp hit 100 graduates this month", "clarify", ["Diego"]),
    Turn(
        105,
        "Theo is transitioning to staff engineer — stepping down from management",
        "state",
        ["Theo"],
        note="Theo no longer a manager (leaving management)",
    ),
    # (A future turn might replace Theo, but we leave the boss slot at Theo-leaving-management
    #  for the test: 'who is User's current boss?' -> Theo (still, until replaced).
    # But the scenario will introduce a replacement:
    Turn(
        106,
        "So they promoted someone internal — Nadia is my new manager",
        "supersede",
        ["Theo", "Nadia", "User"],
        note="boss role: @Theo -> @Nadia",
    ),
    Turn(
        107,
        "Nadia was previously a senior PM, not an engineer, first time managing eng",
        "clarify",
        ["Nadia"],
    ),
    Turn(108, "Had a great 1:1 with Nadia", "filler", ["Nadia", "User"]),
    Turn(109, "Priya mentioned she's giving a talk at NeurIPS", "clarify", ["Priya"]),
    Turn(110, "Luna celebrated her gotcha-day with tuna", "filler", ["Luna"]),
]

# Canonical ground-truth snapshot AT END OF CONVERSATION
# This is what the grader uses for current-state questions.
GROUND_TRUTH_CURRENT = {
    "User.role": "software engineer at Anthropic",
    "User.location": "New York",
    "User.boss": "Nadia",
    "User.partner": "Jamie (married)",
    "Jamie.job": "full-time at Pentagram",
    "Jamie.relationship_to_user": "married (spouse)",
    "Marcus.current_job": "at a startup (left Anthropic)",
    "Alice.current_job": "founder/CEO at Paradox Labs",
    "Priya.employer": "Anthropic",
    "Priya.role": "technical lead",
    "Priya.team": "User's team (transferred from safety)",
    "Priya.friend_source": "met at conference in Portland",
    "Diego.current_job": "running a coding bootcamp",
    "Diego.role_to_user": "friend (was gym trainer)",
    "Luna.species": "cat (tortoiseshell)",
    "Theo.previous_role": "User's manager (before Nadia); now staff engineer",
    "Ben.role": "interim boss briefly between Marcus and Alice",
    "Nadia.background": "senior PM, first-time eng manager",
    "Miso.species": "cat (black Maine coon)",
}

# Boss history for User (chronological)
BOSS_HISTORY = ["Marcus", "Ben", "Alice", "Theo", "Nadia"]

# Supersede/invalidate relationships the architectures must track
KEY_CHAINS = {
    "user_role": {
        "chain": [
            "nurse",
            "nurse practitioner",
            "registered nurse (invalidated)",
            "software engineer at Anthropic",
        ],
        "current": "software engineer at Anthropic",
        "note": "Turn 50 invalidates ALL nursing claims.",
    },
    "user_boss": {
        "chain": BOSS_HISTORY,
        "current": "Nadia",
    },
    "user_partner": {
        "chain": ["partner", "engaged", "married"],
        "current": "married to Jamie",
    },
    "user_location": {
        "chain": ["Seattle", "New York"],
        "current": "New York",
    },
    "jamie_job": {
        "chain": ["freelance graphic designer", "full-time at Pentagram"],
        "current": "full-time at Pentagram",
    },
    "priya_employer": {
        "chain": ["Google", "Anthropic"],
        "current": "Anthropic",
    },
    "priya_role": {
        "chain": ["senior engineer", "technical lead"],
        "current": "technical lead",
    },
    "diego_role_to_user": {
        "chain": ["gym trainer", "friend"],
        "current": "friend",
    },
}


# ---------------------------------------------------------------------------
# Test questions
# ---------------------------------------------------------------------------


@dataclass
class Question:
    qid: str
    kind: str  # "current", "history", "supersede", "multi", "entity"
    question: str
    # Canonical answer; grader checks for substring matches + absence of wrong answer
    expected_contains: list[
        str
    ]  # required phrases/names (OR within a sub-list? use AND)
    expected_absent: list[str] = field(default_factory=list)


QUESTIONS: list[Question] = [
    # -- Current-state (8) --
    Question(
        "Q01",
        "current",
        "What is User's current job?",
        ["software engineer", "Anthropic"],
        ["nurse", "hospital"],
    ),
    Question(
        "Q02", "current", "Where does User currently live?", ["New York"], ["Seattle"]
    ),
    Question(
        "Q03",
        "current",
        "Who is User's current manager/boss?",
        ["Nadia"],
        ["Marcus", "Alice", "Theo", "Ben"],
    ),
    Question(
        "Q04",
        "current",
        "What is User's relationship status with Jamie?",
        ["married"],
        ["engaged only", "just dating"],
    ),
    Question(
        "Q05", "current", "Where does Jamie work?", ["Pentagram"], ["freelance only"]
    ),
    Question("Q06", "current", "Where does Priya work now?", ["Anthropic"], ["Google"]),
    Question("Q07", "current", "What is Priya's current role?", ["technical lead"], []),
    Question(
        "Q08",
        "current",
        "What does Diego do now?",
        ["bootcamp"],
        ["gym trainer (currently)"],
    ),
    # -- Supersede-chain history (5) --
    Question(
        "Q09",
        "history",
        "What is User's full boss history, in order?",
        ["Marcus", "Ben", "Alice", "Theo", "Nadia"],
        [],
    ),
    Question(
        "Q10", "history", "Before Alice, who were User's bosses?", ["Marcus", "Ben"], []
    ),
    Question(
        "Q11",
        "history",
        "What jobs has Priya had during the conversation?",
        ["Google", "Anthropic"],
        [],
    ),
    Question(
        "Q12",
        "history",
        "What's the history of User and Jamie's relationship?",
        ["partner", "engaged", "married"],
        [],
    ),
    Question(
        "Q13",
        "supersede",
        "Was User ever a nurse practitioner?",
        ["no", "invalidated"],
        ["yes, User is a nurse practitioner"],
    ),
    # -- Multi-entity (4) --
    Question(
        "Q14",
        "multi",
        "Which of User's friends or coworkers also work at Anthropic?",
        ["Priya"],
        [],
    ),
    Question("Q15", "multi", "Has Marcus ever worked with Alice?", ["yes"], ["no"]),
    Question("Q16", "multi", "Who lives with User?", ["Jamie", "Luna"], []),
    Question(
        "Q17",
        "multi",
        "What entities are in User's household?",
        ["Jamie", "Luna", "Miso"],
        [],
    ),
    # -- Entity-centric / summary (3) --
    Question(
        "Q18",
        "entity",
        "Tell me everything I know about Alice.",
        ["Paradox Labs", "Scorpio"],
        [],
    ),
    Question(
        "Q19",
        "entity",
        "Tell me about Diego.",
        ["gym trainer", "bootcamp", "friend"],
        [],
    ),
    Question("Q20", "entity", "Who is Ben?", ["interim", "manager", "briefly"], []),
]
