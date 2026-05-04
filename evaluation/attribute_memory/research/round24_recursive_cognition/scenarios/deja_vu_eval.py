"""deja_vu_eval — synthetic memories that test cross-domain structural
pattern recognition. Memories are loaded directly as facts (bypassing
the writer) so we can isolate the deja-vu retrieval mechanism.

Each query has a list of EXPECTED memory ids — the structural targets that
should surface as DEEP. False positives that are GENUINELY structural
(per the LLM judge) aren't penalized.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Memory:
    mid: str  # short label, used for grading
    text: str


# Memory corpus: target memories carrying recognizable structural patterns +
# distractors that share surface words but not structural form.
MEMORIES: list[Memory] = [
    # ENERGY-STORAGE FORM: 1/2 × constant × state²
    Memory(
        "e_spring",
        "When you compress a spring by distance x, it stores potential energy U = (1/2) k x², where k is the spring's stiffness.",
    ),
    Memory(
        "e_capacitor",
        "A capacitor stores electrical energy when charged. The energy is U = (1/2) C V² — half the capacitance times the voltage squared.",
    ),
    Memory(
        "e_rubberband",
        "Stretching a rubber band stores elastic energy that scales with the square of how far you stretched it, scaled by the band's elasticity coefficient.",
    ),
    Memory(
        "e_flywheel",
        "A spinning flywheel stores rotational kinetic energy as (1/2) I ω² — half its moment of inertia times angular velocity squared.",
    ),
    # DIMENSION/COUNTING ARGUMENTS — rank-nullity, fiber dimensions, pigeonhole
    Memory(
        "d_ranknul",
        "The rank-nullity theorem states that for a linear map T from a finite-dimensional vector space V to W, dim(ker T) + dim(im T) = dim V. The dimension of the domain splits into what gets crushed and what survives.",
    ),
    Memory(
        "d_pigeonhole",
        "The pigeonhole principle: if n items go into k bins where n > k, at least one bin must contain two or more items. A counting argument that forces collision when you've outrun your container.",
    ),
    Memory(
        "d_pca",
        "Principal Component Analysis projects high-dimensional data onto a lower-dimensional subspace by finding the directions of maximum variance — a dimension reduction guided by what's discarded vs preserved.",
    ),
    # INVERSE / IMPLICIT FUNCTION THEOREMS — local linearization arguments
    Memory(
        "d_ift",
        "The inverse function theorem says that if a smooth map's derivative is invertible at a point, the map itself is locally invertible near that point — a local diffeomorphism.",
    ),
    Memory(
        "d_implicit",
        "The implicit function theorem says: if F(x, y) = 0 and the Jacobian of F with respect to y is invertible at a solution, then locally you can solve for y as a smooth function of x.",
    ),
    Memory(
        "d_constrank",
        "The constant-rank theorem says that a smooth map of constant rank r between manifolds looks locally like the projection (x_1,...,x_n) ↦ (x_1,...,x_r,0,...,0) in suitable coordinates.",
    ),
    Memory(
        "d_submersion",
        "A smooth submersion (surjective derivative everywhere) has fibers that are smooth submanifolds of dimension dim(domain) − dim(target) — fiber dimension follows from the rank of the derivative.",
    ),
    # UNIVERSAL PROPERTIES / CATEGORICAL CONSTRUCTIONS
    Memory(
        "u_setprod",
        "In set theory, the Cartesian product A × B is characterized by a universal property: for any set X with maps to A and to B, there's a unique map X → A × B making the obvious diagram commute.",
    ),
    Memory(
        "u_freegroup",
        "A free group on a set S is characterized by a universal property: any function from S into a group G extends uniquely to a group homomorphism from the free group to G.",
    ),
    # OPTIMIZATION / PATH-OF-LEAST-RESISTANCE
    Memory(
        "p_river",
        "Dad says rivers find the path of least resistance — they always choose to go around obstacles rather than through them, even if the long way is much longer.",
    ),
    Memory(
        "p_lightning",
        "Lightning takes the path of lowest impedance to ground — it branches and probes before completing the circuit, often appearing to choose the most jagged route.",
    ),
    Memory(
        "p_water",
        "When you tilt a tray with marbles, the marbles flow downward but they don't all take the same route — they fan out around obstacles, each finding a local low-resistance path.",
    ),
    # NARRATIVE TROPE: mentor removed at protagonist transition
    Memory(
        "t_starwars",
        "In Star Wars, Obi-Wan Kenobi trains Luke to use the Force. During the climactic confrontation on the Death Star, Obi-Wan sacrifices himself, allowing Luke to escape and continue alone.",
    ),
    Memory(
        "t_lotr",
        "In Fellowship of the Ring, Gandalf guides the hobbits through dangerous territory. In the Mines of Moria, he falls into the abyss battling the Balrog, and the rest of the Fellowship continues without him.",
    ),
    Memory(
        "t_karatekid",
        "In The Karate Kid, Mr. Miyagi teaches Daniel discipline and skill. By the tournament, Daniel has to compete on his own, applying everything Miyagi taught.",
    ),
    Memory(
        "t_lionking",
        "In The Lion King, Mufasa teaches young Simba about responsibility. After Mufasa is killed in the wildebeest stampede, Simba flees and must return as an adult to reclaim his birthright.",
    ),
    # PERSONAL/CONVERSATIONAL DISTRACTORS (no structural matches)
    Memory("x_pizza", "User had pizza for lunch yesterday and it was greasy."),
    Memory("x_cat", "User's cat Luna knocked over a plant in the morning."),
    Memory("x_lease", "User signed a lease on a new apartment in Park Slope."),
    Memory("x_marcus", "User's manager Marcus moved to a different team last quarter."),
    Memory("x_pottery", "User started taking a pottery class on Sundays."),
    Memory(
        "x_octopus", "User watched a documentary about deep-sea octopuses last weekend."
    ),
    Memory("x_subway", "User's commute today took 45 minutes due to a subway delay."),
    Memory("x_spanish", "User started learning Spanish on Duolingo this month."),
    Memory("x_quentin", "Quentin is User's new mentor at work."),
    Memory("x_promo", "User got their staff promotion approved at Stripe."),
    # MATH/PHYSICS SURFACE DISTRACTORS — same domain as targets but no structural overlap
    Memory(
        "xs_mapreduce",
        "User read about MapReduce — it splits work across nodes, then aggregates results.",
    ),
    Memory(
        "xs_pca_short",
        "User skimmed a paper on neural-network loss landscapes during training.",
    ),
    Memory(
        "xs_banach",
        "Banach fixed-point theorem: a contraction on a complete metric space has a unique fixed point.",
    ),
    Memory(
        "xs_quotient",
        "User reread the chapter on quotient groups — left cosets of a normal subgroup form a group structure.",
    ),
    Memory(
        "xs_continuity",
        "User reread the proof of the intermediate value theorem before class.",
    ),
    Memory(
        "xs_grouptheory", "User has been working through the chapter on Sylow theorems."
    ),
    Memory(
        "xs_typehint",
        "User attended a workshop on Python type hints — talking about generics, variance, duck typing.",
    ),
    Memory(
        "xs_ivt",
        "User wrote a quick proof on a napkin that the rationals are dense in the reals.",
    ),
]

# Extended distractor pool for scale stress testing.
# Diverse surface domains, mostly UNRELATED, some plausibly SURFACE-similar.
SCALE_DISTRACTORS: list[Memory] = [
    Memory(f"sd{i:03d}", text)
    for i, text in enumerate(
        [
            # personal life
            "User went rock climbing for the first time and was surprised at how technical it was.",
            "User's brother sent a meme about software engineers in their natural habitat.",
            "User's recurring sinus infection came back; doctor recommended saline rinse routine.",
            "User noticed their basil plant is getting leggy and needs pinching.",
            "User had a strange dream involving a labyrinth and someone they knew in college.",
            "User's hometown coffee shop is closing after 30 years.",
            "User's friend Maya recommended a podcast about urban planning history.",
            "User finally returned the book they borrowed from their professor in 2019.",
            "User got stuck on a tricky chord change while practicing guitar.",
            "User's sister announced her engagement at Sunday brunch.",
            # math / cs
            "User's algorithms class covered amortized analysis with the dynamic array doubling argument.",
            "User read about the halting problem and the diagonalization proof.",
            "User looked up the proof that there are infinitely many primes — Euclid's classic.",
            "User got confused about the difference between metric and topological completeness.",
            "User's friend was working on a SAT solver project for class.",
            "User read a blog post about the CAP theorem in distributed systems.",
            "User saw a tweet thread about the Y combinator (lambda calculus, not the company).",
            "User's professor mentioned that L1 regularization induces sparsity.",
            "User was puzzling over why the determinant equals the product of eigenvalues.",
            "User read a Wikipedia article about Cantor's diagonal argument.",
            # physics / engineering
            "User watched a YouTube video about feedback control systems and PID tuning.",
            "User read about how phased-array antennas steer beams electronically.",
            "User saw a demo of standing waves on a vibrating string.",
            "User read about how MRI machines use gradient fields to encode position.",
            "User looked up why airfoils generate lift — pressure differential at curved upper surface.",
            "User's friend explained how a rectifier converts AC to DC using diodes.",
            "User read about thermodynamic cycles: Carnot, Otto, Brayton.",
            "User watched a video about the Tacoma Narrows bridge resonance failure.",
            "User read about how diffraction gratings split light into spectra.",
            "User skimmed a paper on superconductor theory and BCS pairing.",
            # biology / medicine
            "User read that bacteria use quorum sensing to coordinate behavior at population thresholds.",
            "User watched a documentary about CRISPR and how Cas9 finds target sequences.",
            "User read a piece on how metabolic pathways form regulated networks.",
            "User read about how the immune system distinguishes self from non-self via thymic selection.",
            "User noticed their resting heart rate has been slowly dropping with consistent training.",
            "User read about how kidney function declines linearly with age.",
            "User read about how forests recover after fires — succession patterns.",
            "User read a piece on how ant colonies allocate workers without central control.",
            "User read about how nerve action potentials propagate via the Hodgkin-Huxley model.",
            "User watched a video on how plants track the sun (heliotropism).",
            # narrative / media
            "User started reading a Murakami novel and noticed his typical recurring motifs.",
            "User watched a Coen Brothers film and was struck by how morally ambiguous everyone was.",
            "User saw a documentary about street photographers and how they wait for the decisive moment.",
            "User read a New Yorker piece about a writer's lifelong obsession with one type of story.",
            "User watched a heist movie where everything was set up in act one.",
            "User listened to a podcast where the host always builds tension before the reveal.",
            "User reread a short story they thought they remembered and noticed new layers.",
            "User watched a foreign film and missed cultural references but enjoyed the rhythm.",
            "User read a comic book where the supervillain reveals their plan in exposition.",
            "User watched a stage play whose climax depended on a single object.",
            # work / professional
            "User had a productive 1:1 with their skip-level manager about career trajectory.",
            "User's annual review included surprising peer feedback about communication style.",
            "User attended a Q3 planning workshop that ran way over time.",
            "User's team did a retrospective and identified the same three issues as last quarter.",
            "User got pulled into an incident response on a holiday weekend.",
            "User's PR was approved with comments about edge cases they hadn't considered.",
            "User volunteered to mentor a new grad joining next month.",
            "User's offsite was held at a vineyard which was nice but distracting.",
            "User received a Slack message that needed a 24h cooldown before responding.",
            "User's colleague is leaving for a competitor — there's a vacuum to fill.",
            # daily life / random
            "User's bus was rerouted due to a marathon they forgot was happening.",
            "User finally got around to ordering a replacement passport before travel.",
            "User signed up for the local farmers market CSA box this season.",
            "User's neighbor brought over surplus tomatoes from their garden.",
            "User's dentist appointment got rescheduled due to a power outage.",
            "User got a haircut and didn't recognize themselves in the mirror at first.",
            "User's gym had its weights rearranged in a way that made workouts inefficient.",
            "User's flight overhead bin was already full when they boarded.",
            "User had a disagreement with their barber about the right length.",
            "User's package was delivered to the wrong unit — neighbor brought it over.",
            # creative / hobby
            "User started a watercolor painting and ruined it by overworking one corner.",
            "User joined a board game night and was crushed at Catan.",
            "User finished a long jigsaw puzzle and felt strangely deflated afterward.",
            "User experimented with sourdough hydration percentages — 80% was too wet.",
            "User wrote a poem and was uncertain whether it was finished.",
            "User went to a pottery studio and made a wonky bowl they kept anyway.",
            "User restarted their Duolingo streak after a long break.",
            "User practiced violin scales for 20 minutes — not glamorous but necessary.",
            "User tried to bake a pie crust and the butter was too warm.",
            "User started a journal again and immediately abandoned it after three entries.",
            # finance / admin
            "User's auto-pay for utilities failed for unclear reasons.",
            "User reviewed last quarter's spending and noticed unused subscriptions.",
            "User's tax refund came back smaller than expected.",
            "User signed up for a HSA contribution adjustment before year-end.",
            "User discussed retirement planning with their financial advisor.",
            "User's credit card had a fraud alert for a $4 charge in another country.",
            "User finally cancelled the gym membership they hadn't used since March.",
            "User got a check from a class action settlement they had forgotten about.",
            "User's homeowner's insurance went up significantly this year.",
            "User considered switching cell carriers but the math didn't quite work out.",
            # food / cooking misc
            "User had okay ramen at a place with great reviews — wondered if expectations mismatched.",
            "User experimented with miso butter on roasted vegetables.",
            "User's cast iron pan finally has a nice seasoning after months of use.",
            "User tried a kimchi stew recipe and it was perfect on a cold day.",
            "User burned the garlic again — needs to add it later in the cooking.",
            "User's bread starter went dormant in the fridge — needed three feedings to revive.",
            "User finally figured out how to make poached eggs reliably.",
            "User's mother visited and reorganized the kitchen, and User can't find anything now.",
            "User's friend brought a dish to the potluck that was suspiciously good.",
            "User decided that they actually do like olives now.",
        ]
    )
]
ALL_MEMORIES = MEMORIES + SCALE_DISTRACTORS


@dataclass
class Question:
    qid: str
    question: str
    expected: list[str]  # mid labels expected to appear in DEEP set


QUESTIONS: list[Question] = [
    Question(
        qid="Q1_inductor",
        question="What's the formula for energy stored in an inductor as a function of current passing through it?",
        expected=["e_spring", "e_capacitor", "e_rubberband", "e_flywheel"],
    ),
    Question(
        qid="Q2_diffgeom",
        question="I'm working with a smooth map between manifolds and trying to count the dimension of a fiber. The map has constant rank but I'm getting confused about how the dimensions add up.",
        # All members of the rank/dimension family are acceptable structural targets:
        expected=["d_ranknul", "d_constrank", "d_submersion", "d_ift", "d_implicit"],
    ),
    Question(
        qid="Q3_topproduct",
        question="I'm working with topological spaces and trying to define what 'the right' product topology should be. What's the abstract property that should determine it?",
        expected=["u_setprod", "u_freegroup"],
    ),
    Question(
        qid="Q4_proposal",
        question="I'm trying to push a controversial proposal through at work. The CTO is blocking it. What's the right way to handle this?",
        expected=["p_river", "p_lightning", "p_water"],
    ),
    Question(
        qid="Q5_mentortrope",
        question="I'm watching a new fantasy series where the protagonist's wise mentor — the only one who really understood the protagonist — was just killed by the villain in episode 6. The protagonist seems devastated. What story arc am I in?",
        expected=["t_starwars", "t_lotr", "t_karatekid", "t_lionking"],
    ),
    Question(
        qid="Q6_borsukulam",
        question="I'm trying to prove that any continuous map from the n-sphere to R^n must send some pair of antipodal points to the same value. What's a good way to think about why this has to happen?",
        expected=["d_pigeonhole"],  # also d_ranknul (dimension obstruction)
    ),
    Question(
        qid="Q7_localsolve",
        question="I have a smooth function from R^n to R^k with k < n, and at one point its Jacobian has full rank k. I want to understand the structure of the level set near that point — which variables can I eliminate, and what shape does the rest take?",
        expected=["d_implicit", "d_ift", "d_constrank", "d_submersion"],
    ),
]


def grade(deep_results: list[tuple[str, str, str]], expected: list[str]) -> dict:
    """deep_results = list of (mid, text, reason). expected = list of mids."""
    deep_mids = {mid for mid, _, _ in deep_results}
    hits = [m for m in expected if m in deep_mids]
    missed = [m for m in expected if m not in deep_mids]
    extras = [mid for mid in deep_mids if mid not in expected]
    return {
        "n_expected": len(expected),
        "n_hit": len(hits),
        "n_missed": len(missed),
        "n_extra": len(extras),
        "hits": hits,
        "missed": missed,
        "extras": extras,
    }
