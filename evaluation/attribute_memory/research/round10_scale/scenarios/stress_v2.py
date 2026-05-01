"""Adversarial stress scenarios v2.

Adds noise that defeats simple retrieval:
- S-paraphrase: state-change entries are paraphrased many times
- S-semantic-distractors: many semantically-similar non-changing entries
- S-cross-entity-collision: multiple entities with same predicate
- S-name-conflict: two entities sharing a first name (different roles)

Each entry still has a ground-truth predicate when applicable.
"""

from __future__ import annotations

import random

from generators import (
    ENTITY_POOL,
    Entry,
    Question,
    Ref,
    _mentions,
    _uuid,
)

# ---------------------------------------------------------------------------
# S-paraphrase: paraphrase the same job-change in multiple ways scattered
# in the log, plus dense filler. The reader must still surface the LATEST.
# ---------------------------------------------------------------------------


def gen_paraphrase(
    n_entries: int, seed: int = 11
) -> tuple[list[Entry], list[Question]]:
    rng = random.Random(seed)
    entries: list[Entry] = []
    ts = 0

    # Chain: 6 employers, each transition gets one canonical "supersede" entry
    # plus 2-3 paraphrase entries (clarify) sprinkled later.
    employers = ["Stripe", "Anthropic", "Meta", "OpenAI", "Apple", "Cursor"]

    canonical_uuids = []  # the supersede-chain heads per transition

    # Schedule: 6 canonical transitions + (n_entries - 6 - 12) filler + 12 paraphrases
    n_paraphrase = 12
    n_canonical = len(employers)
    n_filler = max(0, n_entries - n_canonical - n_paraphrase)
    schedule = ["C"] * n_canonical + ["P"] * n_paraphrase + ["F"] * n_filler
    rng.shuffle(schedule)

    # We need to fire canonical transitions in order. Paraphrase entries refer
    # to whichever canonical was most-recently-fired.
    canon_cursor = 0
    prev_uuid = None

    paraphrase_templates = [
        "@User mentioned they're enjoying their work at {emp}.",
        "@User said the {emp} office has good coffee.",
        "Apparently @User joined {emp} a while back.",
        "@User's badge for {emp} arrived in the mail.",
        "@User is on a project for {emp} this quarter.",
    ]

    semantic_distractors = [
        "@User is reading an article about Stripe's IPO.",
        "@User dreamed about working at Apple.",
        "@User saw an Anthropic ad on the subway.",
        "@User's friend has a Meta T-shirt.",
        "@User criticized OpenAI's new policy.",
        "@User wondered what working at Cursor is like.",
        "@User read a job listing at Figma.",
        "@User passed the Google office on their walk.",
        "@User's neighbor works at Notion.",
        "@User went to a tech meetup with people from Stripe.",
    ]

    distractor_names = ENTITY_POOL[:30]

    for slot in schedule:
        ts += 1
        if slot == "C" and canon_cursor < n_canonical:
            emp = employers[canon_cursor]
            canon_cursor += 1
            u = _uuid("e_", ts)
            refs = [Ref(prev_uuid, "supersede")] if prev_uuid else []
            text = f"@User started a new job at {emp}."
            e = Entry(
                u,
                ts,
                text,
                _mentions("User"),
                refs,
                predicate="@User.employer",
                value=emp,
                is_current=(canon_cursor == n_canonical),
            )
            entries.append(e)
            canonical_uuids.append(u)
            prev_uuid = u
        elif slot == "P" and canon_cursor > 0:
            emp = employers[canon_cursor - 1]
            tmpl = rng.choice(paraphrase_templates)
            text = tmpl.format(emp=emp)
            u = _uuid("e_", ts)
            # paraphrase entries clarify the canonical
            refs = [Ref(canonical_uuids[canon_cursor - 1], "clarify")]
            e = Entry(u, ts, text, _mentions("User"), refs)
            entries.append(e)
        else:
            # filler — half generic, half semantic distractors
            if rng.random() < 0.5 and semantic_distractors:
                t = rng.choice(semantic_distractors)
                e = Entry(_uuid("e_", ts), ts, t, _mentions("User"), [])
            else:
                name = rng.choice(distractor_names)
                t = f"@User and @{name} chatted about weekend plans."
                e = Entry(_uuid("e_", ts), ts, t, _mentions("User", name), [])
            entries.append(e)
        if len(entries) >= n_entries:
            break

    qs = [
        Question(
            "P_Q01",
            "current",
            "What is User's current employer?",
            expected_contains=[employers[-1]],
            expected_absent=[],
            scenario_tag="paraphrase",
        ),
        Question(
            "P_Q02",
            "supersede",
            f"Is User still working at {employers[0]}?",
            expected_contains=["no"],
            scenario_tag="paraphrase",
        ),
        Question(
            "P_Q03",
            "supersede",
            f"Did User ever work at {employers[2]}?",
            expected_contains=["yes"],
            scenario_tag="paraphrase",
        ),
        Question(
            "P_Q04",
            "history",
            "List User's employers in chronological order, from earliest to most recent.",
            expected_contains=employers,
            scenario_tag="paraphrase",
        ),
    ]
    return entries, qs


# ---------------------------------------------------------------------------
# S-cross-entity-collision: User AND Jamie both have employer chains. Test
# whether retrieval surfaces the right entity's chain.
# ---------------------------------------------------------------------------


def gen_cross_entity(
    n_entries: int, seed: int = 12
) -> tuple[list[Entry], list[Question]]:
    rng = random.Random(seed)
    entries: list[Entry] = []
    ts = 0

    user_chain = ["Stripe", "Anthropic", "Meta", "OpenAI"]
    jamie_chain = ["Adobe", "Figma", "Pentagram"]
    priya_chain = ["Google", "Anthropic", "Cursor"]

    chains = [
        ("User", user_chain),
        ("Jamie", jamie_chain),
        ("Priya", priya_chain),
    ]
    n_trans = sum(len(c) for _, c in chains)

    # schedule
    n_filler = max(0, n_entries - n_trans)
    slots = ["T"] * n_trans + ["F"] * n_filler
    rng.shuffle(slots)

    # ordered transitions per entity
    cursors = {name: 0 for name, _ in chains}
    prev_uuid = {name: None for name, _ in chains}
    distractors = ENTITY_POOL[:30]

    for slot in slots:
        ts += 1
        if slot == "T":
            avail = [n for n, c in chains if cursors[n] < len(c)]
            if not avail:
                slot = "F"
            else:
                name = rng.choice(avail)
                chain = dict(chains)[name]
                emp = chain[cursors[name]]
                cursors[name] += 1
                u = _uuid("e_", ts)
                refs = [Ref(prev_uuid[name], "supersede")] if prev_uuid[name] else []
                text = f"@{name} started a new job at {emp}."
                e = Entry(
                    u,
                    ts,
                    text,
                    _mentions(name),
                    refs,
                    predicate=f"@{name}.employer",
                    value=emp,
                    is_current=(cursors[name] == len(chain)),
                )
                entries.append(e)
                prev_uuid[name] = u
                continue
        # filler
        n2 = rng.choice(distractors)
        text = rng.choice(
            [
                f"@User had lunch with @{n2}.",
                f"@{n2} said hi.",
                f"@User and @{n2} discussed a book.",
            ]
        )
        e = Entry(_uuid("e_", ts), ts, text, _mentions("User", n2), [])
        entries.append(e)
        if len(entries) >= n_entries:
            break

    qs = [
        Question(
            "E_Q01",
            "current",
            "What is User's current employer?",
            expected_contains=[user_chain[-1]],
            expected_absent=[jamie_chain[-1]],
            scenario_tag="cross_entity",
        ),
        Question(
            "E_Q02",
            "current",
            "What is Jamie's current employer?",
            expected_contains=[jamie_chain[-1]],
            expected_absent=[user_chain[-1]],
            scenario_tag="cross_entity",
        ),
        Question(
            "E_Q03",
            "current",
            "What is Priya's current employer?",
            expected_contains=[priya_chain[-1]],
            expected_absent=[],
            scenario_tag="cross_entity",
        ),
        Question(
            "E_Q04",
            "history",
            "List Jamie's employers in chronological order.",
            expected_contains=jamie_chain,
            expected_absent=[],
            scenario_tag="cross_entity",
        ),
        Question(
            "E_Q05",
            "supersede",
            f"Did User ever work at {jamie_chain[0]}?",
            expected_contains=["no"],
            scenario_tag="cross_entity",
        ),
    ]
    return entries, qs


# ---------------------------------------------------------------------------
# S-rare-fact: A single isolated entry contains a fact that's queried later.
# Tests retrieval recall when the gold entry is one-of-1000.
# ---------------------------------------------------------------------------


def gen_rare_fact(n_entries: int, seed: int = 13) -> tuple[list[Entry], list[Question]]:
    rng = random.Random(seed)
    entries: list[Entry] = []
    ts = 0

    # 5 rare facts about distinct entities, each with ONE entry. Buried in noise.
    rare_facts = [
        ("Marcus", "Marcus was born in Lisbon.", "Lisbon"),
        ("Priya", "Priya plays the cello in a quartet.", "cello"),
        ("Diego", "Diego is allergic to shellfish.", "shellfish"),
        ("Jamie", "Jamie's middle name is Avery.", "Avery"),
        ("Luna", "Luna has a chipped left ear from a fight.", "chipped"),
    ]

    distractors = ENTITY_POOL[:40]

    # Place each rare fact at a deterministic location
    rare_positions = sorted(rng.sample(range(20, n_entries - 5), len(rare_facts)))

    rare_lookup = dict(zip(rare_positions, rare_facts))

    for i in range(n_entries):
        ts += 1
        if i in rare_lookup:
            name, text, _ = rare_lookup[i]
            mentions = ["User", name] if name != "User" else ["User"]
            e = Entry(_uuid("e_", ts), ts, text, _mentions(*mentions), [])
            entries.append(e)
        else:
            n2 = rng.choice(distractors)
            t = rng.choice(
                [
                    f"@User and @{n2} chatted today.",
                    f"@{n2} said hi.",
                    f"@User saw @{n2} at the cafe.",
                    f"@{n2} sent @User a meme.",
                    f"@User mentioned to @{n2} they're tired.",
                ]
            )
            e = Entry(_uuid("e_", ts), ts, t, _mentions("User", n2), [])
            entries.append(e)

    qs = [
        Question(
            "R_Q01",
            "entity",
            "Where was Marcus born?",
            expected_contains=["Lisbon"],
            scenario_tag="rare_fact",
        ),
        Question(
            "R_Q02",
            "entity",
            "What instrument does Priya play?",
            expected_contains=["cello"],
            scenario_tag="rare_fact",
        ),
        Question(
            "R_Q03",
            "entity",
            "What is Diego allergic to?",
            expected_contains=["shellfish"],
            scenario_tag="rare_fact",
        ),
        Question(
            "R_Q04",
            "entity",
            "What is Jamie's middle name?",
            expected_contains=["Avery"],
            scenario_tag="rare_fact",
        ),
        Question(
            "R_Q05",
            "entity",
            "Does Luna have any distinguishing features?",
            expected_contains=["chipped"],
            scenario_tag="rare_fact",
        ),
    ]
    return entries, qs


STRESS_GENERATORS = {
    "paraphrase": gen_paraphrase,
    "cross_entity": gen_cross_entity,
    "rare_fact": gen_rare_fact,
}
