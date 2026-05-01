"""Adversarial stress v3 — really try to break AEN-1.

- S-decay: gold entry is among the OLDEST entries (low ts), with many
  semantically-similar but irrelevant entries layered on top.
- S-no-tag: entries omit @Mentions (writer noise) and use pronouns.
- S-misleading-distractor: distractor entries semantically EXTREMELY close
  to the gold (different value, same template).
"""

from __future__ import annotations

import random

from generators import (
    EMPLOYER_POOL,
    ENTITY_POOL,
    Entry,
    Question,
    Ref,
    _mentions,
    _uuid,
)

# ---------------------------------------------------------------------------
# S-decay: the GOLD fact is at the start of a long log of distractor entries
# that semantically resemble the gold but are about other entities.
# ---------------------------------------------------------------------------


def gen_decay(n_entries: int, seed: int = 21) -> tuple[list[Entry], list[Question]]:
    rng = random.Random(seed)
    entries: list[Entry] = []
    ts = 0

    # Gold: User has employer = "Plover Robotics" (a unique name not in
    # distractors). Stated ONCE at turn 5. Then never restated.
    gold_employer = "Plover Robotics"
    gold_position = 5

    distractor_employers = [
        "Anthropic",
        "Google",
        "Stripe",
        "Meta",
        "OpenAI",
        "Apple",
        "Notion",
        "Cursor",
    ]
    distractor_names = ENTITY_POOL[:30]

    for i in range(n_entries):
        ts += 1
        if i == gold_position:
            u = _uuid("e_", ts)
            text = f"@User started a new job at {gold_employer}."
            e = Entry(
                u,
                ts,
                text,
                _mentions("User"),
                [],
                predicate="@User.employer",
                value=gold_employer,
            )
            entries.append(e)
        else:
            # distractors that semantically resemble gold but about OTHER entities
            n2 = rng.choice(distractor_names)
            emp = rng.choice(distractor_employers)
            templates = [
                f"@{n2} started a new job at {emp}.",
                f"@{n2} works at {emp} now.",
                f"@User mentioned that @{n2} is at {emp}.",
                f"@User had coffee with @{n2} who works at {emp}.",
                f"@{n2} loves working at {emp}.",
            ]
            t = rng.choice(templates)
            e = Entry(_uuid("e_", ts), ts, t, _mentions("User", n2), [])
            entries.append(e)

    qs = [
        Question(
            "DC_Q01",
            "current",
            "What is User's current employer?",
            expected_contains=["Plover"],
            scenario_tag="decay",
        ),
        Question(
            "DC_Q02",
            "supersede",
            "Has User ever worked at Plover Robotics?",
            expected_contains=["yes"],
            scenario_tag="decay",
        ),
    ]
    return entries, qs


# ---------------------------------------------------------------------------
# S-misleading: distractor entries use the SAME template as gold but with
# different @entity. Tests entity-specific filtering precision.
# ---------------------------------------------------------------------------


def gen_misleading(
    n_entries: int, seed: int = 22
) -> tuple[list[Entry], list[Question]]:
    rng = random.Random(seed)
    entries: list[Entry] = []
    ts = 0

    # User has a chain at companies; many other entities also have chains.
    # All entries use IDENTICAL template "<entity> started at <X>".
    # Tests whether retrieval filters by @User vs not.
    user_chain = ["Stripe", "Anthropic", "Meta", "OpenAI"]
    decoy_chains = {
        n: [rng.choice(EMPLOYER_POOL) for _ in range(rng.randint(2, 4))]
        for n in ENTITY_POOL[:20]
    }

    n_user_trans = len(user_chain)
    n_decoy_trans = sum(len(c) for c in decoy_chains.values())
    n_filler = max(0, n_entries - n_user_trans - n_decoy_trans)

    schedule = ["U"] * n_user_trans + ["D"] * n_decoy_trans + ["F"] * n_filler
    rng.shuffle(schedule)

    user_cursor = 0
    user_prev = None
    decoy_cursors = dict.fromkeys(decoy_chains, 0)
    decoy_prev = dict.fromkeys(decoy_chains)
    decoy_keys = list(decoy_chains.keys())

    for slot in schedule:
        ts += 1
        if slot == "U" and user_cursor < n_user_trans:
            emp = user_chain[user_cursor]
            user_cursor += 1
            u = _uuid("e_", ts)
            refs = [Ref(user_prev, "supersede")] if user_prev else []
            text = f"@User started a new job at {emp}."
            e = Entry(
                u,
                ts,
                text,
                _mentions("User"),
                refs,
                predicate="@User.employer",
                value=emp,
                is_current=(user_cursor == n_user_trans),
            )
            entries.append(e)
            user_prev = u
            continue
        if slot == "D":
            avail = [n for n in decoy_keys if decoy_cursors[n] < len(decoy_chains[n])]
            if not avail:
                slot = "F"
            else:
                name = rng.choice(avail)
                emp = decoy_chains[name][decoy_cursors[name]]
                decoy_cursors[name] += 1
                u = _uuid("e_", ts)
                refs = [Ref(decoy_prev[name], "supersede")] if decoy_prev[name] else []
                text = f"@{name} started a new job at {emp}."
                e = Entry(
                    u,
                    ts,
                    text,
                    _mentions(name),
                    refs,
                    predicate=f"@{name}.employer",
                    value=emp,
                    is_current=(decoy_cursors[name] == len(decoy_chains[name])),
                )
                entries.append(e)
                decoy_prev[name] = u
                continue
        # filler
        n2 = rng.choice(decoy_keys + ["User"])
        text = f"@{n2} had a normal day."
        e = Entry(_uuid("e_", ts), ts, text, _mentions(n2), [])
        entries.append(e)
        if len(entries) >= n_entries:
            break

    qs = [
        Question(
            "M_Q01",
            "current",
            "What is User's current employer?",
            expected_contains=[user_chain[-1]],
            scenario_tag="misleading",
        ),
        Question(
            "M_Q02",
            "history",
            "List User's employers in chronological order.",
            expected_contains=user_chain,
            scenario_tag="misleading",
        ),
        Question(
            "M_Q03",
            "supersede",
            f"Did User ever work at {user_chain[1]}?",
            expected_contains=["yes"],
            scenario_tag="misleading",
        ),
        # The killer: ask about a decoy entity's employer
        Question(
            "M_Q04",
            "current",
            "What is User's current employer (not anyone else's)?",
            expected_contains=[user_chain[-1]],
            expected_absent=[],
            scenario_tag="misleading",
        ),
    ]
    return entries, qs


# ---------------------------------------------------------------------------
# S-untagged-pronoun: writer omitted @mentions in some entries, using "she"/"he".
# Tests retrieval when entity-tagging is incomplete.
# ---------------------------------------------------------------------------


def gen_untagged(n_entries: int, seed: int = 23) -> tuple[list[Entry], list[Question]]:
    rng = random.Random(seed)
    entries: list[Entry] = []
    ts = 0

    # A focal entity Priya has 3 transitions; some entries about her are
    # untagged ("she got a promotion").
    priya_chain = ["Google", "Anthropic", "Cursor"]

    # First transition: TAGGED
    # Subsequent transitions: half untagged
    distractors = ENTITY_POOL[:30]

    n_priya_trans = len(priya_chain)
    n_untagged = 4  # filler entries about Priya without @-tag
    n_filler = max(0, n_entries - n_priya_trans - n_untagged)
    slots = ["P"] * n_priya_trans + ["UP"] * n_untagged + ["F"] * n_filler
    rng.shuffle(slots)

    cursor = 0
    prev_uuid = None
    for slot in slots:
        ts += 1
        if slot == "P" and cursor < n_priya_trans:
            emp = priya_chain[cursor]
            cursor += 1
            u = _uuid("e_", ts)
            refs = [Ref(prev_uuid, "supersede")] if prev_uuid else []
            # First mention is fully tagged; subsequent use pronouns
            if cursor == 1:
                text = f"@Priya started a new job at {emp}."
                mentions = _mentions("Priya")
            else:
                text = f"She started a new job at {emp}."
                # NOTE: NO mentions! This is the failure case.
                mentions = []
            e = Entry(
                u,
                ts,
                text,
                mentions,
                refs,
                predicate="@Priya.employer",
                value=emp,
                is_current=(cursor == n_priya_trans),
            )
            entries.append(e)
            prev_uuid = u
        elif slot == "UP":
            # untagged filler about Priya — text mentions her by name in
            # natural prose but not as @Tag (writer drift)
            t = rng.choice(
                [
                    "She likes the coffee at the new office.",
                    "She's been working late this month.",
                    "She got a fancy new monitor.",
                    "She mentioned a favorite restaurant.",
                ]
            )
            e = Entry(_uuid("e_", ts), ts, t, [], [])  # NO @ tags
            entries.append(e)
        else:
            # generic filler
            n2 = rng.choice(distractors)
            t = f"@User and @{n2} chatted."
            e = Entry(_uuid("e_", ts), ts, t, _mentions("User", n2), [])
            entries.append(e)
        if len(entries) >= n_entries:
            break

    qs = [
        Question(
            "U_Q01",
            "current",
            "Where does Priya currently work?",
            expected_contains=[priya_chain[-1]],
            scenario_tag="untagged",
        ),
        Question(
            "U_Q02",
            "history",
            "List Priya's employers in chronological order.",
            expected_contains=priya_chain,
            scenario_tag="untagged",
        ),
    ]
    return entries, qs


STRESS_V3 = {
    "decay": gen_decay,
    "misleading": gen_misleading,
    "untagged": gen_untagged,
}
