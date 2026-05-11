"""Cross-config sweep v2: v10 vs v14c vs v16 across gpt-5 family.

Sweep: {v10, v14c, v16} x {gpt-5.4-nano, gpt-5-nano, gpt-5-mini} x
       {low, medium} x 6 cases = 108 LLM calls.

Cases include 2 new binding tests that surface the "wife Anne" splitting
issue raised by the user.

Scoring: each (prompt, model, reasoning, case) gets PASS/FAIL on
qualitative checks specific to the case:
  - B1 binding-trip: every derivative mentioning Anne also says "wife"
  - B2 binding-role: every derivative mentioning Sarah also says "manager"
  - D1 mustang: ≤4 derivatives (anti-atomization)
  - C5 bare: output is exactly "Tokyo" (no hallucination padding)
  - F2 chess: exactly 1 derivative, near-verbatim
  - A4 acronyms: each acronym (JFK/POTUS/CMC/SAC) has its expansion
                somewhere among derivatives

We report per-prompt pass rate aggregated across (model, reasoning).
The prompt with highest aggregate pass rate is most robust.
"""

from __future__ import annotations

import asyncio
import json
import os
import textwrap
from typing import Any

import openai
from dotenv import load_dotenv
from probe_deriver_v64_compound_ids import PROMPT_DERIVER as V64
from probe_deriver_v65_completeness import PROMPT_DERIVER as V65

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")

PROMPTS = [
    ("v64", V64),
    ("v65", V65),
]

MODELS = ["gpt-5.4-nano", "gpt-5-nano", "gpt-5-mini"]
REASONING_LEVELS = ["low", "medium"]


# --- Cases with scoring rules ---

B1_BINDING_TRIP = (
    "B1_binding_trip",
    (
        "Last March I went to Tokyo with my wife Anne and our dog Mochi. "
        "We stayed at the Park Hyatt for 5 nights and ate ramen at Ichiran "
        "in Shibuya."
    ),
    "Every derivative mentioning Anne also keeps 'wife'; same for Mochi/'dog'.",
)

B2_BINDING_ROLE = (
    "B2_binding_role",
    "My manager Sarah approved the Q3 budget last week, and her boss the VP also signed off.",
    "Every derivative mentioning Sarah also says 'manager'.",
)

D1_MUSTANG = (
    "D1_mustang",
    textwrap.dedent("""\
        We finally finished restoring the 1967 Ford Mustang my grandfather \
        left me. It took almost three years from the day we towed it out \
        of the barn in upstate New York. The engine block had completely \
        seized up from a decade of moisture, so we ended up rebuilding the \
        289 V8 from scratch, with new pistons, rings, and a Holley \
        four-barrel carburetor. Bodywork was the worst part — the rear \
        quarter panels were so corroded we had to fabricate replacements \
        from sheet metal at a friend's shop in Poughkeepsie. We went with \
        the original Wimbledon White over a black interior, which was the \
        color my grandfather always said he wanted but never got around \
        to repainting. The chrome trim was sent out to a specialty \
        re-plater in Pennsylvania. We rebuilt the suspension with new \
        bushings, replaced the rear leaf springs, and put in a Borg-Warner \
        T-10 four-speed transmission to match the original spec. Last \
        weekend I finally drove it down to the Hudson Valley cars-and-coffee \
        and parked it next to my friend Marcus's '69 Camaro. It still pulls \
        slightly to the right under hard braking and the radio is a modern \
        replacement, but the rest is as close to factory as we could get. \
        My grandmother cried when she saw it.\
        """).strip(),
    "≤4 derivatives (cohesive narrative).",
)

C5_BARE = (
    "C5_bare",
    "Tokyo",
    "Output is exactly 'Tokyo' (no hallucinated description).",
)

F2_CHESS = (
    "F2_chess",
    "In chess, the move 25.g4 is played to gain space on the kingside and restrict the opponent's pawn structure.",
    "Exactly 1 derivative; preserves both 'kingside' and 'pawn'.",
)

A4_ACRONYMS = (
    "A4_acronyms",
    "TIL JFK was POTUS during the CMC in '62; SAC went to DEFCON 2.",
    "Acronyms have expansions: JFK/Kennedy, POTUS/President, CMC/Cuban Missile, SAC/Strategic Air Command.",
)

P1_PREFERENCE_CENTERED = (
    "P1_preference_centered",
    (
        "I've always preferred pour-over coffee from Stumptown over chain "
        "shops, especially their Hair Bender blend. I find chain shop "
        "espresso bitter and over-extracted; Stumptown's hand-pour technique "
        "with a 30-second bloom brings out chocolate and citrus notes I "
        "actually look forward to in the morning."
    ),
    "ONE derivative — this segment is centered on a preference. Should NOT atomize into separate facts about Stumptown, Hair Bender, chains, etc.",
)


def score_p1_preference_centered(derivs: list[str]) -> tuple[bool, str]:
    """Preference-centered segments should collapse to 1-2 derivatives.
    Atomization to ≥3 separate fact-derivatives is the failure mode.
    Also: 'Stumptown' and 'pour-over' (the key preference) should
    appear in the same derivative."""
    n = len(derivs)
    if n > 2:
        return False, f"atomized: {n} derivatives (preference-centered → 1-2)"
    if n < 1:
        return False, "0 derivatives"
    # Stumptown should appear with the preference signal
    has_stumptown_with_preference = any(
        "stumptown" in d.lower() and ("prefer" in d.lower() or "pour-over" in d.lower())
        for d in derivs
    )
    if not has_stumptown_with_preference:
        return False, "Stumptown not co-mentioned with pour-over/preference"
    return True, f"{n} derivatives"


FIL1_PURE_AFFIRMATIVE = (
    "FIL1_pure_affirmative",
    "yes",
    "Pure filler — emit 0 derivatives.",
)


FIL2_REACTIVE_FILLER = (
    "FIL2_reactive_filler",
    "Great point!",
    "Pure reaction — emit 0 derivatives.",
)


FIL3_AFFIRMATIVE_WITH_CONTENT = (
    "FIL3_affirmative_with_content",
    "ok, leaving Tuesday at 5 for the Boston trip",
    "Affirmative + content — emit ≥1 derivative covering the plan.",
)


FIL4_NEGATION_WITH_CONTENT = (
    "FIL4_negation_with_content",
    "no, I changed my mind about the Tuesday meeting with Dana",
    "Negation + content — emit ≥1 derivative covering the change.",
)


def score_fil_pure(derivs: list[str]) -> tuple[bool, str]:
    """Pure filler — should emit 0 derivatives."""
    if len(derivs) == 0:
        return True, "0 derivatives ✓"
    return False, f"emitted {len(derivs)} derivative(s) for pure filler: {derivs[:3]}"


def score_fil_with_content(derivs: list[str]) -> tuple[bool, str]:
    """Affirmative with content — should emit ≥1 covering the plan,
    and the derivative should contain 'Tuesday' (the time) and 'Boston' (scope)."""
    if len(derivs) == 0:
        return False, "0 derivatives — content was dropped"
    blob = " ".join(derivs).lower()
    if "tuesday" not in blob:
        return False, "missing 'Tuesday'"
    if "boston" not in blob:
        return False, "missing 'Boston'"
    return True, f"{len(derivs)} derivative(s), content preserved"


def score_fil4_negation_with_content(derivs: list[str]) -> tuple[bool, str]:
    """Negation + content — should emit ≥1 covering the change of mind,
    with 'Tuesday' (the time) and 'Dana' (co-actor)."""
    if len(derivs) == 0:
        return False, "0 derivatives — content was dropped"
    blob = " ".join(derivs).lower()
    if "tuesday" not in blob:
        return False, "missing 'Tuesday'"
    if "dana" not in blob:
        return False, "missing 'Dana'"
    if "chang" not in blob and "no" not in blob and "mind" not in blob:
        return False, "missing change-of-mind signal"
    return True, f"{len(derivs)} derivative(s), content preserved"


PR1_CLOCK = (
    "PR1_clock",
    (
        "I finished refurbishing the antique grandfather clock my aunt "
        "Eleanor gave me. It took me about 8 months working evenings and "
        "weekends. The brass pendulum was tarnished, so I polished it "
        "back to bright. The walnut case needed three coats of stain. "
        "The original chime mechanism still worked but the gears needed "
        "cleaning. I set it up in the entryway last Sunday."
    ),
    "Each atomized derivative must retain 'clock' or specific part-anchor (pendulum/case/chime/gears). Bare 'it'/'I' without anchor = FAIL.",
)


PR2_TREEHOUSE = (
    "PR2_treehouse",
    (
        "My friend Sarah and I spent Saturday building a treehouse in "
        "her backyard. We used pine for the frame and cedar for the "
        "panels. We finished in the late afternoon and her dog wouldn't "
        "stop barking at it."
    ),
    "Each atomized derivative must retain 'treehouse' or specific co-actor ('Sarah'). Bare 'we'/'it' without anchor = FAIL.",
)


PR3_APPLE_PIE = (
    "PR3_apple_pie",
    (
        "I made my grandmother's apple pie yesterday. I started by "
        "making the crust with butter and flour. I peeled and sliced 6 "
        "Granny Smith apples and tossed them with cinnamon and sugar. I "
        "baked it at 375 for 50 minutes until golden."
    ),
    "Each atomized derivative must retain 'pie' or specific anchor (apples/crust/cinnamon). Bare 'I made it'/'baked it' = FAIL.",
)


def _score_pronoun_anchor(
    derivs: list[str], anchors: list[str], case_label: str
) -> tuple[bool, str]:
    """Check that every atomized derivative retains at least one of the
    central anchors. The failure mode is pronoun substitution: a deriv
    saying 'It took 8 months' or 'We finished at 5' without any anchor
    word is unsearchable for queries about the central topic.

    If only 1 derivative emitted, scope is implicit -> pass.
    If multiple derivatives, EACH must contain at least one anchor.
    """
    n = len(derivs)
    if n == 0:
        return False, "0 derivatives"
    if n == 1:
        # Not atomized; check that the single deriv has the anchor
        if not any(a.lower() in derivs[0].lower() for a in anchors):
            return (
                False,
                f"single deriv missing all anchors {anchors}: '{derivs[0][:80]}'",
            )
        return True, "1 derivative (cohesive)"
    # Atomized: every deriv must have an anchor
    bad = []
    for i, d in enumerate(derivs):
        dl = d.lower()
        if not any(a.lower() in dl for a in anchors):
            bad.append((i, d[:80]))
    if bad:
        first_idx, first_text = bad[0]
        return (
            False,
            f"{len(bad)}/{n} derivs lack all anchors {anchors}; first: deriv #{first_idx} '{first_text}'",
        )
    return True, f"{n} derivs all anchored"


def score_pr1_clock(derivs: list[str]) -> tuple[bool, str]:
    anchors = ["clock", "pendulum", "walnut case", "chime", "gears"]
    return _score_pronoun_anchor(derivs, anchors, "PR1_clock")


def score_pr2_treehouse(derivs: list[str]) -> tuple[bool, str]:
    anchors = ["treehouse", "sarah", "frame", "panels"]
    return _score_pronoun_anchor(derivs, anchors, "PR2_treehouse")


def score_pr3_apple_pie(derivs: list[str]) -> tuple[bool, str]:
    anchors = ["pie", "crust", "apples", "cinnamon"]
    return _score_pronoun_anchor(derivs, anchors, "PR3_apple_pie")


CASES = [
    B1_BINDING_TRIP,
    B2_BINDING_ROLE,
    D1_MUSTANG,
    C5_BARE,
    F2_CHESS,
    P1_PREFERENCE_CENTERED,
    FIL1_PURE_AFFIRMATIVE,
    FIL2_REACTIVE_FILLER,
    FIL3_AFFIRMATIVE_WITH_CONTENT,
    FIL4_NEGATION_WITH_CONTENT,
    PR1_CLOCK,
    PR2_TREEHOUSE,
    PR3_APPLE_PIE,
]


def score_b1_binding_trip(derivs: list[str]) -> tuple[bool, str]:
    """Stricter: tests TWO things.
    (a) Binding glue: every deriv mentioning Anne also mentions 'wife'.
    (b) Scope propagation: every deriv about a trip event (Park Hyatt,
        Ichiran, Shibuya, ramen) mentions Anne, because she was on the
        trip and a query 'did Anne stay at the Park Hyatt?' should match.
    """
    for d in derivs:
        dl = d.lower()
        if "anne" in dl and "wife" not in dl:
            return False, f"deriv has Anne without 'wife': '{d[:80]}'"
        if "mochi" in dl and "dog" not in dl:
            return False, f"deriv has Mochi without 'dog': '{d[:80]}'"
    trip_event_keywords = ["park hyatt", "ichiran", "ramen", "shibuya"]
    for d in derivs:
        dl = d.lower()
        is_trip_event_only = (
            any(kw in dl for kw in trip_event_keywords)
            and "anne" not in dl
            and "tokyo" not in dl  # at minimum, scope (Tokyo) must travel
        )
        if is_trip_event_only:
            return False, f"trip-event deriv drops both Anne and Tokyo: '{d[:80]}'"
        # If the deriv is about a trip event, Anne (co-actor of scope) should appear.
        if any(kw in dl for kw in trip_event_keywords) and "anne" not in dl:
            return False, f"trip-event deriv drops Anne (co-actor): '{d[:80]}'"
    if not any("anne" in d.lower() for d in derivs):
        return False, "no derivative mentions Anne"
    return True, "ok"


def score_b2_binding_role(derivs: list[str]) -> tuple[bool, str]:
    """Every deriv mentioning Sarah keeps 'manager'."""
    for d in derivs:
        dl = d.lower()
        if "sarah" in dl and "manager" not in dl:
            return False, f"deriv has Sarah without 'manager': '{d[:80]}'"
    if not any("sarah" in d.lower() for d in derivs):
        return False, "no derivative mentions Sarah"
    return True, "ok"


def score_d1_mustang(derivs: list[str]) -> tuple[bool, str]:
    n = len(derivs)
    if n > 4:
        return False, f"atomized: {n} derivatives (>4)"
    if n < 1:
        return False, "0 derivatives"
    return True, f"{n} derivatives"


def score_c5_bare(derivs: list[str]) -> tuple[bool, str]:
    """Tokyo as bare entity — no hallucinated description like 'Tokyo is the capital of Japan'."""
    if len(derivs) == 0:
        return False, "0 derivatives"
    # Allowed forms: "Tokyo", "Tokyo.", or quoted variants
    for d in derivs:
        d_clean = d.strip().rstrip(".").strip()
        if d_clean.lower() != "tokyo":
            return False, f"hallucinated description: '{d}'"
    return True, "ok"


def score_f2_chess(derivs: list[str]) -> tuple[bool, str]:
    if len(derivs) != 1:
        return False, f"emitted {len(derivs)} derivatives (want exactly 1)"
    d = derivs[0].lower()
    if "kingside" not in d:
        return False, "missing 'kingside'"
    if "pawn" not in d:
        return False, "missing 'pawn'"
    if "25.g4" not in d:
        return False, "missing '25.g4'"
    return True, "ok"


ACRONYM_EXPANSIONS = [
    ("JFK", ["kennedy"]),
    ("POTUS", ["president"]),
    ("CMC", ["cuban missile", "cuban-missile"]),
    ("SAC", ["strategic air command"]),
]


def score_a4_acronyms(derivs: list[str]) -> tuple[bool, str]:
    all_text = " ".join(derivs).lower()
    missing = []
    for acro, expansions in ACRONYM_EXPANSIONS:
        if acro.lower() not in all_text:
            missing.append(f"{acro}-itself")
            continue
        if not any(exp in all_text for exp in expansions):
            missing.append(acro)
    if missing:
        return False, f"missing expansions for: {','.join(missing)}"
    return True, "ok"


SCORERS = {
    "B1_binding_trip": score_b1_binding_trip,
    "B2_binding_role": score_b2_binding_role,
    "D1_mustang": score_d1_mustang,
    "C5_bare": score_c5_bare,
    "F2_chess": score_f2_chess,
    "A4_acronyms": score_a4_acronyms,
    "P1_preference_centered": score_p1_preference_centered,
    "FIL1_pure_affirmative": score_fil_pure,
    "FIL2_reactive_filler": score_fil_pure,
    "FIL3_affirmative_with_content": score_fil_with_content,
    "FIL4_negation_with_content": score_fil4_negation_with_content,
    "PR1_clock": score_pr1_clock,
    "PR2_treehouse": score_pr2_treehouse,
    "PR3_apple_pie": score_pr3_apple_pie,
}


DERIVATIVES_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["derivatives"],
    "properties": {"derivatives": {"type": "array", "items": {"type": "string"}}},
}


async def derive(
    client, prompt_template: str, segment: str, model: str, reasoning_effort: str
):
    kwargs: dict[str, Any] = {
        "model": model,
        "input": prompt_template.format(segment=segment),
        "reasoning": {"effort": reasoning_effort},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "derivatives",
                "schema": DERIVATIVES_SCHEMA,
                "strict": True,
            }
        },
    }
    try:
        resp = await client.responses.create(**kwargs)
        payload = json.loads(resp.output_text)
        return list(payload.get("derivatives", [])), None
    except Exception as exc:
        return None, str(exc)[:140]


async def main():
    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(8)

    print("# CROSS-CONFIG SWEEP v2")
    print(f"# Prompts: {len(PROMPTS)} ({', '.join(p for p, _ in PROMPTS)})")
    print(f"# Models:  {len(MODELS)} ({', '.join(MODELS)})")
    print(f"# Reasoning: {REASONING_LEVELS}")
    print(f"# Cases:   {len(CASES)} ({', '.join(c[0] for c in CASES)})")
    print(
        f"# Total calls: {len(PROMPTS) * len(MODELS) * len(REASONING_LEVELS) * len(CASES)}"
    )
    print()

    async def one(prompt_name, template, case_label, segment, model, effort):
        async with sem:
            derivs, err = await derive(client, template, segment, model, effort)
            return (prompt_name, case_label, model, effort, derivs, err)

    tasks = []
    for prompt_name, template in PROMPTS:
        for case_label, segment, _ in CASES:
            for model in MODELS:
                for effort in REASONING_LEVELS:
                    tasks.append(
                        one(prompt_name, template, case_label, segment, model, effort)
                    )

    results = await asyncio.gather(*tasks)

    # Index: by[prompt][case][model][effort] = (derivs, err)
    by: dict = {}
    for prompt_name, case_label, model, effort, derivs, err in results:
        by.setdefault(prompt_name, {}).setdefault(case_label, {}).setdefault(model, {})[
            effort
        ] = (derivs, err)

    # Score each.
    print("## PER-CONFIG SCORES")
    print()
    header_configs = []
    for model in MODELS:
        for effort in REASONING_LEVELS:
            header_configs.append(
                f"{model.split('-')[1]}-{model.split('-')[2][:4]}@{effort[:1]}"
            )
    print(
        f"{'prompt':>6s}  {'case':>20s}  "
        + "  ".join(f"{h:>9s}" for h in header_configs)
    )
    print("-" * (6 + 2 + 20 + 2 + 11 * len(header_configs)))

    pass_counts = {}
    fail_details = {}

    for prompt_name, _ in PROMPTS:
        for case_label, _, _ in CASES:
            scorer = SCORERS[case_label]
            row = []
            for model in MODELS:
                for effort in REASONING_LEVELS:
                    derivs, err = by[prompt_name][case_label][model][effort]
                    if err:
                        row.append("ERR")
                        fail_details.setdefault((prompt_name, case_label), []).append(
                            f"  [{model}@{effort}] ERR: {err[:80]}"
                        )
                        continue
                    ok, msg = scorer(derivs)
                    if ok:
                        row.append("PASS")
                        pass_counts.setdefault(prompt_name, 0)
                        pass_counts[prompt_name] += 1
                    else:
                        row.append("FAIL")
                        fail_details.setdefault((prompt_name, case_label), []).append(
                            f"  [{model}@{effort}] FAIL: {msg}"
                        )
            print(
                f"{prompt_name:>6s}  {case_label:>20s}  "
                + "  ".join(f"{r:>9s}" for r in row)
            )
        print()

    total_configs = len(MODELS) * len(REASONING_LEVELS) * len(CASES)
    print("## AGGREGATE PASS RATE")
    print()
    for prompt_name, _ in PROMPTS:
        n_pass = pass_counts.get(prompt_name, 0)
        pct = 100.0 * n_pass / total_configs
        print(f"  {prompt_name}: {n_pass}/{total_configs} ({pct:.1f}%)")
    print()

    print("## FAILURE DETAILS")
    print()
    for (prompt_name, case_label), details in sorted(fail_details.items()):
        print(f"### {prompt_name}  {case_label}")
        for d in details:
            print(d)
        print()

    print("## SAMPLE OUTPUTS")
    print()
    sample_cases = ["B1_binding_trip", "D1_mustang", "C5_bare"]
    for prompt_name, _ in PROMPTS:
        for case_label in sample_cases:
            for model in MODELS:
                for effort in REASONING_LEVELS:
                    derivs, err = by[prompt_name][case_label][model][effort]
                    if err:
                        continue
                    print(
                        f"### {prompt_name} / {case_label} / {model}@{effort}  (n={len(derivs)})"
                    )
                    for d in derivs[:5]:
                        print(f"  - {d[:170]}")
                    if len(derivs) > 5:
                        print(f"  ... +{len(derivs) - 5} more")
                    print()

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
