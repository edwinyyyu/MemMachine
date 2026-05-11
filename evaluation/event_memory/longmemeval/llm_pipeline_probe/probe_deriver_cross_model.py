"""Cross-model robustness test: v10 vs v14c vs v15.

The user's concern: a prompt that 'happens to nudge' one model into the
correct behavior is brittle. We want one that is interpreted the same by
most models.

Test setup:
  - 5 discriminating cases that surface different failure modes
  - 3 candidate prompts: v10 (rules), v14c (goal-first+compression), v15 (demos)
  - 4 models spanning the families: gpt-5.4-nano, gpt-5-nano, gpt-4.1-nano,
    gpt-4o-mini

For each (prompt, model, case): record N derivatives + the actual outputs.
The robust prompt is the one with most CONSISTENT n_derivatives across
models on the same case, AND the outputs that don't fall into known
failure modes (atomization, hallucination, near-clones).

Reasoning effort: gpt-5.x supports it (use "medium"), gpt-4.x doesn't (skip).
"""

from __future__ import annotations

import asyncio
import json
import os
import textwrap
from typing import Any

import openai
from dotenv import load_dotenv

# Import the three candidate prompts.
from probe_deriver_v10_tight import PROMPT_DERIVER as V10
from probe_deriver_v14c_fixed import PROMPT_DERIVER as V14C
from probe_deriver_v15_demos import PROMPT_DERIVER as V15

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


# Models to test. (model_id, supports_reasoning, reasoning_effort)
MODELS = [
    ("gpt-5.4-nano", True, "medium"),
    ("gpt-5-nano", True, "low"),
    ("gpt-4.1-nano", False, None),
    ("gpt-4o-mini", False, None),
]


PROMPTS = [
    ("v10", V10),
    ("v14c", V14C),
    ("v15", V15),
]


# 5 cases that surface different failure modes (anti-fragment, hallucination,
# non-prose, scope propagation, acronym bridging).
CASES = [
    ("C1_focused", "I went to Tokyo last March with my wife Anne.", "ideal: 1 deriv"),
    ("C5_bare", "Tokyo", "ideal: 1 deriv 'Tokyo'"),
    (
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
        "ideal: 1-3 cohesive derivs (NOT atomized)",
    ),
    (
        "C3_cipher",
        "Khoor, zruog! Wklv lv d phvvdjh.",
        "ideal: 1-2 derivs (decoded + description)",
    ),
    (
        "A4_acronym",
        "TIL JFK was POTUS during the CMC in '62; SAC went to DEFCON 2.",
        "ideal: 2 derivs, both forms inline",
    ),
]


DERIVATIVES_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["derivatives"],
    "properties": {"derivatives": {"type": "array", "items": {"type": "string"}}},
}


async def derive(
    client,
    prompt_template: str,
    segment: str,
    model: str,
    supports_reasoning: bool,
    reasoning_effort: str | None,
):
    """Call the model with structured json_schema output. Return list of derivatives."""
    kwargs: dict[str, Any] = {
        "model": model,
        "input": prompt_template.format(segment=segment),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "derivatives",
                "schema": DERIVATIVES_SCHEMA,
                "strict": True,
            }
        },
    }
    if supports_reasoning and reasoning_effort:
        kwargs["reasoning"] = {"effort": reasoning_effort}

    try:
        resp = await client.responses.create(**kwargs)
        payload = json.loads(resp.output_text)
        return list(payload.get("derivatives", [])), None
    except Exception as exc:
        return None, str(exc)


async def main():
    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(6)

    print("# CROSS-MODEL PROMPT ROBUSTNESS TEST")
    print(f"# Models:  {', '.join(m for m, _, _ in MODELS)}")
    print(f"# Prompts: {', '.join(n for n, _ in PROMPTS)}")
    print(f"# Cases:   {len(CASES)} discriminating segments")
    print()

    # Run everything in parallel.
    async def one(
        prompt_name, prompt_template, case_label, segment, model, supports_r, effort
    ):
        async with sem:
            derivs, err = await derive(
                client, prompt_template, segment, model, supports_r, effort
            )
            return (prompt_name, case_label, model, derivs, err)

    tasks = []
    for prompt_name, prompt_template in PROMPTS:
        for case_label, segment, _ideal in CASES:
            for model, supports_r, effort in MODELS:
                tasks.append(
                    one(
                        prompt_name,
                        prompt_template,
                        case_label,
                        segment,
                        model,
                        supports_r,
                        effort,
                    )
                )

    results = await asyncio.gather(*tasks)

    # Organize results: results[prompt][case][model] = (n_derivs, derivs_or_err)
    by = {}
    for prompt_name, case_label, model, derivs, err in results:
        by.setdefault(prompt_name, {}).setdefault(case_label, {})[model] = (derivs, err)

    # Compact summary table: n_derivatives by (prompt, case, model)
    print("## N_DERIVATIVES SUMMARY (target_consistency in parens)")
    print()
    header_models = "  ".join(f"{m[:14]:>14s}" for m, _, _ in MODELS)
    print(f"{'prompt':>6s}  {'case':>11s}  {header_models}   ideal")
    print("-" * (6 + 2 + 11 + 2 + len(header_models) + 3 + 30))

    for prompt_name, _ in PROMPTS:
        for case_label, _, ideal in CASES:
            row_models = []
            for model, _, _ in MODELS:
                derivs, err = by[prompt_name][case_label][model]
                if err:
                    row_models.append(f"{'ERR':>14s}")
                else:
                    row_models.append(f"{len(derivs):>14d}")
            row_str = "  ".join(row_models)
            print(f"{prompt_name:>6s}  {case_label:>11s}  {row_str}   {ideal}")
        print()

    # Cross-model variance per prompt: variance of n_derivatives across models per case.
    print("## CROSS-MODEL VARIANCE (lower = more robust)")
    print()
    print(f"{'prompt':>6s}  {'case':>11s}  {'min':>4s}  {'max':>4s}  {'spread':>6s}")
    print("-" * 40)
    for prompt_name, _ in PROMPTS:
        total_spread = 0
        for case_label, _, _ in CASES:
            counts = []
            for model, _, _ in MODELS:
                derivs, err = by[prompt_name][case_label][model]
                if not err and derivs is not None:
                    counts.append(len(derivs))
            if counts:
                spread = max(counts) - min(counts)
                total_spread += spread
                print(
                    f"{prompt_name:>6s}  {case_label:>11s}  {min(counts):>4d}  {max(counts):>4d}  {spread:>6d}"
                )
        print(
            f"{prompt_name:>6s}  {'TOTAL':>11s}  {'-':>4s}  {'-':>4s}  {total_spread:>6d}"
        )
        print()

    # Full outputs for inspection.
    print("## FULL OUTPUTS")
    print()
    for prompt_name, _ in PROMPTS:
        print(f"### PROMPT = {prompt_name}")
        for case_label, segment, ideal in CASES:
            print(f"  case = {case_label}  |  ideal: {ideal}")
            for model, _, _ in MODELS:
                derivs, err = by[prompt_name][case_label][model]
                if err:
                    print(f"    [{model}] ERROR: {err[:120]}")
                else:
                    print(f"    [{model}] n={len(derivs)}")
                    for d in derivs[:6]:
                        print(f"        - {d[:160]}")
                    if len(derivs) > 6:
                        print(f"        ... +{len(derivs) - 6} more")
            print()
        print()

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
