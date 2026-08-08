"""Compare check prompts (original vs reframed) on nano and Claude.

Isolates two confounds:
  (a) Prompt strength -- maybe original Loop B/C were too weak. Test
      with mechanical-reformulated prompts.
  (b) Same-model bias -- maybe nano can't run judgmental checks
      regardless of prompt. Compare against Claude with same prompts.

Cells:
  - nano  + ORIGINAL Loop B
  - nano  + REFRAMED Loop B
  - claude + REFRAMED Loop B
  - nano  + ORIGINAL Loop C
  - nano  + REFRAMED Loop C
  - claude + REFRAMED Loop C

Test inputs: V7 drafts captured from test_deictic_loop.out, 10 cases
each. The drafts are the OUTPUT of V7 generation -- the question is
whether each checker correctly identifies which drafts contain a
problem.

Expected ground truth:
  Loop C generic-you check:
    - D6:3 vocative on addressed "Are you excited" -> CLEAN
    - D6:10 advice with no vocatives -> CLEAN
    - D29:6 generic "when you get" no vocative -> CLEAN
    - D30:18 generic "all you need" no vocative -> CLEAN
    (Original Loop C false-flagged D6:3.)

  Loop B wrong-context check:
    - D28:10 "it inspired others" substituted to "this car inspired others"
      -> FLAG (it = the blog post, not the car)
    - D28:20 "I found it" -> "I found this car" -> CLEAN (it = the car)
    - D28:3 partial "my fur kids" substitution -> CLEAN
    - D27:5 nothing substituted -> CLEAN
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent
        / "longmemeval/llm_pipeline_probe"),
)

from dotenv import load_dotenv
from openai import AsyncOpenAI


CASES = [
    {
        "id": "D6:3",
        "label": "addressed-you 'Are you excited'",
        "speaker": "Nate", "addressee": "Joanna",
        "source": "Congrats! How did it go? Are you excited?",
        "draft": "Congrats! How did it go? Are you excited, Joanna?",
        "neighbors": "- Joanna: Hey Nate, I just had a writing audition for a gig today!\n- Nate: Wow, congratulations Joanna!\n",
        "loop_c_truth": "CLEAN",
        "loop_b_truth": "CLEAN",
    },
    {
        "id": "D30:18",
        "label": "generic 'all you need'",
        "speaker": "John", "addressee": "James",
        "source": "Not at all, all you need is a gamepad and a sense of timing.",
        "draft": "Not at all, all you need is a gamepad and a sense of timing.",
        "neighbors": "- James: I want to try FIFA 23 but worry about complex controls.\n- John: It's pretty easy to pick up.\n",
        "loop_c_truth": "CLEAN",
        "loop_b_truth": "CLEAN",
    },
    {
        "id": "D28:10",
        "label": "wrong-context 'this car inspired'",
        "speaker": "Dave", "addressee": "Calvin",
        "source": "I recently posted about how I made this car look like a beast, and it was great to hear it inspired others to start their own DIY projects.",
        "draft": "I recently posted about how I made this car look like a beast, and it was great to hear this car inspired others to start their own DIY projects.",
        "neighbors": "- Dave: I recently started a blog on car mods. Just take a look at this beautiful car!\n- Calvin: Cool, Dave! Your blog is awesome.\n",
        "loop_c_truth": "CLEAN",
        "loop_b_truth": "FLAG",
    },
    {
        "id": "D28:3",
        "label": "partial 'my fur kids' substitution",
        "speaker": "Audrey", "addressee": "Andrew",
        "source": "Here's a pic of them, looking all groomed. Look at those shiny coats! To top it off, they were really good at the salon - I always worry about them in new places.",
        "draft": "Here's a pic of them, looking all groomed. Look at those shiny coats! To top it off, my fur kids were really good at the salon - I always worry about them in new places.",
        "neighbors": "- Audrey: Last Friday I took my fur kids to the pet salon - they were so psyched.\n- Andrew: Do you have any pictures of them all groomed up?\n",
        "loop_c_truth": "CLEAN",
        "loop_b_truth": "CLEAN",
    },
    {
        "id": "D27:5",
        "label": "no substitution 'What city is it'",
        "speaker": "Calvin", "addressee": "Dave",
        "source": "Wow, that view looks awesome! What city is it? Have you taken any good pictures lately?",
        "draft": "Wow, that view looks awesome! What city is it? Have you taken any good pictures lately?",
        "neighbors": "- Dave: I've been getting into photography. Look at this magnificent sunset I captured.\n",
        "loop_c_truth": "CLEAN",
        "loop_b_truth": "CLEAN",
    },
    {
        "id": "D6:10",
        "label": "advice generic-you no vocatives",
        "speaker": "Joanna", "addressee": "Nate",
        "source": "Practicing and gathering feedback will make you better. Have faith in yourself and continue following your writing dreams - it's tough but worth it.",
        "draft": "Practicing and gathering feedback will make you better. Have faith in yourself and continue following your writing dreams - it's tough but worth it.",
        "neighbors": "- Nate: Any tips for someone who wants to write more seriously?\n",
        "loop_c_truth": "CLEAN",
        "loop_b_truth": "CLEAN",
    },
    {
        "id": "D28:20",
        "label": "3x bare 'it' -> 'this car'",
        "speaker": "Dave", "addressee": "Calvin",
        "source": "I found it last week, and it was in bad shape, but I saw the potential. I spent ages restoring it.",
        "draft": "I found this car last week, and this car was in bad shape, but I saw the potential. I spent ages restoring this car.",
        "neighbors": "- Dave: I recently posted about how I made this car look like a beast.\n- Calvin: Cool! Was the work hard?\n",
        "loop_c_truth": "CLEAN",
        "loop_b_truth": "CLEAN",
    },
    {
        "id": "D29:6",
        "label": "generic 'when you get those moments'",
        "speaker": "Nate", "addressee": "Joanna",
        "source": "It's incredible when you get those moments of joy.",
        "draft": "It's incredible when you get those moments of joy.",
        "neighbors": "- Joanna: It was so exciting when my screenplay first appeared on screen!\n",
        "loop_c_truth": "CLEAN",
        "loop_b_truth": "CLEAN",
    },
]


PROMPT_ORIG_C = """\
You will verify a deictic-resolution rewrite for generic-"you" \
vocative errors -- places where a vocative ", {addressee}," was \
inserted next to a "you" that is actually GENERIC (meaning \
"anyone" / "a person").

ORIGINAL MESSAGE: {source}

REWRITE: {draft}

For each vocative ", {addressee}," in the REWRITE that appears next \
to a "you" or "your" or "yourself", apply this test:
- Substitute "a person" for the "you" in the original sentence.
- If the substituted sentence still makes sense as a general \
statement (advice, hypothetical, general truth), the "you" is \
GENERIC and the vocative is wrong.

If all vocatives are on actually-addressed "you"s, respond: CLEAN

Otherwise respond with:
GENERIC_YOU_VOCATIVES:
- "<context of the generic 'you'>" — vocative is wrong because <reason>
- ..."""


PROMPT_REFRAMED_C = """\
You will check a deictic-resolution rewrite for incorrectly-inserted \
vocatives next to "you" pronouns.

ORIGINAL MESSAGE: {source}
REWRITE: {draft}

For each ", {addressee}," vocative in the REWRITE next to a \
"you" / "your" / "yours" / "yourself", apply this MECHANICAL \
syntactic test:

STEP 1 -- Addressed-question test: does the sentence containing the \
vocative START with one of these auxiliary patterns?
  Are/Do/Does/Did/Have/Has/Had/Can/Could/Will/Would/Should/Were/Was/Aren't/Don't/Didn't/Haven't/Hasn't/Can't/Won't/Wouldn't/Shouldn't \
+ "you"
If YES -> the "you" is ADDRESSED, vocative is CORRECT, do NOT flag.

STEP 2 -- Generic-pattern test: does the sentence match one of these \
patterns (with optional words inside brackets)?
  - "you have to [verb]"
  - "you need [to]? [verb]"
  - "you want to [verb]"
  - "all you need is [noun]"
  - "all you have to do is [verb]"
  - "when you [verb]"
  - "if you [verb]"
  - "you get [noun]"
  - "yourself" in an imperative-style advice sentence (no question, \
no specific recent action)
  - "your [noun]" in advice phrasing ("follow your dreams", "trust \
your gut")
If YES -> the "you" is GENERIC, vocative is WRONG, FLAG it.

STEP 3 -- If NEITHER step 1 NOR step 2 matches, DEFAULT to CORRECT \
(don't flag). Most "you" in a 2-person chat is addressed unless it \
clearly matches a generic pattern.

Output format:

If all vocatives pass: respond exactly CLEAN

Otherwise:
WRONG_VOCATIVES:
- sentence: "<full sentence>" -- matches generic pattern "<pattern>"
- ..."""


PROMPT_ORIG_B = """\
You will verify a deictic-resolution rewrite for semantically \
inappropriate substitutions -- where a pronoun was replaced with a \
noun phrase that does not fit the pronoun's syntactic role or \
semantic type.

ORIGINAL MESSAGE: {source}

PRIOR TURNS:
{neighbors}

REWRITE: {draft}

For each substitution in the REWRITE (places where a pronoun in the \
ORIGINAL MESSAGE became a noun phrase), check:
- Does the noun phrase fit the syntactic role of the pronoun?
- Is the noun phrase the SEMANTICALLY APPROPRIATE antecedent given \
the ORIGINAL MESSAGE context, not just any literal match in PRIOR \
TURNS?

If all substitutions are semantically correct, respond: CLEAN

Otherwise respond with:
WRONG_SUBSTITUTIONS:
- "<original pronoun in context>" → "<substituted phrase>" because <reason>
- ..."""


PROMPT_REFRAMED_B = """\
You will check a deictic-resolution rewrite for substitutions where \
the noun phrase doesn't fit the type required by the original pronoun.

ORIGINAL MESSAGE: {source}
PRIOR TURNS:
{neighbors}
REWRITE: {draft}

STEP 1 -- Enumerate every substitution in the REWRITE. A \
substitution is a position where a pronoun (it/they/them/he/she/this\
/that/these/those/here/there) in the ORIGINAL MESSAGE was replaced \
by a noun phrase in the REWRITE.

For each substitution, output a 3-line block:
  ORIGINAL: "<pronoun + 4 words of context from ORIGINAL>"
  SUBSTITUTED: "<noun phrase used in REWRITE>"
  REQUIRED TYPE: <based on surrounding verb/adjective, what kind of \
referent must this be?> -- choose from: place / person / object / \
event / time / quote / abstract-concept

STEP 2 -- For each block, classify the SUBSTITUTED TYPE: place / \
person / object / event / time / quote / abstract-concept.

STEP 3 -- For each block, compare REQUIRED TYPE vs SUBSTITUTED TYPE.

Output format:

If all substitutions match types (no mismatch), respond exactly CLEAN

Otherwise:
TYPE_MISMATCHES:
- "<ORIGINAL>" -> "<SUBSTITUTED>": required <REQUIRED_TYPE>, got \
<SUBSTITUTED_TYPE>
- ..."""


async def _call_nano(client, prompt):
    resp = await client.chat.completions.create(
        model="gpt-5.4-nano", reasoning_effort="low",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip(), resp.usage


async def _call_claude(prompt):
    proc = await asyncio.create_subprocess_exec(
        "claude", "-p", prompt,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ},
        cwd="/tmp/fresh-test",
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
    except asyncio.TimeoutError:
        proc.kill()
        return "(TIMEOUT)"
    if proc.returncode != 0:
        return f"(EXIT {proc.returncode})"
    return out.decode().strip()


def _classify(verdict: str) -> str:
    """Map verdict -> CLEAN / FLAG."""
    v = verdict.strip().upper()
    if v.startswith("CLEAN") or "CLEAN" in v.split("\n")[0]:
        return "CLEAN"
    return "FLAG"


async def main():
    load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    print(f"{'='*92}")
    print("LOOP C -- generic-you vocative check")
    print(f"{'='*92}")
    print(f"{'case':<40} {'truth':<6} {'nano-orig':<10} "
          f"{'nano-rfr':<10} {'cld-rfr':<10}")
    c_results = []
    for case in CASES:
        truth = case["loop_c_truth"]
        orig_prompt = PROMPT_ORIG_C.format(**case)
        rfr_prompt = PROMPT_REFRAMED_C.format(**case)
        nano_orig, _ = await _call_nano(client, orig_prompt)
        nano_rfr, _ = await _call_nano(client, rfr_prompt)
        cld_rfr = await _call_claude(rfr_prompt)
        n_o = _classify(nano_orig)
        n_r = _classify(nano_rfr)
        c_r = _classify(cld_rfr)
        c_results.append((case["id"], truth, n_o, n_r, c_r,
                          nano_orig, nano_rfr, cld_rfr))
        print(f"{case['id'] + ' ' + case['label']:<40} {truth:<6} "
              f"{n_o:<10} {n_r:<10} {c_r:<10}")

    print(f"\n{'='*92}")
    print("LOOP B -- wrong-context substitution check")
    print(f"{'='*92}")
    print(f"{'case':<40} {'truth':<6} {'nano-orig':<10} "
          f"{'nano-rfr':<10} {'cld-rfr':<10}")
    b_results = []
    for case in CASES:
        truth = case["loop_b_truth"]
        orig_prompt = PROMPT_ORIG_B.format(**case)
        rfr_prompt = PROMPT_REFRAMED_B.format(**case)
        nano_orig, _ = await _call_nano(client, orig_prompt)
        nano_rfr, _ = await _call_nano(client, rfr_prompt)
        cld_rfr = await _call_claude(rfr_prompt)
        n_o = _classify(nano_orig)
        n_r = _classify(nano_rfr)
        c_r = _classify(cld_rfr)
        b_results.append((case["id"], truth, n_o, n_r, c_r,
                          nano_orig, nano_rfr, cld_rfr))
        print(f"{case['id'] + ' ' + case['label']:<40} {truth:<6} "
              f"{n_o:<10} {n_r:<10} {c_r:<10}")

    # Compute agreement with truth per cell
    print(f"\n{'='*92}")
    print("AGREEMENT WITH GROUND TRUTH (count of correct verdicts)")
    print(f"{'='*92}")
    n_cases = len(CASES)
    for loop_name, results in [("C", c_results), ("B", b_results)]:
        n_o_correct = sum(1 for r in results if r[1] == r[2])
        n_r_correct = sum(1 for r in results if r[1] == r[3])
        c_r_correct = sum(1 for r in results if r[1] == r[4])
        print(f"Loop {loop_name}: nano-orig {n_o_correct}/{n_cases}, "
              f"nano-rfr {n_r_correct}/{n_cases}, "
              f"claude-rfr {c_r_correct}/{n_cases}")

    # Show full verdicts for divergent cases
    print(f"\n{'='*92}")
    print("DETAILED VERDICTS for cases where checkers disagree")
    print(f"{'='*92}")
    for loop_name, results in [("C", c_results), ("B", b_results)]:
        for cid, truth, n_o, n_r, c_r, n_o_v, n_r_v, c_r_v in results:
            if not (n_o == n_r == c_r):
                print(f"\n--- Loop {loop_name} {cid} (truth={truth}) ---")
                print(f"NANO ORIG ({n_o}): {n_o_v[:200]}")
                print(f"NANO REFR ({n_r}): {n_r_v[:200]}")
                print(f"CLAUDE REFR ({c_r}): {c_r_v[:200]}")


if __name__ == "__main__":
    asyncio.run(main())
