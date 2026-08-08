"""Nano-only self-eval-loop deictic resolver test.

Architecture per case:
  1. INITIAL: nano produces V7-prompt rewrite (single call)
  2. LOOP A -- HALLUCINATION CHECK: nano enumerates substituted noun
     phrases in the draft, verifies each appears word-for-word in
     source or PRIOR TURNS, lists those that don't. If any, nano
     produces a corrected draft removing them.
  3. LOOP B -- WRONG-CONTEXT CHECK: nano verifies each pronoun-
     substitution is semantically appropriate as antecedent. If any
     mismatches, nano produces a corrected draft reverting them.
  4. LOOP C -- GENERIC-YOU CHECK: nano enumerates each "you, X,"
     vocative in the draft, applies the "substitute 'a person'" self-
     check, lists any vocatives on generic "you"s. If any, nano
     produces a corrected draft removing those vocatives.

Each check loop: 1 detect call + (if issues) 1 fix call. Max 2
iterations per loop. All nano @ low reasoning.

Per-case cost ceiling: 1 (initial) + 3 loops * 2 calls = 7 nano
calls. Average probably 4-5.

Tested on 10 cases covering all the failure modes from the V7 audit.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent
        / "longmemeval/llm_pipeline_probe"),
)

from dotenv import load_dotenv
from openai import AsyncOpenAI


NB = 8

# 10 failure-mode cases (covers hallucination, generic-you,
# wrong-context, ownership, edge cases).
TARGETS = [
    ("conv-42", "D6:3",   "addressed-you vocative"),
    ("conv-47", "D30:18", "generic-you (gamepad) -- WAS V7 fail"),
    ("conv-50", "D28:10", "demonstrative + ownership (car)"),
    ("conv-44", "D28:3",  "3p antecedent (fur kids in PRIOR)"),
    ("conv-50", "D27:5",  "bare 'it' no antecedent (city)"),
    ("conv-42", "D6:10",  "advice generic-you (yourself) -- WAS V7 fail"),
    ("conv-44", "D19:29", "wrong-ownership 'my fur babies' -- WAS V7 fail"),
    ("conv-50", "D28:20", "3x bare 'it' (found car)"),
    ("conv-42", "D29:6",  "generic-you (moments of joy) -- WAS V7 fail"),
    ("conv-44", "D18:6",  "Glad you + They (mid-message)"),
]


PROMPT_V7_GENERATE = """\
SPEAKER: {speaker}
ADDRESSEE: {addressee}

{speaker} sent the MESSAGE below to {addressee}. Rewrite the MESSAGE \
for a reader who has not seen the PRIOR TURNS. The MESSAGE words \
must appear in your output in their original order. The only edits \
you may make are vocative insertions and pronoun-for-name \
substitutions.

Rules (compressed):
1. ADDRESSED "you" (question to {addressee} or claim about their \
specific recent action): KEEP the pronoun, INSERT ", {addressee}, " \
right after it.
2. GENERIC "you" (advice / hypothetical / general statement; \
substitute "a person" makes sense): KEEP unchanged.
3. 3p pronoun (he/she/they/etc.) referring to a person or thing \
the PRIOR TURNS named WORD-FOR-WORD: REPLACE with that name or \
noun phrase. If no word-for-word match → KEEP.
4. Demonstrative (this/that/here/there) or bare "it" referring to \
something the PRIOR TURNS named WORD-FOR-WORD: REPLACE with that \
phrase, or "my X" if PRIOR TURNS establish {speaker} as owner/maker.
5. Ambiguous referent → KEEP unchanged. Never invent.
6. 3p referring to {speaker} themselves → first person "I/my/me". \
NEVER write "{speaker}" as self-reference.
7. KEEP unchanged: "I/my/me", "we/us/our", temporal references, \
existing vocatives at start, all other words/punctuation/emoji.

Output ONLY the rewritten message on a single line.

PRIOR TURNS:
{neighbors_block}MESSAGE FROM {speaker} TO {addressee}:
{passage}"""


PROMPT_LOOP_A_DETECT_HALLUCINATION = """\
You will verify a deictic-resolution rewrite for hallucinated \
content -- specific noun phrases that do NOT appear word-for-word \
in the source or prior turns.

ORIGINAL MESSAGE: {passage}

PRIOR TURNS:
{neighbors_block}

REWRITE: {draft}

For each noun phrase, name, or specific detail in the REWRITE that \
is NOT in the ORIGINAL MESSAGE, check whether it appears \
word-for-word in the PRIOR TURNS above.

If a phrase in REWRITE is not in ORIGINAL MESSAGE and not in PRIOR \
TURNS, it is hallucinated. List it.

If REWRITE is clean (every added or substituted phrase appears \
word-for-word in PRIOR TURNS), respond with exactly: CLEAN

Otherwise respond with:
HALLUCINATIONS:
- "<phrase>" — not found in source or prior turns
- ...

Be strict. Even a single capitalized name not in PRIOR TURNS \
counts as hallucination."""


PROMPT_LOOP_A_FIX = """\
Below is a deictic-resolution rewrite with hallucinated content \
that was added without source.

ORIGINAL MESSAGE: {passage}

PRIOR TURNS:
{neighbors_block}

CURRENT REWRITE: {draft}

HALLUCINATIONS to remove:
{issues}

Produce a corrected rewrite that REVERTS the hallucinated \
substitutions back to the original pronoun or phrase from the \
ORIGINAL MESSAGE. Keep all other parts of the rewrite unchanged.

Output ONLY the corrected rewrite on a single line."""


PROMPT_LOOP_B_DETECT_WRONG_CONTEXT = """\
You will verify a deictic-resolution rewrite for semantically \
inappropriate substitutions -- where a pronoun was replaced with a \
noun phrase that does not fit the pronoun's syntactic role or \
semantic type.

ORIGINAL MESSAGE: {passage}

PRIOR TURNS:
{neighbors_block}

REWRITE: {draft}

For each substitution in the REWRITE (places where a pronoun in the \
ORIGINAL MESSAGE became a noun phrase), check:
- Does the noun phrase fit the syntactic role of the pronoun? \
("What city is X?" → X must be a place. "I love X" → X must be the \
object the speaker mentions. "they said Y" → "they" must refer to \
a person/group/animal.)
- Is the noun phrase the SEMANTICALLY APPROPRIATE antecedent given \
the ORIGINAL MESSAGE context, not just any literal match in PRIOR \
TURNS?

If all substitutions are semantically correct, respond: CLEAN

Otherwise respond with:
WRONG_SUBSTITUTIONS:
- "<original pronoun in context>" → "<substituted phrase>" because <reason>
- ..."""


PROMPT_LOOP_B_FIX = """\
Below is a deictic-resolution rewrite with semantically wrong \
substitutions.

ORIGINAL MESSAGE: {passage}

PRIOR TURNS:
{neighbors_block}

CURRENT REWRITE: {draft}

WRONG SUBSTITUTIONS to revert:
{issues}

Produce a corrected rewrite that REVERTS each wrong substitution \
back to the original pronoun from the ORIGINAL MESSAGE. Keep all \
other parts of the rewrite unchanged.

Output ONLY the corrected rewrite on a single line."""


PROMPT_LOOP_C_DETECT_GENERIC_YOU = """\
You will verify a deictic-resolution rewrite for generic-"you" \
vocative errors -- places where a vocative ", {addressee}," was \
inserted next to a "you" that is actually GENERIC (meaning \
"anyone" / "a person").

ORIGINAL MESSAGE: {passage}

REWRITE: {draft}

For each vocative ", {addressee}," in the REWRITE that appears next \
to a "you" or "your" or "yourself", apply this test:
- Substitute "a person" for the "you" in the original sentence.
- If the substituted sentence still makes sense as a general \
statement (advice, hypothetical, general truth), the "you" is \
GENERIC and the vocative is wrong.

Examples of GENERIC "you" patterns (the vocative is WRONG):
- "you have to / need to [verb]"
- "all you need is X"
- "when you [verb]"
- "if you [verb]"
- "you get [noun]" (general experience)
- "have faith in yourself" / "your dreams" (advice)
- "make you better" (advice)

If all vocatives are on actually-addressed "you"s, respond: CLEAN

Otherwise respond with:
GENERIC_YOU_VOCATIVES:
- "<context of the generic 'you'>" — vocative is wrong because <reason>
- ..."""


PROMPT_LOOP_C_FIX = """\
Below is a deictic-resolution rewrite with vocatives wrongly \
inserted next to generic "you" pronouns.

ORIGINAL MESSAGE: {passage}

CURRENT REWRITE: {draft}

GENERIC-"YOU" VOCATIVES to remove:
{issues}

Produce a corrected rewrite that REMOVES each wrongly-inserted \
", {addressee}," vocative from the listed generic-"you" positions. \
Keep all other parts unchanged.

Output ONLY the corrected rewrite on a single line."""


def _parse_locomo_time(s: str) -> datetime:
    return datetime.strptime(s.strip(), "%I:%M %p on %d %B, %Y")


def _build_events(conv: dict) -> dict[str, dict]:
    out = {}
    c = conv["conversation"]
    for k in c:
        if k.startswith("session_") and not k.endswith("_date_time"):
            n = k.split("_")[1]
            date_str = c.get(f"session_{n}_date_time", "")
            ts = _parse_locomo_time(date_str)
            session = c[k]
            if not isinstance(session, list):
                continue
            for turn in session:
                out[turn["dia_id"]] = {
                    "speaker": turn["speaker"],
                    "text": turn["text"],
                    "timestamp": ts,
                }
    return out


def _surrounding(dia_map, target_id, nb):
    session_prefix = target_id.split(":")[0]
    ordered = sorted(
        (k for k in dia_map if k.startswith(session_prefix + ":")),
        key=lambda x: int(x.split(":")[1]),
    )
    idx = ordered.index(target_id)
    before_ids = ordered[max(0, idx - nb):idx]
    return [(dia_map[i]["speaker"], dia_map[i]["text"]) for i in before_ids]


def _format_neighbors(before: list) -> str:
    lines = [f"- {p}: {t}" for p, t in before]
    return "\n".join(lines) + "\n" if lines else ""


def _derive_addressee(speaker: str, before: list) -> str:
    for p, _ in before:
        if p and p != speaker:
            return p
    return "the other person"


class CostTracker:
    def __init__(self):
        self.n_calls = 0
        self.in_tokens = 0
        self.out_tokens = 0

    def add(self, usage):
        self.n_calls += 1
        self.in_tokens += usage.prompt_tokens
        self.out_tokens += usage.completion_tokens

    def usd(self):
        # gpt-5.4-nano pricing (approximate)
        return self.in_tokens * 0.05 / 1_000_000 + \
               self.out_tokens * 0.40 / 1_000_000


async def _call_nano(client, prompt: str, tracker: CostTracker) -> str:
    resp = await client.chat.completions.create(
        model="gpt-5.4-nano",
        reasoning_effort="low",
        messages=[{"role": "user", "content": prompt}],
    )
    tracker.add(resp.usage)
    return resp.choices[0].message.content.strip()


async def _resolve_with_loops(
    client, source: str, speaker: str, addressee: str,
    neighbors_block: str, tracker: CostTracker, verbose: bool = False,
) -> str:
    # Initial generation
    draft = await _call_nano(client, PROMPT_V7_GENERATE.format(
        speaker=speaker, addressee=addressee, passage=source,
        neighbors_block=neighbors_block,
    ), tracker)
    if verbose:
        print(f"  [V7]: {draft}")

    # Loop A -- hallucination check (max 2 iterations)
    for _ in range(2):
        verdict = await _call_nano(client,
            PROMPT_LOOP_A_DETECT_HALLUCINATION.format(
                passage=source, neighbors_block=neighbors_block,
                draft=draft,
            ), tracker)
        if verbose:
            print(f"  [A-det]: {verdict[:120]}")
        if verdict.strip().upper().startswith("CLEAN"):
            break
        # Extract issues
        issues = verdict.split("HALLUCINATIONS:", 1)
        issues = issues[1].strip() if len(issues) > 1 else verdict
        new_draft = await _call_nano(client, PROMPT_LOOP_A_FIX.format(
            passage=source, neighbors_block=neighbors_block,
            draft=draft, issues=issues,
        ), tracker)
        if verbose:
            print(f"  [A-fix]: {new_draft}")
        if new_draft == draft:
            break
        draft = new_draft

    # Loop B -- wrong-context substitution check (max 2 iterations)
    for _ in range(2):
        verdict = await _call_nano(client,
            PROMPT_LOOP_B_DETECT_WRONG_CONTEXT.format(
                passage=source, neighbors_block=neighbors_block,
                draft=draft,
            ), tracker)
        if verbose:
            print(f"  [B-det]: {verdict[:120]}")
        if verdict.strip().upper().startswith("CLEAN"):
            break
        issues = verdict.split("WRONG_SUBSTITUTIONS:", 1)
        issues = issues[1].strip() if len(issues) > 1 else verdict
        new_draft = await _call_nano(client, PROMPT_LOOP_B_FIX.format(
            passage=source, neighbors_block=neighbors_block,
            draft=draft, issues=issues,
        ), tracker)
        if verbose:
            print(f"  [B-fix]: {new_draft}")
        if new_draft == draft:
            break
        draft = new_draft

    # Loop C -- generic-you vocative check (max 2 iterations)
    for _ in range(2):
        verdict = await _call_nano(client,
            PROMPT_LOOP_C_DETECT_GENERIC_YOU.format(
                addressee=addressee, passage=source, draft=draft,
            ), tracker)
        if verbose:
            print(f"  [C-det]: {verdict[:120]}")
        if verdict.strip().upper().startswith("CLEAN"):
            break
        issues = verdict.split("GENERIC_YOU_VOCATIVES:", 1)
        issues = issues[1].strip() if len(issues) > 1 else verdict
        new_draft = await _call_nano(client, PROMPT_LOOP_C_FIX.format(
            addressee=addressee, passage=source, draft=draft,
            issues=issues,
        ), tracker)
        if verbose:
            print(f"  [C-fix]: {new_draft}")
        if new_draft == draft:
            break
        draft = new_draft

    return draft


async def main():
    load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")
    with open(
        "/Users/eyu/edwinyyyu/mmcc/segment_store/evaluation/data/"
        "locomo10_c2sub.json"
    ) as f:
        bench = json.load(f)
    conv_data = {c["sample_id"]: c for c in bench}
    dia_maps = {cid: _build_events(c) for cid, c in conv_data.items()}

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    overall_tracker = CostTracker()

    t0 = time.time()
    for cid, dia, label in TARGETS:
        if cid not in dia_maps or dia not in dia_maps[cid]:
            continue
        msg = dia_maps[cid][dia]
        before = _surrounding(dia_maps[cid], dia, NB)
        addressee = _derive_addressee(msg["speaker"], before)
        neighbors_block = _format_neighbors(before)

        print(f"\n{'='*78}")
        print(f"[{label}]  {dia}  ({msg['speaker']} -> {addressee})")
        print(f"SRC: {msg['text']}")
        print()

        case_tracker = CostTracker()
        final = await _resolve_with_loops(
            client, msg["text"], msg["speaker"], addressee,
            neighbors_block, case_tracker, verbose=True,
        )
        print(f"  FINAL: {final}")
        print(f"  [case cost] calls={case_tracker.n_calls} "
              f"tok_in={case_tracker.in_tokens} "
              f"tok_out={case_tracker.out_tokens} "
              f"usd=${case_tracker.usd():.6f}")
        overall_tracker.n_calls += case_tracker.n_calls
        overall_tracker.in_tokens += case_tracker.in_tokens
        overall_tracker.out_tokens += case_tracker.out_tokens

    elapsed = time.time() - t0
    print(f"\n{'='*78}")
    print(f"TOTAL: {overall_tracker.n_calls} calls, "
          f"{overall_tracker.in_tokens} in + "
          f"{overall_tracker.out_tokens} out tokens, "
          f"${overall_tracker.usd():.4f} over {len(TARGETS)} cases, "
          f"{elapsed:.1f}s")
    print(f"  Per case: {overall_tracker.n_calls / len(TARGETS):.1f} calls, "
          f"${overall_tracker.usd() / len(TARGETS):.6f}")
    n_segments_c2sub = 2500
    est_pass = overall_tracker.usd() / len(TARGETS) * n_segments_c2sub
    print(f"  Projected c2sub ingest cost (2500 segs): ${est_pass:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
