"""Dialog-style prompting for deictic resolution.

V7-V8 stuffed all rules + examples into a single user prompt. This
variant uses proper chat-completions multi-message structure: system
prompt, then user/assistant pairs as few-shot demonstrations, then the
final user turn with the actual case.

The hypothesis: chat models interpret prior 'assistant' turns as
their own past outputs and continue the pattern. Demonstrating
careful (verbatim-preserving, conservative-when-uncertain) behavior
as past 'assistant' outputs primes the model toward that behavior.

Few-shot examples use Alice/Bob/Charlie/Dana/Eve/Frank/George/Hannah
and non-c2sub domains (chess club, standing desk, charcoal grill,
coastal trail). None of these appear in c2sub conversations.

Tested directly against gpt-5.4-nano @ low chat-completions on a
diverse 8-case subset.
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


# Diverse 8-case subset covering distinct failure modes
TARGETS = [
    ("conv-42", "D6:3",   "addressed-you vocative"),
    ("conv-47", "D30:18", "generic-you (gamepad)"),
    ("conv-50", "D28:10", "demonstrative + ownership (car)"),
    ("conv-44", "D28:3",  "3p antecedent in prior (fur kids)"),
    ("conv-50", "D27:5",  "bare 'it' no antecedent (city)"),
    ("conv-42", "D6:10",  "generic-you in advice (yourself)"),
    ("conv-44", "D20:2",  "vocative + 'my face' verbatim"),
    ("conv-50", "D28:20", "3x bare 'it' (found car)"),
]
NB = 8


SYSTEM_PROMPT = """\
You rewrite a chat message so a reader who has not seen the prior \
turns can understand every reference. Apply these rules:

1. ADDRESSED "you" / "your" / "yours" / "yourself" (a claim about \
the addressee or a question to them): keep the pronoun AND add \
", {addressee_placeholder}, " as a vocative immediately after it. \
Examples: "you" -> "you, {addressee_placeholder},"; "your X" -> \
"your, {addressee_placeholder}'s, X".

2. GENERIC "you" (means "anyone" / "a person"; typical in advice \
or general statements): keep unchanged. Test: if "a person" \
substitutes for "you" naturally, it is generic.

3. THIRD-PERSON pronouns (he/she/they/him/her/them/his/hers/theirs) \
referring to a person or thing named in the PRIOR TURNS: replace \
with the name or noun phrase the prior turns established. The \
substituted phrase must appear word-for-word in the PRIOR TURNS and \
must be semantically appropriate as the antecedent.

4. DEMONSTRATIVES (this/that/these/those/here/there) or bare "it" \
referring to a specific referent the PRIOR TURNS introduced: replace \
with the noun phrase the prior turns established, word-for-word, \
semantically appropriate. If the prior turns establish the speaker \
owns / made / built / restored / planted the referent, use "my X" \
instead.

5. If a 3p pronoun, demonstrative, or bare "it" has no clear or \
semantically-appropriate antecedent in the PRIOR TURNS, KEEP it \
unchanged. Never invent. Never substitute a noun phrase that \
matches literally but is semantically wrong.

6. NEVER write the speaker's own name as a self-reference. If a 3p \
pronoun refers to the speaker themselves, rewrite to first person \
"I" / "my" / "me".

7. Keep everything else verbatim: wording, punctuation, \
capitalization, emoji, attached-media descriptions. Do not \
paraphrase. Do not drop sentences. Do not add commentary.

Output ONLY the rewritten message on a single line. No reasoning, \
no preamble, no JSON, no quotes.

In each turn the SPEAKER and ADDRESSEE are named. Substitute the \
actual addressee name for {addressee_placeholder} in your output."""


# Few-shot turns -- non-c2sub names and domains.
# Each: (user content, assistant content)
FEW_SHOT = [
    # Example 1: addressed-you + demonstrative ownership (grill is Bob's,
    # but speaker is Alice asking about Bob's grill, so it's "the
    # charcoal grill" not "my X")
    (
        "SPEAKER: Alice\n"
        "ADDRESSEE: Bob\n\n"
        "PRIOR TURNS:\n"
        "- Bob: I bought a charcoal grill last Saturday. It's been "
        "great for weekend cookouts.\n"
        "- Alice: That sounds awesome!\n\n"
        "MESSAGE: Did you season it before the first cook?",
        "Did you, Bob, season the charcoal grill before the first "
        "cook?"
    ),
    # Example 2: wrong-context keep -- "It's quiet" refers to vibe
    # (which Dana asked about, never named as noun in prior turns),
    # NOT to the "beautiful old building" that appears literally
    (
        "SPEAKER: Charlie\n"
        "ADDRESSEE: Dana\n\n"
        "PRIOR TURNS:\n"
        "- Charlie: I joined a chess club last month. Beautiful old "
        "building downtown.\n"
        "- Dana: Cool, what's the vibe like there?\n\n"
        "MESSAGE: It's quiet. People play seriously. You should come "
        "sometime.",
        "It's quiet. People play seriously. You, Dana, should come "
        "sometime."
    ),
    # Example 3: generic-you (advice context) + speaker's own first
    # person stays. "you have to ease into it" is generic
    (
        "SPEAKER: Eve\n"
        "ADDRESSEE: Frank\n\n"
        "PRIOR TURNS:\n"
        "- Eve: I started using a standing desk last week.\n"
        "- Frank: Nice! Has it helped with focus?\n\n"
        "MESSAGE: Definitely. You have to ease into it though - my "
        "legs were sore for days.",
        "Definitely. You have to ease into it though - my legs were "
        "sore for days."
    ),
    # Example 4: demonstrative with no word-for-word antecedent in
    # prior turns -- prior says "where was that?" without naming.
    # The speaker provides the answer; "That" stays unchanged
    # because there's no prior noun phrase to substitute it with
    (
        "SPEAKER: George\n"
        "ADDRESSEE: Hannah\n\n"
        "PRIOR TURNS:\n"
        "- George: Hannah, did you see the photo I sent yesterday?\n"
        "- Hannah: Oh yes! Where was that?\n\n"
        "MESSAGE: That was at the coastal trail near my cabin. I "
        "shot it last weekend.",
        "That was at the coastal trail near my cabin. I shot it "
        "last weekend."
    ),
]


def _parse_locomo_time(s: str) -> datetime:
    return datetime.strptime(s.strip(), "%I:%M %p on %d %B, %Y")


def _build_events_for_conv(conv: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
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
                    "dia_id": turn["dia_id"],
                }
    return out


def _surrounding(dia_map, target_id, nb):
    """Before-only (matches the fixed locomo_ingest default)."""
    session_prefix = target_id.split(":")[0]
    ordered = sorted(
        (k for k in dia_map if k.startswith(session_prefix + ":")),
        key=lambda x: int(x.split(":")[1]),
    )
    idx = ordered.index(target_id)
    before_ids = ordered[max(0, idx - nb):idx]
    return [(dia_map[i]["speaker"], dia_map[i]["text"]) for i in before_ids]


def _derive_addressee(speaker: str, before: list) -> str:
    for producer, _ in before:
        if producer and producer != speaker:
            return producer
    return "the other person"


def _format_user_turn(speaker: str, addressee: str, before: list,
                     message: str) -> str:
    prior_lines = "\n".join(f"- {p}: {t}" for p, t in before)
    return (
        f"SPEAKER: {speaker}\n"
        f"ADDRESSEE: {addressee}\n\n"
        f"PRIOR TURNS:\n{prior_lines}\n\n"
        f"MESSAGE: {message}"
    )


def _build_messages(speaker: str, addressee: str, before: list,
                   message: str) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_content, asst_content in FEW_SHOT:
        msgs.append({"role": "user", "content": user_content})
        msgs.append({"role": "assistant", "content": asst_content})
    msgs.append({
        "role": "user",
        "content": _format_user_turn(speaker, addressee, before, message),
    })
    return msgs


async def _call_nano(client: AsyncOpenAI, messages: list[dict]) -> str:
    resp = await client.chat.completions.create(
        model="gpt-5.4-nano",
        reasoning_effort="low",
        messages=messages,
    )
    return resp.choices[0].message.content.strip()


async def main():
    load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")
    with open(
        "/Users/eyu/edwinyyyu/mmcc/segment_store/evaluation/data/"
        "locomo10_c2sub.json"
    ) as f:
        bench = json.load(f)
    conv_data = {c["sample_id"]: c for c in bench}
    dia_maps = {cid: _build_events_for_conv(c) for cid, c in conv_data.items()}

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    for cid, dia, label in TARGETS:
        if cid not in dia_maps:
            continue
        dia_map = dia_maps[cid]
        if dia not in dia_map:
            continue
        msg = dia_map[dia]
        before = _surrounding(dia_map, dia, NB)
        addressee = _derive_addressee(msg["speaker"], before)
        messages = _build_messages(
            msg["speaker"], addressee, before, msg["text"]
        )
        print(f"\n{'='*78}", flush=True)
        print(f"[{label}]  {dia}  ({msg['speaker']} -> {addressee})",
              flush=True)
        print(f"SRC: {msg['text']}", flush=True)
        out = await _call_nano(client, messages)
        print(f"DIALOG: {out}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
