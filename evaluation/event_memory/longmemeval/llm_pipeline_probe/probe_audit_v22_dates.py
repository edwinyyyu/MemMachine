"""Audit v22-dates segmenter on a sample of real LoCoMo messages.

Compares v22 vs v22-dates outputs side-by-side so we can verify:
  1. v22-dates removes the message date {date} from outputs
  2. v22-dates emits event-mentioned dates only when they differ
  3. v22-dates uses the canonical `on YYYY-MM-DD` inline form

Reuses the segments-cache to pick real LoCoMo messages with rich date
content (events mentioned that differ from message date).
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import re
import sys

import openai
from langchain_text_splitters import RecursiveCharacterTextSplitter

from memmachine_server.common.language_model.openai_responses_language_model import (
    OpenAIResponsesLanguageModel,
    OpenAIResponsesLanguageModelParams,
)

from probe_segmenter_rewrite_v22_dates import (
    PROMPT_REWRITE_V22_DATES,
    _RewriteResponse,
)


def sample_locomo_messages(n: int, seed: int = 0) -> list[dict]:
    """Pull message-level records from locomo10.json with date context.

    Each record: {speaker, date (ISO YYYY-MM-DD), text}
    """
    src = json.load(
        open(
            "/Users/eyu/edwinyyyu/mmcc/segment_store/evaluation/data/locomo10.json"
        )
    )
    rng = random.Random(seed)
    msgs: list[dict] = []
    for conv in src:
        for sess_key, turns in conv.get("conversation", {}).items():
            if not isinstance(turns, list):
                continue
            for turn in turns:
                if not isinstance(turn, dict):
                    continue
                spk = turn.get("speaker")
                txt = turn.get("text") or turn.get("clean_text")
                ts = turn.get("date_time") or turn.get("timestamp")
                if not (spk and txt and ts):
                    continue
                # Try to extract YYYY-MM-DD from timestamp
                m = re.search(r"(\d{4}-\d{2}-\d{2})", str(ts))
                if not m:
                    continue
                msgs.append({"speaker": spk, "date": m.group(1), "text": txt})
    # Bias the sample toward messages that include relative-time references
    rel_pat = re.compile(
        r"\b(last|next|this)\s+(week|month|year|summer|winter|spring|fall|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|night|weekend)\b"
        r"|\b(yesterday|today|tomorrow|tonight)\b"
        r"|\b(\d+)\s+(days|weeks|months|years)\s+ago\b"
        r"|\b(last|recent|past)\b",
        re.IGNORECASE,
    )
    rich = [m for m in msgs if rel_pat.search(m["text"])]
    plain = [m for m in msgs if m not in rich]
    rng.shuffle(rich)
    rng.shuffle(plain)
    picked = rich[: n // 2] + plain[: n - n // 2]
    return picked[:n]


async def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    model = sys.argv[2] if len(sys.argv) > 2 else "gpt-5.4-nano"
    reasoning = sys.argv[3] if len(sys.argv) > 3 else "low"
    msgs = sample_locomo_messages(n)
    print(f"== n={len(msgs)} sample messages, model={model} reasoning={reasoning}\n")

    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    lm = OpenAIResponsesLanguageModel(
        OpenAIResponsesLanguageModelParams(
            client=client, model=model, reasoning_effort=reasoning,
        )
    )

    iso_pat = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
    forbidden_patterns = [
        ("sentence-prefix `On YYYY-MM-DD,`", re.compile(r"^On \d{4}-\d{2}-\d{2}[,.]")),
        ("paren `(Date: ...)`", re.compile(r"\(Date: \d{4}-\d{2}-\d{2}\)")),
        ("paren `(Event date: ...)`", re.compile(r"\(Event date: \d{4}-\d{2}-\d{2}\)")),
        ("bracket `[YYYY-MM-DD]`", re.compile(r"\[\d{4}-\d{2}-\d{2}\]")),
        ("`as of YYYY-MM-DD`", re.compile(r"as of \d{4}-\d{2}-\d{2}")),
    ]
    relative_pat = re.compile(
        r"\b(last|next|this) (week|month|year|summer|winter|spring|fall|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|night|weekend)\b"
        r"|\b(yesterday|today|tomorrow|tonight)\b"
        r"|\b(\d+) (days|weeks|months|years) ago\b",
        re.IGNORECASE,
    )

    async def run_one(msg):
        prompt = PROMPT_REWRITE_V22_DATES.format(
            speaker=msg["speaker"], date=msg["date"],
            passage=msg["text"], neighbors_block="",
        )
        resp = await lm.generate_parsed_response(
            output_format=_RewriteResponse,
            user_prompt=prompt, max_attempts=3,
        )
        return msg, resp

    results = await asyncio.gather(*[run_one(m) for m in msgs])

    msg_date_leaks = 0
    forbidden_hits = {n_: 0 for n_, _ in forbidden_patterns}
    bare_relative = 0
    total_segments = 0
    correctly_dated = 0

    print("=== PER-MESSAGE OUTPUTS ===\n")
    for msg, resp in results:
        print(f"-- {msg['speaker']} on {msg['date']} --")
        print(f"   {msg['text'][:180]}")
        if resp is None:
            print("   [parse failure]")
            print()
            continue
        for mem in resp.memories:
            total_segments += 1
            issues: list[str] = []
            if msg["date"] in mem:
                msg_date_leaks += 1
                issues.append("MSG_DATE_LEAK")
            for name, pat in forbidden_patterns:
                if pat.search(mem):
                    forbidden_hits[name] += 1
                    issues.append(f"FORBIDDEN:{name}")
            if relative_pat.search(mem):
                bare_relative += 1
                issues.append("BARE_RELATIVE")
            for d in iso_pat.findall(mem):
                if d != msg["date"]:
                    correctly_dated += 1
            tag = f" !! {issues}" if issues else ""
            print(f"   -> {mem}{tag}")
        if not resp.memories:
            print("   -> (empty list)")
        print()

    print()
    print("=== SUMMARY ===")
    print(f"  total messages:           {len(msgs)}")
    print(f"  total segments emitted:   {total_segments}")
    print(f"  msg-date leaks:           {msg_date_leaks}")
    print(f"  forbidden formats:")
    for name, hits in forbidden_hits.items():
        print(f"    {name:>40}: {hits}")
    print(f"  bare relative phrases:    {bare_relative}")
    print(f"  event-date emits (!=msg): {correctly_dated}")


if __name__ == "__main__":
    asyncio.run(main())
