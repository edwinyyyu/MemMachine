"""Audit v22-fp on a small sample of real LoCoMo turns.

Pulls 12 turns from locomo10.json (mix of dated-event and message-time
content) and runs the v22-fp segmenter on each. Reports the 1p outputs
and flags rule violations (msg-date leak, forbidden date forms, bare
relatives, 3p slips for the speaker's own references).
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import sys
from datetime import datetime

import openai

from memmachine_server.common.language_model.openai_responses_language_model import (
    OpenAIResponsesLanguageModel,
    OpenAIResponsesLanguageModelParams,
)

from probe_segmenter_rewrite_v22_fp import (
    PROMPT_REWRITE_V22_FP,
    _RewriteResponse,
)


def sample_turns(n: int, seed: int = 0) -> list[dict]:
    import re as _re
    from datetime import datetime as _dt
    src = json.load(
        open(
            "/Users/eyu/edwinyyyu/mmcc/segment_store/evaluation/data/locomo10.json"
        )
    )
    turns: list[dict] = []
    for conv in src:
        conv_dict = conv.get("conversation", {})
        # session_N is a list of turns; session_N_date_time is e.g. "1:56 pm on 8 May, 2023"
        for sess_key, val in conv_dict.items():
            if not (sess_key.startswith("session_") and not sess_key.endswith("_date_time") and isinstance(val, list)):
                continue
            ts_raw = conv_dict.get(f"{sess_key}_date_time", "")
            m = _re.search(r"(\d{1,2})\s+([A-Za-z]+),\s*(\d{4})", str(ts_raw))
            if not m:
                continue
            try:
                date_iso = _dt.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%d %B %Y").strftime("%Y-%m-%d")
            except Exception:
                continue
            for turn in val:
                if not isinstance(turn, dict):
                    continue
                spk = turn.get("speaker")
                txt = turn.get("text")
                if not (spk and txt):
                    continue
                turns.append({"speaker": spk, "date": date_iso, "text": txt})
    rng = random.Random(seed)
    rich = re.compile(
        r"\b(last|next|this|past)\s+(week|month|year|summer|winter|spring|fall|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|night|weekend)\b"
        r"|\b(yesterday|today|tomorrow|tonight)\b"
        r"|\b(\d+)\s+(days|weeks|months|years)\s+ago\b",
        re.IGNORECASE,
    )
    rich_t = [t for t in turns if rich.search(t["text"])]
    plain_t = [t for t in turns if not rich.search(t["text"])]
    rng.shuffle(rich_t)
    rng.shuffle(plain_t)
    return (rich_t[: n // 2] + plain_t[: n - n // 2])[:n]


async def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    model = sys.argv[2] if len(sys.argv) > 2 else "gpt-5.4-nano"
    reasoning = sys.argv[3] if len(sys.argv) > 3 else "low"
    turns = sample_turns(n)
    print(f"== n={len(turns)} turns, model={model} reasoning={reasoning}\n")

    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    lm = OpenAIResponsesLanguageModel(
        OpenAIResponsesLanguageModelParams(
            client=client, model=model, reasoning_effort=reasoning,
        )
    )

    forbidden_patterns = [
        ("sentence-prefix `On YYYY-MM-DD,`", re.compile(r"^On \d{4}-\d{2}-\d{2}[,.]")),
        ("paren `(Date: ...)`", re.compile(r"\(Date: \d{4}-\d{2}-\d{2}\)")),
        ("paren `(Event date: ...)`", re.compile(r"\(Event date: \d{4}-\d{2}-\d{2}\)")),
        ("bracket `[YYYY-MM-DD]`", re.compile(r"\[\d{4}-\d{2}-\d{2}\]")),
        ("`as of YYYY-MM-DD`", re.compile(r"as of \d{4}-\d{2}-\d{2}")),
    ]
    bare_relative = re.compile(
        r"\b(last|next|this) (week|month|year|summer|winter|spring|fall|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|night|weekend)\b"
        r"|\b(yesterday|today|tomorrow|tonight)\b"
        r"|\b(\d+) (days|weeks|months|years) ago\b",
        re.IGNORECASE,
    )

    async def run_one(t):
        prompt = PROMPT_REWRITE_V22_FP.format(
            speaker=t["speaker"], date=t["date"],
            passage=t["text"], neighbors_block="",
        )
        resp = await lm.generate_parsed_response(
            output_format=_RewriteResponse,
            user_prompt=prompt, max_attempts=3,
        )
        return t, resp

    results = await asyncio.gather(*[run_one(t) for t in turns])

    counts = {"msg_date_leak": 0, "bare_relative": 0, "speaker_3p_slip": 0}
    forbidden = {n_: 0 for n_, _ in forbidden_patterns}
    total = 0

    for t, resp in results:
        print(f"-- {t['speaker']} on {t['date']} --")
        print(f"   {t['text'][:200]}")
        if resp is None:
            print("   [parse failure]\n")
            continue
        if not resp.memories:
            print("   -> (empty list)\n")
            continue
        for mem in resp.memories:
            total += 1
            issues: list[str] = []
            if t["date"] in mem:
                counts["msg_date_leak"] += 1
                issues.append("MSG_DATE_LEAK")
            for name, pat in forbidden_patterns:
                if pat.search(mem):
                    forbidden[name] += 1
                    issues.append(f"FORBIDDEN:{name}")
            if bare_relative.search(mem):
                counts["bare_relative"] += 1
                issues.append("BARE_RELATIVE")
            # Detect 3p slip: speaker name appearing as the subject when 1p was expected
            # Heuristic: if the speaker's name appears followed by a 3p verb (is/has/does/went/said/...)
            spk = t["speaker"]
            if re.search(rf"\b{re.escape(spk)}\b\s+(is|was|has|had|does|did|went|said|told|asked|wanted|likes|loves|recommends|suggests|plans|knows|feels|thinks)\b", mem):
                counts["speaker_3p_slip"] += 1
                issues.append("SPEAKER_3P_SLIP")
            tag = f"   !! {issues}" if issues else ""
            print(f"   -> {mem}{tag}")
        print()

    print("=== SUMMARY ===")
    print(f"  total segments: {total}")
    print(f"  msg_date_leak: {counts['msg_date_leak']}")
    print(f"  bare_relative: {counts['bare_relative']}")
    print(f"  speaker_3p_slip: {counts['speaker_3p_slip']}")
    print(f"  forbidden formats:")
    for k, v in forbidden.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    asyncio.run(main())
