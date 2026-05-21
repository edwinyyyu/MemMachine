"""Audit + compare multiple v22-fp prompt variants on the same turn sample.

Runs N turns through each named variant module and reports compliance
metrics side-by-side (msg_date_leak, bare_relative, forbidden_format,
speaker_3p_slip, segments-per-turn, parse failures, soft-relative).
"""

from __future__ import annotations

import asyncio
import importlib
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


def sample_turns(n: int, seed: int = 0) -> list[dict]:
    src = json.load(
        open(
            "/Users/eyu/edwinyyyu/mmcc/segment_store/evaluation/data/locomo10.json"
        )
    )
    turns: list[dict] = []
    for conv in src:
        cd = conv.get("conversation", {})
        for sess_key, val in cd.items():
            if not (sess_key.startswith("session_") and not sess_key.endswith("_date_time") and isinstance(val, list)):
                continue
            ts_raw = cd.get(f"{sess_key}_date_time", "")
            m = re.search(r"(\d{1,2})\s+([A-Za-z]+),\s*(\d{4})", str(ts_raw))
            if not m:
                continue
            try:
                date_iso = datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%d %B %Y").strftime("%Y-%m-%d")
            except Exception:
                continue
            for turn in val:
                if not isinstance(turn, dict):
                    continue
                spk, txt = turn.get("speaker"), turn.get("text")
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


FORBIDDEN = [
    ("sentence_prefix", re.compile(r"(?m)^On \d{4}-\d{2}-\d{2}[,.]")),
    ("paren_Date", re.compile(r"\(Date: \d{4}-\d{2}-\d{2}\)")),
    ("paren_EventDate", re.compile(r"\(Event date: \d{4}-\d{2}-\d{2}\)")),
    ("bracket", re.compile(r"\[\d{4}-\d{2}-\d{2}\]")),
    ("as_of", re.compile(r"as of \d{4}-\d{2}-\d{2}")),
]
BARE_RELATIVE = re.compile(
    r"\b(last|next|this) (week|month|year|summer|winter|spring|fall|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|night|weekend)\b"
    r"|\b(yesterday|today|tomorrow|tonight)\b"
    r"|\b(\d+) (days|weeks|months|years) ago\b",
    re.IGNORECASE,
)
SOFT_RELATIVE = re.compile(
    r"\b(recently|lately|nowadays|currently|these days|the other day|a while back|a while ago|a couple (?:of )?(?:days|weeks|months|years))\b",
    re.IGNORECASE,
)


async def run_variant(turns: list[dict], module_name: str, prompt_attr: str, response_attr: str, lm) -> dict:
    mod = importlib.import_module(module_name)
    prompt_template = getattr(mod, prompt_attr)
    response_cls = getattr(mod, response_attr)

    async def one(t):
        try:
            prompt = prompt_template.format(
                speaker=t["speaker"], date=t["date"],
                passage=t["text"], neighbors_block="",
            )
        except KeyError:
            # Some variants might use slightly different fields; just skip
            return t, None
        resp = await lm.generate_parsed_response(
            output_format=response_cls, user_prompt=prompt, max_attempts=3,
        )
        return t, resp

    results = await asyncio.gather(*[one(t) for t in turns])

    metrics = {
        "total_segments": 0,
        "parse_failures": 0,
        "msg_date_leak": 0,
        "bare_relative": 0,
        "soft_relative": 0,
        "speaker_3p_slip": 0,
        "forbidden": {n: 0 for n, _ in FORBIDDEN},
        "segs_per_turn": [],
    }
    for t, resp in results:
        if resp is None:
            metrics["parse_failures"] += 1
            metrics["segs_per_turn"].append(0)
            continue
        mems = getattr(resp, "memories", None) or []
        metrics["segs_per_turn"].append(len(mems))
        spk = re.escape(t["speaker"])
        speaker_3p_pat = re.compile(rf"\b{spk}\b\s+(is|was|has|had|does|did|went|said|told|asked|wants|likes|loves|recommends|suggests|plans|knows|feels|thinks)\b")
        for m in mems:
            metrics["total_segments"] += 1
            if t["date"] in m:
                metrics["msg_date_leak"] += 1
            if BARE_RELATIVE.search(m):
                metrics["bare_relative"] += 1
            if SOFT_RELATIVE.search(m):
                metrics["soft_relative"] += 1
            if speaker_3p_pat.search(m):
                metrics["speaker_3p_slip"] += 1
            for name, pat in FORBIDDEN:
                if pat.search(m):
                    metrics["forbidden"][name] += 1
    return metrics


async def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    model = sys.argv[2] if len(sys.argv) > 2 else "gpt-5.4-nano"
    reasoning = sys.argv[3] if len(sys.argv) > 3 else "low"

    variants = [
        ("v22-fp v1", "probe_segmenter_rewrite_v22_fp", "PROMPT_REWRITE_V22_FP", "_RewriteResponse"),
        ("v22-fp rulesfirst", "probe_segmenter_rewrite_v22_fp_rulesfirst", "PROMPT_REWRITE_V22_FP_RULESFIRST", "_RewriteResponse"),
        ("v22-fp cot", "probe_segmenter_rewrite_v22_fp_cot", "PROMPT_REWRITE_V22_FP_COT", "_RewriteResponse"),
        ("v22-fp min", "probe_segmenter_rewrite_v22_fp_min", "PROMPT_REWRITE_V22_FP_MIN", "_RewriteResponse"),
    ]

    turns = sample_turns(n)
    print(f"== n={len(turns)} LoCoMo turns, model={model} reasoning={reasoning}\n")

    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    lm = OpenAIResponsesLanguageModel(
        OpenAIResponsesLanguageModelParams(
            client=client, model=model, reasoning_effort=reasoning,
        )
    )

    all_metrics: dict[str, dict] = {}
    for label, mod, pa, ra in variants:
        try:
            m = await run_variant(turns, mod, pa, ra, lm)
            all_metrics[label] = m
        except Exception as e:
            print(f"ERROR running {label}: {e}")
            all_metrics[label] = None

    print(f'{"variant":<22} {"segs":>5} {"parses_fail":>11} {"msg_date":>9} {"bare_rel":>9} {"soft_rel":>9} {"spk_3p":>7} {"fmt":>5} {"segs/turn":>9}')
    for label, _, _, _ in variants:
        m = all_metrics.get(label)
        if m is None:
            print(f'{label:<22} ERROR')
            continue
        avg = sum(m["segs_per_turn"]) / max(len(m["segs_per_turn"]), 1)
        fmt_total = sum(m["forbidden"].values())
        print(f'{label:<22} {m["total_segments"]:>5} {m["parse_failures"]:>11} {m["msg_date_leak"]:>9} {m["bare_relative"]:>9} {m["soft_relative"]:>9} {m["speaker_3p_slip"]:>7} {fmt_total:>5} {avg:>9.2f}')


if __name__ == "__main__":
    asyncio.run(main())
