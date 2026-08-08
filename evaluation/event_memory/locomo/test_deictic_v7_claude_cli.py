"""V7 prompt via fresh Claude CLI (--bare) on 25 manual cases.

Tests whether Claude can do the deictic-resolved task that gpt-5.4-nano/
mini/full all imperfectly solved. Uses --bare to strip CLAUDE.md,
auto-memory, and hook context so the model sees only the V7 prompt +
case data. Each case runs in its own subprocess invocation (no shared
session state).

Usage:
  uv run python test_deictic_v7_claude_cli.py > test_deictic_v7_claude.out
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent
        / "longmemeval/llm_pipeline_probe"),
)

from probe_deictic_resolved_verbatim_v7 import PROMPT_DEICTIC_RESOLVED_V7


NB = 8
CLAUDE = "claude"
# Run claude from a clean project path so the user-level auto-memory
# (keyed off project path) does not match -- avoids leaking LoCoMo /
# xpurp2 / VERBATIM context from the dev project memory directory.
CLEAN_CWD = "/tmp/fresh-test"

# Identical to the 25 cases used for gpt-5.4-nano/mini/full comparisons.
TARGETS = [
    ("conv-42", "D28:32", "you addressee"),
    ("conv-47", "D30:18", "generic you (gamepad)"),
    ("conv-42", "D9:14", "this series"),
    ("conv-50", "D30:14", "that = photo"),
    ("conv-42", "D6:3", "Are you excited?"),
    ("conv-42", "D14:14", "they = the guys at tournament"),
    ("conv-50", "D28:10", "this car = my car"),
    ("conv-44", "D18:6", "Glad you + They (dogs)"),
    ("conv-50", "D2:6", "You Calvin x3"),
    ("conv-42", "D29:6", "generic you (moments of joy)"),
    ("conv-42", "D26:19", "vocative Nate at start"),
    ("conv-42", "D11:16", "we + don't you think"),
    ("conv-44", "D19:29", "them snuggled + They + our lives"),
    ("conv-44", "D20:5", "they + That spot + them + you"),
    ("conv-44", "D25:5", "Have you ever tried it"),
    ("conv-44", "D18:5", "no deictic"),
    ("conv-42", "D6:10", "advice generic-you yourself"),
    ("conv-44", "D20:2", "Hey Andrew + Bet you + my face"),
    ("conv-50", "D27:5", "that view + Have you taken pictures"),
    ("conv-42", "D18:11", "Thanks for sharing your love"),
    ("conv-44", "D23:22", "Do you ever + escape for you"),
    ("conv-50", "D28:20", "I found it last week"),
    ("conv-44", "D28:3", "pic of them + those coats"),
    ("conv-50", "D27:1", "You? Anything new"),
    ("conv-50", "D8:5", "That studio + Boston parks"),
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
    """Before-only (`nb` prior turns, 0 future). Matches the historical
    `nb8b` ingest convention. Production default in locomo_ingest.py is
    `--neighbor-direction both` but the c2sub ingests should be using
    `before` to avoid temporal leakage from future turns."""
    session_prefix = target_id.split(":")[0]
    ordered = sorted(
        (k for k in dia_map if k.startswith(session_prefix + ":")),
        key=lambda x: int(x.split(":")[1]),
    )
    idx = ordered.index(target_id)
    before_ids = ordered[max(0, idx - nb):idx]
    return (
        [(dia_map[i]["speaker"], dia_map[i]["text"]) for i in before_ids],
        [],
    )


def _format_neighbors(before, after) -> str:
    lines: list[str] = []
    if before:
        lines.append("PRIOR TURNS (context only):")
        for producer, text in before:
            lines.append(f"- {producer}: {text}")
        lines.append("")
    if after:
        lines.append("LATER TURNS (context only):")
        for producer, text in after:
            lines.append(f"- {producer}: {text}")
        lines.append("")
    return "\n".join(lines) + ("\n" if lines else "")


def _derive_addressee(speaker: str, before, after) -> str:
    for producer, _ in list(before) + list(after):
        if producer and producer != speaker:
            return producer
    return "the other person"


async def _run_claude(prompt: str) -> str:
    proc = await asyncio.create_subprocess_exec(
        CLAUDE, "-p", prompt,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ},
        cwd=CLEAN_CWD,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=180)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return "(TIMEOUT)"
    if proc.returncode != 0:
        return f"(EXIT {proc.returncode}: {stderr.decode()[:200]})"
    return stdout.decode().strip()


async def process_one(msg, before, after, label):
    speaker = msg["speaker"]
    addressee = _derive_addressee(speaker, before, after)
    neighbors_block = _format_neighbors(before, after)
    prompt = PROMPT_DEICTIC_RESOLVED_V7.format(
        speaker=speaker, addressee=addressee, passage=msg["text"],
        neighbors_block=neighbors_block,
    )
    print(f"\n{'='*78}", flush=True)
    print(f"[{label}]  {msg['dia_id']}  ({speaker} -> {addressee})", flush=True)
    print(f"SRC: {msg['text']}", flush=True)
    output = await _run_claude(prompt)
    print(f"CLAUDE: {output}", flush=True)


async def main():
    bench_path = (
        "/Users/eyu/edwinyyyu/mmcc/segment_store/evaluation/data/"
        "locomo10_c2sub.json"
    )
    with open(bench_path) as f:
        bench = json.load(f)
    conv_data = {c["sample_id"]: c for c in bench}
    dia_maps = {cid: _build_events_for_conv(c) for cid, c in conv_data.items()}

    # Run sequentially -- Claude CLI may not parallelize cleanly.
    for cid, dia, label in TARGETS:
        if cid not in dia_maps:
            continue
        dia_map = dia_maps[cid]
        if dia not in dia_map:
            continue
        msg = dia_map[dia]
        before, after = _surrounding(dia_map, dia, NB)
        await process_one(msg, before, after, label)


if __name__ == "__main__":
    asyncio.run(main())
