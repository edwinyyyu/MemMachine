"""V7 prompt on gpt-5.4-nano @ low with BEFORE-ONLY neighbors.

The prior test_deictic_v7.out was generated with both-direction neighbors
(harness bug), which let the model grab phrases from future turns
(e.g. "Scout, Toby, Buddy" at D28:3). This rerun uses 8 prior + 0
future, matching the intended `nb8b` ingest convention.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent
        / "longmemeval/llm_pipeline_probe"),
)

from dotenv import load_dotenv
from openai import AsyncOpenAI

from memmachine_server.common.language_model.openai_chat_completions_language_model import (
    OpenAIChatCompletionsLanguageModel,
    OpenAIChatCompletionsLanguageModelParams,
)
from memmachine_server.episodic_memory.event_memory.data_types import (
    Event,
    SurroundingEvent,
    SurroundingEventsContext,
    TextBlock,
)
from probe_deictic_resolved_verbatim_v7 import DeicticResolvedV7Segmenter


NB = 8

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
    session_prefix = target_id.split(":")[0]
    ordered = sorted(
        (k for k in dia_map if k.startswith(session_prefix + ":")),
        key=lambda x: int(x.split(":")[1]),
    )
    idx = ordered.index(target_id)
    before_ids = ordered[max(0, idx - nb):idx]
    return (
        [SurroundingEvent(producer=dia_map[i]["speaker"], text=dia_map[i]["text"])
         for i in before_ids],
        [],
    )


async def run_one(seg, msg, before, after, label):
    event = Event(
        uuid=uuid4(),
        timestamp=msg["timestamp"],
        timestamp_timezone_offset=0,
        producer=msg["speaker"],
        produced_for=set(),
        context=SurroundingEventsContext(
            producer=msg["speaker"], before=before, after=after
        ),
        blocks=[TextBlock(text=msg["text"])],
        properties={},
    )
    print(f"\n{'='*78}", flush=True)
    print(f"[{label}]  {msg['dia_id']}  ({msg['speaker']})", flush=True)
    print(f"SRC: {msg['text']}", flush=True)
    out = await seg.segment(event)
    print(f"NANO: {out[0].block.text if out else '(empty)'}", flush=True)


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
    lm = OpenAIChatCompletionsLanguageModel(
        OpenAIChatCompletionsLanguageModelParams(
            client=client, model="gpt-5.4-nano", reasoning_effort="low",
        )
    )
    seg = DeicticResolvedV7Segmenter(language_model=lm)

    for cid, dia, label in TARGETS:
        if cid not in dia_maps:
            continue
        dia_map = dia_maps[cid]
        if dia not in dia_map:
            continue
        msg = dia_map[dia]
        before, after = _surrounding(dia_map, dia, NB)
        await run_one(seg, msg, before, after, label)


if __name__ == "__main__":
    asyncio.run(main())
