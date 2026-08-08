"""Quick test of deictic-resolved v2 prompt on 10 failure cases.

Pulls neighbor context from c2sub source, runs the v2 segmenter only,
prints (source, v2 output) so you can eyeball whether v2 fixes the
substitution failures and hallucinations seen in v1.

Usage:
  uv run python test_deictic_v2.py
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
from probe_deictic_resolved_verbatim_v2 import DeicticResolvedV2Segmenter
from probe_deictic_resolved_verbatim_v3 import DeicticResolvedV3Segmenter
from probe_deictic_resolved_verbatim_v4 import DeicticResolvedV4Segmenter
from probe_deictic_resolved_verbatim_v5 import DeicticResolvedV5Segmenter
from probe_deictic_resolved_verbatim_v6 import DeicticResolvedV6Segmenter
from probe_deictic_resolved_verbatim_v7 import DeicticResolvedV7Segmenter
from probe_deictic_resolved_verbatim_multipass import (
    DeicticResolvedMultipassSegmenter,
)
from probe_deictic_resolved_verbatim import DeicticResolvedSegmenter as V1


NB = 8


# Initial 10 = failure modes; +15 = stress test of v3 on broader cases
TARGETS = [
    ("conv-42", "D28:32", "you → addressee (Nate)"),
    ("conv-47", "D30:18", "generic you (must KEEP, not sub speaker name)"),
    ("conv-42", "D9:14", "this series → fantasy book series"),
    ("conv-50", "D30:14", "that → the photo"),
    ("conv-42", "D6:3", "you → Joanna; vocative insert"),
    ("conv-42", "D14:14", "they → the guys at the tournament"),
    ("conv-50", "D28:10", "this car → Dave's Subaru (NOT Calvin's)"),
    ("conv-44", "D18:6", "neighbors' dogs (no sub) + your → Andrew's"),
    ("conv-50", "D2:6", "vocative Calvin + You → Calvin"),
    ("conv-42", "D29:6", "you generic + my turtles (keep)"),
    # Stress: multi-pronoun + temporal
    ("conv-42", "D26:19", "thanks/come over; that = coming over"),
    ("conv-42", "D11:16", "Hey + we + don't you + I'll"),
    ("conv-44", "D19:29", "them snuggled up + They bring joy"),
    ("conv-44", "D20:5", "That spot + them to play + you take them"),
    ("conv-44", "D25:5", "tried it (sushi from neighbors)"),
    # Stress: 3p referring to speaker self
    ("conv-44", "D18:5", "their words/they = letter writer (3p other)"),
    # Stress: no deictic at all
    ("conv-42", "D6:10", "advice text, you generic in 'will make you better'"),
    # Stress: vocative-at-start + you
    ("conv-44", "D20:2", "Hey Andrew + have fun at beach trip + Bet you"),
    # Stress: multiple potential antecedents in neighbors
    ("conv-50", "D27:5", "that view + you take pictures"),
    # Stress: deep filler
    ("conv-42", "D18:11", "thanks for sharing + your love for desserts"),
    # Stress: 3p = speaker themself
    ("conv-44", "D23:22", "They bring me joy + Do you ever visit + for you"),
    # Stress: 'it' object reference across turns
    ("conv-50", "D28:20", "I found it last week (= the car)"),
    # Stress: photo/attached media descriptions
    ("conv-44", "D28:3", "Here's a pic of them + those coats + them in salon"),
    # Stress: question + temporal
    ("conv-50", "D27:1", "How's it going? + I had a blast last week"),
    # Stress: this place
    ("conv-50", "D8:5", "That studio + Boston parks"),
]


def _parse_locomo_time(date_time_str: str) -> datetime:
    return datetime.strptime(date_time_str.strip(), "%I:%M %p on %d %B, %Y")


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
    `nb8b` ingest convention."""
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


async def run_one(seg_v7, seg_mp, msg, before, after, label):
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
    print(f"\n{'='*78}")
    print(f"[{label}]  {msg['dia_id']}  ({msg['speaker']})")
    print(f"SRC: {msg['text']}")
    print()
    v7_out = await seg_v7.segment(event)
    print(f"V7: {v7_out[0].block.text if v7_out else '(empty)'}")
    mp_out = await seg_mp.segment(event)
    print(f"MP: {mp_out[0].block.text if mp_out else '(empty)'}")


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
    seg_v7 = DeicticResolvedV7Segmenter(language_model=lm)
    seg_mp = DeicticResolvedMultipassSegmenter(language_model=lm)

    for cid, dia, label in TARGETS:
        if cid not in dia_maps:
            print(f"skip {cid}/{dia}: conv not in c2sub", file=sys.stderr)
            continue
        dia_map = dia_maps[cid]
        if dia not in dia_map:
            print(f"skip {cid}/{dia}: dia_id not found", file=sys.stderr)
            continue
        msg = dia_map[dia]
        before, after = _surrounding(dia_map, dia, NB)
        await run_one(seg_v7, seg_mp, msg, before, after, label)


if __name__ == "__main__":
    asyncio.run(main())
