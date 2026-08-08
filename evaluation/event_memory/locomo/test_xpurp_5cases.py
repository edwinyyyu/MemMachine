"""Quick test of SIDECAR xpurp deriver on 5 failing C-bucket cases.

For each (conv, dia_id) gold message: build SurroundingEventsContext
with ~8 prior + 8 later turns from the same session, run both the
original SIDECAR prompt and the xpurp prompt, print outputs.

Usage:
  uv run python test_xpurp_5cases.py
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
from probe_terse_decoupled_slim_v3_sidecar import (
    SidecarSegmenter as BaseSidecar,
)
from probe_terse_decoupled_slim_v3_sidecar_xpurp2 import (
    SidecarSegmenter as Xpurp2Sidecar,
)
from probe_terse_decoupled_slim_v3_sidecar_sc import (
    SidecarSegmenter as ScSidecar,
)


NB = 8


# Target (conv, dia_id) pairs from the C-bucket deriver-failure cases
TARGETS = [
    ("conv-42", "D26:19", "q78 plan to share recipes"),
    ("conv-42", "D28:32", "q16 plan to watch turtles"),
    ("conv-50", "D28:20", "q73 found car for blog"),
    ("conv-47", "D30:18", "q35 FIFA 23 advice"),
    ("conv-42", "D11:16", "q36 drama/screenplay after hike"),
]


def _parse_locomo_time(date_time_str: str) -> datetime:
    """LoCoMo format: '5:54 pm on 9 November, 2022'."""
    # robust to slight variation
    s = date_time_str.replace(" ", " ").strip()
    return datetime.strptime(s, "%I:%M %p on %d %B, %Y")


def _build_events_for_conv(conv: dict) -> dict[str, dict]:
    """Map dia_id -> {speaker, text, timestamp} for one conv."""
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


def _surrounding_events(
    dia_map: dict[str, dict], target_id: str, nb: int
) -> tuple[list, list]:
    """Pull ~nb prior + nb later events from the same session."""
    session_prefix = target_id.split(":")[0]  # e.g. "D26"
    ordered_ids = sorted(
        (k for k in dia_map if k.startswith(session_prefix + ":")),
        key=lambda x: int(x.split(":")[1]),
    )
    idx = ordered_ids.index(target_id)
    before_ids = ordered_ids[max(0, idx - nb):idx]
    after_ids = ordered_ids[idx + 1: idx + 1 + nb]

    def to_evt(m):
        return SurroundingEvent(producer=m["speaker"], text=m["text"])

    before = [to_evt(dia_map[i]) for i in before_ids]
    after = [to_evt(dia_map[i]) for i in after_ids]
    return before, after


async def _seg_print(label, seg, event):
    out = await seg.segment(event)
    print(f"\n--- {label} ---")
    if not out:
        print("  (no items)")
        return
    for s in out:
        ctx = s.context
        embed = ctx.text_to_embed if hasattr(ctx, "text_to_embed") else ""
        print(f"  block.text: {s.block.text!r}")
        print(f"  embed: {embed[:500]!r}")


async def run_one(seg_orig, seg_xp2, seg_sc, msg, before, after, label):
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
    print(f"SRC: {msg['text']!r}")
    await _seg_print("ORIGINAL", seg_orig, event)
    await _seg_print("XPURP2 (strengthened narrow)", seg_xp2, event)
    await _seg_print("SC (generalized self-containment)", seg_sc, event)


async def main():
    load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")
    bench_path = (
        "/Users/eyu/edwinyyyu/mmcc/segment_store/evaluation/data/"
        "locomo10_c2sub.json"
    )
    with open(bench_path) as f:
        bench = json.load(f)
    conv_data = {c["sample_id"]: c for c in bench}
    conv_dia_maps = {
        cid: _build_events_for_conv(c) for cid, c in conv_data.items()
    }

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    lm = OpenAIChatCompletionsLanguageModel(
        OpenAIChatCompletionsLanguageModelParams(
            client=client, model="gpt-5.4-nano", reasoning_effort="low",
        )
    )
    seg_orig = BaseSidecar(language_model=lm)
    seg_xp2 = Xpurp2Sidecar(language_model=lm)
    seg_sc = ScSidecar(language_model=lm)

    for cid, dia_id, label in TARGETS:
        dia_map = conv_dia_maps[cid]
        msg = dia_map[dia_id]
        before, after = _surrounding_events(dia_map, dia_id, NB)
        await run_one(seg_orig, seg_xp2, seg_sc, msg, before, after, label)


if __name__ == "__main__":
    asyncio.run(main())
