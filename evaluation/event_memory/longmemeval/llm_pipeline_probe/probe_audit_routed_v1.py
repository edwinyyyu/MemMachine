"""Audit routed-v1 deriver outputs on real LoCoMo segments.

Samples N segments from a segments-cache JSON, runs the routed-v1
deriver, and prints:
  - segment text
  - routed shapes emitted
  - any rule violations detected (lowercase tags, count caps, etc.)

Use to validate instruction adherence per content type before judging
recall results.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import sys
from pathlib import Path

import openai

from memmachine_server.common.language_model.openai_responses_language_model import (
    OpenAIResponsesLanguageModel,
    OpenAIResponsesLanguageModelParams,
)
from memmachine_server.episodic_memory.event_memory.data_types import (
    decode_block as _decode_block_fn,
    decode_context as _decode_context_fn,
    TextBlock,
)

from probe_deriver_routed_v1 import (
    GenericDeriver,
    PROMPT_ROUTED_V1,
    _RoutedResponse,
)


def load_segment_texts(cache_path: str, n: int, seed: int = 0) -> list[str]:
    with open(cache_path) as f:
        records = json.load(f)
    rng = random.Random(seed)
    sampled = rng.sample(records, min(n, len(records)))
    texts: list[str] = []
    for r in sampled:
        blk = _decode_block_fn(
            json.loads(base64.b64decode(r["block_blob"]).decode("utf-8"))
        )
        if isinstance(blk, TextBlock):
            texts.append(blk.text)
    return texts


def violations(seg_text: str, response: _RoutedResponse) -> list[str]:
    issues: list[str] = []
    shapes_emitted = [d.shape for d in response.derivatives]

    # Duplicate shape
    if len(shapes_emitted) != len(set(shapes_emitted)):
        issues.append(f"DUP_SHAPE: {shapes_emitted}")

    for d in response.derivatives:
        if d.shape == "tag_suffix":
            if not d.values:
                issues.append("tag_suffix empty values")
            for tag in d.values:
                if tag != tag.lower():
                    issues.append(f"tag NOT lowercase: {tag!r}")
                wc = len(tag.split())
                if wc > 3:
                    issues.append(f"tag > 3 words: {tag!r}")
                if wc == 0:
                    issues.append(f"tag empty")
            if len(d.values) > 6:
                issues.append(f"tag_suffix > 6 values: {len(d.values)}")
        elif d.shape == "multi_axis":
            if len(d.values) > 2:
                issues.append(f"multi_axis > 2 values: {len(d.values)}")
            if not d.values:
                issues.append("multi_axis empty values")
            for axis in d.values:
                if len(axis.split()) > 12:
                    issues.append(f"axis > 12 words: {axis!r}")
        elif d.shape == "list_extraction":
            if len(d.values) != 1:
                issues.append(f"list_extraction values != 1: {len(d.values)}")
        elif d.shape == "none":
            if d.values:
                issues.append(f"none with values: {d.values}")

    return issues


async def main() -> None:
    cache_path = sys.argv[1] if len(sys.argv) > 1 else (
        "/Users/eyu/edwinyyyu/mmcc/segment_store/evaluation/event_memory/"
        "locomo/segments-cache-v22-nb8b-g4.json"
    )
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    model = sys.argv[3] if len(sys.argv) > 3 else "gpt-5-nano"
    reasoning = sys.argv[4] if len(sys.argv) > 4 else "low"

    seg_texts = load_segment_texts(cache_path, n=n)
    print(f"== {len(seg_texts)} segments from {Path(cache_path).name}")
    print(f"== model={model} reasoning={reasoning}\n")

    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    lm = OpenAIResponsesLanguageModel(
        OpenAIResponsesLanguageModelParams(
            client=client, model=model, reasoning_effort=reasoning,
        )
    )

    counts = {"tag_suffix": 0, "multi_axis": 0, "list_extraction": 0, "none": 0, "empty": 0}
    all_issues: list[str] = []
    sample_outputs: list[tuple[str, list[dict], list[str]]] = []

    async def run_one(text: str):
        prompt = PROMPT_ROUTED_V1.format(segment_text=text)
        resp = await lm.generate_parsed_response(
            output_format=_RoutedResponse,
            user_prompt=prompt,
            max_attempts=3,
        )
        return text, resp

    results = await asyncio.gather(*[run_one(t) for t in seg_texts])

    for text, resp in results:
        if resp is None:
            all_issues.append("PARSE_FAILURE")
            sample_outputs.append((text, [], ["PARSE_FAILURE"]))
            continue
        if not resp.derivatives:
            counts["empty"] += 1
            shapes_this = ["(empty-array=NONE)"]
        else:
            shapes_this = []
            for d in resp.derivatives:
                counts[d.shape] = counts.get(d.shape, 0) + 1
                shapes_this.append(f"{d.shape}={d.values}")
        iss = violations(text, resp)
        all_issues.extend(iss)
        sample_outputs.append((text, [d.model_dump() for d in resp.derivatives], iss))

    print("=== SHAPE COUNTS ===")
    for k, v in counts.items():
        pct = 100 * v / max(len(seg_texts), 1)
        print(f"  {k:>18}: {v:>3}  ({pct:.1f}%)")
    print()
    print("=== ISSUES ===")
    if not all_issues:
        print("  none")
    else:
        from collections import Counter
        for issue, count in Counter(all_issues).most_common():
            print(f"  {count:>3}x  {issue}")
    print()
    print("=== SAMPLE OUTPUTS (first 12) ===")
    for text, shapes, iss in sample_outputs[:12]:
        print(f"SEG: {text[:200]}")
        print(f"  -> {shapes}")
        if iss:
            print(f"  !! {iss}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
