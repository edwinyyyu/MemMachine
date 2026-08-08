"""Smoke probe: does gpt-5 work as the mem0 BEAM judge under the hardcoded
max_completion_tokens=4096? Measures reasoning-token headroom before a full run.
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv
from llm_provider import PROVIDER_OPENAI, make_chat_client
from mem0.beam_evaluate import (
    BEAM_JUDGE_SYSTEM_PROMPT,
    _JSON_OBJECT_FORMAT,
    _MAX_TOKENS,
    _build_judge_user_prompt,
    _parse_json_loose,
)

MODEL = "gpt-5"
SAMPLE_CATEGORIES = [
    "abstention",
    "temporal_reasoning",
    "multi_session_reasoning",
    "summarization",
    "event_ordering",
]


async def main():
    load_dotenv()
    data = json.load(open("10m-out/mem0_10m-v200-e5-l200-r0.json"))
    jobs = []
    for cat in SAMPLE_CATEGORIES:
        for item in data[cat][:2]:
            q = item["question"]
            ans = str(item.get("model_answer", ""))
            for rub in item.get("rubric", []):
                jobs.append((cat, q, rub, ans))

    client = make_chat_client(PROVIDER_OPENAI)
    sem = asyncio.Semaphore(10)

    async def run(cat, q, rub, ans):
        user = _build_judge_user_prompt(q, rub, ans)
        async with sem:
            try:
                res = await client.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": BEAM_JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                    ],
                    response_format=_JSON_OBJECT_FORMAT,
                    max_tokens=_MAX_TOKENS,
                )
            except Exception as e:
                return (cat, -1, False, f"API_ERROR: {e}")
        content = res["content"]
        ct = res["completion_tokens"]
        if not content:
            return (cat, ct, False, "EMPTY_CONTENT")
        try:
            parsed = _parse_json_loose(content)
            score = parsed.get("score") if isinstance(parsed, dict) else None
            return (cat, ct, True, f"score={score}")
        except Exception as e:
            return (cat, ct, False, f"PARSE_FAIL: {e}")

    rows = await asyncio.gather(*(run(*j) for j in jobs))
    await client.close()

    cts = [r[1] for r in rows if r[1] >= 0]
    bad = [r for r in rows if not r[2]]
    print(f"\n=== gpt-5 judge smoke: {len(rows)} calls, max_completion_tokens cap={_MAX_TOKENS} ===")
    print(f"failed/empty calls : {len(bad)}/{len(rows)}")
    if cts:
        cts_sorted = sorted(cts)
        print(f"completion_tokens  : min={cts_sorted[0]}  "
              f"median={cts_sorted[len(cts_sorted)//2]}  max={cts_sorted[-1]}")
        near_cap = sum(1 for c in cts if c > _MAX_TOKENS * 0.85)
        print(f"calls within 15% of cap: {near_cap}/{len(cts)}")
    for cat, ct, ok, note in rows:
        flag = "" if ok else "  <-- PROBLEM"
        print(f"  {cat:26s} ct={ct:>5}  {note}{flag}")


if __name__ == "__main__":
    asyncio.run(main())
