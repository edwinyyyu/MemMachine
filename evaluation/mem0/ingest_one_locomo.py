"""Ingest one LoCoMo conversation into Mem0 cloud, bench-compatible.

Mirrors `benchmarks/locomo/run.py` ingestion: CHUNK_SIZE=1 turn per call,
sessions ordered chronologically, each chunk timestamped with the session
date. Uses V3 endpoint with event polling so each chunk completes before
the next starts.
"""

import asyncio
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
API_KEY = os.environ["MEM0_API_KEY"]
HOST = os.getenv("MEM0_HOST", "https://api.mem0.ai").rstrip("/")

LOCOMO_DATA = Path(__file__).parent.parent / "data" / "locomo10.json"
CONV_IDX = int(os.getenv("LOCOMO_CONV_IDX", "0"))
RUN_TAG = os.getenv("LOCOMO_RUN_TAG", "probe")
EVENT_POLL_INTERVAL = 0.5
EVENT_POLL_TIMEOUT = 180.0

CHUNK_SIZE = 1


def parse_locomo_date(date_str: str) -> int | None:
    """Return unix epoch (UTC) for a locomo session date string, or None."""
    if not date_str:
        return None
    for fmt in (
        "%I:%M %p on %d %B, %Y",
        "%H:%M on %d %B, %Y",
        "%d %B, %Y",
        "%I:%M %p on %B %d, %Y",
    ):
        try:
            dt = datetime.strptime(date_str, fmt)
            return int(dt.replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            continue
    return None


def session_to_chunks(
    turns: list[dict], speaker_a: str, speaker_b: str
) -> list[list[dict]]:
    """Convert a session's turns into chunked message lists (bench format)."""
    messages: list[dict] = []
    for turn in turns:
        speaker = turn.get("speaker", "")
        text = turn.get("text", "")
        blip = turn.get("blip_caption", "")
        query = turn.get("query", "")
        if query and blip:
            photo = f"[Sharing image - query: {query}. The image shows: {blip}]"
        elif query:
            photo = f"[Sharing image - query for: {query}]"
        elif blip:
            photo = f"[Sharing image that shows: {blip}]"
        else:
            photo = ""
        if photo:
            text = f"{text} {photo}" if text else photo
        if not text:
            continue
        role = "user" if speaker == speaker_a else "assistant"
        messages.append({"role": role, "content": f"{speaker}: {text}"})

    return [messages[i : i + CHUNK_SIZE] for i in range(0, len(messages), CHUNK_SIZE)]


def get_sorted_sessions(conversation: dict) -> list[tuple[str, str, list[dict]]]:
    """Extract sessions and sort them chronologically by their date string."""
    keys = [k for k in conversation if re.match(r"^session_\d+$", k)]
    paired = []
    for k in keys:
        paired.append((k, conversation.get(f"{k}_date_time", ""), conversation[k]))

    def sort_key(item: tuple) -> tuple:
        ts = parse_locomo_date(item[1])
        if ts is not None:
            return (0, ts)
        num = int(re.search(r"\d+", item[0]).group())
        return (1, num)

    paired.sort(key=sort_key)
    return paired


async def add_and_wait(
    session: aiohttp.ClientSession,
    messages: list[dict],
    user_id: str,
    timestamp: int | None,
) -> dict | None:
    """POST /v3/memories/ then poll /v1/event/{id}/ until SUCCEEDED or FAILED."""
    payload: dict = {"messages": messages, "user_id": user_id}
    if timestamp is not None:
        payload["timestamp"] = timestamp

    async with session.post(f"{HOST}/v3/memories/", json=payload) as resp:
        resp.raise_for_status()
        data = await resp.json()
    event_id = data.get("event_id")
    if not event_id:
        return data

    start = time.monotonic()
    while (time.monotonic() - start) < EVENT_POLL_TIMEOUT:
        async with session.get(f"{HOST}/v1/event/{event_id}/") as resp:
            resp.raise_for_status()
            event = await resp.json()
        status = event.get("status", "UNKNOWN")
        if status in ("SUCCEEDED", "FAILED"):
            return event
        await asyncio.sleep(EVENT_POLL_INTERVAL)
    return None


async def main() -> None:
    with open(LOCOMO_DATA) as f:
        data = json.load(f)

    entry = data[CONV_IDX]
    conv = entry["conversation"]
    speaker_a = conv["speaker_a"]
    speaker_b = conv["speaker_b"]
    user_id = f"locomo_{CONV_IDX}_{RUN_TAG}"

    sessions = get_sorted_sessions(conv)
    all_chunks = [
        (sk, sd, c)
        for sk, sd, turns in sessions
        for c in session_to_chunks(turns, speaker_a, speaker_b)
    ]
    total = len(all_chunks)
    print(f"Conversation {CONV_IDX}: {speaker_a} & {speaker_b}")
    print(f"Sessions: {len(sessions)}  total chunks: {total}")
    print(f"user_id: {user_id}")
    print()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Token {API_KEY}",
    }
    timeout = aiohttp.ClientTimeout(total=300)
    connector = aiohttp.TCPConnector(limit=0)

    fail_count = 0
    extracted_count = 0
    durations: list[float] = []
    t_start = time.monotonic()

    async with aiohttp.ClientSession(
        headers=headers, timeout=timeout, connector=connector
    ) as session:
        for i, (sess_key, sess_date, chunk) in enumerate(all_chunks, 1):
            if any(not m.get("content", "").strip() for m in chunk):
                continue
            ts = parse_locomo_date(sess_date)
            t0 = time.monotonic()
            try:
                event = await add_and_wait(session, chunk, user_id, ts)
            except Exception as e:
                fail_count += 1
                print(f"  [{i}/{total}] ERROR: {type(e).__name__}: {str(e)[:120]}")
                continue
            dt = time.monotonic() - t0
            durations.append(dt)

            if event is None:
                fail_count += 1
                status = "TIMEOUT"
                n = 0
            else:
                status = event.get("status", "UNKNOWN")
                results = event.get("results", []) or []
                n = len(results)
                if status == "SUCCEEDED":
                    extracted_count += n
                else:
                    fail_count += 1

            if i <= 3 or i % 25 == 0 or i == total:
                msg_preview = chunk[0]["content"][:80]
                print(
                    f"  [{i:>3}/{total}] {dt:>5.1f}s {status:<10} "
                    f"+{n} memories | {sess_key} | {msg_preview!r}"
                )

    elapsed = time.monotonic() - t_start
    print()
    print(
        f"Ingested: {len(durations)} chunks in {elapsed:.1f}s ({elapsed / max(len(durations), 1):.2f}s/chunk avg)"
    )
    print(f"Memories extracted: {extracted_count}")
    print(f"Failures: {fail_count}")
    if durations:
        durations_sorted = sorted(durations)
        n = len(durations_sorted)
        print(
            f"Per-chunk latency: min={min(durations):.2f}s "
            f"p50={durations_sorted[n // 2]:.2f}s "
            f"p90={durations_sorted[int(n * 0.9)]:.2f}s "
            f"max={max(durations):.2f}s"
        )


if __name__ == "__main__":
    asyncio.run(main())
