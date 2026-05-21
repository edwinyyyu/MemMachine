"""Re-answer an existing locomo_search.py output with reformatted timestamps.

Retrieval is fixed (timestamp display does not affect retrieval). This lets us
test timestamp-display formats cheaply -- no re-ingest, no re-retrieval.

The displayed raw-event timestamp is currently the Babel "full" date style:
``[Thursday, May 25, 2023, 1:14 PM]`` (~14 tokens). Compressing it frees
budget so a higher-K retrieval fits under the >=340 token cap.

ts-format choices:
  verbose  -- identity transform (CONTROL: isolates re-answer sampling variance)
  iso      -- [2023-05-25 13:14]      full date + 24h time, no weekday/monthname
  isowd    -- [Thu 2023-05-25 13:14]  iso + weekday abbreviation
  isodate  -- [2023-05-25]            date only, time dropped
  datevar  -- in-context date variables: each unique date defined once in a
              header (``date1 = Thursday, May 25, 2023``), referenced per line
              as ``[date1 13:14]``. Tests whether named-date binding helps
              temporal reasoning. Note: at K=7 retrieval (~1.8 events/date)
              the per-date definition overhead is mostly unamortized.

Usage:
  uv run python reanswer_tsfmt.py --data-path SEARCH.json --target-path OUT.json \
      --ts-format iso --model gpt-5-mini --concurrency 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time

from dotenv import load_dotenv
from openai import AsyncOpenAI

# Verbatim copy of locomo_search.py ANSWER_PROMPT_SIMPLE (the production default).
ANSWER_PROMPT_SIMPLE = """
You are a helpful assistant with access to extensive conversation history.
When answering questions, carefully review the conversation history to identify and use any relevant user preferences, interests, or specific details they have mentioned.

<history>
{memories}
</history>

Question: {question}
"""

_WEEKDAYS = "Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"
_MONTHS = (
    "January|February|March|April|May|June|July|August|September|"
    "October|November|December"
)
# Matches the Babel "full" date + "short" time style emitted by locomo_search.
_TS_RE = re.compile(
    r"\[((?:" + _WEEKDAYS + r"), (?:" + _MONTHS + r") \d{1,2}, \d{4}), "
    r"(\d{1,2}):(\d{2})[\s  ]*([AP]M)\]"
)
_MONTH_NUM = {
    m: i + 1
    for i, m in enumerate(
        "January February March April May June July August September "
        "October November December".split()
    )
}
_MONTH_ABBR = {
    m: m[:3]
    for m in (
        "January February March April May June July August September "
        "October November December".split()
    )
}
_WD_ABBR = {
    "Monday": "Mon", "Tuesday": "Tue", "Wednesday": "Wed", "Thursday": "Thu",
    "Friday": "Fri", "Saturday": "Sat", "Sunday": "Sun",
}


def _reformat(match: re.Match, ts_format: str) -> str:
    date_part, hour, minute, ampm = match.groups()
    weekday, rest = date_part.split(", ", 1)
    month_name, day_str, year_str = re.match(
        r"([A-Za-z]+) (\d{1,2}), (\d{4})", rest
    ).groups()
    month = _MONTH_NUM[month_name]
    day = int(day_str)
    year = int(year_str)
    hour24 = int(hour) % 12 + (12 if ampm == "PM" else 0)
    minute = int(minute)
    iso_date = f"{year:04d}-{month:02d}-{day:02d}"
    iso_time = f"{hour24:02d}:{minute:02d}"
    if ts_format == "verbose":
        return match.group(0)
    if ts_format == "iso":
        return f"[{iso_date} {iso_time}]"
    if ts_format == "isowd":
        return f"[{_WD_ABBR[weekday]} {iso_date} {iso_time}]"
    if ts_format == "isodate":
        return f"[{iso_date}]"
    if ts_format == "isodate_nb":
        # date-only ISO, NO brackets -- tests whether the brackets earn ~2 tok/line
        return iso_date
    if ts_format == "mediumdate":
        # Babel "medium" date-only: [May 8, 2023] -- tests medium vs iso surface form
        return f"[{_MONTH_ABBR[month_name]} {day}, {year}]"
    raise ValueError(f"unknown ts-format: {ts_format}")


def _datevar_parts(match: re.Match) -> tuple[str, str]:
    """Return (verbose_date, 'HH:MM') for a timestamp match."""
    date_part, hour, minute, ampm = match.groups()
    hour24 = int(hour) % 12 + (12 if ampm == "PM" else 0)
    return date_part, f"{hour24:02d}:{int(minute):02d}"


def _reformat_datevar(memories: str) -> tuple[str, int]:
    """In-context date variables: define each unique date once, reference per line.

    Header lists ``dateN = <verbose date>`` in chronological order of first
    appearance; every timestamp becomes ``[dateN HH:MM]``.
    """
    matches = list(_TS_RE.finditer(memories))
    if not matches:
        return memories, 0
    var_of: dict[str, str] = {}
    for m in matches:
        verbose_date, _ = _datevar_parts(m)
        if verbose_date not in var_of:
            var_of[verbose_date] = f"date{len(var_of) + 1}"

    def repl(m: re.Match) -> str:
        verbose_date, hhmm = _datevar_parts(m)
        return f"[{var_of[verbose_date]} {hhmm}]"

    body = _TS_RE.sub(repl, memories)
    header_lines = [f"{var} = {date}" for date, var in var_of.items()]
    header = "Dates (each line below is timestamped with one of these):\n" + "\n".join(
        header_lines
    )
    return f"{header}\n\n{body}", len(matches)


def reformat_context(memories: str, ts_format: str) -> tuple[str, int]:
    """Return (reformatted_context, num_timestamps_replaced)."""
    if ts_format == "datevar":
        return _reformat_datevar(memories)

    count = 0

    def repl(m: re.Match) -> str:
        nonlocal count
        count += 1
        return _reformat(m, ts_format)

    return _TS_RE.sub(repl, memories), count


async def reanswer(
    client: AsyncOpenAI,
    model: str,
    sem: asyncio.Semaphore,
    item: dict,
    ts_format: str,
    prepend: str = "",
) -> tuple[dict, int]:
    new_ctx, n = reformat_context(item.get("conversation_memories", ""), ts_format)
    if prepend:
        new_ctx = f"{prepend}\n{new_ctx}"
    async with sem:
        start = time.monotonic()
        for attempt in range(5):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": ANSWER_PROMPT_SIMPLE.format(
                                memories=new_ctx, question=item["question"]
                            ),
                        }
                    ],
                )
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 4:
                    raise
                await asyncio.sleep(min(2**attempt, 30))
        latency = time.monotonic() - start
    new_item = dict(item)
    new_item["conversation_memories"] = new_ctx
    new_item["model_answer"] = (resp.choices[0].message.content or "").strip()
    new_item["llm_latency"] = latency
    return new_item, n


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--target-path", required=True)
    parser.add_argument(
        "--ts-format",
        required=True,
        choices=[
            "verbose", "iso", "isowd", "isodate", "isodate_nb",
            "mediumdate", "datevar",
        ],
    )
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument(
        "--prepend",
        default="",
        help="Optional line prepended to the context (e.g. a chronological-"
        "ordering header). Tests display framing without re-retrieval.",
    )
    args = parser.parse_args()

    load_dotenv()
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sem = asyncio.Semaphore(args.concurrency)

    with open(args.data_path) as f:
        data = json.load(f)

    total_replaced = 0
    done = 0
    out: dict = {cat: [None] * len(items) for cat, items in data.items()}
    index = [(cat, i) for cat, items in data.items() for i in range(len(items))]

    async def run_one(cat: str, i: int, item: dict) -> None:
        nonlocal total_replaced, done
        new_item, n = await reanswer(
            client, args.model, sem, item, args.ts_format, args.prepend
        )
        out[cat][i] = new_item
        total_replaced += n
        done += 1
        if done % 200 == 0:
            print(f"  {done}/{len(index)} re-answered")

    await asyncio.gather(
        *(run_one(cat, i, data[cat][i]) for cat, i in index)
    )

    with open(args.target_path, "w") as f:
        json.dump(out, f, ensure_ascii=False)

    print(f"ts-format={args.ts_format}  timestamps replaced={total_replaced}")
    print(f"Saved to {args.target_path}")


if __name__ == "__main__":
    asyncio.run(main())
