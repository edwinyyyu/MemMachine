"""Pull representative chunks from LongMemEval to stress-test LLM segmenter/deriver.

Categories we want to exercise:
  T - tables / structured data
  N - dense numeric content (lists of numbers, prices, stats)
  C - code blocks
  M - multi-topic long messages (where a good segmenter would cut)
  L - lists with itemized facts
  S - short coherent self-contained
"""

from __future__ import annotations

import re
import sys

sys.path.insert(
    0, "/Users/eyu/edwinyyyu/mmcc/segment_store/evaluation/event_memory/longmemeval"
)
from longmemeval_models import load_longmemeval_dataset

LME = (
    "/Users/eyu/edwinyyyu/mmcc/extra_memory/evaluation/data/longmemeval_s_cleaned.json"
)


def collect() -> dict[str, list[str]]:
    qs = load_longmemeval_dataset(LME)
    seen: set[str] = set()
    for q in qs:
        for sid in q.session_id_map:
            for turn in q.get_session(sid):
                t = turn.content.strip()
                if t and len(t) > 80:
                    seen.add(t)
    texts = list(seen)

    table = [t for t in texts if t.count("|") >= 6 and "---" in t][:30]
    code = [t for t in texts if "```" in t or "def " in t or "function " in t][:30]
    nums = [t for t in texts if len(re.findall(r"\$?\d[\d,]*(?:\.\d+)?", t)) >= 6][:30]
    long_multi = [t for t in texts if len(t) > 1500 and t.count("\n") >= 4][:30]
    lists = [
        t
        for t in texts
        if t.count("\n") >= 5 and re.search(r"^\s*\d+\.", t, re.MULTILINE)
    ][:30]
    short = [t for t in texts if 80 <= len(t) <= 200][:30]

    return {
        "T table": table,
        "C code": code,
        "N nums": nums,
        "M multi": long_multi,
        "L lists": lists,
        "S short": short,
    }


if __name__ == "__main__":
    bins = collect()
    for k, v in bins.items():
        print(f"{k}: {len(v)} chunks")
