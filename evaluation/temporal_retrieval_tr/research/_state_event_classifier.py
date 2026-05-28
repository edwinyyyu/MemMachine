"""Heuristic state-shape classifier for doc text.

Identifies docs whose temporal anchor describes a STATE (continuous
holding-condition) rather than an EVENT (a happening). Designed to be
high-precision — false negatives OK, false positives bad.

A state-shape doc should not get temporal credit for event-locator
queries. The classifier is used by `_state_event_ab.py` to filter
state-doc match scores in an A/B test.
"""
from __future__ import annotations

import re

# High-precision regex patterns. Each captures a distinctive linguistic
# marker of a continuous condition or era reference.
STATE_REGEXES: list[re.Pattern] = [
    # Possessive/personal state assertions with explicit duration
    re.compile(r'\bI was (married|single|divorced|engaged) (from|in|between)\b', re.I),
    re.compile(r'\b(was|were) married (from|in) \d{4}\b', re.I),
    re.compile(r'\bmarried from \d{4}\b', re.I),

    # Residence states
    re.compile(r'\blived in \w+', re.I),  # "lived in NYC", etc.
    re.compile(r'\b(I|we) lived (in|at) \w+\b.* (from|for|since)\b', re.I),

    # Employment states
    re.compile(r'\bworked at \w+', re.I),
    re.compile(r'\bemployed (at|by) \w+\b.* (from|for|since)\b', re.I),
    re.compile(r'\bI worked at \w+ (from|for|since)\b', re.I),
    re.compile(r'\bI was (at|with) \w+\b.* (from|for|since)\b', re.I),

    # Era / context references
    re.compile(r'\bduring my time at\b', re.I),
    re.compile(r'\bback when I\b', re.I),
    re.compile(r'\bwhile (at|in|living in)\b', re.I),

    # Duration assertions
    re.compile(r'\bfor \d+ years\b', re.I),
    re.compile(r'\bfor \d+ months\b', re.I),
    re.compile(r'\bfor (the past|the last) \d+ (years|months)\b', re.I),
    re.compile(r'\bover \d+ years\b', re.I),

    # Open-ended state ("since X")
    re.compile(r'\b(I|we) (have|has) been .* since \d{4}\b', re.I),
    re.compile(r'\bsince \d{4}\b.*\b(I|we) (have|has) been\b', re.I),

    # Generic "<subject> was <descriptor> from YYYY" — catches
    # "Sarah was the chief legal officer from 2019 to 2024" etc.
    # 1-6 tokens between (was|were) and 'from YYYY'.
    re.compile(r'\b(was|were)\s+\w+(\s+\w+){0,5}\s+from\s+\d{4}\b', re.I),

    # "I was X" with date range
    re.compile(r'\bI was \w+(\s+\w+){0,4}\s+from\s+\d{4}\b', re.I),
]


def is_state_shape(text: str) -> bool:
    """Returns True iff doc text contains state-shape markers."""
    return any(p.search(text) for p in STATE_REGEXES)


def classify_corpus(docs: list[dict]) -> dict[str, bool]:
    """Returns {doc_id: is_state} mapping."""
    return {d["doc_id"]: is_state_shape(d["text"]) for d in docs}


if __name__ == "__main__":
    # Stand-alone diagnostic: enumerate state-classified docs per bench.
    import json
    from temporal_retrieval.research._common import DATA_DIR
    from temporal_retrieval_tr.research.bench import BENCH_NAMES

    print(f"\n=== State-shape classifier (heuristic) over {len(BENCH_NAMES)} benches ===\n",
          flush=True)
    total_docs = 0
    total_state = 0
    for bench in BENCH_NAMES:
        path = DATA_DIR / f"{bench}_docs.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            docs = [json.loads(line) for line in f]
        state_classified = [d for d in docs if is_state_shape(d["text"])]
        total_docs += len(docs)
        total_state += len(state_classified)
        if state_classified:
            print(f"--- {bench} ({len(state_classified)}/{len(docs)} state-shape) ---",
                  flush=True)
            for d in state_classified[:5]:
                print(f"    {d['doc_id']:30s}  {d['text'][:90]}", flush=True)
            if len(state_classified) > 5:
                print(f"    ... +{len(state_classified)-5} more", flush=True)
    print(f"\n  TOTAL: {total_state}/{total_docs} = {total_state/total_docs:.1%}",
          flush=True)
