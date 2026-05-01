import json

from negation import has_negation_cue

files = [
    "edge_conjunctive_temporal_queries.jsonl",
    "edge_multi_te_doc_queries.jsonl",
    "edge_relative_time_queries.jsonl",
    "edge_era_refs_queries.jsonl",
    "hard_bench_queries.jsonl",
    "temporal_essential_queries.jsonl",
]
for f in files:
    n = 0
    nc = 0
    fires = []
    for line in open(f"data/{f}"):
        q = json.loads(line)
        n += 1
        if has_negation_cue(q["text"]):
            nc += 1
            fires.append(q["text"])
    print(f"{f}: {nc}/{n} cues fired")
    for t in fires:
        print(f"  - {t!r}")
