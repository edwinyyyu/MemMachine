import json

from negation import has_negation_cue, parse_negation_query

with open("data/negation_temporal_queries.jsonl") as f:
    for line in f:
        q = json.loads(line)
        cue = has_negation_cue(q["text"])
        pos, excl = parse_negation_query(q["text"])
        print(f"cue={cue}")
        print(f"  Q: {q['text']}")
        print(f"  +: {pos!r}")
        print(f"  -: {excl!r}")

# regression check: don't fire on non-negation queries
samples = [
    "What did I do in 2023?",
    "Latest meeting with Acme",
    "Did I not finish the report on time?",
    "Last week's grocery run",
    "What did I present at the conference?",
]
print("\n=== Regression check ===")
for s in samples:
    cue = has_negation_cue(s)
    pos, excl = parse_negation_query(s)
    print(f"cue={cue}  Q: {s}  +: {pos!r}  -: {excl!r}")
