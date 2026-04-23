"""Why are the segment sets so disjoint? Check if certain sessions
have disproportionately many sentence-derivatives that flood the index."""

import json
from collections import Counter

with open("raw-v250.json") as f:
    raw_v250 = json.load(f)
with open("raw-v200-cons0_85.json") as f:
    raw_cons = json.load(f)

raw_v250_by_id = {r["question_id"]: r for r in raw_v250}
raw_cons_by_id = {r["question_id"]: r for r in raw_cons}


def turn_key(props):
    return f"{props['longmemeval_session_id']}:{props['turn_id']}"


# For a few diff questions, analyze session representation in results
# vs what we'd expect from the underlying data
diff_qids = ["031748ae", "1d4e3b97", "gpt4_18c2b244", "61f8c8f8", "19b5f2b3"]

for qid in diff_qids:
    raw_v = raw_v250_by_id[qid]
    raw_c = raw_cons_by_id[qid]
    answer_turns = set(raw_v["answer_turn_indices"])
    answer_sessions = set(tk.rsplit(":", 1)[0] for tk in answer_turns)

    # Count segments per session in v250 results
    v_session_counts = Counter()
    c_session_counts = Counter()
    for sc in raw_v["segment_contexts"]:
        for seg in sc["segments"]:
            v_session_counts[seg["properties"]["longmemeval_session_id"]] += 1
    for sc in raw_c["segment_contexts"]:
        for seg in sc["segments"]:
            c_session_counts[seg["properties"]["longmemeval_session_id"]] += 1

    print(f"\n{'=' * 60}")
    print(f"Question: {qid}")
    print(f"Query: {raw_v['question'][:100]}")
    print(f"Answer sessions: {answer_sessions}")
    print(f"\nv250 sessions (total {raw_v['total_ranked_contexts']} segments):")
    for sid, count in v_session_counts.most_common():
        marker = " *** ANSWER SESSION" if sid in answer_sessions else ""
        print(f"  {sid}: {count} segments{marker}")

    print(f"\ncons sessions (total {raw_c['total_ranked_contexts']} segments):")
    for sid, count in c_session_counts.most_common():
        marker = " *** ANSWER SESSION" if sid in answer_sessions else ""
        print(f"  {sid}: {count} segments{marker}")

    # What rank does the first answer session segment appear at in v250?
    first_answer_rank_v = None
    for sc in raw_v["segment_contexts"]:
        for seg in sc["segments"]:
            if seg["properties"]["longmemeval_session_id"] in answer_sessions:
                first_answer_rank_v = sc["rank"]
                break
        if first_answer_rank_v is not None:
            break

    first_answer_rank_c = None
    for sc in raw_c["segment_contexts"]:
        for seg in sc["segments"]:
            if seg["properties"]["longmemeval_session_id"] in answer_sessions:
                first_answer_rank_c = sc["rank"]
                break
        if first_answer_rank_c is not None:
            break

    print(
        f"\nFirst answer-session segment: v250 rank={first_answer_rank_v}, cons rank={first_answer_rank_c}"
    )
