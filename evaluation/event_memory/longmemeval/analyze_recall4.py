"""Analyze ranking quality: where do answer turns land in each system?"""

import json
import statistics
from collections import defaultdict

with open("raw-v250.json") as f:
    raw_v250 = json.load(f)
with open("raw-v200-cons0_85.json") as f:
    raw_cons = json.load(f)

raw_v250_by_id = {r["question_id"]: r for r in raw_v250}
raw_cons_by_id = {r["question_id"]: r for r in raw_cons}


def turn_key(props):
    return f"{props['longmemeval_session_id']}:{props['turn_id']}"


def get_answer_turn_ranks(raw_item):
    answer_turns = set(raw_item["answer_turn_indices"])
    found = {}
    for sc in raw_item["segment_contexts"]:
        rank = sc["rank"]
        for seg in sc["segments"]:
            tk = turn_key(seg["properties"])
            if tk in answer_turns and tk not in found:
                found[tk] = rank
    return found


# For ALL questions, compare answer turn ranks when BOTH systems find them
both_found_v250_ranks = []
both_found_cons_ranks = []
only_v250_ranks = []
only_cons_ranks = []
v250_better = 0
cons_better = 0
same_rank = 0

per_question_stats = []

for qid in raw_v250_by_id:
    raw_v = raw_v250_by_id[qid]
    raw_c = raw_cons_by_id[qid]

    v_ranks = get_answer_turn_ranks(raw_v)
    c_ranks = get_answer_turn_ranks(raw_c)

    answer_turns = set(raw_v["answer_turn_indices"])

    for tk in answer_turns:
        vr = v_ranks.get(tk)
        cr = c_ranks.get(tk)

        if vr is not None and cr is not None:
            both_found_v250_ranks.append(vr)
            both_found_cons_ranks.append(cr)
            if vr < cr:
                v250_better += 1
            elif cr < vr:
                cons_better += 1
            else:
                same_rank += 1
        elif vr is not None and cr is None:
            only_v250_ranks.append(vr)
        elif cr is not None and vr is None:
            only_cons_ranks.append(cr)

print("=== Answer Turn Rank Comparison (when both find the turn) ===")
print(f"Total answer turns found by both: {len(both_found_v250_ranks)}")
print(f"v250 ranks better: {v250_better}")
print(f"cons ranks better: {cons_better}")
print(f"same rank: {same_rank}")
print(
    f"\nv250 rank stats: mean={statistics.mean(both_found_v250_ranks):.1f}, median={statistics.median(both_found_v250_ranks):.1f}"
)
print(
    f"cons rank stats: mean={statistics.mean(both_found_cons_ranks):.1f}, median={statistics.median(both_found_cons_ranks):.1f}"
)

# Rank difference distribution
rank_diffs = [v - c for v, c in zip(both_found_v250_ranks, both_found_cons_ranks)]
print("\nRank diff (v250 - cons, positive = cons better):")
print(
    f"  mean={statistics.mean(rank_diffs):.2f}, median={statistics.median(rank_diffs):.1f}"
)
print(f"  stdev={statistics.stdev(rank_diffs):.2f}")

# Histogram of rank diffs
buckets = defaultdict(int)
for d in rank_diffs:
    if d < -20:
        buckets["<-20"] += 1
    elif d < -10:
        buckets["-20 to -10"] += 1
    elif d < -5:
        buckets["-10 to -5"] += 1
    elif d < 0:
        buckets["-5 to 0"] += 1
    elif d == 0:
        buckets["0"] += 1
    elif d <= 5:
        buckets["0 to 5"] += 1
    elif d <= 10:
        buckets["5 to 10"] += 1
    elif d <= 20:
        buckets["10 to 20"] += 1
    else:
        buckets[">20"] += 1

print("\nRank diff histogram:")
for bucket in [
    "<-20",
    "-20 to -10",
    "-10 to -5",
    "-5 to 0",
    "0",
    "0 to 5",
    "5 to 10",
    "10 to 20",
    ">20",
]:
    print(f"  {bucket:>12}: {buckets[bucket]:>4}")

print("\n=== Turns found by only one system ===")
print(
    f"Only in v250: {len(only_v250_ranks)} turns, mean rank={statistics.mean(only_v250_ranks):.1f}"
    if only_v250_ranks
    else "Only in v250: 0"
)
print(
    f"Only in cons: {len(only_cons_ranks)} turns, mean rank={statistics.mean(only_cons_ranks):.1f}"
    if only_cons_ranks
    else "Only in cons: 0"
)

if only_cons_ranks:
    print(
        f"  Only-cons rank distribution: min={min(only_cons_ranks)}, max={max(only_cons_ranks)}, median={statistics.median(only_cons_ranks):.1f}"
    )

# Normalize ranks by total_ranked_contexts to see percentile position
print("\n=== Normalized Rank (rank / total_ranked_contexts) ===")
norm_v250 = []
norm_cons = []
for qid in raw_v250_by_id:
    raw_v = raw_v250_by_id[qid]
    raw_c = raw_cons_by_id[qid]
    v_ranks = get_answer_turn_ranks(raw_v)
    c_ranks = get_answer_turn_ranks(raw_c)
    v_total = raw_v["total_ranked_contexts"]
    c_total = raw_c["total_ranked_contexts"]

    answer_turns = set(raw_v["answer_turn_indices"])
    for tk in answer_turns:
        vr = v_ranks.get(tk)
        cr = c_ranks.get(tk)
        if vr is not None:
            norm_v250.append(vr / v_total)
        if cr is not None:
            norm_cons.append(cr / c_total)

print(
    f"v250 normalized: mean={statistics.mean(norm_v250):.4f}, median={statistics.median(norm_v250):.4f}"
)
print(
    f"cons normalized: mean={statistics.mean(norm_cons):.4f}, median={statistics.median(norm_cons):.4f}"
)

# How many answer turns are in top-10 / top-20 / top-50 for each?
print("\n=== Answer Turns in Top-K ===")
for k in [5, 10, 20, 50]:
    v_in = (
        sum(1 for r in both_found_v250_ranks if r < k)
        + sum(1 for r in only_v250_ranks if r < k)
        if only_v250_ranks
        else sum(1 for r in both_found_v250_ranks if r < k)
    )
    c_in = (
        sum(1 for r in both_found_cons_ranks if r < k)
        + sum(1 for r in only_cons_ranks if r < k)
        if only_cons_ranks
        else sum(1 for r in both_found_cons_ranks if r < k)
    )

    # More carefully:
    v_in_k = 0
    c_in_k = 0
    for qid in raw_v250_by_id:
        raw_v = raw_v250_by_id[qid]
        raw_c = raw_cons_by_id[qid]
        v_ranks = get_answer_turn_ranks(raw_v)
        c_ranks = get_answer_turn_ranks(raw_c)
        answer_turns = set(raw_v["answer_turn_indices"])
        for tk in answer_turns:
            vr = v_ranks.get(tk)
            cr = c_ranks.get(tk)
            if vr is not None and vr < k:
                v_in_k += 1
            if cr is not None and cr < k:
                c_in_k += 1
    print(f"  k={k}: v250={v_in_k}, cons={c_in_k}, diff={c_in_k - v_in_k}")
