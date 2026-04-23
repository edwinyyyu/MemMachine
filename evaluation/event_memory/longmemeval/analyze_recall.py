"""Analyze per-question recall differences between v250 and v200-cons0_85."""

import json

import numpy as np

with open("recall-v250.json") as f:
    v250 = json.load(f)
with open("recall-v200-cons0_85.json") as f:
    cons = json.load(f)

# Compare per-question first_hit_rank
overall_v250 = {q["question_id"]: q for q in v250["overall"]["per_question"]}
overall_cons = {q["question_id"]: q for q in cons["overall"]["per_question"]}

common_ids = set(overall_v250.keys()) & set(overall_cons.keys())
print(f"Total questions: {len(common_ids)}")

# 1. First hit rank comparison
better = []  # cons better (lower first_hit_rank)
worse = []  # cons worse
same = []
only_v250 = []  # v250 finds it, cons doesn't
only_cons = []  # cons finds it, v250 doesn't

for qid in sorted(common_ids):
    r_v250 = overall_v250[qid].get("first_hit_rank")
    r_cons = overall_cons[qid].get("first_hit_rank")
    n_turns = overall_v250[qid]["num_answer_turns"]

    if r_v250 is None and r_cons is None:
        continue
    if r_v250 is None and r_cons is not None:
        only_cons.append((qid, r_cons, n_turns))
        continue
    if r_cons is None and r_v250 is not None:
        only_v250.append((qid, r_v250, n_turns))
        continue

    diff = r_v250 - r_cons  # positive = cons is better
    if diff > 0:
        better.append((qid, r_v250, r_cons, diff, n_turns))
    elif diff < 0:
        worse.append((qid, r_v250, r_cons, diff, n_turns))
    else:
        same.append((qid, r_v250, r_cons, n_turns))

print("\n=== First Hit Rank Comparison ===")
print(f"Cons better (lower rank): {len(better)}")
print(f"Cons worse (higher rank): {len(worse)}")
print(f"Same: {len(same)}")
print(f"Only v250 finds: {len(only_v250)}")
print(f"Only cons finds: {len(only_cons)}")

if better:
    diffs = [b[3] for b in better]
    print(
        f"\nWhen cons is better: mean improvement = {np.mean(diffs):.1f} ranks, median = {np.median(diffs):.1f}"
    )
if worse:
    diffs = [abs(w[3]) for w in worse]
    print(
        f"When cons is worse: mean degradation = {np.mean(diffs):.1f} ranks, median = {np.median(diffs):.1f}"
    )

# 2. Per-question recalled count at various k values
print("\n=== Recalled Turns at Various K ===")
for k in [5, 10, 20, 50]:
    v250_recalled = 0
    cons_recalled = 0
    v250_better_count = 0
    cons_better_count = 0

    for qid in common_ids:
        q_v250 = overall_v250[qid]
        q_cons = overall_cons[qid]
        n_turns = q_v250["num_answer_turns"]
        if n_turns == 0:
            continue

        # Count how many answer turns are found by rank <= k
        # We need the raw data for this...
        # Use first_hit_rank as proxy: if first_hit_rank <= k, at least 1 found

    # Actually let's use the recall_at_k directly
    r_v250 = (
        v250["overall"]["recall_at_k"][k]
        if k < len(v250["overall"]["recall_at_k"])
        else None
    )
    r_cons = (
        cons["overall"]["recall_at_k"][k]
        if k < len(cons["overall"]["recall_at_k"])
        else None
    )
    print(f"k={k}: v250={r_v250:.4f}, cons={r_cons:.4f}, diff={r_cons - r_v250:+.4f}")

# 3. Per category comparison at k=10, k=20, k=50
print("\n=== Per Category Recall Difference (cons - v250) ===")
categories = [k for k in v250.keys() if k != "mode"]
print(f"{'category':<30} {'k=5':>8} {'k=10':>8} {'k=20':>8} {'k=50':>8}")
for cat in categories:
    diffs = []
    for k in [5, 10, 20, 50]:
        d = cons[cat]["recall_at_k"][k] - v250[cat]["recall_at_k"][k]
        diffs.append(d)
    print(
        f"{cat:<30} {diffs[0]:>+8.4f} {diffs[1]:>+8.4f} {diffs[2]:>+8.4f} {diffs[3]:>+8.4f}"
    )

# 4. Look at questions where cons is much better or worse
print("\n=== Top 10 Questions Where Cons is Better (by rank diff) ===")
for qid, r_v250, r_cons, diff, n_turns in sorted(better, key=lambda x: -x[3])[:10]:
    print(
        f"  {qid}: v250 rank={r_v250}, cons rank={r_cons}, improvement={diff}, answer_turns={n_turns}"
    )

print("\n=== Top 10 Questions Where Cons is Worse (by rank diff) ===")
for qid, r_v250, r_cons, diff, n_turns in sorted(worse, key=lambda x: x[3])[:10]:
    print(
        f"  {qid}: v250 rank={r_v250}, cons rank={r_cons}, degradation={abs(diff)}, answer_turns={n_turns}"
    )

# 5. Compare recalled counts (not just first hit)
print("\n=== Recalled Turn Counts ===")
v250_recalled_list = [overall_v250[qid].get("recalled", 0) for qid in common_ids]
cons_recalled_list = [overall_cons[qid].get("recalled", 0) for qid in common_ids]
print(f"v250 total recalled: {sum(v250_recalled_list)}")
print(f"cons total recalled: {sum(cons_recalled_list)}")
print(f"v250 mean recalled per question: {np.mean(v250_recalled_list):.2f}")
print(f"cons mean recalled per question: {np.mean(cons_recalled_list):.2f}")

# 6. Look at the recalled count differences
recalled_diffs = []
for qid in common_ids:
    r_v250 = overall_v250[qid].get("recalled", 0)
    r_cons = overall_cons[qid].get("recalled", 0)
    n_turns = overall_v250[qid]["num_answer_turns"]
    if n_turns > 0:
        recalled_diffs.append((qid, r_v250, r_cons, r_cons - r_v250, n_turns))

cons_more = [(qid, rv, rc, d, n) for qid, rv, rc, d, n in recalled_diffs if d > 0]
v250_more = [(qid, rv, rc, d, n) for qid, rv, rc, d, n in recalled_diffs if d < 0]
same_recalled = [(qid, rv, rc, d, n) for qid, rv, rc, d, n in recalled_diffs if d == 0]
print(
    f"\nCons recalls more turns: {len(cons_more)} questions (total +{sum(d for _, _, _, d, _ in cons_more)} turns)"
)
print(
    f"v250 recalls more turns: {len(v250_more)} questions (total {sum(d for _, _, _, d, _ in v250_more)} turns)"
)
print(f"Same recalled: {len(same_recalled)} questions")
