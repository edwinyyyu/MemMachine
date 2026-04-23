"""Why does cons retrieve more ranked_contexts than v250 for the same questions?"""

import json
import statistics

with open("raw-v250.json") as f:
    raw_v250 = json.load(f)
with open("raw-v200-cons0_85.json") as f:
    raw_cons = json.load(f)

raw_v250_by_id = {r["question_id"]: r for r in raw_v250}
raw_cons_by_id = {r["question_id"]: r for r in raw_cons}

# Compare ranked_contexts counts
diffs = []
for qid in raw_v250_by_id:
    v = raw_v250_by_id[qid]["total_ranked_contexts"]
    c = raw_cons_by_id[qid]["total_ranked_contexts"]
    diffs.append((qid, v, c, c - v))

# Distribution of differences
d_vals = [d[3] for d in diffs]
print("ranked_contexts difference (cons - v250):")
print(f"  mean={statistics.mean(d_vals):.1f}, median={statistics.median(d_vals):.1f}")
print(f"  min={min(d_vals)}, max={max(d_vals)}")
print(f"  cons > v250: {sum(1 for d in d_vals if d > 0)}")
print(f"  cons < v250: {sum(1 for d in d_vals if d < 0)}")
print(f"  cons == v250: {sum(1 for d in d_vals if d == 0)}")

# The key question: with derive_sentences, how many derivatives does each segment
# produce without consolidation? If each segment has N sentence-derivatives,
# then 250 derivative hits resolve to at most 250/N unique segments.
# With consolidation, derivatives are merged across segments, so 200 derivative
# hits can resolve to MORE unique segments (fan-out).

# Let's count: for v250, how many derivatives map to each unique segment?
# We can estimate this from the data: if v250 has 250 derivative hits (the limit)
# that resolve to X segments, then on average each segment has 250/X derivatives
# in the top results.

# Actually we can count directly from the raw data structure.
# In v250 (no consolidation, 1:1 derivative-to-segment), total_ranked_contexts
# equals the number of unique segments found, which equals the number of unique
# derivatives found (after dedup). The vector search limit is 250, so at most
# 250 derivatives were returned.

# For cons, the vector search limit is 200 derivatives, but each derivative can
# map to multiple segments. So total_ranked_contexts can exceed 200.

print("\n=== How often does cons exceed its vector search limit of 200? ===")
cons_over_200 = sum(1 for r in raw_cons if r["total_ranked_contexts"] > 200)
print(f"  Questions with >200 ranked_contexts: {cons_over_200}")
cons_over_100 = sum(1 for r in raw_cons if r["total_ranked_contexts"] > 100)
print(f"  Questions with >100 ranked_contexts: {cons_over_100}")

print("\n=== How often does v250 hit its vector search limit? ===")
# v250 max is 250 derivative hits -> at most 250 segments (1:1)
# If total_ranked_contexts < 250, the collection had fewer than 250 derivatives
v250_at_limit = sum(1 for r in raw_v250 if r["total_ranked_contexts"] >= 250)
v250_near_limit = sum(1 for r in raw_v250 if r["total_ranked_contexts"] >= 200)
print(f"  Questions with >=250 ranked_contexts: {v250_at_limit}")
print(f"  Questions with >=200 ranked_contexts: {v250_near_limit}")

# Key insight: in v250, derive_sentences creates multiple derivatives per segment.
# The vector search returns 250 derivatives. Many of these are different sentences
# from the SAME segment, so after dedup they collapse. The effective segment count
# is much less than 250.

# In cons, sentence-derivatives get consolidated ACROSS segments. So a single
# consolidated derivative can map to segments from different conversations turns.
# 200 derivative hits can fan out to >200 unique segments.

# Let's verify: what's the ratio of ranked_contexts to vector_search_limit?
print("\n=== Ratio of ranked_contexts to vector search limit ===")
v250_ratios = [r["total_ranked_contexts"] / 250 for r in raw_v250]
cons_ratios = [r["total_ranked_contexts"] / 200 for r in raw_cons]
print(
    f"v250 (limit=250): mean ratio={statistics.mean(v250_ratios):.3f}, median={statistics.median(v250_ratios):.3f}"
)
print(
    f"cons (limit=200): mean ratio={statistics.mean(cons_ratios):.3f}, median={statistics.median(cons_ratios):.3f}"
)

# For the 18 diff questions specifically
print("\n=== Ratio for the 18 diff questions ===")
diff_qids = {
    d[0]
    for d in diffs
    if raw_cons_by_id[d[0]]["total_ranked_contexts"]
    > raw_v250_by_id[d[0]]["total_ranked_contexts"]
}

# Actually, let me look at the specific 18 questions
recall_v250 = json.load(open("recall-v250.json"))
recall_cons = json.load(open("recall-v200-cons0_85.json"))
overall_v250 = {q["question_id"]: q for q in recall_v250["overall"]["per_question"]}
overall_cons = {q["question_id"]: q for q in recall_cons["overall"]["per_question"]}

diff_18 = [
    qid
    for qid in overall_v250
    if overall_cons[qid].get("recalled", 0) > overall_v250[qid].get("recalled", 0)
]

print("\nThe 18 diff questions:")
print(f"{'qid':<20} {'v250_ctx':>8} {'cons_ctx':>8} {'v250/250':>8} {'cons/200':>8}")
for qid in sorted(diff_18):
    vc = raw_v250_by_id[qid]["total_ranked_contexts"]
    cc = raw_cons_by_id[qid]["total_ranked_contexts"]
    print(f"{qid:<20} {vc:>8} {cc:>8} {vc / 250:>8.3f} {cc / 200:>8.3f}")

# What percentage of ALL questions have cons retrieving more unique segments?
print("\n=== Unique segment coverage ===")
# For each question: what fraction of total conversation segments are retrieved?
# We don't know total segments in the conversation, but we can compare the two.
print("Correlation between collection size and recall diff:")
# Larger collections = more likely to hit vector search limit = more dedup waste
large_colls = [
    (
        qid,
        raw_v250_by_id[qid]["total_ranked_contexts"],
        raw_cons_by_id[qid]["total_ranked_contexts"],
    )
    for qid in raw_v250_by_id
]
# Sort by v250 ranked_contexts (proxy for collection size)
large_colls.sort(key=lambda x: x[1], reverse=True)

# Top 20 largest by v250
print("\nTop 20 largest collections (by v250 ranked_contexts):")
print(f"{'qid':<20} {'v250_ctx':>8} {'cons_ctx':>8} {'diff':>8}")
for qid, vc, cc in large_colls[:20]:
    print(f"{qid:<20} {vc:>8} {cc:>8} {cc - vc:>+8}")

# Bottom 20
print("\nBottom 20 smallest collections:")
for qid, vc, cc in large_colls[-20:]:
    print(f"{qid:<20} {vc:>8} {cc:>8} {cc - vc:>+8}")
