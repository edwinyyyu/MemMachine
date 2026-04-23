"""Deep dive into the 18 questions where cons recalls more turns."""

import json

with open("recall-v250.json") as f:
    v250 = json.load(f)
with open("recall-v200-cons0_85.json") as f:
    cons = json.load(f)

overall_v250 = {q["question_id"]: q for q in v250["overall"]["per_question"]}
overall_cons = {q["question_id"]: q for q in cons["overall"]["per_question"]}

# Find the 18 questions where cons recalls more
diff_questions = []
for qid in overall_v250:
    rv = overall_v250[qid].get("recalled", 0)
    rc = overall_cons[qid].get("recalled", 0)
    n = overall_v250[qid]["num_answer_turns"]
    if rc > rv:
        diff_questions.append((qid, rv, rc, n))

print(f"Questions where cons recalls more: {len(diff_questions)}")
print(
    f"{'question_id':<20} {'v250_recalled':>14} {'cons_recalled':>14} {'answer_turns':>13}"
)
for qid, rv, rc, n in sorted(diff_questions, key=lambda x: -(x[2] - x[1])):
    print(f"{qid:<20} {rv:>14} {rc:>14} {n:>13}")

# Now look at per-category breakdown for these questions
print("\n=== Question types for the 18 diff questions ===")
# Need to find these in category-level per_question
diff_qids = {q[0] for q in diff_questions}
categories = [k for k in v250.keys() if k not in ("mode", "overall")]
for cat in categories:
    cat_qids = {q["question_id"] for q in v250[cat]["per_question"]}
    overlap = diff_qids & cat_qids
    if overlap:
        print(f"{cat}: {len(overlap)} questions - {sorted(overlap)}")

# Now load raw data to understand WHY these segments are missed
print("\n=== Loading raw search data ===")
with open("raw-v250.json") as f:
    raw_v250 = json.load(f)
with open("raw-v200-cons0_85.json") as f:
    raw_cons = json.load(f)

raw_v250_by_id = {r["question_id"]: r for r in raw_v250}
raw_cons_by_id = {r["question_id"]: r for r in raw_cons}


def turn_key(props):
    return f"{props['longmemeval_session_id']}:{props['turn_id']}"


def get_answer_turn_ranks(raw_item):
    """Return dict of answer_turn_key -> rank of first appearance."""
    answer_turns = set(raw_item["answer_turn_indices"])
    found = {}
    for sc in raw_item["segment_contexts"]:
        rank = sc["rank"]
        for seg in sc["segments"]:
            tk = turn_key(seg["properties"])
            if tk in answer_turns and tk not in found:
                found[tk] = rank
    return found


# For each diff question, show where answer turns appear in each ranking
for qid, rv, rc, n in sorted(diff_questions, key=lambda x: -(x[2] - x[1])):
    print(f"\n--- {qid} (answer_turns={n}, v250_recalled={rv}, cons_recalled={rc}) ---")

    raw_v = raw_v250_by_id[qid]
    raw_c = raw_cons_by_id[qid]

    v_ranks = get_answer_turn_ranks(raw_v)
    c_ranks = get_answer_turn_ranks(raw_c)

    answer_turns = set(raw_v["answer_turn_indices"])

    print(
        f"  v250: total_ranked_contexts={raw_v['total_ranked_contexts']}, total_segments={raw_v['total_segments']}"
    )
    print(
        f"  cons: total_ranked_contexts={raw_c['total_ranked_contexts']}, total_segments={raw_c['total_segments']}"
    )

    print("  Answer turn ranks:")
    for tk in sorted(answer_turns):
        vr = v_ranks.get(tk, "NOT FOUND")
        cr = c_ranks.get(tk, "NOT FOUND")
        marker = ""
        if vr == "NOT FOUND" and cr != "NOT FOUND":
            marker = " <-- ONLY IN CONS"
        elif cr == "NOT FOUND" and vr != "NOT FOUND":
            marker = " <-- ONLY IN V250"
        elif vr != "NOT FOUND" and cr != "NOT FOUND" and vr != cr:
            marker = f" (diff={vr - cr})"
        print(f"    {tk}: v250={vr}, cons={cr}{marker}")

# Summary stats on total_ranked_contexts and segments
print("\n=== Overall Collection Size Stats ===")
v250_contexts = [r["total_ranked_contexts"] for r in raw_v250]
cons_contexts = [r["total_ranked_contexts"] for r in raw_cons]
v250_segs = [r["total_segments"] for r in raw_v250]
cons_segs = [r["total_segments"] for r in raw_cons]
import statistics

for name, v_list, c_list in [
    ("ranked_contexts", v250_contexts, cons_contexts),
    ("total_segments", v250_segs, cons_segs),
]:
    print(f"\n{name}:")
    print(
        f"  v250: min={min(v_list)}, max={max(v_list)}, mean={statistics.mean(v_list):.1f}, median={statistics.median(v_list):.1f}"
    )
    print(
        f"  cons: min={min(c_list)}, max={max(c_list)}, mean={statistics.mean(c_list):.1f}, median={statistics.median(c_list):.1f}"
    )

# For the 18 diff questions, look at what segments cons finds that v250 doesn't
print("\n=== Detailed: How do the extra segments appear in cons? ===")
for qid, rv, rc, n in sorted(diff_questions, key=lambda x: -(x[2] - x[1]))[:5]:
    raw_v = raw_v250_by_id[qid]
    raw_c = raw_cons_by_id[qid]
    answer_turns = set(raw_v["answer_turn_indices"])

    # Find answer turns that cons finds but v250 doesn't
    v_found = set()
    c_found = {}
    for sc in raw_v["segment_contexts"]:
        for seg in sc["segments"]:
            tk = turn_key(seg["properties"])
            if tk in answer_turns:
                v_found.add(tk)

    for sc in raw_c["segment_contexts"]:
        for seg in sc["segments"]:
            tk = turn_key(seg["properties"])
            if tk in answer_turns and tk not in c_found:
                c_found[tk] = sc["rank"]

    only_in_cons = {tk: c_found[tk] for tk in c_found if tk not in v_found}
    print(f"\n--- {qid} ---")
    print(
        f"  v250 ranked_contexts: {raw_v['total_ranked_contexts']}, cons ranked_contexts: {raw_c['total_ranked_contexts']}"
    )
    print(f"  Answer turns only in cons: {only_in_cons}")

    # For each answer turn only in cons, find which ranked_context contains it
    # and check if it's riding with other segments (fan-out)
    for tk, rank in only_in_cons.items():
        sc_cons = raw_c["segment_contexts"][rank]
        num_segs = len(sc_cons["segments"])
        seg_turns = [turn_key(s["properties"]) for s in sc_cons["segments"]]
        print(
            f"    Turn {tk} at cons rank {rank}: context has {num_segs} segments, turns={seg_turns}"
        )

        # Is this segment's turn present ANYWHERE in v250 raw data?
        found_in_v250 = False
        for sc in raw_v["segment_contexts"]:
            for seg in sc["segments"]:
                if turn_key(seg["properties"]) == tk:
                    found_in_v250 = True
                    print(
                        f"    -> Also in v250 at rank {sc['rank']} (so it IS retrieved, just not matched as answer?)"
                    )
                    break
            if found_in_v250:
                break
        if not found_in_v250:
            print("    -> NOT in v250 raw results at all (not retrieved)")
