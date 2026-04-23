"""Analyze WHY v250 retrieves different segments than cons.
1. What do v250's top results look like for the 18 diff questions?
   Are they generic conversational patterns?
2. Per-question segment count comparison for the 18 diff questions.
3. Look at common text patterns in v250's top results."""

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


# Find the 18 diff questions
diff_qids = set()
for qid in raw_v250_by_id:
    raw_v = raw_v250_by_id[qid]
    raw_c = raw_cons_by_id[qid]
    v_ranks = get_answer_turn_ranks(raw_v)
    c_ranks = get_answer_turn_ranks(raw_c)
    answer_turns = set(raw_v["answer_turn_indices"])
    for tk in answer_turns:
        if tk not in v_ranks and tk in c_ranks:
            diff_qids.add(qid)

# 1. For these 18 questions, what sessions do v250's top-10 segments come from?
# Are they from the answer session or noise sessions?
print("=== v250 top-10: how many are from the ANSWER session vs other sessions? ===")
for qid in sorted(diff_qids):
    raw_v = raw_v250_by_id[qid]
    raw_c = raw_cons_by_id[qid]
    answer_turns = set(raw_v["answer_turn_indices"])

    # Extract answer session IDs
    answer_sessions = set()
    for tk in answer_turns:
        session_id = tk.rsplit(":", 1)[0]
        answer_sessions.add(session_id)

    v_top10_sessions = Counter()
    c_top10_sessions = Counter()
    for sc in raw_v["segment_contexts"][:10]:
        for seg in sc["segments"]:
            sid = seg["properties"]["longmemeval_session_id"]
            v_top10_sessions[sid] += 1
    for sc in raw_c["segment_contexts"][:10]:
        for seg in sc["segments"]:
            sid = seg["properties"]["longmemeval_session_id"]
            c_top10_sessions[sid] += 1

    v_answer_count = sum(v_top10_sessions[s] for s in answer_sessions)
    c_answer_count = sum(c_top10_sessions[s] for s in answer_sessions)

    print(f"\n{qid}: answer_sessions={answer_sessions}")
    print(
        f"  v250 top-10: {v_answer_count}/10 from answer sessions, {10 - v_answer_count}/10 from other"
    )
    print(
        f"  cons top-10: {c_answer_count}/10 from answer sessions, {10 - c_answer_count}/10 from other"
    )

# 2. Look at text patterns - are v250's results generic?
print("\n\n=== Text patterns in v250 vs cons top-10 for diff questions ===")
v250_texts = []
cons_texts = []
for qid in sorted(diff_qids):
    raw_v = raw_v250_by_id[qid]
    raw_c = raw_cons_by_id[qid]
    for sc in raw_v["segment_contexts"][:10]:
        for seg in sc["segments"]:
            text = seg.get("text", "")
            v250_texts.append(text)
    for sc in raw_c["segment_contexts"][:10]:
        for seg in sc["segments"]:
            text = seg.get("text", "")
            cons_texts.append(text)

# Check average text length
v_lens = [len(t) for t in v250_texts]
c_lens = [len(t) for t in cons_texts]
print(f"v250 avg text length: {sum(v_lens) / len(v_lens):.0f}")
print(f"cons avg text length: {sum(c_lens) / len(c_lens):.0f}")

# Look for generic opener patterns
generic_starters = [
    "That's ",
    "That sounds",
    "Wow,",
    "These ",
    "Sure!",
    "Sure,",
    "I think ",
    "I'd ",
    "I'm glad",
    "Great ",
    "Yes,",
    "Absolutely",
    "Good ",
    "Thank",
    "Glad ",
    "No worries",
    "It's ",
    "Makes sense",
]


def count_generic(texts):
    count = 0
    for t in texts:
        t_stripped = t.strip()
        for starter in generic_starters:
            if t_stripped.startswith(starter):
                count += 1
                break
    return count


v_generic = count_generic(v250_texts)
c_generic = count_generic(cons_texts)
print(
    f"v250 generic starters: {v_generic}/{len(v250_texts)} ({v_generic / len(v250_texts) * 100:.1f}%)"
)
print(
    f"cons generic starters: {c_generic}/{len(cons_texts)} ({c_generic / len(cons_texts) * 100:.1f}%)"
)

# 3. How many unique sessions appear in top-10 for each?
print("\n=== Session diversity in top-10 ===")
for qid in sorted(diff_qids):
    raw_v = raw_v250_by_id[qid]
    raw_c = raw_cons_by_id[qid]

    v_sessions = set()
    c_sessions = set()
    for sc in raw_v["segment_contexts"][:10]:
        for seg in sc["segments"]:
            v_sessions.add(seg["properties"]["longmemeval_session_id"])
    for sc in raw_c["segment_contexts"][:10]:
        for seg in sc["segments"]:
            c_sessions.add(seg["properties"]["longmemeval_session_id"])

    print(f"{qid}: v250={len(v_sessions)} sessions, cons={len(c_sessions)} sessions")

# 4. Across ALL 500 questions, compare session concentration in top-50
print("\n=== Session concentration in top-50 across ALL questions ===")
v_answer_in_top50 = 0
c_answer_in_top50 = 0
v_total = 0
c_total = 0

for qid in raw_v250_by_id:
    raw_v = raw_v250_by_id[qid]
    raw_c = raw_cons_by_id[qid]
    answer_turns = set(raw_v["answer_turn_indices"])
    answer_sessions = set(tk.rsplit(":", 1)[0] for tk in answer_turns)

    for sc in raw_v["segment_contexts"][:50]:
        v_total += 1
        for seg in sc["segments"]:
            if seg["properties"]["longmemeval_session_id"] in answer_sessions:
                v_answer_in_top50 += 1
                break

    for sc in raw_c["segment_contexts"][:50]:
        c_total += 1
        for seg in sc["segments"]:
            if seg["properties"]["longmemeval_session_id"] in answer_sessions:
                c_answer_in_top50 += 1
                break

print(
    f"v250: {v_answer_in_top50}/{v_total} top-50 segments from answer sessions ({v_answer_in_top50 / v_total * 100:.1f}%)"
)
print(
    f"cons: {c_answer_in_top50}/{c_total} top-50 segments from answer sessions ({c_answer_in_top50 / c_total * 100:.1f}%)"
)

# 5. For the 18 diff questions, what's in v250 but NOT in cons?
print("\n=== Segments in v250 but NOT in cons (for diff questions) ===")
for qid in sorted(list(diff_qids)[:5]):
    raw_v = raw_v250_by_id[qid]
    raw_c = raw_cons_by_id[qid]

    v_turns = set()
    c_turns = set()
    for sc in raw_v["segment_contexts"]:
        for seg in sc["segments"]:
            v_turns.add(turn_key(seg["properties"]))
    for sc in raw_c["segment_contexts"]:
        for seg in sc["segments"]:
            c_turns.add(turn_key(seg["properties"]))

    only_v = v_turns - c_turns
    only_c = c_turns - v_turns
    print(f"\n{qid}: v250 has {len(v_turns)} turns, cons has {len(c_turns)} turns")
    print(f"  Only in v250: {len(only_v)}, Only in cons: {len(only_c)}")
    print(f"  Shared: {len(v_turns & c_turns)}")
