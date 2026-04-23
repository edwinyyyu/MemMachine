"""Check if the only-in-cons answer segments are seed segments (=representative of their derivative)
vs riding on another segment's derivative.

Key question: if the answer segment IS the seed (rank 0 or 1),
then ITS OWN derivative scored high. But v250 didn't find it.
This would mean the same embedding exists in both indexes but
only ranks high in the consolidated (smaller) index."""

import json

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


# Find the 30 only-in-cons turns
only_cons_turns = []
for qid in raw_v250_by_id:
    raw_v = raw_v250_by_id[qid]
    raw_c = raw_cons_by_id[qid]
    v_ranks = get_answer_turn_ranks(raw_v)
    c_ranks = get_answer_turn_ranks(raw_c)
    answer_turns = set(raw_v["answer_turn_indices"])

    for tk in answer_turns:
        if tk not in v_ranks and tk in c_ranks:
            only_cons_turns.append((qid, tk, c_ranks[tk]))

# For each, check if the answer segment is the seed of its ranked context
# If context has 1 segment, the seed IS the answer segment
# If context has >1 segments, check if seed_segment_uuid matches
print(f"Total only-in-cons turns: {len(only_cons_turns)}\n")

seed_is_answer = 0
seed_is_other = 0

for qid, tk, rank in sorted(only_cons_turns, key=lambda x: x[2]):
    raw_c = raw_cons_by_id[qid]
    sc = raw_c["segment_contexts"][rank]

    # Find which segment in the context is the answer turn
    answer_seg_uuid = None
    for seg in sc["segments"]:
        if turn_key(seg["properties"]) == tk:
            answer_seg_uuid = seg["uuid"]
            break

    is_seed = answer_seg_uuid == sc["seed_segment_uuid"]
    num_segs = len(sc["segments"])

    if is_seed:
        seed_is_answer += 1
    else:
        seed_is_other += 1

    # Check what v250 has at the same rank
    raw_v = raw_v250_by_id[qid]
    v_at_rank = None
    if rank < len(raw_v["segment_contexts"]):
        v_sc = raw_v["segment_contexts"][rank]
        v_seg = v_sc["segments"][0] if v_sc["segments"] else None
        if v_seg:
            v_text = v_seg.get("text", "")[:80]
            v_turn = turn_key(v_seg["properties"])
            v_at_rank = f"{v_turn}: {v_text}"

    print(
        f"qid={qid}, turn={tk}, cons_rank={rank}, is_seed={is_seed}, context_size={num_segs}"
    )
    if not is_seed:
        # Show what the seed is
        seed_uuid = sc["seed_segment_uuid"]
        for seg in sc["segments"]:
            if seg["uuid"] == seed_uuid:
                print(
                    f"  SEED: turn={turn_key(seg['properties'])}: {seg.get('text', '')[:120]}"
                )
                break

    # Show the answer text
    for seg in sc["segments"]:
        if seg["uuid"] == answer_seg_uuid:
            print(f"  ANSWER: {seg.get('text', '')[:120]}")
            break

    if v_at_rank:
        print(f"  v250@rank{rank}: {v_at_rank}")

print("\n=== Summary ===")
print(f"Answer segment IS the seed (own derivative scored high): {seed_is_answer}")
print(f"Answer segment is NOT the seed (riding another's derivative): {seed_is_other}")
