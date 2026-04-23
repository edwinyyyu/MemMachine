"""Look at the 30 extra answer turns cons finds: their text, query, and neighbors."""

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

only_cons_turns.sort(key=lambda x: x[2])
print(f"Total only-in-cons turns: {len(only_cons_turns)}")
print(f"Rank distribution: {[t[2] for t in only_cons_turns]}")

# For each, show query, answer turn text, and neighboring ranked contexts in cons
for qid, tk, rank in only_cons_turns:
    raw_c = raw_cons_by_id[qid]
    raw_v = raw_v250_by_id[qid]

    print(f"\n{'=' * 80}")
    print(f"Question ID: {qid}")
    print(f"Question: {raw_c['question']}")
    print(f"Expected Answer: {raw_c['answer']}")
    print(f"Answer Turn Key: {tk}, Cons Rank: {rank}")
    print(
        f"v250 ranked_contexts: {raw_v['total_ranked_contexts']}, cons ranked_contexts: {raw_c['total_ranked_contexts']}"
    )

    # Find the answer turn's text in cons
    answer_sc = raw_c["segment_contexts"][rank]
    print(f"\n--- Answer segment at cons rank {rank} ---")
    print(f"  seed_segment_uuid: {answer_sc['seed_segment_uuid']}")
    for seg in answer_sc["segments"]:
        text = seg.get("text", "")
        props = seg["properties"]
        ctx = seg.get("context")
        source = ctx.get("source", "?") if ctx else "?"
        print(f"  [{source}] turn={turn_key(props)}: {text[:200]}")

    # Show a few neighbors in cons (ranks around this one)
    print("\n--- Neighboring ranks in cons ---")
    for r in range(max(0, rank - 2), min(len(raw_c["segment_contexts"]), rank + 3)):
        sc = raw_c["segment_contexts"][r]
        marker = " <<<" if r == rank else ""
        for seg in sc["segments"]:
            text = seg.get("text", "")
            props = seg["properties"]
            ctx = seg.get("context")
            source = ctx.get("source", "?") if ctx else "?"
            is_answer = turn_key(props) in set(raw_c["answer_turn_indices"])
            ans_marker = " [ANSWER]" if is_answer else ""
            print(
                f"  rank {r}{marker}{ans_marker}: [{source}] turn={turn_key(props)}: {text[:150]}"
            )

    # Check: is this turn's text present ANYWHERE in v250 results?
    # (maybe under a different segment/derivative)
    answer_turn_text = answer_sc["segments"][0].get("text", "")
    found_in_v250 = False
    for sc in raw_v["segment_contexts"]:
        for seg in sc["segments"]:
            if seg.get("text") == answer_turn_text:
                found_in_v250 = True
                print(f"\n  -> Same text found in v250 at rank {sc['rank']}")
                break
        if found_in_v250:
            break
    if not found_in_v250:
        # Check if any segment from the same turn_key is in v250
        for sc in raw_v["segment_contexts"]:
            for seg in sc["segments"]:
                if turn_key(seg["properties"]) == tk:
                    print(
                        f"\n  -> Same turn_key in v250 at rank {sc['rank']}: {seg.get('text', '')[:150]}"
                    )
                    found_in_v250 = True
                    break
            if found_in_v250:
                break
        if not found_in_v250:
            print("\n  -> Turn completely absent from v250 results")

    # Show what v250 has at similar ranks for comparison
    print("\n--- v250 top 5 ranks ---")
    for r in range(min(5, len(raw_v["segment_contexts"]))):
        sc = raw_v["segment_contexts"][r]
        for seg in sc["segments"]:
            text = seg.get("text", "")
            props = seg["properties"]
            ctx = seg.get("context")
            source = ctx.get("source", "?") if ctx else "?"
            is_answer = turn_key(props) in set(raw_v["answer_turn_indices"])
            ans_marker = " [ANSWER]" if is_answer else ""
            print(
                f"  rank {r}{ans_marker}: [{source}] turn={turn_key(props)}: {text[:150]}"
            )
