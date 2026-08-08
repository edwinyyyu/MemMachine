"""Normalize the feedback_retrieval synth corpus into a unified item/query record file.

Unified item:  {id, content_type, text, cluster, domain}
Unified query: {id, gold_id, content_type, situation, difficulty}

Drops queries whose gold_id is not present in the item set (reported).
"""
import json
import os

SYNTH = "/Users/eyu/edwinyyyu/mmcc/feedback_retrieval/synth"
OUT = os.path.join(SYNTH, "normalized_corpus.json")

CONV_FILES = [f"corpus_part_{i}.json" for i in range(4)]


def load(name):
    with open(os.path.join(SYNTH, name)) as f:
        return json.load(f)


def main():
    items = []
    queries = []

    # convention_or_principle (4 parts)
    for fn in CONV_FILES:
        d = load(fn)
        for l in d["lessons"]:
            items.append({
                "id": l["id"],
                "content_type": "convention_or_principle",
                "text": l["lesson"],
                "cluster": l.get("subcluster"),
                "domain": l.get("domain"),
            })
        for q in d["queries"]:
            queries.append({
                "id": q["id"],
                "gold_id": q["gold_lesson_id"],
                "content_type": "convention_or_principle",
                "situation": q["situation"],
                "difficulty": q["difficulty"],
            })

    # declarative_fact
    d = load("corpus_facts.json")
    for it in d["items"]:
        items.append({
            "id": it["id"],
            "content_type": "declarative_fact",
            "text": it["text"],
            # cluster field is present and informative ("orders-service/limits");
            # fall back to entity if absent.
            "cluster": it.get("cluster") or it.get("entity"),
            "domain": it.get("world"),
        })
    for q in d["queries"]:
        queries.append({
            "id": q["id"],
            "gold_id": q["gold_id"],
            "content_type": "declarative_fact",
            "situation": q["situation"],
            "difficulty": q["difficulty"],
        })

    # procedure
    d = load("corpus_procedures.json")
    for it in d["items"]:
        items.append({
            "id": it["id"],
            "content_type": "procedure",
            "text": it["text"],
            "cluster": it.get("cluster"),
            "domain": it.get("domain"),
        })
    for q in d["queries"]:
        queries.append({
            "id": q["id"],
            "gold_id": q["gold_id"],
            "content_type": "procedure",
            "situation": q["situation"],
            "difficulty": q["difficulty"],
        })

    # stance_feedback
    d = load("corpus_stance.json")
    for it in d["items"]:
        items.append({
            "id": it["id"],
            "content_type": "stance_feedback",
            "text": it["text"],
            "cluster": it.get("cluster") or it.get("failure_mode"),
            "domain": None,
        })
    for q in d["queries"]:
        queries.append({
            "id": q["id"],
            "gold_id": q["gold_id"],
            "content_type": "stance_feedback",
            "situation": q["situation"],
            "difficulty": q["difficulty"],
        })

    # integrity: unique ids
    item_ids = [it["id"] for it in items]
    assert len(item_ids) == len(set(item_ids)), "duplicate item ids!"
    item_id_set = set(item_ids)

    # drop queries with missing gold
    kept = []
    dropped = 0
    dropped_detail = []
    for q in queries:
        if q["gold_id"] in item_id_set:
            kept.append(q)
        else:
            dropped += 1
            dropped_detail.append((q["id"], q["gold_id"]))

    out = {"items": items, "queries": kept}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)

    print(f"items: {len(items)}")
    print(f"queries kept: {len(kept)}, dropped (missing gold): {dropped}")
    if dropped_detail:
        print("dropped:", dropped_detail[:20])
    from collections import Counter
    print("items by content_type:", dict(Counter(it["content_type"] for it in items)))
    print("queries by content_type:", dict(Counter(q["content_type"] for q in kept)))
    print("queries by difficulty:", dict(Counter(q["difficulty"] for q in kept)))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
