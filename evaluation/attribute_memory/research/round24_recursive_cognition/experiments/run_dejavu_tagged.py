"""Tag-based deja-vu retrieval evaluator.

Tags every memory at load time using the controlled vocabulary in
aen6_prose_v2.STRUCTURAL_TAG_VOCAB. Tags the query at retrieval time.
Retrieval = inverted-index lookup (union of buckets of memories whose
tags overlap query's tags), then LLM judge.

Compares against `run_dejavu.py` which uses pure-embedding kNN.

Usage:
  uv run python run_dejavu_tagged.py [--scale]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND24 = HERE.parent
RESEARCH = ROUND24.parent
ROUND7 = RESEARCH / "round7"

sys.path.insert(0, str(ROUND24 / "architectures"))
sys.path.insert(0, str(ROUND24 / "scenarios"))
sys.path.insert(0, str(RESEARCH / "round23_prose_facts" / "architectures"))
sys.path.insert(0, str(ROUND7 / "experiments"))

import aen6_prose_v2 as v2  # noqa: E402
import deja_vu_eval as scenario  # noqa: E402
from _common import Budget, Cache  # noqa: E402

CACHE_DIR = ROUND24 / "cache"


def main():
    cache = Cache(CACHE_DIR / "dejavu_tagged.json")
    budget = Budget(max_llm=2000, max_embed=400, stop_at_llm=1900, stop_at_embed=380)

    use_scale = "--scale" in sys.argv
    memory_pool = scenario.ALL_MEMORIES if use_scale else scenario.MEMORIES

    store = v2.MemoryStore()
    store.registry.register("m_user_root")
    store.registry.mention_to_entity["m_user_root"] = "e_user"
    store.registry.entity_members["e_user"] = {"m_user_root"}
    store.registry.entity_members.pop("e_m_user_root", None)

    facts = []
    for i, m in enumerate(memory_pool):
        facts.append(
            v2.Fact(
                fact_uuid=m.mid,
                ts=i,
                text=m.text,
                mention_ids=[],
                collection="observations",
            )
        )
    store.collections["observations"] = v2.build_collection(
        "observations",
        facts,
        [],
        store.registry,
        cache,
        budget,
    )
    cache.save()

    print(f"Loaded {len(facts)} memories.")
    print(
        f"Tagging memories with structural-pattern vocabulary ({len(v2.STRUCTURAL_TAG_VOCAB)} tags)..."
    )
    t0 = time.time()
    tag_index = v2.build_tag_index(facts, cache, budget)
    cache.save()
    print(f"  Tagged in {time.time() - t0:.1f}s.")
    stats = tag_index.stats()
    print(f"  Index stats: {stats}")
    # Print tag distribution
    print("  Tag distribution (memories per tag):")
    for tag, fids in sorted(
        tag_index.tag_to_fact_uuids.items(), key=lambda kv: -len(kv[1])
    ):
        print(f"    {len(fids):3d}  {tag}")
    print()

    total_hit = 0
    total_expected = 0
    total_extras = 0

    for q in scenario.QUESTIONS:
        print(f"=== {q.qid} ===")
        print(f"Q: {q.question}")
        print(f"Expected DEEP: {q.expected}")
        # Print query's tags
        q_tags = v2.tag_text(q.question, cache, budget)
        print(f"Query tags: {q_tags}")
        bucket_size = len(tag_index.lookup(q_tags))
        print(
            f"Tag-bucket size: {bucket_size} / {len(facts)} ({100 * bucket_size / max(1, len(facts)):.1f}%)"
        )
        cache.save()

        t0 = time.time()
        results = v2.retrieve_deja_vu_tagged(
            q.question,
            store,
            tag_index,
            cache,
            budget,
            collections=["observations"],
        )
        cache.save()
        dt = time.time() - t0

        deep_results = [(f.fact_uuid, f.text, reason) for (f, reason, _tags) in results]
        report = scenario.grade(deep_results, q.expected)
        print(f"DEEP returned ({len(deep_results)}, in {dt:.1f}s):")
        for mid, text, reason in deep_results:
            marker = " ★" if mid in q.expected else ""
            tags = tag_index.fact_uuid_to_tags.get(mid, [])
            print(f"  [{mid}]{marker}  tags={tags}  {text[:60]}")
            print(f"      → {reason[:100]}")
        print(
            f"hit={report['n_hit']}/{report['n_expected']}  missed={report['missed']}  extras={len(report['extras'])}"
        )
        print()
        total_hit += report["n_hit"]
        total_expected += report["n_expected"]
        total_extras += report["n_extra"]

    print(
        f"=== TOTAL: {total_hit}/{total_expected} expected DEEP recovered, {total_extras} extras ==="
    )
    print(f"Cost: ${budget.cost():.3f}, llm calls: {budget.llm_calls}")


if __name__ == "__main__":
    main()
