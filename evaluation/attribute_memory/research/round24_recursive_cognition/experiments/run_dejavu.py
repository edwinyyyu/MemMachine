"""Deja-vu retrieval evaluator. Bypasses the writer — loads memories
directly as Fact objects and tests retrieve_deja_vu / judge_structural_match
end-to-end against the eval scenario.
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
RESULTS_DIR = ROUND24 / "results"


def main():
    cache = Cache(CACHE_DIR / "dejavu.json")
    budget = Budget(max_llm=300, max_embed=200, stop_at_llm=290, stop_at_embed=190)

    # Build a MemoryStore directly from the scenario's memories.
    store = v2.MemoryStore()
    store.registry.register("m_user_root")
    store.registry.mention_to_entity["m_user_root"] = "e_user"
    store.registry.entity_members["e_user"] = {"m_user_root"}
    store.registry.entity_members.pop("e_m_user_root", None)

    use_scale = "--scale" in sys.argv
    memory_pool = scenario.ALL_MEMORIES if use_scale else scenario.MEMORIES

    facts: list[v2.Fact] = []
    mentions: list[v2.Mention] = []
    for i, m in enumerate(memory_pool):
        # Use the memory's stable label as the fact_uuid so we can grade by mid.
        f = v2.Fact(
            fact_uuid=m.mid,
            ts=i,
            text=m.text,
            mention_ids=[],
            collection="observations",
        )
        facts.append(f)
    store.collections["observations"] = v2.build_collection(
        "observations",
        facts,
        mentions,
        store.registry,
        cache,
        budget,
    )
    cache.save()

    multi_probe = "--multi-probe" in sys.argv
    n_variants = 4

    print(f"Loaded {len(facts)} memories.")
    mode = "multi-probe (query + variants)" if multi_probe else "single-probe"
    print(f"Running {len(scenario.QUESTIONS)} deja-vu queries (top_k=20, {mode}).")
    print()

    total_hit = 0
    total_expected = 0
    total_extras = 0

    for q in scenario.QUESTIONS:
        print(f"=== {q.qid} ===")
        print(f"Q: {q.question}")
        print(f"Expected DEEP: {q.expected}")
        if multi_probe:
            variants = v2.expand_query_for_deja_vu(
                q.question, cache, budget, n=n_variants
            )
            print("Query variants:")
            for v in variants:
                print(f"  • {v}")
            cache.save()
        t0 = time.time()
        results = v2.retrieve_deja_vu(
            q.question,
            store,
            cache,
            budget,
            top_k=20,
            multi_probe=multi_probe,
            n_variants=n_variants,
        )
        cache.save()
        dt = time.time() - t0

        deep_results: list[tuple[str, str, str]] = []
        for f, reason in results:
            deep_results.append((f.fact_uuid, f.text, reason))

        report = scenario.grade(deep_results, q.expected)
        print(f"DEEP returned ({len(deep_results)}, in {dt:.1f}s):")
        for mid, text, reason in deep_results:
            marker = " ★" if mid in q.expected else ""
            print(f"  [{mid}]{marker}  {text[:80]}")
            print(f"      → {reason[:110]}")
        print(
            f"hit={report['n_hit']}/{report['n_expected']}  missed={report['missed']}  extras_kept={report['extras']}"
        )
        print()
        total_hit += report["n_hit"]
        total_expected += report["n_expected"]
        total_extras += report["n_extra"]

    print(
        f"=== TOTAL: {total_hit}/{total_expected} expected DEEP matches recovered, {total_extras} extra DEEP labels ==="
    )
    print(f"Cost: ${budget.cost():.3f}, llm calls: {budget.llm_calls}")


if __name__ == "__main__":
    main()
