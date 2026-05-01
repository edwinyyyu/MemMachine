"""Diagnose per-bench what's happening with T scoring."""

import json

with open(
    "/Users/eyu/edwinyyyu/mmcc/extra_memory/evaluation/temporal_extraction/results/missing_patterns_eval.json"
) as f:
    out = json.load(f)

for name in [
    "causal_relative",
    "latest_recent",
    "open_ended_date",
    "negation_temporal",
]:
    r = out[name]
    print(f"\n========= {name} =========")
    print(
        f"rerank_only R@1={r['rerank_only_r1']}/{r['n']}, T_lblend R@1={r['t_lblend_r1']}/{r['n']}"
    )
    # Count gold T scores: how many had nonzero T
    gold_t_nonzero = sum(1 for q in r["per_query"] if (q["gold_t"] or 0) > 0)
    print(f"queries where gold has nonzero T: {gold_t_nonzero}/{r['n']}")

    # Sample 3 queries for inspection
    for i, q in enumerate(r["per_query"][:3]):
        print(f"\n  Q[{i}] {q['query']}")
        print(f"    gold_text: {q['gold_text']}")
        print(
            f"    rerank_only_rank={q['rerank_only_rank']}  t_lblend_rank={q['t_lblend_rank']}"
        )
        gc = q["gold_components"]
        print(
            f"    gold T components: iv_raw={gc['iv_raw']:.3f}, tag={gc['tag']:.3f}, lat={gc['lat']:.3f}  (gold_t={q['gold_t']:.3f})"
        )
        print("    top-3 by T:")
        for t in q["top3_by_t"]:
            star = " <-- GOLD" if t["is_gold"] else ""
            print(
                f"      [{t['t_total']:.3f}] iv={t['iv_raw']:.3f} tag={t['tag']:.3f} lat={t['lat']:.3f}{star}"
            )
            print(f"          {t['text_snip']}")
