"""Per-bench summary stats."""

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
    n = r["n"]
    print(f"\n========= {name} =========")
    print(
        f"n={n}  rerank_only R@1={r['rerank_only_r1']}/{n}  T_lblend R@1={r['t_lblend_r1']}/{n}"
    )

    # Stats
    gold_zero_t = sum(1 for q in r["per_query"] if (q["gold_t"] or 0) == 0)
    gold_zero_iv = sum(1 for q in r["per_query"] if q["gold_components"]["iv_raw"] == 0)
    gold_zero_lat = sum(1 for q in r["per_query"] if q["gold_components"]["lat"] == 0)
    gold_zero_tag = sum(1 for q in r["per_query"] if q["gold_components"]["tag"] == 0)

    # When gold has nonzero T, does T's top-1 still beat gold?
    gold_outranked_with_signal = 0
    gold_outranked_total = 0
    for q in r["per_query"]:
        gt = q["gold_t"]
        if gt is None:
            continue
        top1 = q["top3_by_t"][0]
        if not top1["is_gold"]:
            gold_outranked_total += 1
            if top1["t_total"] > 0:
                gold_outranked_with_signal += 1

    print(f"  gold has T=0: {gold_zero_t}/{n}")
    print(
        f"    of which iv=0: {gold_zero_iv}, lat=0: {gold_zero_lat}, tag=0: {gold_zero_tag}"
    )
    print(f"  T's top-1 != gold: {gold_outranked_total}/{n}")
    print(
        f"    of which top-1 had nonzero T (active wrong pick): {gold_outranked_with_signal}/{n}"
    )
