"""Re-run only the v2_binding_K3_w7_w14 variant (timed out earlier)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND19 = HERE.parent
RESEARCH = ROUND19.parent
ROUND16A = RESEARCH / "round16a_sliding_window"
ROUND14 = RESEARCH / "round14_chain_density"
ROUND7 = RESEARCH / "round7"

sys.path.insert(0, str(ROUND19 / "architectures"))
sys.path.insert(0, str(ROUND16A / "scenarios"))
sys.path.insert(0, str(ROUND7 / "experiments"))

import multi_batch_coref  # noqa: E402
from _common import Budget  # noqa: E402
from run_v2 import run_variant, save  # noqa: E402


def main():
    budget = Budget(max_llm=400, max_embed=200, stop_at_llm=370, stop_at_embed=180)

    coref_turns = multi_batch_coref.generate()
    coref_gt = multi_batch_coref.ground_truth(coref_turns)
    coref_qs = multi_batch_coref.build_questions(coref_gt)

    # Load existing v2 results to merge.
    results_path = ROUND19 / "results" / "run_v2.json"
    if results_path.exists():
        results = json.loads(results_path.read_text())
    else:
        results = {"variants": {}}
    results.setdefault("variants", {})

    res = run_variant(
        "v2_binding_K3_w7_w14",
        turns=coref_turns,
        gt=coref_gt,
        qs=coref_qs,
        w_past=7,
        w_future=14,
        k=3,
        budget=budget,
        do_qa=True,
    )
    results["variants"]["v2_binding_K3_w7_w14"] = res
    save(results, budget)
    print(f"\n[done] cost=${budget.cost():.3f} llm={budget.llm_calls}")


if __name__ == "__main__":
    main()
