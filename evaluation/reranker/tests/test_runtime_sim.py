"""The runtime pool in the core simulation's terms: rerank 4 x k (or a flat 200), keep k."""

import numpy as np

from rbench import runtime_sim
from rbench.rankers import Depth


def test_keep_reranks_the_whole_pool_and_keeps_the_first_k():
    logits = np.asarray([0.1, 0.9, 0.5, 0.7, 0.3, 0.8, 0.2, 0.6])
    r = runtime_sim.Keep("rerank", "q4-mem", Depth(8), keep=2, alias="rerank:q4-mem@n2")
    assert r.id == "rerank:q4-mem@n2"
    assert r.order(8, np.zeros(8), logits, np.zeros(8)).tolist() == [1, 5]  # the best two of all 8 scored
    assert r.order(4, np.zeros(8), logits, np.zeros(8)).tolist() == [1, 3]  # a pool of 4: the best two of those


def test_the_results_set_names_the_limit_and_scores_the_pool():
    rankers = {r.id: r for r in runtime_sim.rankers_for(["q4-mem"], (1,))}
    assert rankers["rerank:q4-mem@n5"].depth.value == 20 and rankers["rerank:q4-mem@n5"].keep == 5  # 4 x k
    assert rankers["rerank:q4-mem~200@n5"].depth.value == 200 and rankers["rerank:q4-mem~200@n5"].keep == 5
    assert rankers["first_stage@n5"].depth.value == 5  # the first stage at the same limit: its first k
    assert rankers["rerank:q4-mem@n50+x1"].depth.value == 200 and rankers["rerank:q4-mem@n50+x1"].expand == 1
    assert len(rankers) == len(runtime_sim.rankers_for(["q4-mem"], (1,)))  # ids are unique
