"""The depth a query needs: the rank of its deepest needed seed, every and first."""

import math

import numpy as np

from rbench.depth import _quantile, needed
from rbench.metrics import Gold


def test_every_needs_one_answers_spans_and_first_any_span():
    # answer A: spans in segments {3} and {7, 8}; answer B: one span in segment 1
    g = Gold(((((3, 5),), ((7, 5), (8, 5))), (((1, 5),),)))
    order = [5, 3, 9, 1, 7, 2, 8]  # ranks: 5→1, 3→2, 9→3, 1→4, 7→5, 2→6, 8→7
    every, first = needed(order, g)
    assert every == 4  # B complete at rank 4 (A needs rank 7)
    assert first == 2  # A's first span (segment 3) at rank 2


def test_unreached_evidence_is_infinite():
    g = Gold(((((42, 5),),),))
    assert all(math.isinf(x) for x in needed([1, 2, 3], g))


def test_depth_at_a_target_is_an_order_statistic():
    d = np.sort(np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 100], dtype=float))
    assert _quantile(d, 0.9) == 9 and _quantile(d, 0.95) == 100 and _quantile(d, 0.5) == 5
