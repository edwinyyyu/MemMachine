"""Budget first: the largest k whose results fit, found over every k (tokens need not grow with k)."""

import numpy as np

from rbench import runtime_fit_report as rfr


def test_the_largest_fitting_k_is_found_even_when_tokens_dip():
    tokens = np.asarray([[100.0, 250.0, 220.0, 400.0], [50.0, np.nan, np.nan, np.nan]])  # row 1: a pool of one
    assert rfr.largest_fitting(tokens, 230).tolist() == [2, 0]  # k = 3 fits though k = 2 does not
    assert rfr.largest_fitting(tokens, 40).tolist() == [-1, -1]
    values = np.asarray([[0.0, 1.0, 0.5, 1.0], [1.0, 0.0, 0.0, 0.0]])
    assert rfr.take(values, rfr.largest_fitting(tokens, 230)).tolist() == [0.5, 1.0]
    assert rfr.take(values, rfr.largest_fitting(tokens, 40)).tolist() == [0.0, 0.0]  # nothing fits: nothing read
