"""Paired bootstrap intervals that resample documents.

Queries that share a document (or a dialogue, or a repository snapshot) are correlated:
they share its chunks and its ranking, so resampling queries independently would
understate the uncertainty. Every interval resamples clusters with replacement and takes
all of a cluster's queries along; the cluster is set per dataset (datasets.DatasetInfo.cluster).
"""

from __future__ import annotations

import numpy as np

RESAMPLES = 1000
MIN_CLUSTERS = 20  # fewer and a percentile interval is not meaningful


def interval(values: np.ndarray, clusters: np.ndarray, seed: int = 0) -> tuple[float, float]:
    """95% interval of the mean of `values` (e.g. paired differences) over clusters."""
    labels, inverse = np.unique(clusters, return_inverse=True)
    if len(labels) < MIN_CLUSTERS:
        return float("nan"), float("nan")
    sums = np.bincount(inverse, weights=values.astype(float), minlength=len(labels))
    counts = np.bincount(inverse, minlength=len(labels))
    draws = np.random.default_rng(seed).integers(0, len(labels), size=(RESAMPLES, len(labels)))
    means = sums[draws].sum(1) / counts[draws].sum(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def delta(values: np.ndarray, clusters: np.ndarray) -> str:
    lo, hi = interval(values, clusters)
    if np.isnan(lo):
        return f"{values.mean():+.3f} [n/a: {len(np.unique(clusters))} clusters]"
    return f"{values.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]"
