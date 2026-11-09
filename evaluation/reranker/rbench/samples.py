"""Named query samples. Queries are taken in a fixed hash order, so smaller samples nest in
larger ones and models scored on different samples are compared on the shared queries. There
is no dev/test split (decided 2026-11-06): every sampled query is evaluated.

  tiny   8     tests the pipeline
  proto  100   the pilot
  panel  300
  large  500
  all    every query

Datasets with fewer queries than a sample asks for contribute all they have.
"""

from __future__ import annotations

import hashlib

from rbench.schema import queries

SAMPLES = {"tiny": 8, "proto": 100, "panel": 300, "large": 500, "all": 10**9}
ORDER = list(SAMPLES)  # each nests in the next


def sample_order(dataset: str) -> list[int]:
    """Every query index of a dataset in the fixed hash order samples take them in."""
    rows = queries(dataset)
    return sorted(range(len(rows)), key=lambda i: hashlib.sha1(f"sample:{rows[i]['qid']}".encode()).hexdigest())


def sample(dataset: str, name: str) -> list[int]:
    """Indices into queries(dataset), in file order."""
    return sorted(sample_order(dataset)[: SAMPLES[name]])
