"""Named query samples. Queries are taken in a fixed hash order, so smaller samples nest in
larger ones and models scored on different samples are compared on the shared queries. There
is no dev/test split (decided 2026-11-06): every sampled query is evaluated.

  tiny   8     tests the pipeline
  proto  100   the pilot
  panel  300
  large  500
  all    every query
  tune   the 101st to 200th: held out from the pilot, for tuning the reranker's instruction
         (2026-11-11); inside panel, large and all, but not nested with proto

Datasets with fewer queries than a sample asks for contribute all they have.
"""

from __future__ import annotations

import hashlib

from rbench.schema import queries

SAMPLES: dict[str, int | tuple[int, int]] = {"tiny": 8, "proto": 100, "panel": 300, "large": 500, "all": 10**9,
                                             "tune": (100, 200)}
ORDER = [k for k, v in SAMPLES.items() if isinstance(v, int)]  # the nested ones: each in the next


def sample_order(dataset: str) -> list[int]:
    """Every query index of a dataset in the fixed hash order samples take them in."""
    rows = queries(dataset)
    return sorted(range(len(rows)), key=lambda i: hashlib.sha1(f"sample:{rows[i]['qid']}".encode()).hexdigest())


def sample(dataset: str, name: str) -> list[int]:
    """Indices into queries(dataset), in file order: the first N of the hash order, or a slice of it."""
    n = SAMPLES[name]
    start, stop = (0, n) if isinstance(n, int) else n
    return sorted(sample_order(dataset)[start:stop])
