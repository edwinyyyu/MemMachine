"""Named query samples. Queries are taken in a fixed hash order, so smaller samples nest in
larger ones and models scored on different samples are compared on the shared queries.

  proto  0 dev / 100 test   local prototype
  panel  0 dev / 300 test   the large-model comparison panel
  large  100 dev / 400 test
  main   150 dev / 500 test the full grid for the cheap model
  all    every query

Datasets with fewer queries than a sample asks for contribute all they have.
"""

from __future__ import annotations

import hashlib

from rbench.schema import queries

SAMPLES = {"tiny": (0, 8), "proto": (0, 100), "panel": (0, 300), "large": (100, 400), "main": (150, 500),
           "all": (10**9, 10**9)}
ORDER = ["tiny", "proto", "panel", "large", "main", "all"]  # each nests in the next


def sample(dataset: str, name: str) -> list[int]:
    """Indices into queries(dataset), in file order."""
    dev, test = SAMPLES[name]
    rows = queries(dataset)
    order = sorted(range(len(rows)), key=lambda i: hashlib.sha1(f"sample:{rows[i]['qid']}".encode()).hexdigest())
    keep = [i for i in order if rows[i]["split"] == "dev"][:dev] + [i for i in order if rows[i]["split"] == "test"][:test]
    return sorted(keep)
