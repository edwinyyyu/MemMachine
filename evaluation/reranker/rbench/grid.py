"""The experiment grid. `python -m rbench estimate <stage>` prices a stage; `run` executes it.

LongMemEval-M first (2026-11-06): Qwen3-Reranker-8B on all 470 answerable questions, each
searching its own history through MemMachine's EventMemory itself (memory.py, histories.py).

Varied:
  segment size     500 characters (EventMemory's default) in the core; 250, 1,000 and 2,000
  chunker          EventMemory's TextSegmenter (seg<N>, recursive splitting) throughout; the simple
                   chunker (cut<N>, a cut every N characters within each turn) as an option
  reranking        none; the seeds (rerank, then read seeds or expand them); or each seed's
                   expand_context-3 window (expand, then rerank: what EventMemory's query does
                   whenever expand_context > 0; its default is 0)
  pool depth       the top 50 seeds in the core (vector search limits 5-50, EventMemory's default
                   20 among them, at every segment length); the top 200 in the depth stage.
                   Seed counts, not tokens: the vector search limit is a count, and equal
                   reranker input is compared through each configuration's measured tokens
Free (replayed from cached scores): every shallower overfetch, top-k, reader budget, and the
first stage and rerank-then-expand at expand_context 1, 2, 3 and 6 (simulate_histories.py).
A reranked configuration's vector search limit is its depth (EventMemory reranks every seed
it retrieves); the first stage's is whatever the budget reads. Windows are scored (paid) at
expand_context 3 only. The reranker is the 8B; the 0.6B is an optional cheap middle option.
Scores are cached by content, so later combinations pay only for new pairs
(`python -m rbench adhoc ... --expand E`).

A Run reranks one (dataset, scope, variant, sample) at some chunkings with some models: each
query's top `pool_seeds` seeds (its deepest vector search limit). expand > 0 means the reranker
scores each seed's window (expand, then rerank), as EventMemory's query does. Every dataset
runs through EventMemory itself (partitions.py); the inactive datasets are run with
`python -m rbench adhoc`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from rbench.datasets import info


@dataclass(frozen=True)
class Run:
    dataset: str
    scope: str
    variant: str
    chunkings: tuple[str, ...]
    models: tuple[str, ...]
    sample: str = "large"
    expand: int = 0  # > 0: score each seed's EventMemory window (expand_context = expand), not the seed
    pool_seeds: int = 50  # score each query's top pool_seeds seeds (its deepest vector search limit)
    note: str = ""


def default(dataset: str) -> tuple[str, str]:
    """A dataset's natural scope and the variant a user or agent would actually send."""
    ds = info(dataset)
    return ds.scope, ds.default_variant


def runs(datasets, chunkings, models, **settings) -> list[Run]:
    """One Run per dataset at its natural scope and default variant."""
    return [Run(d, *default(d), tuple(chunkings), tuple(models), **settings) for d in datasets]


LME = "longmemeval_m"


def lme(chunkings: tuple[str, ...], models: tuple[str, ...], expand: int, **settings) -> Run:
    """A LongMemEval-M run on all 470 answerable questions. chunkings: seg<N> (EventMemory's
    TextSegmenter at N characters) or cut<N> (the simple chunker: memory.segmenter)."""
    return Run(LME, "history", "question", chunkings, models, sample="all", expand=expand, **settings)


# LongMemEval-M first (the other datasets are inactive until we decide which to add back).
# expand 0 scores seeds (rerank, then read the windows or the seeds); expand 3 scores each seed's
# EventMemory window (expand, then rerank). Depths count seed tokens in both, so the two are
# compared at the same overfetch; every shallower depth and every reader budget is replayed.
# There is no dev split (decided 2026-11-06): samples take questions in a fixed hash order, so
# the pilot's 100 are part of the core's 470.
STAGES: dict[str, list[Run]] = {
    # Free and fast: the local 0.6B on 8 LongMemEval-S histories, to test the pipeline itself.
    "lme-test": [Run("longmemeval_s", "history", "question", ("seg500", "cut500"), ("q06-local",), sample="tiny",
                     pool_seeds=30, expand=e) for e in (0, 3)],
    # Free: the reranker's input format on the local 0.6B (the user's rule: whichever of the model
    # card's prompt and raw pairs does better on a small sample becomes the format). LongMemEval-S,
    # whose histories embed locally in minutes; the questions are LongMemEval-M's.
    "format-test": [Run("longmemeval_s", "history", "question", ("seg500",), ("q06-local", "q06-local-raw"),
                        sample="proto", pool_seeds=50, expand=0)],
    # Paid, cheap: every planned configuration on 100 questions first.
    "lme-pilot": [replace(lme(("seg500",), ("q8",), e), sample="proto") for e in (0, 3)],
    "lme-core": [lme(("seg500",), ("q8",), e) for e in (0, 3)],
    "lme-sizes": [lme(("seg250", "seg1000", "seg2000"), ("q8",), e) for e in (0, 3)],
    "lme-depth": [lme(("seg500",), ("q8",), e, pool_seeds=200) for e in (0, 3)],
    # Optional: the 0.6B as the cheap middle option.
    "lme-small": [lme(("seg500",), ("q06",), e) for e in (0, 3)],
    # Optional: the simple chunker at EventMemory's default limit. Reports state each chunking's
    # measured size, and comparisons that pair by size use the measured one.
    "lme-cut": [lme(("cut500",), ("q8",), e) for e in (0, 3)],
}


def adhoc(datasets: list[str], chunkings: list[str], models: list[str], variant: str | None = None,
          scope: str | None = None, **settings) -> list[Run]:
    """Runs for any combination, for `python -m rbench adhoc`."""
    out = runs(datasets, chunkings, models, **settings)
    return [replace(r, variant=variant or r.variant, scope=scope or r.scope) for r in out]
