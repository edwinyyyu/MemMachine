"""The experiment grid. `python -m rbench estimate <stage>` prices a stage; `run` executes it.

LongMemEval-M first (2026-11-06): Qwen3-Reranker-8B on all 470 answerable questions, each
searching its own history through MemMachine's EventMemory itself (memory.py, histories.py).

Varied:
  segment size     500 characters (EventMemory's default) in the core; 250, 1,000 and 2,000
  chunker          EventMemory's TextSegmenter (seg<N>, recursive splitting) throughout; the simple
                   chunker (cut<M>, a cut every M characters within each turn) as an option, its M
                   fitted so its measured mean segment matches seg<N>'s (sizes.py, config.CUT)
  reranking        none; the seeds (rerank, then read seeds or expand them); or each seed's
                   expand_context-3 window (expand, then rerank: what EventMemory's query does
                   whenever expand_context > 0; its default is 0)
  pool depth       fixed: 200 seeds, 100 windows (config.EVAL.pool_seeds / pool_windows; the user's
                   decision, 2026-11-10); depth.py measures how far reading each reaches and how
                   much evidence it holds, reported beside it. Every shallower vector search limit
                   replays.
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

from rbench.config import cut_for
from rbench.datasets import info
from rbench.embedding import PAID as PAID_EMBEDDER


@dataclass(frozen=True)
class Run:
    dataset: str
    scope: str
    variant: str
    chunkings: tuple[str, ...]
    models: tuple[str, ...]
    sample: str = "large"
    expand: int = 0  # > 0: score each seed's EventMemory window (expand_context = expand), not the seed
    embedder: str | None = None  # the first stage (embedding.EMBEDDERS); None: embedding.DEFAULT, the local server
    pool_seeds: int | None = None  # contexts scored per query (its deepest vector search limit); None: the
    # configured pool of its scoring position (config.EVAL.pool_seeds 200, pool_windows 100)
    note: str = ""


def default(dataset: str) -> tuple[str, str]:
    """A dataset's natural scope and the variant a user or agent would actually send."""
    ds = info(dataset)
    return ds.scope, ds.default_variant


def runs(datasets, chunkings, models, **settings) -> list[Run]:
    """One Run per dataset at its natural scope and default variant."""
    return [Run(d, *default(d), tuple(chunkings), tuple(models), **settings) for d in datasets]


LME = "longmemeval_m"
LMS = "longmemeval_s"


def lme(chunkings: tuple[str, ...], models: tuple[str, ...], expand: int, **settings) -> Run:
    """A LongMemEval-M run on all 470 answerable questions. chunkings: seg<N> (EventMemory's
    TextSegmenter at N characters) or cut<N> (the simple chunker: memory.segmenter)."""
    settings.setdefault("embedder", PAID_EMBEDDER)
    return Run(LME, "history", "question", chunkings, models, sample="all", expand=expand, **settings)


# LongMemEval-M first (the other datasets are inactive until we decide which to add back).
# expand 0 scores seeds (rerank, then read the windows or the seeds); expand 3 scores each seed's
# EventMemory window (expand, then rerank). Depths count seed tokens in both, so the two are
# compared at the same overfetch; every shallower depth and every reader budget is replayed.
# There is no dev split (decided 2026-11-06): samples take questions in a fixed hash order, so
# the pilot's 100 are part of the core's 470.
STAGES: dict[str, list[Run]] = {
    # Free and fast: the local 0.6B on 8 LongMemEval-S histories, to test the pipeline itself.
    "lme-test": [Run(LMS, "history", "question", ("seg500", cut_for(LMS, "seg500")), ("q06-local",), sample="tiny",
                     expand=e) for e in (0, 3)],
    # Free, about an hour on this machine per chunking: the local 0.6B on 100 LongMemEval-S
    # questions at the pools the paid run uses (fill depth), to see full-depth curves before
    # paying; EventMemory's segments and the size-matched simple chunker.
    "lme-s-proto": [Run(LMS, "history", "question", ("seg500", cut_for(LMS, "seg500")), ("q06-local",),
                        sample="proto", expand=e) for e in (0, 3)],
    # Free, longer (hours of the local 0.6B): the other segment lengths on the same questions.
    "lme-s-sizes": [Run(LMS, "history", "question", ("seg250", "seg1000", "seg2000"), ("q06-local",),
                        sample="proto", expand=e) for e in (0, 3)],
    # Free: the reranker's input format on the local 0.6B (the user's rule: whichever of the model
    # card's prompt and raw pairs does better on a small sample becomes the format). LongMemEval-S,
    # whose histories embed locally in minutes; the questions are LongMemEval-M's.
    "format-test": [Run("longmemeval_s", "history", "question", ("seg500",), ("q06-local", "q06-local-raw"),
                        sample="proto", pool_seeds=50, expand=0)],
    # The minimal paid run (the user, 2026-11-10): the seeds scored by both the 4B and the 8B
    # (rerank, then read the seeds or their windows at every expand_context, free), no windows
    # scored. The pilot first: the same on 100 questions.
    "lme-pilot": [replace(lme(("seg500",), ("q4", "q8"), 0), sample="proto")],
    "lme-core": [lme(("seg500",), ("q4", "q8"), 0)],
    # The 4B's instruction, tuned on questions 101-200 (held out from the pilot; part of the core,
    # so their embedding is the core's): the top 50 seeds scored with each candidate and the two
    # references (the user's request, 2026-11-11; the decision rule: METHODOLOGY section 8).
    "lme-tune": [replace(lme(("seg500",), ("q4", "q4-mem", "q4-c1", "q4-c2", "q4-c3", "q4-c4", "q4-c5", "q4-c8"), 0),
                         sample="tune", pool_seeds=50)],
    # Optional, later: the reranker scoring each seed's expand_context-3 window (expand, then rerank).
    "lme-windows": [lme(("seg500",), ("q8",), 3)],
    "lme-sizes": [lme(("seg250", "seg1000", "seg2000"), ("q8",), e) for e in (0, 3)],
    # Optional: the 0.6B as the cheap middle option.
    "lme-small": [lme(("seg500",), ("q06",), e) for e in (0, 3)],
    # Optional: the simple chunker, size-matched to EventMemory's default segments (its length
    # fitted to their measured mean: sizes.py). Reports state each chunking's measured size.
    "lme-cut": [lme((cut_for(LME, "seg500"),), ("q8",), e) for e in (0, 3)],
}


def adhoc(datasets: list[str], chunkings: list[str], models: list[str], variant: str | None = None,
          scope: str | None = None, **settings) -> list[Run]:
    """Runs for any combination, for `python -m rbench adhoc`."""
    out = runs(datasets, chunkings, models, **settings)
    return [replace(r, variant=variant or r.variant, scope=scope or r.scope) for r in out]
