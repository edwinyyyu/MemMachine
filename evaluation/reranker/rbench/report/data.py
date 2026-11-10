"""Loading simulated results and the views every report section uses.

A configuration is (chunking, ranker). Its per-query rows carry the
event costs, so coverage at any reader budget B is the share of rows with cost <= B: the
budget is a continuous knob, not a fixed grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rbench.paths import SIM
from rbench.rerankers import RERANKERS

CONFIG = ["chunking", "ranker"]


@dataclass
class Results:
    embedder: str
    dataset: str
    scope: str
    variant: str
    sample: str
    df: pd.DataFrame

    @property
    def expansions(self) -> list[int]:
        """expand_context values simulated (above 0)."""
        return sorted(int(e) for e in self.df.expand.unique() if e > 0)

    @property
    def chunkings(self) -> list[str]:
        return sorted(self.df.chunking.unique(), key=lambda c: (int("".join(ch for ch in c if ch.isdigit())), c))

    @property
    def models(self) -> list[str]:
        found = set(self.df.loc[self.df.kind == "rerank", "model"].dropna())
        return [m for m in RERANKERS if m in found]

    def rows(self, chunking: str, ranker: str) -> pd.DataFrame:
        """One configuration's rows, sorted by qid so configurations align query by query."""
        d = self.df
        return d[(d.chunking == chunking) & (d.ranker == ranker)].sort_values("qid")

    def select(self, kind: str | None = None, model: str | None = None,
               expand: int | None = 0) -> pd.DataFrame:
        """Rows of one kind and model, without expansion unless asked otherwise."""
        d = self.df
        if expand is not None:
            d = d[d.expand == expand]
        if kind is not None:
            d = d[d.kind == kind]
        if model is not None:
            d = d[d.model == model]
        return d

    def models_at(self, chunking: str) -> list[str]:
        """Models with plain reranked rows at this chunking (sweeps score only some models)."""
        d = self.select(kind="rerank")
        found = set(d.loc[d.chunking == chunking, "model"])
        return [m for m in RERANKERS if m in found]

    def common_depth(self, chunking: str, models: list[str] | None = None) -> str | None:
        """The deepest vector search limit every model scored at this chunking was scored to
        (models without scores there are left out rather than hiding the others)."""
        d = self.select(kind="rerank")
        d = d[d.chunking == chunking]
        shared = None
        for m in [m for m in (models or self.models_at(chunking)) if m in self.models_at(chunking)]:
            depths = set(d.loc[d.model == m, "depth"].dropna())
            shared = depths if shared is None else shared & depths
        if not shared:
            return None
        return max(shared, key=lambda x: int(x[1:]))

    def limits(self, chunking: str, wanted: tuple[int, ...] | None = None) -> list[str]:
        """Vector search limits to show for a chunking, as depth labels ("n20"): the wanted ones
        some model there was scored to (all of them when no model was), plus each model's
        deepest pool, of seeds and of windows (a model scored less deep is simply absent there).
        wanted defaults to config.EVAL.limits (EventMemory's default limit is 20)."""
        from rbench import config

        wanted = config.EVAL.limits if wanted is None else wanted
        models = self.models_at(chunking)
        if not models:
            return [f"n{v}" for v in sorted(wanted)]
        have: set[str] = set()
        out: set[str] = set()
        for m in models:
            for kind in ("rerank", "xrerank"):
                depths = self.depths_at(chunking, m, kind)
                have |= set(depths)
                if depths:
                    out.add(depths[-1])
        out |= {f"n{v}" for v in wanted if f"n{v}" in have}
        return sorted(out, key=lambda d: int(d[1:]))

    def pool_depth(self, chunking: str, model: str, kind: str = "rerank") -> str | None:
        """A model's deepest pool at a chunking for one scoring position (rerank: the seeds;
        xrerank: the windows, scored to their own, shallower pool)."""
        depths = self.depths_at(chunking, model, kind)
        return depths[-1] if depths else None

    def depths_at(self, chunking: str, model: str, kind: str = "rerank") -> list[str]:
        """Count depths (vector search limits) a model's reranked rows (kind rerank: seeds;
        xrerank: windows) exist at, shallowest first."""
        d = self.select(kind=kind, model=model, expand=None)
        return sorted({x for x in d.loc[d.chunking == chunking, "depth"] if x.startswith("n")}, key=lambda x: int(x[1:]))


def load(embedder: str, dataset: str, scope: str, variant: str, sample: str) -> Results:
    base = SIM / embedder / dataset / scope / variant / sample
    frames = []
    for path in sorted(base.glob("*.parquet")):
        if "__" in path.stem:  # written before the instruction and render sweep was removed: superseded
            continue
        frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError(f"no simulated results under {base}")
    return Results(embedder, dataset, scope, variant, sample, pd.concat(frames, ignore_index=True))


def coverage(costs: np.ndarray, budgets: np.ndarray) -> np.ndarray:
    """Share of queries whose cost is within each budget."""
    s = np.sort(costs)
    return np.searchsorted(s, budgets, "right") / len(s)


def price(model: str | None) -> float:
    """$ per million reranker tokens (a local model: its stand-in's, rerankers.Reranker.priced_as)."""
    return 0.0 if model is None else RERANKERS[model].price


def query_labels(r: Results, chunking: str, ranker: str = "first_stage") -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """A configuration's rows (default: the first stage) and, aligned with them, each query's type labels
    (features.py, and the dataset's own question types where it has them)."""
    from rbench.features import query_features
    from rbench.samples import sample
    from rbench.schema import queries

    idx = sample(r.dataset, r.sample)
    position = {queries(r.dataset)[i]["qid"]: j for j, i in enumerate(idx)}
    base = r.rows(chunking, ranker)
    at = np.asarray([position[q] for q in base.qid])
    labels = {k: np.asarray(v)[at] for k, v in query_features(r.dataset, r.variant, idx).items()}
    if "question_type" in base:  # a dataset's own query categories (LongMemEval's question types)
        labels["question_type"] = base["question_type"].to_numpy()
    return base, labels


def default_segment() -> str:
    """The chunking name of EventMemory's default segment length (TextSegmenter's default)."""
    import inspect

    import rbench.memory  # noqa: F401  (imports MemMachine with its optional-agent logger silenced)

    from memmachine_server.episodic_memory.event_memory.segmenter.text_segmenter import TextSegmenter  # noqa: I001

    return f"seg{inspect.signature(TextSegmenter).parameters['max_chunk_length'].default}"


def _measured(r: Results, chunking_name: str, column: str, expand: int = 0) -> float | None:
    """A per-question measured column's mean over questions (None if not recorded)."""
    d = r.df[(r.df.chunking == chunking_name) & (r.df.expand == expand)]
    if d.empty or column not in d or d[column].isna().all():
        return None
    return float(d.groupby("qid")[column].first().mean())


def segment_size(r: Results, chunking_name: str) -> tuple[float, float] | None:
    """A chunking's measured mean segment size over the partitions searched (characters,
    tokens), averaged over queries; None for results simulated before sizes were recorded."""
    chars, tokens = _measured(r, chunking_name, "seg_chars"), _measured(r, chunking_name, "seg_tokens")
    return None if chars is None or tokens is None else (chars, tokens)


def unit_tokens(r: Results, chunking_name: str, expand: int = 0) -> float | None:
    """Measured mean segment-text tokens one retrieved seed's context holds at an expand_context
    (the seed alone at 0)."""
    return _measured(r, chunking_name, "unit_tokens", expand)


def size_label(r: Results, chunking_name: str, expand: int = 0) -> str:
    """What one unit of a configuration measures: the chunker's limit (seg: at most N
    characters; cut: every N) with the measured mean segment of the partitions searched and of
    the seeds retrieved, or the measured mean window when expanded."""
    if expand:
        tokens = unit_tokens(r, chunking_name, expand)
        return "not measured" if tokens is None else f"window: {tokens:,.0f} tokens"
    size = segment_size(r, chunking_name)
    limit = f"≤{chunking_name[3:]} chars" if chunking_name.startswith("seg") else f"cut every {chunking_name[3:]} chars"
    if size is None:
        return f"{limit} (not measured)"
    seeds = unit_tokens(r, chunking_name)
    return (f"{limit}; searched {size[0]:,.0f} chars, {size[1]:,.0f} tokens"
            + (f"; seeds {seeds:,.0f} tokens" if seeds else ""))
