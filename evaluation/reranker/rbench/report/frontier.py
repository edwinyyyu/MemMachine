"""When is a reranker worth its cost? Break-even price ratios and relative cost.

Cost per query = reader tokens B x p_llm + reranker tokens R x p_rr. Scaling both prices by
the same factor changes no decision, so everything depends on the ratio x = p_llm / p_rr
(for example a $3/M reader with the 8B at $0.05/M is x = 60).

Candidates. No reranker: the first stage at every segment length and expansion (seeds read
alone or with their EventMemory windows). Reranked: every depth of reranking the seeds (read
alone or expanded) and of reranking their windows (expand, then rerank: what EventMemory does
with expand_context > 0), at every segment length.

For a coverage target tau (every-gold by default: all of a question's evidence is needed):
  1. Pick the no-reranker configuration whose reader budget for tau is smallest (B0), and the
     reranked configuration with the smallest R / (B0 - B): the one that pays off at the
     lowest ratio.
  2. r* = R / (B0 - B) for those two, infinite if the reranked one is not cheaper in reader
     tokens.
  3. Bootstrap question clusters for an interval on r*.
There is no dev split (decided 2026-11-06): the choice and the measurement use the same
questions, and every bootstrap resample makes its own choice, so the interval includes the
selection's variability. The point estimate still takes a minimum over many more reranked than
unreranked configurations, so it leans optimistic for the reranker.

cost_curves gives, per LLM price and target, the total cost with the reranker (its cheapest
configuration) and without it (the cheapest first-stage configuration): the reranker saves
money where the first is lower, which happens exactly above the break-even price.

Reranker tokens are counted two ways: as an API bills them (every pair whole) and as a
server with a prefix cache computes them (shared prompt and query once per query).
Budgets are exact: coverage is known at any budget, so a target's budget is an order statistic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rbench.report.data import Results
from rbench.report import readings as rd
from rbench.report.stats import RESAMPLES

BUDGET_GRID = np.unique(np.round(np.geomspace(50, 16000, 121)))  # 1/14-octave steps
TARGETS = np.round(np.arange(0.10, 0.991, 0.01), 2)  # every-gold coverage at realistic budgets can be well under 0.5
EVENT = "every"
RERANKED = ("rerank", "xrerank")


def min_budgets(costs: np.ndarray, targets: np.ndarray = TARGETS) -> np.ndarray:
    """For each target tau, the smallest reader budget whose coverage reaches it (inf if it is
    not reached): an order statistic of the costs (readings.min_budgets; a share reading's
    costs are per evidence turn)."""
    c = costs if np.asarray(costs).dtype == object else np.asarray(costs, dtype=np.float64)
    return rd.min_budgets(c, targets)


@dataclass
class Config:
    name: tuple[str, str]  # (chunking, ranker)
    costs: np.ndarray  # the reading's costs per query (sorted by qid; readings.costs)
    tokens: np.ndarray  # reranker tokens per query


def setting(results: Results) -> tuple[str, int]:
    """The (chunking, expand_context) the cost comparison holds fixed (config.EVAL.cost_*):
    EventMemory's default segment length unless set (or the first simulated chunking)."""
    from rbench import config
    from rbench.report.data import default_segment

    c = config.EVAL.cost_chunking or default_segment()
    return (c if c in results.chunkings else results.chunkings[0]), config.EVAL.cost_expand


def configurations(results: Results, model: str | None, token_field: str = "rr_tokens", event: str = EVENT,
                   fixed: tuple[str, int] | None = None, limit: int | None = None) -> list[Config]:
    """Every configuration of one model (model None: every no-reranker configuration), at one
    (chunking, expand_context) when `fixed`, so only the vector search limit varies; else
    across every chunking and expansion. limit (with fixed): the reranked side at that one
    vector search limit, its seeds scored (one configuration, nothing re-chosen per target)."""
    d = results.select(expand=None)
    if fixed is not None:
        d = d[(d.chunking == fixed[0]) & (d.expand == fixed[1])]
    d = d[d.kind == "first_stage"] if model is None else d[d.kind.isin(RERANKED) & (d.model == model)]
    if model is not None and limit is not None:
        d = d[(d.kind == "rerank") & (d.depth == f"n{limit}")]
    out = []
    for name, g in d.groupby(["chunking", "ranker"]):
        g = g.sort_values("qid")
        out.append(Config(name, rd.costs(g, event), g[token_field].to_numpy()))
    return out


def cost_limits(results: Results, model: str, event: str = EVENT) -> list[tuple[int, str]]:
    """The vector search limits the cost comparison is drawn at, each held fixed for every
    target and price (the user's decision, 2026-11-09: no per-target switching): EventMemory's
    default 20, the best L chosen once overall (best.overall_limit), and the pool; (L, what
    it is), each L once."""
    from rbench.report import best

    c, e = setting(results)
    depths = [int(d[1:]) for d in results.depths_at(c, model, "rerank")]
    if not depths:
        return []
    picks = [(20, "EventMemory's default"), (best.overall_limit(results, model, c, event, e), "the best overall"),
             (depths[-1], "the pool")]
    out: dict[int, list[str]] = {}
    for L, what in picks:
        if L is not None and L in depths:
            out.setdefault(L, []).append(what)
    return [(L, " = ".join(names)) for L, names in out.items()]


def _clusters(results: Results) -> np.ndarray:
    d = results.select(kind="first_stage")
    return d[(d.chunking == d.chunking.iloc[0]) & (d.ranker == "first_stage")].sort_values("qid").cluster.to_numpy()


def _choose(base: list[Config], rr: list[Config], index: np.ndarray | None):
    """Per target: (index of the chosen no-reranker config, index of the chosen reranked one or None)."""
    take = (lambda a: a) if index is None else (lambda a: a[index])
    b0 = np.asarray([min_budgets(take(c.costs)) for c in base])  # [config, target]
    b1 = np.asarray([min_budgets(take(c.costs)) for c in rr]) if rr else np.zeros((0, len(TARGETS)))
    tok = np.asarray([take(c.tokens).mean() for c in rr])
    out = []
    for i in range(len(TARGETS)):
        k0 = int(np.argmin(b0[:, i]))
        reach = np.isfinite(b1[:, i])
        if np.isfinite(b0[k0, i]):  # the reranked configuration that pays off at the lowest ratio
            ok = reach & (b1[:, i] < b0[k0, i])
            key = np.where(ok, tok / np.maximum(b0[k0, i] - b1[:, i], 1e-12), np.inf)
        else:  # the first stage alone misses the target: the cheapest reranked configuration that reaches it
            ok = reach
            key = np.where(ok, tok, np.inf)
        out.append((k0, int(np.argmin(key)) if ok.any() else None))
    return out


def _r_star(base: list[Config], rr: list[Config], chosen, index: np.ndarray | None) -> np.ndarray:
    take = (lambda a: a) if index is None else (lambda a: a[index])
    out = np.full(len(TARGETS), np.nan)
    cache: dict[tuple[str, int], np.ndarray] = {}

    def budgets(kind: str, k: int, configs: list[Config]) -> np.ndarray:
        if (kind, k) not in cache:
            cache[(kind, k)] = min_budgets(take(configs[k].costs))
        return cache[(kind, k)]

    for i, (k0, k1) in enumerate(chosen):
        b0 = budgets("base", k0, base)[i]
        if k1 is None:
            out[i] = np.inf if np.isfinite(b0) else np.nan
            continue
        b1 = budgets("rr", k1, rr)[i]
        if not np.isfinite(b0):
            out[i] = 0.0 if np.isfinite(b1) else np.nan  # 0: only the reranker reaches the target
        else:
            out[i] = take(rr[k1].tokens).mean() / (b0 - b1) if b1 < b0 else np.inf
    return out


def break_even(results: Results, model: str, token_field: str = "rr_tokens", with_interval: bool = True,
               event: str = EVENT, fixed: tuple[str, int] | None = None, limit: int | None = None):
    """(r* per target, lower, upper, chosen no-reranker names, chosen reranked names).

    r* is inf where the chosen reranked configuration is not cheaper, 0 where only the
    reranker reaches the target (cheaper at any price), and nan where nothing does."""
    base, rr = (configurations(results, None, event=event, fixed=fixed),
                configurations(results, model, token_field, event, fixed, limit))
    chosen = _choose(base, rr, None)
    point = _r_star(base, rr, chosen, None)
    lo = hi = np.full(len(TARGETS), np.nan)
    if with_interval:
        clusters = _clusters(results)
        labels, inverse = np.unique(clusters, return_inverse=True)
        members = [np.flatnonzero(inverse == k) for k in range(len(labels))]
        rng = np.random.default_rng(0)
        draws = []
        for _ in range(RESAMPLES):
            index = np.concatenate([members[k] for k in rng.integers(0, len(labels), len(labels))])
            draws.append(_r_star(base, rr, _choose(base, rr, index), index))
        draws = np.asarray(draws)
        with np.errstate(invalid="ignore"):
            # order statistics, not interpolated: a percentile between a price and the "never"
            # stand-in (1e12) would be a huge finite price, drawn as a stray line
            lo, hi = np.nanpercentile(np.where(np.isinf(draws), 1e12, draws), [2.5, 97.5], axis=0, method="nearest")
    names = [(base[k0].name, rr[k1].name if k1 is not None else None) for k0, k1 in chosen]
    return point, lo, hi, [a for a, _ in names], [b for _, b in names]


def cost_curves(results: Results, model: str, llm_prices: list[float], rr_price: float, token_field: str = "rr_tokens",
                event: str = EVENT, fixed: tuple[str, int] | None = None,
                limit: int | None = None) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Total cost in $ per 1,000 questions to reach each coverage target, without and with the
    reranker: {llm price -> (no reranker [target], reranker [target])}, prices in $ per million
    tokens. Each side is its cheapest configuration at that price (segment length, expansion and,
    for the reranker, depth), chosen and measured on the same questions; nan where no
    configuration of that side reaches the target."""
    base, rr = (configurations(results, None, event=event, fixed=fixed),
                configurations(results, model, token_field, event, fixed, limit))
    b0 = np.asarray([min_budgets(c.costs) for c in base])
    b1 = np.asarray([min_budgets(c.costs) for c in rr]) if rr else np.zeros((0, len(TARGETS)))
    tok = np.asarray([c.tokens.mean() for c in rr])
    out = {}
    for p in llm_prices:
        no, yes = np.full(len(TARGETS), np.nan), np.full(len(TARGETS), np.nan)
        for i in range(len(TARGETS)):
            cheapest = float(np.min(b0[:, i]))
            if np.isfinite(cheapest):
                no[i] = p * cheapest / 1e3  # $/M tokens x tokens / 1e6 x 1,000 questions
            if rr:
                total = p * b1[:, i] + rr_price * tok
                best = float(np.min(np.where(np.isfinite(total), total, np.inf)))
                if np.isfinite(best):
                    yes[i] = best / 1e3
        out[p] = (no, yes)
    return out


def saved_interval(results: Results, model: str, llm_prices: list[float], rr_price: float, token_field: str = "rr_tokens",
                   event: str = EVENT, fixed: tuple[str, int] | None = None,
                   limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The 95% bootstrap interval of the % of total cost reranking saves, per (target, price):
    clusters resampled, each side's cheapest configuration re-chosen in every resample (as
    break_even does). +100 where only the reranker reaches the target in a resample, -inf where
    only the first stage does, nan where neither."""
    base, rr = (configurations(results, None, event=event, fixed=fixed),
                configurations(results, model, token_field, event, fixed, limit))
    if not base or not rr:  # nothing to compare at this setting (e.g. the model is scored at other chunkings)
        empty = np.full((len(TARGETS), len(llm_prices)), np.nan)
        return empty, empty.copy()
    labels, inverse = np.unique(_clusters(results), return_inverse=True)
    members = [np.flatnonzero(inverse == k) for k in range(len(labels))]
    rng = np.random.default_rng(0)
    p = np.asarray(llm_prices, dtype=np.float64)
    draws = np.empty((RESAMPLES, len(TARGETS), len(p)))
    for i in range(RESAMPLES):
        index = np.concatenate([members[k] for k in rng.integers(0, len(labels), len(labels))])
        b0 = np.min([min_budgets(c.costs[index]) for c in base], axis=0)  # [target]
        b1 = np.asarray([min_budgets(c.costs[index]) for c in rr])  # [config, target]
        tok = np.asarray([c.tokens[index].mean() for c in rr])
        no = b0[:, None] * p[None, :]
        with np.errstate(invalid="ignore"):
            yes = np.min(b1[:, :, None] * p[None, None, :] + rr_price * tok[:, None, None], axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            saved = 100 * (1 - yes / no)
        saved = np.where(np.isinf(no) & np.isfinite(yes), 100.0, saved)
        saved = np.where(np.isfinite(no) & np.isinf(yes), -np.inf, saved)
        draws[i] = np.where(np.isinf(no) & np.isinf(yes), np.nan, saved)
    with np.errstate(invalid="ignore"):
        lo, hi = np.nanpercentile(draws, [2.5, 97.5], axis=0, method="nearest")
    return lo, hi
