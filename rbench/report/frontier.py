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
With a dev split, step 1 uses dev and steps 2-3 use test with the choice fixed. Without one
(LongMemEval: decided 2026-11-06), the choice and the measurement use the same questions,
and every bootstrap resample makes its own choice, so the interval includes the selection's
variability. The point estimate still takes a minimum over many more reranked than
unreranked configurations, so it leans optimistic for the reranker.

ratio(tau, x) is the cost of the cheapest configuration at ratio x (the reranker allowed) over
the chosen no-reranker one; 1 means the reranker does not pay.

Reranker tokens are counted two ways: as an API bills them (every pair whole) and as a
server with a prefix cache computes them (shared prompt and query once per query).
Budgets are exact: coverage is known at any budget, so a target's budget is an order statistic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rbench.report.data import Results
from rbench.report.stats import RESAMPLES

BUDGET_GRID = np.unique(np.round(np.logspace(np.log2(64), np.log2(16384), 97, base=2)))
TARGETS = np.round(np.arange(0.50, 0.991, 0.01), 2)
EVENT = "every"
RERANKED = ("rerank", "xrerank")


def min_budgets(costs: np.ndarray, targets: np.ndarray = TARGETS) -> np.ndarray:
    """For each target tau, the smallest reader budget whose coverage reaches it: coverage at B
    is the share of costs <= B, so that budget is exactly the ceil(tau N)-th smallest cost
    (inf if it is not reached)."""
    s = np.sort(np.asarray(costs, dtype=np.float64))
    k = np.ceil(np.asarray(targets) * len(s) - 1e-9).astype(int) - 1
    return s[np.clip(k, 0, len(s) - 1)] if len(s) else np.full(len(targets), np.inf)


@dataclass
class Config:
    name: tuple[str, str]  # (chunking, ranker)
    costs: dict[str, np.ndarray]  # split -> event cost per query (sorted by qid)
    tokens: dict[str, np.ndarray]  # split -> reranker tokens per query


def configurations(results: Results, model: str | None, token_field: str = "rr_tokens", event: str = EVENT) -> list[Config]:
    """Every configuration of one model (reranking seeds or windows, read alone or expanded,
    default instruction and render); model None means every no-reranker configuration."""
    d = results.select(split=None, expand=None)
    d = d[d.kind == "first_stage"] if model is None else d[d.kind.isin(RERANKED) & (d.model == model)]
    out = []
    for name, g in d.groupby(["chunking", "ranker"]):
        g = g.sort_values("qid")
        out.append(Config(name, {s: g.loc[g.split == s, event].to_numpy() for s in ("dev", "test")},
                          {s: g.loc[g.split == s, token_field].to_numpy() for s in ("dev", "test")}))
    return out


def _clusters(results: Results) -> np.ndarray:
    d = results.select(kind="first_stage")
    return d[(d.chunking == d.chunking.iloc[0]) & (d.ranker == "first_stage")].sort_values("qid").cluster.to_numpy()


def _choose(base: list[Config], rr: list[Config], split: str, index: np.ndarray | None):
    """Per target: (index of the chosen no-reranker config, index of the chosen reranked one or None)."""
    take = (lambda a: a) if index is None else (lambda a: a[index])
    b0 = np.asarray([min_budgets(take(c.costs[split])) for c in base])  # [config, target]
    b1 = np.asarray([min_budgets(take(c.costs[split])) for c in rr]) if rr else np.zeros((0, len(TARGETS)))
    tok = np.asarray([take(c.tokens[split]).mean() for c in rr])
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
            cache[(kind, k)] = min_budgets(take(configs[k].costs["test"]))
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
            out[i] = take(rr[k1].tokens["test"]).mean() / (b0 - b1) if b1 < b0 else np.inf
    return out


def break_even(results: Results, model: str, token_field: str = "rr_tokens", with_interval: bool = True,
               event: str = EVENT):
    """(r* per target, lower, upper, chosen no-reranker names, chosen reranked names).

    r* is inf where the chosen reranked configuration is not cheaper, 0 where only the
    reranker reaches the target (cheaper at any price), and nan where nothing does."""
    held_out = results.has_dev
    pick = "dev" if held_out else "test"
    base, rr = configurations(results, None, event=event), configurations(results, model, token_field, event)
    chosen = _choose(base, rr, pick, None)
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
            draws.append(_r_star(base, rr, chosen if held_out else _choose(base, rr, "test", index), index))
        draws = np.asarray(draws)
        with np.errstate(invalid="ignore"):
            lo, hi = np.nanpercentile(np.where(np.isinf(draws), 1e12, draws), [2.5, 97.5], axis=0)
    names = [(base[k0].name, rr[k1].name if k1 is not None else None) for k0, k1 in chosen]
    return point, lo, hi, [a for a, _ in names], [b for _, b in names]


def cost_ratio(results: Results, model: str, ratios: np.ndarray, token_field: str = "rr_tokens",
               event: str = EVENT) -> np.ndarray:
    """[target, ratio] -> cost (on test) of the cheapest configuration, the reranker allowed,
    over the chosen no-reranker one (nan where the first stage cannot reach the target).
    Chosen on dev when there is one, else on the same questions."""
    pick = "dev" if results.has_dev else "test"
    base, rr = configurations(results, None, event=event), configurations(results, model, token_field, event)
    out = np.full((len(TARGETS), len(ratios)), np.nan)
    budgets = {id(c): (min_budgets(c.costs[pick]), min_budgets(c.costs["test"]), c.tokens[pick].mean(), c.tokens["test"].mean())
               for c in base + rr}
    for i in range(len(TARGETS)):
        c0 = min(base, key=lambda c: budgets[id(c)][0][i])
        b0_pick, b0_test = budgets[id(c0)][0][i], budgets[id(c0)][1][i]
        if not np.isfinite(b0_test):
            continue  # nothing to compare against
        for j, x in enumerate(ratios):
            best, best_cost = c0, x * b0_pick  # inf when the first stage misses the target
            for c in rr:
                b = budgets[id(c)][0][i]
                if np.isfinite(b) and x * b + budgets[id(c)][2] < best_cost:
                    best, best_cost = c, x * b + budgets[id(c)][2]
            b_test, r_test = budgets[id(best)][1][i], (budgets[id(best)][3] if best is not c0 else 0.0)
            out[i, j] = (x * b_test + r_test) / (x * b0_test) if np.isfinite(b_test) else np.nan
    return out
