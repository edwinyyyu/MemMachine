"""Budget first, exactly: the charts and tables of the every-k table (runtime_fit).

As the core report's matched-token measures, every question gets the same reader budget B: it
reads the most results that fit in B, whole, from the pool its k needs:
  first   the first stage's first k seeds, the largest k whose k results fit in B (k <= 400)
  4k      the best k of the first 4 x k seeds, the largest such k (k <= 50: past the tokens of 50
          results the runtime pool stops growing, its 200 seeds all scored)
  200     the best k of a flat 200, the largest such k (k <= 200)
For the same money the side with a reranker pays its pairs first and buys the largest k whose
pairs and reader tokens fit; the side without one buys the most results the money does.

The largest fitting k is found per question by scanning its k (a k's tokens need not grow with k
when the pool changes with it), so every budget and every k is exact, not interpolated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rbench import config
from rbench.report import readings as rd
from rbench.report.data import price
from rbench.report.stats import MIN_CLUSTERS, RESAMPLES, delta
from rbench.runtime import KS, POOL_COLOR, ROLE
from rbench.runtime_fit import EVENTS

BUDGETS = np.unique(np.round(np.geomspace(250, 16000, 241)))  # the token axis: exact at every point drawn
POLICY_TEXT = {"first": "no reranker: the most first-stage results that fit",
               "4k": "the runtime pool: the most results that fit, the best of 4 × that many (at most 50 results: "
                     "only 200 seeds per question were scored)",
               "200": "a flat 200: the most of its best results that fit"}
CAP_NOTE = ("Why the orange line goes flat: only each question's first 200 seeds were scored, so the runtime pool "
            "(4 × k) stops at k = 50 results. Shaded: past the mean tokens of those 50 results, where it cannot grow; "
            "its line is flat there by construction (scoring deeper seeds would extend it), not because reranking "
            "stops helping.")


def cap_tokens(fit: Fit, m: str, e: int) -> float:
    """Mean reader tokens of the runtime pool's 50 results: past them it cannot grow."""
    return float(np.nanmean(fit.arrays("4k", m, e, "T")[:, -1]))


def shade_cap(ax, start: float, end: float) -> None:
    """The region past the runtime pool's 50 results, shaded and labelled where it is."""
    from rbench.report import figures as fg

    if start >= end:
        return
    ax.axvspan(start, end, color=fg.GRID, alpha=0.45, linewidth=0, zorder=0)
    ax.annotate("runtime pool\nat its 50 results", xy=(start, 0.03), xycoords=("data", "axes fraction"),
                xytext=(3, 0), textcoords="offset points", fontsize=6.5, color=fg.INK2, va="bottom")


class Fit:
    """The every-k table as per-question arrays [question, k] (questions in qid order)."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.qids = np.sort(df.qid.unique())
        first = df[(df.policy == "first") & (df.expand == 0) & (df.k == 1)].set_index("qid").loc[self.qids]
        self.clusters = first.cluster.to_numpy()
        self.types = first.question_type.fillna("other").to_numpy()
        self._cache: dict = {}

    def arrays(self, policy: str, model: str | None, e: int, column: str) -> np.ndarray:
        key = (policy, model, e, column)
        if key not in self._cache:
            d = self.df
            x = d[(d.policy == policy) & (d.expand == e) & ((d.model == model) if model else d.model.isna())]
            values = x.pivot(index="qid", columns="k", values=column).reindex(self.qids)
            self._cache[key] = values.to_numpy(dtype=np.float64)
        return self._cache[key]

    def value(self, policy: str, model: str | None, e: int, reading: str) -> np.ndarray:
        """[question, k]: each question's value with its k results read in full (1/0 for events,
        the share of evidence turns for shares, recall and precision of evidence characters)."""
        if reading in EVENTS:
            return np.isfinite(self.arrays(policy, model, e, reading)).astype(np.float64)
        column = {"share": "share_full", "share_touch": "share_touch"}.get(reading, reading)
        return np.nan_to_num(self.arrays(policy, model, e, column))


def largest_fitting(cost: np.ndarray, budget: float) -> np.ndarray:
    """Per question: the index of the largest k whose cost fits the budget, -1 if none does."""
    fits = cost <= budget  # nan (k beyond the pool) never fits
    return np.where(fits, np.arange(cost.shape[1])[None, :], -1).max(axis=1)


def take(values: np.ndarray, index: np.ndarray, empty: float = 0.0) -> np.ndarray:
    return np.where(index >= 0, values[np.arange(len(values)), np.maximum(index, 0)], empty)


def at_budgets(fit: Fit, policy: str, model: str | None, e: int, reading: str, budgets) -> np.ndarray:
    """[question, budget]: the value of the most results that fit in each budget (none: 0)."""
    tokens, values = fit.arrays(policy, model, e, "T"), fit.value(policy, model, e, reading)
    return np.stack([take(values, largest_fitting(tokens, b)) for b in budgets], 1)


def gain_cell(fit: Fit, a, b, reading: str, budget: float, keep: np.ndarray | None = None) -> str:
    """Δ (a minus b, each a (policy, model, e)) at one budget, with its interval."""
    d = at_budgets(fit, *a, reading, [budget])[:, 0] - at_budgets(fit, *b, reading, [budget])[:, 0]
    return delta(d if keep is None else d[keep], fit.clusters if keep is None else fit.clusters[keep])


def fitted_spend(fit: Fit, m: str, pool: str, reading: str, prices: np.ndarray, spends_of, usd: bool = False):
    """frontier.EqualSpend with each side buying as many results as the money does: at LLM price
    p and money worth S reader tokens, the side with a reranker buys the largest k whose pairs
    (in reader-token equivalents, r / p each; at each k the mean over questions, as
    frontier.equal_spend) and reader tokens fit in S; the side without one
    the largest k whose reader tokens do. Nan where the money cannot pay the reranker for one
    result's pool. spends_of(p): the spends at price p (reader-token equivalents); usd: the
    result's spends are the dollars per 1,000 questions (as frontier.equal_spend_usd)."""
    from rbench.report.frontier import EqualSpend

    t_rr = fit.arrays(pool, m, 0, "T")
    # the reranker's cost at each k: its mean pair tokens over the questions, as frontier.equal_spend charges it
    r_rr = np.broadcast_to(np.nanmean(fit.arrays(pool, m, 0, "R"), 0)[None, :], t_rr.shape)
    v_rr, t_fs, v_fs = fit.value(pool, m, 0, reading), fit.arrays("first", None, 0, "T"), fit.value("first", None, 0, reading)
    rr_price, smallest = price(m), float(np.nanmean(r_rr[:, 0]))  # the smallest pool's mean pair tokens
    diffs = []
    for p in prices:
        shift = rr_price / p  # reader tokens one pair token costs at this price
        cost = shift * r_rr + t_rr
        row = []
        for s in spends_of(p):
            if shift * smallest > s:  # blank, as frontier.equal_spend: the money cannot pay the reranker
                row.append(np.full(len(t_rr), np.nan))
                continue
            row.append(take(v_rr, largest_fitting(cost, s)) - take(v_fs, largest_fitting(t_fs, s)))
        diffs.append(np.stack(row, 1))
    d = np.stack(diffs, 1)  # [question, price, spend]
    gain = d.mean(0)  # a cell is blank for every question or for none
    same = np.isfinite(gain) & np.all(np.nan_to_num(d) == 0, axis=0)
    labels, inverse = np.unique(fit.clusters, return_inverse=True)
    lo = hi = np.full_like(gain, np.nan)
    if len(labels) >= MIN_CLUSTERS:
        sums = np.zeros((len(labels), *gain.shape))
        np.add.at(sums, inverse, np.nan_to_num(d))
        counts = np.bincount(inverse, minlength=len(labels)).astype(np.float64)
        draws = np.random.default_rng(0).multinomial(len(labels), np.full(len(labels), 1 / len(labels)), RESAMPLES)
        means = np.tensordot(draws, sums, axes=1) / (draws @ counts)[:, None, None]
        lo, hi = np.percentile(means, [2.5, 97.5], axis=0)
        lo, hi = np.where(np.isnan(gain), np.nan, lo), np.where(np.isnan(gain), np.nan, hi)
    spends = np.asarray(spends_of(prices[0])) * prices[0] / 1e3 if usd else np.asarray(spends_of(prices[0]))
    tokens = smallest if pool == "200" else float("nan")  # a flat pool's reranker costs the same at every k
    return EqualSpend(gain, same, lo, hi, tokens, np.asarray(prices), spends)


# ---------------------------------------------------------------- the report's tables


def summary_rows(fit: Fit, m: str, reading: str) -> list[str]:
    budgets = config.EVAL.budgets
    first = ("first", None, 0)

    def cells(policy):
        return " | ".join(f"{at_budgets(fit, policy, m, 0, reading, [b]).mean():.3f} vs "
                          f"{at_budgets(fit, *first, reading, [b]).mean():.3f}: "
                          f"{gain_cell(fit, (policy, m, 0), first, reading, b)}" for b in budgets)

    return [f"| budget first: the most results that fit, from a pool of 4 × that many | {cells('4k')} |",
            f"| budget first: the most of a flat 200's best results that fit | {cells('200')} |",
            "| budget first: 4 × k minus the flat 200 | "
            + " | ".join(gain_cell(fit, ("4k", m, 0), ("200", m, 0), reading, b) for b in budgets) + " |"]


def section(fit: Fit, models: list[str], number) -> list[str]:
    budgets = config.EVAL.budgets
    lines = [f"## {number}. Budget first, exactly: the most results that fit\n",
             "As the 2026-11-12 report's matched-token measures: every question gets the same reader budget and reads "
             "the most results that fit in it, whole. The runtime pool takes them from a pool of 4 × that many: at most "
             "50 results, because only each question's first 200 seeds were scored, so past the tokens of 50 results "
             "(about 4,100 without windows) its numbers stop growing by construction. The flat 200 takes them from its "
             "best 200; the first stage from its own order. The largest "
             "fitting k is found per question over every k, so every cell is exact. Δ: minus the first stage, paired by "
             "question, with its 95% interval.\n"]
    for reading in config.READINGS:
        lines += [f"**{reading}**\n", "| model · expand_context | budget | first stage | 4 × k Δ | flat 200 Δ | "
                  "4 × k minus flat 200 |", "|---|---|---|---|---|---|"]
        for m in models:
            for e in (0, *sorted(config.EVAL.expansions)):
                for b in budgets:
                    first = ("first", None, e)
                    lines.append(f"| {m} · {e} | {b:,} | {at_budgets(fit, *first, reading, [b]).mean():.3f} | "
                                 f"{gain_cell(fit, ('4k', m, e), first, reading, b)} | "
                                 f"{gain_cell(fit, ('200', m, e), first, reading, b)} | "
                                 f"{gain_cell(fit, ('4k', m, e), ('200', m, e), reading, b)} |")
            if reading != "every":
                break  # the other readings for the first model only
        lines.append("")
    return lines


# ---------------------------------------------------------------- the charts


def _lines(ax, xs, curves):
    from rbench.report import figures as fg

    for ys, color in curves:
        fg._plot(ax, xs, ys, color=color)


def _keys():
    from rbench.report import figures as fg

    return [fg._key(ROLE["none"]), fg._key(POOL_COLOR["4k"]), fg._key(POOL_COLOR["200"])]


def coverage_tokens(fit: Fit, models: list[str], out, head: str, reading: str, save) -> str:
    from rbench.report import figures as fg

    expansions = (0, *sorted(config.EVAL.expansions))
    fig, axes = fg._figure(len(models), len(expansions), panel=(3.6, 3.0), sharex=True, sharey=True)
    for i, m in enumerate(models):
        for j, e in enumerate(expansions):
            ax = axes[i][j]
            _lines(ax, BUDGETS, [(at_budgets(fit, "first", None, e, reading, BUDGETS).mean(0), ROLE["none"]),
                                 (at_budgets(fit, "4k", m, e, reading, BUDGETS).mean(0), POOL_COLOR["4k"]),
                                 (at_budgets(fit, "200", m, e, reading, BUDGETS).mean(0), POOL_COLOR["200"])])
            fg._token_log_x(ax)
            fg._share_y(ax)
            shade_cap(ax, cap_tokens(fit, m, e), BUDGETS[-1])
            ax.set_title(f"{fg.LABEL[m]} · expand_context {e}" if j == 0 else f"expand_context {e}", loc="left")
            if i == len(models) - 1:
                ax.set_xlabel(f"reader tokens ({fg.READER_NAME}; log scale)\nexact at every budget drawn")
            if j == 0:
                ax.set_ylabel(("mean over questions: " if rd.is_share(reading) else "share of questions: ")
                              + fg.EVENT_TEXT[reading])
    path = out / f"runtime_coverage_tokens_{reading}.png"
    save(fig, path, f"Coverage ({fg.EVENT_TEXT[reading]}) vs reader tokens, budget first, at MemMachine's runtime pool "
         f"— {head}", _keys(), [POLICY_TEXT["first"], POLICY_TEXT["4k"], POLICY_TEXT["200"]], 1,
         ["Does reranking read more of the evidence within the same reader tokens, at the pool MemMachine's runtime "
          "reranks?",
          "Budget first, as the 2026-11-12 charts: every question gets the same budget (x, log scale) and reads the most "
          "results that fit in it, whole. The largest k that fits is found per question over every k, so each point is "
          "exact.",
          "Blue: no reranker, the first stage's own order. Orange: the runtime pool, the best k of the first 4 × k seeds. "
          "Aqua: the best k of a flat 200 (the 2026-11-12 pool).",
          CAP_NOTE,
          f"y: {fg.EVENT_TEXT[reading]}. Rows: the two rerankers; columns: expand_context (neighbouring segments read with "
          "each result). The reranker's own cost: runtime_cost_equal*.png."])
    return str(path)


def rpf(fit: Fit, models: list[str], out, head: str, save) -> list[str]:
    from rbench.report import figures as fg

    expansions = (0, *sorted(config.EVAL.expansions))
    ks = np.arange(1, 51)
    written = []
    for m in models:
        for axis in ("tokens", "k"):
            fig, axes = fg._figure(3, len(expansions), panel=(3.4, 2.5), sharex=True)
            for j, e in enumerate(expansions):
                curves = []
                for policy, model, color in (("first", None, ROLE["none"]), ("4k", m, POOL_COLOR["4k"]),
                                             ("200", m, POOL_COLOR["200"])):
                    if axis == "tokens":
                        rec, pre = (at_budgets(fit, policy, model, e, x, BUDGETS) for x in ("recall", "precision"))
                    else:
                        rec, pre = (fit.value(policy, model, e, x)[:, :50] for x in ("recall", "precision"))
                    f1 = np.where(rec + pre > 0, 2 * rec * pre / np.maximum(rec + pre, 1e-12), 0.0)
                    curves.append(((rec.mean(0), pre.mean(0), f1.mean(0)), color))
                xs = BUDGETS if axis == "tokens" else ks
                for i, name in enumerate(("recall (evidence characters)", "precision", "F1")):
                    ax = axes[i][j]
                    _lines(ax, xs, [(c[i], color) for c, color in curves])
                    fg._token_log_x(ax) if axis == "tokens" else fg._log_x(ax, 10, list(KS))
                    if axis == "tokens":
                        shade_cap(ax, cap_tokens(fit, m, e), BUDGETS[-1])
                    if j == 0:
                        ax.set_ylabel(name)
                axes[0][j].set_title(f"{fg.LABEL[m]} · expand_context {e}" if j == 0 else f"expand_context {e}",
                                     loc="left")
                axes[2][j].set_xlabel(f"reader tokens ({fg.READER_NAME}; log scale)" if axis == "tokens"
                                      else "k results, read in full (log; every k)")
            for row in axes:
                top = max([np.nanmax(line.get_ydata()) for ax in row for line in ax.get_lines()], default=1.0)
                for ax in row:
                    fg._share_y(ax, min(1.0, top))
            path = out / f"runtime_rpf_{axis}_{m}.png"
            save(fig, path, f"Evidence-character recall, precision and F1 vs {'reader tokens' if axis == 'tokens' else 'k'}"
                 f" at MemMachine's runtime pool, {fg.LABEL[m]} — {head}", _keys(),
                 [POLICY_TEXT["first"], POLICY_TEXT["4k"], POLICY_TEXT["200"]] if axis == "tokens" else
                 ["no reranker: the first stage's first k", "the runtime pool: the best k of 4 × k",
                  "a flat 200: its best k"], 1,
                 ["Recall: the share of the evidence's characters read. Precision: the share of the characters read that "
                  "are evidence. F1: their harmonic mean, per question, averaged.",
                  ("x: the reader's budget; budget first, as the 2026-11-12 charts: every question reads the most results "
                   "that fit, whole, exact at every budget drawn." if axis == "tokens" else
                   "x: the limit k, the results returned and read in full (each with its window), exact at every k from 1 "
                   "to 50; the runtime pool reranks 4 × k seeds for each k."),
                  "Blue: no reranker. Orange: the runtime pool. Aqua: a flat 200. Columns: expand_context.",
                  CAP_NOTE if axis == "tokens" else
                  "x stops at k = 50: only each question's first 200 seeds were scored, so the runtime pool (4 × k) "
                  "goes no further."])
            written.append(str(path))
    return written


def rerank_cost(fit: Fit, models: list[str], out, head: str, save) -> str:
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    from rbench.report import figures as fg

    budgets = config.EVAL.budgets
    ks = np.arange(1, 51)
    fig, axes = fg._figure(len(models), len(budgets), panel=(3.8, 3.4), sharey=True)
    for i, m in enumerate(models):
        usd = np.nanmean(fit.arrays("4k", m, 0, "R"), 0) * price(m) / 1e6 * 1000  # $ per 1,000 questions at each k
        per_k = float(np.mean(usd / ks))
        for j, b in enumerate(budgets):
            ax = axes[i][j]
            curves = []
            for policy, model, color in (("first", None, ROLE["none"]), ("4k", m, POOL_COLOR["4k"]),
                                         ("200", m, POOL_COLOR["200"])):
                cost = fit.arrays(policy, model, 0, "every")[:, :50]  # the reader cost at which all evidence is read
                curves.append(((cost <= b).mean(0), color))
            _lines(ax, ks, curves)
            fg._log_x(ax, 10, list(KS))
            fg._share_y(ax)
            ax.set_title(f"{fg.LABEL[m]} · reader budget {b:,}", loc="left")
            ax.set_xlabel("limit k (results returned; every k)")
            top = ax.secondary_xaxis("top", functions=(lambda x, s=per_k: x * s, lambda y, s=per_k: y / s))
            top.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            top.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            top.xaxis.set_minor_formatter(NullFormatter())
            top.set_xlabel(r"runtime pool: reranker \$ per 1,000 questions", fontsize=8, color=fg.INK2)
            if j == 0:
                ax.set_ylabel(fg._coverage("every"))
    path = out / "runtime_rerank_cost.png"
    save(fig, path, f"Coverage vs the limit k at MemMachine's runtime pool, and the reranker's cost — {head}", _keys(),
         ["no reranker: the first stage's first k", "the runtime pool: the best k of 4 × k",
          "a flat 200: its best k"], 3,
         ["What does a larger limit buy, with and without reranking, and what does the reranker cost at the runtime pool?",
          "Each panel: one reader budget; the reader reads the k results in order until it. x: the limit k, every k from "
          "1 to 50 (the same on all three lines).",
          "Top axis: the runtime pool's reranker cost at that k, in $ per 1,000 questions at DeepInfra's list price (4 "
          "pairs per result). A flat 200 costs the same at every k: "
          + "; ".join(f"{fg.LABEL[m]} ${float(np.nanmean(fit.arrays('200', m, 0, 'R')[:, 0])) * price(m) / 1e3:.2g}"
                      for m in models) + " per 1,000 questions.",
          "Blue: no reranker. Orange: the runtime pool. Aqua: a flat 200."])
    return str(path)


def expansion_compare(fit: Fit, models: list[str], out, head: str, save) -> str:
    from rbench.report import figures as fg
    from rbench.report import palette

    expansions = (0, *sorted(config.EVAL.expansions))
    fig, axes = fg._figure(len(models), 2, panel=(4.4, 3.2), sharex=True, sharey=True)
    blues = palette.family(ROLE["none"], len(expansions))
    for i, m in enumerate(models):
        for j, pool in enumerate(("4k", "200")):
            ax = axes[i][j]
            shades = palette.family(POOL_COLOR[pool], len(expansions))
            for e, blue, shade in zip(expansions, blues, shades, strict=True):
                _lines(ax, BUDGETS, [(at_budgets(fit, "first", None, e, "every", BUDGETS).mean(0), blue),
                                     (at_budgets(fit, pool, m, e, "every", BUDGETS).mean(0), shade)])
            fg._token_log_x(ax)
            fg._share_y(ax)
            ax.set_title(f"{fg.LABEL[m]} · {'the runtime pool (4 × k)' if pool == '4k' else 'a flat 200'}", loc="left")
            if i == len(models) - 1:
                ax.set_xlabel(f"reader tokens ({fg.READER_NAME}; log scale)")
            if j == 0:
                ax.set_ylabel(fg._coverage("every"))
    handles = ([fg._key(c) for c in blues] + [fg._key(c) for c in palette.family(POOL_COLOR["4k"], len(expansions))]
               + [fg._key(c) for c in palette.family(POOL_COLOR["200"], len(expansions))])
    labels = ([f"no reranker, expand_context {e}" for e in expansions] + [f"runtime pool, expand_context {e}"
                                                                          for e in expansions]
              + [f"flat 200, expand_context {e}" for e in expansions])
    labels = [x + (" (at most 50 results)" if x.startswith("runtime pool") else "") for x in labels]
    path = out / "runtime_expansion_compare.png"
    save(fig, path, f"Expand_context compared, budget first, at MemMachine's runtime pool — {head}", handles, labels, 3,
         ["Does reading each result with its neighbouring segments pay for its extra text, with and without reranking?",
          "Budget first: every question reads the most results that fit in the budget (x), whole, each with its window "
          "(rerank the seeds, then expand; windows were not scored).",
          "Blues: no reranker. Oranges: the runtime pool (left). Aquas: a flat 200 (right). Within each, darker is less "
          "expansion.",
          "Why the oranges go flat: only each question's first 200 seeds were scored, so the runtime pool (4 × k) stops "
          "at 50 results; past their tokens (more with wider windows) its lines are flat by construction, not because "
          "reranking stops helping."])
    return str(path)


def query_types(fit: Fit, models: list[str], out, head: str, save) -> str:
    from rbench.report import figures as fg
    from rbench.report import palette
    from rbench.report.stats import interval

    budgets = config.EVAL.budgets
    names = sorted(set(fit.types))
    shades = palette.family(POOL_COLOR["4k"], len(budgets))
    fig, axes = fg._figure(1, len(models), panel=(5.0, 0.5 * len(names) + 1.6), sharey=True)
    for j, m in enumerate(models):
        ax = axes[0][j]
        for s, (b, shade) in enumerate(zip(budgets, shades, strict=True)):
            d = at_budgets(fit, "4k", m, 0, "every", [b])[:, 0] - at_budgets(fit, "first", None, 0, "every", [b])[:, 0]
            for y, t in enumerate(names):
                keep = fit.types == t
                lo, hi = interval(d[keep], fit.clusters[keep])
                yy = y + (s - (len(budgets) - 1) / 2) * 0.16
                ax.plot([lo, hi], [yy, yy], color=shade, linewidth=1.2, alpha=fg.ALPHA)
                ax.plot([d[keep].mean()], [yy], marker="o", markersize=5, color=shade, markerfacecolor=fg.SURFACE,
                        zorder=3)
        ax.axvline(0, color=fg.INK2, linewidth=0.6)
        ax.set_yticks(range(len(names)), [f"{t} ({int((fit.types == t).sum())})" for t in names])
        ax.invert_yaxis()
        ax.set_title(f"{fg.LABEL[m]} · the runtime pool, budget first", loc="left")
        ax.set_xlabel("reranker gain (all evidence read in full), reranked minus the first stage")
    path = out / "runtime_query_types.png"
    save(fig, path, f"Reranker gain by question type at MemMachine's runtime pool — {head}",
         [fg._key(c, marker="o") for c in shades], [f"reader budget {b:,}" for b in budgets], 4,
         ["Where does reranking help, by LongMemEval's question types, at the runtime pool?",
          "Budget first: at each budget every question reads the most results that fit, whole; the runtime pool takes "
          "them from 4 × that many. Each point: the gain at one budget (darker: smaller), with its 95% interval (paired "
          "bootstrap over questions); the number of questions of each type in brackets.",
          "At 4,000 tokens some questions reach the runtime pool's cap of 50 results (only 200 seeds per question were "
          "scored)."])
    return str(path)


def charts(fit: Fit, models: list[str], out, head: str, save) -> list[str]:
    for old in [*out.glob("runtime_rpf_*_x*.png"), out / "runtime_rpf_tokens.png", out / "runtime_rpf_k.png"]:
        if old.exists():
            old.unlink()  # replaced by the per-model runtime_rpf_<axis>_<model>.png
    written = [coverage_tokens(fit, models, out, head, r, save) for r in config.READINGS]
    written += rpf(fit, models, out, head, save)
    written += [rerank_cost(fit, models, out, head, save), expansion_compare(fit, models, out, head, save),
                query_types(fit, models, out, head, save)]
    return written
