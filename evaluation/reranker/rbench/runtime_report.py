"""The core's measurements under MemMachine's runtime pool: reports and charts of the runtime
results set (runtime_sim), with a flat pool of 200 beside it for comparison.

In that set the limit k is what MemMachine's `limit` is: the results returned, on both sides.
The first stage returns its first k seeds; the runtime pool reranks the first 4 x k and keeps k;
the flat pool reranks the first 200 and keeps k. The reader reads the k results in order until
its token budget. Every measure is the core report's, computed by the core's own code
(readings, frontier.equal_spend, stats.interval) on those rows:

  coverage at equal reader tokens, at the same limit k on both sides (core section 3)
  the reranker's cost per limit (core sections 7, rerank_cost)
  the same money per question: break-even LLM prices (core section 7, cost_equal*)
  the pool's ceiling (core section 2)
  query types (core section 9), expansion (core sections 11 and 12), 4B vs 8B, the held-out
  questions, and gold-character recall and precision (rpf_*)

MemMachine's default limit is 20 (main/memmachine.py: search_limit), a pool of 80. "Best k" is
chosen once for all questions on the same questions it is measured on, so it leans optimistic,
as the core's best L does.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd

from rbench import config
from rbench import partitions as pt
from rbench.report import frontier
from rbench.report import readings as rd
from rbench.report.data import Results, load, price
from rbench.report.stats import delta
from rbench.runtime import KS, OVERFETCH, POOL
from rbench import runtime_fit_report as rfr
from rbench.runtime_sim import FLAT, SAMPLE

DATASET, CHUNKING, SCOPE, VARIANT = "longmemeval_m", "seg500", "history", "question"
DEFAULT_K = 20  # MemMachine's default limit (main/memmachine.py: search_limit = limit or 20)
GRID = (5, 10, 20, 50)  # the limits drawn as panels
POOLS = ("4k", "200")
POOL_TEXT = {"4k": "the runtime pool (rerank 4 × k, keep k)", "200": "a flat pool (rerank 200, keep k)"}
RUN = f"{DATASET}__{SCOPE}__{VARIANT}__all"


def tail(e: int) -> str:
    return f"+x{e}" if e else ""


def rid(m: str, k: int, pool: str = "4k", e: int = 0) -> str:
    return f"rerank:{m}{FLAT if pool == '200' else ''}@n{k}{tail(e)}"


def fid(k: int, e: int = 0) -> str:
    return f"first_stage@n{k}{tail(e)}"


def rows(r: Results, name: str) -> pd.DataFrame:
    return r.rows(CHUNKING, name).sort_values("qid").reset_index(drop=True)


def at(x: pd.DataFrame, reading: str, budgets) -> np.ndarray:
    """[question, budget]: each question's value at each budget (frontier.per_question)."""
    return frontier.per_question(rd.costs(x, reading), np.asarray(budgets, dtype=float))


def gain(r: Results, a: str, b: str, reading: str, budget: float, keep: np.ndarray | None = None) -> np.ndarray:
    """Per question: configuration a minus b at one budget (paired by question; keep: a mask)."""
    x, y = rows(r, a), rows(r, b)
    d = at(x, reading, [budget])[:, 0] - at(y, reading, [budget])[:, 0]
    return d if keep is None else d[keep]


def cell(r: Results, a: str, b: str, reading: str, budget: float, keep: np.ndarray | None = None) -> str:
    x = rows(r, a)
    clusters = x.cluster.to_numpy() if keep is None else x.cluster.to_numpy()[keep]
    return delta(gain(r, a, b, reading, budget, keep), clusters)


def value(r: Results, name: str, reading: str, budget: float, keep: np.ndarray | None = None) -> float:
    v = at(rows(r, name), reading, [budget])[:, 0]
    return float((v if keep is None else v[keep]).mean())


def best_k(r: Results, name_of, reading: str, budgets=None) -> int:
    """The limit with the highest coverage averaged over the budgets (chosen once for all
    questions; with one budget, the best k at that budget)."""
    budgets = budgets or config.EVAL.budgets
    scores = {k: float(np.mean([value(r, name_of(k), reading, b) for b in budgets])) for k in KS}
    return max(scores, key=scores.get)


def uncapped(r: Results, limits=KS) -> Results:
    """The same results with the first stage at each limit replaced by the first stage reading as
    far as its budget goes (no limit; its whole search, 400 seeds): the baseline for the same
    money, so the side without a reranker may spend all of it, on more results instead of a
    reranker (the user's rule, 2026-11-18: same cost means same cost). frontier.equal_spend
    compares rerank:<m>@n<k> with first_stage@n<k>, so the rows are renamed to those."""
    df = r.df
    expansions = sorted({int(e) for e in df.expand.unique()})
    names = {fid(k, e) for k in limits for e in expansions}
    out = [df[~df.ranker.isin(names)]]
    for e in expansions:
        src = df[df.ranker == f"first_stage{tail(e)}"]
        out += [src.assign(ranker=fid(k, e), depth=f"n{k}") for k in limits]
    return Results(r.embedder, r.dataset, r.scope, r.variant, r.sample, pd.concat(out, ignore_index=True))


def usd_per_1000(r: Results, m: str, k: int, pool: str) -> float:
    return float(rows(r, rid(m, k, pool)).rr_tokens.mean()) * price(m) / 1e6 * 1000


def ceilings(embedder: str) -> pd.DataFrame:
    """Per question and limit: whether the pool holds all the evidence (4 x k and the flat 200),
    and whether it is reachable with k results at all (every evidence segment among a pool's
    seeds, and no more evidence segments than k): the best any reranker keeping k could do."""
    from rbench import runtime
    from rbench.samples import sample
    from rbench.schema import queries

    rows_, fmt = queries(DATASET), pt.formats(DATASET)
    out = []
    for key, members in pt.groups(DATASET, sample(DATASET, "all"), SCOPE).items():
        inputs = asyncio.run(runtime._question_inputs(embedder, DATASET, CHUNKING, key, members, rows_, fmt))
        for qi, x in zip(members, inputs, strict=True):
            need = {s for span in x["spans"] for s in span}
            where = {s: i for i, s in enumerate(x["seeds"])}
            deepest = max((where.get(s, 10**9) for s in need), default=0) + 1  # pool depth holding all of it
            for k in KS:
                for pool, depth in (("4k", OVERFETCH * k), ("200", POOL)):
                    out.append((rows_[qi]["qid"], k, pool, float(deepest <= depth),
                                float(deepest <= depth and len(need) <= k)))
    return pd.DataFrame(out, columns=["qid", "k", "pool", "holds", "reachable"])


# ---------------------------------------------------------------- the report


def _summary(r: Results, r0: Results | None, models: list[str], reading: str, fit=None) -> list[str]:
    budgets = config.EVAL.budgets
    lines = ["## Summary\n",
             f"Coverage ({reading}: {_reading_text(reading)}) at {', '.join(f'{b:,}' for b in budgets)} reader tokens. "
             "Δ: reranked minus the first stage at the same limit k, paired by question, with its 95% interval.\n",
             "With a token budget, a larger k only adds candidates: the reader stops at the budget either way. So the "
             f"best k is the largest, 50, whose runtime pool is 200: the 2026-11-12 report's setting, with the same "
             f"numbers. What the runtime pool changes is the cost at the k an application asks for, such as "
             f"MemMachine's default {DEFAULT_K}.\n"]
    for m in models:
        bk = {b: best_k(r, lambda k, m=m: rid(m, k), reading, [b]) for b in budgets}
        bf = {b: best_k(r, fid, reading, [b]) for b in budgets}
        b2 = {b: best_k(r, lambda k, m=m: rid(m, k, "200"), reading, [b]) for b in budgets}
        lines += [f"**{m}**\n", "| | " + " | ".join(f"@{b:,}" for b in budgets) + " |", "|---|" + "---|" * len(budgets)]
        lines.append(f"| MemMachine's default, k {DEFAULT_K} (pool {OVERFETCH * DEFAULT_K}) | "
                     + " | ".join(f"{value(r, rid(m, DEFAULT_K), reading, b):.3f} vs {value(r, fid(DEFAULT_K), reading, b):.3f}: "
                                  f"{cell(r, rid(m, DEFAULT_K), fid(DEFAULT_K), reading, b)}" for b in budgets) + " |")
        lines.append("| best k, each side (leans optimistic) | "
                     + " | ".join(f"{value(r, rid(m, bk[b]), reading, b):.3f} (k {bk[b]}) vs "
                                  f"{value(r, fid(bf[b]), reading, b):.3f} (k {bf[b]}): "
                                  f"{value(r, rid(m, bk[b]), reading, b) - value(r, fid(bf[b]), reading, b):+.3f}"
                                  for b in budgets) + " |")
        lines.append(f"| comparison: a flat 200, keep k {DEFAULT_K} | "
                     + " | ".join(f"{value(r, rid(m, DEFAULT_K, '200'), reading, b):.3f}: "
                                  f"{cell(r, rid(m, DEFAULT_K, '200'), fid(DEFAULT_K), reading, b)}" for b in budgets) + " |")
        lines.append(f"| comparison: 4 × k minus a flat 200, both k {DEFAULT_K} | "
                     + " | ".join(cell(r, rid(m, DEFAULT_K), rid(m, DEFAULT_K, "200"), reading, b) for b in budgets) + " |")
        lines.append("| comparison: a flat 200, keep the best k (leans optimistic) | "
                     + " | ".join(f"{value(r, rid(m, b2[b], '200'), reading, b):.3f} (k {b2[b]})" for b in budgets) + " |")
        if fit is not None:
            lines += rfr.summary_rows(fit, m, reading)
        if r0 is not None and not r0.rows(CHUNKING, f"rerank:{m}@n200").empty:
            lines.append("| the 2026-11-12 report: a flat 200, read until the budget (L 200 on both sides) | "
                         + " | ".join(f"{value(r0, f'rerank:{m}@n200', reading, b):.3f} vs "
                                      f"{value(r0, 'first_stage@n200', reading, b):.3f}: "
                                      f"{cell(r0, f'rerank:{m}@n200', 'first_stage@n200', reading, b)}"
                                      for b in budgets) + " |")
        lines.append("")
        lines += _break_even_lines(r, r0, m, reading, fit)
    return lines


def _reading_text(reading: str) -> str:
    from rbench.report.figures import EVENT_TEXT

    return EVENT_TEXT.get(reading, reading)


def _break_even(es, spend: float) -> str:
    if es is None:
        return "–"
    point, clear = frontier.equal_break_even(es, spend)
    lowest = float(es.prices[0])

    def price_text(p: float) -> str:  # the grid's lowest price: it pays at every price shown
        return f"at every price shown (from ${lowest:.2f}/M)" if p <= lowest else f"from ${p:.2f}/M"

    if not np.isfinite(point):
        return "never in range"
    return price_text(point) + (f"; clearly {price_text(clear)}" if np.isfinite(clear) else "; never clearly")


def _break_even_lines(r: Results, r0: Results | None, m: str, reading: str, fit=None) -> list[str]:
    spends = (1000, 4000)
    ru = uncapped(r)
    lines = [f"**The same money per question** ({m} at ${price(m):g}/M pair tokens): the LLM input price from which "
             "reranking gets more evidence for money worth these reader tokens, the reranker's cost included. The side "
             "without a reranker spends all the money too, reading as many results as it buys (no limit k).\n",
             "| limit k | pool | reranker $ per 1,000 questions | " + " | ".join(f"money worth {s:,} tokens" for s in spends)
             + " |", "|---|---|---|" + "---|" * len(spends)]
    for k in KS:
        for pool in POOLS:
            if pool == "200" and OVERFETCH * k == POOL:
                continue  # the runtime pool at this k is the flat 200 itself
            es = frontier.equal_spend(ru, m + (FLAT if pool == "200" else ""), price(m), k, reading, (CHUNKING, 0))
            lines.append(f"| {k} | {OVERFETCH * k if pool == '4k' else POOL} | ${usd_per_1000(r, m, k, pool):.3g} | "
                         + " | ".join(_break_even(es, s) for s in spends) + " |")
    if fit is not None:  # each side buys as many results as the money does (budget first, for money)
        for pool in POOLS:
            es = rfr.fitted_spend(fit, m, pool, reading, frontier.SPEND_PRICES, lambda p: frontier.SPENDS)
            lines.append(f"| as many as the money buys | {'4 × that k' if pool == '4k' else POOL} | "
                         f"{'grows with k' if pool == '4k' else '$' + format(usd_per_1000(r, m, 50, '200'), '.3g')} | "
                         + " | ".join(_break_even(es, s) for s in spends) + " |")
    if r0 is not None and not r0.rows(CHUNKING, f"rerank:{m}@n200").empty:
        es = frontier.equal_spend(uncapped(r0, (200,)), m, price(m), 200, reading, (CHUNKING, 0))
        lines.append("| 2026-11-12 report: L 200, read until the budget (the same baseline) | 200 | "
                     f"${float(rows(r0, f'rerank:{m}@n200').rr_tokens.mean()) * price(m) / 1e6 * 1000:.3g} | "
                     + " | ".join(_break_even(es, s) for s in spends) + " |")
    return [*lines, ""]


def _grid(r: Results, models: list[str], reading: str, number) -> list[str]:
    budgets = config.EVAL.budgets
    lines = [f"## {number}. Equal reader tokens at the same limit k ({reading}: {_reading_text(reading)})\n",
             "The first stage reads its first k seeds, the reranked side its k kept, each in order until the budget. "
             "Δ: reranked minus the first stage; last column, the runtime pool minus the flat 200 (both reranked, k "
             "kept); 95% intervals.\n"]
    for m in models:
        lines += [f"**{m}**\n", "| k (pool 4 × k) | budget | first stage | runtime pool (4 × k) Δ | flat 200 Δ | "
                  "4 × k minus flat 200 |", "|---|---|---|---|---|---|"]
        for k in KS:
            for b in budgets:
                same = OVERFETCH * k == POOL
                lines.append(f"| {k} ({OVERFETCH * k}) | {b:,} | {value(r, fid(k), reading, b):.3f} | "
                             f"{cell(r, rid(m, k), fid(k), reading, b)} | {cell(r, rid(m, k, '200'), fid(k), reading, b)} | "
                             + ("identical (the same pool)" if same else cell(r, rid(m, k), rid(m, k, '200'), reading, b))
                             + " |")
        lines.append("")
    return lines


def _costs(r: Results, models: list[str], number: int) -> list[str]:
    lines = [f"## {number}. The reranker's cost per limit\n",
             "Pairs scored per question, their mean tokens, and $ per 1,000 questions at DeepInfra's list price "
             f"({'; '.join(f'{m} ${price(m):g}/M' for m in models)}).\n",
             "| k | pool 4 × k: pairs · tokens | " + " | ".join(f"{m}, 4 × k" for m in models) + " | flat 200: pairs · tokens | "
             + " | ".join(f"{m}, flat 200" for m in models) + " |", "|---|---|" + "---|" * len(models) + "---|"
             + "---|" * len(models)]
    for k in KS:
        a, b = rows(r, rid(models[0], k)), rows(r, rid(models[0], k, "200"))
        lines.append(f"| {k} | {a.pairs.mean():.0f} · {a.rr_tokens.mean():,.0f} | "
                     + " | ".join(f"${usd_per_1000(r, m, k, '4k'):.3g}" for m in models)
                     + f" | {b.pairs.mean():.0f} · {b.rr_tokens.mean():,.0f} | "
                     + " | ".join(f"${usd_per_1000(r, m, k, '200'):.3g}" for m in models) + " |")
    return [*lines, ""]


def _ceiling_section(c: pd.DataFrame, number: int) -> list[str]:
    lines = [f"## {number}. The pool's ceiling (all evidence)\n",
             "Holds: the share of questions whose pool holds every evidence segment, so a reranker reading the whole pool "
             "could find it all. Reachable with k: it holds them and they number no more than k, so a reranker keeping "
             "k results could return them all: the most any reranker can do at that limit.\n",
             "| k | 4 × k: holds · reachable with k | flat 200: holds · reachable with k |", "|---|---|---|"]
    for k in KS:
        a, b = c[(c.k == k) & (c.pool == "4k")], c[(c.k == k) & (c.pool == "200")]
        lines.append(f"| {k} ({OVERFETCH * k}) | {a.holds.mean():.3f} · {a.reachable.mean():.3f} | "
                     f"{b.holds.mean():.3f} · {b.reachable.mean():.3f} |")
    return [*lines, ""]


def _query_type_section(r: Results, models: list[str], reading: str, number: int) -> list[str]:
    budgets = config.EVAL.budgets
    lines = [f"## {number}. Query types: reranker gain at the same limit k ({reading})\n",
             f"Δ: reranked minus the first stage at k {DEFAULT_K} (MemMachine's default), per LongMemEval question "
             "type; the last column the flat 200 at the same k and 1,000 tokens.\n"]
    base = rows(r, fid(DEFAULT_K))
    types = base.question_type.fillna("other").to_numpy()
    for m in models:
        lines += [f"**{m}**\n", "| question type | n | " + " | ".join(f"@{b:,}" for b in budgets) + " | flat 200 @1,000 |",
                  "|---|---|" + "---|" * len(budgets) + "---|"]
        for t in sorted(set(types)):
            keep = types == t
            lines.append(f"| {t} | {keep.sum()} | "
                         + " | ".join(cell(r, rid(m, DEFAULT_K), fid(DEFAULT_K), reading, b, keep) for b in budgets)
                         + f" | {cell(r, rid(m, DEFAULT_K, '200'), fid(DEFAULT_K), reading, 1000, keep)} |")
        lines.append("")
    return lines


def _expansion_section(r: Results, models: list[str], reading: str, number: int) -> list[str]:
    budgets = config.EVAL.budgets
    expansions = (0, *sorted(config.EVAL.expansions))
    lines = [f"## {number}. Expansion: each kept seed read with its window ({reading})\n",
             "expand_context e: each of the k results read with e neighbouring segments either side (rerank the seeds, "
             "then expand; windows were not scored). Cells: the first stage's coverage · reranked Δ (4 × k) · reranked Δ "
             "(flat 200).\n"]
    for m in models:
        for k in (5, DEFAULT_K):
            lines += [f"**{m}, k {k}**\n", "| expand_context | " + " | ".join(f"@{b:,}" for b in budgets) + " |",
                      "|---|" + "---|" * len(budgets)]
            for e in expansions:
                lines.append(f"| {e} | " + " | ".join(
                    f"{value(r, fid(k, e), reading, b):.3f} · {cell(r, rid(m, k, '4k', e), fid(k, e), reading, b)} · "
                    f"{cell(r, rid(m, k, '200', e), fid(k, e), reading, b)}" for b in budgets) + " |")
            lines.append("")
    return lines


def _models_section(r: Results, models: list[str], reading: str, number: int) -> list[str]:
    if len(models) < 2:
        return []
    a, b = models[:2]
    budgets = config.EVAL.budgets
    lines = [f"## {number}. {b} minus {a}, the runtime pool ({reading})\n",
             "| k (pool) | " + " | ".join(f"@{x:,}" for x in budgets) + " |", "|---|" + "---|" * len(budgets)]
    for k in KS:
        lines.append(f"| {k} ({OVERFETCH * k}) | " + " | ".join(cell(r, rid(b, k), rid(a, k), reading, x) for x in budgets)
                     + " |")
    return [*lines, ""]


def _held_out_section(r: Results, models: list[str], reading: str, number: int) -> list[str]:
    from rbench.samples import sample
    from rbench.schema import queries

    rows_ = queries(DATASET)
    tuned = {rows_[i]["qid"] for name in ("proto", "tune") for i in sample(DATASET, name)}
    held = ~rows(r, fid(DEFAULT_K)).qid.isin(tuned).to_numpy()
    budgets = config.EVAL.budgets
    lines = [f"## {number}. Selection check: the {held.sum()} questions the instruction was not chosen on ({reading})\n",
             f"Δ: reranked (4 × k) minus the first stage at the same k, on the held-out questions only; beside it the "
             f"{(~held).sum()} it was chosen on.\n",
             "| model · k | " + " | ".join(f"@{b:,} held out · chosen on" for b in budgets) + " |",
             "|---|" + "---|" * len(budgets)]
    for m in models:
        for k in (5, DEFAULT_K):
            lines.append(f"| {m} · {k} | " + " | ".join(
                f"{cell(r, rid(m, k), fid(k), reading, b, held)} · {cell(r, rid(m, k), fid(k), reading, b, ~held)}"
                for b in budgets) + " |")
    return [*lines, ""]


def report(r: Results, r0: Results | None, models: list[str], c: pd.DataFrame, reading: str = "every",
           fit=None) -> list[str]:
    n = r.df.qid.nunique()
    lines = [f"# The core's measurements at MemMachine's runtime pool — {DATASET} · {CHUNKING} · {n} questions\n",
             "**How to read the numbers** (definitions: METHODOLOGY sections 2 and 8.10)\n",
             "- **Limit k:** the results returned, MemMachine's `limit` (default 20), the same on both sides. The first "
             "stage returns its first k seeds. **The runtime pool** reranks the first 4 × k (MemMachine's overfetch) and "
             "keeps k. **A flat pool**, for comparison, reranks the first 200 and keeps k.",
             "- The reader reads the k results in order until its token budget (Qwen3 tokenizer, the rendered context). "
             "Coverage: the share of questions with all the evidence read in full (every) within the budget.",
             "- The reranker's cost: its measured pair tokens over the pool it scores, at DeepInfra's list price.",
             "- Δ: paired by question, with its 95% bootstrap interval. *Best k* is chosen on the same questions, so it "
             "leans optimistic.",
             "- The 2026-11-12 report's flat 200 (L 200) differs in one more way: it read all 200 reranked seeds until "
             "the budget, with no limit k. It is shown where it is the comparison.",
             f"- Recall at top-k (the k results, no budget): `{RUN}__runtime_topk.md`.\n"]
    lines += _summary(r, r0, models, reading, fit)
    lines += _ceiling_section(c, 1)
    lines += _grid(r, models, reading, 2)
    for i, other in enumerate(x for x in rd.available(r.df, config.READINGS) if x != reading):
        lines += _grid(r, models[:1], other, f"2.{i + 1}")
    lines += _costs(r, models, 3)
    lines += _query_type_section(r, models, reading, 4)
    lines += _expansion_section(r, models, reading, 5)
    lines += _models_section(r, models, reading, 6)
    lines += _held_out_section(r, models, reading, 7)
    if fit is not None:
        lines += rfr.section(fit, models, 8)
    return lines


# ---------------------------------------------------------------- the charts


def charts(r: Results, models: list[str], out, head: str, fit) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    written = rfr.charts(fit, models, out, head, _save)
    for pool in POOLS:
        for usd in (False, True):
            written.append(_cost_map(r, models, out, head, "every", pool, usd, fit))
    return written


def _money(text: str) -> str:
    """Chart text with its dollar signs escaped: two on one wrapped line would otherwise be read
    by matplotlib as a math expression."""
    return text.replace("$", r"\$")


def _save(fig, path, title: str, handles, labels, ncol: int, note: list[str]) -> None:
    from rbench.report import figures as fg

    fg._save(fig, path, _money(title), handles, [_money(x) for x in labels], ncol=ncol, note=[_money(x) for x in note])


def _diverging(pool: str):
    """The cost maps' scale, one meaning per color as in every other chart: blue where no reranker
    gets more for the money, the pool's color (orange: the runtime pool; aqua: a flat 200) where
    reranking does, a neutral gray between (the dataviz rule: two hues and a neutral midpoint)."""
    from matplotlib.colors import LinearSegmentedColormap

    # dark, base, light of each role's hue, held to the hue (a lightness family drifts: blue to lavender,
    # orange to red): the blue ramp is the dataviz palette's (the 2026-11-12 maps' blues)
    ramps = {"none": ("#104281", "#2a78d6", "#9ec5f4"), "4k": ("#9c3d0c", "#eb6834", "#f8b892"),
             "200": ("#0d6e4b", "#1baf7a", "#9fe3c6")}
    blue, other = ramps["none"], ramps[pool]
    return LinearSegmentedColormap.from_list(f"none_gray_{pool}", [blue[0], blue[1], blue[2], "#f0efec", other[2],
                                                                    other[1], other[0]])


def _spend_map(ax, es, few: bool, norm, cmap):
    """figures._spend_map with the scale as a parameter (_diverging): the gain where clear, gray where
    its interval includes 0, a hatch where both sides read the same, the break-even line."""
    import matplotlib
    from matplotlib.colors import LinearSegmentedColormap

    from rbench.report import figures as fg

    clear = (np.isfinite(es.gain) if few else (es.lo > 0) | (es.hi < 0)) & ~es.same
    im = ax.pcolormesh(es.prices, es.spends, np.where(clear, es.gain, np.nan).T, cmap=cmap, norm=norm, shading="nearest")
    ax.pcolormesh(es.prices, es.spends, np.where(np.isfinite(es.gain) & ~clear & ~es.same, 1.0, np.nan).T,
                  shading="nearest", cmap=LinearSegmentedColormap.from_list("muted", [fg.MUTED, fg.MUTED]))
    ax.pcolormesh(es.prices, es.spends, np.where(es.same, 1.0, np.nan).T, shading="nearest",
                  cmap=LinearSegmentedColormap.from_list("same", [fg.SAME, fg.SAME]))
    if es.same.any():
        with matplotlib.rc_context({"hatch.color": fg.SAME_HATCH, "hatch.linewidth": 0.6}):
            ax.contourf(es.prices, es.spends, es.same.T.astype(float), levels=[0.5, 1.5], colors="none", hatches=["///"])
    edge = np.nan_to_num(es.gain if few else es.lo, nan=-1.0).T
    line = bool((edge > 0).any() and (edge <= 0).any())
    if line:
        ax.contour(es.prices, es.spends, edge, levels=[0.0], colors=[fg.INK], linewidths=fg.LINE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(es.prices[0], es.prices[-1])
    ax.set_ylim(es.spends[0], es.spends[-1])
    return im, bool(es.same.any()), line


def _cost_map(r: Results, models: list[str], out, head: str, reading: str, pool: str, usd: bool, fit=None) -> str:
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter, NullFormatter

    from rbench.report import figures as fg
    from rbench.rerankers import RERANKERS

    few = fg._few(r)
    ru = uncapped(r)
    panels = []
    spends_of = (lambda p: frontier.USD_PER_1000 * 1e3 / p) if usd else (lambda p: frontier.SPENDS)
    for m in models:
        for k, what in ((5, "a small limit"), (DEFAULT_K, "MemMachine's default")):
            name = m + (FLAT if pool == "200" else "")
            es = (frontier.equal_spend_usd if usd else frontier.equal_spend)(ru, name, price(m), k, reading, (CHUNKING, 0))
            panels.append((m, k, what, es))
        if fit is not None:
            panels.append((m, "fitted", "as many results as the money buys",
                           rfr.fitted_spend(fit, m, pool, reading, frontier.SPEND_PRICES, spends_of, usd)))
    columns = max(sum(p[0] == m for p in panels) for m in models)
    low = min(-0.05, float(np.nanmin([np.nanmin(p[3].gain) for p in panels])))
    high = max(0.05, float(np.nanmax([np.nanmax(p[3].gain) for p in panels])))
    norm = TwoSlopeNorm(0, low, high)
    fig, axes = fg._figure(len(models), columns, panel=(4.6, 3.8), sharey=True)
    lines, identical, im = False, False, None
    for row, m in enumerate(models):
        mine = [p for p in panels if p[0] == m]
        for col in range(columns):
            ax = axes[row][col]
            if col >= len(mine):
                ax.set_visible(False)
                continue
            _, k, what, es = mine[col]
            im, same_here, line_here = _spend_map(ax, es, few, norm, _diverging(pool))
            identical, lines = identical or same_here, lines or line_here
            if usd:
                for t in fg.READER_TOKENS_SHOWN:
                    dollars = t * es.prices / 1e6 * 1000
                    inside = (dollars >= es.spends[0]) & (dollars <= es.spends[-1])
                    if inside.sum() >= 2:
                        ax.plot(es.prices[inside], dollars[inside], color=fg.INK2, linewidth=0.6, linestyle=(0, (2, 2)),
                                alpha=0.8)
                        i = int(np.flatnonzero(inside)[len(np.flatnonzero(inside)) // 2])
                        ax.annotate(f"{t:,} tokens", (es.prices[i], dollars[i]), fontsize=6.5, color=fg.INK2, rotation=32,
                                    rotation_mode="anchor", xytext=(2, 2), textcoords="offset points")
                own = price(m) * es.tokens / 1e6 * 1000
                if es.spends[0] <= own <= es.spends[-1]:
                    ax.axhline(own, color=fg.INK, linewidth=0.8, linestyle=":")
                    ax.annotate(_money(f"reranker cost: ${own:.2g} per 1,000 questions"), (es.prices[0], own), fontsize=6.5,
                                color=fg.INK, xytext=(3, 3), textcoords="offset points")
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _money(f"${v:g}")))
            else:
                for dollars in fg.DOLLARS_PER_1000:
                    tokens = dollars / 1000 / (es.prices / 1e6)
                    inside = (tokens >= es.spends[0]) & (tokens <= es.spends[-1])
                    if inside.sum() >= 2:
                        ax.plot(es.prices[inside], tokens[inside], color=fg.INK2, linewidth=0.6, linestyle=(0, (2, 2)),
                                alpha=0.8)
                        i = int(np.flatnonzero(inside)[len(np.flatnonzero(inside)) // 2])
                        ax.annotate(_money(f"${dollars:g} per 1,000 questions"), (es.prices[i], tokens[i]), fontsize=6.5,
                                    color=fg.INK2, rotation=-38, rotation_mode="anchor", xytext=(2, 2),
                                    textcoords="offset points")
                ax.set_yticks([500, 1000, 2000, 4000, 8000, 16000], ["500", "1,000", "2,000", "4,000", "8,000", "16,000"])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_facecolor(fg.SURFACE)
            ax.grid(False)
            if k == "fitted":
                ax.set_title(f"{fg.LABEL[m]} · reranked, {what}: the reranker scores "
                             + ("4 × that many seeds" if pool == "4k" else f"{POOL} seeds"), loc="left")
            else:
                scored = OVERFETCH * k if pool == "4k" else POOL
                ax.set_title(f"{fg.LABEL[m]} · reranked, k {k} kept ({what}): the reranker scores {scored} seeds, "
                             f"{es.tokens:,.0f} tokens per question", loc="left")
            ax.set_xlabel(_money("LLM input price ($/M)"))
            if col == 0:
                ax.set_ylabel(_money("budget, $ per 1,000 questions\n(reranker included)") if usd else
                              f"budget per question, as reader tokens ({fg.READER_NAME})\n(money / LLM price; the "
                              "reranked side reads fewer)")
    pole = "orange" if pool == "4k" else "aqua"
    fig.colorbar(im, ax=[a for row in axes for a in row],
                 label=f"reranking minus no reranker, for the same money ({fg.EVENT_TEXT[reading]}): {pole}, reranking "
                       "gets more; blue, no reranker does")
    legend = [(Patch(color=fg.MUTED), "no clear difference (the gain's 95% interval includes 0)")] if not few else []
    if identical:
        legend.append((Patch(facecolor=fg.SAME, edgecolor=fg.SAME_HATCH, hatch="///", linewidth=0),
                       "identical on every question: both sides read the same evidence"))
    legend.append((Patch(facecolor=fg.SURFACE, edgecolor=fg.GRID),
                   "blank: below the reranker's own cost (the dotted line)" if usd
                   else "blank: the budget cannot even pay for the reranker"))
    if lines:
        legend.insert(0, (fg._key(fg.INK), "break-even: reranking clearly gets more for the money inside the line"))
    path = out / ("runtime_cost_equal" + ("_usd" if usd else "") + ("_pool200" if pool == "200" else "") + ".png")
    _save(fig, path,
             f"Coverage for the same money per question{', in dollars' if usd else ''}, reranked vs not, "
             f"{POOL_TEXT[pool]} — {head}",
             [h for h, _ in legend], [lab for _, lab in legend], ncol=2,
             note=["Does reranking get more evidence for the same total money, its own cost included, when the reranker "
                   + ("scores MemMachine's runtime pool, 4 × k seeds for k results?" if pool == "4k" else
                      "scores a flat 200 seeds and keeps k? (The comparison for runtime_cost_equal"
                      + ("_usd" if usd else "") + ".png.)"),
                   "How to read a cell: an LLM input price (x) and a budget per question (y, "
                   + ("in $ per 1,000 questions)." if usd else "shown as the reader tokens it buys at that price).")
                   + " Without reranking, the reader spends the whole budget on the first stage's results, as many as "
                   "it buys (no limit k; up to its whole search): the same money buys more results instead of a "
                   "reranker. With reranking, the reranker is paid first and the reader gets the rest, for the k results "
                   "it kept.",
                   "Each panel: one limit k on the reranked side (the results it returns; MemMachine's default is 20); "
                   "its title gives the seeds the reranker scores and their tokens per question. Where the reranked side "
                   "has read all k results, more money helps only the side without a reranker.",
                   f"Color: {pole}, the side with the reranker reads more of the evidence ({fg.EVENT_TEXT[reading]}) "
                   "for that money; blue, the side without one does (the same colors as every other chart: blue no "
                   f"reranker, {pole} the cross-encoder on {'the runtime pool' if pool == '4k' else 'a flat 200'}). "
                   "Gray: no clear difference (the 95% interval includes 0). Black line: break-even; inside it "
                   "reranking clearly gets more for the money.",
                   ("Dashed diagonals: the reader tokens the budget buys without a reranker. Dotted line: the reranker's "
                    "own cost per 1,000 questions." if usd else
                    "Dashed lines: constant money, in $ per 1,000 questions (reader tokens x LLM price)."),
                   "The reranker's cost: DeepInfra's list price ("
                   + "; ".join(f"{fg.LABEL[m]} ${RERANKERS[m].price:g}/M pair tokens" for m in models)
                   + ") times its tokens per question.",
                   "Right column: as many results as the money buys, each side: with reranking, the largest k whose "
                   + ("4 × k pairs" if pool == "4k" else "200 pairs") + " and k results the budget pays for; without, "
                   "the most results its money buys (budget first, as the 2026-11-12 charts; exact for every question)."])
    return str(path)


def write(embedder: str, models: tuple[str, ...] = ("q4-mem", "q8-mem")) -> list[str]:
    """The report (summary/<embedder>/runtime/<run>__runtime_core.md) and its charts
    (figures/<embedder>/runtime/<run>/runtime_*.png) from the runtime results set."""
    from rbench import runtime

    r = load(embedder, DATASET, SCOPE, VARIANT, SAMPLE)
    try:
        r0 = load(embedder, DATASET, SCOPE, VARIANT, "all")
    except FileNotFoundError:
        r0 = None
    models = [m for m in models if m in r.models]
    from rbench import runtime_fit

    target = runtime_fit.path(embedder)
    if not target.exists():
        runtime_fit.run(embedder, models=tuple(models))
    fit = rfr.Fit(pd.read_parquet(target))
    c = ceilings(embedder)
    out = runtime.summary_dir(embedder)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{RUN}__runtime_core.md"
    path.write_text("\n".join(report(r, r0, models, c, fit=fit)) + "\n")
    head = f"{DATASET} · {CHUNKING} · {r.df.qid.nunique()} questions"
    return [str(path), *charts(r, models, runtime.figures_dir(embedder, DATASET, "all"), head, fit)]
