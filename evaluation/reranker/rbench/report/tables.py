"""The markdown report for one (embedder, dataset, scope, variant, sample).

Sections, each answering one question (README lists which question maps where):
  1 First stage alone: coverage by reader budget per chunking (the embedding/chunking benchmark)
  2 Headroom: how often the gold is inside the rerank pool at all
  3 Equal reader tokens: first stage vs each model, at equal vector search limits
  4 Token savings: first-stage budget needed to match the reranked coverage
  5 Overfetch: coverage as rerank depth grows, with its reranker tokens
  6 Equal reranker tokens: the best chunking and depth within each reranker budget
  7 Worth the money: break-even LLM/reranker price ratio by coverage target
  8 Chunkers: recursive minus fixed at matched size, without and with each reranker
  9 Query types: reranker gain by question type, form, length, context and parts
 10 Help and harm: how often reranking makes the gold cheaper or dearer
 11 Expansion: plain chunks vs small seeds expanded to the same read size, reranked before
    or after expansion (separates the embedded, reranked and read units)
 13 Equal counts: vector search limit, seeds kept, segments read
 14 EventMemory settings at each reader budget
 15 Reranker latency (a by-product)
(12, LongMemEval-style session recall, was removed; numbers are kept so references stay valid.)
Coverage is over all questions (there is no dev split, by decision): choices among
configurations are made and measured on the same questions, and each such section says so. Intervals: stats.interval.
History datasets add the expanded configurations to sections 3 and 6-7: the first stage
and rerank-then-expand at every replayed expand_context, expand-then-rerank where scored.
"""

from __future__ import annotations

import numpy as np

from rbench.paths import SUMMARY
from rbench import config
from rbench.report import frontier
from rbench.report import readings as rd
from rbench.report.data import Results, query_labels, size_label, unit_tokens
from rbench.report.stats import delta
from rbench.text import READER_NAME

EVENTS = rd.ALL  # the event readings (complete, touched) and the share readings (readings.py)
RERANK_BUDGETS = (1000, 2000, 4000, 8000, 16000, 32000, 64000)
SHOW_TARGETS = (0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def _name(event: str) -> str:
    """A reading's name in prose: first-gold, every-gold, ... or the share of evidence turns."""
    return {"share": "share-of-evidence-turns (complete)", "share_touch": "share-of-evidence-turns (touched)"}.get(
        event, f"{event}-gold")


def _intro(*lines: str) -> str:
    """A section's introduction: one short bullet per idea, most important first (the user's
    rule, 2026-11-10: no nested asides; details go to METHODOLOGY)."""
    return "\n".join(f"- {x}" for x in lines if x) + "\n"


def _cov(rows, event: str, budgets) -> list[str]:
    return [f"{x:.3f}" for x in rd.coverage(rd.costs(rows, event), budgets)]


def _pairs(r: Results) -> list[tuple[str, str]]:
    """(seg<N>, cut<M>): EventMemory's TextSegmenter and the simple chunker size-matched to it
    (config.cut_for: M fitted so the mean segments match)."""
    return [(c, config.cut_for(r.dataset, c)) for c in r.chunkings
            if c.startswith("seg") and config.cut_for(r.dataset, c) in r.chunkings]


def _sizes(r: Results, a: str, b: str) -> str:
    """Two chunkings' measured mean segments (retrieved seeds), flagged when not size-matched."""
    x, y = unit_tokens(r, a), unit_tokens(r, b)
    if not x or not y:
        return "sizes not measured"
    flag = "" if abs(y / x - 1) <= config.EVAL.size_tolerance else f", NOT size-matched ({y / x:.2f}×)"
    return f"{x:,.0f} vs {y:,.0f} tokens{flag}"


def first_stage(r: Results, event: str) -> list[str]:
    out = ["## 1. First stage alone (embedding and chunking)\n",
           _intro(f"{_name(event)} coverage by reader budget, with no reranker.",
                  "A row marked +xE reads each seed with its window of E neighbouring segments.",
                  "Measured unit: the mean segment searched and retrieved, or the mean window.",
                  ("Δ rows: the simple chunker minus TextSegmenter, at the same mean segment size." if _pairs(r) else ""),
                  ("The simple chunker splits more evidence turns, so it loses on readings in full: section 8 gives the "
                   "touched readings too." if _pairs(r) else "")),
           "| chunking | measured unit | " + " | ".join(map(str, config.EVAL.columns)) + " |", "|---|---|" + "---|" * len(config.EVAL.columns)]
    for c in r.chunkings:
        for e in (0, *r.expansions):
            name = "first_stage" + (f"+x{e}" if e else "")
            rows = r.rows(c, name)
            if not rows.empty:
                out.append(f"| {c}{name.removeprefix('first_stage')} | {size_label(r, c, e)} | "
                           + " | ".join(_cov(rows, event, config.EVAL.columns)) + " |")
    for seg, cut in _pairs(r):
        a, b = r.rows(seg, "first_stage"), r.rows(cut, "first_stage")
        out.append(f"| Δ {cut}−{seg} | {_sizes(r, seg, cut)} | " + " | ".join(
            delta(rd.hit(rd.costs(b, event), x) - rd.hit(rd.costs(a, event), x), a.cluster.to_numpy())
            for x in config.EVAL.columns) + " |")
    return out + [""]


HEADROOM = {"first": "some gold span complete", "every": "every gold span of one answer complete",
            "first_touch": "some gold span touched", "every_touch": "every gold span of one answer touched",
            "share": "their evidence turns complete (mean share)", "share_touch": "their evidence turns touched (mean share)"}


def headroom(r: Results, event: str) -> list[str]:
    out = ["## 2. Headroom: the gold among the seeds retrieved\n",
           _intro("A reranker can only reorder the seeds the vector search returned.",
                  f"Per vector search limit L: the share of questions whose top L seeds hold {HEADROOM[event]}.",
                  "Also: the reranker's pairs per question, and how often the first seed already holds evidence."),
           "| chunking | L | " + " | ".join(f"{m}: pairs / {event} among the L seeds" for m in r.models)
           + " | first segment overlaps gold |", "|---|---|" + "---|" * (len(r.models) + 1)]
    for c in r.chunkings:
        first = (r.rows(c, "first_stage")["rank"] == 1).mean()
        for limit in r.limits(c, kinds=("rerank",)):
            oracle = r.rows(c, f"oracle@{limit}")
            if oracle.empty:
                continue
            in_pool = rd.reachable(rd.costs(oracle, event)).mean()  # the oracle reads exactly its pool
            cells = []
            for m in r.models:
                rows = r.rows(c, f"rerank:{m}@{limit}")
                cells.append("–" if rows.empty else f"{rows.pairs.mean():.0f} / {in_pool:.3f}")
            out.append(f"| {c} | {limit[1:]} | " + " | ".join(cells) + f" | {first:.3f} |")
    return out + [""]


def _expanded(r: Results, depth: str | None, models: list[str]) -> list[str]:
    """Ranker names of the expanded configurations, per expand_context: the first stage,
    rerank-then-expand and expand-then-rerank (rows that were not simulated are skipped later)."""
    out = []
    for e in r.expansions:
        out.append(f"first_stage+x{e}")
        if depth:
            out += [f"{kind}:{m}@{depth}+x{e}" for m in models for kind in ("rerank", "xrerank")]
    return out


def _chooser(r: Results) -> str:
    return "the same questions (no dev split)"


def equal_reader_tokens(r: Results, event: str) -> list[str]:
    out = ["## 3. Equal reader tokens, at equal vector search limits\n",
           _intro("Same for every row of a block: the vector search limit L and the reader budget (columns).",
                  "Rows: the first stage in vector order; rerank the seeds; or expand, then rerank the windows.",
                  "+xE: each seed read with its window of E neighbouring segments.",
                  "Δ: minus the first stage at the same L and expand_context, with its 95% interval.",
                  "bound: the best any order of the same seeds could do, with seeds alone."),
           "| chunking | ranker | pairs | " + " | ".join(map(str, config.EVAL.columns)) + " | "
           + " | ".join(f"Δ@{b}" for b in config.EVAL.budgets) + " |", "|---|---|---|" + "---|" * (len(config.EVAL.columns) + len(config.EVAL.budgets))]
    for c in r.chunkings:
        for limit in r.limits(c):
            for e in (0, *r.expansions):
                if not e and limit not in r.limits(c, kinds=("rerank",)):  # a windows' pool: no setting of the seeds
                    continue
                tail = f"+x{e}" if e else ""
                base = r.rows(c, f"first_stage@{limit}{tail}")
                names = ([f"first_stage@{limit}{tail}"] + [f"rerank:{m}@{limit}{tail}" for m in r.models_at(c)]
                         + ([f"xrerank:{m}@{limit}{tail}" for m in r.models_at(c)] if e else [])
                         + ([] if e else [f"bound@{limit}"]))
                for name in names:
                    rows = r.rows(c, name)
                    if rows.empty or base.empty:
                        continue
                    ds = [] if name.startswith("first_stage") else [
                        delta(rd.hit(rd.costs(rows, event), b) - rd.hit(rd.costs(base, event), b),
                              base.cluster.to_numpy()) for b in config.EVAL.budgets]
                    out.append(f"| {c} | {name} | {rows.pairs.mean():.0f} | " + " | ".join(_cov(rows, event, config.EVAL.columns))
                               + " | " + " | ".join(ds or [""] * len(config.EVAL.budgets)) + " |")
    return out + [""]


def token_savings(r: Results, event: str) -> list[str]:
    grid = frontier.BUDGET_GRID
    out = ["## 4. Token savings\n",
           _intro("How many reader tokens does the first stage need to match the reranked coverage?",
                  "Same for both: the vector search limit L.",
                  "Cells: that budget as a multiple of the reranked budget; 'never' if it does not get there."),
           "| chunking | model | L | " + " | ".join(f"at {b}" for b in config.EVAL.columns) + " |", "|---|---|---|" + "---|" * len(config.EVAL.columns)]
    for c in r.chunkings:
        for limit in r.limits(c, kinds=("rerank",)):
            base = rd.coverage(rd.costs(r.rows(c, f"first_stage@{limit}"), event), grid)
            for m in r.models_at(c):
                rows = r.rows(c, f"rerank:{m}@{limit}")
                if rows.empty:
                    continue
                cells = []
                for b, target in zip(config.EVAL.columns, rd.coverage(rd.costs(rows, event), config.EVAL.columns), strict=True):
                    hit = np.flatnonzero(base >= target - 1e-12)
                    cells.append("never" if not len(hit) else f"{grid[hit[0]]:,.0f} ({grid[hit[0]] / b:.1f}×)")
                out.append(f"| {c} | {m} | {limit[1:]} | " + " | ".join(cells) + " |")
    return out + [""]


def overfetch(r: Results, event: str) -> list[str]:
    budgets = config.EVAL.budgets
    out = ["## 5. Overfetch\n",
           _intro("What does a deeper vector search buy?",
                  f"Coverage at {' / '.join(f'{b:,}' for b in budgets)} reader tokens as the limit grows.",
                  "Each cell: reranked, then the first stage over the same seeds, then the reranker's tokens per question.",
                  "'=': every seed has been read.")]
    for m in r.models:
        d = r.select(kind="rerank", model=m)
        depths = sorted({x for x in d.depth if x.startswith("n")}, key=lambda x: int(x[1:]))
        out += [f"**{m}**\n", "| chunking | | " + " | ".join(f"L {x[1:]}" for x in depths) + " |", "|---|---|" + "---|" * len(depths)]
        for c in r.chunkings:
            reranked, first, last = [], [], None
            for x in depths:
                rows, base = r.rows(c, f"rerank:{m}@{x}"), r.rows(c, f"first_stage@{x}")
                if rows.empty:
                    reranked.append("–")
                    first.append("–")
                elif last is not None and rows.pairs.mean() == last:
                    reranked.append("=")
                    first.append("=")
                else:
                    last = rows.pairs.mean()
                    reranked.append(" / ".join(_cov(rows, event, budgets)) + f" ({rows.rr_tokens.mean():,.0f})")
                    first.append(" / ".join(_cov(base, event, budgets)))
            out.append(f"| {c} | reranked | " + " | ".join(reranked) + " |")
            out.append(f"| {c} | first stage | " + " | ".join(first) + " |")
        out.append("")
    return out


def equal_reranker_tokens(r: Results, event: str) -> list[str]:
    out = ["## 6. Equal reranker tokens: candidates vs chunk size\n",
           _intro("For a given reranker budget, which chunking and setting reads the most evidence?",
                  "Rows: reranker tokens per question. Cells: coverage of the best setting that fits.",
                  f"Chosen on {_chooser(r)}.")]
    test = r.select(expand=None)
    families = sorted({c[:3] for c in r.chunkings})
    for m in r.models:
        out += [f"**{m}**\n", "| reranker tokens | chunker | " + " | ".join(f"reader {b}" for b in config.EVAL.budgets) + " |",
                "|---|---|" + "---|" * len(config.EVAL.budgets)]
        cands = r.select(model=m, expand=None)
        cands = cands[cands.kind.isin(frontier.RERANKED)]
        cost = cands.groupby(["chunking", "ranker"]).rr_tokens.mean()
        for R in RERANK_BUDGETS:
            for family in ([*families, "any"] if len(families) > 1 else ["any"]):
                cells = []
                for b in config.EVAL.budgets:
                    best = None
                    for (c, name), mean_tokens in cost.items():
                        if mean_tokens > R or (family != "any" and not c.startswith(family)):
                            continue
                        score = rd.hit(rd.costs(cands[(cands.chunking == c) & (cands.ranker == name)], event), b).mean()
                        if best is None or score > best[0]:
                            best = (score, c, name)
                    if best is None:
                        cells.append("–")
                    else:
                        t = test[(test.chunking == best[1]) & (test.ranker == best[2])]
                        cells.append(f"{rd.hit(rd.costs(t, event), b).mean():.3f} ({best[1]} {best[2].replace(f':{m}', '')})")
                out.append(f"| {R:,} | {family} | " + " | ".join(cells) + " |")
        out.append("")
    return out


PRICES_SHOWN = (1.0, 3.0, 10.0, 30.0)  # LLM input prices ($/M) section 7 gives the gain at


def worth_it(r: Results, event: str) -> list[str]:
    """Section 7 from the equal-spend computation (frontier.equal_spend, cost_equal.png)."""
    from rbench.report.data import price

    fixed = frontier.setting(r)
    spends = sorted({*config.EVAL.budgets, config.EVAL.token_top})
    out = ["## 7. Worth the money: the same money per question\n",
           _intro("Does reranking get more evidence for the same money per question?",
                  "Same for both: the money per question, and the vector search limit (one block per limit).",
                  "Without reranking all of it buys reader tokens; with reranking, part pays the reranker.",
                  "Spend: the money as reader tokens at that price. Gain: coverage reranked minus without, with its 95% "
                  "interval.",
                  "Break-even: the lowest price from which reranking gets more for the money; 'clearly': its interval "
                  "is above 0.",
                  "Chart: cost_equal.png.")]

    def fmt(x: float) -> str:
        return "n/a" if np.isnan(x) else "–" if np.isinf(x) else f"${x:,.2f}/M"

    for m in r.models:
        for L, what in frontier.cost_limits(r, m, event):
            es = frontier.equal_spend(r, m, price(m), L, event, fixed)
            if es is None:
                continue
            out += [f"**{m} at L {L} on both sides ({what}; {es.tokens:,.0f} reranker tokens per question)**\n",
                    "| spend (reader tokens, " + READER_NAME + ") | break-even | clearly from | "
                    + " | ".join(f"gain at ${p:g}/M" for p in PRICES_SHOWN) + " |",
                    "|---|---|---|" + "---|" * len(PRICES_SHOWN)]
            for T in spends:
                if frontier.same_everywhere(es, T):
                    out.append(f"| {T:,} | no difference: both read all {L} seeds | | " + " | ".join(
                        "0" for _ in PRICES_SHOWN) + " |")
                    continue
                point, clear = frontier.equal_break_even(es, T)
                j = int(np.argmin(np.abs(es.spends - T)))
                cells = []
                for p in PRICES_SHOWN:
                    i = int(np.argmin(np.abs(es.prices - p)))
                    g, lo, hi = es.gain[i, j], es.lo[i, j], es.hi[i, j]
                    cells.append("–" if np.isnan(g) else f"{g:+.3f}" + ("" if np.isnan(lo) else f" [{lo:+.3f}, {hi:+.3f}]"))
                out.append(f"| {T:,} | {fmt(point)} | {fmt(clear)} | " + " | ".join(cells) + " |")
            out.append("")
    return out + [""]




def chunkers(r: Results, event: str) -> list[str]:
    if not _pairs(r):
        return []
    readings = list(dict.fromkeys([event, *rd.available(r.df, ("every_touch", "share_touch"))]))
    out = ["## 8. Chunkers with a reranker: the simple chunker minus EventMemory's TextSegmenter\n",
           _intro("The simple chunker minus EventMemory's TextSegmenter, at the same mean segment size.",
                  "The simple chunker splits more evidence turns, so readings in full favour TextSegmenter.",
                  "The touched readings count a turn once any piece is read: compare both.",
                  "Measured segment sizes per pair. Details: METHODOLOGY section 3."),
           "| pair | measured sizes (seg vs cut) | reading | ranker | " + " | ".join(f"Δ@{b}" for b in config.EVAL.budgets) + " |",
           "|---|---|---|---|" + "---|" * len(config.EVAL.budgets)]
    for c, other in _pairs(r):
        shared = [m for m in r.models_at(c) if m in r.models_at(other)]
        depth = r.common_depth(c, shared) if shared else None
        for reading in readings:
            for name in ["first_stage"] + ([f"rerank:{m}@{depth}" for m in shared] if depth else []):
                a, b = r.rows(c, name), r.rows(other, name)
                if a.empty or b.empty:
                    continue
                out.append(f"| {c} vs {other} | {_sizes(r, c, other)} | {reading} | {name} | " + " | ".join(
                    delta(rd.hit(rd.costs(b, reading), x) - rd.hit(rd.costs(a, reading), x), a.cluster.to_numpy())
                    for x in config.EVAL.budgets) + " |")
    return out + [""]


def query_types(r: Results, event: str) -> list[str]:
    budgets = config.EVAL.budgets
    out = ["## 9. Query types: reranker gain\n",
           _intro("Which kinds of question does reranking help?",
                  "Labels: the dataset's question types, plus simple features of the question.",
                  f"Gain: coverage reranked minus the first stage over the same seeds, at {', '.join(f'{b:,}' for b in budgets)} "
                  "reader tokens.",
                  "Seeds: at the best vector search limit overall. Windows: at their pool.",
                  "Cells: the first stage's coverage, then Δ with its 95% interval.")]
    for c in r.chunkings:
        limits = r.limits(c)
        if not limits:
            continue
        for e, kind, what in ((0, "rerank", "seeds reranked"), (3, "xrerank", "windows reranked, expand_context 3")):
            tail = f"+x{e}" if e else ""
            for m in r.models_at(c):
                from rbench.report import best

                chosen = best.overall_limit(r, m, c, event) if kind == "rerank" else None
                limit = f"n{chosen}" if chosen else r.pool_depth(c, m, kind)
                if limit is None:
                    continue
                base, labels = query_labels(r, c, f"first_stage@{limit}{tail}")
                rows = r.rows(c, f"{kind}:{m}@{limit}{tail}")
                if rows.empty or base.empty:
                    continue
                out += [f"**{m}, {c}, L {limit[1:]}: {what}**\n",
                        "| feature | value | n | " + " | ".join(f"@{b:,}" for b in budgets) + " |",
                        "|---|---|---|" + "---|" * len(budgets)]
                for feature in ("question_type", "form", "length", "context", "parts"):
                    if feature not in labels:
                        continue
                    for v in sorted(set(labels[feature])):
                        mask = labels[feature] == v
                        if mask.sum() < 10:
                            continue
                        cells = []
                        for b in budgets:
                            hit = rd.hit(rd.costs(base, event), b)
                            gain = rd.hit(rd.costs(rows, event), b) - hit
                            cells.append(f"{hit[mask].mean():.3f}, {delta(gain[mask], base.cluster.to_numpy()[mask])}")
                        out.append(f"| {feature} | {v} | {int(mask.sum())} | " + " | ".join(cells) + " |")
                out.append("")
    return out


def help_harm(r: Results, event: str) -> list[str]:
    budgets = config.EVAL.budgets
    out = ["## 10. Help and harm\n",
           _intro("For how many questions does reranking help, and for how many does it hurt?",
                  "Same for both: the vector search limit L.",
                  "Helped / hurt: reranking lowers / raises the reader tokens the question needs.",
                  "Lost / gained at a budget: the question was within it before reranking and not after, or the reverse."),
           "| chunking | model | L | helped | hurt | " + " | ".join(f"lost / gained @{b:,}" for b in budgets) + " |",
           "|---|---|---|---|---|" + "---|" * len(budgets)]
    for c in r.chunkings:
        for limit in r.limits(c, kinds=("rerank",)):
            a = rd.costs(r.rows(c, f"first_stage@{limit}"), event)
            for m in r.models_at(c):
                rows = r.rows(c, f"rerank:{m}@{limit}")
                if rows.empty:
                    continue
                b = rd.costs(rows, event)
                cells = [f"{(rd.hit(a, x) > rd.hit(b, x)).mean():.3f} / {(rd.hit(b, x) > rd.hit(a, x)).mean():.3f}"
                         for x in budgets]
                # helped / hurt: the event's cost lowered / raised (a share reading has no single cost)
                moved = "– | –" if rd.is_share(event) else f"{(b < a).mean():.3f} | {(b > a).mean():.3f}"
                out.append(f"| {c} | {m} | {limit[1:]} | {moved} | " + " | ".join(cells) + " |")
    return out + [""]


def expansion(r: Results, event: str) -> list[str]:
    from rbench.report.figures import expansion_rows

    panels = expansion_rows(r)
    if not panels:
        return []
    out = ["## 11. Expansion: embedded vs reranked vs read unit\n",
           _intro(f"Read {panels[0].seed}'s seeds with their windows, or plain chunks of the same size?",
                  "Same for all: reader tokens.",
                  "Plain chunks are embedded, reranked and read whole; the closest measured size is used.",
                  "Sizes are measured. Overlapping windows are read once."),
           "| model | expand_context | approach | measured unit (" + READER_NAME + " tokens) | reranker tokens | "
           + " | ".join(f"@{b}" for b in config.EVAL.budgets) + " |",
           "|---|---|---|---|---|" + "---|" * len(config.EVAL.budgets)]
    flagged = []
    for p in panels:
        for label, c, name in p.items:
            rows = r.rows(c, name)
            size = p.plain_size if c == p.plain else p.window
            out.append(f"| {p.model} | {p.expand} | {label} ({c} {name}) | "
                       + (f"{size:,.0f}" if size else "not measured") + f" | {rows.rr_tokens.mean():,.0f} | "
                       + " | ".join(_cov(rows, event, config.EVAL.budgets)) + " |")
        if p.plain is not None and not p.matched:
            flagged.append(f"expand_context {p.expand}: {p.sizes(flag=False)}, {p.ratio:.2f}×")
    if all(p.plain is None for p in panels):
        out.append(f"\nNo other {panels[0].seed[:3]} chunking than {panels[0].seed} was simulated, so there are no plain "
                   "chunks of the same chunker to compare with the windows.")
    for f in dict.fromkeys(flagged):
        out.append(f"\n**Not size-matched** ({f}; tolerance {config.EVAL.size_tolerance:.0%}). Coverage is still "
                   "compared at equal reader tokens, but the plain chunks are not the windows' size.")
    return out + [""]


GLOSSARY = """**How to read the numbers** (definitions: METHODOLOGY section 2)

- Every number is a per-question value, averaged over the questions.
- Evidence: the turns marked as holding the answer.
- *Reader tokens*: the tokens of the context EventMemory renders for the reader, counted with Qwen3's tokenizer.
- *In full*: every segment of an evidence turn read. *Touched*: any of them read.
- *every*: all of a question's evidence read. *first*: some of it. *share*: the share of its evidence turns read.
- *Recall*: the share of the evidence's characters read. *Precision*: the share of what is read that is evidence.
- Intervals: 95%, from resampling the questions.
"""


KEPT, SEGMENTS = (5, 10), (10, 20, 50)


def _count_rows(r: Results, c: str, L: int) -> list[str]:
    names = [f"first_stage@n{L}", *(f"rerank:{m}@n{L}" for m in r.models_at(c))]
    for e in [3] if 3 in r.expansions else []:
        names += [f"first_stage@n{L}+x{e}", *(f"{k}:{m}@n{L}+x{e}" for m in r.models_at(c) for k in ("rerank", "xrerank"))]
    return names


def expansion_compare(r: Results, event: str) -> list[str]:
    """Coverage by expand_context, compared directly (figures.expansion_compare's values)."""
    from rbench.report.figures import expansion_lines

    if not r.models or not (r.df.expand > 0).any():
        return []
    m, budgets = r.models[0], config.EVAL.budgets
    out = ["## 12. Expand_context compared directly\n",
           _intro("Which expand_context reads the evidence most cheaply?",
                  f"{_name(event)} coverage at {', '.join(f'{b:,}' for b in budgets)} reader tokens, one row per "
                  "expand_context.",
                  "Same within a block: the vector search limit and how the seeds are ordered.",
                  f"Reader tokens count a window's extra text. Model: {m}. Chart: expansion_compare.png."),
           "| chunking | L | approach | expand_context | " + " | ".join(f"@{b:,}" for b in budgets) + " |",
           "|---|---|---|---|" + "---|" * len(budgets)]
    for c in r.chunkings:
        for L, what in frontier.cost_limits(r, m, event):
            for approach, lines in expansion_lines(r, c, m, L):
                for e, name in lines:
                    out.append(f"| {c} | {L} ({what}) | {approach} | {e} | "
                               + " | ".join(_cov(r.rows(c, name), event, budgets)) + " |")
    return out + [""]


def equal_counts(r: Results, event: str) -> list[str]:
    """Coverage at equal counts: the vector search limit L (seeds retrieved, all of them
    reranked), the seeds kept after reranking k, and the segments read after expansion m."""
    if f"{event}_ctx" not in r.df:
        return []
    out = ["## 13. Equal counts: vector search limit, seeds kept, segments read\n",
           _intro(f"{_name(event)} coverage by counts instead of tokens.",
                  "L: the vector search limit. The reranker scores all L seeds; nothing beyond them is read.",
                  "k: the seeds kept after reranking. m: the segments read, EventMemory's max_num_segments.",
                  "'all': every one of the L seeds read. Without a reranker: the same seeds in vector order."),
           "| chunking | L | ranker | " + " | ".join(f"k≤{k}" for k in KEPT) + " | " + " | ".join(f"m≤{m}" for m in SEGMENTS)
           + " | all |", "|---|---|---|" + "---|" * (len(KEPT) + len(SEGMENTS) + 1)]
    for c in r.chunkings:
        for L in (int(x[1:]) for x in r.limits(c, kinds=("rerank",))):
            for name in _count_rows(r, c, L):
                rows = r.rows(c, name)
                if rows.empty:
                    continue
                cells = [f"{(rows[f'{event}_ctx'] <= k).mean():.3f}" for k in KEPT]
                cells += [f"{(rows[f'{event}_k'] <= m).mean():.3f}" for m in SEGMENTS]
                cells.append(f"{rd.reachable(rd.costs(rows, event)).mean():.3f}")
                out.append(f"| {c} | {L} | {name.split('@')[0]}{'+x' + name.split('+x')[1] if '+x' in name else ''} | "
                           + " | ".join(cells) + " |")
    return out + [""]




def _pick(choices: dict, factor: str, which: str) -> str:
    """A factor's best or cheapest-equivalent value's label ('–' when not simulated)."""
    ch = choices.get(factor)
    return "–" if ch is None else ch.values[getattr(ch, which)].label


def recommended(r: Results, event: str) -> list[str]:
    """The EventMemory settings to use at each reader budget, per model, from the best-settings
    choices (section 15): vector_search_limit and max_num_segments the cheapest within the
    best's 95% bound, segment length and expand_context the best."""
    from rbench.report import best

    rows = []
    for m in r.models:
        for b in config.EVAL.budgets:
            choices = {f: best.choose(r, f, v) for f, v in best.factors(r, m, event, b).items()}
            if not choices:
                continue
            rows.append(f"| {m} | {b:,} | {_pick(choices, 'L', 'cheapest')} | {_pick(choices, 'm', 'cheapest')} | "
                        f"{_pick(choices, 'segment', 'best')} | {_pick(choices, 'reader sees', 'best')} | "
                        f"{_pick(choices, 'reranker sees', 'best')} |")
    if not rows:
        return []
    return ["**Best value of each factor, one at a time** (from section 15, chosen once for all questions at each "
            "budget). Each is chosen with every other factor at the core setting, so they were not evaluated together and "
            "a combination of them may do differently: read each column on its own. The vector search limit and "
            "max_num_segments are the cheapest within the best's 95% bound where reranking does not lose to the first "
            "stage over the same seeds (max_num_segments at the deepest pool); the segment length and what the reader sees "
            "(expand_context) the best; the scoring position compares seeds and windows at expand_context 3 and one L.\n",
            "| model | reader budget | vector_search_limit | max_num_segments | segment length | reader sees | reranker sees |",
            "|---|---|---|---|---|---|---|", *rows, ""]


def settings(r: Results, event: str) -> list[str]:
    """What each configuration needs EventMemory set to at a reader budget: max_num_segments
    (segments read within it) and, for the first stage, vector_search_limit (seeds reached)."""
    budgets = [b for b in config.EVAL.budgets if f"segs_t{b}" in r.df]  # counts are stored at the budgets configured when simulated
    if not budgets:
        return []
    skipped = [b for b in config.EVAL.budgets if b not in budgets]
    out = ["## 14. EventMemory settings at each reader budget\n",
           _intro("What to set EventMemory to, at each reader budget.",
                  f"Each cell: {_name(event)} coverage, then max_num_segments and seeds read: the median, and in "
                  "brackets the 10th percentile.",
                  "A max_num_segments that keeps 90% of questions within the budget is the 10th percentile.",
                  "The vector search limit: the seeds read without a reranker, the depth in the name with one.",
                  (f"Not shown: {', '.join(map(str, skipped))}, not stored when this was simulated." if skipped else "")),
           "| chunking | ranker | " + " | ".join(f"@{b}" for b in budgets) + " |",
           "|---|---|" + "---|" * len(budgets)]
    out = out[:1] + recommended(r, event) + out[1:]  # under the section's heading
    for c in r.chunkings:
        names = ["first_stage", *(f"first_stage+x{e}" for e in r.expansions if e == 3)]
        for m in r.models_at(c):  # EventMemory's default limit and each scoring position's deepest pool
            for kind, tail in (("rerank", ""), ("xrerank", "+x3")):
                deepest = r.pool_depth(c, m, kind)
                names += [f"{kind}:{m}@{d}{tail}" for d in sorted({x for x in ("n20", deepest) if x}, key=lambda x: int(x[1:]))]
        for name in names:
            rows = r.rows(c, name)
            if rows.empty:
                continue
            cells = []
            for b in budgets:
                segs, ctx = rows[f"segs_t{b}"], rows[f"ctx_t{b}"]
                cells.append(f"{rd.hit(rd.costs(rows, event), b).mean():.2f} · {segs.median():.0f} ({segs.quantile(0.1):.0f}) · "
                             f"{ctx.median():.0f} ({ctx.quantile(0.1):.0f})")
            out.append(f"| {c} | {name} | " + " | ".join(cells) + " |")
    return out + [""]


def latency(r: Results) -> list[str]:
    """Reranker request times recorded while scoring this dataset: a by-product, not a
    deployment measurement (requests run 16 at a time, include queueing and any retries)."""
    from rbench.rerankers import RERANKERS, ScoreCache

    rows = []
    for m in RERANKERS:
        if m not in r.models:
            continue
        calls = ScoreCache(m).db.execute("SELECT pairs, tokens, seconds FROM calls WHERE note LIKE ? AND seconds > 0",
                                         (f"{r.dataset}/%",)).fetchall()
        if calls:
            pairs, tokens, seconds = (np.asarray(x, dtype=float) for x in zip(*calls, strict=True))
            rows.append(f"| {m} ({RERANKERS[m].backend}) | {len(calls):,} | {np.median(pairs):.0f} | {np.median(tokens):,.0f} | "
                        f"{np.median(seconds):.2f} | {np.percentile(seconds, 90):.2f} | {np.median(pairs / seconds):.0f} |")
    if not rows:
        return []
    return ["## 17. Reranker latency (a by-product)\n",
            _intro("Times of the reranker requests made for this dataset: a by-product, not a deployment's latency.",
                   "DeepInfra requests run many at once and include queueing and retries; local ones are this machine.",
                   "Per question, EventMemory would rerank its seeds in one or a few such requests."),
            "| model | requests | pairs per request (median) | tokens per request | seconds (median) | seconds (p90) | "
            "pairs per second (median) |", "|---|---|---|---|---|---|---|", *rows, ""]


def clear(r: Results) -> None:
    """Remove a run's summaries before they are written again, so none from an earlier version
    is left beside the new ones."""
    for old in (SUMMARY / r.embedder).glob(f"{r.dataset}__{r.scope}__{r.variant}__{r.sample}__*.md"):
        old.unlink()


def _value_cell(v) -> str:
    """A factor value's cell: its label, reranked coverage · the first stage's under it."""
    first = "–" if v.first is None else f"{v.first.mean():.3f}"
    return f"{v.label}: {v.reranked.mean():.3f} · {first}"


def best_settings(r: Results, event: str) -> list[str]:
    """Which value of each factor works best, one factor at a time (best.py)."""
    from rbench.report import best

    if not r.models:
        return []
    core = frontier.setting(r)
    out = ["## 15. Best settings, one factor at a time\n",
           _intro("Which value of each setting works best?",
                  f"Each setting varied alone; the others at the core setting: {core[0]}, expand_context {core[1]}, the "
                  "reranker scoring the seeds at its pool.",
                  "Best: the highest reranked coverage. Cheapest equivalent: the cheapest value within the best's 95% "
                  "interval where reranking does not lose.",
                  "Chosen once for all questions at a budget, on the same questions: the best leans optimistic.",
                  "Chart: best_settings.png.")]
    for m in r.models:
        for b in config.EVAL.budgets:
            choices = {f: best.choose(r, f, v) for f, v in best.factors(r, m, event, b).items()}
            if not choices:
                continue
            out += [f"**{m}, {b:,} reader tokens**\n",
                    "| factor | values | best (reranked · first stage) | cheapest equivalent (reranked · first stage) "
                    "| first stage's best |", "|---|---|---|---|---|"]
            for f, ch in choices.items():
                firsts = [(v.first.mean(), v.label) for v in ch.values if v.first is not None]
                top = max(firsts) if firsts else None
                out.append(f"| {f} | {len(ch.values)} | {_value_cell(ch.values[ch.best])} | "
                           f"{_value_cell(ch.values[ch.cheapest])} | "
                           + (f"{top[1]}: {top[0]:.3f}" if top else "–") + " |")
            (rc, rv, fc, fv) = best.overall(r, m, event, b)
            out.append(f"\nBest combination simulated: reranked {rc[0]} {rc[1]} {rv:.3f}; first stage alone "
                       f"{fc[0]} {fc[1]} {fv:.3f}.\n")
    return out + [""]


def _topk_cell(g: dict, i: int | None, j: int) -> str:
    """One cell of the top-k table: reranked recall at pool L (row i) and k (column j), with its
    gain over the first stage; k = L says it equals the first stage by construction."""
    if i is None or np.isnan(g["reranked"][i, j]):
        return "–"
    if g["limits"][i] == g["ks"][j]:
        return f"{g['reranked'][i, j]:.3f} (= first stage: k = L)"
    ci = "" if np.isnan(g["low"][i, j]) else f" [{g['low'][i, j]:+.3f}, {g['high'][i, j]:+.3f}]"
    return f"{g['reranked'][i, j]:.3f}: {g['gain'][i, j]:+.3f}{ci}"


def topk_recall(r: Results) -> list[str]:
    """Recall with and without reranking across top-k, read both ways (report/topk.py)."""
    from rbench.report import topk

    if not r.models or not topk.available(r):
        return []
    core = frontier.setting(r)
    out = ["## 16. Recall at top-k, with and without reranking (the asker's metric)\n",
           _intro("Recall with and without reranking across top-k: the asker's metric.",
                  "Top-k read two ways: L, the vector search limit with every seed reranked; k, the seeds kept after "
                  "reranking.",
                  "Recall: the share of a question's evidence turns found in the first k seeds; also all evidence in full.",
                  "At k = L everything retrieved is kept, so reranking cannot change recall: the gain there is 0.",
                  f"{core[0]}, expand_context 0. Δ: reranked minus the first stage, with its 95% interval.")]
    for m in r.models:
        for reading in topk.READINGS:
            g = topk.grid(r, m, reading)
            if not g["limits"]:
                continue
            bests = topk.best_limit(g)
            limits, ks = g["limits"], g["ks"]
            pool = len(limits) - 1
            at20 = limits.index(20) if 20 in limits else None
            out += [f"**{m}: {topk.NAMES[reading]}, recall at k seeds kept**\n",
                    "| k | first stage (= top-k as L) | L 20, reranked | best L, reranked | pool, reranked |",
                    "|---|---|---|---|---|"]
            for j, k in enumerate(ks):
                best_i = limits.index(bests[j]) if bests[j] is not None else None
                out.append(f"| {k} | {g['first'][j]:.3f} | {_topk_cell(g, at20, j)} | "
                           + (f"L {bests[j]}: " if best_i is not None else "") + f"{_topk_cell(g, best_i, j)} | "
                           f"L {limits[pool]}: {_topk_cell(g, pool, j)} |")
            out.append("")
        g = topk.grid(r, m, "share")
        if g["limits"]:
            cols = [k for k in topk.SLICE_KS if k in g["ks"]]
            out += [f"**{m}: evidence turns found in full, recall vs L at fixed k**\n",
                    "| L | " + " | ".join(f"k {k}: reranked (Δ)" for k in cols) + " |", "|---|" + "---|" * len(cols)]
            for i, L in enumerate(g["limits"]):
                cells = []
                for k in cols:
                    j = g["ks"].index(k)
                    v = g["reranked"][i, j]
                    cells.append("–" if np.isnan(v) else f"{v:.3f} ({g['gain'][i, j]:+.3f})"
                                 + (" k = L" if L == k else ""))
                out.append(f"| {L} | " + " | ".join(cells) + " |")
            out.append("")
    return out


SUMMARY_TARGETS = (0.5, 0.7, 0.9)  # coverage targets the summary gives the break-even price at


def _verdict(lo: float, hi: float) -> str:
    if np.isnan(lo):
        return "too few clusters to tell"
    return "helps" if lo > 0 else "hurts" if hi < 0 else "no clear difference"


def _ci(lo: float, hi: float, digits: int = 3) -> str:
    """A 95% interval and its verdict; with fewer than stats.MIN_CLUSTERS clusters there is none."""
    from rbench.report.stats import MIN_CLUSTERS

    if np.isnan(lo):
        return f"(no interval: fewer than {MIN_CLUSTERS} clusters)"
    return f"[{lo:+.{digits}f}, {hi:+.{digits}f}] {_verdict(lo, hi)}"


def summary(r: Results, event: str) -> list[str]:
    """The questions answered first, a fixed template filled with this report's numbers: per
    model, the gain at EventMemory's defaults and at the best settings at the realistic
    budgets, the break-even LLM price at a few targets, and the caveats."""
    from rbench.report import best
    from rbench.report.data import price
    from rbench.report.stats import interval

    if not r.models:
        return []
    core = frontier.setting(r)
    default_limit = "n20"
    budgets = config.EVAL.budgets
    out = ["## Summary\n",
           _intro("Does reranking improve retrieval enough to justify its cost?",
                  f"Measure: {_name(event)} coverage at {', '.join(f'{b:,}' for b in budgets)} reader tokens ({READER_NAME}).",
                  "Δ: reranked minus the first stage over the same seeds, with its 95% interval.",
                  "*helps* / *hurts*: the interval excludes 0.")]
    for m in r.models:
        out.append(f"**{m}**\n")
        rows, base = r.rows(core[0], f"rerank:{m}@{default_limit}"), r.rows(core[0], f"first_stage@{default_limit}")
        lines = ["| | " + " | ".join(f"@{b:,}" for b in budgets) + " |", "|---|" + "---|" * len(budgets)]
        if not rows.empty and not base.empty:
            cells = []
            for b in budgets:
                d = rd.hit(rd.costs(rows, event), b) - rd.hit(rd.costs(base, event), b)
                lo, hi = interval(d, base.cluster.to_numpy())
                cells.append("identical on every question (both have read the whole pool)" if not d.any() else
                             f"{d.mean():+.3f} {_ci(lo, hi)}")
            lines.append(f"| EventMemory's defaults ({core[0]}, L 20, expand_context 0) | " + " | ".join(cells) + " |")
        cells = []
        for b in budgets:
            rc, rv, fc, fv = best.overall(r, m, event, b)
            if rc is None or fc is None:
                cells.append("–")
                continue
            d = rd.hit(rd.costs(r.rows(*rc), event), b) - rd.hit(rd.costs(r.rows(*fc), event), b)
            lo, hi = interval(d, r.rows(*fc).cluster.to_numpy())
            cells.append(f"{rv:.3f} vs {fv:.3f}: {d.mean():+.3f} {_ci(lo, hi)} "
                         f"({rc[1].split('@')[1]} vs {fc[1].replace('first_stage', 'first stage')})")
        lines.append("| best settings, each side (section 15; leans optimistic) | " + " | ".join(cells) + " |")
        out += lines + [""]
        at = []
        for L, what in frontier.cost_limits(r, m, event):
            es = frontier.equal_spend(r, m, price(m), L, event, core)
            if es is None:
                continue
            parts = []
            for T in (config.EVAL.budgets[1] if len(config.EVAL.budgets) > 1 else config.EVAL.budgets[0],
                      config.EVAL.budgets[-1]):
                if frontier.same_everywhere(es, T):
                    parts.append(f"money worth {T:,} reader tokens: no difference (both read all {L} seeds)")
                    continue
                point, clear = frontier.equal_break_even(es, T)
                text = "never in $0.05-$50/M" if np.isinf(point) else f"from ${point:.2f}/M"
                if not np.isnan(clear):
                    text += ", clearly " + ("never in range" if np.isinf(clear) else f"from ${clear:.2f}/M")
                parts.append(f"money worth {T:,} reader tokens: {text}")
            at.append(f"At L {L} ({what}): " + "; ".join(parts) + ".")
        out.append(f"**For the same money per question** ({m} at ${price(m):g}/M, the same vector search limit on both "
                   "sides). The LLM input price from which reranking gets more evidence:\n")
        out.append(_intro(*at, "Details: section 7 and cost_equal.png."))
    from rbench.report import topk

    if topk.available(r):
        for m in r.models:
            g = topk.grid(r, m, "share")
            if not g["limits"]:
                continue
            parts = []
            for k in topk.SLICE_KS:
                if k in g["ks"] and 20 in g["limits"] and k < 20:
                    i, j = g["limits"].index(20), g["ks"].index(k)
                    parts.append(f"k {k}: {g['first'][j]:.2f} → {g['reranked'][i, j]:.2f} ({g['gain'][i, j]:+.2f} "
                                 f"{_ci(g['low'][i, j], g['high'][i, j], 2)})")
            out.append(f"**Recall at top-k** ({m}, evidence turns read in full):\n")
            out.append(_intro("Top-k as the seeds kept after reranking the default 20: " + ("; ".join(parts) or "–") + ".",
                              "Top-k as the vector search limit: everything retrieved is kept, so reranking cannot "
                              "change recall.",
                              "Details: section 16."))
    pool = r.pool_depth(core[0], r.models[0], "rerank")
    bound = r.rows(core[0], f"bound@{pool}") if pool else None
    from rbench import depth

    shares = depth.pool_shares(r.embedder, r.dataset, core[0], r.sample, 0, int(pool[1:])) if pool else None
    fills = (f". Reading it reaches {config.EVAL.token_top:,} reader tokens for {shares[0]:.0%} of questions"
             if shares else "")
    ceiling = (f"The pool: {pool[1:]} seeds. It holds all the evidence for "
               f"{rd.reachable(rd.costs(bound, event)).mean():.0%} of questions, so no reranker can do better{fills}."
               if bound is not None and not bound.empty else "")
    from rbench.report.figures import not_run

    missing = not_run(r)
    out += ["**Notes**\n", _intro(ceiling.strip(),
                                    "Settings are chosen once for all questions, on the same questions they are measured "
                                    "on, so the best settings lean optimistic. The best value of each setting: section 14.",
                                    ("Not run in this stage: " + "; ".join(missing) + "." if missing else ""))]
    return out


def write(r: Results, event: str = "first") -> str:
    head = [f"# {r.dataset} · {r.scope} · {r.variant} · {r.sample} · {_name(event)}\n", GLOSSARY,
            f"Embedder {r.embedder}. {r.df.qid.nunique()} questions, all evaluated (no dev split). "
            f"Rerankers: {', '.join(r.models) or 'none'}.\n"]
    body = (summary(r, event) + first_stage(r, event) + headroom(r, event) + equal_reader_tokens(r, event) + token_savings(r, event)
            + overfetch(r, event) + equal_reranker_tokens(r, event) + (worth_it(r, event) if event == "every" else [])
            + chunkers(r, event) + query_types(r, event) + help_harm(r, event) + expansion(r, event)
            + expansion_compare(r, event)
            + equal_counts(r, event) + settings(r, event) + best_settings(r, event)
            + (topk_recall(r) if event == "every" else [])
            + (latency(r) if event == "every" else []))
    path = SUMMARY / r.embedder / f"{r.dataset}__{r.scope}__{r.variant}__{r.sample}__{event}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(head + body) + "\n")
    return str(path)


def index() -> str:
    """results/README.md: every simulated run's reports (one per reading, each opening with its
    summary) and its charts folder, regenerated after each report run."""
    from rbench.paths import FIGURES, RESULTS

    lines = ["# Results\n",
             "Generated after each report run (`python -m rbench run <stage>`). Each report opens with a summary of the "
             "questions it answers (the gain at EventMemory's defaults and at the best settings, the break-even LLM "
             "price, the caveats), then the sections; the every-gold report also holds the asker's metric, recall at "
             "top-k (section 16). Charts are in each run's folder.\n"]
    for embedder in sorted(p.name for p in SUMMARY.iterdir() if p.is_dir()):
        runs = sorted({f.name.rsplit("__", 1)[0] for f in (SUMMARY / embedder).glob("*__*.md")})
        if not runs:
            continue
        lines.append(f"## First stage {embedder}\n")
        for run in runs:
            reports = sorted((SUMMARY / embedder).glob(f"{run}__*.md"))
            links = ", ".join(f"[{f.stem.rsplit('__', 1)[1]}](summary/{embedder}/{f.name})" for f in reports)
            charts = FIGURES / embedder / run
            lines.append(f"- **{run.replace('__', ' · ')}**: reports {links}"
                         + (f"; [charts](figures/{embedder}/{run}/)" if charts.exists() else ""))
        lines.append("")
    path = RESULTS / "README.md"
    path.write_text("\n".join(lines) + "\n")
    return str(path)
