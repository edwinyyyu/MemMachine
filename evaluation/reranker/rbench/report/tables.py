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

EVENTS = rd.ALL  # the event readings (complete, touched) and the share readings (readings.py)
RERANK_BUDGETS = (1000, 2000, 4000, 8000, 16000, 32000, 64000)
SHOW_TARGETS = (0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def _name(event: str) -> str:
    """A reading's name in prose: first-gold, every-gold, ... or the share of evidence turns."""
    return {"share": "share-of-evidence-turns (complete)", "share_touch": "share-of-evidence-turns (touched)"}.get(
        event, f"{event}-gold")


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
           f"{_name(event)} coverage by reader budget, no reranker"
           + "; `+xE` reads each seed's EventMemory window (expand_context E)"
           + (". Δ rows: the simple chunker minus EventMemory's TextSegmenter, size-matched (the cut's length fitted so "
              "its mean segment matches TextSegmenter's: sizes.py), with both measured sizes. " + SPLITTING
              + " Section 8 gives the touched readings beside this one.\n" if _pairs(r) else ".\n"),
           "Measured unit: the chunker's limit, the mean segment of the partitions searched and of the seeds "
           "retrieved; for windows, the mean window of the seeds retrieved (segment-text tokens).\n"
           if any(c.startswith(("seg", "cut")) for c in r.chunkings) else "",
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
           "A reranker can only reorder the seeds the vector search returned. At each vector search limit L, the share of "
           f"questions for which the top L seeds hold {HEADROOM[event]} (so a reranker could deliver it), with the "
           "reranker's pairs per question; and the share whose first-ranked segment already overlaps gold.\n",
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
           "At each vector search limit L, every row reads only the top L seeds of the same vector search: in vector "
           "order (`first_stage@nL`) or reranked (`rerank`: the seeds; `xrerank`: their windows, expand then rerank). "
           "`+xE`: read with EventMemory windows of expand_context E. Δ: minus the first stage at the same L and expansion "
           "[95%]. `bound` rows are the cheapest reading of the same pool any ranking could achieve, for this reading, with "
           "seeds alone (bounds.py); there is none with windows, since an exact one isn't computable here.\n",
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
    out = ["## 4. Token savings\n", "At each vector search limit L: reader tokens the first stage over the same L seeds "
           "needs to match the reranked coverage (× the reranked budget); 'never' if it does not.\n",
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
    out = ["## 5. Overfetch\n", f"Coverage at {' / '.join(f'{b:,}' for b in budgets)} reader tokens as the vector search "
           "limit (the rerank depth) grows: reranked, then the first stage over the same seeds; reranker tokens per query "
           "in parentheses; '=' where the pool is exhausted.\n"]
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
           f"Best chunking and reranked configuration (seeds or windows, any expansion, any depth) whose mean reranker "
           f"tokens fit each budget, chosen on {_chooser(r)}; cells are coverage.\n"]
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


def worth_it(r: Results, event: str) -> list[str]:
    pick = ("the same questions they are measured on (no dev split), so the best L leans optimistic for the reranker; "
            "the 95% interval resamples questions with each block's L fixed")
    out = ["## 7. Worth the money: break-even price ratio\n",
           f"{_name(event)} coverage targets. Reranking is cheaper overall when (LLM input price) / (reranker price) "
           "exceeds r* (chart: the break-even lines in cost_saving.png). Both sides at one setting "
           f"({frontier.setting(r)[0]}, expand_context {frontier.setting(r)[1]}; config.EVAL.cost_chunking / cost_expand): "
           "one vector search limit L on both sides, held fixed for every target: the reranker over the top L seeds against "
           "the first stage over the same seeds in vector order, each reading as far as the target needs (one block per L: "
           "EventMemory's default, the L whose reranked coverage averaged over the realistic budgets is highest, and the "
           "pool; chosen on "
           f"{pick}). '–': reranking at that L is not cheaper; 'only': only reranking reaches the target (cheaper at any "
           "price); 'n/a': neither reaches it (above the L's ceiling, given per block).\n"]
    fixed = frontier.setting(r)

    def fmt(x: float) -> str:
        return "n/a" if np.isnan(x) else "–" if np.isinf(x) or x >= 1e12 else "only" if x == 0 else f"{x:,.1f}"

    for m in r.models:
        for L, what in frontier.cost_limits(r, m, event):
            api = frontier.break_even(r, m, "rr_tokens", event=event, fixed=fixed, limit=L)
            cached = frontier.break_even(r, m, "rr_prefix_tokens", False, event=event, fixed=fixed, limit=L)
            top = frontier.ceiling(r, L, event)
            out += [f"**{m} at L {L} on both sides ({what}"
                    + (f"; ceiling {top:.2f}: no reading of the top {L} seeds passes it" if top is not None else "") + ")**\n",
                    "| target | r* API [95%] | r* prefix-cached |", "|---|---|---|"]
            for t in SHOW_TARGETS:
                i = int(np.argmin(np.abs(frontier.TARGETS - t)))
                point, lo, hi, _, _ = api
                out.append(f"| {t} | {fmt(point[i])} [{fmt(lo[i])}, {fmt(hi[i])}] | {fmt(cached[0][i])} |")
            out.append("")
    return out + [""]


SPLITTING = ("Equal mean segment size does not mean equal splitting of evidence turns: a cut every M characters "
             "splits every turn longer than M, where TextSegmenter keeps turns up to its limit whole (on LongMemEval-S, "
             "cut354 splits 23% of the evidence turns into several segments, seg500 5%: METHODOLOGY section 3), so "
             "readings in full (every piece of a turn read) favour TextSegmenter and touched readings (any piece) do "
             "not: both are given.")


def chunkers(r: Results, event: str) -> list[str]:
    if not _pairs(r):
        return []
    readings = list(dict.fromkeys([event, *rd.available(r.df, ("every_touch", "share_touch"))]))
    out = ["## 8. Chunkers with a reranker: the simple chunker minus EventMemory's TextSegmenter\n",
           "Size-matched: the cut's length is fitted so its mean segment over the partitions searched matches "
           "TextSegmenter's (sizes.py), since the recursive splitter's segments run well under its limit. Measured mean "
           f"segments of the seeds retrieved are given per pair. {SPLITTING}\n",
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
           "Labels are the dataset's own question types and rule-based features (features.py). Gain: coverage reranked "
           "minus the first stage over the same seeds, at "
           f"{', '.join(f'{b:,}' for b in budgets)} reader tokens. Seeds reranked: at the best vector search limit chosen "
           "once overall (the L whose reranked coverage averaged over these budgets is highest, as in the summary and "
           "query_types.png); the deepest pool would mix in deep-overfetch harm. Windows reranked: at their pool. Cells: "
           "first-stage coverage, Δ [95%].\n"]
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
    out = ["## 10. Help and harm\n", f"At each vector search limit L: the share of questions whose {event} cost the reranker "
           "lowers or raises against the first stage over the same seeds, and that it makes cross each reader budget either "
           "way (lost: within the budget before reranking, not after; gained: the reverse).\n",
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
           f"Seeds are {panels[0].seed}'s segments, embedded small and read as EventMemory windows; 'plain' chunks are "
           "embedded, reranked and read whole. Sizes are measured (mean segment-text tokens per retrieved seed, or its "
           "window); the plain chunking is the one of the same chunker closest to the window's measured size. Reader cost "
           "counts overlapping "
           "windows once.\n",
           "| model | expand_context | approach | measured unit (tokens) | reranker tokens | "
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


GLOSSARY = """**Metrics** (full definitions: rbench/metrics.py). Every number is a per-query value averaged over
the evaluated queries. A gold span is *complete* when every chunk overlapping it has been read and *touched* when any has
(LongMemEval's notion of a retrieved unit; optimistic on long gold units). *first* / *every*: some span of some
acceptable answer / every span of one answer, complete (`first`, `every`) or touched (`first_touch`,
`every_touch`), within B reader tokens (whole chunks in ranked order). *share* / *share_touch*: the share of a
question's evidence turns (spans of the answer read in full first) complete / touched within B, each turn counted once,
averaged over questions; with one evidence turn they equal *every* / *every_touch* (rbench/report/readings.py).
*Recall* is the share of gold characters
read (between the two readings); *precision* the share of characters read that are gold. Intervals are 95% paired
bootstraps over the dataset's clusters (questions for LongMemEval; documents, dialogues or issues elsewhere).
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
           f"{_name(event)} coverage at {', '.join(f'{b:,}' for b in budgets)} reader tokens for each expand_context, at "
           "the cost comparison's vector search limits (EventMemory's default, the best L overall, the pool), for the first "
           "stage, rerank then expand, and expand then rerank (where windows were scored; its x0 row is the seeds "
           f"reranked). {m}. Reader tokens include a window's extra text. Chart: expansion_compare.png.\n",
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
           f"{_name(event)} coverage normalized by counts instead of tokens. L: vector search limit (seeds retrieved; a "
           "reranker scores all L, and nothing beyond L is read). k: seeds kept after reranking (contexts given to the "
           "LLM). m: segments read after expansion (EventMemory's max_num_segments). 'all': every one of the L seeds "
           "read (up to 16k reader tokens). Without a reranker, the same L seeds in similarity order.\n",
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
           f"Each cell: {_name(event)} coverage · max_num_segments · seeds read, as the median (10th percentile) over "
           "questions. Segment counts vary with the text, so a limit that keeps 90% of questions within the budget is "
           "the 10th percentile. Without a reranker, the seeds read are the vector_search_limit to set; with one, the "
           "vector_search_limit is the depth in the name (every one of those seeds is reranked)."
           + (f" Not shown: {', '.join(map(str, skipped))} (segment counts are stored only at the budgets configured when simulated)." if skipped else "")
           + "\n",
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
            "Times of the scoring requests made for this dataset. Not a self-hosted deployment: DeepInfra requests run "
            "concurrently under an adaptive limit (16 at first, up to 150: limits.py) and include queueing and any retries; "
            "local ones are this machine. Per question, EventMemory "
            "would rerank its vector_search_limit seeds in one or a few such requests.\n",
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
           f"Each factor varied alone; the others at the core setting ({core[0]}, expand_context {core[1]}, the reranker "
           "scoring the seeds, at its deepest pool). Per value: the reranked coverage and the first stage's under the "
           "same value. *Best*: the highest reranked coverage. *Cheapest equivalent*: the cheapest value (fewer seeds or "
           "segments, or fewer reranker tokens) whose coverage reaches the best's 95% lower bound. The bound comes from "
           "bootstrap resamples of the questions, each re-choosing the best value: choices are made on the same "
           "questions they are measured on (no dev split), so the best leans optimistic. Every setting is chosen once for "
           "all questions at a budget, never per question, since a service runs one configuration (only the bound rows "
           "are per-question optima, and are labelled so). Factors that need seed counts (k, m) are shown for the event "
           "readings only.\n"]
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
           f"{core[0]}, expand_context 0. Top-k read both ways: **L**, the vector search limit (seeds retrieved, all "
           "reranked), and **k**, the seeds kept after (re)ranking. Recall: the share of a question's evidence turns found "
           "within the first k seeds, in full or touched (LongMemEval's turn recall), and every-gold (all turns in full), "
           "averaged over questions. Without a reranker top-k is both (the first k seeds). **At k = L everything "
           "retrieved is kept, so reranking cannot change recall, only order: the gain there is 0 by construction**, "
           "and reading top-k as L alone the two coincide at every k. L and k are chosen once for all questions. "
           "Δ: reranked minus the first stage [95% interval].\n"]
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
           f"Does reranking improve retrieval enough to justify its cost? {_name(event)} coverage; reader budgets "
           f"{', '.join(f'{b:,}' for b in budgets)} tokens; Δ = reranked minus the first stage over the same seeds "
           "[95% interval over questions]; *helps* / *hurts*: the interval excludes 0.\n"]
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
        from rbench.report.stats import MIN_CLUSTERS

        few = len(np.unique(frontier._clusters(r))) < MIN_CLUSTERS  # no interval, as for the gains
        targets = list(frontier.TARGETS)
        at = []
        for L, what in frontier.cost_limits(r, m, event):
            point, lo, hi, _, _ = frontier.break_even(r, m, "rr_tokens", event=event, fixed=core, limit=L)
            prices = []
            for t in SUMMARY_TARGETS:
                if t not in targets:
                    continue
                k = targets.index(t)
                p, a, z = (x[k] * price(m) for x in (point, lo, hi))
                text = ("never" if not np.isfinite(p) or p >= 1e11 else "at any price" if p == 0 else f"${p:.2f}/M")
                if text.startswith("$") and np.isfinite(a) and np.isfinite(z) and few is False:
                    text += f" [{a:.2f}, {'never' if z >= 1e11 * price(m) else f'{z:.2f}'}]"
                prices.append(f"{t:.0%}: {text}")
            at.append(f"at L {L} ({what}): " + "; ".join(prices))
        out.append(f"Break-even LLM input price (reranking is cheaper above it; {core[0]}, expand_context {core[1]}, "
                   f"{m} at ${price(m):g}/M, the same L on both sides at every target): " + " | ".join(at)
                   + ". Section 7 and cost_saving.png give every target.\n")
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
            out.append(f"Recall at top-k ({m}; evidence turns found in full; section 16). Top-k as the vector search "
                       "limit, everything kept: reranking cannot change recall (only order), a gain of 0 by construction. "
                       "Top-k as the seeds kept after reranking the default 20: " + ("; ".join(parts) or "–") + ".\n")
    pool = r.pool_depth(core[0], r.models[0], "rerank")
    bound = r.rows(core[0], f"bound@{pool}") if pool else None
    from rbench import depth

    shares = depth.pool_shares(r.embedder, r.dataset, core[0], r.sample, 0, int(pool[1:])) if pool else None
    fills = (f"; reading it reaches {config.EVAL.token_top:,} reader tokens for {shares[0]:.0%} of questions"
             if shares else "")
    ceiling = (f" The reranker's pool ({pool[1:]} seeds, fixed) holds all the evidence for "
               f"{rd.reachable(rd.costs(bound, event)).mean():.0%} of questions (the bound), so no reranker of it can do "
               f"better{fills}." if bound is not None and not bound.empty else "")
    out += ["The best value of each factor per budget, one at a time: section 14. Every setting is chosen once for all questions "
            "(never per question), on the same questions it is measured on (no dev split): the best-settings line leans "
            "optimistic." + ceiling + "\n"]
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
