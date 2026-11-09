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
from rbench.report.data import Results, coverage, mean_unit_tokens, query_labels
from rbench.report.stats import delta

EVENTS = ("first", "every", "first_touch", "every_touch")  # complete and touched readings (metrics.py)
RERANK_BUDGETS = (1000, 2000, 4000, 8000, 16000, 32000, 64000)
SHOW_TARGETS = (0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def _cov(rows, event: str, budgets) -> list[str]:
    return [f"{x:.3f}" for x in coverage(rows[event].to_numpy(), np.asarray(budgets))]


def _pairs(r: Results) -> list[str]:
    """Fixed-window chunkings that have a size-matched recursive one (flat datasets)."""
    return [c for c in r.chunkings if c.startswith("fix") and "rec" + c[3:] in r.chunkings]


def first_stage(r: Results, event: str) -> list[str]:
    out = ["## 1. First stage alone (embedding and chunking)\n",
           f"{event}-gold coverage by reader budget, no reranker"
           + (". Δ rows: recursive minus fixed at matched size.\n" if _pairs(r) else
              "; `+xE` reads each seed's EventMemory window (expand_context E).\n"),
           "| chunking | mean tokens | " + " | ".join(map(str, config.EVAL.columns)) + " |", "|---|---|" + "---|" * len(config.EVAL.columns)]
    for c in r.chunkings:
        for name in ["first_stage", *(f"first_stage+x{e}" for e in r.expansions)]:
            rows = r.rows(c, name)
            if not rows.empty:
                out.append(f"| {c}{name.removeprefix('first_stage')} | {mean_unit_tokens(r, c)} | "
                           + " | ".join(_cov(rows, event, config.EVAL.columns)) + " |")
    for c in r.chunkings:
        if c.startswith("fix") and "rec" + c[3:] in r.chunkings:
            a, b = r.rows(c, "first_stage"), r.rows("rec" + c[3:], "first_stage")
            out.append(f"| Δ rec−fix {c[3:]} | | " + " | ".join(
                delta((b[event].to_numpy() <= x).astype(float) - (a[event].to_numpy() <= x), a.cluster.to_numpy())
                for x in config.EVAL.columns) + " |")
    return out + [""]


HEADROOM = {"first": "some gold span complete", "every": "every gold span of one answer complete",
            "first_touch": "some gold span touched", "every_touch": "every gold span of one answer touched"}


def headroom(r: Results, event: str) -> list[str]:
    out = ["## 2. Headroom: the gold among the seeds retrieved\n",
           "A reranker can only reorder the seeds the vector search returned. At each vector search limit L, the share of "
           f"questions for which the top L seeds hold {HEADROOM[event]} (so a reranker could deliver it), with the "
           "reranker's pairs per question; and the share whose first-ranked segment already overlaps gold.\n",
           "| chunking | L | " + " | ".join(f"{m}: pairs / {event} among the L seeds" for m in r.models)
           + " | first segment overlaps gold |", "|---|---|" + "---|" * (len(r.models) + 1)]
    for c in r.chunkings:
        first = (r.rows(c, "first_stage")["rank"] == 1).mean()
        for limit in r.limits(c):
            oracle = r.rows(c, f"oracle@{limit}")
            if oracle.empty:
                continue
            in_pool = (oracle[f"{event}_k"].to_numpy() <= oracle.oracle_pairs.to_numpy()).mean()
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
           "[95%]. In the every-gold table only, `oracle` rows (seeds alone) are a perfect reranker: it reads only the "
           "evidence seeds, so no reranker reads all the evidence in full with fewer tokens; elsewhere it is no bound and "
           "is not shown.\n",
           "| chunking | ranker | pairs | " + " | ".join(map(str, config.EVAL.columns)) + " | "
           + " | ".join(f"Δ@{b}" for b in config.EVAL.budgets) + " |", "|---|---|---|" + "---|" * (len(config.EVAL.columns) + len(config.EVAL.budgets))]
    for c in r.chunkings:
        for limit in r.limits(c):
            for e in (0, *r.expansions):
                tail = f"+x{e}" if e else ""
                base = r.rows(c, f"first_stage@{limit}{tail}")
                names = ([f"first_stage@{limit}{tail}"] + [f"rerank:{m}@{limit}{tail}" for m in r.models_at(c)]
                         + ([f"xrerank:{m}@{limit}{tail}" for m in r.models_at(c)] if e else [])
                         + ([f"oracle@{limit}"] if event == "every" and not e else []))
                for name in names:
                    rows = r.rows(c, name)
                    if rows.empty or base.empty:
                        continue
                    ds = [] if name.startswith("first_stage") else [
                        delta((rows[event].to_numpy() <= b).astype(float) - (base[event].to_numpy() <= b),
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
        for limit in r.limits(c):
            base = coverage(r.rows(c, f"first_stage@{limit}")[event].to_numpy(), grid)
            for m in r.models_at(c):
                rows = r.rows(c, f"rerank:{m}@{limit}")
                if rows.empty:
                    continue
                cells = []
                for b, target in zip(config.EVAL.columns, coverage(rows[event].to_numpy(), np.asarray(config.EVAL.columns)), strict=True):
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
                "|---|---|---|---|---|---|"]
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
                        score = (cands[(cands.chunking == c) & (cands.ranker == name)][event] <= b).mean()
                        if best is None or score > best[0]:
                            best = (score, c, name)
                    if best is None:
                        cells.append("–")
                    else:
                        t = test[(test.chunking == best[1]) & (test.ranker == best[2])]
                        cells.append(f"{(t[event] <= b).mean():.3f} ({best[1]} {best[2].replace(f':{m}', '')})")
                out.append(f"| {R:,} | {family} | " + " | ".join(cells) + " |")
        out.append("")
    return out


def worth_it(r: Results, event: str) -> list[str]:
    pick = ("the same questions they are measured on (no dev split), with a 95% interval whose every resample makes "
            "its own choice; the minimum over many reranked configurations leans optimistic for the reranker")
    out = ["## 7. Worth the money: break-even price ratio\n",
           f"{event}-gold coverage targets. Reranking is cheaper overall when (LLM input price) / (reranker price) "
           "exceeds r* (chart: break_even.png). Candidates: every no-reranker configuration (segment length, "
           "expansion) against every reranked one (seeds or windows, expansion, depth). Configurations are chosen on "
           f"{pick}. '–': the chosen reranked configuration is not cheaper; 'only': only reranking reaches the target "
           "(cheaper at any price); 'n/a': nothing reaches it.\n",
           "| target | " + " | ".join(f"{m}: r* API [95%] · prefix-cached · chosen" for m in r.models) + " |",
           "|---|" + "---|" * len(r.models)]
    results = {m: (frontier.break_even(r, m, "rr_tokens", event=event),
                   frontier.break_even(r, m, "rr_prefix_tokens", False, event=event)) for m in r.models}

    def fmt(x: float) -> str:
        return "n/a" if np.isnan(x) else "–" if np.isinf(x) or x >= 1e12 else "only" if x == 0 else f"{x:,.1f}"

    for t in SHOW_TARGETS:
        i = int(np.argmin(np.abs(frontier.TARGETS - t)))
        cells = []
        for (api, cached) in results.values():
            point, lo, hi, base_names, rr_names = api
            chosen = " ".join(base_names[i]) + (" vs " + " ".join(rr_names[i]) if rr_names[i] else "")
            cells.append(f"{fmt(point[i])} [{fmt(lo[i])}, {fmt(hi[i])}] · {fmt(cached[0][i])} · {chosen}")
        out.append(f"| {t} | " + " | ".join(cells) + " |")
    return out + [""]


def chunkers(r: Results, event: str) -> list[str]:
    if not _pairs(r):
        return []  # history datasets have one segmenter (EventMemory's)
    out = ["## 8. Chunkers with a reranker: recursive minus fixed\n",
           "| size | ranker | " + " | ".join(f"Δ@{b}" for b in config.EVAL.budgets) + " |", "|---|---|---|---|---|"]
    for c in r.chunkings:
        other = "rec" + c[3:]
        if not c.startswith("fix") or other not in r.chunkings:
            continue
        depth = r.common_depth(c)
        shared = [m for m in r.models_at(c) if m in r.models_at(other)]
        depth = r.common_depth(c, shared) if shared else None
        for name in ["first_stage"] + ([f"rerank:{m}@{depth}" for m in shared] if depth else []):
            a, b = r.rows(c, name), r.rows(other, name)
            if a.empty or b.empty:
                continue
            out.append(f"| {c[3:]} | {name} | " + " | ".join(
                delta((b[event].to_numpy() <= x).astype(float) - (a[event].to_numpy() <= x), a.cluster.to_numpy())
                for x in config.EVAL.budgets) + " |")
    return out + [""]


def query_types(r: Results, event: str) -> list[str]:
    budgets = config.EVAL.budgets
    out = ["## 9. Query types: reranker gain\n",
           "Labels are the dataset's own question types and rule-based features (features.py). Gain: coverage reranked "
           "minus the first stage over the same seeds (vector search limit 20, EventMemory's default, where scored; else "
           f"the deepest), at {', '.join(f'{b:,}' for b in budgets)} reader tokens. Cells: first-stage coverage, Δ [95%].\n"]
    for c in r.chunkings:
        limits = r.limits(c)
        if not limits:
            continue
        limit = "n20" if "n20" in limits else limits[-1]
        for e, kind, what in ((0, "rerank", "seeds reranked"), (3, "xrerank", "windows reranked, expand_context 3")):
            tail = f"+x{e}" if e else ""
            base_name = f"first_stage@{limit}{tail}"
            base, labels = query_labels(r, c, base_name)
            for m in r.models_at(c):
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
                            hit = base[event].to_numpy() <= b
                            gain = (rows[event].to_numpy() <= b).astype(float) - hit
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
        for limit in r.limits(c):
            a = r.rows(c, f"first_stage@{limit}")[event].to_numpy()
            for m in r.models_at(c):
                rows = r.rows(c, f"rerank:{m}@{limit}")
                if rows.empty:
                    continue
                b = rows[event].to_numpy()
                cells = [f"{((a <= x) & (b > x)).mean():.3f} / {((a > x) & (b <= x)).mean():.3f}" for x in budgets]
                out.append(f"| {c} | {m} | {limit[1:]} | {(b < a).mean():.3f} | {(b > a).mean():.3f} | " + " | ".join(cells) + " |")
    return out + [""]


def expansion(r: Results, event: str) -> list[str]:
    from rbench.report.figures import expansion_rows

    groups = expansion_rows(r)
    if not groups:
        return []
    out = ["## 11. Expansion: embedded vs reranked vs read unit\n",
           "Seeds are the smallest expanded chunking. 'Plain' chunks are embedded, reranked and read whole; expanded seeds are embedded small "
           "and read as EventMemory windows. Reader cost counts overlapping windows once.\n",
           "| model | read unit | approach | reranker tokens | " + " | ".join(f"@{b}" for b in config.EVAL.budgets) + " |",
           "|---|---|---|---|---|---|---|---|"]
    for m, unit, items in groups:
        for label, c, name in items:
            rows = r.rows(c, name)
            out.append(f"| {m} | ~{unit} | {label} ({c} {name}) | {rows.rr_tokens.mean():,.0f} | "
                       + " | ".join(_cov(rows, event, config.EVAL.budgets)) + " |")
    return out + [""]


GLOSSARY = """**Metrics** (full definitions: rbench/metrics.py). Every number is a per-query value averaged over
the evaluated (test) queries. A gold span is *complete* when every chunk overlapping it has been read and *touched* when any has
(LongMemEval's notion of a retrieved unit; optimistic on long gold units). *first* / *every*: some span of some
acceptable answer / every span of one answer, complete (`first`, `every`) or touched (`first_touch`,
`every_touch`), within B reader tokens (whole chunks in ranked order). *Recall* is the share of gold characters
read (between the two readings); *precision* the share of characters read that are gold. Intervals are 95% paired
bootstraps over the dataset's clusters (questions for LongMemEval; documents, dialogues or issues elsewhere).
"""


KEPT, SEGMENTS = (5, 10), (10, 20, 50)


def _count_rows(r: Results, c: str, L: int) -> list[str]:
    names = [f"first_stage@n{L}", *(f"rerank:{m}@n{L}" for m in r.models_at(c))]
    for e in [3] if 3 in r.expansions else []:
        names += [f"first_stage@n{L}+x{e}", *(f"{k}:{m}@n{L}+x{e}" for m in r.models_at(c) for k in ("rerank", "xrerank"))]
    return names


def equal_counts(r: Results, event: str) -> list[str]:
    """Coverage at equal counts: the vector search limit L (seeds retrieved, all of them
    reranked), the seeds kept after reranking k, and the segments read after expansion m."""
    if f"{event}_ctx" not in r.df:
        return []
    out = ["## 13. Equal counts: vector search limit, seeds kept, segments read\n",
           f"{event}-gold coverage normalized by counts instead of tokens. L: vector search limit (seeds retrieved; a "
           "reranker scores all L, and nothing beyond L is read). k: seeds kept after reranking (contexts given to the "
           "LLM). m: segments read after expansion (EventMemory's max_num_segments). 'all': every one of the L seeds "
           "read (up to 16k reader tokens). Without a reranker, the same L seeds in similarity order.\n",
           "| chunking | L | ranker | " + " | ".join(f"k≤{k}" for k in KEPT) + " | " + " | ".join(f"m≤{m}" for m in SEGMENTS)
           + " | all |", "|---|---|---|" + "---|" * (len(KEPT) + len(SEGMENTS) + 1)]
    for c in r.chunkings:
        for L in (int(x[1:]) for x in r.limits(c)):
            for name in _count_rows(r, c, L):
                rows = r.rows(c, name)
                if rows.empty:
                    continue
                cells = [f"{(rows[f'{event}_ctx'] <= k).mean():.3f}" for k in KEPT]
                cells += [f"{(rows[f'{event}_k'] <= m).mean():.3f}" for m in SEGMENTS]
                cells.append(f"{np.isfinite(rows[event]).mean():.3f}")
                out.append(f"| {c} | {L} | {name.split('@')[0]}{'+x' + name.split('+x')[1] if '+x' in name else ''} | "
                           + " | ".join(cells) + " |")
    return out + [""]




def settings(r: Results, event: str) -> list[str]:
    """What each configuration needs EventMemory set to at a reader budget: max_num_segments
    (segments read within it) and, for the first stage, vector_search_limit (seeds reached)."""
    budgets = [b for b in config.EVAL.budgets if f"segs_t{b}" in r.df]  # counts are stored at metrics.BUDGETS only
    if not budgets:
        return []
    skipped = [b for b in config.EVAL.budgets if b not in budgets]
    out = ["## 14. EventMemory settings at each reader budget\n",
           f"Each cell: {event}-gold coverage · max_num_segments · seeds read, as the median (10th percentile) over "
           "questions. Segment counts vary with the text, so a limit that keeps 90% of questions within the budget is "
           "the 10th percentile. Without a reranker, the seeds read are the vector_search_limit to set; with one, the "
           "vector_search_limit is the depth in the name (every one of those seeds is reranked)."
           + (f" Not shown: {', '.join(map(str, skipped))} (segment counts are stored at metrics.BUDGETS only)." if skipped else "")
           + "\n",
           "| chunking | ranker | " + " | ".join(f"@{b}" for b in budgets) + " |",
           "|---|---|" + "---|" * len(budgets)]
    for c in r.chunkings:
        depth = r.common_depth(c)
        names = ["first_stage", *(f"first_stage+x{e}" for e in r.expansions if e == 3)]
        for m in r.models_at(c):
            for d in sorted({x for x in ("n20", depth) if x}, key=lambda x: int(x[1:])):
                names += [f"rerank:{m}@{d}", f"xrerank:{m}@{d}+x3"]
        for name in names:
            rows = r.rows(c, name)
            if rows.empty:
                continue
            cells = []
            for b in budgets:
                segs, ctx = rows[f"segs_t{b}"], rows[f"ctx_t{b}"]
                cells.append(f"{(rows[event] <= b).mean():.2f} · {segs.median():.0f} ({segs.quantile(0.1):.0f}) · "
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
    return ["## 15. Reranker latency (a by-product)\n",
            "Times of the scoring requests made for this dataset. Not a self-hosted deployment: DeepInfra requests run 16 "
            "at a time and include queueing and any retries; local ones are this machine. Per question, EventMemory "
            "would rerank its vector_search_limit seeds in one or a few such requests.\n",
            "| model | requests | pairs per request (median) | tokens per request | seconds (median) | seconds (p90) | "
            "pairs per second (median) |", "|---|---|---|---|---|---|---|", *rows, ""]


def write(r: Results, event: str = "first") -> str:
    head = [f"# {r.dataset} · {r.scope} · {r.variant} · {r.sample} · {event} gold\n", GLOSSARY,
            f"Embedder {r.embedder}. {r.df.qid.nunique()} questions, all evaluated (no dev split). "
            f"Rerankers: {', '.join(r.models) or 'none'}.\n"]
    body = (first_stage(r, event) + headroom(r, event) + equal_reader_tokens(r, event) + token_savings(r, event)
            + overfetch(r, event) + equal_reranker_tokens(r, event) + (worth_it(r, event) if event == "every" else [])
            + chunkers(r, event) + query_types(r, event) + help_harm(r, event) + expansion(r, event)
            + equal_counts(r, event) + settings(r, event)
            + (latency(r) if event == "every" else []))
    path = SUMMARY / r.embedder / f"{r.dataset}__{r.scope}__{r.variant}__{r.sample}__{event}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(head + body) + "\n")
    return str(path)
