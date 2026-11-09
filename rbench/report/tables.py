"""The markdown report for one (embedder, dataset, scope, variant, sample).

Sections, each answering one question (README lists which question maps where):
  1 First stage alone: coverage by reader budget per chunking (the embedding/chunking benchmark)
  2 Headroom: how often the gold is inside the rerank pool at all
  3 Equal reader tokens: first stage vs each model (common depth) vs the gold-first reference
  4 Token savings: first-stage budget needed to match the reranked coverage
  5 Overfetch: coverage as rerank depth grows, with its reranker tokens
  6 Equal reranker tokens: the best chunking and depth within each reranker budget
  7 Worth the money: break-even LLM/reranker price ratio by coverage target
  8 Chunkers: recursive minus fixed at matched size, without and with each reranker
  9 Query types: reranker gain by form, length, context, parts and first-stage difficulty
 10 Help and harm: how often reranking makes the gold cheaper or dearer
 11 Expansion: plain chunks vs small seeds expanded to the same read size, reranked before
    or after expansion (separates the embedded, reranked and read units)
Coverage is over the evaluated (test) questions. Choices among configurations are made on dev
when the dataset has a dev split, else on the same questions (LongMemEval has none, by
decision), and each such section says which. Intervals: stats.interval.
History datasets add the expanded configurations to sections 3 and 6-7 and 12: the first stage
and rerank-then-expand at every replayed expand_context, expand-then-rerank where scored.
"""

from __future__ import annotations

import numpy as np

from rbench.paths import SUMMARY
from rbench.report import frontier
from rbench.report.data import Results, coverage, mean_unit_tokens, query_labels
from rbench.report.stats import delta

BUDGETS = (256, 512, 1024, 2048, 4096, 8192)
EVENTS = ("first", "every", "first_touch", "every_touch")  # complete and touched readings (metrics.py)
DELTA_BUDGETS = (512, 1024, 4096)
RERANK_BUDGETS = (1000, 2000, 4000, 8000, 16000, 32000, 64000)
SHOW_TARGETS = (0.6, 0.7, 0.8, 0.9, 0.95)


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
           "| chunking | mean tokens | " + " | ".join(map(str, BUDGETS)) + " |", "|---|---|" + "---|" * len(BUDGETS)]
    for c in r.chunkings:
        for name in ["first_stage", *(f"first_stage+x{e}" for e in r.expansions)]:
            rows = r.rows(c, name)
            if not rows.empty:
                out.append(f"| {c}{name.removeprefix('first_stage')} | {mean_unit_tokens(r, c)} | "
                           + " | ".join(_cov(rows, event, BUDGETS)) + " |")
    for c in r.chunkings:
        if c.startswith("fix") and "rec" + c[3:] in r.chunkings:
            a, b = r.rows(c, "first_stage"), r.rows("rec" + c[3:], "first_stage")
            out.append(f"| Δ rec−fix {c[3:]} | | " + " | ".join(
                delta((b[event].to_numpy() <= x).astype(float) - (a[event].to_numpy() <= x), a.cluster.to_numpy())
                for x in BUDGETS) + " |")
    return out + [""]


HEADROOM = {"first": "some gold span complete", "every": "every gold span of one answer complete",
            "first_touch": "some gold span touched", "every_touch": "every gold span of one answer touched"}


def headroom(r: Results, event: str) -> list[str]:
    out = ["## 2. Headroom: the gold inside the rerank pool\n",
           "A reranker can only reorder what the first stage retrieved. Share of test queries for which the pool the "
           f"model scored holds {HEADROOM[event]} (so a reranker could deliver it), and the share whose first-ranked "
           "chunk already overlaps gold.\n",
           "| chunking | " + " | ".join(f"{m}: pool pairs / {event} in pool" for m in r.models) + " | first chunk overlaps gold |",
           "|---|" + "---|" * (len(r.models) + 1)]
    for c in r.chunkings:
        base = r.rows(c, "first_stage")
        cells = []
        for m in r.models:
            depth = r.common_depth(c, [m])
            oracle = r.rows(c, f"oracle@{depth}") if depth else None
            rows = r.rows(c, f"rerank:{m}@{depth}") if depth else None
            if oracle is None or oracle.empty:
                cells.append("–")
                continue
            in_pool = (oracle[f"{event}_k"].to_numpy() <= oracle.oracle_pairs.to_numpy()).mean()
            cells.append(f"{rows.pairs.mean():.0f} / {in_pool:.3f}")
        out.append(f"| {c} | " + " | ".join(cells) + f" | {(base['rank'] == 1).mean():.3f} |")
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
    return "dev" if r.has_dev else "the same questions (no dev split)"


def equal_reader_tokens(r: Results, event: str) -> list[str]:
    out = ["## 3. Equal reader tokens\n",
           "Models at the deepest pool all of them were scored to. Δ: minus the first stage without expansion [95%]. "
           "`oracle` is the gold-first reference ordering of the same pool (rankers.py): how much reordering can buy, "
           "not a strict bound. `+xE`: read with EventMemory windows of expand_context E; `rerank` scores the seeds "
           "(rerank, then expand), `xrerank` the windows (expand, then rerank). A reranked row reads only its pool.\n",
           "| chunking | ranker | pairs | " + " | ".join(map(str, BUDGETS)) + " | "
           + " | ".join(f"Δ@{b}" for b in DELTA_BUDGETS) + " |", "|---|---|---|" + "---|" * (len(BUDGETS) + 3)]
    for c in r.chunkings:
        base = r.rows(c, "first_stage")
        depth = r.common_depth(c)
        names = ["first_stage"] + ([f"rerank:{m}@{depth}" for m in r.models_at(c)]
                                   + [f"fuse0.5:{m}@{depth}" for m in r.models_at(c)] + [f"oracle@{depth}"] if depth else [])
        names += _expanded(r, depth, r.models_at(c))
        for name in names:
            rows = r.rows(c, name)
            if rows.empty:
                continue
            ds = [] if name == "first_stage" else [
                delta((rows[event].to_numpy() <= b).astype(float) - (base[event].to_numpy() <= b), base.cluster.to_numpy())
                for b in DELTA_BUDGETS]
            out.append(f"| {c} | {name} | {rows.pairs.mean():.0f} | " + " | ".join(_cov(rows, event, BUDGETS)) + " | "
                       + " | ".join(ds or [""] * len(DELTA_BUDGETS)) + " |")
    return out + [""]


def token_savings(r: Results, event: str) -> list[str]:
    grid = frontier.BUDGET_GRID
    out = ["## 4. Token savings\n", "Reader tokens the first stage alone needs to match the reranked coverage "
           "(× the reranked budget); '>16k' if it never does.\n",
           "| chunking | model | " + " | ".join(f"at {b}" for b in BUDGETS) + " |", "|---|---|" + "---|" * len(BUDGETS)]
    for c in r.chunkings:
        base = coverage(r.rows(c, "first_stage")[event].to_numpy(), grid)
        depth = r.common_depth(c)
        for m in r.models_at(c) if depth else []:
            rr = r.rows(c, f"rerank:{m}@{depth}")[event].to_numpy()
            cells = []
            for b, target in zip(BUDGETS, coverage(rr, np.asarray(BUDGETS)), strict=True):
                hit = np.flatnonzero(base >= target - 1e-12)
                cells.append(">16k" if not len(hit) else f"{grid[hit[0]]:,.0f} ({grid[hit[0]] / b:.1f}×)")
            out.append(f"| {c} | {m} | " + " | ".join(cells) + " |")
    return out + [""]


def overfetch(r: Results, event: str) -> list[str]:
    out = ["## 5. Overfetch\n", "Coverage at 1,024 / 4,096 reader tokens by rerank depth (chunks); reranker "
           "tokens per query in parentheses; '=' where the pool is exhausted.\n"]
    for m in r.models:
        d = r.select(split=None, kind="rerank", model=m)
        depths = sorted({x for x in d.depth if x.startswith("n")}, key=lambda x: int(x[1:]))
        out += [f"**{m}**\n", "| chunking | none | " + " | ".join(depths) + " |", "|---|---|" + "---|" * len(depths)]
        for c in r.chunkings:
            cells, last = [" / ".join(_cov(r.rows(c, "first_stage"), event, (1024, 4096)))], None
            for x in depths:
                rows = r.rows(c, f"rerank:{m}@{x}")
                if rows.empty:
                    cells.append("–")
                elif last is not None and rows.pairs.mean() == last:
                    cells.append("=")
                else:
                    last = rows.pairs.mean()
                    cells.append(" / ".join(_cov(rows, event, (1024, 4096))) + f" ({rows.rr_tokens.mean() / 1000:.1f}k)")
            out.append(f"| {c} | " + " | ".join(cells) + " |")
        out.append("")
    return out


def equal_reranker_tokens(r: Results, event: str) -> list[str]:
    split = "dev" if r.has_dev else "test"
    out = ["## 6. Equal reranker tokens: candidates vs chunk size\n",
           f"Best chunking and reranked configuration (seeds or windows, any expansion, any depth) whose mean reranker "
           f"tokens fit each budget, chosen on {_chooser(r)}; cells are test coverage.\n"]
    test = r.select(expand=None)
    families = sorted({c[:3] for c in r.chunkings})
    for m in r.models:
        out += [f"**{m}**\n", "| reranker tokens | chunker | " + " | ".join(f"reader {b}" for b in (512, 1024, 2048, 4096)) + " |",
                "|---|---|---|---|---|---|"]
        cands = r.select(split=split, model=m, expand=None)
        cands = cands[cands.kind.isin(frontier.RERANKED)]
        cost = cands.groupby(["chunking", "ranker"]).rr_tokens.mean()
        for R in RERANK_BUDGETS:
            for family in ([*families, "any"] if len(families) > 1 else ["any"]):
                cells = []
                for b in (512, 1024, 2048, 4096):
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
                out.append(f"| {R // 1000}k | {family} | " + " | ".join(cells) + " |")
        out.append("")
    return out


def worth_it(r: Results, event: str) -> list[str]:
    pick = ("dev, and r* is measured on test with a 95% interval" if r.has_dev else
            "the same questions they are measured on (no dev split), with a 95% interval whose every resample makes "
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
           "| size | ranker | " + " | ".join(f"Δ@{b}" for b in DELTA_BUDGETS) + " |", "|---|---|---|---|---|"]
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
                for x in DELTA_BUDGETS) + " |")
    return out + [""]


def query_types(r: Results, event: str) -> list[str]:
    out = ["## 9. Query types: reranker gain at 1,024 reader tokens\n",
           "Labels are rule-based (features.py). Difficulty is the first-stage rank of the first gold chunk; it is "
           "defined by the baseline's own outcome, so its rank-1 slice can only lose. Cells: n, first-stage "
           "coverage, Δ [95%].\n"]
    for c in [c for c in ("seg500", "rec64", "fix256", "rec256", "fix1024") if c in r.chunkings] or r.chunkings[:1]:
        _, labels = query_labels(r, c)
        depth = r.common_depth(c)
        pairs = [(model, base_name, f"{kind}:{model}@{depth}{tail}", what) for model in (r.models_at(c) if depth else [])
                 for base_name, kind, tail, what in (
                     ("first_stage", "rerank", "", "seeds reranked, no expansion"),
                     ("first_stage+x3", "xrerank", "+x3", "windows reranked, expand_context 3 (EventMemory with expansion)"))]
        for m, base_name, name, what in pairs:
            rows, base = r.rows(c, name), r.rows(c, base_name)
            if rows.empty or base.empty:
                continue
            hit = base[event].to_numpy() <= 1024
            gain = (rows[event].to_numpy() <= 1024).astype(float) - hit
            out += [f"**{m}, {c}: {what}, against the first stage {base_name.removeprefix('first_stage') or 'without expansion'}**\n",
                    "| feature | value | n | first stage | Δ |", "|---|---|---|---|---|"]
            for feature in ("question_type", "form", "length", "context", "parts", "difficulty"):
                if feature not in labels:
                    continue
                for v in sorted(set(labels[feature])):
                    mask = labels[feature] == v
                    if mask.sum() >= 10:
                        out.append(f"| {feature} | {v} | {int(mask.sum())} | {hit[mask].mean():.3f} | "
                                   f"{delta(gain[mask], base.cluster.to_numpy()[mask])} |")
            out.append("")
    return out


def help_harm(r: Results, event: str) -> list[str]:
    out = ["## 10. Help and harm\n", f"Share of test queries whose {event} cost the reranker lowers or raises, and "
           "that cross 1,024 reader tokens either way.\n",
           "| chunking | model | helped | hurt | lost@1024 | gained@1024 |", "|---|---|---|---|---|---|"]
    for c in r.chunkings:
        depth = r.common_depth(c)
        a = r.rows(c, "first_stage")[event].to_numpy()
        for m in r.models_at(c) if depth else []:
            b = r.rows(c, f"rerank:{m}@{depth}")[event].to_numpy()
            out.append(f"| {c} | {m} | {(b < a).mean():.3f} | {(b > a).mean():.3f} | {((a <= 1024) & (b > 1024)).mean():.3f} | "
                       f"{((a > 1024) & (b <= 1024)).mean():.3f} |")
    return out + [""]


def expansion(r: Results, event: str) -> list[str]:
    from rbench.report.figures import expansion_rows

    groups = expansion_rows(r)
    if not groups:
        return []
    out = ["## 11. Expansion: embedded vs reranked vs read unit\n",
           "Seeds are the smallest expanded chunking. 'Plain' chunks are embedded, reranked and read whole; expanded seeds are embedded small "
           "and read as EventMemory windows. Reader cost counts overlapping windows once.\n",
           "| model | read unit | approach | reranker tokens | " + " | ".join(f"@{b}" for b in (512, 1024, 2048, 4096)) + " |",
           "|---|---|---|---|---|---|---|---|"]
    for m, unit, items in groups:
        for label, c, name in items:
            rows = r.rows(c, name)
            out.append(f"| {m} | ~{unit} | {label} ({c} {name}) | {rows.rr_tokens.mean():,.0f} | "
                       + " | ".join(_cov(rows, event, (512, 1024, 2048, 4096))) + " |")
    return out + [""]


GLOSSARY = """**Metrics** (full definitions: rbench/metrics.py). Every number is a per-query value averaged over
the evaluated (test) queries. A gold span is *complete* when every chunk overlapping it has been read and *touched* when any has
(LongMemEval's notion of a retrieved unit; optimistic on long gold units). *first* / *every*: some span of some
acceptable answer / every span of one answer, complete (`first`, `every`) or touched (`first_touch`,
`every_touch`), within B reader tokens (whole chunks in ranked order). *Recall* is the share of gold characters
read (between the two readings); *precision* the share of characters read that are gold. Intervals are 95% paired
bootstraps over the dataset's clusters (questions for LongMemEval; documents, dialogues or issues elsewhere).
"""


def _recall_cell(rows, column: str, k: int) -> str:
    """Share of questions whose event happened within k admitted segments; '–' when some
    question stopped reading before k without it (its value at k is unknown)."""
    known = (rows[column] <= k) | (rows.admitted >= k)
    return f"{(rows[column] <= k).mean():.3f}" if known.all() else "–"


def session_recall(r: Results) -> list[str]:
    """LongMemEval-style recall at k: an evidence session counts once any of its segments is
    admitted (touched), as LongMemEval's own retrieval metrics count retrieved units. A cell
    is '–' when some question's reading stopped (budget or pool) before k segments without
    reaching the event, since its value at k is then unknown."""
    if "sess_any_k" not in r.df or r.df.sess_any_k.isna().all():
        return []
    ks = (5, 10, 20, 50, 100)
    out = ["## 12. LongMemEval-style session recall@k\n",
           "k counts admitted segments, not sessions, so this is not LongMemEval's published recall@k (k sessions or "
           "rounds). recall_any@k: some evidence session touched within k; recall_all@k: every evidence session touched "
           "within k. '–': some question stopped reading (16k reader tokens, or the end of a reranked pool) before k.\n",
           "| chunking | ranker | " + " | ".join(f"any@{k}" for k in ks) + " | " + " | ".join(f"all@{k}" for k in ks) + " |",
           "|---|---|" + "---|" * (2 * len(ks))]
    for c in r.chunkings:
        depth = r.common_depth(c)
        names = ["first_stage"] + ([f"rerank:{m}@{depth}" for m in r.models_at(c)] if depth else [])
        names += _expanded(r, depth, r.models_at(c))
        for name in names:
            rows = r.rows(c, name)
            if rows.empty:
                continue
            out.append(f"| {c} | {name} | " + " | ".join(_recall_cell(rows, "sess_any_k", k) for k in ks) + " | "
                       + " | ".join(_recall_cell(rows, "sess_all_k", k) for k in ks) + " |")
    return out + [""]


COUNT_LIMITS = (10, 20, 30, 50)  # vector search limits shown (EventMemory's default is 20)
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
        for L in COUNT_LIMITS:
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


SETTING_BUDGETS = (512, 1024, 2048, 4096)


def settings(r: Results, event: str) -> list[str]:
    """What each configuration needs EventMemory set to at a reader budget: max_num_segments
    (segments read within it) and, for the first stage, vector_search_limit (seeds reached)."""
    if "segs_t512" not in r.df:
        return []
    out = ["## 14. EventMemory settings at each reader budget\n",
           f"Each cell: {event}-gold coverage · max_num_segments · seeds read, as the median (10th percentile) over "
           "questions. Segment counts vary with the text, so a limit that keeps 90% of questions within the budget is "
           "the 10th percentile. Without a reranker, the seeds read are the vector_search_limit to set; with one, the "
           "vector_search_limit is the depth in the name (every one of those seeds is reranked).\n",
           "| chunking | ranker | " + " | ".join(f"@{b}" for b in SETTING_BUDGETS) + " |",
           "|---|---|" + "---|" * len(SETTING_BUDGETS)]
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
            for b in SETTING_BUDGETS:
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
            f"Embedder {r.embedder}. {r.test.qid.nunique()} test and {r.df[r.df.split == 'dev'].qid.nunique()} dev queries. "
            f"Rerankers: {', '.join(r.models) or 'none'}.\n"]
    body = (first_stage(r, event) + headroom(r, event) + equal_reader_tokens(r, event) + token_savings(r, event)
            + overfetch(r, event) + equal_reranker_tokens(r, event) + (worth_it(r, event) if event == "every" else [])
            + chunkers(r, event) + query_types(r, event) + help_harm(r, event) + expansion(r, event)
            + (session_recall(r) if event == "every" else []) + equal_counts(r, event) + settings(r, event)
            + (latency(r) if event == "every" else []))
    path = SUMMARY / r.embedder / f"{r.dataset}__{r.scope}__{r.variant}__{r.sample}__{event}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(head + body) + "\n")
    return str(path)
