"""Charts for one (embedder, dataset, scope, variant, sample); the markdown is their table view.

  coverage_tokens_{first,every}.png  coverage vs reader tokens, panel per chunking: first stage,
                                     each model at the common depth, the gold-first reference,
                                     and dashed, the same with expand_context 3 (windows reranked)
  rpf_tokens.png, rpf_k.png          gold-character recall, precision and F1 vs tokens and vs k
  rerank_cost.png                    coverage at fixed reader budgets vs reranker tokens as
                                     overfetch grows; one line per chunking, so equal-x points
                                     compare many small chunks with few large ones
  break_even.png                     the LLM input price above which reranking is cheaper overall,
                                     vs coverage target (API billing); and the same as a price
                                     ratio for a self-hosted reranker with a prefix cache
  cost_saving.png                    % saved by allowing a reranker, over LLM price x coverage target
  query_types.png                    reranker gain at 1,024 tokens by query type, 95% intervals
  first_stage_sizes.png              first-stage coverage vs chunk size at fixed reader budgets, per
                                     chunker: where an optimal embedding chunk size shows
  expansion.png                      plain chunks vs small seeds expanded to the same read size, with
                                     no reranker, reranking before expansion, and after it
  coverage_counts.png                coverage at equal counts: seeds kept after reranking, segments
                                     read, and 10 seeds kept as the vector search limit grows

Colors follow the dataviz reference palette and were checked with its validator: models
take categorical slots in fixed order (first stage, 0.6B, 4B, 8B, then the local 0.6B); the
gold-first reference is neutral gray; chunk sizes use a one-hue ordinal ramp (fixed solid with circles,
recursive dashed with squares); savings use a diverging blue (saves) / red (costs more)
scale around a neutral gray midpoint.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

from rbench.metrics import BUDGETS, KS  # noqa: E402
from rbench.paths import FIGURES  # noqa: E402
from rbench.report import frontier  # noqa: E402
from rbench.report.data import Results, coverage, mean_unit_tokens, price, query_labels  # noqa: E402
from rbench.report.stats import interval  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"first_stage": "#2a78d6", "q06": "#eb6834", "q4": "#1baf7a", "q8": "#eda100", "q06-local": "#e87ba4"}
ORACLE = "#8f8e89"
LABEL = {"first_stage": "first stage only", "q06-local": "Qwen3-Reranker-0.6B (local)", "q06": "Qwen3-Reranker-0.6B",
         "q4": "Qwen3-Reranker-4B", "q8": "Qwen3-Reranker-8B", "oracle": "gold-first reference (same pool)"}
RAMP = {3: ("#86b6ef", "#2a78d6", "#104281"), 5: ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")}
DIVERGING = LinearSegmentedColormap.from_list("red_gray_blue", ["#a12a2a", "#e34948", "#f2b8b5", "#f0efec", "#9ec5f4",
                                                                  "#3987e5", "#104281"])  # equal steps per arm
LLM_PRICES = {"$0.10": 0.1, "$1": 1.0, "$3": 3.0, "$15": 15.0}
PRIMARY = "every"  # single-event charts: every gold turn read (all of a question's evidence is needed)
SHOWN_EXPANSION = 3  # coverage and recall charts add this expand_context, dashed, when simulated

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 1, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titlecolor": INK, "lines.linewidth": 2, "lines.solid_capstyle": "round",
    "legend.frameon": False, "legend.labelcolor": INK2,
})


def _human(v: float) -> str:
    return f"{v / 1000:g}k" if v >= 10000 else f"{v:g}"


def _log_x(ax, base: int = 10, ticks=None) -> None:
    ax.set_xscale("log", base=base)
    if ticks is not None:
        ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _human(v)))
    ax.xaxis.set_minor_formatter(NullFormatter())


def _log_y(ax) -> None:
    """Log y axis labelled at 1, 2 and 5 of each decade, so a range within one decade still
    has numbers."""
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def _save(fig, path, title: str, handles=None, labels=None, ncol: int = 4) -> None:
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    if handles:
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=ncol, fontsize=8.5)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _rankers(r: Results, c: str) -> list[tuple[str, str, str, str]]:
    """(legend key, ranker, color, line style): solid without expansion; dashed with
    EventMemory windows (the first stage expanded; expand, then rerank)."""
    depth, e = r.common_depth(c), SHOWN_EXPANSION
    out = [("first_stage", "first_stage", SERIES["first_stage"], "-")]
    if depth:
        out += [(m, f"rerank:{m}@{depth}", SERIES[m], "-") for m in r.models_at(c)] + [("oracle", f"oracle@{depth}", ORACLE, "-")]
    if e in r.expansions:
        out.append((f"first_stage+x{e}", f"first_stage+x{e}", SERIES["first_stage"], "--"))
        if depth:
            out += [(f"{m}+x{e}", f"xrerank:{m}@{depth}+x{e}", SERIES[m], "--") for m in r.models_at(c)]
    return [x for x in out if not r.rows(c, x[1]).empty]


def _label(key: str) -> str:
    if "+x" not in key:
        return LABEL[key]
    base, e = key.split("+x")
    return f"{LABEL[base]}, expand_context {e}" + (", windows reranked" if base != "first_stage" else "")


def _keys(r: Results, handles: dict) -> list[str]:
    e = SHOWN_EXPANSION
    order = ("first_stage", *r.models, "oracle", f"first_stage+x{e}", *(f"{m}+x{e}" for m in r.models))
    return [k for k in order if k in handles]


def coverage_by_tokens(r: Results, out, head: str) -> None:
    budgets = frontier.BUDGET_GRID[frontier.BUDGET_GRID >= 128]
    fams = sorted({c[:3] for c in r.chunkings})
    sizes = sorted({c[3:] for c in r.chunkings}, key=int)
    for event in ("first", "every", "first_touch", "every_touch"):
        fig, axes = plt.subplots(len(fams), len(sizes), figsize=(3 * len(sizes), 2.5 * len(fams) + 0.9), squeeze=False,
                                 sharex=True, sharey=True)
        handles = {}
        for i, fam in enumerate(fams):
            for j, size in enumerate(sizes):
                ax, c = axes[i][j], fam + size
                if c not in r.chunkings:
                    ax.set_visible(False)
                    continue
                for key, name, color, ls in _rankers(r, c):
                    y = coverage(r.rows(c, name)[event].to_numpy(), budgets)
                    (handles[key],) = ax.plot(budgets, y, color=color, linestyle=ls, linewidth=1.5 if key == "oracle" else 2)
                _log_x(ax, 2, [128, 512, 2048, 8192])
                ax.set_ylim(0, 1.02)
                ax.set_title(f"{c} (mean {mean_unit_tokens(r, c)})", loc="left")
                if i == len(fams) - 1:
                    ax.set_xlabel("reader tokens")
                if j == 0:
                    ax.set_ylabel(f"{event.replace('_', ' ')} coverage")
        keys = _keys(r, handles)
        fig.tight_layout(rect=(0, 0, 1, 0.86))
        _save(fig, out / f"coverage_tokens_{event}.png", f"Coverage ({event.replace('_', ' ')}) vs reader tokens — {head}",
              [handles[k] for k in keys], [_label(k) for k in keys], ncol=3)


def recall_precision_f1(r: Results, out, head: str) -> None:
    for axis, fmt, xs, xlabel, ticks in (("tokens", "t{}", BUDGETS, "reader tokens", [128, 512, 2048, 8192]),
                                         ("k", "k{}", KS, "chunks read (k)", [1, 2, 5, 10, 20, 50])):
        fig, axes = plt.subplots(3, len(r.chunkings), figsize=(2.7 * len(r.chunkings), 7.4), squeeze=False, sharex=True)
        handles = {}
        for j, c in enumerate(r.chunkings):
            for key, name, color, ls in _rankers(r, c):
                rows = r.rows(c, name)
                rec = np.stack([rows[f"recall_{fmt.format(v)}"].to_numpy() for v in xs], 1)
                pre = np.stack([rows[f"precision_{fmt.format(v)}"].to_numpy() for v in xs], 1)
                f1 = np.where(rec + pre > 0, 2 * rec * pre / np.maximum(rec + pre, 1e-12), 0.0)
                for i, values in enumerate((rec, pre, f1)):
                    (handles[key],) = axes[i][j].plot(xs, values.mean(0), color=color, linestyle=ls, marker="o", markersize=4,
                                                      markeredgecolor=SURFACE, linewidth=1.5 if key == "oracle" else 2)
            axes[0][j].set_title(c, loc="left")
            for i, label in enumerate(("recall", "precision", "F1")):
                _log_x(axes[i][j], 2, ticks)
                if j == 0:
                    axes[i][j].set_ylabel(label)
            axes[2][j].set_xlabel(xlabel)
        for row in axes:
            top = max(ax.get_ylim()[1] for ax in row)
            for ax in row:
                ax.set_ylim(0, top)
        keys = _keys(r, handles)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        _save(fig, out / f"rpf_{axis}.png", f"Gold-character recall, precision and F1 vs {xlabel} — {head}",
              [handles[k] for k in keys], [_label(k) for k in keys], ncol=3)


def _size_styles(chunkings: list[str]) -> dict[str, dict]:
    sizes = sorted({int(c[3:]) for c in chunkings})
    ramp = RAMP[3] if len(sizes) <= 3 else RAMP[5]
    color = dict(zip(sizes[-len(ramp):], ramp))  # beyond 5 sizes, the smallest go unplotted
    return {c: {"color": color[int(c[3:])], "ls": "-" if c.startswith("fix") else "--",
                "marker": "o" if c.startswith("fix") else "s"} for c in chunkings if int(c[3:]) in color}


def rerank_cost(r: Results, out, head: str) -> None:
    if not r.models:
        return
    styles, budgets = _size_styles(r.chunkings), (1024, 4096)
    fig, axes = plt.subplots(len(r.models), 2, figsize=(9.2, 3.2 * len(r.models) + 0.9), squeeze=False, sharey=True)
    handles = {}
    for i, m in enumerate(r.models):
        for j, b in enumerate(budgets):
            ax = axes[i][j]
            for c, s in styles.items():
                rows = r.select(kind="rerank", model=m)
                rows = rows[rows.chunking == c]
                pts = sorted({(g.rr_tokens.mean(), (g[PRIMARY] <= b).mean()) for _, g in rows.groupby("ranker")
                              if g.depth.iloc[0].startswith("n")})
                if not pts:
                    continue
                xs, ys = zip(*pts, strict=True)
                base = (r.rows(c, "first_stage")[PRIMARY] <= b).mean()
                ax.plot([xs[0] / 2.2, xs[0]], [base, ys[0]], color=s["color"], linestyle=":", linewidth=1)
                ax.plot([xs[0] / 2.2], [base], marker=s["marker"], color=s["color"], markerfacecolor=SURFACE,
                        markersize=6, markeredgewidth=1.5, linestyle="none")
                (handles[c],) = ax.plot(xs, ys, color=s["color"], linestyle=s["ls"], marker=s["marker"], markersize=5,
                                        markeredgecolor=SURFACE, markeredgewidth=1.2)
            _log_x(ax)
            ax.set_title(f"{LABEL[m]} · reader budget {b}", loc="left")
            ax.set_xlabel("reranker tokens per query (API billing)")
            if j == 0:
                ax.set_ylabel(f"{PRIMARY}-gold coverage")
    keys = [c for c in r.chunkings if c in handles]
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, out / "rerank_cost.png", f"Coverage vs reranker tokens as overfetch grows (hollow: no reranker) — {head}",
          [handles[k] for k in keys], keys, ncol=6)


def break_even(r: Results, out, head: str) -> None:
    if not r.models:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), squeeze=False)
    handles = {}
    for m in r.models:
        api, lo, hi, _, _ = frontier.break_even(r, m, "rr_tokens", event=PRIMARY)
        cached = frontier.break_even(r, m, "rr_prefix_tokens", with_interval=False, event=PRIMARY)[0]
        finite = lambda v: np.where(np.isfinite(v) & (v > 0) & (v < 1e12), v, np.nan)  # noqa: E731
        ax = axes[0][0]
        (handles[m],) = ax.step(frontier.TARGETS, finite(api) * price(m), where="mid", color=SERIES[m])
        ax.fill_between(frontier.TARGETS, finite(lo) * price(m), finite(hi) * price(m), step="mid", color=SERIES[m],
                        alpha=0.15, linewidth=0)
        axes[0][1].step(frontier.TARGETS, finite(cached), where="mid", color=SERIES[m])
    inside = blended_transform_factory(axes[0][0].transAxes, axes[0][0].transData)  # x in the panel, y a price
    for name, value in LLM_PRICES.items():
        axes[0][0].axhline(value, color=INK2, linewidth=0.8, alpha=0.5)
        axes[0][0].text(0.01, value * 1.08, f"LLM input {name}/M", fontsize=7.5, color=INK2, transform=inside)
    _log_y(axes[0][0])
    axes[0][0].set_ylabel("break-even LLM input price ($/M tokens)")
    axes[0][0].set_title("API billing at DeepInfra prices: reranking pays above the line", loc="left")
    _log_y(axes[0][1])
    axes[0][1].set_ylabel("break-even ratio (LLM price ÷ reranker price)")
    axes[0][1].set_title("Self-hosted with a prefix cache", loc="left")
    for ax in axes[0]:
        ax.set_xlabel(f"{PRIMARY}-gold coverage target")
    split = "chosen on dev, measured on test" if r.has_dev else "chosen and measured on the same questions"
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    _save(fig, out / "break_even.png", f"When is a reranker worth it? ({split}; band: 95%; gaps: never cheaper) — {head}",
          [handles[m] for m in r.models], [LABEL[m] for m in r.models])


def cost_saving(r: Results, out, head: str) -> None:
    if not r.models:
        return
    llm = np.logspace(-2, 2, 81)  # $ per million reader tokens
    fig, axes = plt.subplots(1, len(r.models), figsize=(3.6 * len(r.models) + 0.8, 3.6), squeeze=False, sharey=True)
    for k, m in enumerate(r.models):
        ratio = frontier.cost_ratio(r, m, llm / price(m), "rr_tokens", event=PRIMARY)
        ax = axes[0][k]
        im = ax.pcolormesh(llm, frontier.TARGETS, 100 * (1 - ratio), cmap=DIVERGING, norm=TwoSlopeNorm(0, -50, 50),
                           shading="nearest")
        ax.set_facecolor(SURFACE)  # blank where the first stage alone never reaches the target
        _log_x(ax)
        ax.grid(False)
        ax.set_title(LABEL[m], loc="left")
        ax.set_xlabel("LLM input price ($/M tokens)")
        if k == 0:
            ax.set_ylabel(f"{PRIMARY}-gold coverage target")
    fig.colorbar(im, ax=axes[0].tolist(), label="% cost saved by allowing the reranker (red: costs more)")
    split = "chosen on dev, measured on test" if r.has_dev else "chosen and measured on the same questions"
    _save(fig, out / "cost_saving.png", f"Cost saved by reranking ({split}; blank: first stage never reaches the target) — {head}")


def query_types(r: Results, out, head: str) -> None:
    if not r.models or r.test.qid.nunique() < 20:
        return
    c = next((c for c in ("seg500", "rec64", "fix256", "rec256", "fix1024") if c in r.chunkings), r.chunkings[0])
    depth = r.common_depth(c)
    if not depth:
        return
    base, labels = query_labels(r, c)
    groups = [(f, v) for f in ("question_type", "form", "length", "context", "parts", "difficulty") if f in labels
              for v in sorted(set(labels[f])) if (labels[f] == v).sum() >= 10]
    if not groups:
        return
    hit = (base[PRIMARY].to_numpy() <= 1024).astype(float)
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(groups) + 1.6))
    handles = {}
    for k, m in enumerate(r.models):
        gain = (r.rows(c, f"rerank:{m}@{depth}")[PRIMARY].to_numpy() <= 1024).astype(float) - hit
        for g, (f, v) in enumerate(groups):
            mask = labels[f] == v
            lo, hi = interval(gain[mask], base.cluster.to_numpy()[mask])
            y = g + (k - (len(r.models) - 1) / 2) * 0.22
            if not np.isnan(lo):
                ax.plot([lo, hi], [y, y], color=SERIES[m])
            (handles[m],) = ax.plot([gain[mask].mean()], [y], "o", color=SERIES[m], markersize=6, markeredgecolor=SURFACE)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(range(len(groups)), [f"{f}: {v} (n={int((labels[f] == v).sum())})" for f, v in groups])
    ax.invert_yaxis()
    ax.set_xlabel(f"gain in {PRIMARY}-gold coverage at 1,024 reader tokens ({c} @{depth}), 95% interval")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, out / "query_types.png", f"Reranker gain by query type — {head}", [handles[m] for m in r.models],
          [LABEL[m] for m in r.models])


def first_stage_sizes(r: Results, out, head: str) -> None:
    sizes = sorted({int(c[3:]) for c in r.chunkings if c[:3] in ("fix", "rec", "seg")})
    if len(sizes) < 3:
        return
    budgets = (512, 1024, 2048, 4096)
    fig, axes = plt.subplots(1, len(budgets), figsize=(3.1 * len(budgets), 3.2), squeeze=False, sharey=True)
    handles = {}
    for j, b in enumerate(budgets):
        ax = axes[0][j]
        for fam, color, marker, ls in (("fix", SERIES["first_stage"], "o", "-"), ("rec", SERIES["q06"], "s", "--"),
                                       ("seg", SERIES["q4"], "D", "-")):
            xs = [n for n in sizes if f"{fam}{n}" in r.chunkings]
            ys = [(r.rows(f"{fam}{n}", "first_stage")[PRIMARY] <= b).mean() for n in xs]
            (handles[fam],) = ax.plot(xs, ys, color=color, marker=marker, linestyle=ls, markersize=5, markeredgecolor=SURFACE)
        _log_x(ax, 2, sizes)
        ax.set_title(f"reader budget {b}", loc="left")
        ax.set_xlabel("chunk size (mean tokens, nominal)")
        if j == 0:
            ax.set_ylabel(f"{PRIMARY}-gold coverage, no reranker")
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    _save(fig, out / "first_stage_sizes.png", f"First-stage coverage vs chunk size (no reranker) — {head}",
          [handles[k] for k in handles], [{"fix": "fixed windows", "rec": "recursive, size-matched",
                                           "seg": "EventMemory segments (characters)"}[k] for k in handles])


EXPANSION_APPROACHES = (("plain chunks of the read size", "#2a78d6"), ("seeds, expanded (no reranker)", "#8f8e89"),
                        ("rerank seeds, then expand", "#1baf7a"), ("expand, then rerank windows", "#eda100"))


def expansion_rows(r: Results) -> list[tuple[str, int, list[tuple[str, str, str]]]]:
    """Per (model, read size): the four approaches as (label, chunking, ranker) when present.
    Seeds are EventMemory's default 500-character segments when expanded, else the smallest
    chunking with expansions; the plain comparison is the chunking of the window's nominal
    size (4x for expand 3, 16x for expand 15)."""
    expanded = sorted({c for c in r.df.loc[r.df.expand > 0, "chunking"]}, key=lambda c: int(c[3:]))
    if not expanded:
        return []
    seed = "seg500" if "seg500" in expanded else expanded[0]
    fam, base = seed[:3], int(seed[3:])
    out = []
    for m in r.models:
        for e, plain in ((3, f"{fam}{base * 4}"), (15, f"{fam}{base * 16}")):
            if not (r.df.expand == e).any():
                continue
            depth = r.common_depth(seed, [m])
            items = [(EXPANSION_APPROACHES[0][0], plain, f"rerank:{m}@{r.common_depth(plain, [m])}") if plain in r.chunkings
                     else (EXPANSION_APPROACHES[0][0], seed, "missing"),
                     (EXPANSION_APPROACHES[1][0], seed, f"first_stage+x{e}"),
                     (EXPANSION_APPROACHES[2][0], seed, f"rerank:{m}@{depth}+x{e}"),
                     (EXPANSION_APPROACHES[3][0], seed, f"xrerank:{m}@{depth}+x{e}")]
            items = [(a, c, n) for a, c, n in items if not r.rows(c, n).empty]
            if len(items) > 1:
                out.append((m, base * (1 + e), items))
    return out


def expansion(r: Results, out, head: str) -> None:
    rows = expansion_rows(r)
    if not rows:
        return
    budgets = frontier.BUDGET_GRID[frontier.BUDGET_GRID >= 128]
    fig, axes = plt.subplots(1, len(rows), figsize=(3.4 * len(rows), 3.4), squeeze=False, sharey=True)
    handles = {}
    colors = dict(EXPANSION_APPROACHES)
    for j, (m, unit, items) in enumerate(rows):
        ax = axes[0][j]
        for label, c, name in items:
            (handles[label],) = ax.plot(budgets, coverage(r.rows(c, name)[PRIMARY].to_numpy(), budgets), color=colors[label])
        _log_x(ax, 2, [128, 512, 2048, 8192])
        ax.set_title(f"{LABEL[m]} · window ≈ {unit}", loc="left")
        ax.set_xlabel("reader tokens")
        if j == 0:
            ax.set_ylabel(f"{PRIMARY}-gold coverage")
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    _save(fig, out / "expansion.png", f"Embedded vs reranked vs read unit — {head}",
          [handles[k] for k, _ in EXPANSION_APPROACHES if k in handles], [k for k, _ in EXPANSION_APPROACHES if k in handles])


def coverage_counts(r: Results, out, head: str) -> None:
    """Every-gold coverage at equal counts instead of tokens, per chunking: against seeds kept
    after reranking (k), against segments read after expansion (m), and at k = 10 kept seeds
    as the vector search limit L grows."""
    if f"{PRIMARY}_ctx" not in r.df or not r.models:
        return
    m = r.models[-1]
    fig, axes = plt.subplots(len(r.chunkings), 3, figsize=(11.5, 3.1 * len(r.chunkings) + 0.9), squeeze=False)
    handles = {}
    for i, c in enumerate(r.chunkings):
        depth, limits = r.common_depth(c, [m]), [d for d in r.depths_at(c, m)]
        series = [("first_stage", "first_stage", SERIES["first_stage"], "-"),
                  (m, f"rerank:{m}@{depth}", SERIES[m], "-"),
                  ("first_stage+x3", "first_stage+x3", SERIES["first_stage"], "--"),
                  (f"{m}+x3", f"xrerank:{m}@{depth}+x3", SERIES[m], "--")]
        for j, (column, xs, xlabel) in enumerate(((f"{PRIMARY}_ctx", np.arange(1, 51), "seeds kept after reranking (k)"),
                                                   (f"{PRIMARY}_k", np.arange(1, 201), "segments read (m, max_num_segments)"))):
            ax = axes[i][j]
            for key, name, color, ls in series:
                rows = r.rows(c, name)
                if rows.empty:
                    continue
                values = rows[column].to_numpy()
                (handles[key],) = ax.plot(xs, [(values <= x).mean() for x in xs], color=color, linestyle=ls)
            ax.set_xlabel(xlabel)
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{c}: equal {'seeds kept' if j == 0 else 'segments'}", loc="left")
            if j == 0:
                ax.set_ylabel(f"{PRIMARY}-gold coverage")
        ax = axes[i][2]
        values = [int(d[1:]) for d in limits]
        for key, name, color, ls in (("first_stage", "first_stage@{d}", SERIES["first_stage"], "-"),
                                     (m, f"rerank:{m}@{{d}}", SERIES[m], "-"),
                                     ("first_stage+x3", "first_stage@{d}+x3", SERIES["first_stage"], "--"),
                                     (f"{m}+x3", f"xrerank:{m}@{{d}}+x3", SERIES[m], "--")):
            ys = [(r.rows(c, name.format(d=d))[f"{PRIMARY}_ctx"] <= 10).mean() if not r.rows(c, name.format(d=d)).empty
                  else np.nan for d in limits]
            if values and not np.all(np.isnan(ys)):
                (handles[key],) = ax.plot(values, ys, color=color, linestyle=ls, marker="o", markersize=4,
                                          markeredgecolor=SURFACE)
        ax.set_xlabel("vector search limit L (seeds retrieved and reranked)")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{c}: 10 seeds kept, as L grows", loc="left")
    keys = [k for k in ("first_stage", m, "first_stage+x3", f"{m}+x3") if k in handles]
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, out / "coverage_counts.png", f"Coverage at equal counts instead of tokens — {head}",
          [handles[k] for k in keys], [_label(k) for k in keys], ncol=4)


def write(r: Results) -> str:
    out = FIGURES / r.embedder / f"{r.dataset}__{r.scope}__{r.variant}__{r.sample}"
    out.mkdir(parents=True, exist_ok=True)
    head = f"{r.dataset} · {r.scope} · {r.variant} · {r.test.qid.nunique()} test queries"
    for draw in (coverage_by_tokens, recall_precision_f1, rerank_cost, break_even, cost_saving, query_types,
                 first_stage_sizes, expansion, coverage_counts):
        draw(r, out, head)
    return str(out)
