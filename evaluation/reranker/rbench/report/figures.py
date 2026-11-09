"""Charts for one (embedder, dataset, scope, variant, sample); the markdown is their table view.

  coverage_tokens_{first,every}.png  coverage vs reader tokens, panel per chunking: first stage,
                                     each model at the common depth,
                                     and dashed, the same with expand_context 3 (windows reranked)
  rpf_tokens.png, rpf_k.png          gold-character recall, precision and F1 vs tokens and vs k
  rerank_cost.png                    coverage at fixed reader budgets vs reranker tokens as
                                     overfetch grows; one line per chunking, so equal-x points
                                     compare many small chunks with few large ones
  break_even.png                     the LLM input price above which reranking is cheaper overall,
                                     vs coverage target (API-billed and prefix-cached reranker tokens)
  cost_saving.png                    % saved by allowing a reranker, over LLM price x coverage target
  query_types.png                    reranker gain at 1,000 tokens by query type, 95% intervals
  first_stage_sizes.png              first-stage coverage vs chunk size at fixed reader budgets, per
                                     chunker: where an optimal embedding chunk size shows
  expansion.png                      plain chunks vs small seeds expanded to the same read size, with
                                     no reranker, reranking before expansion, and after it
  coverage_counts.png                coverage at equal counts: seeds kept after reranking, segments
                                     read, and 10 seeds kept as the vector search limit grows

Colors follow the dataviz reference palette and were checked with its validator: models
take categorical slots in fixed order (first stage, 0.6B, 4B, 8B, then the local 0.6B); the
perfect reranker is neutral gray; chunk sizes use a one-hue ordinal ramp; savings use a
diverging blue (saves) / red (costs more) scale around a neutral gray midpoint.

Reading the charts: lines are thin with a light outline so crossings stay apart; each ranking
also has a line-only marker shape (MARKERS), and dashes mean expanded windows. Coverage curves
are exact step functions (one step per question); recall, precision and F1 are measured at
every minor gridline (every 250 reader tokens, every k: metrics.CURVE_BUDGETS, CURVE_KS).
Markers only identify the series (every 1,000 tokens, every 5 counts). Token and count axes are
linear. Under each title, a line says what is held equal and what is not; the legend under the
panels gives each line's vector search limit.

python -m rbench show <dataset> opens the same charts in zoomable matplotlib windows (show()).
"""

from __future__ import annotations

import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator, NullFormatter  # noqa: E402

from rbench.metrics import CURVE_BUDGETS, CURVE_KS  # noqa: E402
from rbench.paths import FIGURES  # noqa: E402
from rbench.report import frontier  # noqa: E402
from rbench.report.data import Results, coverage, mean_unit_tokens, price, query_labels  # noqa: E402
from rbench.report.stats import interval  # noqa: E402
from rbench.rerankers import RERANKERS  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"first_stage": "#2a78d6", "q06": "#eb6834", "q4": "#1baf7a", "q8": "#eda100", "q06-local": "#e87ba4"}
ORACLE = "#8f8e89"
# A marker shape per ranking, so overlapping lines stay identifiable without color. Line-only
# shapes (caltrops "1"-"4", x, +) cover nothing beneath them, so overlapping points stay
# readable, and their centre is the data point. Most curves here are flat or stepped, so the
# caltrops with a vertical prong (up "2", down "1") go to the models that matter most; left
# "3" (a prong along the line) to the local 0.6B; + (its bar along flat lines) to the gray reference.
MARKERS = {"first_stage": "x", "q8": "2", "q4": "1", "q06": "3", "q06-local": "3", "oracle": "+"}
LABEL = {"first_stage": "first stage only", "q06-local": "Qwen3-Reranker-0.6B (local)", "q06": "Qwen3-Reranker-0.6B",
         "q4": "Qwen3-Reranker-4B", "q8": "Qwen3-Reranker-8B"}
RAMP = {3: ("#86b6ef", "#2a78d6", "#104281"), 5: ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")}
DIVERGING = LinearSegmentedColormap.from_list("red_gray_blue", ["#a12a2a", "#e34948", "#f2b8b5", "#f0efec", "#9ec5f4",
                                                                  "#3987e5", "#104281"])  # equal steps per arm
BREAK_EVEN_TOP = 100.0  # $/M: the break-even chart's ceiling ("never cheaper" bands reach it)
PRIMARY = "every"  # single-event charts: every gold turn read (all of a question's evidence is needed)
# What each coverage event means, in words (metrics.py has the definitions).
EVENT_TEXT = {"first": "some evidence read in full", "every": "all evidence read in full",
              "first_touch": "some evidence at least partly read", "every_touch": "all evidence at least partly read"}
EVENT_NOTE = "Evidence: the turns marked as holding the answer; in full = all their segments read, partly = any."
COVERAGE = f"coverage: {EVENT_TEXT[PRIMARY]}"
SHOWN_EXPANSION = 3  # coverage and recall charts add this expand_context, dashed, when simulated

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 1, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titlecolor": INK, "lines.linewidth": 1.3, "lines.solid_capstyle": "round",
    "lines.markeredgewidth": 0.6, "lines.markeredgecolor": SURFACE,
    "legend.frameon": False, "legend.labelcolor": INK2, "legend.fontsize": 9, "legend.handlelength": 4.2,
    "legend.handletextpad": 0.6, "legend.columnspacing": 2.4, "figure.constrained_layout.h_pad": 0.08,
})
DASH = (0, (5, 3))  # the dashed style: long enough dashes to read in the legend's line samples
PANEL = (4.4, 3.2)  # inches per panel; figures are at least MIN_WIDTH wide so the legend never dwarfs the plot
MIN_WIDTH = 7.5
TOKEN_TOP = 8000  # reader-token axes show 0..8k (500-4,000 is the realistic range); data runs to 16k
TOKEN_MARKS = list(range(1000, 16000 + 1, 1000))  # where step curves carry their series marker
LINE, THIN = 1.3, 1.0  # line widths (points): thin enough that overlapping lines stay apart


def _hollow(kwargs: dict) -> dict:
    """Thin line-only markers in the line's color (unless a fill is asked for)."""
    if kwargs.get("marker"):
        kwargs.setdefault("markerfacecolor", "none")
        kwargs.setdefault("markeredgecolor", kwargs.get("color"))
        kwargs.setdefault("markeredgewidth", 1.2)
        kwargs.setdefault("markersize", {"x": 6, "+": 7}.get(kwargs["marker"], 8))  # x and + look larger at one size
    return kwargs


def _plot(ax, x, y, **kwargs):
    """ax.plot with a surface-colored outline on the line, so crossing and overlapping lines
    stay distinct, and hollow markers drawn on top without it."""
    width = kwargs.get("linewidth", LINE)
    marks = {k: kwargs.pop(k) for k in ("marker", "markersize", "markerfacecolor", "markeredgecolor", "markeredgewidth",
                                        "markevery") if k in kwargs}
    lines = ax.plot(x, y, path_effects=[patheffects.withStroke(linewidth=width + 1.4, foreground=SURFACE)], **kwargs)
    if marks.get("marker"):
        ax.plot(x, y, linestyle="none", zorder=lines[0].get_zorder() + 0.1, **_hollow({**marks, "color": kwargs.get("color")}))
    return lines


def _steps(ax, costs: np.ndarray, lo: float, hi: float, marks=None, marker: str | None = None, **kwargs):
    """Coverage as the exact step function of per-question costs on [lo, hi]: it rises by
    1/questions at each question's cost, so it is measured at every budget, not interpolated.
    With a marker, the series' shape is drawn at `marks` (the labelled ticks) to identify it."""
    finite = np.sort(costs[np.isfinite(costs)])
    xs = np.concatenate([[lo], finite[(finite > lo) & (finite <= hi)], [hi]])
    lines = _plot(ax, xs, coverage(costs, xs), drawstyle="steps-post", **kwargs)
    if marker and marks is not None:
        marks = np.asarray([m for m in marks if lo <= m <= hi], dtype=float)
        ax.plot(marks, coverage(costs, marks), linestyle="none", zorder=3, **_hollow({"marker": marker, "color": kwargs.get("color")}))
    return lines


def _marker(key: str) -> str:
    return MARKERS.get(key.split("+x")[0], "o")


_SHOWING = False  # set by show(): figures stay open in windows instead of being written


def _figure(rows: int, cols: int, panel: tuple[float, float] = PANEL, **kwargs):
    """A constrained-layout figure of rows x cols panels; the title goes above it, the legend below."""
    width = max(MIN_WIDTH, cols * panel[0])
    return plt.subplots(rows, cols, figsize=(width, rows * panel[1] + 0.5), squeeze=False, layout="constrained", **kwargs)


def _human(v: float) -> str:
    """Plain numbers with thousands separators: "1,000" is never ambiguous, "1k" can be (1,000 or 1,024?)."""
    return f"{v:,.0f}" if v >= 1 else f"{v:g}"


def _log_x(ax, base: int = 10, ticks=None) -> None:
    ax.set_xscale("log", base=base)
    if ticks is not None:
        ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _human(v)))
    ax.xaxis.set_minor_formatter(NullFormatter())


def _token_x(ax, top: int = TOKEN_TOP) -> None:
    """Reader tokens on a linear axis: labelled every 500 (every 1,000 in narrow panels), minor gridlines every 250 (where
    recall and precision are measured: metrics.CURVE_BUDGETS). Data beyond `top` is drawn and
    can be panned to in `show`."""
    ax.set_xlim(0, top)
    wide = ax.get_position().width * ax.figure.get_figwidth() >= 5.5  # inches: room for a label every 500
    ax.xaxis.set_major_locator(MultipleLocator(500 if wide else 1000))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _human(v) if v else "0"))
    ax.xaxis.set_minor_locator(MultipleLocator(250))
    ax.grid(which="minor", color=GRID, linewidth=0.5, alpha=0.7)


def _k_x(ax, top: int = 50) -> None:
    """A count on a linear axis: labelled every 5 (every 20 past 50), minor gridlines at every
    count (every 5 past 50)."""
    ax.set_xlim(0, top + 0.5)
    ax.xaxis.set_major_locator(MultipleLocator(5 if top <= 50 else 20))
    ax.xaxis.set_minor_locator(MultipleLocator(1 if top <= 50 else 5))
    ax.grid(which="minor", color=GRID, linewidth=0.5, alpha=0.7)


def _share_y(ax, top: float = 1.0) -> None:
    """A 0..top share axis with gridlines every 0.1 (every 0.05 or 0.02 for small ranges)."""
    step = 0.1 if top > 0.4 else 0.05 if top > 0.15 else 0.02
    ax.set_ylim(0, top * 1.02)
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_minor_locator(MultipleLocator(step / 2))
    ax.grid(which="minor", axis="y", color=GRID, linewidth=0.5, alpha=0.7)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}".rstrip("0").rstrip(".") if v else "0"))


def _log_y(ax) -> None:
    """Log y axis labelled at 1, 2 and 5 of each decade, so a range within one decade still
    has numbers."""
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def _blank() -> Line2D:
    return Line2D([], [], linestyle="none")


def _key(color: str, ls="-", marker=None) -> Line2D:
    """A legend sample: thicker than the plotted line so color and dash pattern read at a glance."""
    return Line2D([], [], linestyle=ls, linewidth=1.8, **_hollow({"marker": marker, "color": color, "markersize": 9}))


def _save(fig, path, title: str, handles=None, labels=None, ncol: int | None = None,
          groups: list[tuple[str, list, list[str]]] | None = None, note: str = "") -> None:
    """Title above the panels and legend below them (the figure must use constrained layout). `groups`
    are (heading, handles, labels) shown as side-by-side columns, e.g. what the color means
    and what the line style means; or pass plain handles and labels."""
    chars = int(fig.get_figwidth() * 12.5)  # characters per line at this font size: text never widens the figure
    text = textwrap.fill(title, chars) + (f"\n{textwrap.fill(note, chars)}" if note else "")
    fig.suptitle(text, x=0.01, ha="left", fontsize=10.5, color=INK, linespacing=1.45)
    legend_rows = (max(len(h) for _, h, _ in groups) + 1 if groups else
                   -(-len(handles) // (ncol or min(len(handles), 3))) if handles else 0)
    # Title, note and legend add height instead of taking it from the panels.
    fig.set_figheight(fig.get_figheight() + 0.23 * (text.count("\n") + 1) + 0.25 * legend_rows + 0.2)
    if groups:
        height = max(len(h) for _, h, _ in groups) + 1
        entries = []
        for heading, hs, ls in groups:
            column = [(_blank(), heading)] + list(zip(hs, ls, strict=True))
            entries += column + [(_blank(), "")] * (height - len(column))
        legend = fig.legend([h for h, _ in entries], [lab for _, lab in entries], loc="outside lower left", ncol=len(groups))
        for i, text in enumerate(legend.get_texts()):
            if i % height == 0:
                text.set_fontweight("bold")
                text.set_color(INK)
    elif handles:
        fig.legend(handles, labels, loc="outside lower left", ncol=ncol or min(len(handles), 3))
    if _SHOWING:
        fig.canvas.manager.set_window_title(path.stem)
        return
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _rankers(r: Results, c: str, limit: str, bound: bool = False) -> list[tuple[str, str, str, str]]:
    """(legend key, ranker, color, line style) at one vector search limit: the first stage in
    vector order and each model's reranking, over the same top seeds; solid reads the seeds,
    dashed their EventMemory windows (expand, then rerank). bound adds the perfect reranker,
    which is a bound only for "all evidence read in full" with seeds alone (see rankers.py):
    pass it only there."""
    e = SHOWN_EXPANSION
    out = [("first_stage", f"first_stage@{limit}", SERIES["first_stage"], "-")]
    out += [(m, f"rerank:{m}@{limit}", SERIES[m], "-") for m in r.models_at(c)]
    if bound:
        out.append(("oracle", f"oracle@{limit}", ORACLE, "-"))
    if e in r.expansions:
        out.append((f"first_stage+x{e}", f"first_stage@{limit}+x{e}", SERIES["first_stage"], DASH))
        out += [(f"{m}+x{e}", f"xrerank:{m}@{limit}+x{e}", SERIES[m], DASH) for m in r.models_at(c)]
    return [x for x in out if not r.rows(c, x[1]).empty]


def _priced(m: str) -> str:
    """A model's label for charts that put a price on its tokens: a local model has none of its
    own, so it is priced as DeepInfra's model of the same size, and says so."""
    if RERANKERS[m].backend == "local":
        return f"{LABEL[m]}, priced as DeepInfra's 0.6B (${price(m):g}/M)"
    return f"{LABEL[m]} (DeepInfra ${price(m):g}/M)"


def _ranker_groups(r: Results, keys: set[str]) -> list[tuple[str, list, list[str]]]:
    """Two legend columns for charts drawn with _rankers: color and marker = how the seeds are
    ordered, line style = what is read (the seeds, or their windows)."""
    e = SHOWN_EXPANSION
    bases = {k.split("+x")[0] for k in keys}
    color_keys = [k for k in ("first_stage", *r.models, "oracle") if k in bases]
    names = {"first_stage": "first stage (vector search order)",
             "oracle": "perfect reranker, seeds only: a bound for the solid lines (windows can reach evidence it cannot)"}
    labels = [names.get(k, LABEL.get(k, k)) for k in color_keys]
    groups = [("order (color, marker)", [_key(SERIES.get(k, ORACLE), marker=MARKERS.get(k, "o")) for k in color_keys], labels)]
    if any("+x" in k for k in keys):
        groups.append(("reading (line)", [_key(INK2), _key(INK2, DASH)],
                       ["seeds (expand_context 0)", f"windows (expand_context {e}); a reranker scores the windows"]))
    return groups


SAME_LIMIT = ("Each panel is one vector search limit L: every line reads only the top L seeds of the same vector search, "
              "in vector order (first stage) or reranked. Dashed: seeds read with their windows.")
BOUND = (" Gray: a perfect reranker over the same seeds, which reads only the evidence seeds: no reranker can read all "
         "the evidence in full with fewer tokens (drawn only here; with windows, or for other measures, it is no bound).")
TOKENS_EQUAL = "Held equal: reader tokens, and the vector search limit (panel). Not equal: reranker tokens. " + SAME_LIMIT


def _limit_grid(r: Results) -> tuple[list[str], list[str]]:
    """(chunkings, vector search limits) for panel grids: a row per chunking, a column per limit."""
    return r.chunkings, sorted({x for c in r.chunkings for x in r.limits(c)}, key=lambda d: int(d[1:]))


def coverage_by_tokens(r: Results, out, head: str) -> None:
    chunkings, limits = _limit_grid(r)
    for event in ("first", "every", "first_touch", "every_touch"):
        fig, axes = _figure(len(chunkings), len(limits), panel=(4.0, 3.2), sharex=True, sharey=True)
        keys = set()
        for i, c in enumerate(chunkings):
            for j, limit in enumerate(limits):
                ax = axes[i][j]
                if limit not in r.limits(c):
                    ax.set_visible(False)
                    continue
                for key, name, color, ls in _rankers(r, c, limit, bound=event == "every"):
                    _steps(ax, r.rows(c, name)[event].to_numpy(), 0, 16000, TOKEN_MARKS, _marker(key), color=color,
                           linestyle=ls, linewidth=THIN if key.startswith("oracle") else LINE)
                    keys.add(key)
                _token_x(ax)
                _share_y(ax)
                ax.set_title(f"{c} ({mean_unit_tokens(r, c)}) · vector search limit {limit[1:]}", loc="left")
                if i == len(chunkings) - 1:
                    ax.set_xlabel("reader tokens (rendered context)\nexact at every budget; markers every 1,000")
                if j == 0:
                    ax.set_ylabel(f"share of questions: {EVENT_TEXT[event]}")
        _save(fig, out / f"coverage_tokens_{event}.png", f"Coverage ({EVENT_TEXT[event]}) vs reader tokens — {head}",
              groups=_ranker_groups(r, keys), note=TOKENS_EQUAL + (BOUND if event == "every" else "") + " " + EVENT_NOTE)


def recall_precision_f1(r: Results, out, head: str) -> None:
    for c in r.chunkings:
        limits = r.limits(c)
        for axis, fmt, grid, xlabel in (("tokens", "t{}", CURVE_BUDGETS, "reader tokens (rendered context)"),
                                        ("k", "k{}", CURVE_KS, "segments read (k, window neighbours included)")):
            xs = [v for v in grid if f"recall_{fmt.format(v)}" in r.df]  # results simulated before the finer grid
            fig, axes = _figure(3, len(limits), panel=(4.0, 2.6), sharex=True)
            keys = set()
            for j, limit in enumerate(limits):
                for key, name, color, ls in _rankers(r, c, limit):
                    rows = r.rows(c, name)
                    rec = np.stack([rows[f"recall_{fmt.format(v)}"].to_numpy() for v in xs], 1)
                    pre = np.stack([rows[f"precision_{fmt.format(v)}"].to_numpy() for v in xs], 1)
                    f1 = np.where(rec + pre > 0, 2 * rec * pre / np.maximum(rec + pre, 1e-12), 0.0)
                    every = [n for n, v in enumerate(xs) if v % (1000 if axis == "tokens" else 5) == 0]
                    for i, values in enumerate((rec, pre, f1)):
                        _plot(axes[i][j], xs, values.mean(0), color=color, linestyle=ls, marker=_marker(key),
                              markevery=every, linewidth=THIN if key.startswith("oracle") else LINE)
                    keys.add(key)
                axes[0][j].set_title(f"{c} · vector search limit {limit[1:]}", loc="left")
                for i, label in enumerate(("recall (gold characters)", "precision", "F1")):
                    _token_x(axes[i][j]) if axis == "tokens" else _k_x(axes[i][j], max(CURVE_KS))
                    if j == 0:
                        axes[i][j].set_ylabel(label)
                axes[2][j].set_xlabel(f"{xlabel}\nmeasured at every minor gridline")
            for row in axes:
                tops = [line.get_ydata().max() for ax in row for line in ax.get_lines()]
                for ax in row:
                    _share_y(ax, min(1.0, max(tops, default=1.0)))
            note = (TOKENS_EQUAL if axis == "tokens" else
                    "Held equal: segments read (x; window neighbours count), and the vector search limit (panel). "
                    "Not equal: reader tokens, reranker tokens. " + SAME_LIMIT)
            name = f"rpf_{axis}.png" if len(r.chunkings) == 1 else f"rpf_{axis}_{c}.png"
            _save(fig, out / name, f"Gold-character recall, precision and F1 vs {xlabel} — {head}",
                  groups=_ranker_groups(r, keys), note=note)


def _size_styles(chunkings: list[str]) -> dict[str, dict]:
    sizes = sorted({int(c[3:]) for c in chunkings})
    ramp = RAMP[3] if len(sizes) <= 3 else RAMP[5]
    color = dict(zip(sizes[-len(ramp):], ramp))  # beyond 5 sizes, the smallest go unplotted
    return {c: {"color": color[int(c[3:])], "ls": DASH if c.startswith("rec") else "-",
                "marker": "1" if c.startswith("rec") else "2"} for c in chunkings if int(c[3:]) in color}


def rerank_cost(r: Results, out, head: str) -> None:
    if not r.models:
        return
    styles, budgets = _size_styles(r.chunkings), (1000, 4000)
    fig, axes = _figure(len(r.models), 2, sharey=True)
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
                ax.axhline(base, color=s["color"], linestyle=":", linewidth=THIN)  # the same segments, no reranker
                _plot(ax, xs, ys, color=s["color"], linestyle=s["ls"], marker=s["marker"])
            _log_x(ax)
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            _share_y(ax)
            ax.set_title(f"{LABEL[m]} · reader budget {b:,}", loc="left")
            ax.set_xlabel("reranker tokens per query (API billing)\nmarkers: the vector search limits measured")
            if j == 0:
                ax.set_ylabel(COVERAGE)
    keys = list(styles)
    _save(fig, out / "rerank_cost.png", f"Coverage vs reranker tokens as the vector search limit grows — {head}",
          [_key(styles[k]["color"], styles[k]["ls"], styles[k]["marker"]) for k in keys], keys, ncol=min(len(keys), 5),
          note="Held equal: reader tokens (per panel). x: reranker tokens per query, one marker per vector search limit "
               "(5, 10, 20, ...). Dotted line: the same segments with no reranker (first stage, no limit).")


def break_even(r: Results, out, head: str) -> None:
    """The LLM input price above which reranking is cheaper overall, per coverage target: solid
    with reranker tokens as an API bills them, dashed with a prefix cache (shared prompt and
    question computed once per question), both at the reranker's price per token."""
    if not r.models:
        return
    fig, axes = _figure(1, 1, panel=(7.5, 3.8))
    ax = axes[0][0]
    finite = lambda v: np.where(np.isfinite(v) & (v > 0) & (v < 1e12), v, np.nan)  # noqa: E731
    x = frontier.TARGETS
    lines = {}
    for m in r.models:
        api, lo, hi, _, _ = frontier.break_even(r, m, "rr_tokens", event=PRIMARY)
        cached = frontier.break_even(r, m, "rr_prefix_tokens", with_interval=False, event=PRIMARY)[0]
        lines[m] = finite(api) * price(m)
        _plot(ax, x, lines[m], drawstyle="steps-mid", color=SERIES[m])
        _plot(ax, x, finite(cached) * price(m), drawstyle="steps-mid", color=SERIES[m], linestyle=DASH)
        band_lo, band_hi = finite(lo) * price(m), np.where(hi >= 1e12, BREAK_EVEN_TOP, hi * price(m))
        ax.fill_between(x, band_lo, band_hi, where=np.isfinite(band_lo) & np.isfinite(lines[m]), step="mid", color=SERIES[m],
                        alpha=0.12, linewidth=0)  # only where there is a line: elsewhere the estimate is "never"
    m = r.models[-1]
    ax.text(0.99, 0.97, "above a line: reranking is cheaper overall\nbelow it: the first stage alone is cheaper",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK2)
    ax.set_xlim(x[0] - 0.02, 1.0)
    ax.set_yscale("log")
    ax.set_ylim(0.01, BREAK_EVEN_TOP)
    _log_y(ax)
    ax.set_ylabel("break-even LLM input price ($ per million tokens)")
    ax.set_xlabel(f"coverage target ({EVENT_TEXT[PRIMARY]})")
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    split = "the same questions"
    handles = [_key(SERIES[k]) for k in r.models] + [_key(INK2), _key(INK2, DASH)]
    labels = ([_priced(k) for k in r.models] + ["reranker tokens as an API bills them (band: 95% interval)",
               "with a prefix cache (prompt and question computed once)"])
    _save(fig, out / "break_even.png", f"Above which LLM price does reranking pay for itself? — {head}", handles, labels,
          ncol=2, note=f"Held equal: coverage (x: the share of questions with {EVENT_TEXT[PRIMARY]}). y: the LLM input "
                       "price at which reaching that coverage costs the same (reader + reranker tokens) with and without "
                       f"the reranker, each at its cheapest configuration, chosen on {split}. Above the line reranking is "
                       "cheaper; no line: reranking is never cheaper at that target. Band: 95% interval of the line (reaching "
                       "the top: it includes never).")


def cost_saving(r: Results, out, head: str) -> None:
    """Over a continuous range of LLM prices and coverage targets: the cost of reaching the target
    with the reranker (its cheapest configuration) relative to without it. Blue saves, red costs
    more; gray where only the reranker reaches the target."""
    if not r.models:
        return
    llm = np.logspace(-2, 2, 81)  # $ per million reader tokens
    fig, axes = _figure(1, len(r.models), panel=(4.6, 3.8), sharey=True)
    for k, m in enumerate(r.models):
        curves = frontier.cost_curves(r, m, list(llm), price(m), event=PRIMARY)
        no = np.stack([curves[p][0] for p in llm], 1)  # [target, price]
        yes = np.stack([curves[p][1] for p in llm], 1)
        ax = axes[0][k]
        with np.errstate(invalid="ignore", divide="ignore"):
            saved = 100 * (1 - yes / no)
        im = ax.pcolormesh(llm, frontier.TARGETS, saved, cmap=DIVERGING, norm=TwoSlopeNorm(0, -50, 50), shading="nearest")
        only = np.where(np.isnan(no) & np.isfinite(yes), 1.0, np.nan)
        ax.pcolormesh(llm, frontier.TARGETS, only, cmap=LinearSegmentedColormap.from_list("only", [ORACLE, ORACLE]),
                      shading="nearest", alpha=0.6)
        ax.set_facecolor(SURFACE)  # blank where neither reaches the target
        _log_x(ax)
        ax.grid(False)
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.set_title(_priced(m), loc="left")
        ax.set_xlabel("LLM input price ($ per million tokens)")
        if k == 0:
            ax.set_ylabel(f"coverage target ({EVENT_TEXT[PRIMARY]})")
    fig.colorbar(im, ax=axes[0].tolist(), label="% of total cost saved by reranking (red: reranking costs more)")
    split = "the same questions"
    _save(fig, out / "cost_saving.png", f"Cost saved by reranking, at every LLM price and coverage target — {head}",
          [Patch(color=ORACLE, alpha=0.6)], ["only the reranker reaches this target"], ncol=1,
          note="Held equal: the coverage target (y) and the LLM input price (x). Color: total cost (reader + reranker "
               "tokens) of reaching the target with the reranker, relative to without it, each at its cheapest "
               f"configuration, chosen on {split}. Blank: neither reaches the target. Matches the "
               "break-even line.")


def query_types(r: Results, out, head: str) -> None:
    if not r.models or r.df.qid.nunique() < 20:
        return
    c = next((c for c in ("seg500", "rec64", "fix256", "rec256", "fix1024") if c in r.chunkings), r.chunkings[0])
    depth = r.common_depth(c)
    if not depth:
        return
    base, labels = query_labels(r, c)
    groups = [(f, v) for f in ("question_type", "form", "length", "context", "parts") if f in labels
              for v in sorted(set(labels[f])) if (labels[f] == v).sum() >= 10]
    if not groups:
        return
    hit = (base[PRIMARY].to_numpy() <= 1000).astype(float)
    fig, axes = _figure(1, 1, panel=(7.5, 0.34 * len(groups) + 1.2))
    ax = axes[0][0]
    for k, m in enumerate(r.models):
        gain = (r.rows(c, f"rerank:{m}@{depth}")[PRIMARY].to_numpy() <= 1000).astype(float) - hit
        for g, (f, v) in enumerate(groups):
            mask = labels[f] == v
            lo, hi = interval(gain[mask], base.cluster.to_numpy()[mask])
            y = g + (k - (len(r.models) - 1) / 2) * 0.22
            if not np.isnan(lo):
                ax.plot([lo, hi], [y, y], color=SERIES[m], linewidth=LINE)
            ax.plot([gain[mask].mean()], [y], "o", color=SERIES[m], markersize=5.5)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(range(len(groups)), [f"{f}: {v} (n={int((labels[f] == v).sum())})" for f, v in groups])
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_xlabel(f"gain in {COVERAGE} at 1,000 reader tokens ({c} @{depth}); dot: mean, line: 95% interval")
    _save(fig, out / "query_types.png", f"Reranker gain by query type — {head}", [_key(SERIES[m], marker="o") for m in r.models],
          [LABEL[m] for m in r.models],
          note=f"Held equal: 1,000 reader tokens. The reranker reads only its top {depth[1:]}; the first stage has no limit.")


def first_stage_sizes(r: Results, out, head: str) -> None:
    sizes = sorted({int(c[3:]) for c in r.chunkings if c[:3] in ("fix", "rec", "seg")})
    if len(sizes) < 3:
        return
    budgets = (500, 1000, 2000, 4000)
    fig, axes = _figure(1, len(budgets), panel=(3.4, 3.2), sharey=True)
    styles = {"fix": (SERIES["first_stage"], "x", "-"), "rec": (SERIES["q06"], "1", DASH), "seg": (SERIES["q4"], "2", "-")}
    present = set()
    for j, b in enumerate(budgets):
        ax = axes[0][j]
        for fam, (color, marker, ls) in styles.items():
            xs = [n for n in sizes if f"{fam}{n}" in r.chunkings]
            if not xs:
                continue
            ys = [(r.rows(f"{fam}{n}", "first_stage")[PRIMARY] <= b).mean() for n in xs]
            _plot(ax, xs, ys, color=color, marker=marker, linestyle=ls)
            present.add(fam)
        _log_x(ax, 2, sizes)
        _share_y(ax)
        ax.set_title(f"reader budget {b:,}", loc="left")
        ax.set_xlabel("segment / chunk size; markers: sizes measured")
        if j == 0:
            ax.set_ylabel(f"{COVERAGE}, no reranker")
    names = {"fix": "fixed windows", "rec": "recursive, size-matched", "seg": "EventMemory segments (characters)"}
    keys = [k for k in styles if k in present]
    _save(fig, out / "first_stage_sizes.png", f"First-stage coverage vs segment size (no reranker) — {head}",
          [_key(styles[k][0], styles[k][2], styles[k][1]) for k in keys], [names[k] for k in keys],
          note="Held equal: reader tokens (per panel). No reranker; the first stage reads as deep as the budget allows.")


EXPANSION_APPROACHES = (("plain chunks of the read size", "#2a78d6"), ("seeds, expanded (no reranker)", "#8f8e89"),
                        ("rerank seeds, then expand", "#1baf7a"), ("expand, then rerank windows", "#eda100"))
EXPANSION_MARKERS = dict(zip((a for a, _ in EXPANSION_APPROACHES), ("x", "+", "2", "1"), strict=True))


def expansion_rows(r: Results) -> list[tuple[str, int, list[tuple[str, str, str]]]]:
    """Per (model, read size): the four approaches as (label, chunking, ranker) when present.
    Seeds are EventMemory's default 500-character segments when expanded, else the smallest
    chunking with expansions; the plain comparison is the chunking of the window's nominal
    size (4x for expand 3)."""
    expanded = sorted({c for c in r.df.loc[r.df.expand > 0, "chunking"]}, key=lambda c: int(c[3:]))
    if not expanded:
        return []
    seed = "seg500" if "seg500" in expanded else expanded[0]
    fam, base = seed[:3], int(seed[3:])
    out = []
    for m in r.models:
        for e, plain in ((3, f"{fam}{base * 4}"),):
            if not (r.df.expand == e).any():
                continue
            depth = r.common_depth(seed, [m])
            items = [(EXPANSION_APPROACHES[0][0], plain, f"rerank:{m}@{r.common_depth(plain, [m])}") if plain in r.chunkings
                     else (EXPANSION_APPROACHES[0][0], seed, "missing"),
                     (EXPANSION_APPROACHES[1][0], seed, f"first_stage@{depth}+x{e}"),
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
    fig, axes = _figure(1, len(rows), sharey=True)
    colors = dict(EXPANSION_APPROACHES)
    present = set()
    for j, (m, unit, items) in enumerate(rows):
        ax = axes[0][j]
        for label, c, name in items:
            _steps(ax, r.rows(c, name)[PRIMARY].to_numpy(), 0, 16000, TOKEN_MARKS, EXPANSION_MARKERS[label],
                   color=colors[label])
            present.add(label)
        _token_x(ax)
        _share_y(ax)
        ax.set_title(f"{LABEL[m]} · window ≈ {unit}", loc="left")
        ax.set_xlabel("reader tokens: exact steps, one per question; markers every 1k")
        if j == 0:
            ax.set_ylabel(COVERAGE)
    keys = [k for k, _ in EXPANSION_APPROACHES if k in present]
    _save(fig, out / "expansion.png", f"Embedded vs reranked vs read unit — {head}",
          [_key(colors[k], marker=EXPANSION_MARKERS[k]) for k in keys], keys, ncol=2,
          note="Held equal: reader tokens (x). Not equal: reranker tokens and vector search limit.")


def coverage_counts(r: Results, out, head: str) -> None:
    """Coverage at equal counts instead of tokens, per chunking: against seeds kept (k) and
    segments read after expansion (m) at one vector search limit L (EventMemory's default, 20,
    where scored), and at k = 10 kept seeds as L grows."""
    if f"{PRIMARY}_ctx" not in r.df:
        return
    fig, axes = _figure(len(r.chunkings), 3, panel=(4.0, 3.2))
    keys = set()
    for i, c in enumerate(r.chunkings):
        limits = r.limits(c)
        if not limits:
            continue
        limit = "n20" if "n20" in limits else limits[-1]
        series = _rankers(r, c, limit, bound=PRIMARY == "every")
        for j, (column, top, xlabel) in enumerate(((f"{PRIMARY}_ctx", min(50, int(limit[1:])), "seeds kept (k)"),
                                                    (f"{PRIMARY}_k", 200, "segments read (m = max_num_segments)"))):
            ax = axes[i][j]
            for key, name, color, ls in series:
                _steps(ax, r.rows(c, name)[column].to_numpy(), 0, top, range(5 if top <= 50 else 20, top + 1, 5 if top <= 50 else 20),
                       _marker(key), color=color, linestyle=ls, linewidth=THIN if key.startswith("oracle") else LINE)
                keys.add(key)
            _k_x(ax, top)
            ax.set_xlabel(f"{xlabel}\nexact at every count; markers identify series")
            _share_y(ax)
            ax.set_title(f"{c} · vector search limit {limit[1:]}: equal {'seeds kept' if j == 0 else 'segments'}", loc="left")
            if j == 0:
                ax.set_ylabel(COVERAGE)
        ax = axes[i][2]
        values = [int(d[1:]) for d in limits]
        for key, _, color, ls in series:
            ys = [(r.rows(c, n)[f"{PRIMARY}_ctx"] <= 10).mean() for n in (_name_at(key, d) for d in limits)]
            _plot(ax, values, ys, color=color, linestyle=ls, marker=_marker(key),
                  linewidth=THIN if key.startswith("oracle") else LINE)
        _log_x(ax, 10, values)
        _share_y(ax)
        ax.set_xlabel("vector search limit L (seeds retrieved; a reranker scores all of them)\nmarkers: the limits measured")
        ax.set_title(f"{c}: 10 seeds kept, as L grows", loc="left")
    _save(fig, out / "coverage_counts.png", f"Coverage at equal counts instead of tokens — {head}",
          groups=_ranker_groups(r, keys),
          note="Held equal: seeds kept (left), segments read (middle), vector search limit L (each panel; right: "
               "as it grows). Not equal: reader tokens and reranker tokens. " + SAME_LIMIT + (BOUND if PRIMARY == "every" else ""))


def _name_at(key: str, limit: str) -> str:
    """The ranker name of a _rankers legend key at another vector search limit."""
    base, _, e = key.partition("+x")
    tail = f"+x{e}" if e else ""
    if base == "first_stage":
        return f"first_stage@{limit}{tail}"
    if base == "oracle":
        return f"oracle@{limit}{tail}"
    return f"{'xrerank' if e else 'rerank'}:{base}@{limit}{tail}"


CHARTS = {"coverage": "coverage_by_tokens", "rpf": "recall_precision_f1", "rerank_cost": "rerank_cost",
          "break_even": "break_even", "cost_saving": "cost_saving", "query_types": "query_types",
          "sizes": "first_stage_sizes", "expansion": "expansion", "counts": "coverage_counts"}


def _head(r: Results) -> str:
    return f"{r.dataset} · {r.scope} · {r.variant} · {r.df.qid.nunique()} questions"


def write(r: Results) -> str:
    out = FIGURES / r.embedder / f"{r.dataset}__{r.scope}__{r.variant}__{r.sample}"
    out.mkdir(parents=True, exist_ok=True)
    for name in CHARTS.values():
        globals()[name](r, out, _head(r))
    return str(out)


def show(r: Results, charts: list[str] | None = None) -> None:
    """Open the charts in interactive matplotlib windows (zoom, pan, save from the toolbar)
    instead of writing PNGs. charts: keys of CHARTS (default: all)."""
    global _SHOWING
    for backend in ("macosx", "QtAgg", "TkAgg"):
        try:
            plt.switch_backend(backend)
            break
        except (ImportError, RuntimeError):
            continue
    else:
        raise SystemExit("no interactive matplotlib backend is available here; open the PNGs instead")
    _SHOWING = True
    try:
        out = FIGURES / r.embedder / f"{r.dataset}__{r.scope}__{r.variant}__{r.sample}"  # window titles only
        for key in charts or list(CHARTS):
            globals()[CHARTS[key]](r, out, _head(r))
        plt.show()
    finally:
        _SHOWING = False
