"""Charts for one (embedder, dataset, scope, variant, sample); the markdown is their table view.
The values they are drawn at (reader budgets, vector search limits, readings, expansions) come
from rbench.config.EVAL. Files (a suffix _<chunking> when several chunkings were simulated):

  coverage_tokens_<reading>.png  coverage vs reader tokens: a row per expand_context, a column per
                                 vector search limit L; in each panel every line reads only the top
                                 L seeds: the first stage in vector order, each reranker scoring the
                                 seeds (solid) or their windows (dashed, where scored)
  rpf_<tokens|k>[_x<E>].png      gold-character recall, precision and F1 vs tokens and vs k
  coverage_counts.png            coverage at equal counts (seeds kept, segments read, L)
  rerank_cost.png                coverage vs reranker tokens as L grows, beside the first stage at
                                 the same L, at each configured budget
  gain_sizes.png                 reranker gain (same L) vs segment size, at each budget
  break_even.png                 the LLM input price above which reranking is cheaper overall
  cost_saving.png                % saved by reranking, over LLM price x coverage target
  query_types.png                reranker gain (same L) by question type, at each budget
  first_stage_sizes.png          first-stage coverage vs segment size (no reranker)
  expansion.png                  plain chunks vs small seeds expanded to the same read size
Single-reading charts are drawn for each reading in EVAL.readings, with a _<reading> suffix
except for "every".

Colors follow the dataviz reference palette and were checked with its validator: models take
their color, marker and label from their registry entry (rerankers.RERANKERS); the perfect
reranker is neutral gray; ordered values (segment sizes, budgets) use a one-hue ordinal ramp;
savings use a diverging blue (saves) / red (costs more) scale around a neutral gray midpoint.

Reading the charts: lines are thin with a light outline so crossings stay apart; each ranking
also has a line-only marker shape (MARKERS). Coverage curves are exact step functions (one step
per question); recall, precision and F1 are measured at every minor gridline (every 250 reader
tokens, every k: metrics.CURVE_BUDGETS, CURVE_KS). Markers only identify the series. Token and
count axes are linear. Under each title, a line says what is held equal and what is not.

python -m rbench show <dataset> opens the same charts in zoomable matplotlib windows (show()).
"""

from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator, NullFormatter  # noqa: E402

from rbench import config  # noqa: E402
from rbench.metrics import CURVE_BUDGETS, CURVE_KS  # noqa: E402
from rbench.paths import FIGURES  # noqa: E402
from rbench.report import frontier  # noqa: E402
from rbench.report.data import (  # noqa: E402
    Results,
    coverage,
    default_segment,
    price,
    query_labels,
    size_label,
    unit_tokens,
)
from rbench.report.stats import interval  # noqa: E402
from rbench.rerankers import RERANKERS  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# Models take their color, marker and label from their registry entry (rerankers.RERANKERS).
SERIES = {"first_stage": "#2a78d6", **{m: spec.color for m, spec in RERANKERS.items()}}
ORACLE = "#8f8e89"
# A marker shape per ranking, so overlapping lines stay identifiable without color. Line-only
# shapes (caltrops "1"-"4", x, +) cover nothing beneath them, so overlapping points stay
# readable, and their centre is the data point. Most curves here are flat or stepped, so the
# caltrops with a vertical prong (up "2", down "1") go to the models that matter most; left
# "3" (a prong along the line) to the local 0.6B; + (its bar along flat lines) to the gray reference.
MARKERS = {"first_stage": "x", "oracle": "+", **{m: spec.marker for m, spec in RERANKERS.items()}}
LABEL = {"first_stage": "first stage only", **{m: spec.label for m, spec in RERANKERS.items()}}
CHUNKERS = {"fix": ("#2a78d6", "x", "-"), "rec": ("#eb6834", "1", (0, (5, 3))), "seg": ("#1baf7a", "2", "-")}  # color, marker, line
RAMP = {3: ("#86b6ef", "#2a78d6", "#104281"), 5: ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")}
DIVERGING = LinearSegmentedColormap.from_list("red_gray_blue", ["#a12a2a", "#e34948", "#f2b8b5", "#f0efec", "#9ec5f4",
                                                                  "#3987e5", "#104281"])  # equal steps per arm
BREAK_EVEN_TOP = 100.0  # $/M: the break-even chart's ceiling ("never cheaper" bands reach it)
# What each coverage event means, in words (metrics.py has the definitions).
EVENT_TEXT = {"first": "some evidence read in full", "every": "all evidence read in full",
              "first_touch": "some evidence at least partly read", "every_touch": "all evidence at least partly read"}
EVENT_NOTE = "Evidence: the turns marked as holding the answer; in full = all their segments read, partly = any."

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
    return MARKERS.get(key.split("~")[0], "o")


_SHOWING = False  # set by show(): figures stay open in windows instead of being written


def _figure(rows: int, cols: int, panel: tuple[float, float] = PANEL, **kwargs):
    """A constrained-layout figure of rows x cols panels; the title goes above it, the legend below."""
    width = max(MIN_WIDTH, cols * panel[0])
    return plt.subplots(rows, cols, figsize=(width, rows * panel[1] + 0.5), squeeze=False, layout="constrained", **kwargs)


def _human(v: float) -> str:
    """Plain numbers with thousands separators: "1,000" is never ambiguous, "1k" can be (1,000 or 1,024?)."""
    return f"{v:,.0f}" if v >= 1 else f"{v:g}"


def _sized(r: Results, c: str) -> str:
    """A chunking named with its measured size (data.size_label)."""
    return f"{c} ({size_label(r, c)})"


def _unit_title(r: Results, c: str, e: int) -> str:
    """A panel title's expansion, with the measured window when expanded."""
    if not e:
        return "expand_context 0"
    window = unit_tokens(r, c, e)
    return f"expand_context {e}" + (f" (window {window:,.0f} tok)" if window else "")


def _size_ticks(ax, r: Results, chunkings: list[str]) -> None:
    """Label a segment-size axis with each limit's measured mean segment tokens beneath it."""
    sizes = sorted(chunkings, key=lambda c: int(c[3:]))
    ax.set_xticks([int(c[3:]) for c in sizes],
                  [f"{c[3:]}\n{t:,.0f} tok" if (t := unit_tokens(r, c)) else c[3:] for c in sizes])


def _log_x(ax, base: int = 10, ticks=None) -> None:
    ax.set_xscale("log", base=base)
    if ticks is not None:
        ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _human(v)))
    ax.xaxis.set_minor_formatter(NullFormatter())


def _token_x(ax, top: int | None = None) -> None:
    """Reader tokens on a linear axis: labelled every 500 (every 1,000 in narrow panels), minor gridlines every 250 (where
    recall and precision are measured: metrics.CURVE_BUDGETS). Data beyond `top` is drawn and
    can be panned to in `show`."""
    ax.set_xlim(0, top or config.EVAL.token_top)
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


def _coverage(reading: str) -> str:
    return f"coverage: {EVENT_TEXT[reading]}"


def _file(base: str, reading: str | None = None, chunking: str | None = None, r: Results | None = None) -> str:
    """A chart's file name: the reading as a suffix unless it is every, the chunking when several."""
    name = base + (f"_{reading}" if reading and reading != "every" else "")
    return name + (f"_{chunking}" if chunking and r is not None and len(r.chunkings) > 1 else "") + ".png"


def _expansions(r: Results) -> list[int]:
    """expand_context values to draw: 0 and the configured ones that were simulated."""
    return [0, *(e for e in r.expansions if e in config.EVAL.expansions)]


def _series(r: Results, c: str, limit: str, e: int, bound: bool = False) -> list[tuple[str, str, str, str]]:
    """(legend key, ranker, color, line style) at one vector search limit and expand_context e.
    Every line reads only the top `limit` seeds: the first stage in vector order; each model
    scoring the seeds (solid: rerank, then read the seeds or their windows); each model scoring
    the windows (dashed: expand, then rerank; only where they were scored). bound adds the
    perfect reranker, which is a bound only for "all evidence read in full" with seeds alone
    (rankers.py): pass it only there."""
    tail = f"+x{e}" if e else ""
    out = [("first_stage", f"first_stage@{limit}{tail}", SERIES["first_stage"], "-")]
    out += [(m, f"rerank:{m}@{limit}{tail}", SERIES[m], "-") for m in r.models_at(c)]
    if e:
        out += [(f"{m}~windows", f"xrerank:{m}@{limit}{tail}", SERIES[m], DASH) for m in r.models_at(c)]
    if bound and not e:
        out.append(("oracle", f"oracle@{limit}", ORACLE, "-"))
    return [x for x in out if not r.rows(c, x[1]).empty]


def _name_at(key: str, limit: str, e: int) -> str:
    """The ranker name of a _series legend key at another vector search limit."""
    base, _, what = key.partition("~")
    tail = f"+x{e}" if e else ""
    if base in ("first_stage", "oracle"):
        return f"{base}@{limit}{tail}"
    return f"{'xrerank' if what else 'rerank'}:{base}@{limit}{tail}"


def _priced(m: str) -> str:
    """A model's label for charts that put a price on its tokens: a local model has none of its
    own, so it is priced as its stand-in on DeepInfra, and says so."""
    spec = RERANKERS[m]
    if spec.usd_per_mtok is None:
        return f"{spec.label}, priced as {RERANKERS[spec.priced_as].label} on DeepInfra (${price(m):g}/M)"
    return f"{spec.label} (DeepInfra ${price(m):g}/M)"


def _ranker_groups(r: Results, keys: set[str]) -> list[tuple[str, list, list[str]]]:
    """Two legend columns for charts drawn with _series: color and marker = how the seeds are
    ordered, line style = what the reranker scores."""
    bases = {k.split("~")[0] for k in keys}
    color_keys = [k for k in ("first_stage", *r.models, "oracle") if k in bases]
    names = {"first_stage": "first stage (vector search order)",
             "oracle": "perfect reranker, seeds only: a bound for the solid lines at expand_context 0"}
    labels = [names.get(k, LABEL.get(k, k)) for k in color_keys]
    groups = [("order (color, marker)", [_key(SERIES.get(k, ORACLE), marker=MARKERS.get(k, "o")) for k in color_keys], labels)]
    if any("~windows" in k for k in keys):
        groups.append(("what the reranker scores (line)", [_key(INK2), _key(INK2, DASH)],
                       ["the seeds (rerank, then read the seeds or their windows)",
                        "the windows (expand, then rerank)"]))
    return groups


SAME_LIMIT = ("Each panel is one vector search limit L and one expand_context E: every line reads only the top L seeds of the "
              "same vector search, each with its window of E neighbouring segments, in vector order (first stage) or reranked.")
BOUND = (" Gray: a perfect reranker over the same seeds, which reads only the evidence seeds: no reranker reads all the "
         "evidence in full with fewer tokens (drawn only at expand_context 0, and only for this reading).")
TOKENS_EQUAL = "Held equal: reader tokens, the vector search limit and the expansion (panel). Not equal: reranker tokens. " + SAME_LIMIT


def _limit_grid(r: Results) -> list[str]:
    """Vector search limits for panel columns: those shown for any chunking."""
    return sorted({x for c in r.chunkings for x in r.limits(c, config.EVAL.limits)}, key=lambda d: int(d[1:]))


def coverage_by_tokens(r: Results, out, head: str) -> None:
    limits, expansions = _limit_grid(r), _expansions(r)
    for c in r.chunkings:
        for reading in config.READINGS:
            fig, axes = _figure(len(expansions), len(limits), panel=(4.0, 3.0), sharex=True, sharey=True)
            keys = set()
            for i, e in enumerate(expansions):
                for j, limit in enumerate(limits):
                    ax = axes[i][j]
                    if limit not in r.limits(c, config.EVAL.limits):
                        ax.set_visible(False)
                        continue
                    for key, name, color, ls in _series(r, c, limit, e, bound=reading == "every"):
                        _steps(ax, r.rows(c, name)[reading].to_numpy(), 0, 16000, TOKEN_MARKS, _marker(key), color=color,
                               linestyle=ls, linewidth=THIN if key == "oracle" else LINE)
                        keys.add(key)
                    _token_x(ax)
                    _share_y(ax)
                    ax.set_title(f"{c} · L {limit[1:]} · {_unit_title(r, c, e)}", loc="left")
                    if i == len(expansions) - 1:
                        ax.set_xlabel("reader tokens (rendered context)\nexact at every budget; markers every 1,000")
                    if j == 0:
                        ax.set_ylabel(f"share of questions: {EVENT_TEXT[reading]}")
            _save(fig, out / (f"coverage_tokens_{reading}" + (f"_{c}" if len(r.chunkings) > 1 else "") + ".png"),
                  f"Coverage ({EVENT_TEXT[reading]}) vs reader tokens, {_sized(r, c)} — {head}", groups=_ranker_groups(r, keys),
                  note=TOKENS_EQUAL + (BOUND if reading == "every" else "") + " " + EVENT_NOTE)


def recall_precision_f1(r: Results, out, head: str) -> None:
    for c in r.chunkings:
        limits = r.limits(c, config.EVAL.limits)
        for e in _expansions(r):
            for axis, fmt, grid, xlabel in (("tokens", "t{}", CURVE_BUDGETS, "reader tokens (rendered context)"),
                                            ("k", "k{}", CURVE_KS, "segments read (k, window neighbours included)")):
                xs = [v for v in grid if f"recall_{fmt.format(v)}" in r.df]  # results simulated before the finer grid
                fig, axes = _figure(3, len(limits), panel=(4.0, 2.6), sharex=True)
                keys = set()
                for j, limit in enumerate(limits):
                    for key, name, color, ls in _series(r, c, limit, e):
                        rows = r.rows(c, name)
                        rec = np.stack([rows[f"recall_{fmt.format(v)}"].to_numpy() for v in xs], 1)
                        pre = np.stack([rows[f"precision_{fmt.format(v)}"].to_numpy() for v in xs], 1)
                        f1 = np.where(rec + pre > 0, 2 * rec * pre / np.maximum(rec + pre, 1e-12), 0.0)
                        every = [n for n, v in enumerate(xs) if v % (1000 if axis == "tokens" else 5) == 0]
                        for i, values in enumerate((rec, pre, f1)):
                            _plot(axes[i][j], xs, values.mean(0), color=color, linestyle=ls, marker=_marker(key),
                                  markevery=every, linewidth=LINE)
                        keys.add(key)
                    axes[0][j].set_title(f"{c} · L {limit[1:]} · {_unit_title(r, c, e)}", loc="left")
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
                        "Held equal: segments read (x; window neighbours count), the vector search limit and the expansion "
                        "(panel). Not equal: reader tokens, reranker tokens. " + SAME_LIMIT)
                name = f"rpf_{axis}" + (f"_x{e}" if e else "")
                _save(fig, out / _file(name, None, c, r), f"Gold-character recall, precision and F1 vs {xlabel}, {_sized(r, c)} — {head}",
                      groups=_ranker_groups(r, keys), note=note)


def coverage_counts(r: Results, out, head: str, reading: str) -> None:
    """Coverage at equal counts instead of tokens: against seeds kept (k) and segments read (m)
    at one vector search limit L (EventMemory's default, 20, where scored), and at k = 10 kept
    seeds as L grows; a row per expand_context."""
    if f"{reading}_ctx" not in r.df:
        return
    for c in r.chunkings:
        limits = r.limits(c, config.EVAL.limits)
        if not limits:
            continue
        limit = "n20" if "n20" in limits else limits[-1]
        expansions = _expansions(r)
        fig, axes = _figure(len(expansions), 3, panel=(4.0, 3.0))
        keys = set()
        for i, e in enumerate(expansions):
            series = _series(r, c, limit, e, bound=reading == "every")
            for j, (column, top, xlabel) in enumerate(((f"{reading}_ctx", min(50, int(limit[1:])), "seeds kept (k)"),
                                                        (f"{reading}_k", 200, "segments read (m = max_num_segments)"))):
                ax = axes[i][j]
                for key, name, color, ls in series:
                    _steps(ax, r.rows(c, name)[column].to_numpy(), 0, top,
                           range(5 if top <= 50 else 20, top + 1, 5 if top <= 50 else 20), _marker(key), color=color,
                           linestyle=ls, linewidth=THIN if key == "oracle" else LINE)
                    keys.add(key)
                _k_x(ax, top)
                ax.set_xlabel(f"{xlabel}\nexact at every count")
                _share_y(ax)
                ax.set_title(f"{c} · L {limit[1:]} · {_unit_title(r, c, e)}: equal {'seeds kept' if j == 0 else 'segments'}",
                             loc="left")
                if j == 0:
                    ax.set_ylabel(_coverage(reading))
            ax = axes[i][2]
            values = [int(d[1:]) for d in limits]
            for key, _, color, ls in series:
                names = [_name_at(key, d, e) for d in limits]
                ys = [(r.rows(c, n)[f"{reading}_ctx"] <= 10).mean() if not r.rows(c, n).empty else np.nan for n in names]
                _plot(ax, values, ys, color=color, linestyle=ls, marker=_marker(key), linewidth=THIN if key == "oracle" else LINE)
            _log_x(ax, 10, values)
            _share_y(ax)
            ax.set_xlabel("vector search limit L (seeds retrieved; a reranker scores all of them)\nmarkers: the limits measured")
            ax.set_title(f"{c} · {_unit_title(r, c, e)}: 10 seeds kept, as L grows", loc="left")
        _save(fig, out / _file("coverage_counts", reading, c, r), f"Coverage at equal counts instead of tokens, {_sized(r, c)} — {head}",
              groups=_ranker_groups(r, keys),
              note="Held equal: seeds kept (left), segments read (middle), the vector search limit L (each panel; right: "
                   "as it grows) and the expansion (row). Not equal: reader tokens and reranker tokens. " + SAME_LIMIT
                   + (BOUND if reading == "every" else ""))


def _size_styles(chunkings: list[str]) -> dict[str, dict]:
    sizes = sorted({int(c[3:]) for c in chunkings})
    ramp = RAMP[3] if len(sizes) <= 3 else RAMP[5]
    color = dict(zip(sizes[-len(ramp):], ramp))  # beyond 5 sizes, the smallest go unplotted
    return {c: {"color": color[int(c[3:])], "ls": DASH if c.startswith("rec") else "-",
                "marker": "1" if c.startswith("rec") else "2"} for c in chunkings if int(c[3:]) in color}


def rerank_cost(r: Results, out, head: str, reading: str) -> None:
    """Coverage at each configured reader budget against reranker tokens, as the vector search
    limit grows (one marker per limit), with the first stage at the same limits drawn at the
    same x: what the same seeds give without a reranker."""
    if not r.models:
        return
    styles, budgets = _size_styles(r.chunkings), config.EVAL.budgets
    fig, axes = _figure(len(r.models), len(budgets), panel=(3.8, 3.2), sharey=True)
    for i, m in enumerate(r.models):
        for j, b in enumerate(budgets):
            ax = axes[i][j]
            for c, s in styles.items():
                pts = []
                for d in r.depths_at(c, m):
                    rows, base = r.rows(c, f"rerank:{m}@{d}"), r.rows(c, f"first_stage@{d}")
                    if not rows.empty and not base.empty:
                        pts.append((rows.rr_tokens.mean(), (rows[reading] <= b).mean(), (base[reading] <= b).mean()))
                if not pts:
                    continue
                xs, ys, bs = zip(*sorted(pts), strict=True)
                _plot(ax, xs, ys, color=s["color"], linestyle=s["ls"], marker=s["marker"])
                _plot(ax, xs, bs, color=s["color"], linestyle=":", linewidth=THIN, marker="x", markersize=4)
            _log_x(ax)
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            _share_y(ax)
            ax.set_title(f"{LABEL[m]} · reader budget {b:,}", loc="left")
            ax.set_xlabel("reranker tokens per query (API billing)")
            if j == 0:
                ax.set_ylabel(_coverage(reading))
    keys = list(styles)
    handles = [_key(styles[k]["color"], styles[k]["ls"], styles[k]["marker"]) for k in keys] + [_key(INK2, ":", "x")]
    labels = [f"{_sized(r, k)}, reranked" for k in keys] + ["first stage at the same vector search limits (no reranker tokens)"]
    _save(fig, out / _file("rerank_cost", reading), f"Coverage vs reranker tokens as the vector search limit grows — {head}",
          handles, labels, ncol=2,
          note="Held equal: reader tokens (per panel) and, at each marker, the vector search limit (5, 10, 20, ...). x: the "
               "reranker's tokens per query at that limit; the dotted line is the first stage over the same seeds, drawn at "
               "the same x though it spends no reranker tokens. Seeds only (expand_context 0).")


def gain_sizes(r: Results, out, head: str, reading: str) -> None:
    """Reranker gain against segment size: coverage reranked minus coverage of the first stage
    over the same seeds (same vector search limit), at each configured budget, with 95%
    intervals; a panel per (model, limit)."""
    sized = sorted((c for c in r.chunkings if c.startswith("seg")), key=lambda c: int(c[3:]))
    if len(sized) < 2 or not r.models:
        return
    limits = [d for d in _limit_grid(r) if all(d in r.limits(c, config.EVAL.limits) for c in sized)]
    if not limits:
        return
    budgets = config.EVAL.budgets
    ramp = dict(zip(budgets, RAMP[5] if len(budgets) > 3 else RAMP[3], strict=False))
    fig, axes = _figure(len(r.models), len(limits), panel=(3.8, 3.2), sharey=True)
    sizes = [int(c[3:]) for c in sized]
    for i, m in enumerate(r.models):
        for j, d in enumerate(limits):
            ax = axes[i][j]
            for k, b in enumerate(budgets):
                means, los, his = [], [], []
                for c in sized:
                    rows, base = r.rows(c, f"rerank:{m}@{d}"), r.rows(c, f"first_stage@{d}")
                    gain = (rows[reading].to_numpy() <= b).astype(float) - (base[reading].to_numpy() <= b)
                    lo, hi = interval(gain, base.cluster.to_numpy())
                    means.append(gain.mean())
                    los.append(gain.mean() - lo if np.isfinite(lo) else 0.0)
                    his.append(hi - gain.mean() if np.isfinite(hi) else 0.0)
                xs = [x * (1 + 0.03 * (k - (len(budgets) - 1) / 2)) for x in sizes]  # side by side
                ax.errorbar(xs, means, yerr=[los, his], color=ramp.get(b, INK2), linewidth=THIN, capsize=0, fmt="none")
                _plot(ax, xs, means, color=ramp.get(b, INK2), marker="2")
            ax.axhline(0, color=INK2, linewidth=0.8)
            _log_x(ax, 2, sizes)
            _size_ticks(ax, r, sized)
            ax.set_title(f"{LABEL[m]} · L {d[1:]}", loc="left")
            ax.set_xlabel("segment limit (characters) and measured mean segment of the retrieved seeds")
            if j == 0:
                ax.set_ylabel(f"gain in {_coverage(reading)}")
    _save(fig, out / _file("gain_sizes", reading), f"Reranker gain vs segment size — {head}",
          [_key(ramp.get(b, INK2), marker="2") for b in budgets], [f"reader budget {b:,}" for b in budgets], ncol=4,
          note="Held equal: the vector search limit (panel) and the reader budget (line). y: coverage reranked minus coverage "
               "of the first stage over the same seeds, mean with 95% interval over questions. Seeds only (expand_context 0).")


def break_even(r: Results, out, head: str, reading: str) -> None:
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
        api, lo, hi, _, _ = frontier.break_even(r, m, "rr_tokens", event=reading)
        cached = frontier.break_even(r, m, "rr_prefix_tokens", with_interval=False, event=reading)[0]
        lines[m] = finite(api) * price(m)
        _plot(ax, x, lines[m], drawstyle="steps-mid", color=SERIES[m])
        _plot(ax, x, finite(cached) * price(m), drawstyle="steps-mid", color=SERIES[m], linestyle=DASH)
        band_lo, band_hi = finite(lo) * price(m), np.where(hi >= 1e12, BREAK_EVEN_TOP, hi * price(m))
        ax.fill_between(x, band_lo, band_hi, where=np.isfinite(band_lo) & np.isfinite(lines[m]), step="mid", color=SERIES[m],
                        alpha=0.12, linewidth=0)  # only where there is a line: elsewhere the estimate is "never"
    ax.text(0.99, 0.97, "above a line: reranking is cheaper overall\nbelow it: the first stage alone is cheaper",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK2)
    ax.set_xlim(x[0] - 0.02, 1.0)
    ax.set_yscale("log")
    ax.set_ylim(0.01, BREAK_EVEN_TOP)
    _log_y(ax)
    ax.set_ylabel("break-even LLM input price ($ per million tokens)")
    ax.set_xlabel(f"coverage target ({EVENT_TEXT[reading]})")
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    handles = [_key(SERIES[k]) for k in r.models] + [_key(INK2), _key(INK2, DASH)]
    labels = ([_priced(k) for k in r.models] + ["reranker tokens as an API bills them (band: 95% interval)",
              "with a prefix cache (prompt and question computed once)"])
    _save(fig, out / _file("break_even", reading), f"Above which LLM price does reranking pay for itself? — {head}", handles,
          labels, ncol=2,
          note=f"Held equal: coverage (x: the share of questions with {EVENT_TEXT[reading]}). y: the LLM input price at which "
               "reaching that coverage costs the same (reader + reranker tokens) with and without the reranker, each at its "
               "cheapest configuration (any segment length, expansion and vector search limit), chosen on the same questions. "
               "Above the line reranking is cheaper; no line: never cheaper at that target. Band: 95% interval of the line "
               "(reaching the top: it includes never).")


def cost_saving(r: Results, out, head: str, reading: str) -> None:
    """Over a continuous range of LLM prices and coverage targets: the cost of reaching the target
    with the reranker (its cheapest configuration) relative to without it. Blue saves, red costs
    more; gray where only the reranker reaches the target."""
    if not r.models:
        return
    llm = np.logspace(-2, 2, 81)  # $ per million reader tokens
    fig, axes = _figure(1, len(r.models), panel=(4.6, 3.8), sharey=True)
    for k, m in enumerate(r.models):
        curves = frontier.cost_curves(r, m, list(llm), price(m), event=reading)
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
            ax.set_ylabel(f"coverage target ({EVENT_TEXT[reading]})")
    fig.colorbar(im, ax=axes[0].tolist(), label="% of total cost saved by reranking (red: reranking costs more)")
    _save(fig, out / _file("cost_saving", reading), f"Cost saved by reranking, at every LLM price and coverage target — {head}",
          [Patch(color=ORACLE, alpha=0.6)], ["only the reranker reaches this target"], ncol=1,
          note="Held equal: the coverage target (y) and the LLM input price (x). Color: total cost (reader + reranker tokens) "
               "of reaching the target with the reranker, relative to without it, each at its cheapest configuration (any "
               "segment length, expansion and vector search limit), chosen on the same questions. Blank: neither reaches the "
               "target. Matches the break-even line.")


def query_types(r: Results, out, head: str, reading: str) -> None:
    """Reranker gain by question type at each configured budget: reranked minus the first stage
    over the same seeds (EventMemory's default limit, 20, where scored)."""
    if not r.models or r.df.qid.nunique() < 20:
        return
    c = default_segment() if default_segment() in r.chunkings else r.chunkings[0]
    limits = r.limits(c, config.EVAL.limits)
    if not limits:
        return
    limit = "n20" if "n20" in limits else limits[-1]
    base, labels = query_labels(r, c, f"first_stage@{limit}")
    groups = [(f, v) for f in ("question_type", "form", "length", "context", "parts") if f in labels
              for v in sorted(set(labels[f])) if (labels[f] == v).sum() >= 10]
    if not groups:
        return
    budgets = config.EVAL.budgets
    fig, axes = _figure(1, len(budgets), panel=(3.4, 0.34 * len(groups) + 1.2), sharey=True)
    for j, b in enumerate(budgets):
        ax = axes[0][j]
        hit = (base[reading].to_numpy() <= b).astype(float)
        for k, m in enumerate(r.models):
            gain = (r.rows(c, f"rerank:{m}@{limit}")[reading].to_numpy() <= b).astype(float) - hit
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
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.set_title(f"reader budget {b:,}", loc="left")
        ax.set_xlabel("gain in coverage")
    _save(fig, out / _file("query_types", reading), f"Reranker gain by question type ({c}, L {limit[1:]}) — {head}",
          [_key(SERIES[m], marker="o") for m in r.models], [LABEL[m] for m in r.models],
          note=f"Held equal: the reader budget (panel) and the vector search limit ({limit[1:]}): gain = {_coverage(reading)} "
               "reranked minus the first stage over the same seeds. Dot: mean; line: 95% interval. Seeds only.")


def first_stage_sizes(r: Results, out, head: str, reading: str) -> None:
    sizes = sorted({int(c[3:]) for c in r.chunkings if c[:3] in ("fix", "rec", "seg")})
    if len(sizes) < 3:
        return
    budgets = config.EVAL.budgets
    fig, axes = _figure(1, len(budgets), panel=(3.4, 3.2), sharey=True)
    styles = CHUNKERS
    present = set()
    for j, b in enumerate(budgets):
        ax = axes[0][j]
        for fam, (color, marker, ls) in styles.items():
            xs = [n for n in sizes if f"{fam}{n}" in r.chunkings]
            if not xs:
                continue
            ys = [(r.rows(f"{fam}{n}", "first_stage")[reading] <= b).mean() for n in xs]
            _plot(ax, xs, ys, color=color, marker=marker, linestyle=ls)
            present.add(fam)
        _log_x(ax, 2, sizes)
        if set(present) == {"seg"}:
            _size_ticks(ax, r, [f"seg{n}" for n in sizes])
        _share_y(ax)
        ax.set_title(f"reader budget {b:,}", loc="left")
        ax.set_xlabel("segment / chunk size; markers: sizes measured")
        if j == 0:
            ax.set_ylabel(f"{_coverage(reading)}, no reranker")
    names = {"fix": "fixed windows", "rec": "recursive, size-matched", "seg": "EventMemory segments (characters)"}
    keys = [k for k in styles if k in present]
    _save(fig, out / _file("first_stage_sizes", reading), f"First-stage coverage vs segment size (no reranker) — {head}",
          [_key(styles[k][0], styles[k][2], styles[k][1]) for k in keys], [names[k] for k in keys],
          note="Held equal: reader tokens (per panel). No reranker: the first stage reads as deep as the budget allows "
               "(this chart compares segment sizes, not rerankers).")


EXPANSION_APPROACHES = (("plain chunks of the read size", "#2a78d6"), ("seeds, expanded (no reranker)", "#8f8e89"),
                        ("rerank seeds, then expand", "#1baf7a"), ("expand, then rerank windows", "#eda100"))
EXPANSION_MARKERS = dict(zip((a for a, _ in EXPANSION_APPROACHES), ("x", "+", "2", "1"), strict=True))


@dataclass(frozen=True)
class ExpansionPanel:
    """One (model, expand_context) comparison. seed: the chunking whose seeds are expanded;
    window: their measured mean window (segment-text tokens); plain: the unexpanded chunking
    whose measured unit is closest to the window, and plain_size its measured size."""

    model: str
    expand: int
    seed: str
    window: float | None
    plain: str | None
    plain_size: float | None
    items: tuple[tuple[str, str, str], ...]

    @property
    def ratio(self) -> float | None:
        return self.plain_size / self.window if self.plain_size and self.window else None

    @property
    def matched(self) -> bool:
        return self.ratio is not None and abs(self.ratio - 1) <= config.EVAL.size_tolerance

    def sizes(self) -> str:
        window = f"window {self.window:,.0f} tok" if self.window else "window not measured"
        if self.plain is None:
            return f"{window}; no unexpanded chunking to compare"
        plain = f"plain {self.plain} {self.plain_size:,.0f} tok"
        return f"{window}; {plain}" + ("" if self.matched else f" — NOT size-matched ({self.ratio:.2f}×)")


def expansion_rows(r: Results) -> list[ExpansionPanel]:
    """Per (model, expand_context): the four approaches as (label, chunking, ranker) when
    present. Seeds are EventMemory's default segments when expanded, else the smallest chunking
    with expansions; the plain comparison is the unexpanded chunking whose measured mean unit
    (data.unit_tokens) is closest to the seeds' measured mean window."""
    expanded = sorted({c for c in r.df.loc[r.df.expand > 0, "chunking"]}, key=lambda c: int(c[3:]))
    if not expanded:
        return []
    seed = default_segment() if default_segment() in expanded else expanded[0]
    sized = [(c, unit_tokens(r, c)) for c in r.chunkings if c != seed]
    sized = [(c, t) for c, t in sized if t]
    out = []
    for e in r.expansions:
        if not ((r.df.chunking == seed) & (r.df.expand == e)).any():
            continue
        window = unit_tokens(r, seed, e)
        plain, plain_size = (min(sized, key=lambda ct: abs(math.log(ct[1] / window))) if sized and window
                             else (None, None))
        for m in r.models:
            depth = r.common_depth(seed, [m])
            items = [(EXPANSION_APPROACHES[1][0], seed, f"first_stage@{depth}+x{e}"),
                     (EXPANSION_APPROACHES[2][0], seed, f"rerank:{m}@{depth}+x{e}"),
                     (EXPANSION_APPROACHES[3][0], seed, f"xrerank:{m}@{depth}+x{e}")]
            if plain is not None:
                items.insert(0, (EXPANSION_APPROACHES[0][0], plain, f"rerank:{m}@{r.common_depth(plain, [m])}"))
            items = tuple((a, c, n) for a, c, n in items if not r.rows(c, n).empty)
            if len(items) > 1:
                out.append(ExpansionPanel(m, e, seed, window, plain, plain_size, items))
    return out


def expansion(r: Results, out, head: str, reading: str) -> None:
    panels = expansion_rows(r)
    if not panels:
        return
    expands = sorted({p.expand for p in panels})
    models = [m for m in r.models if any(p.model == m for p in panels)]
    fig, axes = _figure(len(expands), len(models), sharey=True)
    colors = dict(EXPANSION_APPROACHES)
    present = set()
    for p in panels:
        i, j = expands.index(p.expand), models.index(p.model)
        ax = axes[i][j]
        for label, c, name in p.items:
            _steps(ax, r.rows(c, name)[reading].to_numpy(), 0, 16000, TOKEN_MARKS, EXPANSION_MARKERS[label],
                   color=colors[label])
            present.add(label)
        _token_x(ax)
        _share_y(ax)
        ax.set_title(f"{LABEL[p.model]} · expand_context {p.expand}\n{p.sizes()}", loc="left")
        ax.set_xlabel("reader tokens: exact steps, one per question; markers every 1,000")
        if j == 0:
            ax.set_ylabel(_coverage(reading))
    for i, e in enumerate(expands):  # panels without a comparison stay empty
        for j, m in enumerate(models):
            if not any(p.expand == e and p.model == m for p in panels):
                axes[i][j].set_visible(False)
    keys = [k for k, _ in EXPANSION_APPROACHES if k in present]
    unmatched = any(not p.matched for p in panels if p.plain is not None)
    _save(fig, out / _file("expansion", reading), f"Embedded vs reranked vs read unit — {head}",
          [_key(colors[k], marker=EXPANSION_MARKERS[k]) for k in keys], keys, ncol=2,
          note="Sizes are measured: mean segment-text tokens per retrieved seed (its window when expanded), over every "
               "seed the first stage retrieved. Plain chunks: the unexpanded chunking closest to the window's measured size"
               + (f"; panels flag pairs more than {config.EVAL.size_tolerance:.0%} apart" if unmatched else "")
               + ". Held equal: reader tokens (x). Not equal: reranker tokens and vector search limit.")


ALL_READINGS = ("coverage", "rpf")  # drawn for every reading or measure at once
CHARTS = {"coverage": "coverage_by_tokens", "rpf": "recall_precision_f1", "counts": "coverage_counts",
          "rerank_cost": "rerank_cost", "gain_sizes": "gain_sizes", "break_even": "break_even", "cost_saving": "cost_saving",
          "query_types": "query_types", "sizes": "first_stage_sizes", "expansion": "expansion"}


def _head(r: Results) -> str:
    return f"{r.dataset} · {r.scope} · {r.variant} · {r.df.qid.nunique()} questions"


def _draw(r: Results, out, keys) -> None:
    for key in keys:
        chart = globals()[CHARTS[key]]
        if key in ALL_READINGS:
            chart(r, out, _head(r))
        else:
            for reading in config.EVAL.readings:
                chart(r, out, _head(r), reading)


def write(r: Results) -> str:
    out = FIGURES / r.embedder / f"{r.dataset}__{r.scope}__{r.variant}__{r.sample}"
    out.mkdir(parents=True, exist_ok=True)
    _draw(r, out, list(CHARTS))
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
        _draw(r, out, charts or list(CHARTS))
        plt.show()
    finally:
        _SHOWING = False
