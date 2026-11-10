"""Charts for one (embedder, dataset, scope, variant, sample); the markdown is their table view.
The values they are drawn at (reader budgets, vector search limits, readings, expansions) come
from rbench.config.EVAL. Files (a suffix _<chunking> when several chunkings were simulated):

  coverage_tokens_<reading>.png  coverage vs reader tokens: a row per expand_context, a column per
                                 vector search limit L; in each panel every line reads only the top
                                 L seeds: the first stage in vector order, each reranker scoring the
                                 seeds (up caltrop) or their windows (down caltrop, where scored)
  rpf_<tokens|k>[_x<E>].png      gold-character recall, precision and F1 vs tokens and vs k
  coverage_counts.png            coverage at equal counts (seeds kept, segments read, L)
  rerank_cost.png                coverage vs reranker tokens as L grows, beside the first stage at
                                 the same L, at each configured budget
  gain_sizes.png                 reranker gain (same L) vs segment size, at each budget
  cost_equal.png                 coverage gained by reranking for the same money, over LLM price x spend
  query_types.png                reranker gain (same L) by question type, at each budget
  first_stage_sizes.png          first-stage coverage vs segment size (no reranker)
  expansion.png                  plain chunks vs small seeds expanded to the same read size
Single-reading charts are drawn for each reading in EVAL.readings, with a _<reading> suffix
except for "every".

Colors follow the dataviz reference palette and were checked with its validator: models take
their color, markers and label from their registry entry (rerankers.RERANKERS); the perfect
reranker is neutral gray; ordered values (segment sizes, budgets) use a one-hue ordinal ramp;
savings use a diverging blue (saves) / red (costs more) scale around a neutral gray midpoint.

Reading the charts: every data line is thin, solid and semi-transparent (one width, one alpha),
with no outline, so where lines coincide both colors show. Color is the ranking: the first
stage's blue, each model its own hue, the bound gray; a family of settings (a model's seeds and
windows, expand_context values, k values, chunkers, budgets) takes close but distinguishable
shades of one hue (palette.py, checked by tests/test_palette.py). Lines measured everywhere
(coverage steps, one per question; recall, precision and F1 at every minor gridline:
metrics.CURVE_BUDGETS, CURVE_KS) have no markers; lines joining sampled measurements (vector
search limits, sizes, budgets) carry a hollow marker on each measured point, in the line's
color (the user's rule, 2026-11-10). Token and count axes are linear. Under each title, a line
says what is held equal and what is not.

python -m rbench show <dataset> opens the same charts in zoomable matplotlib windows (show()).
"""

from __future__ import annotations

import itertools
import math
import textwrap
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator, NullFormatter  # noqa: E402

from rbench import config  # noqa: E402
from rbench.metrics import CURVE_BUDGETS, CURVE_KS  # noqa: E402
from rbench.paths import FIGURES  # noqa: E402
from rbench.report import frontier  # noqa: E402
from rbench.report import palette  # noqa: E402
from rbench.report import readings as rd  # noqa: E402
from rbench.report.data import (  # noqa: E402
    Results,
    default_segment,
    price,
    query_labels,
    unit_tokens,
)
from rbench.report.stats import interval  # noqa: E402
from rbench.rerankers import RERANKERS  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# Models take their color, markers and label from their registry entry (rerankers.RERANKERS).
SERIES = {"first_stage": "#2a78d6", **{m: spec.color for m, spec in RERANKERS.items()}}
ORACLE = "#8f8e89"
MUTED = "#e4e3df"  # cost chart: no clear difference (the grid's gray, lighter than the diverging scale's ends)
FIRST_ONLY = "#c6b3ea"  # cost chart: only the first stage reaches the target (a hue outside the diverging scale)
# Markers sit only on measured points of sampled curves (hollow, in the line's color); lines
# measured everywhere (coverage steps, the recall grids) have none: each line has its own color
# (the user's rule, 2026-11-10).
LABEL = {"first_stage": "first stage only", **{m: spec.label for m, spec in RERANKERS.items()}}
# color and marker per chunker (left / right caltrop, apart from the rankings' shapes); lines solid
CHUNKERS = {"seg": ("#1baf7a", "3", "-"), "cut": ("#2a78d6", "4", "-")}
CHUNKER_NAMES = {"seg": "EventMemory segments (TextSegmenter)", "cut": "simple chunker (a cut every N characters)"}
RAMP = {3: ("#86b6ef", "#2a78d6", "#104281"), 5: ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")}
DIVERGING = LinearSegmentedColormap.from_list("red_gray_blue", ["#a12a2a", "#e34948", "#f2b8b5", "#f0efec", "#9ec5f4",
                                                                  "#3987e5", "#104281"])  # equal steps per arm
# What each coverage event means, in words (metrics.py has the definitions).
EVENT_TEXT = {"first": "some evidence read in full", "every": "all evidence read in full",
              "first_touch": "some evidence at least partly read", "every_touch": "all evidence at least partly read",
              "share": "share of evidence turns read in full", "share_touch": "share of evidence turns at least partly read"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 1, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titlecolor": INK, "lines.linewidth": 1.0, "lines.solid_capstyle": "round",
    "lines.markeredgewidth": 0.6, "lines.markeredgecolor": SURFACE,
    "legend.frameon": False, "legend.labelcolor": INK2, "legend.fontsize": 9, "legend.handlelength": 4.2,
    "legend.handletextpad": 0.6, "legend.columnspacing": 2.4, "figure.constrained_layout.h_pad": 0.08,
})
PANEL = (4.4, 3.2)  # inches per panel; figures are at least MIN_WIDTH wide so the legend never dwarfs the plot
MIN_WIDTH = 7.5
LINE = 1.0  # the one line width (points): thin, with no outline, so overlapping lines stay whole
ALPHA = 0.75  # every data line, the same: where lines coincide both colors show (the user's choice, 2026-11-10)


def _hollow(kwargs: dict) -> dict:
    """Thin line-only markers in the line's color (unless a fill is asked for)."""
    if kwargs.get("marker"):
        kwargs.setdefault("markerfacecolor", "none")
        kwargs.setdefault("markeredgecolor", kwargs.get("color"))
        kwargs.setdefault("markeredgewidth", 1.2)
        kwargs.setdefault("markersize", {"x": 6, "+": 7}.get(kwargs["marker"], 8))  # x and + look larger at one size
    return kwargs


def _plot(ax, x, y, **kwargs):
    """ax.plot of a thin plain line (no outline: an outline cuts gaps into the lines beneath),
    with its hollow line-only markers drawn on top."""
    kwargs.setdefault("linewidth", LINE)
    kwargs.setdefault("alpha", ALPHA)
    marks = {k: kwargs.pop(k) for k in ("marker", "markersize", "markerfacecolor", "markeredgecolor", "markeredgewidth",
                                        "markevery") if k in kwargs}
    lines = ax.plot(x, y, **kwargs)
    if marks.get("marker"):
        ax.plot(x, y, linestyle="none", zorder=lines[0].get_zorder() + 0.1, alpha=kwargs["alpha"],
                **_hollow({**marks, "color": kwargs.get("color")}))
    return lines


def _steps(ax, costs: np.ndarray, lo: float, hi: float, marks=None, marker: str | None = None, **kwargs):
    """Coverage as the exact step function of a reading's costs on [lo, hi] (readings.py): it
    rises at each question's cost (or evidence turn's), so it is measured at every budget, not
    interpolated. With a marker, the series' shape is drawn at `marks` (the labelled ticks)."""
    finite = rd.steps(costs)
    xs = np.concatenate([[lo], finite[(finite > lo) & (finite <= hi)], [hi]])
    lines = _plot(ax, xs, rd.coverage(costs, xs), drawstyle="steps-post", **kwargs)
    if marker and marks is not None:
        marks = np.asarray([m for m in marks if lo <= m <= hi], dtype=float)
        ax.plot(marks, rd.coverage(costs, marks), linestyle="none", zorder=3,
                **_hollow({"marker": marker, "color": kwargs.get("color")}))
    return lines


_SHOWING = False  # set by show(): figures stay open in windows instead of being written


def _figure(rows: int, cols: int, panel: tuple[float, float] = PANEL, **kwargs):
    """A constrained-layout figure of rows x cols panels; the title goes above it, the legend below."""
    width = max(MIN_WIDTH, cols * panel[0])
    return plt.subplots(rows, cols, figsize=(width, rows * panel[1] + 0.5), squeeze=False, layout="constrained", **kwargs)


def _human(v: float) -> str:
    """Plain numbers with thousands separators: "1,000" is never ambiguous, "1k" can be (1,000 or 1,024?)."""
    return f"{v:,.0f}" if v >= 1 else f"{v:g}"


def _sized(r: Results, c: str) -> str:
    """A chunking named with its measured mean segment, plainly: "seg500, mean segment 299
    characters" (the full measured unit is in report section 1)."""
    from rbench.report.data import segment_size

    size = segment_size(r, c)
    return f"{c}, mean segment {size[0]:,.0f} characters" if size else c


def _unit_title(r: Results, c: str, e: int) -> str:
    """A panel title's expansion, with the measured window when expanded."""
    if not e:
        return "expand_context 0"
    window = unit_tokens(r, c, e)
    return f"expand_context {e}" + (f" (window {window:,.0f} tok)" if window else "")


def _measured_x(ax, measured: dict[str, float]) -> None:
    """A log axis of measured sizes, a labelled tick at each chunking's measured size (name and
    tokens), padded so the outermost markers clear the frame."""
    ax.set_xscale("log", base=2)
    xs = sorted(measured.values())
    ax.set_xlim(xs[0] / 1.25, xs[-1] * 1.25)
    groups: list[list[tuple[str, float]]] = []  # sizes within 25% of each other share one tick (labels would overprint)
    for c, t in sorted(measured.items(), key=lambda ct: ct[1]):
        if groups and t / groups[-1][0][1] < 1.25:
            groups[-1].append((c, t))
        else:
            groups.append([(c, t)])
    ax.set_xticks([float(np.exp(np.mean([np.log(t) for _, t in g]))) for g in groups],
                  ["\n".join(f"{c} {t:,.0f} tok" for c, t in g) for g in groups])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def _log_x(ax, base: int = 10, ticks=None) -> None:
    ax.set_xscale("log", base=base)
    if ticks is not None:
        ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _human(v)))
    ax.xaxis.set_minor_formatter(NullFormatter())


def _token_x(ax, top: int | None = None) -> None:
    """Reader tokens on a linear axis to `top` (config.EVAL.token_top): labelled every 500,
    1,000 or 2,000, whichever keeps the labels apart in the panel's width, with minor gridlines
    at half the label step (recall and precision are measured every 250: metrics.CURVE_BUDGETS).
    Data beyond `top` is drawn and can be panned to in `show`."""
    top = top or config.EVAL.token_top
    ax.set_xlim(0, top)
    inches = ax.get_position().width * ax.figure.get_figwidth()
    step = next((s for s in (500, 1000, 2000, 4000) if top / s <= 2.2 * inches), 4000)  # about 2 labels per inch
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _human(v) if v else "0"))
    ax.xaxis.set_minor_locator(MultipleLocator(step / 2))
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
    """A legend sample: the line as plotted, with its marker drawn larger."""
    return Line2D([], [], linestyle=ls, linewidth=LINE, alpha=ALPHA,
                  **_hollow({"marker": marker, "color": color, "markersize": 9}))


def _save(fig, path, title: str, handles=None, labels=None, ncol: int | None = None,
          groups: list[tuple[str, list, list[str]]] | None = None, note: str | list[str] = "") -> None:
    """Title above the panels and legend below them (the figure must use constrained layout). `groups`
    are (heading, handles, labels) shown as side-by-side columns, e.g. what the color means
    and what the line style means; or pass plain handles and labels."""
    chars = int(fig.get_figwidth() * 12.5)  # characters per line at this font size: text never widens the figure
    lines = [note] if isinstance(note, str) else note  # a list: one short line per idea, most important first
    text = "\n".join([textwrap.fill(title, chars), *(textwrap.fill(x, chars) for x in lines if x)])
    fig.suptitle(text, x=0.01, ha="left", fontsize=10.5, color=INK, linespacing=1.5)
    legend_rows = (max(len(h) for _, h, _ in groups) + 1 if groups else
                   -(-len(handles) // (ncol or min(len(handles), 3))) if handles else 0)
    # Title, note and legend add height instead of taking it from the panels.
    fig.set_figheight(fig.get_figheight() + 0.28 * (text.count("\n") + 1) + 0.25 * legend_rows + 0.2)
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
    _fit(fig)
    if _SHOWING:
        fig.canvas.manager.set_window_title(path.stem)
        return
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _wrap(t, length: float, renderer, max_lines: int = 4, vertical: bool = False) -> None:
    """Wrap a text at word boundaries until it is no longer than `length` pixels (its height
    when vertical: a rotated y label)."""
    def size() -> float:
        box = t.get_window_extent(renderer)
        return box.height if vertical else box.width

    original = t.get_text()
    if not original or size() <= length:
        return
    for chars in range(len(original), 8, -2):
        t.set_text(textwrap.fill(original, chars))
        if size() <= length or t.get_text().count("\n") + 1 >= max_lines:
            return


def _fit(fig) -> None:
    """Fit every text to its space: panel titles and x labels wrap to their panel's width, and
    legend labels to their share of the figure's width, so no text runs into another or past
    the figure's edge (tests/test_layout.py checks every chart)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        box = ax.get_window_extent(renderer)
        for t in (ax._left_title, ax.title, ax.xaxis.label):
            _wrap(t, box.width, renderer)
        _wrap(ax.yaxis.label, box.height, renderer, vertical=True)
    for legend in fig.legends:
        columns = max(1, legend._ncols)
        texts = legend.get_texts()
        originals = [t.get_text() for t in texts]
        room = fig.bbox.width / columns - 70  # px per label: the column less its line sample and spacing
        for _ in range(8):  # narrower until the legend fits the figure's width
            for t, text in zip(texts, originals, strict=True):
                t.set_text(text)
                _wrap(t, room, renderer)
            fig.canvas.draw()
            if legend.get_window_extent(renderer).width <= fig.bbox.width:
                break
            room *= 0.9
    fig.canvas.draw()


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
    scoring the seeds (up caltrop: rerank, then read the seeds or their windows); each model
    scoring the windows (down caltrop: expand, then rerank; only where they were scored). All
    lines are solid. bound adds the bound (bounds.py): the cheapest reading of the same pool any
    ranking could achieve, with seeds alone only (none with windows)."""
    tail = f"+x{e}" if e else ""
    out = [("first_stage", f"first_stage@{limit}{tail}", line_color("first_stage"), "-")]
    out += [(m, f"rerank:{m}@{limit}{tail}", line_color(m), "-") for m in r.models_at(c)]
    if e:
        out += [(f"{m}~windows", f"xrerank:{m}@{limit}{tail}", line_color(f"{m}~windows"), "-") for m in r.models_at(c)]
    if bound and not e:
        out.append(("bound", f"bound@{limit}", ORACLE, "-"))
    return [x for x in out if not r.rows(c, x[1]).empty]


def _name_at(key: str, limit: str, e: int) -> str:
    """The ranker name of a _series legend key at another vector search limit."""
    base, _, what = key.partition("~")
    tail = f"+x{e}" if e else ""
    if base in ("first_stage", "bound"):
        return f"{base}@{limit}{tail}"
    return f"{'xrerank' if what else 'rerank'}:{base}@{limit}{tail}"


def _priced(m: str) -> str:
    """A model's label for charts that put a price on its tokens: a local model has none of its
    own, so it is priced as its stand-in on DeepInfra, and says so."""
    spec = RERANKERS[m]
    if spec.usd_per_mtok is None:
        return f"{spec.label}, priced as {RERANKERS[spec.priced_as].label} on DeepInfra (${price(m):g}/M)"
    return f"{spec.label} (DeepInfra ${price(m):g}/M)"


def line_color(key: str) -> str:
    """A _series key's line color: the first stage's blue, the bound's gray, and each model's
    hue family (palette.family): its seeds the darker shade, its windows the lighter."""
    base, _, what = key.partition("~")
    if base == "first_stage":
        return SERIES["first_stage"]
    if base == "bound":
        return ORACLE
    return palette.family(SERIES[base], 2)[1 if what == "windows" else 0]




def _ranker_groups(r: Results, keys: set[str]) -> list[tuple[str, list, list[str]]]:
    """Legend columns, one per color family: the first stage; each model, its seeds and its
    windows as two shades; the bound. A swatch per line, no marker key."""
    bases = {k.split("~")[0] for k in keys}
    groups = []
    if "first_stage" in bases:
        groups.append(("first stage", [_key(line_color("first_stage"))], ["vector search order"]))
    for m in [m for m in r.models if m in bases]:
        hs, ls = [_key(line_color(m))], ["rerank the seeds (then read them or their windows)"]
        if f"{m}~windows" in keys:
            hs.append(_key(line_color(f"{m}~windows")))
            ls.append("expand, then rerank the windows")
        groups.append((LABEL[m], hs, ls))
    if "bound" in bases:
        groups.append(("bound", [_key(ORACLE)], [_bound_label(r)]))
    return groups




def _bound_label(r: Results) -> str:
    """The bound's legend label, with the share of questions it is exact for."""
    rows = r.df[(r.df.kind == "bound") & (r.df.expand == 0)]  # simulations before 2026-11-09 have window floors
    if rows.empty:
        return "bound"
    return f"bound over the same pool, seeds alone (exact for {rows.bound_exact.astype(bool).mean():.0%} of questions)"


def _limit_grid(r: Results) -> list[str]:
    """Vector search limits for panel columns: those shown for any chunking."""
    return sorted({x for c in r.chunkings for x in r.limits(c, config.EVAL.limits)}, key=lambda d: int(d[1:]))


def coverage_by_tokens(r: Results, out, head: str) -> None:
    limits, expansions = _limit_grid(r), _expansions(r)
    for c in r.chunkings:
        for reading in rd.available(r.df, config.READINGS):
            fig, axes = _figure(len(expansions), len(limits), panel=(4.0, 3.0), sharex=True, sharey=True)
            keys = set()
            for i, e in enumerate(expansions):
                for j, limit in enumerate(limits):
                    ax = axes[i][j]
                    if limit not in r.limits(c, config.EVAL.limits, ("rerank",) if not e else ("rerank", "xrerank")):
                        ax.set_visible(False)
                        continue
                    for key, name, color, ls in _series(r, c, limit, e, bound=True):
                        _steps(ax, rd.costs(r.rows(c, name), reading), 0, config.EVAL.token_top, color=color, linestyle=ls)
                        keys.add(key)
                    _token_x(ax)
                    _share_y(ax)
                    ax.set_title(f"{c} · L {limit[1:]} · {_unit_title(r, c, e)}", loc="left")
                    if i == len(expansions) - 1:
                        ax.set_xlabel("reader tokens (rendered context)\nexact at every budget")
                    if j == 0:
                        ax.set_ylabel(("mean over questions: " if rd.is_share(reading) else "share of questions: ")
                                      + EVENT_TEXT[reading])
            _save(fig, out / (f"coverage_tokens_{reading}" + (f"_{c}" if len(r.chunkings) > 1 else "") + ".png"),
                  f"Coverage ({EVENT_TEXT[reading]}) vs reader tokens, {_sized(r, c)} — {head}", groups=_ranker_groups(r, keys),
                  note=[
                "Does reranking read more of the evidence within the same reader budget?",
                "Same for both: reader tokens (x), the vector search limit and expand_context (one pair per panel).",
                "Differs: the order the seeds are read in; reranking also pays for reranker tokens.",
                "Blue: the first stage. The model's hue: reranked (lighter: it scored the windows). Gray: the best any "
                "order of the same seeds could do (seeds alone).",
                f"Reading: {EVENT_TEXT[reading]}. Details: report section 3."])


def recall_precision_f1(r: Results, out, head: str) -> None:
    for c in r.chunkings:
        for e in _expansions(r):
            limits = r.limits(c, config.EVAL.limits, ("rerank",) if not e else ("rerank", "xrerank"))
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
                        for i, values in enumerate((rec, pre, f1)):
                            _plot(axes[i][j], xs, values.mean(0), color=color, linestyle=ls, linewidth=LINE)
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
                name = f"rpf_{axis}" + (f"_x{e}" if e else "")
                _save(fig, out / _file(name, None, c, r), f"Gold-character recall, precision and F1 vs {xlabel}, {_sized(r, c)} — {head}",
                      groups=_ranker_groups(r, keys), note=[
                    "Recall, precision and F1 of the evidence's characters read, with and without reranking.",
                    "Same for both: " + ("reader tokens" if axis == "tokens" else "segments read") + " (x), the vector "
                    "search limit and expand_context (panel).",
                    "Blue: the first stage. The model's hue: reranked (lighter: it scored the windows).",
                    "Details: METHODOLOGY section 2."])


def coverage_counts(r: Results, out, head: str, reading: str) -> None:
    """Coverage against segments read (EventMemory's max_num_segments: window neighbours count)
    at one vector search limit L (EventMemory's default, 20, where scored), a panel per
    expand_context: the count view the top-k chart (topk_recall: seeds kept k, and L) does not
    give. Each axis ends at L x (1 + E), the most segments L seeds' windows hold."""
    if f"{reading}_k" not in r.df:
        return
    for c in r.chunkings:
        limits = r.limits(c, config.EVAL.limits)
        if not limits:
            continue
        limit = "n20" if "n20" in limits else limits[-1]
        expansions = _expansions(r)
        fig, axes = _figure(1, len(expansions), panel=(3.4, 3.0), sharey=True)
        keys = set()
        for j, e in enumerate(expansions):
            ax = axes[0][j]
            top = int(limit[1:]) * (1 + e)
            for key, name, color, ls in _series(r, c, limit, e):
                _steps(ax, r.rows(c, name)[f"{reading}_k"].to_numpy(), 0, top, color=color, linestyle=ls)
                keys.add(key)
            _k_x(ax, top)
            ax.set_xlabel("segments read (m = max_num_segments)\nexact at every count")
            _share_y(ax)
            ax.set_title(f"{c} · L {limit[1:]} · {_unit_title(r, c, e)}", loc="left")
            if j == 0:
                ax.set_ylabel(_coverage(reading))
        _save(fig, out / _file("coverage_counts", reading, c, r),
              f"Coverage vs segments read (max_num_segments), {_sized(r, c)} — {head}",
              groups=_ranker_groups(r, keys),
              note=[
                  "How much evidence is read for a given number of segments (EventMemory's max_num_segments)?",
                  f"Same for both: segments read (x), the vector search limit ({limit[1:]}) and expand_context (panel).",
                  "Each axis ends where the windows of the L seeds run out: L x (1 + expand_context).",
                  "Blue: the first stage. The model's hue: reranked (lighter: it scored the windows).",
                  "Seeds kept (k) and the vector search limit: see the top-k chart."])


def _size_line(r: Results, chunkings) -> str:
    """One plain line of measured mean segment sizes, e.g. 'Mean segment: seg500 300 characters
    and 65 tokens, cut354 299 characters and 66 tokens.'"""
    from rbench.report.data import segment_size

    parts = []
    for c in chunkings:
        size = segment_size(r, c)
        parts.append(f"{c} {size[0]:,.0f} characters and {size[1]:,.0f} tokens" if size else f"{c} not measured")
    return "Mean segment: " + ", ".join(parts) + "."


def _families(chunkings: list[str]) -> dict[str, list[str]]:
    """Chunkings by chunker family (seg, cut), each sorted by size limit."""
    out: dict[str, list[str]] = {}
    for c in sorted(chunkings, key=lambda c: int(c[3:])):
        out.setdefault(c[:3], []).append(c)
    return out


def expansion_lines(r: Results, c: str, m: str, L: int) -> list[tuple[str, list[tuple[int, str]]]]:
    """The expansion comparison at one chunking, model and vector search limit: per approach
    (the first stage; rerank, then expand; expand, then rerank), its (expand_context, ranker)
    lines that were simulated. Expand, then rerank exists only where windows were scored; its
    expand_context 0 line is the seeds reranked (the same ranker either way)."""
    expands = [0, *r.expansions]
    tail = lambda e: f"+x{e}" if e else ""  # noqa: E731
    out = [("first stage", [(e, f"first_stage@n{L}{tail(e)}") for e in expands]),
           ("rerank, then expand", [(e, f"rerank:{m}@n{L}{tail(e)}") for e in expands]),
           ("expand, then rerank", [(0, f"rerank:{m}@n{L}")] + [(e, f"xrerank:{m}@n{L}+x{e}") for e in r.expansions])]
    out = [(a, [(e, n) for e, n in lines if not r.rows(c, n).empty]) for a, lines in out]
    return [(a, lines) for a, lines in out if len(lines) > 1]


def expansion_compare(r: Results, out, head: str) -> None:
    """Coverage vs reader tokens with one line per expand_context on the same axes: the first
    stage, rerank then expand, and expand then rerank (where windows were scored), each at the
    cost comparison's fixed vector search limits (frontier.cost_limits: EventMemory's default,
    the best L overall, the pool). Reader tokens on x, so a window's extra text is paid for."""
    if not r.models:
        return
    m = r.models[0]
    for c in r.chunkings:
        if not ((r.df.chunking == c) & (r.df.expand > 0)).any():
            continue
        for reading in rd.available(r.df, config.EVAL.readings):
            limits = frontier.cost_limits(r, m, reading)
            grid = [(L, what, expansion_lines(r, c, m, L)) for L, what in limits]
            grid = [(L, what, lines) for L, what, lines in grid if lines]
            if not grid:
                continue
            columns = max(len(lines) for _, _, lines in grid)
            fig, axes = _figure(len(grid), columns, panel=(4.0, 3.0), sharey=True)
            for i, (L, what, approaches) in enumerate(grid):
                for j in range(columns):
                    ax = axes[i][j]
                    if j >= len(approaches):
                        ax.set_visible(False)
                        continue
                    approach, lines = approaches[j]
                    base = SERIES["first_stage"] if approach == "first stage" else SERIES[m]
                    shades = palette.family(base, len(lines))  # darkest: expand_context 0
                    ends = []
                    for (e, name), color in zip(lines, shades, strict=True):
                        costs = rd.costs(r.rows(c, name), reading)
                        _steps(ax, costs, 0, config.EVAL.token_top, color=color)
                        ends.append((float(rd.coverage(costs, [config.EVAL.token_top])[0]), f"x{e}"))
                    _end_labels(ax, ends)
                    _token_x(ax)
                    _share_y(ax)
                    ax.set_title(f"{c} · L {L} ({what}) · {approach}", loc="left")
                    if i == len(grid) - 1:
                        ax.set_xlabel("reader tokens (rendered context)\nexact at every budget")
                    if j == 0:
                        ax.set_ylabel(_coverage(reading))
            _save(fig, out / _file("expansion_compare", reading, c, r),
                  f"Coverage by expand_context, compared directly ({EVENT_TEXT[reading]}), {_sized(r, c)} — {head}",
                  [_key(palette.family(SERIES["first_stage"], 2)[0]), _key(palette.family(SERIES[m], 2)[0])],
                  ["first stage (blue)", f"{LABEL[m]} (its hue)"], ncol=2,
                  note=[
                      "Which expand_context reads the evidence most cheaply?",
                      "One line per expand_context, labelled at its right end; darker means smaller.",
                      "Same for every line in a panel: the vector search limit (row) and how the seeds are ordered "
                      "(column).",
                      "Reader tokens on x count the windows' extra text too.",
                      "Details: report section 12."])


FACTOR_TITLES = {"L": "vector search limit L (seeds reranked)", "k": "seeds kept k", "m": "segments read m (max_num_segments)",
                 "segment": "segment length", "chunker": "chunker", "reader sees": "what the reader sees",
                 "reranker sees": "what the reranker scores (expand_context 3)"}
NUMERIC = ("L", "k", "m")


def not_run(r: Results) -> list[str]:
    """Factors and comparisons this run's data cannot show (each needs another stage): stated in
    the decision chart and the summary instead of drawn empty."""
    core = frontier.setting(r)[0]
    out = []
    if sum(c.startswith("seg") for c in r.chunkings) < 2:
        out.append("segment length (stage lme-sizes)")
    if config.cut_for(r.dataset, core) not in r.chunkings:
        out.append("chunker (stage lme-cut)")
    if r.models and not any(r.depths_at(core, m, "xrerank") for m in r.models):
        out.append("what the reranker scores: the windows were not scored")
    elif r.models and any(e != 3 for e in r.expansions):
        out.append("expand, then rerank at other expand_context values than 3")
    if len(r.models) < 2:
        out.append("model choice: one reranker scored; add one with --models or stage lme-small")
    return out


def best_settings(r: Results, out, head: str, reading: str) -> None:
    """The decision chart: one panel per factor that has two or more values (best.py: each
    varied alone, the others at the core setting), coverage against its values at each
    configured budget, reranked and the first stage under the same value, with 95% intervals;
    the best value and the cheapest within the best's bound marked, as best.choose picks them,
    once over all questions."""
    from rbench.report import best

    if not r.models:
        return
    budgets = config.EVAL.budgets
    clusters = frontier._clusters(r)
    for m in r.models:
        per = {b: {f: best.choose(r, f, v) for f, v in best.factors(r, m, reading, b).items() if len(v) >= 2}
               for b in budgets}
        factors = [f for f in FACTOR_TITLES if any(f in per[b] for b in budgets)]
        if not factors:
            continue
        cols = min(3, len(factors))
        rows = -(-len(factors) // cols)
        fig, axes = _figure(rows, cols, panel=(4.2, 3.2), sharey=True)
        reranked, first = _shades(SERIES[m], len(budgets)), _shades(SERIES["first_stage"], len(budgets))
        for i, f in enumerate(factors):
            ax = axes[i // cols][i % cols]
            for j, b in enumerate(budgets):
                ch = per[b].get(f)
                if ch is None:
                    continue
                labels = [v.label.split(" ", 1)[1] if f in NUMERIC else v.label for v in ch.values]
                xs = np.asarray([float(x) for x in labels]) if f in NUMERIC else np.arange(len(labels), dtype=float)
                ys = np.asarray([v.reranked.mean() for v in ch.values])
                ci = np.asarray([interval(v.reranked, clusters) for v in ch.values])
                if np.isfinite(ci).all():
                    ax.errorbar(xs, ys, yerr=[ys - ci[:, 0], ci[:, 1] - ys], fmt="none", color=reranked[j],
                                linewidth=LINE, alpha=ALPHA, capsize=0)
                _plot(ax, xs, ys, color=reranked[j], marker="o")
                fy = np.asarray([v.first.mean() if v.first is not None else np.nan for v in ch.values])
                _plot(ax, xs, fy, color=first[j], marker="o")
                for idx, marker, size in ((ch.best, "*", 13), (ch.cheapest, "D", 8)):
                    ax.plot([xs[idx]], [ys[idx]], linestyle="none", alpha=ALPHA,
                            **_hollow({"marker": marker, "color": reranked[j], "markersize": size}))
            if f in NUMERIC:
                _log_x(ax, 10, _round_ticks([int(float(x)) for x in labels]))
            else:
                ax.set_xticks(xs, labels, fontsize=7)
                ax.set_xlim(-0.5, len(labels) - 0.5)
            _share_y(ax)
            ax.set_title(FACTOR_TITLES[f], loc="left")
            if i % cols == 0:
                ax.set_ylabel(_coverage(reading))
        for i in range(len(factors), rows * cols):
            axes[i // cols][i % cols].set_visible(False)
        handles = ([_key(c, marker="o") for c in reranked] + [_key(c, marker="o") for c in first]
                   + [_key(INK2, ls="none", marker="*"), _key(INK2, ls="none", marker="D")])
        labels = ([f"{LABEL[m]}, reader budget {b:,}" for b in budgets] + [f"first stage, reader budget {b:,}" for b in budgets]
                  + ["best value (highest reranked coverage)", "cheapest within the best's 95% bound"])
        missing = not_run(r)
        _save(fig, out / _file(f"best_settings_{m}", reading), f"Best settings, one factor at a time: {LABEL[m]} — {head}",
              handles, labels, ncol=3,
              note=[
                  "Which value of each setting works best?",
                  "Each panel varies one setting; the others stay at the core setting.",
                  "Lines: coverage at each reader budget (lighter: smaller budget), reranked in the model's hue and "
                  "without reranking in blue, with 95% intervals.",
                  "Star: the best value. Diamond: the cheapest value within the best's 95% interval.",
                  ("Not run in this stage: " + "; ".join(missing) + "." if missing else ""),
                  "Details: report sections 14 and 15."])


def chunker_compare(r: Results, out, head: str) -> None:
    """EventMemory's TextSegmenter against the size-matched simple chunker on the same axes:
    coverage vs reader tokens, the first stage and the reranker at the same L (EventMemory's
    default, the best L overall, the pool), for every-gold and the touched and share readings
    beside it (in full penalizes a chunker that splits evidence turns). Only where both are
    simulated."""
    if not r.models:
        return
    m = r.models[0]
    for seg in [c for c in r.chunkings if c.startswith("seg")]:
        cut = config.cut_for(r.dataset, seg)
        if cut not in r.chunkings:
            continue
        readings = rd.available(r.df, ("every", "every_touch", "share", "share_touch"))
        limits = frontier.cost_limits(r, m, "every")
        fig, axes = _figure(len(limits), len(readings), panel=(3.6, 3.0), sharey=True)
        firsts, models = palette.family(SERIES["first_stage"], 2), palette.family(SERIES[m], 2)
        for i, (L, what) in enumerate(limits):
            for j, reading in enumerate(readings):
                ax = axes[i][j]
                for c, shade in ((seg, 0), (cut, 1)):
                    for name, colors in ((f"first_stage@n{L}", firsts), (f"rerank:{m}@n{L}", models)):
                        rows = r.rows(c, name)
                        if not rows.empty:
                            _steps(ax, rd.costs(rows, reading), 0, config.EVAL.token_top, color=colors[shade])
                _token_x(ax)
                _share_y(ax)
                ax.set_title(f"L {L} ({what}) · {EVENT_TEXT[reading]}", loc="left")
                if i == len(limits) - 1:
                    ax.set_xlabel("reader tokens (rendered context)\nexact at every budget")
                if j == 0:
                    ax.set_ylabel("coverage")
        sizes = _size_line(r, [seg, cut])
        _save(fig, out / f"chunker_compare_{seg[3:]}.png",
              f"EventMemory's TextSegmenter vs the size-matched simple chunker, {seg} vs {cut} — {head}",
              [_key(firsts[0]), _key(firsts[1]), _key(models[0]), _key(models[1])],
              [f"first stage, {seg}", f"first stage, {cut}", f"{LABEL[m]}, {seg}", f"{LABEL[m]}, {cut}"], ncol=4,
              note=[
                  "EventMemory's TextSegmenter against the simple chunker of the same mean segment size.",
                  "Same for all lines in a panel: the vector search limit (row) and the reading (column).",
                  "The simple chunker splits more evidence turns, and readings in full need every piece: compare the "
                  "columns that count a turn once any piece is read.",
                  sizes,
                  "Details: report section 8."])


def _shades(base: str, n: int) -> list[str]:
    """n validated shades of a ranking's hue (palette.family), lightest first."""
    return palette.family(base, n)[::-1]


CHUNKER_POINT = {"seg": "o", "cut": "s"}  # marker on measured points, per chunker family (hollow, in the line's color)




def rerank_cost(r: Results, out, head: str, reading: str) -> None:
    """Coverage at each configured reader budget against reranker tokens, as the vector search
    limit grows (one marker per limit), with the first stage at the same limits drawn at the
    same x: what the same seeds give without a reranker. Color is the ranking (the model's
    color reranked, blue the first stage), lightness the chunking, labelled at the line ends."""
    if not r.models:
        return
    chunkings = sorted([c for c in r.chunkings if c[:3] in CHUNKERS], key=lambda c: (unit_tokens(r, c) or 0, c))
    rank = {c: i for i, c in enumerate(chunkings)}  # lighter: smaller measured segments
    budgets = config.EVAL.budgets
    dollars = len({price(m) for m in r.models}) > 1  # models priced differently: compare them in dollars
    fig, axes = _figure(len(r.models), len(budgets), panel=(3.8, 3.2), sharey=True)
    for i, m in enumerate(r.models):
        scale = price(m) / 1e6 * 1000 if dollars else 1.0  # $ per 1,000 questions, or tokens per question
        for j, b in enumerate(budgets):
            ax = axes[i][j]
            ends = []
            for c in chunkings:
                pts = []
                for d in r.seed_depths(c, m):
                    rows, base = r.rows(c, f"rerank:{m}@{d}"), r.rows(c, f"first_stage@{d}")
                    if not rows.empty and not base.empty:
                        pts.append((rows.rr_tokens.mean() * scale, rd.hit(rd.costs(rows, reading), b).mean(),
                                    rd.hit(rd.costs(base, reading), b).mean()))
                if not pts:
                    continue
                xs, ys, bs = zip(*sorted(pts), strict=True)
                _plot(ax, xs, ys, color=_shades(SERIES[m], len(chunkings))[rank[c]], marker="o")
                _plot(ax, xs, bs, color=_shades(SERIES["first_stage"], len(chunkings))[rank[c]], marker="o")
                if len(chunkings) > 1:
                    ends += [(ys[-1], c), (bs[-1], c)]
            _end_labels(ax, ends)
            _log_x(ax)
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            _share_y(ax)
            ax.set_title(f"{LABEL[m]} · reader budget {b:,}", loc="left")
            ax.set_xlabel("reranker $ per 1,000 questions (DeepInfra prices)" if dollars
                          else "reranker tokens per query (API billing)")
            if j == 0:
                ax.set_ylabel(_coverage(reading))
    handles = [_key(SERIES[m], marker="o") for m in r.models] + [_key(SERIES["first_stage"], marker="o")]
    labels = [f"{LABEL[m]}, reranking the seeds" for m in r.models] + [
        "first stage at the same vector search limits (no reranker tokens)"]
    sizes = _size_line(r, chunkings)
    _save(fig, out / _file("rerank_cost", reading), f"Coverage vs reranker tokens as the vector search limit grows — {head}",
          handles, labels, ncol=2,
          note=[
              "What does a deeper vector search buy, against the reranker's cost?",
              "One marker per vector search limit: 5, 10, 20 and so on up to the pool.",
              "x: the reranker's " + ("cost" if dollars else "tokens") + " per question at that limit. The first "
              "stage spends none; it is drawn at the same x for comparison.",
              "Same for both: the reader budget (panel) and the seeds searched.",
              ("Lighter: smaller segments. " if len(chunkings) > 1 else "") + sizes,
              ])


def gain_sizes(r: Results, out, head: str, reading: str) -> None:
    """Reranker gain against measured segment size: coverage reranked minus coverage of the first
    stage over the same seeds (same vector search limit), at each configured budget, with 95%
    intervals; a panel per (model, limit), a line per (budget, chunker family). Drawn only when
    a chunker has two or more sizes (else it is no size comparison: rerank_cost and report
    section 8 compare chunkers)."""
    measured = {c: unit_tokens(r, c) for c in r.chunkings}
    # models scored at two or more measured chunkings (a model scored at one has no gain curve)
    models = [m for m in r.models if sum(1 for c in measured if measured[c] and m in r.models_at(c)) >= 2]
    sized = [c for c, t in measured.items() if t and any(m in r.models_at(c) for m in models)]
    families = _families(sized)
    if not models or max((len(v) for v in families.values()), default=0) < 2:
        return
    limits = [d for d in _limit_grid(r) if all(d in r.limits(c, config.EVAL.limits, ("rerank",)) for c in sized)]
    if not limits:
        return
    budgets = config.EVAL.budgets
    rank = {b: i for i, b in enumerate(budgets)}  # lighter: smaller budget
    fig, axes = _figure(len(models), len(limits), panel=(3.8, 3.2), sharey=True)
    for i, m in enumerate(models):
        for j, d in enumerate(limits):
            ax = axes[i][j]
            for b in budgets:
                for fam, members in families.items():
                    chunkings = [c for c in members if m in r.models_at(c) and not r.rows(c, f"rerank:{m}@{d}").empty]
                    if not chunkings:
                        continue
                    means, los, his = [], [], []
                    for c in chunkings:
                        rows, base = r.rows(c, f"rerank:{m}@{d}"), r.rows(c, f"first_stage@{d}")
                        gain = rd.hit(rd.costs(rows, reading), b) - rd.hit(rd.costs(base, reading), b)
                        lo, hi = interval(gain, base.cluster.to_numpy())
                        means.append(gain.mean())
                        los.append(gain.mean() - lo if np.isfinite(lo) else 0.0)
                        his.append(hi - gain.mean() if np.isfinite(hi) else 0.0)
                    xs = [measured[c] for c in chunkings]
                    color = _shades(SERIES[m], len(budgets))[rank[b]]
                    ax.errorbar(xs, means, yerr=[los, his], color=color, linewidth=LINE, capsize=0, fmt="none")
                    _plot(ax, xs, means, color=color, marker=CHUNKER_POINT[fam])
            ax.axhline(0, color=INK2, linewidth=LINE)
            _measured_x(ax, {c: measured[c] for c in sized})
            ax.set_title(f"{LABEL[m]} · L {d[1:]}", loc="left")
            ax.set_xlabel("measured mean segment of the retrieved seeds (tokens)")
            if j == 0:
                ax.set_ylabel(f"gain in {_coverage(reading)}")
    first = SERIES[models[0]]
    handles = [_key(_shades(first, len(budgets))[rank[b]]) for b in budgets]
    labels = [f"reader budget {b:,}" for b in budgets]
    if len(families) > 1:
        handles += [_key(INK2, marker=CHUNKER_POINT[f]) for f in families]
        labels += [CHUNKER_NAMES[f] for f in families]
    _save(fig, out / _file("gain_sizes", reading), f"Reranker gain vs measured segment size — {head}", handles, labels,
          ncol=4,
          note=[
              "Does reranking help more at some segment lengths?",
              "y: coverage reranked minus the first stage over the same seeds, with 95% intervals.",
              "Same for both: the vector search limit (panel) and the reader budget (lighter: smaller).",
              "x: each chunking's measured mean segment of the retrieved seeds (" + ", ".join(sized) + ")."])




def _few(r: Results) -> bool:
    """Too few question clusters for a percentile interval (stats.MIN_CLUSTERS): a pipeline test."""
    from rbench.report.stats import MIN_CLUSTERS

    return len(np.unique(frontier._clusters(r))) < MIN_CLUSTERS


def cost_equal(r: Results, out, head: str, reading: str) -> None:
    """The headline cost view: reranking against the first stage at the same money per question
    (frontier.equal_spend). One panel per vector search limit L on both sides (EventMemory's
    default, the best L overall, the pool); x the LLM input price, y the spend per question as
    its reader-token equivalent; color the coverage gained by reranking for that money, gray
    where its 95% interval includes 0; the line, where reranking's gain becomes clear (its lower
    bound crosses 0), is the break-even."""
    if not r.models:
        return
    fixed = frontier.setting(r)
    few = _few(r)
    panels = [(m, L, what, frontier.equal_spend(r, m, price(m), L, reading, fixed))
              for m in r.models for L, what in frontier.cost_limits(r, m, reading)]
    panels = [p for p in panels if p[3] is not None]
    if not panels:
        return
    models = [m for m in r.models if any(p[0] == m for p in panels)]
    columns = max(sum(p[0] == m for p in panels) for m in models)
    # each arm its own range, so small gains stay visible beside large losses at the cheap-price edge
    low = min(-0.05, float(np.nanmin([np.nanmin(p[3].gain) for p in panels])))
    high = max(0.05, float(np.nanmax([np.nanmax(p[3].gain) for p in panels])))
    norm = TwoSlopeNorm(0, low, high)
    fig, axes = _figure(len(models), columns, panel=(4.6, 3.8), sharey=True)
    lines = False
    for row, m in enumerate(models):
        mine = [p for p in panels if p[0] == m]
        for col in range(columns):
            ax = axes[row][col]
            if col >= len(mine):
                ax.set_visible(False)
                continue
            _, L, what, es = mine[col]
            clear = np.isfinite(es.gain) if few else (es.lo > 0) | (es.hi < 0)
            im = ax.pcolormesh(es.prices, es.spends, np.where(clear, es.gain, np.nan).T, cmap=DIVERGING, norm=norm,
                               shading="nearest")
            ax.pcolormesh(es.prices, es.spends, np.where(np.isfinite(es.gain) & ~clear, 1.0, np.nan).T, shading="nearest",
                          cmap=LinearSegmentedColormap.from_list("muted", [MUTED, MUTED]))
            edge = np.nan_to_num(es.gain if few else es.lo, nan=-1.0).T  # where the gain (its lower bound) crosses 0
            if (edge > 0).any() and (edge <= 0).any():
                ax.contour(es.prices, es.spends, edge, levels=[0.0], colors=[INK], linewidths=LINE)
                lines = True
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(es.prices[0], es.prices[-1])
            ax.set_ylim(es.spends[0], es.spends[-1])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.set_yticks([500, 1000, 2000, 4000, 8000, 16000], ["500", "1,000", "2,000", "4,000", "8,000", "16,000"])
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_facecolor(SURFACE)
            ax.grid(False)
            ax.set_title(f"{LABEL[m]} · L {L} on both sides ({what}; reranker {es.tokens:,.0f} tokens per question)",
                         loc="left")
            ax.set_xlabel("LLM input price ($/M)")
            if col == 0:
                ax.set_ylabel("spend per question, in reader tokens\n(money / LLM price)")
    fig.colorbar(im, ax=[a for row in axes for a in row],
                 label=f"coverage gained by reranking for the same money ({EVENT_TEXT[reading]})")
    legend = [(Patch(color=MUTED), "no clear difference (the gain's 95% interval includes 0)")] if not few else []
    if lines:
        legend.insert(0, (_key(INK), "break-even: reranking clearly gets more for the money inside the line" if not few
                          else "break-even (point estimate): reranking gets more inside the line"))
    name = "cost_equal" + ("" if fixed == (default_segment(), 0) else f"_{fixed[0]}_x{fixed[1]}")
    _save(fig, out / _file(name, reading),
          f"Coverage for the same money per question, reranked vs not ({fixed[0]}, expand_context {fixed[1]}) — {head}",
          [h for h, _ in legend], [lab for _, lab in legend], ncol=2,
          note=[
              "Does reranking get more evidence for the same money?",
              "Same for both: total spend per question, and the vector search limit (one per panel).",
              "Without reranking, the whole spend buys reader tokens; with reranking, part pays the reranker.",
              f"Color: extra coverage from reranking ({EVENT_TEXT[reading]}). Gray: no clear difference.",
              "Line: where reranking clearly gets more for the money (inside it).",
              ("Pipeline test: fewer than 20 questions, so no intervals and no gray." if few else ""),
              "Long sessions: if retrieved text is re-sent m times, read the price as price x m. Details: report "
              "section 7."])


def query_types(r: Results, out, head: str, reading: str) -> None:
    """Reranker gain by question type at each configured budget: reranked minus the first stage
    over the same seeds, at the best vector search limit chosen once overall (best.overall_limit:
    the L whose reranked coverage averaged over the realistic budgets is highest; the summary's
    best L), for the first model; the deepest pool would mix in deep-overfetch harm."""
    from rbench.report import best
    from rbench.report.stats import MIN_CLUSTERS

    if not r.models or r.df.qid.nunique() < 20:
        return
    c = default_segment() if default_segment() in r.chunkings else r.chunkings[0]
    chosen = best.overall_limit(r, r.models[0], c, reading)
    if chosen is None:
        return
    limit = f"n{chosen}"
    base, labels = query_labels(r, c, f"first_stage@{limit}")
    groups = [(f, v) for f in ("question_type", "form", "length", "context", "parts") if f in labels
              for v in sorted(set(labels[f])) if (labels[f] == v).sum() >= 10]
    if not groups:
        return
    models = [m for m in r.models if not r.rows(c, f"rerank:{m}@{limit}").empty]  # those scored this deep
    if not models:
        return
    budgets = config.EVAL.budgets
    fig, axes = _figure(len(models), len(budgets), panel=(3.4, 0.34 * len(groups) + 1.2), sharey=True)
    for (row, m), (j, b) in itertools.product(enumerate(models), enumerate(budgets)):  # a row per model: no stagger
        ax = axes[row][j]
        hit = rd.hit(rd.costs(base, reading), b)
        gain = rd.hit(rd.costs(r.rows(c, f"rerank:{m}@{limit}"), reading), b) - hit
        for g, (f, v) in enumerate(groups):
            mask = labels[f] == v
            lo, hi = interval(gain[mask], base.cluster.to_numpy()[mask])
            if not np.isnan(lo):
                ax.plot([lo, hi], [g, g], color=SERIES[m], linewidth=LINE, alpha=ALPHA)
            ax.plot([gain[mask].mean()], [g], linestyle="none", alpha=ALPHA,
                    **_hollow({"marker": "o", "color": SERIES[m], "markersize": 6}))
        ax.axvline(0, color=INK2, linewidth=LINE)
        few = lambda f, v: len(np.unique(base.cluster.to_numpy()[labels[f] == v])) < MIN_CLUSTERS  # noqa: E731
        ax.set_yticks(range(len(groups)), [f"{f}: {v} (n={int((labels[f] == v).sum())}"
                                           + ("; no interval" if few(f, v) else "") + ")" for f, v in groups])
        ax.invert_yaxis()
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.set_title(f"{LABEL[m]} · reader budget {b:,}" if len(models) > 1 else f"reader budget {b:,}", loc="left")
        ax.set_xlabel("gain in coverage")
    _save(fig, out / _file("query_types", reading),
          f"Reranker gain by question type ({c}, L {limit[1:]}: the best L overall) — {head}",
          [_key(SERIES[m], marker="o") for m in models], [LABEL[m] for m in models],
          note=[
              "Which kinds of question does reranking help?",
              f"Gain: coverage reranked minus the first stage over the same {limit[1:]} seeds (the best limit overall).",
              "Dot: the mean. Line: its 95% interval. No interval for groups of fewer than 20 questions.",
              "Same for both: the reader budget (panel). Details: report section 9."])


def first_stage_sizes(r: Results, out, head: str, reading: str) -> None:
    """First-stage coverage against each chunking's measured size, a line per chunker family
    (the first stage's blue, lightness and marker the family). Drawn only when a chunker has
    two or more sizes."""
    measured = {c: unit_tokens(r, c) for c in r.chunkings if c[:3] in CHUNKERS}
    families = _families([c for c, t in measured.items() if t])
    if max((len(v) for v in families.values()), default=0) < 2:
        return
    shade = dict(zip(families, _shades(SERIES["first_stage"], len(families)), strict=True))
    budgets = config.EVAL.budgets
    fig, axes = _figure(1, len(budgets), panel=(3.4, 3.2), sharey=True)
    for j, b in enumerate(budgets):
        ax = axes[0][j]
        for fam, chunkings in families.items():
            ys = [rd.hit(rd.costs(r.rows(c, "first_stage"), reading), b).mean() for c in chunkings]
            _plot(ax, [measured[c] for c in chunkings], ys, color=shade[fam], marker=CHUNKER_POINT[fam])
        _measured_x(ax, {c: measured[c] for cs in families.values() for c in cs})
        _share_y(ax)
        ax.set_title(f"reader budget {b:,}", loc="left")
        ax.set_xlabel("measured mean segment / chunk (tokens)")
        if j == 0:
            ax.set_ylabel(f"{_coverage(reading)}, no reranker")
    _save(fig, out / _file("first_stage_sizes", reading), f"First-stage coverage vs measured segment size (no reranker) — {head}",
          [_key(shade[f], marker=CHUNKER_POINT[f]) for f in families],
          [CHUNKER_NAMES[f] for f in families],
          note=[
                    "How does the segment size change retrieval without a reranker?",
                    "Same for every point: the reader budget (panel).",
                    "x: each chunking's measured mean segment of the retrieved seeds.",
                    "Blue shades: the chunker (lighter: the simple chunker)."])


EXPANSION_APPROACHES = (("plain chunks of the read size", None), ("seeds, expanded (no reranker)", None),
                        ("rerank seeds, then expand", None), ("expand, then rerank windows", None))
ORDER_APPROACHES = [label for label, _ in EXPANSION_APPROACHES]


def _approach_color(label: str, m: str) -> str:
    """An expansion approach's color in model m's panel: the first stage's blue, and three shades
    of the model's hue for its reranked approaches (palette.family)."""
    shades = palette.family(SERIES[m], 3)
    return {"rerank seeds, then expand": shades[0], "expand, then rerank windows": shades[1],
            "plain chunks of the read size": shades[2], "seeds, expanded (no reranker)": SERIES["first_stage"]}[label]


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

    def sizes(self, flag: bool = True) -> str:
        window = f"window {self.window:,.0f} tok" if self.window else "window not measured"
        if self.plain is None:
            return f"{window}; no unexpanded {self.seed[:3]} chunking to compare"
        plain = f"plain {self.plain} {self.plain_size:,.0f} tok"
        return f"{window}; {plain}" + ("" if self.matched or not flag else f" — NOT size-matched ({self.ratio:.2f}×)")


def expansion_rows(r: Results) -> list[ExpansionPanel]:
    """Per (model, expand_context): the four approaches as (label, chunking, ranker) when
    present. Seeds are EventMemory's default segments when expanded, else the smallest chunking
    with expansions; the plain comparison is the unexpanded chunking of the same chunker whose
    measured mean unit (data.unit_tokens) is closest to the seeds' measured mean window (another
    chunker would change the boundaries as well as the size)."""
    expanded = sorted({c for c in r.df.loc[r.df.expand > 0, "chunking"]}, key=lambda c: int(c[3:]))
    if not expanded:
        return []
    seed = default_segment() if default_segment() in expanded else expanded[0]
    sized = [(c, unit_tokens(r, c)) for c in r.chunkings if c != seed and c[:3] == seed[:3]]
    sized = [(c, t) for c, t in sized if t]
    out = []
    for e in r.expansions:
        if not ((r.df.chunking == seed) & (r.df.expand == e)).any():
            continue
        window = unit_tokens(r, seed, e)
        plain, plain_size = (min(sized, key=lambda ct: abs(math.log(ct[1] / window))) if sized and window
                             else (None, None))
        for m in r.models:
            depth, windows = r.pool_depth(seed, m, "rerank"), r.pool_depth(seed, m, "xrerank")  # each at its own pool
            items = [(EXPANSION_APPROACHES[1][0], seed, f"first_stage@{depth}+x{e}"),
                     (EXPANSION_APPROACHES[2][0], seed, f"rerank:{m}@{depth}+x{e}"),
                     (EXPANSION_APPROACHES[3][0], seed, f"xrerank:{m}@{windows}+x{e}")]
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
    present = set()
    for p in panels:
        i, j = expands.index(p.expand), models.index(p.model)
        ax = axes[i][j]
        for label, c, name in sorted(p.items, key=lambda item: ORDER_APPROACHES.index(item[0])):
            _steps(ax, rd.costs(r.rows(c, name), reading), 0, config.EVAL.token_top, color=_approach_color(label, p.model))
            present.add(label)
        _token_x(ax)
        _share_y(ax)
        ax.set_title(f"{LABEL[p.model]} · expand_context {p.expand}\n{p.sizes()}", loc="left")
        ax.set_xlabel("reader tokens: exact steps, one per question")
        if j == 0:
            ax.set_ylabel(_coverage(reading))
    for i, e in enumerate(expands):  # panels without a comparison stay empty
        for j, m in enumerate(models):
            if not any(p.expand == e and p.model == m for p in panels):
                axes[i][j].set_visible(False)
    keys = [k for k, _ in EXPANSION_APPROACHES if k in present]
    unmatched = any(not p.matched for p in panels if p.plain is not None)
    one = models[0] if len(models) == 1 else None  # several models: model-colored keys in ink, each panel in its model's color
    _save(fig, out / _file("expansion", reading), f"Embedded vs reranked vs read unit — {head}",
          [_key(_approach_color(k, one or models[0]) if one or k.startswith("seeds,") else INK2) for k in keys], keys,
          ncol=2,
          note=[
              "Read the seeds' windows, or plain chunks of the same size?",
              "Same for all: reader tokens (x).",
              "Differs: what is embedded, what the reranker scores and what is read.",
              "Blue: the first stage. Shades of the model's hue: its reranked approaches.",
              ("Sizes are measured; panels flag pairs more than " + f"{config.EVAL.size_tolerance:.0%}" + " apart."
               if unmatched else "Sizes are measured: mean tokens per retrieved seed's window."),
              "Details: report section 11."])


ROUND = (1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000)


def _round_ticks(values, keep_last: bool = True) -> list[int]:
    """Labelled ticks for a log axis of measured values: the round ones, and the last (a
    pool), dropping round ones too close to it to label apart."""
    values = sorted(values)
    if not values:
        return []
    last = values[-1]
    out = {v for v in values if v in ROUND and (not keep_last or last / v >= 1.3 or v == last)}
    return sorted(out | ({last} if keep_last else set()))


def _log_edges(values) -> np.ndarray:
    """Cell edges for a heatmap of measured values on a log axis: geometric midpoints, the
    outer edges as far out as the neighbouring step."""
    v = np.asarray(values, dtype=float)
    if len(v) == 1:
        return np.asarray([v[0] / 1.5, v[0] * 1.5])
    mids = np.sqrt(v[:-1] * v[1:])
    return np.concatenate([[v[0] ** 2 / mids[0]], mids, [v[-1] ** 2 / mids[-1]]])


def _end_labels(ax, items, gap: float = 0.045) -> None:
    """Text labels at the right end of lines, (y, text) each, in the text ink (the line's color
    carries what it is). Labels too close to read apart are spread evenly around their lines'
    mean, so each stays as near its line as the others allow."""
    items = sorted(items)
    groups = [[i] for i in range(len(items))]
    while True:  # merge neighbouring groups until no two overlap
        spans = [(np.mean([items[i][0] for i in g]), g) for g in groups]
        placed = [(c - gap * (len(g) - 1) / 2, c + gap * (len(g) - 1) / 2, g) for c, g in spans]
        clash = next((n for n in range(len(placed) - 1) if placed[n + 1][0] - placed[n][1] < gap), None)
        if clash is None:
            break
        groups[clash:clash + 2] = [groups[clash] + groups[clash + 1]]
    for low, _, g in placed:
        for n, i in enumerate(g):
            ax.annotate(items[i][1], xy=(1.0, low + gap * n), xycoords=("axes fraction", "data"), xytext=(3, 0),
                        textcoords="offset points", va="center", ha="left", fontsize=7, color=INK2,
                        annotation_clip=False)


def topk_recall(r: Results, out, head: str) -> None:
    """Recall with and without reranking across top-k, the asker's metric (report/topk.py):
    per model, a column per recall reading; the reranker's gain over (L, k), recall vs k at
    fixed L, and recall vs L at fixed k. Color is the ranking (the first stage's blue, the
    model's color reranked) and lightness the setting within it; k and L on log axes."""
    from rbench.report import topk

    if not r.models or not topk.available(r):
        return
    for m in r.models:
        grids = {x: topk.grid(r, m, x) for x in topk.READINGS}
        if not any(g["limits"] for g in grids.values()):
            continue
        first, model = SERIES["first_stage"], SERIES[m]
        point = "o"  # lines here join measured points (k and L values): a marker on each, in the line's color
        fig, axes = _figure(3, len(grids), panel=(4.4, 3.4))
        for j, (reading, g) in enumerate(grids.items()):
            limits, ks = g["limits"], g["ks"]
            ax = axes[0][j]
            # a pipeline test (too few questions) has no interval: every cell colored by its point estimate
            clear = (np.isfinite(g["gain"]) if _few(r) else
                     np.isfinite(g["low"]) & ~((g["low"] <= 0) & (g["high"] >= 0)))
            xe, ye = _log_edges(ks), _log_edges(limits)
            im = ax.pcolormesh(xe, ye, np.where(clear, g["gain"], np.nan), cmap=DIVERGING, norm=TwoSlopeNorm(0, -0.3, 0.3))
            ax.pcolormesh(xe, ye, np.where(np.isfinite(g["gain"]) & ~clear, 1.0, np.nan),
                          cmap=LinearSegmentedColormap.from_list("muted", [MUTED, MUTED]))
            ax.set_xscale("log")
            ax.set_yscale("log")
            for L in limits:  # k = L: everything retrieved is kept, a gain of 0 by construction
                if L in ks:
                    ax.text(L, L, "=", ha="center", va="center", fontsize=8, color=INK2)
            ax.set_xticks(_round_ticks(ks, keep_last=False), [str(k) for k in _round_ticks(ks, keep_last=False)], fontsize=7)
            ax.set_yticks(_round_ticks(limits), [str(L) for L in _round_ticks(limits)], fontsize=7)
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.tick_params(which="minor", length=0)
            ax.set_xlim(xe[0], xe[-1])
            ax.set_ylim(ye[0], ye[-1])
            ax.grid(False)
            ax.set_title(f"{topk.NAMES[reading]}: gain", loc="left")
            ax.set_xlabel("k seeds kept (log)")
            if j == 0:
                ax.set_ylabel("L, vector search limit (log)")
            ax = axes[1][j]
            bests = topk.best_limit(g)
            shades = palette.family(model, 3)  # darkest: best L per k; then L 20; lightest: the pool
            series = [("first stage (top-k as L)", g["first"], first, point)]
            if 20 in limits:
                series.append(("reranked, L 20 (EventMemory's default)", g["reranked"][limits.index(20)], shades[1], point))
            series.append((f"reranked, L {limits[-1]} (the pool)", g["reranked"][-1], shades[2], point))
            series.append(("reranked, best L per k", np.asarray([g["reranked"][limits.index(b), jj] if b else np.nan
                                                                for jj, b in enumerate(bests)]), shades[0], point))
            for _, ys, color, marker in series:
                _plot(ax, ks, ys, color=color, marker=marker)
            _log_x(ax, 10, _round_ticks(ks, keep_last=False))
            _share_y(ax)
            ax.set_title(f"{topk.NAMES[reading]} vs k", loc="left")
            ax.set_xlabel("k seeds kept (log)")
            if j == 0:
                ax.set_ylabel("recall")
            ax = axes[2][j]
            ends = []
            reranked_shades = palette.family(model, len(topk.SLICE_KS))[::-1]  # lighter: fewer kept
            first_shades = palette.family(first, len(topk.SLICE_KS))[::-1]
            for k, shade_r, shade_f in zip(topk.SLICE_KS, reranked_shades, first_shades, strict=True):
                if k not in ks:
                    continue
                col = ks.index(k)
                ys = g["reranked"][:, col]
                _plot(ax, limits, ys, color=shade_r, marker=point)
                # the first stage keeps k of its top L seeds: only where L >= k (else it keeps all L)
                _plot(ax, limits, np.where(np.asarray(limits) >= k, g["first"][col], np.nan), color=shade_f, marker=point)
                last = ys[np.isfinite(ys)]
                ends += [(float(last[-1]), f"k {k}")] if len(last) else []
                ends += [(float(g["first"][col]), f"k {k}")]
            _log_x(ax, 10, _round_ticks(limits))
            _share_y(ax)
            _end_labels(ax, ends)
            ax.set_title(f"{topk.NAMES[reading]} vs L", loc="left")
            ax.set_xlabel("L, vector search limit (log)")
            if j == 0:
                ax.set_ylabel("recall")
        fig.colorbar(im, ax=axes[0].tolist(), label="recall gain (reranked minus first stage)")
        handles = [_key(c, marker=mk) for _, _, c, mk in series] + [Patch(color=MUTED)]
        labels = [lab for lab, _, _, _ in series] + ["no clear difference (95% interval includes 0)"]
        _save(fig, out / f"topk_recall_{m}.png",
              f"Recall with and without reranking across top-k (the asker's metric), {LABEL[m]} — {head}",
              handles, labels, ncol=3,
              note=[
              "Recall with and without reranking across top-k: the asker's metric.",
              "Top-k two ways: L, the vector search limit (every seed reranked), and k, the seeds kept after "
              "reranking.",
              "Top row: the gain at each L and k, colored only where it is clear; '=' where k = L and nothing changes.",
              "Middle: recall vs k. Bottom: recall vs L at k = 5, 10 and 20, labelled at the right.",
              "Blue: the first stage. The model's hue: reranked; lighter shades are other L or k values."
              + (" Pipeline test: no intervals." if _few(r) else ""),
              "Details: report section 16."])


ALL_READINGS = ("coverage", "rpf", "topk", "expansion_compare", "chunker_compare")  # every reading at once
CHARTS = {"coverage": "coverage_by_tokens", "rpf": "recall_precision_f1", "counts": "coverage_counts",
          "rerank_cost": "rerank_cost", "gain_sizes": "gain_sizes", "cost_equal": "cost_equal",
          "query_types": "query_types", "sizes": "first_stage_sizes", "expansion": "expansion", "topk": "topk_recall",
          "expansion_compare": "expansion_compare", "best_settings": "best_settings", "chunker_compare": "chunker_compare"}


def _head(r: Results) -> str:
    return f"{r.dataset} · {r.scope} · {r.variant} · {r.df.qid.nunique()} questions"


def _draw(r: Results, out, keys) -> None:
    for key in keys:
        chart = globals()[CHARTS[key]]
        if key in ALL_READINGS:
            chart(r, out, _head(r))
        else:
            for reading in rd.available(r.df, config.EVAL.readings):
                chart(r, out, _head(r), reading)


def write(r: Results) -> str:
    """Draw every chart of a run into its folder, cleared first so no chart from an earlier
    version of the code or of the results is left beside the new ones."""
    out = FIGURES / r.embedder / f"{r.dataset}__{r.scope}__{r.variant}__{r.sample}"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
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
