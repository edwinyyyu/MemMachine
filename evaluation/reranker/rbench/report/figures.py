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
  cost_saving.png                % saved by reranking, over LLM price x coverage target
  query_types.png                reranker gain (same L) by question type, at each budget
  first_stage_sizes.png          first-stage coverage vs segment size (no reranker)
  expansion.png                  plain chunks vs small seeds expanded to the same read size
Single-reading charts are drawn for each reading in EVAL.readings, with a _<reading> suffix
except for "every".

Colors follow the dataviz reference palette and were checked with its validator: models take
their color, markers and label from their registry entry (rerankers.RERANKERS); the perfect
reranker is neutral gray; ordered values (segment sizes, budgets) use a one-hue ordinal ramp;
savings use a diverging blue (saves) / red (costs more) scale around a neutral gray midpoint.

Reading the charts: every data line is thin and solid, with no outline (an outline cuts gaps
into the lines beneath). Color tells models (and chunkers) apart; a line-only marker shape tells
what a line is: up caltrop, a reranker scoring the seeds; down caltrop, scoring the windows; x,
the first stage; +, the perfect reranker. Every line has one width, and markers sit at the
same points on every line: where two caltrops meet as a six-pointed star, the values are equal.
Coverage curves are exact step functions (one step
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator, NullFormatter  # noqa: E402

from rbench import config  # noqa: E402
from rbench.metrics import CURVE_BUDGETS, CURVE_KS  # noqa: E402
from rbench.paths import FIGURES  # noqa: E402
from rbench.report import frontier  # noqa: E402
from rbench.report import readings as rd  # noqa: E402
from rbench.report.data import (  # noqa: E402
    Results,
    default_segment,
    price,
    query_labels,
    size_label,
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
# Line-only marker shapes cover nothing beneath them and their centre is the data point. A
# model's seeds line takes the first of its registry pair (up caltrop "2"), its windows line the
# second (down caltrop "1"); the first stage x, the perfect reranker + (its bar along flat lines).
MARKERS = {"first_stage": "x", "bound": "+", **{m: spec.markers[0] for m, spec in RERANKERS.items()}}
WINDOW_MARKERS = {m: spec.markers[1] for m, spec in RERANKERS.items()}
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
EVENT_NOTE = "Evidence: the turns marked as holding the answer; in full = all their segments read, partly = any."
SHARE_NOTE = (" Share readings: each question counts the share of its evidence turns read (each turn once), averaged over "
              "questions; with one evidence turn it equals the all/some readings. Partly read is generous on long assistant "
              "turns, in full is strict.")

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
def _token_marks() -> list[int]:
    """Where step curves carry their series marker: every 1,000 reader tokens to the axis end."""
    return list(range(1000, max(config.EVAL.token_top, 1000) + 1, 1000))
LINE = 1.0  # the one line width (points): thin, with no outline, so overlapping lines stay whole


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
    marks = {k: kwargs.pop(k) for k in ("marker", "markersize", "markerfacecolor", "markeredgecolor", "markeredgewidth",
                                        "markevery") if k in kwargs}
    lines = ax.plot(x, y, **kwargs)
    if marks.get("marker"):
        ax.plot(x, y, linestyle="none", zorder=lines[0].get_zorder() + 0.1, **_hollow({**marks, "color": kwargs.get("color")}))
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


def _marker(key: str) -> str:
    base, _, what = key.partition("~")
    return WINDOW_MARKERS[base] if what == "windows" else MARKERS.get(base, "o")


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


def _measured_x(ax, measured: dict[str, float]) -> None:
    """A log axis of measured sizes, a labelled tick at each chunking's measured size (name and
    tokens), padded so the outermost markers clear the frame."""
    ax.set_xscale("log", base=2)
    xs = sorted(measured.values())
    ax.set_xlim(xs[0] / 1.25, xs[-1] * 1.25)
    ax.set_xticks(xs, [f"{c}\n{t:,.0f} tok" for c, t in sorted(measured.items(), key=lambda ct: ct[1])])
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
    return Line2D([], [], linestyle=ls, linewidth=LINE, **_hollow({"marker": marker, "color": color, "markersize": 9}))


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
    out = [("first_stage", f"first_stage@{limit}{tail}", SERIES["first_stage"], "-")]
    out += [(m, f"rerank:{m}@{limit}{tail}", SERIES[m], "-") for m in r.models_at(c)]
    if e:
        out += [(f"{m}~windows", f"xrerank:{m}@{limit}{tail}", SERIES[m], "-") for m in r.models_at(c)]
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


def _ranker_groups(r: Results, keys: set[str]) -> list[tuple[str, list, list[str]]]:
    """Two legend columns for charts drawn with _series: color = how the seeds are ordered (the
    first stage, a model, the bound), marker = what a model's reranker scores."""
    bases = {k.split("~")[0] for k in keys}
    color_keys = [k for k in ("first_stage", *r.models, "bound") if k in bases]
    names = {"first_stage": "first stage (vector search order)", "bound": _bound_label(r)}
    labels = [names.get(k, LABEL.get(k, k)) for k in color_keys]
    handles = [_key(SERIES.get(k, ORACLE), marker=MARKERS[k] if k in ("first_stage", "bound") else None)
               for k in color_keys]
    groups = [("order (color)", handles, labels)]
    models = [m for m in r.models if m in bases]
    pairs = {RERANKERS[m].markers for m in models}
    windows = any("~windows" in k for k in keys)
    if models:
        if len(pairs) == 1:  # one pair for every model: one key
            seeds, win = next(iter(pairs))
            groups.append(("what the reranker scores (marker)",
                           [_key(INK2, marker=seeds)] + ([_key(INK2, marker=win)] if windows else []),
                           ["the seeds (rerank, then read the seeds or their windows)"]
                           + (["the windows (expand, then rerank)"] if windows else [])))
        else:
            hs, ls = [], []
            for m in models:
                hs.append(_key(SERIES[m], marker=MARKERS[m]))
                ls.append(f"{LABEL[m]}: the seeds")
                if windows:
                    hs.append(_key(SERIES[m], marker=WINDOW_MARKERS[m]))
                    ls.append(f"{LABEL[m]}: the windows (expand, then rerank)")
            groups.append(("what the reranker scores (marker)", hs, ls))
    return groups


SAME_LIMIT = ("Each panel is one vector search limit L and one expand_context E: every line reads only the top L seeds of the "
              "same vector search, each with its window of E neighbouring segments, in vector order (first stage) or reranked.")
BOUND = (" Gray, with seeds alone (expand_context 0): the bound, the cheapest reading of the same pool any ranking could "
         "achieve (no reranker's line can pass above it): the evidence seeds read first, each reading's cheapest way. "
         "Where it levels off below 1, the rest of the evidence is not among the L seeds: a ceiling set by the vector "
         "search, not by the reader budget (report section 2 gives that share per L). No bound with windows: an exact "
         "one isn't computable here.")


def _bound_label(r: Results) -> str:
    """The bound's legend label, with the share of questions it is exact for."""
    rows = r.df[(r.df.kind == "bound") & (r.df.expand == 0)]  # simulations before 2026-11-09 have window floors
    if rows.empty:
        return "bound"
    return f"bound over the same pool, seeds alone (exact for {rows.bound_exact.astype(bool).mean():.0%} of questions)"
TOKENS_EQUAL = "Held equal: reader tokens, the vector search limit and the expansion (panel). Not equal: reranker tokens. " + SAME_LIMIT


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
                    if limit not in r.limits(c, config.EVAL.limits):
                        ax.set_visible(False)
                        continue
                    for key, name, color, ls in _series(r, c, limit, e, bound=True):
                        _steps(ax, rd.costs(r.rows(c, name), reading), 0, config.EVAL.token_top, _token_marks(), _marker(key), color=color,
                               linestyle=ls)
                        keys.add(key)
                    _token_x(ax)
                    _share_y(ax)
                    ax.set_title(f"{c} · L {limit[1:]} · {_unit_title(r, c, e)}", loc="left")
                    if i == len(expansions) - 1:
                        ax.set_xlabel("reader tokens (rendered context)\nexact at every budget; markers every 1,000")
                    if j == 0:
                        ax.set_ylabel(("mean over questions: " if rd.is_share(reading) else "share of questions: ")
                                      + EVENT_TEXT[reading])
            _save(fig, out / (f"coverage_tokens_{reading}" + (f"_{c}" if len(r.chunkings) > 1 else "") + ".png"),
                  f"Coverage ({EVENT_TEXT[reading]}) vs reader tokens, {_sized(r, c)} — {head}", groups=_ranker_groups(r, keys),
                  note=TOKENS_EQUAL + BOUND + " " + EVENT_NOTE
                       + (SHARE_NOTE if rd.is_share(reading) else ""))


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
            series = _series(r, c, limit, e)
            for j, (column, top, xlabel) in enumerate(((f"{reading}_ctx", min(50, int(limit[1:])), "seeds kept (k)"),
                                                        (f"{reading}_k", 200, "segments read (m = max_num_segments)"))):
                ax = axes[i][j]
                for key, name, color, ls in series:
                    _steps(ax, r.rows(c, name)[column].to_numpy(), 0, top,
                           range(5 if top <= 50 else 20, top + 1, 5 if top <= 50 else 20), _marker(key), color=color,
                           linestyle=ls)
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
                _plot(ax, values, ys, color=color, linestyle=ls, marker=_marker(key))
            _log_x(ax, 10, values)
            _share_y(ax)
            ax.set_xlabel("vector search limit L (seeds retrieved; a reranker scores all of them)\n"
                          "markers: the limits measured")
            ax.set_title(f"{c} · {_unit_title(r, c, e)}: 10 seeds kept, as L grows", loc="left")
        _save(fig, out / _file("coverage_counts", reading, c, r), f"Coverage at equal counts instead of tokens, {_sized(r, c)} — {head}",
              groups=_ranker_groups(r, keys),
              note="Held equal: seeds kept (left), segments read (middle), the vector search limit L (each panel; right: "
                   "as it grows) and the expansion (row). Not equal: reader tokens and reranker tokens. " + SAME_LIMIT
                   )


def _families(chunkings: list[str]) -> dict[str, list[str]]:
    """Chunkings by chunker family (seg, cut), each sorted by size limit."""
    out: dict[str, list[str]] = {}
    for c in sorted(chunkings, key=lambda c: int(c[3:])):
        out.setdefault(c[:3], []).append(c)
    return out


def _size_styles(chunkings: list[str]) -> dict[str, dict]:
    """Color by size limit; line and marker by chunker family (CHUNKERS)."""
    sizes = sorted({int(c[3:]) for c in chunkings})
    ramp = RAMP[3] if len(sizes) <= 3 else RAMP[5]
    color = dict(zip(sizes[-len(ramp):], ramp))  # beyond 5 sizes, the smallest go unplotted
    return {c: {"color": color[int(c[3:])], "ls": CHUNKERS[c[:3]][2], "marker": CHUNKERS[c[:3]][1]}
            for c in chunkings if int(c[3:]) in color}


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
                        pts.append((rows.rr_tokens.mean(), rd.hit(rd.costs(rows, reading), b).mean(),
                                    rd.hit(rd.costs(base, reading), b).mean()))
                if not pts:
                    continue
                xs, ys, bs = zip(*sorted(pts), strict=True)
                _plot(ax, xs, ys, color=s["color"], linestyle=s["ls"], marker=s["marker"])
                _plot(ax, xs, bs, color=s["color"], linewidth=LINE, marker="x")
            _log_x(ax)
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            _share_y(ax)
            ax.set_title(f"{LABEL[m]} · reader budget {b:,}", loc="left")
            ax.set_xlabel("reranker tokens per query (API billing)")
            if j == 0:
                ax.set_ylabel(_coverage(reading))
    keys = list(styles)
    handles = [_key(styles[k]["color"], styles[k]["ls"], styles[k]["marker"]) for k in keys] + [_key(INK2, marker="x")]
    labels = [f"{_sized(r, k)}, reranked" for k in keys] + ["first stage at the same vector search limits (no reranker tokens)"]
    _save(fig, out / _file("rerank_cost", reading), f"Coverage vs reranker tokens as the vector search limit grows — {head}",
          handles, labels, ncol=2,
          note="Held equal: reader tokens (per panel) and, at each marker, the vector search limit (5, 10, 20, ...). x: the "
               "reranker's tokens per query at that limit; the line with x markers is the first stage over the same seeds, drawn at "
               "the same x though it spends no reranker tokens. Seeds only (expand_context 0).")


def gain_sizes(r: Results, out, head: str, reading: str) -> None:
    """Reranker gain against measured segment size: coverage reranked minus coverage of the first
    stage over the same seeds (same vector search limit), at each configured budget, with 95%
    intervals; a panel per (model, limit), a line per (budget, chunker family)."""
    measured = {c: unit_tokens(r, c) for c in r.chunkings}
    # models scored at two or more measured chunkings (a model scored at one has no gain curve)
    models = [m for m in r.models if sum(1 for c in measured if measured[c] and m in r.models_at(c)) >= 2]
    sized = [c for c, t in measured.items() if t and any(m in r.models_at(c) for m in models)]
    families = _families(sized)
    if not models:
        return
    limits = [d for d in _limit_grid(r) if all(d in r.limits(c, config.EVAL.limits) for c in sized)]
    if not limits:
        return
    budgets = config.EVAL.budgets
    ramp = dict(zip(budgets, RAMP[5] if len(budgets) > 3 else RAMP[3], strict=False))
    fig, axes = _figure(len(models), len(limits), panel=(3.8, 3.2), sharey=True)
    for i, m in enumerate(models):
        for j, d in enumerate(limits):
            ax = axes[i][j]
            for b in budgets:
                for fam, members in families.items():
                    chunkings = [c for c in members if m in r.models_at(c)]
                    if not chunkings:
                        continue
                    means, los, his = [], [], []
                    chunkings = [c for c in chunkings if not r.rows(c, f"rerank:{m}@{d}").empty]  # scored this deep
                    if not chunkings:
                        continue
                    for c in chunkings:
                        rows, base = r.rows(c, f"rerank:{m}@{d}"), r.rows(c, f"first_stage@{d}")
                        gain = rd.hit(rd.costs(rows, reading), b) - rd.hit(rd.costs(base, reading), b)
                        lo, hi = interval(gain, base.cluster.to_numpy())
                        means.append(gain.mean())
                        los.append(gain.mean() - lo if np.isfinite(lo) else 0.0)
                        his.append(hi - gain.mean() if np.isfinite(hi) else 0.0)
                    xs = [measured[c] for c in chunkings]
                    ax.errorbar(xs, means, yerr=[los, his], color=ramp.get(b, INK2), linewidth=LINE, capsize=0, fmt="none")
                    _plot(ax, xs, means, color=ramp.get(b, INK2), marker=CHUNKERS[fam][1])
            ax.axhline(0, color=INK2, linewidth=LINE)
            _measured_x(ax, {c: measured[c] for c in sized})
            ax.set_title(f"{LABEL[m]} · L {d[1:]}", loc="left")
            ax.set_xlabel("measured mean segment of the retrieved seeds (tokens)")
            if j == 0:
                ax.set_ylabel(f"gain in {_coverage(reading)}")
    handles = [_key(ramp.get(b, INK2)) for b in budgets]
    labels = [f"reader budget {b:,}" for b in budgets]
    if len(families) > 1:
        handles += [_key(INK2, CHUNKERS[f][2], CHUNKERS[f][1]) for f in families]
        labels += [CHUNKER_NAMES[f] for f in families]
    _save(fig, out / _file("gain_sizes", reading), f"Reranker gain vs measured segment size — {head}", handles, labels,
          ncol=4,
          note="Held equal: the vector search limit (panel) and the reader budget (color). x: each chunking's measured mean "
               "segment over the seeds the first stage retrieved (" + ", ".join(sized) + "). y: coverage reranked minus "
               "coverage of the first stage over the same seeds, mean with 95% interval over questions. Seeds only "
               "(expand_context 0).")


def cost_saving(r: Results, out, head: str, reading: str) -> None:
    """Over a continuous range of LLM prices and coverage targets: the cost of reaching the
    target with the reranker relative to without it, both at one setting (frontier.setting: a
    chunking and expand_context). One panel per reranked configuration, its vector search limit
    held fixed for every target and price (frontier.cost_limits: EventMemory's default 20, the
    best L overall, the pool), against the first stage in vector order. Blue saves, red costs
    more; gray where the saving's 95% interval includes 0. Over it, the break-even LLM price per
    target (the 0% boundary, frontier.break_even) with its 95% interval, and with a prefix cache,
    drawn only at targets where reranking is clearly cheaper at some higher price."""
    if not r.models:
        return
    fixed = frontier.setting(r)
    llm = np.logspace(-2, 2, 81)  # $ per million reader tokens
    panels = [(m, L, what) for m in r.models for L, what in frontier.cost_limits(r, m, reading)]
    if not panels:
        return
    columns = max(sum(p[0] == m for p in panels) for m in r.models)
    models = [m for m in r.models if any(p[0] == m for p in panels)]
    fig, axes = _figure(len(models), columns, panel=(4.6, 3.8), sharey=True)
    finite = lambda v: np.where(np.isfinite(v) & (v > 0) & (v < 1e12), v, np.nan)  # noqa: E731
    n = r.df.qid.nunique()
    for row, m in enumerate(models):
        mine = [p for p in panels if p[0] == m]
        for col in range(columns):
            ax = axes[row][col]
            if col >= len(mine):
                ax.set_visible(False)
                continue
            _, L, what = mine[col]
            curves = frontier.cost_curves(r, m, list(llm), price(m), event=reading, fixed=fixed, limit=L)
            no = np.stack([curves[p][0] for p in llm], 1)  # [target, price]
            yes = np.stack([curves[p][1] for p in llm], 1)
            with np.errstate(invalid="ignore", divide="ignore"):
                saved = 100 * (1 - yes / no)
            lo_s, hi_s = frontier.saved_interval(r, m, list(llm), price(m), event=reading, fixed=fixed, limit=L)
            clear = ~((lo_s <= 0) & (hi_s >= 0))  # the saving's 95% interval excludes 0
            im = ax.pcolormesh(llm, frontier.TARGETS, np.where(clear, saved, np.nan), cmap=DIVERGING,
                               norm=TwoSlopeNorm(0, -50, 50), shading="nearest")
            layers = ((np.isfinite(saved) & ~clear, MUTED, 1.0),  # no clear difference
                      (np.isnan(no) & np.isfinite(yes), ORACLE, 0.6),  # only the reranker reaches the target
                      (np.isfinite(no) & np.isnan(yes), FIRST_ONLY, 1.0))  # only the first stage does
            for mask, color, alpha in layers:
                ax.pcolormesh(llm, frontier.TARGETS, np.where(mask, 1.0, np.nan), shading="nearest", alpha=alpha,
                              cmap=LinearSegmentedColormap.from_list("layer", [color, color]))
            api, lo, hi, _, _ = frontier.break_even(r, m, "rr_tokens", event=reading, fixed=fixed, limit=L)
            cached = frontier.break_even(r, m, "rr_prefix_tokens", with_interval=False, event=reading, fixed=fixed,
                                         limit=L)[0]
            y = frontier.TARGETS
            # where the line means something: reranking is clearly cheaper somewhere to its right
            cheaper = (clear & (saved > 0)) | (np.isnan(no) & np.isfinite(yes))  # [target, price]
            right = np.asarray([cheaper[i, llm >= finite(api)[i] * price(m)].any() if np.isfinite(finite(api)[i])
                                else False for i in range(len(y))])
            marks = [i for i, t in enumerate(y) if round(t * 100) % 10 == 0]  # a marker at every 0.1 of coverage
            for line in (lo, hi):  # the interval's edges: dotted, no markers
                ax.plot(np.where(right, finite(line), np.nan) * price(m), y, drawstyle="steps-mid", color=INK,
                        linestyle=":", linewidth=LINE)
            for line, marker in ((api, "x"), (cached, "+")):
                _plot(ax, np.where(right, finite(line), np.nan) * price(m), y, drawstyle="steps-mid", color=INK,
                      marker=marker, markevery=marks)
            ax.set_xlim(llm[0], llm[-1])
            ax.set_facecolor(SURFACE)  # blank where neither reaches the target
            _log_x(ax)
            ax.grid(False)
            ax.yaxis.set_major_locator(MultipleLocator(0.1))
            ax.set_title(f"{LABEL[m]} · L {L} ({what})", loc="left")
            ax.set_xlabel("LLM input price ($/M)")
            if col == 0:
                ax.set_ylabel(f"coverage target ({EVENT_TEXT[reading]})")
    fig.colorbar(im, ax=[a for row in axes for a in row], label="% of retrieval cost saved (reader + reranker tokens)")
    name = "cost_saving" + ("" if fixed == (default_segment(), 0) else f"_{fixed[0]}_x{fixed[1]}")
    _save(fig, out / _file(name, reading),
          f"Cost saved by reranking, at every LLM price and coverage target ({fixed[0]}, expand_context {fixed[1]}) — {head}",
          [_key(INK, marker="x"), _key(INK, ":"), _key(INK, marker="+"), Patch(color=MUTED),
           Patch(color=ORACLE, alpha=0.6), Patch(color=FIRST_ONLY)],
          ["break-even LLM price (reranker tokens as an API bills them)", "its 95% interval",
           "break-even with a prefix cache (prompt and question computed once)",
           "no clear difference (the saving's 95% interval includes 0)", "only the reranker reaches this target",
           "only the first stage reaches this target"],
          ncol=2,
          note=f"Held equal: the coverage target (y), the LLM input price (x), the chunking ({fixed[0]}) and "
               f"expand_context ({fixed[1]}); config.EVAL.cost_chunking / cost_expand draw other settings. Each panel is one "
               "reranked configuration, its vector search limit L held fixed at every target and price (its title "
               "names it: EventMemory's default, the L whose reranked coverage averaged over the realistic budgets is "
               "highest, and the pool), against the first stage in vector order, read as deep as the target needs. "
               "Color: the total cost (reader + reranker tokens) of reaching the target with the reranker, relative to "
               "without it, on the same questions it is measured on. Right of the break-even line reranking is cheaper; "
               "the line is drawn only at targets where reranking is clearly cheaper at some higher price (else it would "
               "mark nothing). Colored only where the saving (or extra cost) is clear: its 95% bootstrap interval over "
               f"questions excludes 0; light gray where it does not. Blank: neither side reaches the target. Each step "
               f"is one question (1/{n} of the coverage target). Long sessions: if retrieved context stays in the "
               "conversation and is re-sent m times, read x as price × m (with prompt caching, write w and read c "
               "multipliers: price × (w + c(m−1))). Exact values: report §7.")


def query_types(r: Results, out, head: str, reading: str) -> None:
    """Reranker gain by question type at each configured budget: reranked minus the first stage
    over the same seeds, at the deepest vector search limit scored (the whole pool: a shallow
    limit runs out of seeds before the larger budgets, and both sides would read everything)."""
    if not r.models or r.df.qid.nunique() < 20:
        return
    c = default_segment() if default_segment() in r.chunkings else r.chunkings[0]
    limits = r.limits(c, config.EVAL.limits)
    if not limits:
        return
    limit = limits[-1]  # the deepest scored: a shallow limit runs out of seeds before the larger budgets
    base, labels = query_labels(r, c, f"first_stage@{limit}")
    groups = [(f, v) for f in ("question_type", "form", "length", "context", "parts") if f in labels
              for v in sorted(set(labels[f])) if (labels[f] == v).sum() >= 10]
    if not groups:
        return
    models = [m for m in r.models if not r.rows(c, f"rerank:{m}@{limit}").empty]  # those scored this deep
    if not models:
        return
    budgets = config.EVAL.budgets
    fig, axes = _figure(1, len(budgets), panel=(3.4, 0.34 * len(groups) + 1.2), sharey=True)
    for j, b in enumerate(budgets):
        ax = axes[0][j]
        hit = rd.hit(rd.costs(base, reading), b)
        for k, m in enumerate(models):
            gain = rd.hit(rd.costs(r.rows(c, f"rerank:{m}@{limit}"), reading), b) - hit
            for g, (f, v) in enumerate(groups):
                mask = labels[f] == v
                lo, hi = interval(gain[mask], base.cluster.to_numpy()[mask])
                y = g + (k - (len(models) - 1) / 2) * 0.22
                if not np.isnan(lo):
                    ax.plot([lo, hi], [y, y], color=SERIES[m], linewidth=LINE)
                ax.plot([gain[mask].mean()], [y], "o", color=SERIES[m], markersize=5.5)
        ax.axvline(0, color=INK2, linewidth=LINE)
        ax.set_yticks(range(len(groups)), [f"{f}: {v} (n={int((labels[f] == v).sum())})" for f, v in groups])
        ax.invert_yaxis()
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.set_title(f"reader budget {b:,}", loc="left")
        ax.set_xlabel("gain in coverage")
    _save(fig, out / _file("query_types", reading), f"Reranker gain by question type ({c}, L {limit[1:]}) — {head}",
          [_key(SERIES[m], marker="o") for m in models], [LABEL[m] for m in models],
          note=f"Held equal: the reader budget (panel) and the vector search limit ({limit[1:]}): gain = {_coverage(reading)} "
               "reranked minus the first stage over the same seeds. Dot: mean; line: 95% interval. Seeds only.")


def first_stage_sizes(r: Results, out, head: str, reading: str) -> None:
    """First-stage coverage against each chunking's measured size, a line per chunker family."""
    measured = {c: unit_tokens(r, c) for c in r.chunkings if c[:3] in CHUNKERS}
    families = _families([c for c, t in measured.items() if t])
    if sum(map(len, families.values())) < 2:
        return
    budgets = config.EVAL.budgets
    fig, axes = _figure(1, len(budgets), panel=(3.4, 3.2), sharey=True)
    for j, b in enumerate(budgets):
        ax = axes[0][j]
        for fam, chunkings in families.items():
            color, marker, ls = CHUNKERS[fam]
            ys = [rd.hit(rd.costs(r.rows(c, "first_stage"), reading), b).mean() for c in chunkings]
            _plot(ax, [measured[c] for c in chunkings], ys, color=color, marker=marker, linestyle=ls)
        _measured_x(ax, {c: measured[c] for cs in families.values() for c in cs})
        _share_y(ax)
        ax.set_title(f"reader budget {b:,}", loc="left")
        ax.set_xlabel("measured mean segment / chunk (tokens)")
        if j == 0:
            ax.set_ylabel(f"{_coverage(reading)}, no reranker")
    _save(fig, out / _file("first_stage_sizes", reading), f"First-stage coverage vs measured segment size (no reranker) — {head}",
          [_key(CHUNKERS[f][0], CHUNKERS[f][2], CHUNKERS[f][1]) for f in families], [CHUNKER_NAMES[f] for f in families],
          note="Held equal: reader tokens (per panel). x: each chunking's measured mean unit (for histories, over the seeds "
               "the first stage retrieved; " + ", ".join(c for cs in families.values() for c in cs) + "). No reranker: "
               "the first stage reads as deep as the budget allows (this chart compares segment sizes, not rerankers).")


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
    colors = dict(EXPANSION_APPROACHES)
    present = set()
    for p in panels:
        i, j = expands.index(p.expand), models.index(p.model)
        ax = axes[i][j]
        for label, c, name in p.items:
            _steps(ax, rd.costs(r.rows(c, name), reading), 0, config.EVAL.token_top, _token_marks(), EXPANSION_MARKERS[label],
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
               "seed the first stage retrieved. Plain chunks: the unexpanded chunking of the same chunker closest to the window's "
               "measured size"
               + (f"; panels flag pairs more than {config.EVAL.size_tolerance:.0%} apart" if unmatched else "")
               + ". Held equal: reader tokens (x). Not equal: reranker tokens and vector search limit.")


def _shade(color: str, amount: float) -> str:
    """A lighter tint (amount > 0, toward the surface) or darker shade (< 0, toward the ink) of
    a color, so lines of one ranking stay in its hue."""
    return _mix(color, SURFACE, amount) if amount >= 0 else _mix(color, INK, -amount)


def _mix(color: str, toward: str, amount: float) -> str:
    """A color moved `amount` of the way toward another."""
    from matplotlib.colors import to_hex, to_rgb

    a, b = np.asarray(to_rgb(color)), np.asarray(to_rgb(toward))
    return to_hex(a + (b - a) * amount)


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
        seeds = RERANKERS[m].markers[0]  # every line here scores the seeds
        fig, axes = _figure(3, len(grids), panel=(4.4, 3.4))
        for j, (reading, g) in enumerate(grids.items()):
            limits, ks = g["limits"], g["ks"]
            ax = axes[0][j]
            clear = ~((g["low"] <= 0) & (g["high"] >= 0))
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
            series = [("first stage (top-k as L)", g["first"], first, "x")]
            if 20 in limits:
                series.append(("reranked, L 20 (EventMemory's default)", g["reranked"][limits.index(20)], model, seeds))
            series.append((f"reranked, L {limits[-1]} (the pool)", g["reranked"][-1], _shade(model, 0.55), seeds))
            series.append(("reranked, best L per k", np.asarray([g["reranked"][limits.index(b), jj] if b else np.nan
                                                                for jj, b in enumerate(bests)]),
                           _shade(model, -0.45), seeds))
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
            for k, amount in zip(topk.SLICE_KS, (0.55, 0.0, -0.45), strict=True):  # lighter: fewer kept
                if k not in ks:
                    continue
                col = ks.index(k)
                ys = g["reranked"][:, col]
                _plot(ax, limits, ys, color=_shade(model, amount), marker=seeds)
                _plot(ax, limits, np.full(len(limits), g["first"][col]), color=_shade(first, amount), marker="x")
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
              note=f"{frontier.setting(r)[0]}, expand_context 0. Top-k read both ways: L, the vector search limit (all "
                   "reranked), and k, the seeds kept after (re)ranking. Without a reranker top-k is both. Color is the "
                   f"ranking (blue: the first stage; {LABEL[m]}'s color: reranked), lightness the setting. Top row: the "
                   "reranker's recall gain over (L, k), colored only where its 95% interval excludes 0; '=' marks k = L, "
                   "where everything retrieved is kept, so reranking cannot change recall (0 by construction); k > L "
                   "is blank. Middle row: recall vs k at L 20, at the pool and at the best L for each k (lightest to "
                   "darkest). Bottom row: recall vs L at k 5, 10 and 20 (lightest to darkest, labelled at the right); "
                   "the first stage is flat, the same at every L. L and k are chosen once for all questions. Recall: the "
                   "share of a question's evidence turns found within k seeds (in full, or touched), and every-gold.")


ALL_READINGS = ("coverage", "rpf", "topk")  # drawn for every reading or measure at once
CHARTS = {"coverage": "coverage_by_tokens", "rpf": "recall_precision_f1", "counts": "coverage_counts",
          "rerank_cost": "rerank_cost", "gain_sizes": "gain_sizes", "cost_saving": "cost_saving",
          "query_types": "query_types", "sizes": "first_stage_sizes", "expansion": "expansion", "topk": "topk_recall"}


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
