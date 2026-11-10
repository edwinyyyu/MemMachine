"""Layout check for the charts: no two pieces of text overlap, and none is cut off by the
figure's edge (tests/test_layout.py draws every chart and runs it)."""

from __future__ import annotations

from matplotlib.text import Text


def _drawn_ticks(axis) -> list[Text]:
    """Tick labels matplotlib actually draws: those whose tick lies inside the view."""
    lo, hi = sorted(axis.get_view_interval())
    span = (hi - lo) or 1.0
    out = []
    for tick in [*axis.get_major_ticks(), *axis.get_minor_ticks()]:
        loc = tick.get_loc()
        if loc is None or not lo - 1e-9 * span <= loc <= hi + 1e-9 * span:
            continue
        out += [label for label in (tick.label1, tick.label2) if label.get_visible()]
    return out


def _texts(fig) -> list[tuple[str, object]]:
    """Every drawn, non-empty text of a figure with its bounding box (display pixels): titles,
    axis labels, drawn tick labels, legends, colorbars and free texts."""
    renderer = fig.canvas.get_renderer()
    candidates: list[Text] = [*fig.texts]
    if getattr(fig, "_suptitle", None) is not None:
        candidates.append(fig._suptitle)
    for legend in fig.legends:
        candidates += [*legend.get_texts(), legend.get_title()]
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        candidates += [ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label, *ax.texts]
        candidates += _drawn_ticks(ax.xaxis) + _drawn_ticks(ax.yaxis)
        if ax.get_legend() is not None:
            candidates += [*ax.get_legend().get_texts(), ax.get_legend().get_title()]
    out, seen = [], set()
    for t in candidates:
        if id(t) in seen:
            continue
        seen.add(id(t))
        if not t.get_visible() or not t.get_text().strip():
            continue
        box = t.get_window_extent(renderer)
        if box.width > 0 and box.height > 0:
            out.append((t.get_text(), box))
    return out


def problems(fig, slack: float = 1.0) -> list[str]:
    """Pairs of overlapping texts and texts outside the figure, as messages (empty when fine)."""
    fig.canvas.draw()
    texts = _texts(fig)
    width, height = fig.bbox.width, fig.bbox.height
    out = []
    for text, b in texts:
        if b.x0 < -slack or b.y0 < -slack or b.x1 > width + slack or b.y1 > height + slack:
            out.append(f"cut off: {text[:60]!r}")
    for i, (ta, a) in enumerate(texts):
        for tb, b in texts[i + 1 :]:
            if a.x0 < b.x1 - slack and b.x0 < a.x1 - slack and a.y0 < b.y1 - slack and b.y0 < a.y1 - slack:
                out.append(f"overlap: {ta[:50]!r} / {tb[:50]!r}")
    return out
