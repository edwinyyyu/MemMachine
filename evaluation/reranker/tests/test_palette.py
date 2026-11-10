"""Every color family the charts use is distinguishable pair by pair (report/palette.py: the
dataviz validator's checks, OKLab distance x100): at least FLOOR apart for normal vision and
CVD_FLOOR under simulated deuteranopia and protanopia, for every ranking's hue and 2-5 members."""

import pytest

from rbench.report import figures, palette

BASES = {"first stage": figures.SERIES["first_stage"], **{m: figures.SERIES[m] for m in figures.RERANKERS}}


@pytest.mark.parametrize("name", BASES)
@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_families_are_distinguishable(name, n):
    colors = palette.family(BASES[name], n)
    d = palette.distances(colors)
    assert len(set(colors)) == n
    assert min(d["normal"]) >= palette.FLOOR, (colors, d["normal"])
    assert min(min(d["deuteranopia"]), min(d["protanopia"])) >= palette.CVD_FLOOR, (colors, d)


def test_known_colors_round_trip():
    import numpy as np

    assert np.allclose(palette.oklab("#ffffff"), [1, 0, 0], atol=1e-3)
    assert np.allclose(palette.oklab("#000000"), [0, 0, 0], atol=1e-3)
