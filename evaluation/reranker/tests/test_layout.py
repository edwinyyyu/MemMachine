"""Every chart lays out cleanly: no two pieces of text overlap and none runs past the figure's
edge (titles, axis and tick labels, legends, colorbars). Draws every chart from the lme-test
stage's results; skipped when they have not been simulated."""

import matplotlib.figure as mf
import pytest

from rbench.paths import SIM

SIMULATED = SIM / "gemma300m-raw/longmemeval_s/history/question/tiny"


@pytest.mark.skipif(not SIMULATED.exists(), reason="needs the lme-test stage's simulation")
def test_no_chart_has_overlapping_or_clipped_text(tmp_path, monkeypatch):
    from rbench.report import figures, layout
    from rbench.report.data import load

    found: dict[str, list[str]] = {}
    real = mf.Figure.savefig

    def checked(self, path, *args, **kwargs):
        found[str(path).rsplit("/", 1)[-1]] = layout.problems(self)
        return real(self, path, *args, **kwargs)

    monkeypatch.setattr(mf.Figure, "savefig", checked)
    monkeypatch.setattr(figures, "FIGURES", tmp_path)
    figures.write(load("gemma300m-raw", "longmemeval_s", "history", "question", "tiny"))
    assert len(found) > 10
    bad = {name: p for name, p in found.items() if p}
    assert not bad, bad
