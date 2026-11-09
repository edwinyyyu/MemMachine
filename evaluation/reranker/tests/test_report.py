"""The markdown report renders: every table's header, delimiter and rows have the same number
of columns (GitHub drops a table whose delimiter row is short). Uses the lme-test stage's
simulation; skipped when it has not been run."""

import pytest

from rbench.paths import SIM

SIMULATED = SIM / "gemma300m-raw/longmemeval_s/history/question/tiny/seg500.parquet"


@pytest.mark.skipif(not SIMULATED.exists(), reason="needs the lme-test stage's simulation")
def test_every_table_has_consistent_columns(tmp_path, monkeypatch):
    from rbench.report import tables
    from rbench.report.data import load

    monkeypatch.setattr(tables, "SUMMARY", tmp_path)
    r = load("gemma300m-raw", "longmemeval_s", "history", "question", "tiny")
    for event in tables.EVENTS:
        lines = open(tables.write(r, event)).read().splitlines()
        i = 0
        while i < len(lines):
            if not lines[i].startswith("|"):
                i += 1
                continue
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            counts = {line.count("|") for line in lines[i:j]}
            assert len(counts) == 1, f"{event}: table at line {i + 1} has rows of {sorted(counts)} bars: {lines[i][:80]}"
            i = j


@pytest.mark.skipif(not SIMULATED.exists(), reason="needs the lme-test stage's simulation")
def test_expansion_pairs_plain_chunks_by_measured_size():
    """The plain comparison is the unexpanded chunking closest to the measured window, and a
    pair further apart than config.EVAL.size_tolerance is flagged, whatever the names say."""
    import pandas as pd

    from rbench.report.data import load
    from rbench.report.figures import expansion_rows

    r = load("gemma300m-raw", "longmemeval_s", "history", "question", "tiny")
    seeds = r.df[r.df.chunking == "seg500"]
    windows = {e: seeds.loc[seeds.expand == e, "unit_tokens"].groupby(seeds.qid).first().mean() for e in r.expansions}
    plain = seeds[seeds.expand == 0]
    # two invented chunkings, named against their sizes: "seg9000" measures like the 1-window,
    # "seg100" like the 3-window
    fakes = [plain.assign(chunking=name, unit_tokens=windows[e] * 1.05) for name, e in (("seg9000", 1), ("seg100", 3))]
    r.df = pd.concat([r.df, *fakes], ignore_index=True)
    panels = {p.expand: p for p in expansion_rows(r)}
    assert panels[1].plain == "seg9000" and panels[1].matched
    assert panels[3].plain == "seg100" and panels[3].matched
    assert panels[6].plain == "seg100" and not panels[6].matched and "NOT size-matched" in panels[6].sizes()
