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
