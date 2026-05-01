"""Thin runner for two_speaker_composition experiment.

Delegates to `two_speaker_composition.main()`. Kept separate from the
composition module so that future experiment variations (K-sweeps, single
dataset runs) can import the module without re-running the full eval as a
side effect.

Usage:
    uv run python tspcomp_eval.py
"""

from __future__ import annotations

from two_speaker_composition import main

if __name__ == "__main__":
    main()
