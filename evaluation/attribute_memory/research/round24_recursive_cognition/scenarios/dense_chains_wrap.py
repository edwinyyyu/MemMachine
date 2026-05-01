"""Wrapper to expose round14's dense_chains[:200] for run_one.py regression checks."""

import sys
from pathlib import Path

RESEARCH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RESEARCH / "round14_chain_density" / "scenarios"))
import dense_chains as _dense


def generate():
    return _dense.generate()[:200]


def ground_truth(turns):
    return _dense.ground_truth(turns)


def build_questions(gt):
    return _dense.build_questions(gt)
