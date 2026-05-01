"""Wrapper for multi_batch_coref."""

import sys
from pathlib import Path

RESEARCH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RESEARCH / "round16a_sliding_window" / "scenarios"))
import multi_batch_coref as _mc


def generate():
    return _mc.generate()


def ground_truth(turns):
    return _mc.ground_truth(turns)


def build_questions(gt):
    return _mc.build_questions(gt)
