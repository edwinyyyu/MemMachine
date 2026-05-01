import sys
from pathlib import Path

RESEARCH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RESEARCH / "round16a_sliding_window" / "scenarios"))
import dormant_chains as _dc


def generate():
    return _dc.generate()[:200]


def ground_truth(turns):
    return _dc.ground_truth(turns)


def build_questions(gt):
    return _dc.build_questions(gt)
