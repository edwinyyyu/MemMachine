import sys
from pathlib import Path

RESEARCH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RESEARCH / "round20_cognition_pass" / "scenarios"))
import hypothetical_becomes_real as _hbr


def generate():
    return _hbr.generate()


def ground_truth(turns):
    return _hbr.ground_truth(turns)


def build_questions(gt):
    return _hbr.build_questions(gt)
