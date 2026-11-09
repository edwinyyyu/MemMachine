"""LongMemEval's chat histories as EventMemory events (partitions.py does the rest).

Each turn is an EventMemory event: producer User or Assistant, one text block, a uuid derived
from the question, session and turn. EventMemory orders a partition by (timestamp, event,
index, offset). LongMemEval dates sessions to the minute, gives turns no time, and its files
are not in date order (40 of 94 sampled M histories are not; most have sessions starting in
the same minute). So sessions are sorted by date, ties kept in file order, and each turn's
timestamp is its session's date plus one microsecond per earlier turn in the history: every
turn is ordered, and nothing rendered or embedded changes (renders show the minute, the
embedding text the date).
"""

from __future__ import annotations

import uuid
from datetime import timedelta
from functools import lru_cache

from rbench.datasets.longmemeval import history, session_datetime
from rbench.partitions import NAMESPACE, TICK, Part

from memmachine_server.episodic_memory.event_memory.data_types import Event, ProducerContext, TextBlock  # noqa: I001

PRODUCERS = {"user": "User", "assistant": "Assistant"}


def chronological(sessions: list[dict]) -> list[int]:
    """Session indices in date order, ties in file order."""
    return sorted(range(len(sessions)), key=lambda s: (session_datetime(sessions[s]["date"]), s))


@lru_cache(maxsize=4)
def part(dataset: str, qid: str) -> Part:
    """A history's turns as events in chronological order, labelled "session s, turn t" (s the
    session's index in the history file, t the turn's index in it)."""
    sessions = history(dataset, qid)["sessions"]
    events, labels, texts = [], [], []
    for s in chronological(sessions):
        start = session_datetime(sessions[s]["date"])
        for t, turn in enumerate(sessions[s]["turns"]):
            events.append(Event(uuid=uuid.uuid5(NAMESPACE, f"{dataset}/{qid}/{s}/{t}"), timestamp=start + len(events) * TICK,
                                context=ProducerContext(producer=PRODUCERS[turn["role"]]),
                                blocks=[TextBlock(text=turn["content"])]))
            labels.append(f"session {s}, turn {t}")
            texts.append(turn["content"])
    assert len(events) * TICK < timedelta(seconds=1), "increments must stay below the rendered resolution"
    return Part(qid, tuple(events), tuple(labels), tuple(texts))
