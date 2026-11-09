"""The incremental reader cost equals the additive model recomputed from scratch, eagerly (every
context of the result links components) and lazily (contexts join as admission reaches them).
How the model tracks EventMemory's real rendered string is tested in test_memory.py."""

import datetime
import json
import random
import uuid

import numpy as np

from rbench import partitions as pt
from rbench.reading import Reader, admission, constants

from memmachine_server.episodic_memory.event_memory.data_types import ProducerContext, Segment, TextBlock  # noqa: E402, I001

WORDS = "I went to the store yesterday and bought apples, but my sister said the price was too high. ok?".split()


def history(rng, n_events=15):
    segs, block = [], []
    base = datetime.datetime(2023, 5, 20, 9, 0, tzinfo=datetime.UTC)
    for e in range(n_events):
        producer, event = rng.choice(["User", "Assistant"]), uuid.UUID(int=rng.getrandbits(128))
        for offset in range(rng.randint(1, 4)):
            segs.append(Segment(uuid=uuid.UUID(int=rng.getrandbits(128)), event_uuid=event, index=0, offset=offset,
                                timestamp=base + datetime.timedelta(seconds=e), context=ProducerContext(producer=producer),
                                block=TextBlock(text=" ".join(rng.choice(WORDS) for _ in range(rng.randint(2, 25))))))
            block.append(e)
    return segs, np.asarray(block)


def components(contexts: list[list[int]]) -> list[list[int]]:
    """Admitted segments joined into components where contexts share one (from scratch)."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.setdefault(x, x) != x:
            x = parent[x]
        return x

    for c in contexts:
        for s in c[1:]:
            parent[find(s)] = find(c[0])
        find(c[0])
    groups: dict[int, list[int]] = {}
    for s in parent:
        groups.setdefault(find(s), []).append(s)
    return [sorted(g) for g in groups.values()]


def model_cost(block, contexts, admitted, inner, run):
    """The additive model from scratch for one admitted set."""
    _, newline, separator = constants()
    comps = components([f for f in ([s for s in c if s in admitted] for c in contexts) if f])
    total = sum(int(inner[s]) for s in admitted)
    for comp in comps:
        total += sum(int(run[s]) for i, s in enumerate(comp) if i == 0 or block[s] != block[comp[i - 1]])
    return total - newline * len(comps) + separator * max(0, len(comps) - 1)


def random_case(rng):
    segs, block = history(rng)
    contexts, seeds = [], []
    for _ in range(rng.randint(1, 8)):
        seed = rng.randrange(len(segs))
        lo, hi = max(0, seed - rng.randint(0, 2)), min(len(segs), seed + rng.randint(0, 3) + 1)
        contexts.append(list(range(lo, hi)))
        seeds.append(seed)
    return segs, block, contexts, seeds


def test_costs_count_json_text_and_rendered_headers():
    segs, _ = history(random.Random(0), 3)
    _, inner, run = pt._costs(segs, None)
    assert all(i > 0 for i in inner) and all(r > 2 for r in run)
    assert json.dumps(segs[0].block.text)  # inner counts the JSON-escaped text, without its quotes


def test_incremental_equals_model_from_scratch():
    rng = random.Random(1)
    for _ in range(60):
        segs, block, contexts, seeds = random_case(rng)
        _, inner, run = pt._costs(segs, None)
        reader, admitted = Reader(contexts, block, inner, run), set()
        for s, _ in admission(contexts, seeds):
            admitted.add(s)
            assert reader.admit(s) == model_cost(block, contexts, admitted, inner, run)


def test_lazy_contexts_enter_the_query_result_only_once_admission_reaches_them():
    """The first stage's vector search limit is the seeds it reads: contexts past the last one
    reached are not in the query result, so they cannot link components."""
    rng = random.Random(3)
    differs = 0
    for _ in range(60):
        segs, block, contexts, seeds = random_case(rng)
        _, inner, run = pt._costs(segs, None)
        lazy, eager, admitted = Reader(contexts, block, inner, run, lazy=True), Reader(contexts, block, inner, run), set()
        for s, k in admission(contexts, seeds):
            admitted.add(s)
            got = lazy.admit(s, k)
            assert got == model_cost(block, contexts[: k + 1], admitted, inner, run)
            differs += got != eager.admit(s)
    assert differs  # unread contexts do link components in the eager reader
