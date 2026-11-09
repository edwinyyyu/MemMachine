"""The incremental reader cost equals the additive model recomputed from scratch, and the model
tracks real tokenization of EventMemory's rendered string."""

import datetime
import random

import numpy as np

from rbench import eventmemory as em
from rbench.reading import Reader, constants
from rbench.text import GEMMA, count

WORDS = "I went to the store yesterday and bought apples, but my sister said the price was too high. ok?".split()


def history(rng, n_events=15):
    segs, block = [], []
    base = datetime.datetime(2023, 5, 20, 9, 0, tzinfo=datetime.UTC)
    for e in range(n_events):
        producer = rng.choice(["User", "Assistant"])
        for offset in range(rng.randint(1, 4)):
            segs.append(em.Segment(len(segs), e, 0, offset, base + datetime.timedelta(seconds=e), producer,
                                   " ".join(rng.choice(WORDS) for _ in range(rng.randint(2, 25)))))
            block.append(e)
    return segs, np.asarray(block)


def model_cost(segs, contexts, admitted, inner, run):
    """The additive model from scratch for one admitted set."""
    quotes, newline, separator = constants()
    filtered = [[segs[s] for s in c if s in admitted] for c in contexts]
    comps = em.components([f for f in filtered if f])
    total = sum(int(inner[s]) for s in admitted)
    for comp in comps:
        total += sum(int(run[x.id]) for i, x in enumerate(comp) if i == 0 or x.event != comp[i - 1].event)
    return total - newline * len(comps) + separator * max(0, len(comps) - 1)


def costs_for(segs):
    quotes, newline, _ = constants()
    inner = count([__import__("json").dumps(s.text, ensure_ascii=False)[1:-1] for s in segs], GEMMA)
    run = count([em._header(s, em.RENDER_FORMAT) for s in segs], GEMMA) + quotes + newline
    return inner, run


def random_case(rng):
    segs, block = history(rng)
    contexts, seeds = [], []
    for _ in range(rng.randint(1, 8)):
        seed = rng.randrange(len(segs))
        lo, hi = max(0, seed - rng.randint(0, 2)), min(len(segs), seed + rng.randint(0, 3) + 1)
        contexts.append(list(range(lo, hi)))
        seeds.append(seed)
    return segs, block, contexts, seeds


def test_incremental_equals_model_from_scratch():
    rng = random.Random(1)
    for _ in range(60):
        segs, block, contexts, seeds = random_case(rng)
        inner, run = costs_for(segs)
        order = [s.id for s in em.admission_order([[segs[i] for i in c] for c in contexts], seeds)]
        reader, admitted = Reader(contexts, block, inner, run), set()
        for s in order:
            admitted.add(s)
            assert reader.admit(s) == model_cost(segs, contexts, admitted, inner, run)


def test_model_tracks_real_tokenization():
    rng = random.Random(2)
    errors = []
    for _ in range(40):
        segs, block, contexts, seeds = random_case(rng)
        inner, run = costs_for(segs)
        order = [s.id for s in em.admission_order([[segs[i] for i in c] for c in contexts], seeds)]
        admitted = set(order)
        rendered = em.string_from_segment_contexts([[segs[s] for s in c if s in admitted] for c in contexts])
        real = count([rendered], GEMMA)[0]
        errors.append(abs(model_cost(segs, contexts, admitted, inner, run) - real) / real)
    assert max(errors) < 0.03 and np.mean(errors) < 0.01, (max(errors), np.mean(errors))


def test_lazy_contexts_enter_the_query_result_only_once_admission_reaches_them():
    """The first stage's vector search limit is the seeds it reads: contexts past the last one
    reached are not in the query result, so they cannot link components."""
    rng = random.Random(3)
    differs = 0
    for _ in range(60):
        segs, block, contexts, seeds = random_case(rng)
        inner, run = costs_for(segs)
        pairs = em.admission(contexts, seeds)
        lazy, eager, admitted = Reader(contexts, block, inner, run, lazy=True), Reader(contexts, block, inner, run), set()
        for s, k in pairs:
            admitted.add(s)
            got = lazy.admit(s, k)
            assert got == model_cost(segs, contexts[: k + 1], admitted, inner, run)
            differs += got != eager.admit(s)
    assert differs  # unread contexts do link components in the eager reader


def test_model_matches_string_from_query_result_at_each_limit():
    """The reader cost at segment limit L tracks the tokens of string_from_query_result(L), for
    a reranked pool (every context in the result) and for the first stage (contexts reached)."""
    rng = random.Random(4)
    errors = []
    for _ in range(30):
        segs, block, contexts, seeds = random_case(rng)
        inner, run = costs_for(segs)
        pairs = em.admission(contexts, seeds)
        as_segments = [[segs[i] for i in c] for c in contexts]
        for lazy in (False, True):
            reader = Reader(contexts, block, inner, run, lazy=lazy)
            for L, (s, k) in enumerate(pairs, start=1):
                cost = reader.admit(s, k if lazy else None)
                shown = as_segments[: k + 1] if lazy else as_segments
                text = em.string_from_query_result(shown, seeds[: len(shown)], L)
                real = count([text], GEMMA)[0]
                errors.append(abs(cost - real) / real)
    assert max(errors) < 0.05 and np.mean(errors) < 0.015, (max(errors), np.mean(errors))
