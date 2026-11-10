"""The benchmark against MemMachine's own EventMemory (no services needed): a small synthetic
history in real stores, with a deterministic embedder and reranker standing in for the models.

  - replaying reranked configurations (scores of the deepest first stage, sliced to each
    vector search limit) gives exactly the order EventMemory.query returns at that limit;
  - reading.admission is EventMemory.build_query_result_context at every segment limit;
  - the reader-cost model tracks the tokens of EventMemory.string_from_query_result;
  - a history is encoded once, and the segment table follows a new encoding.
"""

import asyncio
import datetime
import hashlib
import random
import uuid

import numpy as np
import pytest

from rbench import partitions as pt
from rbench.memory import RERANK_FORMAT, ExactSearchEngine, Memory
from rbench.reading import Reader, admission
from rbench.text import GEMMA, count

from memmachine_server.common.data_types import SimilarityMetric  # noqa: E402, I001
from memmachine_server.common.embedder import Embedder  # noqa: E402
from memmachine_server.common.reranker import Reranker  # noqa: E402
from memmachine_server.episodic_memory.event_memory.data_types import Event, ProducerContext, TextBlock  # noqa: E402
from memmachine_server.episodic_memory.event_memory.event_memory import EventMemory  # noqa: E402

WORDS = "I went to the store yesterday and bought apples, but my sister said the price was too high. ok? Then we left.".split()


def _vector(text: str) -> list[float]:
    rng = np.random.default_rng(int(hashlib.sha1(text.encode()).hexdigest()[:8], 16))
    return rng.normal(size=768).tolist()


class FakeEmbedder(Embedder):
    async def _ingest_embed(self, inputs, max_attempts=1):
        return [_vector(t) for t in inputs]

    async def _search_embed(self, queries, max_attempts=1):
        return [_vector(q) for q in queries]

    @property
    def model_id(self) -> str:
        return "fake"

    @property
    def dimensions(self) -> int:
        return 768

    @property
    def similarity_metric(self) -> SimilarityMetric:
        return SimilarityMetric.COSINE


class FakeReranker(Reranker):
    async def score(self, query, candidates):
        return [int(hashlib.sha1((query + c).encode()).hexdigest()[:6], 16) / 2**24 for c in candidates]


def _events(n: int = 40, seed: int = 0) -> list[Event]:
    rng = random.Random(seed)
    base = datetime.datetime(2023, 5, 20, 9, 0, tzinfo=datetime.UTC)
    return [Event(uuid=uuid.UUID(int=rng.getrandbits(128)), timestamp=base + datetime.timedelta(minutes=i),
                  context=ProducerContext(producer=rng.choice(["User", "Assistant"])),
                  blocks=[TextBlock(text=" ".join(rng.choice(WORDS) for _ in range(rng.randint(3, 40))))])
            for i in range(n)]


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def stored(tmp_path):
    events = _events()

    async def go():
        async with Memory("fake", "synthetic", "seg80", root=tmp_path, backend=FakeEmbedder()) as m:
            assert await m.ingest("h", events)
            assert not await m.ingest("h", events)  # encoded once
            partition = await (await m.event_memory()).open_or_create_partition("h")
            segment_store, _, _ = await m.stores()
            part = await segment_store.open_partition("h")
            by_event = await part.get_segment_uuids_by_event_uuids([e.uuid for e in events])
            ids = [u for e in events for u in by_event.get(e.uuid, [])]
            segments = pt._ordered([v[0] for v in (await part.get_segment_contexts(ids)).values()])
            return segments, partition

    return events, tmp_path, _run(go())


def test_exact_search_is_stable_and_survives_a_save(tmp_path):
    engine = ExactSearchEngine(768, SimilarityMetric.COSINE)
    vectors = {k: _vector(str(k % 7)) for k in range(30)}  # many exact ties

    async def go():
        await engine.add(vectors)
        deep = (await engine.search([_vector("3")], limit=20))[0].matches
        shallow = (await engine.search([_vector("3")], limit=8))[0].matches
        assert [m.key for m in shallow] == [m.key for m in deep[:8]]  # a shallower search is a prefix
        await engine.save(str(tmp_path / "i"))
        again = ExactSearchEngine(768, SimilarityMetric.COSINE)
        await again.load(str(tmp_path / "i"))
        assert [m.key for m in (await again.search([_vector("3")], limit=20))[0].matches] == [m.key for m in deep]

    _run(go())


def test_replayed_reranking_matches_eventmemory_query_at_each_limit(tmp_path):
    events = _events()
    question = "what did my sister say about the price?"

    async def go():
        async with Memory("fake", "synthetic", "seg80", root=tmp_path, backend=FakeEmbedder()) as m:
            await m.ingest("h", events)
            plain = await (await m.event_memory()).open_or_create_partition("h")
            reranked = await (await m.event_memory(FakeReranker())).open_or_create_partition("h")
            for e in (0, 3):
                deep = (await plain.query(question, vector_search_limit=30, expand_context=e)).scored_segment_contexts
                texts = [EventMemory.string_from_segment_context(c.segments, format_options=RERANK_FORMAT) for c in deep]
                scores = await FakeReranker().score(question, texts)
                for limit in (3, 7, 15, 30):
                    replay = sorted(range(limit), key=lambda i: -scores[i])  # stable: ties keep similarity order
                    real = (await reranked.query(question, vector_search_limit=limit, expand_context=e)).scored_segment_contexts
                    assert [deep[i].seed_segment_uuid for i in replay] == [c.seed_segment_uuid for c in real]
                    assert [deep[i].segments for i in replay] == [c.segments for c in real]

    _run(go())


def test_admission_is_build_query_result_context_at_every_limit(stored):
    _, _, (segments, partition) = stored
    index = {s.uuid: i for i, s in enumerate(segments)}

    async def go():
        for e in (0, 2, 3, 6):
            result = await partition.query("bought apples", vector_search_limit=12, expand_context=e)
            contexts = [[index[s.uuid] for s in c.segments] for c in result.scored_segment_contexts]
            seeds = [index[c.seed_segment_uuid] for c in result.scored_segment_contexts]
            order = [s for s, _ in admission(contexts, seeds)]
            for limit in range(1, len(order) + 3):
                got = {index[s.uuid] for s in EventMemory.build_query_result_context(result, limit)}
                assert got == set(order[:limit])

    _run(go())


def test_reader_cost_tracks_string_from_query_result(stored):
    """At every segment limit, for a reranked pool (every context in the result) and for the
    first stage (only the contexts reached), within a few percent of real tokenization."""
    _, _, (segments, partition) = stored
    index = {s.uuid: i for i, s in enumerate(segments)}
    _, inner, run = pt._costs(segments, None)
    block = np.asarray([hash(s.event_uuid) for s in segments])
    from memmachine_server.episodic_memory.event_memory.data_types import QueryResult

    async def go():
        errors = []
        for e in (0, 3):
            result = await partition.query("price was too high", vector_search_limit=10, expand_context=e)
            scored = result.scored_segment_contexts
            contexts = [[index[s.uuid] for s in c.segments] for c in scored]
            pairs = admission(contexts, [index[c.seed_segment_uuid] for c in scored])
            for lazy in (False, True):
                reader = Reader(contexts, block, inner, run, lazy=lazy)
                for limit, (s, k) in enumerate(pairs, start=1):
                    cost = reader.admit(s, k if lazy else None)
                    shown = QueryResult(scored_segment_contexts=scored[: k + 1] if lazy else scored)
                    real = count([EventMemory.string_from_query_result(shown, max_num_segments=limit)], GEMMA)[0]
                    errors.append(abs(cost - real) / real)
        return errors

    errors = _run(go())
    assert max(errors) < 0.05 and np.mean(errors) < 0.015, (max(errors), np.mean(errors))


def test_a_new_encoding_refreshes_the_segment_table(tmp_path, monkeypatch):
    events = _events(12)
    monkeypatch.setattr(pt, "formats", lambda dataset: pt.CHAT)
    p = pt.Part("h", tuple(events), tuple(f"turn {i}" for i in range(len(events))),
                tuple(e.blocks[0].text for e in events))

    async def go():
        async with Memory("fake", "synthetic", "seg80", root=tmp_path, backend=FakeEmbedder()) as m:
            await m.ingest("h", events)
            first = await pt.table(m, "synthetic", p)
            assert (await pt.table(m, "synthetic", p)).uuid == first.uuid  # cached
            m.done("h").unlink()  # as if the encoding had been interrupted: encoded again
            assert await m.ingest("h", events)
            again = await pt.table(m, "synthetic", p)
            assert again.texts == first.texts and again.uuid != first.uuid  # new uuids, table refreshed

    _run(go())


def test_simple_chunker_cuts_every_n_characters_within_each_turn():
    """cut<N> pieces are N characters (the last shorter), join back to the turn exactly, and
    never span events; seg<N> is MemMachine's TextSegmenter."""
    from rbench.memory import FixedCutSegmenter, segmenter

    from memmachine_server.episodic_memory.event_memory.segmenter.text_segmenter import TextSegmenter  # noqa: I001

    text = "  Hello there.\n\nThis turn is cut every seven characters, whatever it holds.  "
    event = Event(uuid=uuid.uuid4(), timestamp=datetime.datetime(2023, 5, 20, tzinfo=datetime.UTC),
                  context=ProducerContext(producer="User"), blocks=[TextBlock(text=text)])
    pieces = asyncio.run(segmenter("cut7").segment(event))
    assert isinstance(segmenter("cut7"), FixedCutSegmenter) and isinstance(segmenter("seg500"), TextSegmenter)
    assert "".join(p.block.text for p in pieces) == text
    assert [len(p.block.text) for p in pieces[:-1]] == [7] * (len(pieces) - 1) and 0 < len(pieces[-1].block.text) <= 7
    assert [p.offset for p in pieces] == list(range(len(pieces))) and {p.event_uuid for p in pieces} == {event.uuid}
    with pytest.raises(ValueError):
        segmenter("fix500")


def test_large_partitions_are_encoded_in_batches_under_sqlites_variable_limit(tmp_path, monkeypatch):
    """A partition with more segments than one statement can bind is encoded in batches, and
    the stored segments are the same as one call's."""
    from rbench import memory

    monkeypatch.setattr(memory, "ENCODE_SEGMENTS", 5)
    events = _events(12)

    async def go():
        async with Memory("fake", "synthetic", "seg80", root=tmp_path, backend=FakeEmbedder()) as m:
            batches = await m._batches(events, None)
            assert len(batches) > 1 and [e for b in batches for e in b] == events
            assert await m.ingest("h", events)
            segment_store, _, _ = await m.stores()
            part = await segment_store.open_partition("h")
            by_event = await part.get_segment_uuids_by_event_uuids([e.uuid for e in events])
            stored = sum(len(v) for v in by_event.values())
            expected = sum([len(await m.segmenter.segment(e)) for e in events])
            assert stored == expected

    _run(go())


def test_question_cache_is_keyed_on_the_query_side_and_survives_a_torn_line(tmp_path, monkeypatch):
    """A changed instruction or size gets its own cache (no old vectors reused); a line a crash
    cut short is skipped; embedders that embed documents alike share one store."""
    from dataclasses import replace

    from rbench import embedding as emb
    from rbench import memory

    base = memory.questions_path("qwen3-8b-di")
    monkeypatch.setitem(emb.EMBEDDERS, "qwen3-8b-di", replace(emb.EMBEDDERS["qwen3-8b-di"], query_prompt="Instruct: x\nQuery:"))
    assert memory.questions_path("qwen3-8b-di") != base
    monkeypatch.setitem(emb.EMBEDDERS, "qwen3-8b-di", replace(emb.EMBEDDERS["qwen3-8b-di"], dimensions=512))
    assert memory.questions_path("qwen3-8b-di") != base
    path = tmp_path / "q.jsonl"
    path.write_text('{"q": "a", "v": [1.0, 0.0]}\n{"q": "b", "v": [0.0')  # the second line was cut short
    assert memory._read_questions(path) == {"a": [1.0, 0.0]}
    assert memory.store_root("qwen3-8b-di", "d", "seg500") == memory.store_root("qwen3-8b-di-raw", "d", "seg500")
    assert memory.store_root("gemma300m-raw", "d", "seg500") != memory.store_root("qwen3-8b-di", "d", "seg500")
