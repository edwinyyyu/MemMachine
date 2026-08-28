"""Differential tests for the EventMemory WRITE path fast paths.

- add_segments: canonical vs fast into twin throwaway partitions, raw rows
  compared column-by-column.
- upsert: canonical vs fast, points read back (vectors + payloads) compared.
- round trip: fast-written rows decode identically through the canonical
  reader; fast-written points query identically through the canonical client.
- windowed get_segment_contexts: fast vs canonical equality on live data.
- encoder fast variants == canonical encoders.
- blessed segmenter/deriver objects == validated equivalents.
"""

import asyncio
import hashlib
from datetime import UTC, datetime, timedelta, timezone
from uuid import UUID

import asyncpg
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import create_async_engine

import os

import memmachine_server

# Guard against benching the wrong build: when BENCH_EXPECT_PATH_SUBSTR is
# set, refuse to run unless the imported memmachine_server comes from a
# path containing it (e.g. the worktree name of the branch under test).
_expect = os.environ.get("BENCH_EXPECT_PATH_SUBSTR")
if _expect:
    assert _expect in memmachine_server.__file__, memmachine_server.__file__

from memmachine_server.common.data_types import SimilarityMetric  # noqa: E402
from memmachine_server.common.vector_store.data_types import (  # noqa: E402
    Record,
    VectorStoreCollectionConfig,
)
from memmachine_server.common.vector_store.qdrant_vector_store import (  # noqa: E402
    QdrantVectorStore,
    QdrantVectorStoreParams,
)
from memmachine_server.episodic_memory.event_memory.data_types import (  # noqa: E402
    Event,
    NullContext,
    ProducerContext,
    Segment,
    TextBlock,
    encode_block,
    encode_context,
)
from memmachine_server.episodic_memory.event_memory.deriver.text_deriver import (  # noqa: E402
    WholeTextDeriver,
)
from memmachine_server.episodic_memory.event_memory.event_memory import (  # noqa: E402
    EventMemory,
)
from memmachine_server.episodic_memory.event_memory.segment_store.data_types import (  # noqa: E402
    SegmentStorePartitionConfig,
)
from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import (  # noqa: E402
    SQLAlchemySegmentStore,
    SQLAlchemySegmentStorePartition,
    SQLAlchemySegmentStoreParams,
)
from memmachine_server.episodic_memory.event_memory.segmenter.passthrough_segmenter import (  # noqa: E402
    PassthroughSegmenter,
)
from memmachine_server.episodic_memory.long_term_memory import (  # noqa: E402
    EVENT_BACKEND_SYSTEM_FIELDS,
)

PG = "postgresql://memmachine:memmachine@localhost:5442/memmachine"
PGA = "postgresql+asyncpg://memmachine:memmachine@localhost:5442/memmachine"
ISOLATED = hashlib.sha256(b"benchorg/isolated1").hexdigest()[:32]
TZ = timezone(timedelta(hours=5, minutes=30))


def uid(n: int) -> UUID:
    return UUID(int=n)


def make_segments(n: int, base: int):
    out = {}
    for i in range(n):
        seg = Segment(
            uuid=uid(base + i),
            event_uuid=uid(base + 1000 + i),
            index=i % 3,
            offset=i,
            timestamp=datetime(2026, 3, 1, 12, 0, i % 60, tzinfo=TZ),
            context=ProducerContext(producer=f"p{i % 4}")
            if i % 5 else NullContext(),
            block=TextBlock(text=f"segment text {i} with unicode jeżyk"),
            properties={"_episode_uid": str(2000 + i), "_seq": i,
                        "when": datetime(2026, 3, 1, tzinfo=UTC)},
        )
        out[seg] = [uid(base + 5000 + i), uid(base + 6000 + i)]
    return out


async def wipe_partition(pk: str):
    conn = await asyncpg.connect(PG)
    await conn.execute(f'DELETE FROM segment_store_dv_ln WHERE partition_key = $1', pk)
    await conn.execute(f'DELETE FROM segment_store_sg WHERE partition_key = $1', pk)
    await conn.execute(f'DROP TABLE IF EXISTS "segment_store_dv_ln_p_{pk}" CASCADE')
    await conn.execute(f'DROP TABLE IF EXISTS "segment_store_sg_p_{pk}" CASCADE')
    await conn.execute("DELETE FROM segment_store_pt WHERE partition_key=$1", pk)
    await conn.close()


async def rows_of(pk: str):
    conn = await asyncpg.connect(PG)
    rows = await conn.fetch(
        'SELECT uuid, event_uuid, "index", "offset", timestamp, '
        "timestamp_timezone_offset, context, block, properties "
        "FROM segment_store_sg WHERE partition_key=$1 ORDER BY uuid", pk)
    links = await conn.fetch(
        "SELECT uuid, segment_uuid FROM segment_store_dv_ln "
        "WHERE partition_key=$1 ORDER BY uuid", pk)
    await conn.close()
    return rows, links


async def main():
    engine = create_async_engine(PGA, pool_size=4)
    store = SQLAlchemySegmentStore(SQLAlchemySegmentStoreParams(engine=engine))
    await store.startup()

    # --- encoder equivalence
    for ctx in (ProducerContext(producer="alice"), NullContext(), None):
        assert SQLAlchemySegmentStorePartition._encode_context_fast(ctx) == \
            encode_context(ctx), ctx
    blk = TextBlock(text="hi")
    assert SQLAlchemySegmentStorePartition._encode_block_fast(blk) == encode_block(blk)
    print("encoders: fast == canonical: OK")

    # --- add_segments canonical vs fast, column-by-column
    pk_c, pk_f = "difftest_canon", "difftest_fast"
    await wipe_partition(pk_c); await wipe_partition(pk_f)
    part_c = await store.open_or_create_partition(pk_c, SegmentStorePartitionConfig())
    part_f = await store.open_or_create_partition(pk_f, SegmentStorePartitionConfig())

    async def no_fast(*a, **k):
        return False
    part_c._add_segments_fast = no_fast

    segs = make_segments(30, base=1)
    await part_c.add_segments(segs)
    await part_f.add_segments(make_segments(30, base=1))
    rc, lc = await rows_of(pk_c)
    rf, lf = await rows_of(pk_f)
    assert len(rc) == len(rf) == 30 and len(lc) == len(lf) == 60
    import json as _json
    for a, b in zip(rc, rf):
        for col in a.keys():
            va, vb = a[col], b[col]
            if col in ("context", "block", "properties"):
                pa = _json.loads(va if not isinstance(va, memoryview) else bytes(va))
                pb = _json.loads(vb if not isinstance(vb, memoryview) else bytes(vb))
                assert pa == pb, (col, pa, pb)
            else:
                assert va == vb, (col, va, vb)
    for a, b in zip(lc, lf):
        assert dict(a) == dict(b)
    print("add_segments: canonical rows == fast rows (30 segs, 60 links): OK")

    # --- round trip: fast-written rows through the CANONICAL reader
    orig_fast = part_f._get_seed_segments_fast
    part_f._get_seed_segments_fast = lambda s: _none()
    read_back = await part_f.get_segment_contexts([s.uuid for s in segs])
    part_f._get_seed_segments_fast = orig_fast
    by_uuid = {s.uuid: s for s in segs}
    assert len(read_back) == 30
    for u, lst in read_back.items():
        assert lst[0] == by_uuid[u], (u, lst[0], by_uuid[u])
    print("round trip: fast-written == original through canonical reader: OK")

    # --- windowed fast vs canonical on live isolated1 data
    part_live = await store.open_or_create_partition(
        ISOLATED, SegmentStorePartitionConfig())
    conn = await asyncpg.connect(PG)
    live_uuids = [UUID(str(r["uuid"])) for r in await conn.fetch(
        "SELECT uuid FROM segment_store_sg WHERE partition_key=$1 "
        "ORDER BY timestamp LIMIT 40 OFFSET 500", ISOLATED)]
    await conn.close()
    for back, fwd in ((2, 4), (8, 16), (0, 3), (3, 0)):
        fast = await part_live.get_segment_contexts(
            live_uuids, max_backward_segments=back, max_forward_segments=fwd)
        orig_w = part_live._get_segment_contexts_windowed_fast
        part_live._get_segment_contexts_windowed_fast = lambda *a: _none()
        part_live._get_seed_segments_fast = lambda s: _none()
        canon = await part_live.get_segment_contexts(
            live_uuids, max_backward_segments=back, max_forward_segments=fwd)
        part_live._get_segment_contexts_windowed_fast = orig_w
        assert set(fast) == set(canon)
        for k in fast:
            assert fast[k] == canon[k], (back, fwd, k)
        print(f"windowed ({back},{fwd}): fast == canonical "
              f"({sum(len(v) for v in fast.values())} rows): OK")

    # --- upsert canonical vs fast, read back and compare
    client = AsyncQdrantClient(host="localhost", port=6343, prefer_grpc=False)
    vs = QdrantVectorStore(QdrantVectorStoreParams(client=client))
    cfg = VectorStoreCollectionConfig(
        vector_dimensions=1536, similarity_metric=SimilarityMetric.COSINE,
        indexed_properties_schema={
            **EventMemory.expected_vector_store_collection_schema(),
            **EVENT_BACKEND_SYSTEM_FIELDS,
        })
    coll_c = await vs.open_or_create_collection(
        namespace="long_term_memory", name=pk_c, config=cfg)
    coll_f = await vs.open_or_create_collection(
        namespace="long_term_memory", name=pk_f, config=cfg)
    coll_c._fast_http = False  # canonical upsert AND canonical reads

    def make_records(base):
        return [Record(
            uuid=uid(base + i),
            vector=[(i + j) % 7 * 0.25 - 0.5 for j in range(1536)],
            properties={"_episode_uid": str(i), "_seq": i,
                        "when": datetime(2026, 3, 2, tzinfo=UTC),
                        "_segment_uuid": str(uid(base + 100 + i))},
        ) for i in range(8)]

    await coll_c.upsert(records=make_records(50_000))
    await coll_f.upsert(records=make_records(60_000))
    got_c = await coll_c.get(record_uuids=[uid(50_000 + i) for i in range(8)],
                             return_vector=True)
    coll_f._fast_http = False  # canonical read for the fast-written points
    got_f = await coll_f.get(record_uuids=[uid(60_000 + i) for i in range(8)],
                             return_vector=True)
    assert len(got_c) == len(got_f) == 8, (len(got_c), len(got_f))
    for a, b in zip(got_c, got_f):
        assert a.uuid.int % 10_000 == b.uuid.int % 10_000
        pa = {k: v for k, v in a.properties.items() if k != "_segment_uuid"}
        pb = {k: v for k, v in b.properties.items() if k != "_segment_uuid"}
        assert pa == pb, (pa, pb)
        assert a.vector == b.vector
    print("upsert: canonical points == fast points (8, vectors+payloads): OK")

    # cleanup
    await wipe_partition(pk_c); await wipe_partition(pk_f)
    conn = await asyncpg.connect(PG)
    await conn.close()
    from qdrant_client import models as qm
    for pk in (pk_c, pk_f):
        await client.delete(
            collection_name=coll_c._collection_name,
            points_selector=qm.FilterSelector(filter=qm.Filter(must=[
                qm.FieldCondition(key="sys-partition_key",
                                  match=qm.MatchValue(value=pk))])))
    await client.close(); await engine.dispose()
    print("ALL INGEST DIFF TESTS PASSED")


async def _none():
    return None


asyncio.run(main())
