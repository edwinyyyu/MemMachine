"""Episode-store add fast path: canonical vs fast equality + fetch timing."""
import asyncio, hashlib, time
from datetime import UTC, datetime, timedelta, timezone
import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine
import os

import memmachine_server
# Guard against benching the wrong build: when BENCH_EXPECT_PATH_SUBSTR is
# set, refuse to run unless the imported memmachine_server comes from a
# path containing it (e.g. the worktree name of the branch under test).
_expect = os.environ.get("BENCH_EXPECT_PATH_SUBSTR")
if _expect:
    assert _expect in memmachine_server.__file__, memmachine_server.__file__
from memmachine_server.common.episode_store.episode_sqlalchemy_store import SqlAlchemyEpisodeStore
from memmachine_server.common.episode_store.episode_model import EpisodeEntry, EpisodeType

PG = "postgresql://memmachine:memmachine@localhost:5442/memmachine"
PGA = "postgresql+asyncpg://memmachine:memmachine@localhost:5442/memmachine"
TZ = timezone(timedelta(hours=-7))
KEY_C, KEY_F = "difftest/ep_canon", "difftest/ep_fast"

def entries(n):
    return [EpisodeEntry(
        content=f"episode content {i} jeżyk",
        producer_id=f"prod{i%3}", producer_role="user",
        produced_for_id="assistant" if i % 2 else None,
        created_at=datetime(2026, 4, 1, 10, 0, i, tzinfo=TZ),
        metadata={"k": i, "nested": {"a": [1, i]}} if i % 2 else None,
        episode_type=EpisodeType.MESSAGE if i % 3 else None,
    ) for i in range(12)]

async def main():
    conn = await asyncpg.connect(PG)
    await conn.execute("DELETE FROM episodestore WHERE session_key = ANY($1)", [KEY_C, KEY_F])
    await conn.close()
    engine = create_async_engine(PGA, pool_size=4)
    store = SqlAlchemyEpisodeStore(engine)

    canon_store = SqlAlchemyEpisodeStore(engine)
    async def none_fast(*a, **k): return None
    canon_store._add_episodes_fast = none_fast

    ec = await canon_store.add_episodes(KEY_C, entries(12))
    ef = await store.add_episodes(KEY_F, entries(12))
    assert len(ec) == len(ef) == 12
    for a, b in zip(ec, ef):
        da, db = a.model_dump(), b.model_dump()
        da.pop("uid"); db.pop("uid")  # autoincrement ids differ
        da.pop("session_key"); db.pop("session_key")
        assert da == db, (da, db)
    print("add_episodes: canonical == fast (12 rows, tz/metadata/enum): OK")

    # raw row comparison
    conn = await asyncpg.connect(PG)
    rc = await conn.fetch("SELECT content, producer_id, producer_role, produced_for_id, episode_type, created_at, \"metadata\" FROM episodestore WHERE session_key=$1 ORDER BY id", KEY_C)
    rf = await conn.fetch("SELECT content, producer_id, producer_role, produced_for_id, episode_type, created_at, \"metadata\" FROM episodestore WHERE session_key=$1 ORDER BY id", KEY_F)
    import json
    for a, b in zip(rc, rf):
        for col in a.keys():
            va, vb = a[col], b[col]
            if col == "metadata" and isinstance(va, str):
                va, vb = va and json.loads(va), vb and json.loads(vb)
            assert va == vb, (col, va, vb)
    print("add_episodes: raw rows identical column-by-column: OK")

    # fetch timing: canonical vs fast, same 10 uids
    uids = [e.uid for e in ef[:10]]
    async def timeit(s, n=300):
        for _ in range(30): await s.get_episodes(uids)
        t0 = time.process_time()
        for _ in range(n): await s.get_episodes(uids)
        return (time.process_time() - t0) * 1000 / n
    fast_ms = await timeit(store)
    canon_ms = await timeit(canon_store) if True else 0
    # canonical fetch: force fallback
    canon_store._get_episodes_fast = none_fast
    canon_ms = await timeit(canon_store)
    print(f"episode fetch 10 rows: canonical {canon_ms:.3f} core-ms vs fast {fast_ms:.3f} core-ms")

    conn2 = await asyncpg.connect(PG)
    await conn2.execute("DELETE FROM episodestore WHERE session_key = ANY($1)", [KEY_C, KEY_F])
    await conn2.close(); await conn.close(); await engine.dispose()
    print("EPISODE ADD DIFF TESTS PASSED")

asyncio.run(main())
