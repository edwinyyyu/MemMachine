"""#135 diagnosis: how does the gpt-5-nano segmenter differ from gpt-5-mini?

The gap: nano is ~1pp less accurate AND +42 tok/q wordier. Compare what
the two segmenters actually emit on the IDENTICAL source, to localize
where the slim_v3 prompt is under-applied by the weaker model.

Reads two slim_v3 segment DBs (nano vs mini), aligns segments by source
message (group_idx, dia_id), and reports aggregate stats + side-by-side
examples of the terse field.
"""

from __future__ import annotations

import json
import sqlite3
import statistics

from memmachine_server.episodic_memory.event_memory.data_types import (
    decode_block,
    decode_context,
)

NANO_DB = "locomo-tslimv3-5n-m.sqlite"
MINI_DB = "locomo-tslimv3-5m-m.sqlite"


def load(db: str) -> dict[tuple, list[dict]]:
    """Map (group_idx, dia_id) -> list of {terse, memory, queries}."""
    con = sqlite3.connect(db)
    cols = [r[1] for r in con.execute("PRAGMA table_info(segment_store_sg)")]
    by_key: dict[tuple, list[dict]] = {}
    for row in con.execute("SELECT * FROM segment_store_sg"):
        d = dict(zip(cols, row))
        blk = decode_block(json.loads(d["block"].decode()))
        ctx = decode_context(json.loads(d["context"].decode()))
        props = json.loads(d["properties"]) if d.get("properties") else {}
        # Property values are wrapped as {"v": value, "t": type}.
        def _val(name: str):
            p = props.get(name)
            return p["v"] if isinstance(p, dict) and "v" in p else p
        key = (_val("group_idx"), _val("dia_id"))
        by_key.setdefault(key, []).append(
            {"terse": blk.text, "embed": ctx.text_to_embed}
        )
    con.close()
    return by_key


def stats(by_key: dict[tuple, list[dict]], label: str) -> None:
    segs = [s for v in by_key.values() for s in v]
    terse_words = [len(s["terse"].split()) for s in segs]
    embed_words = [len(s["embed"].split()) for s in segs]
    print(f"--- {label} ---")
    print(f"  source messages : {len(by_key)}")
    print(f"  segments        : {len(segs)}")
    print(f"  segs/message    : {len(segs) / max(len(by_key), 1):.2f}")
    print(f"  terse words     : mean {statistics.mean(terse_words):.1f}  "
          f"median {statistics.median(terse_words)}  "
          f"p90 {sorted(terse_words)[int(len(terse_words) * 0.9)]}")
    print(f"  embed words     : mean {statistics.mean(embed_words):.1f}")
    print()


def main() -> None:
    nano = load(NANO_DB)
    mini = load(MINI_DB)
    stats(nano, "gpt-5-nano @ medium")
    stats(mini, "gpt-5-mini @ medium")

    # Side-by-side terse on shared source messages: the widest
    # nano-minus-mini word-count gaps (where nano is most over-wordy).
    shared = sorted(set(nano) & set(mini))
    rows = []
    for key in shared:
        nw = sum(len(s["terse"].split()) for s in nano[key])
        mw = sum(len(s["terse"].split()) for s in mini[key])
        rows.append((nw - mw, key, nw, mw))
    rows.sort(reverse=True)
    print("=== widest nano-wordier source messages ===")
    for gap, key, nw, mw in rows[:6]:
        print(f"\n[{key}]  nano {nw}w  vs  mini {mw}w  (+{gap})")
        print("  NANO terse:")
        for s in nano[key]:
            print(f"    - {s['terse']}")
        print("  MINI terse:")
        for s in mini[key]:
            print(f"    - {s['terse']}")


if __name__ == "__main__":
    main()
