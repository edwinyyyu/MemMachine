"""Compaction strategies for append-only topic logs.

A compaction strategy takes a list of log entries and returns a rendered
string (the "compacted log" that the reader LLM sees). Some strategies are
cheap (pure Python) and some invoke LLM/embedding calls.

Strategies:
  C1  truncation_last_k   -- keep last K entries verbatim.
  C2  middle_elision      -- first K_a + "... (N elided) ..." + last K_b.
  C3  hierarchical_summ   -- recent verbatim + LLM-summarized middle + old headlines.
  C4  query_gated         -- top-K by embedding similarity + last K'.
  C5  active_consolidate  -- periodic LLM rewrite of oldest half into one entry.
  C6  relation_compact    -- drop INVALIDATED entries, collapse clarify-chains
                             by keeping only the last clarify per chain, no LLM.
  C7  c6_plus_c3          -- C6 first, then hierarchical summarization.

Entry shape:
  {id, day, topic, text, refs: [int], relation: str|None}

Compacted output shape: string ready to stuff into a reader-LLM prompt.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

# ----------------------------- rendering -----------------------------


def render_entry(e: dict[str, Any], include_status: bool = True) -> str:
    tag = ""
    if e.get("relation"):
        rel = e["relation"]
        refs = e.get("refs", [])
        tag = f" [{rel} of {','.join(str(r) for r in refs)}]"
    status = ""
    if include_status and e.get("invalidated"):
        status = " [INVALIDATED]"
    return f"[{e['id']}] (d{e['day']} {e['topic']}) {e['text']}{tag}{status}"


def render_entries(entries: list[dict[str, Any]]) -> str:
    return "\n".join(render_entry(e) for e in entries)


# ----------------------------- structural helpers -----------------------------


def mark_invalidated(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a copy of entries with an `invalidated` flag computed from
    supersede/invalidate relations. supersede also invalidates the referenced
    entry (the superseding entry itself is live).
    """
    out = [dict(e) for e in entries]
    by_id = {e["id"]: e for e in out}
    for e in out:
        rel = e.get("relation")
        if rel in ("supersede", "invalidate"):
            for r in e.get("refs", []):
                if r in by_id:
                    by_id[r]["invalidated"] = True
    for e in out:
        e.setdefault("invalidated", False)
    return out


def collapse_clarify_chains(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the *latest* clarify per root target per topic. Older clarifies
    that point to the same root (transitively) are dropped. Non-clarify entries
    and the latest clarify remain.

    Root: follow refs back through clarify relations until we hit a non-clarify
    ancestor. If multiple refs, use the first.
    """
    by_id = {e["id"]: e for e in entries}

    def root_of(eid: int) -> int:
        seen = set()
        cur = eid
        while cur in by_id:
            if cur in seen:
                break
            seen.add(cur)
            e = by_id[cur]
            if e.get("relation") == "clarify" and e.get("refs"):
                cur = e["refs"][0]
            else:
                return cur
        return cur

    # For each root, find the latest clarify entry (the one with max id that
    # clarifies into that root, directly or transitively).
    clarify_by_root: dict[int, int] = {}
    for e in entries:
        if e.get("relation") == "clarify" and e.get("refs"):
            r = root_of(e["refs"][0])
            prev = clarify_by_root.get(r, -1)
            if e["id"] > prev:
                clarify_by_root[r] = e["id"]

    keep_ids: set[int] = set()
    for e in entries:
        rel = e.get("relation")
        if rel == "clarify" and e.get("refs"):
            r = root_of(e["refs"][0])
            # Keep only if this is the latest clarify for that root
            if clarify_by_root.get(r) == e["id"]:
                keep_ids.add(e["id"])
        else:
            keep_ids.add(e["id"])

    return [e for e in entries if e["id"] in keep_ids]


def drop_invalidated(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop entries flagged invalidated (via mark_invalidated)."""
    return [e for e in entries if not e.get("invalidated")]


# ----------------------------- C1: truncation_last_k -----------------------------


def c1_truncation(entries: list[dict[str, Any]], k: int = 20) -> tuple[str, dict]:
    kept = entries[-k:] if len(entries) > k else entries
    elided = len(entries) - len(kept)
    rendered = render_entries(kept)
    meta = {"kept": len(kept), "elided": elided, "llm_calls": 0, "embed_calls": 0}
    return rendered, meta


# ----------------------------- C2: middle_elision -----------------------------


def c2_middle_elision(
    entries: list[dict[str, Any]], k_a: int = 10, k_b: int = 20
) -> tuple[str, dict]:
    if len(entries) <= k_a + k_b:
        return render_entries(entries), {
            "kept": len(entries),
            "elided": 0,
            "llm_calls": 0,
            "embed_calls": 0,
        }
    first = entries[:k_a]
    last = entries[-k_b:]
    middle = len(entries) - k_a - k_b
    parts = [
        render_entries(first),
        f"... ({middle} entries elided) ...",
        render_entries(last),
    ]
    rendered = "\n".join(parts)
    meta = {"kept": k_a + k_b, "elided": middle, "llm_calls": 0, "embed_calls": 0}
    return rendered, meta


# ----------------------------- C3: hierarchical_summ -----------------------------

HIER_SUMMARY_PROMPT = """You are compacting an append-only log of user facts.

Below is a slice of the log (oldest-to-newest). Summarize it into 3-6 short bullet points that preserve:
- the FINAL state of each fact (what remains true) after all clarify/refine/supersede/invalidate relations in this slice
- the fact that certain things were changed or retracted (mention by attribute, e.g., "previously rented in Ballard; moved")
- any durable identifying details (names, places, numbers)

Do NOT invent details. Each bullet should be self-contained. Be brief: <=20 words per bullet.

LOG SLICE:
{slice}

Return ONLY the bullet points, one per line, prefixed with "- ".
"""

HEADLINE_PROMPT = """You are compacting very old log entries that have been superseded or clarified by more recent ones.

Each entry below may be outdated. In one line per entry, write a minimal headline that preserves the TOPIC and any durable identifier (name/place/number), so the entry is recoverable by a later reader if asked. Prefer <=10 words per line.

ENTRIES:
{slice}

Return one line per entry, prefixed with the entry id in brackets, e.g., "[3] ...".
"""


def c3_hierarchical(
    entries: list[dict[str, Any]],
    recent_k: int = 15,
    headline_k: int = 5,
    llm_call_fn=None,
) -> tuple[str, dict]:
    """
    - oldest headline_k entries -> LLM headlines (one line each)
    - middle -> LLM bullet summary
    - recent recent_k entries -> verbatim
    """
    llm_calls = 0
    if len(entries) <= recent_k + headline_k:
        return render_entries(entries), {
            "kept": len(entries),
            "elided": 0,
            "llm_calls": 0,
            "embed_calls": 0,
        }
    oldest = entries[:headline_k]
    recent = entries[-recent_k:]
    middle = entries[headline_k:-recent_k]

    parts: list[str] = []
    # Oldest headlines
    if oldest:
        if llm_call_fn:
            raw = llm_call_fn(HEADLINE_PROMPT.format(slice=render_entries(oldest)))
            llm_calls += 1
            parts.append("# Old entries (headlines)\n" + raw.strip())
        else:
            parts.append("# Old entries (headlines)\n" + render_entries(oldest))

    # Middle summary
    if middle:
        if llm_call_fn:
            raw = llm_call_fn(HIER_SUMMARY_PROMPT.format(slice=render_entries(middle)))
            llm_calls += 1
            parts.append(
                f"# Middle period (days {middle[0]['day']}-{middle[-1]['day']}, "
                f"{len(middle)} entries summarized)\n" + raw.strip()
            )
        else:
            parts.append("# Middle period\n" + render_entries(middle))

    # Recent verbatim
    parts.append("# Recent (verbatim)\n" + render_entries(recent))
    rendered = "\n\n".join(parts)
    meta = {
        "kept": len(entries),  # all are represented in some form
        "elided": 0,
        "llm_calls": llm_calls,
        "embed_calls": 0,
    }
    return rendered, meta


# ----------------------------- C4: query_gated -----------------------------


def c4_query_gated(
    entries: list[dict[str, Any]],
    query: str,
    entry_embeds: np.ndarray,
    query_embed: np.ndarray,
    top_k: int = 10,
    recent_k: int = 10,
) -> tuple[str, dict]:
    """Retrieve top-K by cosine similarity, + recent_k most recent entries (union)."""
    # Normalize
    a = entry_embeds / (np.linalg.norm(entry_embeds, axis=1, keepdims=True) + 1e-9)
    q = query_embed / (np.linalg.norm(query_embed) + 1e-9)
    sims = a @ q  # [N]
    topk_idx = np.argsort(-sims)[:top_k]
    topk_set = set(int(i) for i in topk_idx)

    recent_idx = set(range(max(0, len(entries) - recent_k), len(entries)))

    selected = sorted(topk_set | recent_idx)
    kept = [entries[i] for i in selected]
    rendered = (
        f"# Query-gated view for: {query!r}\n"
        f"# Showing top-{top_k} relevant + last {recent_k} entries ({len(kept)} total)\n"
        + render_entries(kept)
    )
    meta = {
        "kept": len(kept),
        "elided": len(entries) - len(kept),
        "llm_calls": 0,
        "embed_calls": 0,  # accounted at index-build time
    }
    return rendered, meta


# ----------------------------- C5: active_consolidate -----------------------------

CONSOLIDATE_PROMPT = """You are actively consolidating the OLDEST HALF of an append-only user-facts log to save space.

Below are the oldest entries. Rewrite them as 4-8 concise bullet points that:
- PRESERVE all durable facts that are still live (not later superseded/invalidated). Include specifics.
- NOTE things that were superseded or invalidated, so a later reader knows the change happened. Example: "previously lived in Seattle; moved".
- Do NOT invent details or synthesize new claims.
- Each bullet <=25 words.

OLD ENTRIES:
{slice}

Return ONLY bullet points prefixed with "- ". These will REPLACE the original entries in the log.
"""


def c5_active_consolidate(
    entries: list[dict[str, Any]],
    trigger_size: int = 40,
    llm_call_fn=None,
) -> tuple[str, dict]:
    """If log >= trigger_size, rewrite the oldest HALF into a summary entry
    and keep the recent half verbatim. Simulates having consolidated once.

    Returns rendered compacted log.
    """
    llm_calls = 0
    if len(entries) < trigger_size or not llm_call_fn:
        return render_entries(entries), {
            "kept": len(entries),
            "elided": 0,
            "llm_calls": 0,
            "embed_calls": 0,
        }
    half = len(entries) // 2
    oldest = entries[:half]
    recent = entries[half:]
    raw = llm_call_fn(CONSOLIDATE_PROMPT.format(slice=render_entries(oldest)))
    llm_calls += 1
    summary_block = (
        f"# Consolidated (was {len(oldest)} entries spanning days "
        f"{oldest[0]['day']}-{oldest[-1]['day']})\n" + raw.strip()
    )
    rendered = summary_block + "\n\n# Recent (verbatim)\n" + render_entries(recent)
    meta = {
        "kept": len(recent),
        "elided": len(oldest),  # folded into summary
        "llm_calls": llm_calls,
        "embed_calls": 0,
    }
    return rendered, meta


# ----------------------------- C6: relation_compact -----------------------------


def c6_relation_compact(entries: list[dict[str, Any]]) -> tuple[str, dict]:
    """Pure structural compaction:
      1. mark entries invalidated via supersede/invalidate
      2. drop invalidated entries entirely
      3. collapse clarify chains to keep only the latest clarify per root
    No LLM. No embedding.
    """
    marked = mark_invalidated(entries)
    not_inv = drop_invalidated(marked)
    collapsed = collapse_clarify_chains(not_inv)
    rendered = render_entries(collapsed)
    meta = {
        "kept": len(collapsed),
        "elided": len(entries) - len(collapsed),
        "llm_calls": 0,
        "embed_calls": 0,
    }
    return rendered, meta


# ----------------------------- C7: C6 + C3 -----------------------------


def c7_hybrid(
    entries: list[dict[str, Any]],
    recent_k: int = 15,
    headline_k: int = 5,
    llm_call_fn=None,
) -> tuple[str, dict]:
    """Apply C6 structural compaction, then if still too long, summarize the
    middle with C3."""
    marked = mark_invalidated(entries)
    not_inv = drop_invalidated(marked)
    collapsed = collapse_clarify_chains(not_inv)
    rendered, meta = c3_hierarchical(
        collapsed, recent_k=recent_k, headline_k=headline_k, llm_call_fn=llm_call_fn
    )
    meta["elided"] = len(entries) - meta["kept"]
    return rendered, meta


# ----------------------------- full log (baseline) -----------------------------


def c0_full(entries: list[dict[str, Any]]) -> tuple[str, dict]:
    return render_entries(entries), {
        "kept": len(entries),
        "elided": 0,
        "llm_calls": 0,
        "embed_calls": 0,
    }


# ----------------------------- registry -----------------------------


@dataclass
class Strategy:
    key: str
    name: str
    requires_query: bool
    requires_llm: bool
    fn: Callable


STRATEGIES: list[Strategy] = [
    # C0_full is kept available via c0_full but NOT in the default run to save budget.
    # It establishes the ceiling: if answers aren't there in the full log, they can't
    # be in any compaction of it. The gold keywords serve as the ceiling oracle.
    Strategy(
        "C1_trunc_last20",
        "Truncation last-20",
        False,
        False,
        lambda e, **kw: c1_truncation(e, k=20),
    ),
    Strategy(
        "C2_middle_elision",
        "Middle-elision (10 first + 20 last)",
        False,
        False,
        lambda e, **kw: c2_middle_elision(e, k_a=10, k_b=20),
    ),
    Strategy(
        "C3_hierarchical",
        "Hierarchical summarization (LLM)",
        False,
        True,
        lambda e, llm_call_fn=None, **kw: c3_hierarchical(
            e, recent_k=15, headline_k=5, llm_call_fn=llm_call_fn
        ),
    ),
    Strategy(
        "C4_query_gated", "Query-gated top-10 + recent-10 (embed)", True, False, None
    ),  # computed ad-hoc in evaluate.py
    Strategy(
        "C5_active_consolidate",
        "Active consolidation (LLM rewrite oldest half)",
        False,
        True,
        lambda e, llm_call_fn=None, **kw: c5_active_consolidate(
            e, trigger_size=40, llm_call_fn=llm_call_fn
        ),
    ),
    Strategy(
        "C6_relation_compact",
        "Relation-aware compaction (no LLM)",
        False,
        False,
        lambda e, **kw: c6_relation_compact(e),
    ),
    Strategy(
        "C7_hybrid_C6_C3",
        "C6 structural then C3 summarization",
        False,
        True,
        lambda e, llm_call_fn=None, **kw: c7_hybrid(
            e, recent_k=15, headline_k=5, llm_call_fn=llm_call_fn
        ),
    ),
]
