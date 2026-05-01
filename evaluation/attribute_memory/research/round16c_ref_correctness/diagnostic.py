"""Diagnostic: for each ref-incorrect case in round 15's aen1_active_cap100
run, classify into:

  (a) RIGHT chain head was in active state at write time, but writer picked a
      different chain (or no ref).
  (b) RIGHT chain head was NOT in the active-state block (cap-truncated, or
      entity tag missed, or predicate parse failed).
  (c) Writer emitted a ref to a uuid that doesn't exist in active state at all
      (hallucinated / out-of-snapshot uuid).

We replay the round-15 ingest using the existing cache (free), capturing for
each batch the exact active-state UUID set the writer saw.

Output: results/diagnostic.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND16C = HERE
RESEARCH = ROUND16C.parent
ROUND15 = RESEARCH / "round15_active_chains"
ROUND14 = RESEARCH / "round14_chain_density"
ROUND11 = RESEARCH / "round11_writer_stress"
ROUND7 = RESEARCH / "round7"
sys.path.insert(0, str(ROUND15 / "architectures"))
sys.path.insert(0, str(ROUND14 / "scenarios"))
sys.path.insert(0, str(ROUND14 / "experiments"))
sys.path.insert(0, str(ROUND11 / "architectures"))
sys.path.insert(0, str(ROUND7 / "experiments"))

import aen1_active  # noqa: E402
import aen1_simple  # noqa: E402
import dense_chains  # noqa: E402
import run as r14_run  # noqa: E402
from _common import Budget, Cache  # noqa: E402

RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def replay_with_active_state_capture(
    turns,
    cache,
    budget,
    *,
    batch_size=5,
    rebuild_index_every=4,
    max_active_state_size=100,
):
    """Same ingest logic as aen1_active.ingest_turns, but per-batch we record
    the exact (uuid, predicate, entity) of every active-state head shown to
    the writer at that batch. Returns (log, idx, batch_active_state).

    batch_active_state[batch_no] = {
        "last_turn": int,
        "active_uuids": list of uuids shown in the active-state block,
        "active_heads_by_key": {(entity_tag, predicate): uuid, ...},
        "n_emitted": int,
    }
    """
    pairs = [(t.idx, t.text) for t in turns]
    log: list[aen1_simple.LogEntry] = []
    known: set[str] = {"User"}
    idx = None
    batch_active_state = []

    for batch_no, i in enumerate(range(0, len(pairs), batch_size)):
        batch = pairs[i : i + batch_size]
        # Mirror write_batch: gather the same active-state heads.
        entities = aen1_active.extract_batch_entities(batch)
        heads = aen1_active.gather_active_state(idx, entities, max_active_state_size)
        active_uuids = [e.uuid for e in heads]
        active_heads_by_key: dict[tuple[str, str], str] = {}
        for e in heads:
            if not e.predicate:
                continue
            import re

            m = re.match(r"(@?[A-Za-z0-9_]+)\.(.+)", e.predicate)
            if not m:
                continue
            ent = m.group(1)
            if not ent.startswith("@"):
                ent = "@" + ent
            pred = m.group(2)
            active_heads_by_key[(ent, pred)] = e.uuid

        new_entries, _tele = aen1_active.write_batch(
            batch,
            log,
            idx,
            known,
            cache,
            budget,
            max_active_state_size=max_active_state_size,
        )
        for e in new_entries:
            for m in e.mentions:
                if m.startswith("@"):
                    known.add(m[1:])
        log.extend(new_entries)
        batch_active_state.append(
            {
                "batch_no": batch_no,
                "last_turn": batch[-1][0] if batch else None,
                "first_turn": batch[0][0] if batch else None,
                "active_uuids": active_uuids,
                "active_heads_by_key": {
                    f"{k[0]}.{k[1]}": v for k, v in active_heads_by_key.items()
                },
                "n_emitted": len(new_entries),
            }
        )
        if batch_no % rebuild_index_every == 0:
            idx = aen1_simple.build_index(log, cache, budget)
    idx = aen1_simple.build_index(log, cache, budget)
    return log, idx, batch_active_state


def classify_incorrect(transitions, batch_active_state, by_uuid):
    """For each non-first transition that emitted a ref but ref_correct=False,
    figure out (a/b/c) classification.

    Mapping: each transition has covering_uuid `eXXXX_i`, ts=XXXX. The batch
    that emitted it is the one whose last_turn==ts (or covers ts).
    """
    # Build last_turn -> batch_active_state[i] map
    by_last_turn: dict[int, dict] = {}
    for b in batch_active_state:
        by_last_turn[b["last_turn"]] = b
    # For lookup by ts (covering entry): the writer assigns uuid based on
    # last_turn of the batch — covering.ts == last_turn.

    classifications = []
    cat_counts = {"a": 0, "b": 0, "c": 0, "no_emit": 0, "no_cover": 0}

    for r in transitions:
        if r["is_first"]:
            continue
        if not r["emitted_entry"]:
            classifications.append({**r, "category": "no_cover"})
            cat_counts["no_cover"] += 1
            continue
        if r["ref_correct"]:
            continue  # we only care about errors among non-first
        if not r["emitted_ref"]:
            classifications.append({**r, "category": "no_emit"})
            cat_counts["no_emit"] += 1
            continue

        # ref was emitted but wrong — classify a/b/c
        cov_uuid = r["covering_uuid"]
        # Find the batch whose last_turn == covering.ts
        cov_e = by_uuid.get(cov_uuid)
        if cov_e is None:
            classifications.append(
                {**r, "category": "c", "reason": "covering not in log"}
            )
            cat_counts["c"] += 1
            continue
        batch = by_last_turn.get(cov_e.ts)
        if batch is None:
            # fall back: nearest batch with last_turn >= cov_e.ts
            classifications.append({**r, "category": "c", "reason": "no batch found"})
            cat_counts["c"] += 1
            continue

        active_uuids = set(batch["active_uuids"])
        expected_prev = r["expected_prev_uuid"]
        emitted_refs = r["covering_refs"]

        # Category C: at least one emitted ref is NOT in active_uuids and
        # does not appear in by_uuid up to that point (i.e. fully fabricated).
        # Or: emitted_refs include a uuid that's also not in any prior log.
        all_emitted_in_active = all(u in active_uuids for u in emitted_refs)
        # Hallucination signals:
        any_emitted_unknown = any(u not in by_uuid for u in emitted_refs)

        if expected_prev is None:
            # First transition — shouldn't happen because is_first filter, but
            # just in case.
            classifications.append({**r, "category": "skip"})
            continue

        # Category A: expected_prev IS in active_uuids
        if expected_prev in active_uuids:
            classifications.append(
                {
                    **r,
                    "category": "a",
                    "active_uuids_size": len(active_uuids),
                    "expected_in_active": True,
                    "emitted_refs_in_active": [u in active_uuids for u in emitted_refs],
                    "any_emitted_unknown": any_emitted_unknown,
                }
            )
            cat_counts["a"] += 1
        elif expected_prev in by_uuid:
            # B: head was NOT in active state. Why? Could be:
            #   - cap truncation (size hit max)
            #   - entity not in batch_entities -> head not pulled in
            #   - predicate parse failed (no @entity prefix)
            classifications.append(
                {
                    **r,
                    "category": "b",
                    "active_uuids_size": len(active_uuids),
                    "expected_in_active": False,
                    "emitted_refs_in_active": [u in active_uuids for u in emitted_refs],
                    "any_emitted_unknown": any_emitted_unknown,
                }
            )
            cat_counts["b"] += 1
        else:
            classifications.append(
                {
                    **r,
                    "category": "c",
                    "active_uuids_size": len(active_uuids),
                    "reason": "expected_prev not in by_uuid",
                    "any_emitted_unknown": any_emitted_unknown,
                }
            )
            cat_counts["c"] += 1

    return classifications, cat_counts


def main():
    print("[diagnostic] replaying round-15 cap=100 ingest from cache...")
    turns = dense_chains.generate()
    gt = dense_chains.ground_truth(turns)
    print(f"  turns={len(turns)}")

    # Reuse round 15's cache (only writer + embed calls; should all be hits).
    cache_path = ROUND15 / "cache" / "aen1_active_cap100.json"
    cache = Cache(cache_path)
    # Generous budget; replay should be ~100% cache hits.
    budget = Budget(max_llm=350, max_embed=50, stop_at_llm=350, stop_at_embed=50)

    log, idx, batch_active_state = replay_with_active_state_capture(
        turns,
        cache,
        budget,
        batch_size=5,
        rebuild_index_every=4,
        max_active_state_size=100,
    )
    print(
        f"  log size: {len(log)}, llm calls: {budget.llm_calls}, "
        f"embed calls: {budget.embed_calls}"
    )
    print(f"  cache hits: {cache.hits}, misses: {cache.misses}")
    cache.save()

    metrics = r14_run.collect_metrics(turns, gt, log, bucket_size=100)
    transitions = metrics["transitions"]
    by_uuid = {e.uuid: e for e in log}

    classifications, cat_counts = classify_incorrect(
        transitions,
        batch_active_state,
        by_uuid,
    )

    n_non_first = sum(1 for r in transitions if not r["is_first"])
    n_emitted_entry = sum(
        1 for r in transitions if not r["is_first"] and r["emitted_entry"]
    )
    n_emitted_ref = sum(
        1 for r in transitions if not r["is_first"] and r["emitted_ref"]
    )
    n_correct = sum(1 for r in transitions if not r["is_first"] and r["ref_correct"])
    n_incorrect_emitted = sum(
        1
        for r in transitions
        if not r["is_first"] and r["emitted_ref"] and not r["ref_correct"]
    )

    print(f"\n  non-first transitions: {n_non_first}")
    print(f"  entries emitted: {n_emitted_entry}")
    print(f"  refs emitted: {n_emitted_ref}")
    print(f"  refs correct: {n_correct}")
    print(f"  refs emitted-but-incorrect: {n_incorrect_emitted}")
    print("\n  classification of emitted-but-incorrect refs:")
    print(
        f"    (a) right head was in active state, writer picked wrong: "
        f"{cat_counts['a']}"
    )
    print(f"    (b) right head was NOT in active state: {cat_counts['b']}")
    print(f"    (c) hallucinated/unknown ref uuid: {cat_counts['c']}")
    if n_incorrect_emitted:
        print(
            f"    pct: a={cat_counts['a'] / n_incorrect_emitted:.1%}  "
            f"b={cat_counts['b'] / n_incorrect_emitted:.1%}  "
            f"c={cat_counts['c'] / n_incorrect_emitted:.1%}"
        )
    print(f"  no-cover (no covering entry): {cat_counts['no_cover']}")
    print(f"  no-emit (covering had no ref): {cat_counts['no_emit']}")

    # Drill into category (a) - what predicate mismatches happen?
    cat_a = [c for c in classifications if c.get("category") == "a"]
    print(
        f"\n  Category-A predicate analysis "
        f"(right head in active, picked wrong): {len(cat_a)}"
    )
    pred_pairs = {}
    for c in cat_a[:30]:
        cov_pred = c.get("covering_predicate")
        gt_key = c.get("key")
        emitted_refs = c.get("covering_refs") or []
        # Try to find what predicate the emitted ref's head has
        first_emitted = emitted_refs[0] if emitted_refs else None
        emit_pred = None
        if first_emitted and first_emitted in by_uuid:
            emit_pred = by_uuid[first_emitted].predicate
        pair = f"new_pred={cov_pred} -> wrong_ref_pred={emit_pred}"
        pred_pairs[pair] = pred_pairs.get(pair, 0) + 1
    for k, v in sorted(pred_pairs.items(), key=lambda x: -x[1])[:15]:
        print(f"    {v:3d}x  {k}")

    # Same for category (b) - why was head missing?
    cat_b = [c for c in classifications if c.get("category") == "b"]
    print(f"\n  Category-B head-missing analysis: {len(cat_b)}")
    cap_hit = 0
    for c in cat_b:
        if c.get("active_uuids_size", 0) >= 100:
            cap_hit += 1
    print(f"    of those, batch had size>=cap(100): {cap_hit}")

    # Save raw diagnostic data
    out = {
        "n_non_first": n_non_first,
        "n_emitted_entry": n_emitted_entry,
        "n_emitted_ref": n_emitted_ref,
        "n_correct": n_correct,
        "n_incorrect_emitted": n_incorrect_emitted,
        "category_counts": cat_counts,
        "category_pcts": {
            "a": cat_counts["a"] / n_incorrect_emitted if n_incorrect_emitted else None,
            "b": cat_counts["b"] / n_incorrect_emitted if n_incorrect_emitted else None,
            "c": cat_counts["c"] / n_incorrect_emitted if n_incorrect_emitted else None,
        },
        "predicate_mismatch_pairs": pred_pairs,
        "category_b_cap_hit_count": cap_hit,
        "classifications_sample": classifications[:60],
        "batch_active_state_summary": {
            "n_batches": len(batch_active_state),
            "max_active_uuids": max(len(b["active_uuids"]) for b in batch_active_state),
            "avg_active_uuids": sum(len(b["active_uuids"]) for b in batch_active_state)
            / len(batch_active_state),
        },
    }
    out_path = RESULTS_DIR / "diagnostic.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
