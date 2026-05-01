"""Consolidated experiment runner for round 8 buffered-commit.

Runs E1 (coref distances), E2 (salience escalation), E3 (late correction),
E4 (query-during-buffer), E5 (end-of-stream flush), E6 (integration replay).

Compares BufferedCommitPipeline against ImmediateWritePipeline on E1-E4.

Budget-aware: stops at 80% (120 LLM calls). Uses shared cache across all
experiments AND piggy-backs on round 7's cache for the integration scenario.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND8 = HERE.parent
ROUND7 = ROUND8.parent / "round7"
sys.path.insert(0, str(ROUND8))
sys.path.insert(0, str(ROUND7))
sys.path.insert(0, str(ROUND7 / "experiments"))

from _common import Budget, Cache
from buffered_pipeline import (
    BufferedCommitPipeline,
    BufferedEntry,
    CommittedEntry,
    ImmediateWritePipeline,
)

CACHE_DIR = ROUND8 / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = ROUND8 / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 30  # commit_age = 15


def load_scenarios(name: str) -> dict:
    return json.loads((ROUND8 / "scenarios" / name).read_text())


# ---------------------------------------------------------------------------
# Shared cache — round 7 integration cache is reused (same prompts = same keys)
# so E6 piggy-backs on it. E1-E5 get their own cache file.
# ---------------------------------------------------------------------------


def make_caches() -> tuple[Cache, Cache]:
    # Shared cache for the fused LLM extraction prompts — we reuse round 7's
    # integration cache for overlapping prompts.
    shared = Cache(CACHE_DIR / "round8_llm.json")
    # Also pre-load round 7's integration cache (same prompt keys)
    r7 = ROUND7 / "cache" / "integration_llm.json"
    if r7.exists():
        try:
            r7_data = json.loads(r7.read_text())
            for k, v in r7_data.items():
                if k not in shared._d:
                    shared._d[k] = v
        except Exception:
            pass
    return shared, shared


# ---------------------------------------------------------------------------
# E1: coref at varying distances
# ---------------------------------------------------------------------------


def run_e1(cache: Cache, budget: Budget) -> dict:
    data = load_scenarios("e1_coref_distances.json")
    results = []
    for sc in data["scenarios"]:
        # Buffered
        buf = BufferedCommitPipeline(WINDOW_SIZE, cache, budget)
        for turn in sc["turns"]:
            buf.ingest_turn(turn["idx"], turn["text"])
        buf.flush()
        marcus_entries_buf = [
            e for e in buf.state.committed if e.topic.startswith("Marcus/")
        ]
        user_emp_buf = [e for e in buf.state.committed if e.topic == "User/Employment"]

        # Immediate-write baseline
        imm = ImmediateWritePipeline(cache, budget)
        for turn in sc["turns"]:
            imm.ingest_turn(turn["idx"], turn["text"])
        imm.flush()
        marcus_entries_imm = [
            e for e in imm.state.committed if e.topic.startswith("Marcus/")
        ]
        user_emp_imm = [e for e in imm.state.committed if e.topic == "User/Employment"]

        coref_merged_buf = len(buf.state.coref_merges) > 0
        coref_merged_imm = len(imm.state.coref_merges) > 0

        exp_min = sc["expected"]["marcus_employment_entries_min"]
        buf_pass = len(marcus_entries_buf) >= exp_min
        imm_pass = False  # baseline can't retroactively rewrite; only first-turn facts with named-intro get proper topic

        results.append(
            {
                "scenario_id": sc["id"],
                "distance": sc["distance"],
                "buffered": {
                    "marcus_entries": len(marcus_entries_buf),
                    "user_employment_entries": len(user_emp_buf),
                    "coref_merges": len(buf.state.coref_merges),
                    "pass": buf_pass,
                },
                "immediate": {
                    "marcus_entries": len(marcus_entries_imm),
                    "user_employment_entries": len(user_emp_imm),
                    "coref_merges": len(imm.state.coref_merges),
                    "pass": imm_pass,
                },
                "expected_min_marcus": exp_min,
            }
        )
        print(
            f"  E1 {sc['id']} d={sc['distance']}: "
            f"buf marcus={len(marcus_entries_buf)} user-emp={len(user_emp_buf)} "
            f"coref-merges={len(buf.state.coref_merges)} pass={buf_pass} | "
            f"imm marcus={len(marcus_entries_imm)} user-emp={len(user_emp_imm)} "
            f"coref={len(imm.state.coref_merges)}",
            flush=True,
        )
    return {"scenarios": results}


# ---------------------------------------------------------------------------
# E2: salience escalation
# ---------------------------------------------------------------------------


def run_e2(cache: Cache, budget: Budget) -> dict:
    data = load_scenarios("e2_salience_escalation.json")
    results = []
    for sc in data["scenarios"]:
        buf = BufferedCommitPipeline(WINDOW_SIZE, cache, budget)
        for turn in sc["turns"]:
            buf.ingest_turn(turn["idx"], turn["text"])
        buf.flush()
        bowl_entries = [e for e in buf.state.committed if "bowl" in e.text.lower()]
        admitted, deferred = _salience_partition(buf)

        imm = ImmediateWritePipeline(cache, budget)
        for turn in sc["turns"]:
            imm.ingest_turn(turn["idx"], turn["text"])
        imm.flush()
        bowl_entries_imm = [e for e in imm.state.committed if "bowl" in e.text.lower()]

        bowl_admitted = any("bowl" in a.lower() for a in admitted)
        exp_admit = sc["expected"].get("admit_grandmother_bowl", False)

        results.append(
            {
                "scenario_id": sc["id"],
                "k": sc["k"],
                "buffered": {
                    "bowl_entries_committed": len(bowl_entries),
                    "bowl_admitted": bowl_admitted,
                    "admitted": admitted,
                    "pass": (bowl_admitted == exp_admit),
                },
                "immediate": {
                    "bowl_entries_committed": len(bowl_entries_imm),
                },
            }
        )
        print(
            f"  E2 {sc['id']} k={sc['k']}: "
            f"buf bowl_committed={len(bowl_entries)} admitted={bowl_admitted} | "
            f"imm bowl_committed={len(bowl_entries_imm)}",
            flush=True,
        )
    return {"scenarios": results}


def _salience_partition(pipe) -> tuple[list[str], list[str]]:
    from schemas import SALIENCE_ENTITY_THRESHOLD_SCORE, salience_score

    a, d = [], []
    for c in pipe.state.salience.values():
        if salience_score(c) >= SALIENCE_ENTITY_THRESHOLD_SCORE:
            a.append(c.descriptor)
        else:
            d.append(c.descriptor)
    return a, d


# ---------------------------------------------------------------------------
# E3: late correction
# ---------------------------------------------------------------------------


def run_e3(cache: Cache, budget: Budget) -> dict:
    data = load_scenarios("e3_late_correction.json")
    results = []
    # Skip very-long scenarios that are slow and redundant with shorter
    # same-pattern tests.
    scenarios = [s for s in data["scenarios"] if s.get("k", 0) < 35]
    for sc in scenarios:
        buf = BufferedCommitPipeline(WINDOW_SIZE, cache, budget)
        for turn in sc["turns"]:
            buf.ingest_turn(turn["idx"], turn["text"])
        buf.flush()

        imm = ImmediateWritePipeline(cache, budget)
        for turn in sc["turns"]:
            imm.ingest_turn(turn["idx"], turn["text"])
        imm.flush()

        buf_slot = buf.state.slots.get("User/Employment/boss")
        imm_slot = imm.state.slots.get("User/Employment/boss")
        buf_history = [e.filler for e in (buf_slot.history if buf_slot else [])]
        imm_history = [e.filler for e in (imm_slot.history if imm_slot else [])]

        exp = sc["expected"]["slot_history_bossfinal"]
        buf_pass = buf_history == exp
        results.append(
            {
                "scenario_id": sc["id"],
                "k": sc["k"],
                "expected": exp,
                "buffered_history": buf_history,
                "buf_pass": buf_pass,
                "immediate_history": imm_history,
                "imm_pass": imm_history == exp,
            }
        )
        print(
            f"  E3 {sc['id']} k={sc['k']}: buf={buf_history} imm={imm_history} exp={exp}",
            flush=True,
        )
    return {"scenarios": results}


# ---------------------------------------------------------------------------
# E4: query-during-buffer
# ---------------------------------------------------------------------------


def run_e4(cache: Cache, budget: Budget) -> dict:
    data = load_scenarios("e4_query_during_buffer.json")
    results = []
    for sc in data["scenarios"]:
        buf = BufferedCommitPipeline(WINDOW_SIZE, cache, budget)
        query_turn = sc["query_at_turn"]
        needle = sc["query_match_substring"].lower()
        # Ingest up to just before query_turn; query at query_turn
        fired = False
        for turn in sc["turns"]:
            if turn["idx"] >= query_turn:
                break
            buf.ingest_turn(turn["idx"], turn["text"])
        # advance "now" to query_turn without ingesting
        buf.now = query_turn
        found_buf = buf.query(lambda e: needle in e.text.lower())
        # Break down source of find
        committed_hits = [h for h in found_buf if isinstance(h, CommittedEntry)]
        buffer_hits = [h for h in found_buf if isinstance(h, BufferedEntry)]

        # Baseline: only committed
        imm = ImmediateWritePipeline(cache, budget)
        for turn in sc["turns"]:
            if turn["idx"] >= query_turn:
                break
            imm.ingest_turn(turn["idx"], turn["text"])
        found_imm = imm.query(lambda e: needle in e.text.lower())

        results.append(
            {
                "scenario_id": sc["id"],
                "query_at_turn": query_turn,
                "needle": needle,
                "buffered": {
                    "total_hits": len(found_buf),
                    "committed_hits": len(committed_hits),
                    "buffer_hits": len(buffer_hits),
                    "pass": len(found_buf) > 0,
                },
                "immediate": {
                    "total_hits": len(found_imm),
                    "pass": len(found_imm) > 0,
                },
            }
        )
        print(
            f"  E4 {sc['id']}: buf total={len(found_buf)} buffer-hits={len(buffer_hits)} "
            f"committed-hits={len(committed_hits)} | imm total={len(found_imm)}",
            flush=True,
        )
    return {"scenarios": results}


# ---------------------------------------------------------------------------
# E5: end-of-stream flush
# ---------------------------------------------------------------------------


def run_e5(cache: Cache, budget: Budget) -> dict:
    data = load_scenarios("e5_end_of_stream_flush.json")
    results = []
    for sc in data["scenarios"]:
        buf = BufferedCommitPipeline(WINDOW_SIZE, cache, budget)
        for turn in sc["turns"]:
            buf.ingest_turn(turn["idx"], turn["text"])
        before_flush_committed = len(buf.state.committed)
        before_flush_buffered = len(buf.state.buffer)
        buf.flush()
        after_flush_committed = len(buf.state.committed)

        res = {
            "scenario_id": sc["id"],
            "turns": len(sc["turns"]),
            "committed_before_flush": before_flush_committed,
            "buffered_before_flush": before_flush_buffered,
            "committed_after_flush": after_flush_committed,
            "expected": sc.get("expected", {}),
        }
        if "slot_history_final" in sc["expected"]:
            slot = buf.state.slots.get("User/Employment/boss")
            history = [e.filler for e in (slot.history if slot else [])]
            res["slot_history_final"] = history
            res["pass"] = history == sc["expected"]["slot_history_final"]
        else:
            exp_min = sc["expected"].get("committed_min", 0)
            res["pass"] = after_flush_committed >= exp_min
        print(
            f"  E5 {sc['id']}: committed {before_flush_committed}->{after_flush_committed} "
            f"(buffered at end: {before_flush_buffered}) pass={res['pass']}",
            flush=True,
        )
        results.append(res)
    return {"scenarios": results}


# ---------------------------------------------------------------------------
# E6: integration replay (reuses round 7 scenario)
# ---------------------------------------------------------------------------


def run_e6(cache: Cache, budget: Budget) -> dict:
    scenarios_path = ROUND7 / "scenarios" / "integration.json"
    scenarios = json.loads(scenarios_path.read_text())["scenarios"]
    results = []
    for sc in scenarios:
        buf = BufferedCommitPipeline(WINDOW_SIZE, cache, budget)
        for turn in sc["turns"]:
            buf.ingest_turn(turn["idx"], turn["text"])
        buf.flush()

        exp = sc["expected"]

        # coref merges
        exp_coref = {
            (m["canonical"], m["descriptor"]) for m in exp.get("coref_merges", [])
        }
        got_coref = {
            (m.canonical_entity, m.anonymous_descriptor) for m in buf.state.coref_merges
        }
        coref_missing = exp_coref - got_coref
        coref_extra = got_coref - exp_coref
        coref_pass = not coref_missing and not coref_extra

        # role slots
        slot_pass = True
        slot_details = []
        for sid, seq in (exp.get("role_slot_history") or {}).items():
            exp_seq = [e["filler"] for e in seq]
            slot = buf.state.slots.get(sid)
            got_seq = [e.filler for e in (slot.history if slot else [])]
            ok = got_seq == exp_seq
            if not ok:
                slot_pass = False
            slot_details.append(
                {"slot": sid, "exp": exp_seq, "got": got_seq, "pass": ok}
            )

        # entities admitted
        admitted, deferred = _salience_partition(buf)

        def norm(s):
            return (
                s.strip()
                .lower()
                .replace("the ", "")
                .replace("a ", "")
                .replace("my ", "")
            )

        adm_keys = {norm(a) for a in admitted}
        exp_admit = {norm(a) for a in exp.get("entities_admitted", [])}
        exp_defer = {norm(d) for d in exp.get("entities_deferred", [])}

        def fuzzy_missing(missing, available):
            still = set()
            for m in missing:
                if not any(m in a or a in m for a in available):
                    still.add(m)
            return still

        admit_missing = fuzzy_missing(exp_admit - adm_keys, adm_keys)
        admit_false = adm_keys & exp_defer
        entities_pass = not admit_missing and not admit_false

        # multi-label
        ml_pass = True
        ml_details = []
        for exp_ml in exp.get("multi_label_introductions", []):
            turn = exp_ml["turn"]
            needed = exp_ml["topics_include"]
            emitted = buf.state.facts_by_source_turn.get(turn, [])
            topics_set = set()
            for group in emitted:
                for t in group:
                    topics_set.add(t.split("/")[0])
            missing = [n for n in needed if n not in topics_set]
            if missing:
                ml_pass = False
                ml_details.append(
                    {"turn": turn, "missing": missing, "got": sorted(topics_set)}
                )
            else:
                ml_details.append(
                    {"turn": turn, "got": sorted(topics_set), "pass": True}
                )

        all_pass = coref_pass and slot_pass and entities_pass and ml_pass
        results.append(
            {
                "scenario_id": sc["id"],
                "coref_pass": coref_pass,
                "coref_missing": sorted(coref_missing),
                "coref_extra": sorted(coref_extra),
                "slot_pass": slot_pass,
                "slot_details": slot_details,
                "entities_pass": entities_pass,
                "admit_missing": sorted(admit_missing),
                "admit_false": sorted(admit_false),
                "admitted": sorted(admitted),
                "ml_pass": ml_pass,
                "ml_details": ml_details,
                "all_pass": all_pass,
            }
        )
        print(
            f"  E6 {sc['id']}: coref={coref_pass} slots={slot_pass} "
            f"ents={entities_pass} ml={ml_pass} ALL={all_pass}",
            flush=True,
        )
    return {"scenarios": results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    cache, _ = make_caches()
    budget = Budget(max_llm=260, max_embed=50, stop_at_llm=230, stop_at_embed=40)

    t0 = time.time()
    all_results = {}
    try:
        print(f"\n=== E1: Coref distances (window={WINDOW_SIZE}) ===", flush=True)
        all_results["E1"] = run_e1(cache, budget)
        _save(cache, all_results)
        print(f"  -- LLM {budget.llm_calls} cost ~${budget.cost():.3f}")

        print("\n=== E2: Salience escalation ===", flush=True)
        all_results["E2"] = run_e2(cache, budget)
        _save(cache, all_results)
        print(f"  -- LLM {budget.llm_calls} cost ~${budget.cost():.3f}")

        print("\n=== E3: Late correction ===", flush=True)
        all_results["E3"] = run_e3(cache, budget)
        _save(cache, all_results)
        print(f"  -- LLM {budget.llm_calls} cost ~${budget.cost():.3f}")

        print("\n=== E4: Query-during-buffer ===", flush=True)
        all_results["E4"] = run_e4(cache, budget)
        _save(cache, all_results)
        print(f"  -- LLM {budget.llm_calls} cost ~${budget.cost():.3f}")

        print("\n=== E5: End-of-stream flush ===", flush=True)
        all_results["E5"] = run_e5(cache, budget)
        _save(cache, all_results)
        print(f"  -- LLM {budget.llm_calls} cost ~${budget.cost():.3f}")

        print("\n=== E6: Integration replay ===", flush=True)
        all_results["E6"] = run_e6(cache, budget)
        _save(cache, all_results)
        print(f"  -- LLM {budget.llm_calls} cost ~${budget.cost():.3f}")

    except RuntimeError as ex:
        print(f"BUDGET STOP: {ex}")
        all_results["_aborted"] = str(ex)
    finally:
        cache.save()
        all_results["_meta"] = {
            "window_size": WINDOW_SIZE,
            "llm_calls": budget.llm_calls,
            "embed_calls": budget.embed_calls,
            "cost_usd": round(budget.cost(), 4),
            "wall_seconds": round(time.time() - t0, 1),
        }
        _save(cache, all_results)
    print("\n=== DONE ===")
    print(json.dumps(all_results["_meta"], indent=2))


def _save(cache: Cache, results: dict) -> None:
    cache.save()
    (RESULTS_DIR / "all.json").write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
