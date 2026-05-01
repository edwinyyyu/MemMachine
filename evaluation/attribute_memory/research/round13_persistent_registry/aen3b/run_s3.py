"""Run S3 (LRU stress) on aen3b vs aen3 vs baseline (aen1_simple).

Reuses the round-13 grading harness (align_ids_to_gt + grade_scenario).
Uses round-13's existing aen3 cache for aen3 reruns (warm).
Has its own cache for aen3b (warm-from-aen3 for shared LLM calls / embeds).
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND13 = HERE.parent
ROUND12 = ROUND13.parent / "round12_entity_registry"
ROUND11 = ROUND13.parent / "round11_writer_stress"
ROUND7 = ROUND13.parent / "round7"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROUND13 / "architectures"))
sys.path.insert(0, str(ROUND13 / "scenarios"))
sys.path.insert(0, str(ROUND12 / "architectures"))
sys.path.insert(0, str(ROUND12 / "scenarios"))
sys.path.insert(0, str(ROUND11 / "architectures"))
sys.path.insert(0, str(ROUND7 / "experiments"))

import aen3_persistent  # noqa: E402
import aen3b_persistent  # noqa: E402
from _common import Budget, Cache  # noqa: E402
from coref_stress import Scenario, scenario_s3  # noqa: E402

CACHE_DIR = HERE / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Decision lookup with fuzzy surface matching
# ---------------------------------------------------------------------------


def _norm(s: str) -> str:
    """Lowercase and collapse whitespace; strip leading article and trailing
    punctuation. Conservative — never returns empty unless input is empty."""
    s = s.strip().lower()
    s = " ".join(s.split())
    s = s.strip(".,!?;:")
    return s


def coref_log_decisions(coref_log: dict) -> dict[int, dict[str, str | None]]:
    """Build per-turn decision dict keyed by both the original surface AND
    a normalised lowercase variant, so the grader (which uses GT-surface)
    can find the entry even when the LLM returned 'The X' for GT 'the x'.

    For substring overlaps (GT 'the barista' vs coref 'The barista at the
    cafe by my apartment'), we also index every length-2+ contiguous suffix
    of the coref surface that is a normalised form of a noun phrase prefix
    of the longer one. We do this by also recording, for each decision, all
    NORMALISED tokens-prefix-with-leading-the variants.
    """
    out: dict[int, dict[str, str | None]] = {}
    for tidx, decs in coref_log.items():
        per_turn: dict[str, str | None] = {}
        for d in decs:
            surface = d.surface or ""
            eid = d.entity_id
            # original
            per_turn[surface] = eid
            # normalised
            n = _norm(surface)
            per_turn.setdefault(n, eid)
            # generate noun-phrase prefixes (up to 6 tokens) so that longer
            # coref spans cover shorter GT spans
            tokens = n.split()
            for k in range(2, min(len(tokens), 7) + 1):
                prefix = " ".join(tokens[:k])
                per_turn.setdefault(prefix, eid)
            # also prefixes WITHOUT the leading article
            if tokens and tokens[0] in {"the", "a", "an"}:
                tokens2 = tokens[1:]
                for k in range(2, min(len(tokens2), 6) + 1):
                    prefix = " ".join(tokens2[:k])
                    per_turn.setdefault("the " + prefix, eid)
        out[tidx] = per_turn
    return out


def baseline_coref(scenario: Scenario) -> dict[int, dict[str, str | None]]:
    decisions: dict[int, dict[str, str | None]] = {}
    for t in scenario.turns:
        per_turn: dict[str, str | None] = {}
        for m in t.mentions:
            if m.kind == "pronoun":
                per_turn[m.surface] = None
            else:
                per_turn[m.surface] = "surf::" + m.surface.lower().strip()
        decisions[t.idx] = per_turn
    return decisions


# ---------------------------------------------------------------------------
# Alignment + grading (same as round 12 / 13)
# ---------------------------------------------------------------------------


def _decision_lookup(per_turn: dict[str, str | None], surface: str) -> str | None:
    if surface in per_turn:
        return per_turn[surface]
    n = _norm(surface)
    if n in per_turn:
        return per_turn[n]
    return None


def align_ids_to_gt(
    scenario: Scenario,
    decisions: dict[int, dict[str, str | None]],
) -> dict[str | None, str | None]:
    cooc: dict[str | None, Counter] = defaultdict(Counter)
    for t in scenario.turns:
        per_turn = decisions.get(t.idx, {})
        for m in t.mentions:
            arch_id = _decision_lookup(per_turn, m.surface)
            if arch_id is None or m.entity_id is None:
                continue
            cooc[arch_id][m.entity_id] += 1
    mapping: dict[str | None, str | None] = {}
    used_gt: set[str] = set()
    arch_ids_sorted = sorted(cooc.keys(), key=lambda a: -sum(cooc[a].values()))
    for arch_id in arch_ids_sorted:
        candidates = cooc[arch_id].most_common()
        chosen = None
        for gt, _ in candidates:
            if gt not in used_gt:
                chosen = gt
                break
        if chosen is None and candidates:
            chosen = candidates[0][0]
        mapping[arch_id] = chosen
        if chosen is not None:
            used_gt.add(chosen)
    return mapping


def grade_scenario(
    scenario: Scenario,
    decisions: dict[int, dict[str, str | None]],
) -> dict:
    mapping = align_ids_to_gt(scenario, decisions)
    n_total = n_correct = 0
    n_named_total = n_named_correct = 0
    n_pron_total = n_pron_correct = 0
    n_desc_total = n_desc_correct = 0
    failures: list[dict] = []
    for t in scenario.turns:
        per_turn = decisions.get(t.idx, {})
        for m in t.mentions:
            if m.entity_id is None:
                continue
            arch_id = _decision_lookup(per_turn, m.surface)
            mapped = mapping.get(arch_id) if arch_id else None
            ok = mapped == m.entity_id
            n_total += 1
            n_correct += int(ok)
            if m.kind == "named":
                n_named_total += 1
                n_named_correct += int(ok)
            elif m.kind == "pronoun":
                n_pron_total += 1
                n_pron_correct += int(ok)
            elif m.kind == "descriptor":
                n_desc_total += 1
                n_desc_correct += int(ok)
            if not ok:
                failures.append(
                    {
                        "turn_idx": t.idx,
                        "surface": m.surface,
                        "kind": m.kind,
                        "expected_gt": m.entity_id,
                        "arch_id": arch_id,
                        "mapped_gt": mapped,
                        "turn_text": t.text,
                        "turn_kind": t.kind,
                    }
                )
    return {
        "n_total": n_total,
        "n_correct": n_correct,
        "accuracy": (n_correct / n_total) if n_total else 0.0,
        "named": {
            "total": n_named_total,
            "correct": n_named_correct,
            "rate": (n_named_correct / n_named_total) if n_named_total else 0.0,
        },
        "pronoun": {
            "total": n_pron_total,
            "correct": n_pron_correct,
            "rate": (n_pron_correct / n_pron_total) if n_pron_total else 0.0,
        },
        "descriptor": {
            "total": n_desc_total,
            "correct": n_desc_correct,
            "rate": (n_desc_correct / n_desc_total) if n_desc_total else 0.0,
        },
        "id_mapping": {str(k): v for k, v in mapping.items()},
        "failures": failures,
    }


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------


def run_aen3b(scenario: Scenario, cache: Cache, budget: Budget) -> dict:
    pairs = [(t.idx, t.text) for t in scenario.turns]
    pre_llm = budget.llm_calls
    pre_emb = budget.embed_calls
    log, idx, reg, coref_log = aen3b_persistent.ingest_turns_with_registry(
        pairs,
        cache,
        budget,
        batch_size=5,
        rebuild_index_every=100,
        lru_size=20,
        top_k=5,
        run_writer=False,
    )
    cache.save()
    decisions = coref_log_decisions(coref_log)
    grade = grade_scenario(scenario, decisions)
    delta_llm = budget.llm_calls - pre_llm
    delta_emb = budget.embed_calls - pre_emb
    n_embed_searches = sum(
        1
        for decs in coref_log.values()
        for d in decs
        if getattr(d, "used_embedding_search", False)
    )
    return {
        "arch": "aen3b_persistent",
        "grade": grade,
        "log_size": len(log),
        "n_entities": len(reg.by_id),
        "n_embed_searches": n_embed_searches,
        "registry": aen3b_persistent.registry_snapshot(reg),
        "coref_decisions": {
            str(t): [asdict(d) for d in decs] for t, decs in coref_log.items()
        },
        "llm_calls_delta": delta_llm,
        "embed_calls_delta": delta_emb,
        "llm_per_turn": delta_llm / max(1, len(scenario.turns)),
        "embed_per_turn": delta_emb / max(1, len(scenario.turns)),
    }


def run_aen3(scenario: Scenario, cache: Cache, budget: Budget) -> dict:
    pairs = [(t.idx, t.text) for t in scenario.turns]
    pre_llm = budget.llm_calls
    pre_emb = budget.embed_calls
    log, idx, reg, coref_log = aen3_persistent.ingest_turns_with_registry(
        pairs,
        cache,
        budget,
        batch_size=5,
        rebuild_index_every=100,
        lru_size=20,
        top_k=5,
        run_writer=False,
    )
    cache.save()
    decisions = coref_log_decisions(coref_log)
    grade = grade_scenario(scenario, decisions)
    delta_llm = budget.llm_calls - pre_llm
    delta_emb = budget.embed_calls - pre_emb
    n_embed_searches = sum(
        1
        for decs in coref_log.values()
        for d in decs
        if getattr(d, "used_embedding_search", False)
    )
    return {
        "arch": "aen3_persistent",
        "grade": grade,
        "log_size": len(log),
        "n_entities": len(reg.by_id),
        "n_embed_searches": n_embed_searches,
        "registry": aen3_persistent.registry_snapshot(reg),
        "llm_calls_delta": delta_llm,
        "embed_calls_delta": delta_emb,
        "llm_per_turn": delta_llm / max(1, len(scenario.turns)),
        "embed_per_turn": delta_emb / max(1, len(scenario.turns)),
    }


def run_baseline(scenario: Scenario) -> dict:
    decisions = baseline_coref(scenario)
    grade = grade_scenario(scenario, decisions)
    return {"arch": "aen1_simple", "grade": grade}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    budget = Budget(max_llm=200, max_embed=100, stop_at_llm=199, stop_at_embed=99)

    s = scenario_s3()
    print(f"=== Scenario: {s.name} ({len(s.turns)} turns) ===")
    s_results: dict[str, dict] = {}

    # Warm-start aen3b cache from aen3 cache (round 13). Coref + alias-disambig
    # prompts are unchanged, so those LLM cache keys hit. Only descriptor_pick
    # and embed-search-with-richer-blob may miss.
    aen3_cache_path = ROUND13 / "cache" / f"aen3_{s.name}.json"
    aen3b_cache_path = CACHE_DIR / f"aen3b_{s.name}.json"
    if aen3_cache_path.exists() and not aen3b_cache_path.exists():
        shutil.copy(aen3_cache_path, aen3b_cache_path)
        print(f"  warmed aen3b cache from {aen3_cache_path.name}")

    # ---- aen3b ----
    cache_a3b = Cache(aen3b_cache_path)
    print("  [aen3b_persistent] running...")
    s_results["aen3b_persistent"] = run_aen3b(s, cache_a3b, budget)
    g = s_results["aen3b_persistent"]["grade"]
    es = s_results["aen3b_persistent"]["n_embed_searches"]
    print(
        f"    acc={g['accuracy']:.2%} "
        f"named={g['named']['rate']:.2%} "
        f"pron={g['pronoun']['rate']:.2%} "
        f"desc={g['descriptor']['rate']:.2%} "
        f"({g['n_correct']}/{g['n_total']}) "
        f"LLM/turn={s_results['aen3b_persistent']['llm_per_turn']:.2f} "
        f"emb/turn={s_results['aen3b_persistent']['embed_per_turn']:.2f} "
        f"embed_searches={es}"
    )

    # ---- aen3 (warm cache from round 13) ----
    if aen3_cache_path.exists():
        cache_a3 = Cache(aen3_cache_path)
    else:
        cache_a3 = Cache(CACHE_DIR / f"aen3_{s.name}.json")
    print("  [aen3_persistent] running...")
    s_results["aen3_persistent"] = run_aen3(s, cache_a3, budget)
    g = s_results["aen3_persistent"]["grade"]
    es = s_results["aen3_persistent"]["n_embed_searches"]
    print(
        f"    acc={g['accuracy']:.2%} "
        f"named={g['named']['rate']:.2%} "
        f"pron={g['pronoun']['rate']:.2%} "
        f"desc={g['descriptor']['rate']:.2%} "
        f"({g['n_correct']}/{g['n_total']}) "
        f"LLM/turn={s_results['aen3_persistent']['llm_per_turn']:.2f} "
        f"emb/turn={s_results['aen3_persistent']['embed_per_turn']:.2f} "
        f"embed_searches={es}"
    )

    # ---- baseline ----
    print("  [aen1_simple baseline] grading only...")
    s_results["aen1_simple"] = run_baseline(s)
    g = s_results["aen1_simple"]["grade"]
    print(
        f"    acc={g['accuracy']:.2%} "
        f"named={g['named']['rate']:.2%} "
        f"pron={g['pronoun']['rate']:.2%} "
        f"desc={g['descriptor']['rate']:.2%} "
        f"({g['n_correct']}/{g['n_total']})"
    )

    out = RESULTS_DIR / f"{s.name}.json"
    out.write_text(json.dumps(s_results, indent=2, default=str))
    print(f"\n  wrote {out}")
    print(
        f"  budget so far: LLM={budget.llm_calls} "
        f"embed={budget.embed_calls} cost=${budget.cost():.3f}"
    )


if __name__ == "__main__":
    main()
