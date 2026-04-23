"""Analyze logic_constraint failures: v15/v2f vs cosine baseline.

Compares retrieval at r@20 between v15_control, meta_v2f, and cosine-only.
Extracts actual cues from cache, computes distance metrics, and examines
content of missed source turns.

Usage:
    uv run python analyze_lc_failures.py
"""

import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from associative_recall import SegmentStore, Segment
from best_shot import V15Control, MetaV2f

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main() -> None:
    # Load questions
    with open(DATA_DIR / "questions_puzzle.json") as f:
        questions = json.load(f)
    lc_questions = [q for q in questions if q["category"] == "logic_constraint"]

    # Load segment store
    store = SegmentStore(data_dir=DATA_DIR, npz_name="segments_puzzle.npz")

    # Create architectures
    v15 = V15Control(store)
    v2f = MetaV2f(store)

    output: dict = {"questions": []}

    for q in lc_questions:
        conv_id = q["conversation_id"]
        q_text = q["question"]
        source_ids = set(q["source_chat_ids"])
        q_idx = q["question_index"]

        print(f"\n{'=' * 70}")
        print(f"Q{q_idx} [{conv_id}]: {q_text}")
        print(f"Source turns ({len(source_ids)}): {sorted(source_ids)}")

        # Segment lookup for this convo
        conv_segs = [s for s in store.segments if s.conversation_id == conv_id]
        by_turn = {s.turn_id: s for s in conv_segs}

        # Question embedding
        q_emb = v15.embed_text(q_text)

        # Cosine top-20 baseline
        cosine_res = store.search(q_emb, top_k=20, conversation_id=conv_id)
        cosine_top20 = list(cosine_res.segments)
        cosine_ids = {s.turn_id for s in cosine_top20}
        cosine_hits = cosine_ids & source_ids
        cosine_misses = source_ids - cosine_ids

        print(f"\nCOSINE TOP-20:")
        print(f"  Found ({len(cosine_hits)}/{len(source_ids)}): "
              f"{sorted(cosine_hits)}")
        print(f"  Missed: {sorted(cosine_misses)}")

        # Run v15 (will populate cache)
        v15.reset_counters()
        v15_res = v15.retrieve(q_text, conv_id)
        v15_cues = v15_res.metadata.get("cues", [])
        v15_output = v15_res.metadata.get("output", "")

        # Run v2f (will populate cache)
        v2f.reset_counters()
        v2f_res = v2f.retrieve(q_text, conv_id)
        v2f_cues = v2f_res.metadata.get("cues", [])
        v2f_output = v2f_res.metadata.get("output", "")

        # Fair-backfill r@20 reconstruction: arch's own cue-found segments
        # first, then cosine backfill, truncated to 20.
        def fair_backfill_top20(arch_segs):
            seen: set[int] = set()
            uniq = []
            for s in arch_segs:
                if s.index not in seen:
                    uniq.append(s)
                    seen.add(s.index)
            at_K = uniq[:20]
            arch_idx = {s.index for s in at_K}
            if len(at_K) < 20:
                backfill = [s for s in cosine_top20 if s.index not in arch_idx]
                at_K = at_K + backfill[: 20 - len(at_K)]
            return at_K[:20]

        v15_at20 = fair_backfill_top20(v15_res.segments)
        v2f_at20 = fair_backfill_top20(v2f_res.segments)
        v15_ids = {s.turn_id for s in v15_at20}
        v2f_ids = {s.turn_id for s in v2f_at20}

        # What did each LOSE vs cosine?
        lost_by_v15 = cosine_hits - v15_ids
        lost_by_v2f = cosine_hits - v2f_ids

        # What did each ADD vs cosine?
        gained_by_v15 = (v15_ids & source_ids) - cosine_hits
        gained_by_v2f = (v2f_ids & source_ids) - cosine_hits

        print(f"\nV15 TOP-20 (fair-backfill):")
        print(f"  Recall {len(v15_ids & source_ids)}/{len(source_ids)} = "
              f"{len(v15_ids & source_ids)/len(source_ids):.1%}")
        print(f"  LOST vs cosine: {sorted(lost_by_v15)}")
        print(f"  GAINED vs cosine: {sorted(gained_by_v15)}")

        print(f"\nV2F TOP-20 (fair-backfill):")
        print(f"  Recall {len(v2f_ids & source_ids)}/{len(source_ids)} = "
              f"{len(v2f_ids & source_ids)/len(source_ids):.1%}")
        print(f"  LOST vs cosine: {sorted(lost_by_v2f)}")
        print(f"  GAINED vs cosine: {sorted(gained_by_v2f)}")

        # Cues
        print(f"\nV15 ASSESSMENT + CUES:")
        print(f"{v15_output}")
        print(f"\nV2F ASSESSMENT + CUES:")
        print(f"{v2f_output}")

        # Cue-to-question cosine similarity
        cue_metrics: list[dict] = []
        for label, cues in [("v15", v15_cues), ("v2f", v2f_cues)]:
            for i, cue in enumerate(cues):
                c_emb = v15.embed_text(cue)
                sim_to_q = cos_sim(c_emb, q_emb)
                # What did this cue retrieve at top-10?
                cue_res = store.search(c_emb, top_k=10,
                                       conversation_id=conv_id)
                retr_ids = [s.turn_id for s in cue_res.segments]
                hits = [t for t in retr_ids if t in source_ids]
                cue_metrics.append({
                    "arch": label,
                    "cue_idx": i,
                    "cue": cue,
                    "cos_to_question": round(sim_to_q, 4),
                    "top10_turns": retr_ids,
                    "source_hits_top10": hits,
                })

        print(f"\nCUE METRICS:")
        for cm in cue_metrics:
            print(f"  [{cm['arch']} cue {cm['cue_idx']}] "
                  f"cos(q, cue)={cm['cos_to_question']:.4f}")
            print(f"    cue: {cm['cue'][:150]}")
            print(f"    top10 turns: {cm['top10_turns']}")
            print(f"    source hits: {cm['source_hits_top10']}")

        # For each MISSED source turn (in the lost_by_v2f set), print content
        # and compute: (a) cosine to question, (b) cosine to each cue
        all_missed = (source_ids - v2f_ids) | (source_ids - v15_ids) | cosine_misses
        missed_analysis = []
        print(f"\nMISSED SOURCE TURN CONTENT (all missed across any method):")
        for t in sorted(all_missed):
            if t not in by_turn:
                print(f"  Turn {t}: NOT IN STORE")
                continue
            seg = by_turn[t]
            seg_emb = store.normalized_embeddings[seg.index]
            # Question emb normalized
            qn = q_emb / max(np.linalg.norm(q_emb), 1e-10)
            sim_q = float(np.dot(seg_emb, qn))
            # Cosine to each cue
            cue_sims = {}
            for cm in cue_metrics:
                c_emb = v15.embed_text(cm["cue"])
                cn = c_emb / max(np.linalg.norm(c_emb), 1e-10)
                cue_sims[f"{cm['arch']}_cue{cm['cue_idx']}"] = round(
                    float(np.dot(seg_emb, cn)), 4
                )
            in_cosine = t in cosine_ids
            in_v15 = t in v15_ids
            in_v2f = t in v2f_ids
            status = []
            if in_cosine:
                status.append("COSINE")
            if in_v15:
                status.append("V15")
            if in_v2f:
                status.append("V2F")
            status_str = "+".join(status) if status else "MISSED_ALL"

            print(f"  Turn {t:3d} [{seg.role}] ({status_str}) "
                  f"cos(q)={sim_q:.3f}:")
            print(f"    {seg.text[:220]}")
            print(f"    cue_sims: {cue_sims}")
            missed_analysis.append({
                "turn_id": t,
                "role": seg.role,
                "text": seg.text,
                "in_cosine_top20": in_cosine,
                "in_v15_top20": in_v15,
                "in_v2f_top20": in_v2f,
                "cos_to_question": round(sim_q, 4),
                "cos_to_cues": cue_sims,
            })

        output["questions"].append({
            "question_index": q_idx,
            "conversation_id": conv_id,
            "question": q_text,
            "source_ids": sorted(source_ids),
            "cosine_top20_ids": sorted(cosine_ids),
            "cosine_hits": sorted(cosine_hits),
            "v15_top20_ids": sorted(v15_ids),
            "v2f_top20_ids": sorted(v2f_ids),
            "v15_lost_vs_cosine": sorted(lost_by_v15),
            "v2f_lost_vs_cosine": sorted(lost_by_v2f),
            "v15_gained_vs_cosine": sorted(gained_by_v15),
            "v2f_gained_vs_cosine": sorted(gained_by_v2f),
            "v15_output": v15_output,
            "v2f_output": v2f_output,
            "cue_metrics": cue_metrics,
            "missed_turns": missed_analysis,
        })

    # Save caches
    v15.save_caches()
    v2f.save_caches()

    out_path = RESULTS_DIR / "lc_failure_analysis_raw.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n\nRaw analysis saved: {out_path}")


if __name__ == "__main__":
    main()
