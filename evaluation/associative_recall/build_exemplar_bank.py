"""Build exemplar bank from cached v2f runs.

For each question across all 4 datasets:
  1. Load fairbackfill_meta_v2f_*.json to identify where v2f > cosine baseline
  2. Re-run MetaV2f (cached, no new LLM calls) to extract cue text
  3. Filter to "successful" exemplars: v2f r@50 > baseline r@50
  4. Save with embedded exemplar questions

Output: results/fewshot_exemplar_bank.json

Usage:
    uv run python build_exemplar_bank.py
"""

import json
from pathlib import Path

from associative_recall import SegmentStore
from best_shot import MetaV2f
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

DATASETS = {
    "locomo_30q": {
        "npz": "segments_extended.npz",
        "questions": "questions_extended.json",
        "filter": lambda q: q.get("benchmark") == "locomo",
        "max_questions": 30,
    },
    "synthetic_19q": {
        "npz": "segments_synthetic.npz",
        "questions": "questions_synthetic.json",
        "filter": None,
        "max_questions": None,
    },
    "puzzle_16q": {
        "npz": "segments_puzzle.npz",
        "questions": "questions_puzzle.json",
        "filter": None,
        "max_questions": None,
    },
    "advanced_23q": {
        "npz": "segments_advanced.npz",
        "questions": "questions_advanced.json",
        "filter": None,
        "max_questions": None,
    },
}


def load_dataset(ds_name: str):
    cfg = DATASETS[ds_name]
    store = SegmentStore(data_dir=DATA_DIR, npz_name=cfg["npz"])
    with open(DATA_DIR / cfg["questions"]) as f:
        questions = json.load(f)
    if cfg["filter"]:
        questions = [q for q in questions if cfg["filter"](q)]
    if cfg["max_questions"]:
        questions = questions[: cfg["max_questions"]]
    return store, questions


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    exemplars = []
    bank_questions_to_embed = []

    for ds_name in DATASETS:
        # Load cached fairbackfill_meta_v2f summary to know which questions
        # v2f-helped.
        fb_path = RESULTS_DIR / f"fairbackfill_meta_v2f_{ds_name}.json"
        if not fb_path.exists():
            print(f"  Skipping {ds_name}: no fairbackfill_meta_v2f results.")
            continue

        with open(fb_path) as f:
            fb_data = json.load(f)
        fb_results = {
            (r["conversation_id"], r["question_index"]): r for r in fb_data["results"]
        }

        store, questions = load_dataset(ds_name)
        print(f"\n--- {ds_name}: {len(questions)} questions ---")

        arch = MetaV2f(store)
        new_llm_calls = 0

        for q in questions:
            conv_id = q["conversation_id"]
            qi = q.get("question_index", -1)
            q_text = q["question"]
            category = q.get("category", "unknown")

            key = (conv_id, qi)
            if key not in fb_results:
                continue
            fb_row = fb_results[key]

            # Check if v2f beat baseline (either @20 or @50)
            delta_20 = fb_row["fair_backfill"]["delta_r@20"]
            delta_50 = fb_row["fair_backfill"]["delta_r@50"]
            arch_50 = fb_row["fair_backfill"]["arch_r@50"]

            # Success filter: v2f strictly beat baseline at either K
            # AND arch itself recalled at least something
            is_success = (delta_20 > 0.001 or delta_50 > 0.001) and arch_50 > 0.0
            if not is_success:
                continue

            # Re-run MetaV2f to extract cues from cache
            arch.reset_counters()
            before_llm = arch.llm_calls
            try:
                result = arch.retrieve(q_text, conv_id)
            except Exception as e:
                print(f"  ERROR {conv_id} qi={qi}: {e}", flush=True)
                continue
            after_llm = arch.llm_calls
            cues = result.metadata.get("cues", [])
            if not cues:
                continue

            if after_llm - before_llm > 0:
                new_llm_calls += after_llm - before_llm

            exemplars.append(
                {
                    "dataset": ds_name,
                    "conversation_id": conv_id,
                    "question_index": qi,
                    "question": q_text,
                    "category": category,
                    "cues": cues,
                    "delta_r@20": delta_20,
                    "delta_r@50": delta_50,
                    "arch_r@50": arch_50,
                }
            )
            bank_questions_to_embed.append(q_text)

        arch.save_caches()
        print(f"  {ds_name}: {new_llm_calls} new LLM calls")

    print(f"\nExemplar bank size: {len(exemplars)}")

    # Embed all exemplar questions via one MetaV2f (for embedding reuse)
    # We need an embedding store; pick the first available
    store = SegmentStore(data_dir=DATA_DIR, npz_name="segments_extended.npz")
    arch = MetaV2f(store)
    for ex in exemplars:
        emb = arch.embed_text(ex["question"])
        ex["question_embedding"] = emb.tolist()
    arch.save_caches()

    # Save exemplar bank
    bank_path = RESULTS_DIR / "fewshot_exemplar_bank.json"
    with open(bank_path, "w") as f:
        json.dump(
            {
                "exemplars": exemplars,
                "total": len(exemplars),
                "by_dataset": {
                    ds: sum(1 for e in exemplars if e["dataset"] == ds)
                    for ds in DATASETS
                },
            },
            f,
            indent=2,
            default=str,
        )
    print(f"Saved: {bank_path}")


if __name__ == "__main__":
    main()
