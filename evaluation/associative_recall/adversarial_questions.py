"""Generate adversarial test cases for associative recall.

These test edge cases where keyword-dense cues might fail:
1. Self-referential: the best cue IS the question itself
2. Vocabulary mismatch: answer uses different vocabulary than question
3. Generic language: answer is in very common words
4. Negation: questions about absence of information
5. Ambiguous: cue matches wrong part of conversation
"""

import json
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"


def analyze_existing_failures():
    """Look at existing results to find patterns in failures."""
    results_dir = Path(__file__).resolve().parent / "results"

    # Load best results (v8, neighbor=1)
    results_path = results_dir / "results_gpt-5-mini_v8_n1.json"
    if not results_path.exists():
        print("No v8_n1 results found")
        return

    with open(results_path) as f:
        results = json.load(f)

    print("FAILURE ANALYSIS (v8, neighbor_radius=1)")
    print("=" * 80)

    for r in results:
        b_all = r["baseline_recalls"]["r@all"]
        a_all = r["assoc_recalls"]["r@all"]
        if a_all < 1.0:
            print(f"\nQ: {r['question'][:100]}")
            print(f"  Category: {r['category']}")
            print(f"  Source IDs: {r['source_chat_ids']}")
            print(f"  Baseline r@all: {b_all:.3f}")
            print(f"  Assoc r@all: {a_all:.3f}")
            print(f"  Total retrieved: {r['total_retrieved']}")
            print("  Missed IDs: ", end="")
            # Which source IDs were missed?
            all_retrieved_ids = set()
            for hop in r["hop_details"]:
                all_retrieved_ids.update(hop.get("new_source_hits", []))
            missed = set(r["source_chat_ids"]) - all_retrieved_ids
            print(missed)

            for hop in r["hop_details"]:
                if hop["new_source_hits"]:
                    print(f"  Hop {hop['hop']}: found {hop['new_source_hits']}")
                for cue in hop["cues"][:2]:
                    print(f"    Cue: {cue[:100]}")


def create_adversarial_for_beam():
    """Create adversarial questions for existing BEAM conversations.

    Since we know the conversation content, we can craft questions that
    specifically test failure modes.
    """
    # Load segments to understand conversation content
    data = np.load(DATA_DIR / "segments.npz", allow_pickle=True)
    texts = data["texts"]
    turn_ids = data["turn_ids"]
    conv_ids = data["conversation_ids"]

    # For now, just analyze the first conversation
    mask = conv_ids == "1"
    conv1_texts = texts[mask]
    conv1_tids = turn_ids[mask]

    print(f"\nConversation 1: {len(conv1_texts)} segments")
    print(f"Turn IDs: {int(conv1_tids.min())}-{int(conv1_tids.max())}")

    # Show a sample of segments to understand the conversation
    print("\nSample segments from conv 1:")
    for i in range(min(20, len(conv1_texts))):
        print(f"  [{conv1_tids[i]}] {str(conv1_texts[i])[:100]}")

    # Adversarial question types we should create manually after
    # understanding the conversation content
    adversarial_types = [
        {
            "type": "self_referential",
            "description": "Questions where the cue should match question turns, not answer turns",
            "example": "How many times did I ask about databases?",
        },
        {
            "type": "vocabulary_mismatch",
            "description": "Questions where the answer uses very different vocabulary",
            "example": "What was the first thing I set up for my app? (answer: virtual environment, not 'app setup')",
        },
        {
            "type": "generic_answer",
            "description": "Answers in very common language that many cues would match",
            "example": "What did you say I should do next? (answer: 'you should test it')",
        },
        {
            "type": "negation",
            "description": "Questions about things NOT discussed",
            "example": "Did I ever mention using Docker?",
        },
    ]

    print("\nAdversarial question types to create:")
    for at in adversarial_types:
        print(f"  {at['type']}: {at['description']}")
        print(f"    Example: {at['example']}")


if __name__ == "__main__":
    analyze_existing_failures()
    create_adversarial_for_beam()
