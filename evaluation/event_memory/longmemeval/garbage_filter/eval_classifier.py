"""Score a prompt variant on the adversarial test set.

Asymmetric scoring:
  - false reject (KEEP -> classifier says REJECT) is the costly error
  - false keep (REJECT -> classifier says KEEP) is acceptable

Output: per-bucket counts and the explicit list of mistakes so we can
inspect them and iterate the prompt.
"""

from __future__ import annotations

import argparse
import asyncio

from classifier import classify_many
from dotenv import load_dotenv
from test_set import all_items
from test_set_hard import all_hard

load_dotenv("/Users/eyu/edwinyyyu/mmcc/segment_store/.env")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        default="v2",
        choices=["v1", "v2", "v3", "v5", "v6", "v7", "v8", "v9"],
    )
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--reasoning", default="low")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--set", default="main", choices=["main", "hard", "both"])
    args = parser.parse_args()

    if args.set == "hard":
        items = all_hard()
    elif args.set == "both":
        items = all_items() + all_hard()
    else:
        items = all_items()
    texts = [t for _, t in items]
    labels = [lab for lab, _ in items]

    results = asyncio.run(
        classify_many(
            texts,
            model=args.model,
            prompt=args.prompt,
            reasoning_effort=args.reasoning,
            concurrency=args.concurrency,
        )
    )

    fr, fk = [], []
    n_keep = sum(1 for lab in labels if lab == "KEEP")
    n_reject = sum(1 for lab in labels if lab == "REJECT")
    correct_keep = correct_reject = 0
    for (gold, _), r in zip(items, results, strict=False):
        pred = r.label
        if gold == "KEEP" and pred == "REJECT":
            fr.append(r.text)
        elif gold == "REJECT" and pred == "KEEP":
            fk.append(r.text)
        elif gold == "KEEP" and pred == "KEEP":
            correct_keep += 1
        elif gold == "REJECT" and pred == "REJECT":
            correct_reject += 1

    print(f"prompt={args.prompt} model={args.model} reasoning={args.reasoning}")
    print()
    print(
        f"  KEEP  total {n_keep:3d}  correct {correct_keep:3d}  "
        f"FALSE-REJECT {len(fr):3d}  ({len(fr) / max(n_keep, 1):.1%})"
    )
    print(
        f"  REJECT total {n_reject:3d}  correct {correct_reject:3d}  "
        f"FALSE-KEEP   {len(fk):3d}  ({len(fk) / max(n_reject, 1):.1%})"
    )
    print()
    if fr:
        print("FALSE REJECTS (gold=KEEP, pred=REJECT) — these are the costly errors:")
        for t in fr:
            print(f"  - {t!r}")
        print()
    if fk:
        print("FALSE KEEPS (gold=REJECT, pred=KEEP) — acceptable but want to minimize:")
        for t in fk:
            print(f"  - {t!r}")


if __name__ == "__main__":
    main()
