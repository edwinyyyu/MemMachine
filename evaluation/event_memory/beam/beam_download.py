"""Download BEAM dataset splits from HuggingFace and save as JSON."""

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path

_HF_DATASETS: dict[str, tuple[str, str]] = {
    "100k": ("Mohammadta/BEAM", "100K"),
    "500k": ("Mohammadta/BEAM", "500K"),
    "1m": ("Mohammadta/BEAM", "1M"),
    "10m": ("Mohammadta/BEAM-10M", "10M"),
}


def _require_datasets() -> None:
    if importlib.util.find_spec("datasets") is None:
        print(
            "The `datasets` package is required. Install with:\n"
            "  uv pip install datasets",
            file=sys.stderr,
        )
        sys.exit(1)


def download_split(split: str, output_path: Path) -> int:
    datasets_mod = importlib.import_module("datasets")

    if split not in _HF_DATASETS:
        raise ValueError(f"Unknown split {split!r}. Valid: {sorted(_HF_DATASETS)}")

    repo, hf_split = _HF_DATASETS[split]
    print(f"[{split}] Downloading from HuggingFace {repo} (split={hf_split})…")
    ds = datasets_mod.load_dataset(repo, split=hf_split)
    rows = [dict(row) for row in ds]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f)

    print(f"[{split}] Wrote {len(rows)} conversations to {output_path}")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Download BEAM dataset splits from HuggingFace.",
    )
    parser.add_argument(
        "--split",
        nargs="+",
        required=True,
        choices=sorted(_HF_DATASETS),
        help="One or more BEAM splits to download.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--output",
        help="Output path (only valid with a single --split).",
    )
    group.add_argument(
        "--output-dir",
        help="Directory to write JSON files into. Files are named <split>.json.",
    )
    args = parser.parse_args()

    _require_datasets()

    if args.output is not None:
        if len(args.split) != 1:
            parser.error("--output requires exactly one --split value")
        download_split(args.split[0], Path(args.output))
        return

    out_dir = Path(args.output_dir)
    for split in args.split:
        download_split(split, out_dir / f"{split}.json")


if __name__ == "__main__":
    main()
