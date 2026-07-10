from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data.preprocess import PROPERTY_NAMES, preprocess_excel


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess augmented labels while preserving original evaluation masks and splits.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--evaluation-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-processed-dir", type=Path, default=PROJECT_DIR / "data" / "processed")
    parser.add_argument("--sheet", default="Merged")
    args = parser.parse_args()

    result = preprocess_excel(
        args.input,
        args.output_dir,
        sheet_name=args.sheet,
        properties=PROPERTY_NAMES,
        evaluation_reference_path=args.evaluation_reference,
    )
    graph_cache = args.source_processed_dir / "graph_cache.pt"
    if graph_cache.exists():
        shutil.copy2(graph_cache, args.output_dir / "graph_cache.pt")
    source_splits = args.source_processed_dir / "splits"
    target_splits = args.output_dir / "splits"
    if source_splits.exists():
        target_splits.mkdir(parents=True, exist_ok=True)
        for split_path in source_splits.glob("*.json"):
            shutil.copy2(split_path, target_splits / split_path.name)
    print(result["report"])
    print(f"output_dir: {args.output_dir}")


if __name__ == "__main__":
    main()
