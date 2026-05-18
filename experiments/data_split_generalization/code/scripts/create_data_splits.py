from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data.split import create_split
from src.utils.io import load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create data splits and summary reports for generalization experiments.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split-types", default="random_point,il_pair,cation_family,anion_family")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    base = cfg["_base_dir"]
    seed = args.seed if args.seed is not None else cfg["data"]["seed"]
    clean_csv = resolve_path(cfg["data"]["clean_csv"], base)
    processed_dir = resolve_path(cfg["data"]["processed_dir"], base)
    split_types = [item.strip() for item in args.split_types.split(",") if item.strip()]
    paths = {}
    for split_type in split_types:
        paths[split_type] = create_split(
            split_type,
            clean_csv,
            processed_dir,
            cfg["data"]["train_ratio"],
            cfg["data"]["val_ratio"],
            cfg["data"]["test_ratio"],
            seed,
        )
    print({name: str(path) for name, path in paths.items()})


if __name__ == "__main__":
    main()
