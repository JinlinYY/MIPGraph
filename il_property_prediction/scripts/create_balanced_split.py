from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data.split import create_property_balanced_il_level_split
from src.utils.io import load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a property-balanced IL-level split without molecule leakage.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quantile-bins", type=int, default=4)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--swap-iterations", type=int, default=30000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    base = cfg["_base_dir"]
    split_path, diagnostics_path = create_property_balanced_il_level_split(
        resolve_path(cfg["data"]["clean_csv"], base),
        resolve_path(cfg["data"]["processed_dir"], base),
        cfg["data"]["train_ratio"],
        cfg["data"]["val_ratio"],
        cfg["data"]["test_ratio"],
        args.seed,
        args.quantile_bins,
        args.restarts,
        args.swap_iterations,
    )
    print(f"saved: {split_path}")
    print(f"saved: {diagnostics_path}")


if __name__ == "__main__":
    main()
