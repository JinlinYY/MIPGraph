from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data.split import create_il_level_split, create_row_level_split, load_split
from src.training.trainer import Trainer
from src.utils.io import load_config, resolve_path
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    base = cfg["_base_dir"]
    seed = args.seed if args.seed is not None else cfg["training"].get("seed", cfg["data"]["seed"])
    cfg["training"]["seed"] = seed
    cfg["data"]["seed"] = seed
    if args.run_name:
        cfg["outputs"]["run_name"] = args.run_name
    set_seed(seed)
    arrays_path = resolve_path(cfg["data"]["arrays_path"], base)
    clean_csv = resolve_path(cfg["data"]["clean_csv"], base)
    processed_dir = resolve_path(cfg["data"]["processed_dir"], base)
    graph_cache = resolve_path(cfg["data"]["graph_cache_path"], base)
    if not arrays_path.exists() or not clean_csv.exists():
        raise FileNotFoundError("Preprocessed files are missing. Run scripts/preprocess_data.py first.")
    if not graph_cache.exists():
        raise FileNotFoundError("Graph cache is missing. Run scripts/build_graph_cache.py first.")
    split_name = cfg["data"].get("split_type", "il_level")
    split_path = processed_dir / "splits" / f"{split_name}_seed{seed}.json"
    if not split_path.exists():
        if split_name == "row_level":
            create_row_level_split(clean_csv, processed_dir, cfg["data"]["train_ratio"], cfg["data"]["val_ratio"], cfg["data"]["test_ratio"], seed)
        else:
            create_il_level_split(clean_csv, processed_dir, cfg["data"]["train_ratio"], cfg["data"]["val_ratio"], cfg["data"]["test_ratio"], seed)
    cfg["data"]["arrays_path"] = str(arrays_path)
    run_name = cfg["outputs"].get("run_name", "")
    for k, v in cfg["outputs"].items():
        if k == "run_name":
            continue
        resolved = resolve_path(v, base)
        if run_name and run_name != "default" and k.endswith("_dir"):
            resolved = resolved / run_name
        cfg["outputs"][k] = str(resolved)
    arrays = dict(np.load(arrays_path, allow_pickle=True))
    trainer = Trainer(cfg, arrays, clean_csv, graph_cache, load_split(split_path))
    print(trainer.train())


if __name__ == "__main__":
    main()
