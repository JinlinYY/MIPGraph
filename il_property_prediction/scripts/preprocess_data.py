from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data.preprocess import preprocess_excel
from src.data.split import create_il_level_split, create_row_level_split
from src.utils.io import load_config, resolve_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    base = cfg["_base_dir"]
    raw_path = resolve_path(args.input or cfg["data"]["raw_path"], base)
    processed_dir = resolve_path(cfg["data"]["processed_dir"], base)
    result = preprocess_excel(raw_path, processed_dir, cfg["data"].get("sheet_name", "Merged"), cfg["properties"]["names"])
    create_il_level_split(result["clean_csv"], processed_dir, cfg["data"]["train_ratio"], cfg["data"]["val_ratio"], cfg["data"]["test_ratio"], cfg["data"]["seed"])
    create_row_level_split(result["clean_csv"], processed_dir, cfg["data"]["train_ratio"], cfg["data"]["val_ratio"], cfg["data"]["test_ratio"], cfg["data"]["seed"])
    print(result["report"])


if __name__ == "__main__":
    main()
