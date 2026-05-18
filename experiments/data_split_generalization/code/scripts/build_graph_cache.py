from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.chem.graph_featurizer import build_ion_pair_graph
from src.utils.io import load_config, resolve_path
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    base = cfg["_base_dir"]
    set_seed(cfg["chem"].get("seed", cfg["data"]["seed"]))
    clean_csv = resolve_path(cfg["data"]["clean_csv"], base)
    graph_cache_path = resolve_path(cfg["data"]["graph_cache_path"], base)
    failed_path = resolve_path(cfg["data"]["failed_smiles_path"], base)
    if not clean_csv.exists():
        raise FileNotFoundError(
            f"Preprocessed CSV not found: {clean_csv}\n"
            "Run preprocessing first, for example:\n"
            f"python {PROJECT_DIR / 'scripts' / 'preprocess_data.py'} --config {PROJECT_DIR / 'configs' / 'default.yaml'}"
        )
    df = pd.read_csv(clean_csv)
    smiles_list = sorted(df["IL_SMILES"].dropna().unique())
    cache = {}
    failures = []
    chem = cfg["chem"]
    for smiles in tqdm(smiles_list, desc="Build graph cache"):
        result = build_ion_pair_graph(
            smiles,
            use_3d=chem.get("use_3d", True),
            cutoff=chem.get("cross_ion_cutoff", 5.0),
            seed=chem.get("seed", 42),
            max_attempts=chem.get("max_conformer_attempts", 20),
            optimize_method=chem.get("optimize_method", "UFF"),
            use_cross_edges=cfg["model"].get("use_cross_ion_edges", True),
            cross_ion_mode=chem.get("cross_ion_mode", "deterministic_2d"),
        )
        if result.data is None:
            failures.append({"IL_SMILES": smiles, "reason": result.error})
            continue
        cache[smiles] = result.data
    graph_cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, graph_cache_path)
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    with failed_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["IL_SMILES", "reason"])
        writer.writeheader()
        writer.writerows(failures)
    print(f"saved graphs: {len(cache)}")
    print(f"failed smiles: {len(failures)}")


if __name__ == "__main__":
    main()
