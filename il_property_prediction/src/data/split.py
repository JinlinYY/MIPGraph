from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _split_list(items: np.ndarray, ratios: tuple[float, float, float], seed: int) -> dict[str, list]:
    rng = np.random.default_rng(seed)
    items = np.array(items)
    rng.shuffle(items)
    n = len(items)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    train = items[:n_train]
    val = items[n_train : n_train + n_val]
    test = items[n_train + n_val :]
    return {"train": train.tolist(), "val": val.tolist(), "test": test.tolist()}


def create_il_level_split(
    clean_csv: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Path:
    df = pd.read_csv(clean_csv)
    unique_smiles = df["IL_SMILES"].dropna().unique()
    groups = _split_list(unique_smiles, (train_ratio, val_ratio, test_ratio), seed)
    split = {}
    for name, smiles in groups.items():
        split[name] = df.index[df["IL_SMILES"].isin(smiles)].astype(int).tolist()
    out = Path(output_dir) / "splits" / f"il_level_seed{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
    return out


def create_row_level_split(
    clean_csv: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Path:
    df = pd.read_csv(clean_csv)
    groups = _split_list(df.index.to_numpy(), (train_ratio, val_ratio, test_ratio), seed)
    out = Path(output_dir) / "splits" / f"row_level_seed{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(groups, f, indent=2)
    return out


def load_split(path: str | Path) -> dict[str, list[int]]:
    with Path(path).open("r", encoding="utf-8") as f:
        split = json.load(f)
    return {k: [int(i) for i in v] for k, v in split.items()}
