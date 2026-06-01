from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data.ion_family import add_ion_family_columns
from src.data.split import _ion_smiles_key, write_split_summary
from src.utils.io import load_config, resolve_path


SPLIT_TO_ION = {
    "cation_within_family": "cation",
    "anion_within_family": "anion",
}


def _allocate_counts(n_items: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    if n_items <= 0:
        return 0, 0, 0
    if n_items == 1:
        return 1, 0, 0

    n_train = int(round(n_items * ratios[0]))
    n_val = int(round(n_items * ratios[1]))
    n_train = min(max(1, n_train), n_items - 1)
    n_val = min(max(0, n_val), n_items - n_train - 1)
    n_test = n_items - n_train - n_val

    if n_items >= 3 and n_val == 0:
        if n_train > 1:
            n_train -= 1
            n_val = 1
        elif n_test > 1:
            n_test -= 1
            n_val = 1
    return n_train, n_val, n_test


def _majority_family(values: pd.Series) -> str:
    modes = values.dropna().astype(str).mode()
    if not modes.empty:
        return str(modes.iloc[0])
    return "unknown"


def create_within_family_split(
    clean_csv: str | Path,
    output_dir: str | Path,
    split_type: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Path:
    if split_type not in SPLIT_TO_ION:
        valid = ", ".join(sorted(SPLIT_TO_ION))
        raise ValueError(f"Unknown within-family split_type: {split_type}. Valid values: {valid}")

    ion_type = SPLIT_TO_ION[split_type]
    family_col = "Cation_Family" if ion_type == "cation" else "Anion_Family"
    key_col = "Cation_SMILES_Key" if ion_type == "cation" else "Anion_SMILES_Key"

    df = add_ion_family_columns(pd.read_csv(clean_csv))
    df[key_col] = df.apply(lambda row: _ion_smiles_key(row, ion_type), axis=1)
    empty = df[key_col].astype(str).str.strip() == ""
    if empty.any():
        df.loc[empty, key_col] = [f"missing_{ion_type}_{idx}" for idx in df.index[empty]]

    key_family = df[[key_col, family_col]].groupby(key_col, as_index=False)[family_col].agg(_majority_family)

    rng = np.random.default_rng(seed)
    split_keys: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    family_rows = []
    ratios = (train_ratio, val_ratio, test_ratio)
    for family in sorted(key_family[family_col].dropna().unique()):
        keys = key_family.loc[key_family[family_col] == family, key_col].dropna().astype(str).to_numpy()
        rng.shuffle(keys)
        n_train, n_val, n_test = _allocate_counts(len(keys), ratios)
        groups = {
            "train": keys[:n_train],
            "val": keys[n_train : n_train + n_val],
            "test": keys[n_train + n_val : n_train + n_val + n_test],
        }
        for split_name, split_group in groups.items():
            split_keys[split_name].extend(split_group.tolist())
            family_rows.append(
                {
                    "family": family,
                    "split": split_name,
                    f"unique_{ion_type}_smiles_keys": len(split_group),
                }
            )

    split: dict[str, list[int]] = {}
    for split_name, keys in split_keys.items():
        split[split_name] = df.index[df[key_col].isin(keys)].astype(int).tolist()

    output_dir = Path(output_dir)
    split_path = output_dir / "splits" / f"{split_type}_seed{seed}.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)

    write_split_summary(df, split, split_path.with_suffix(".summary.csv"))
    _write_assignment_report(df, key_col, family_col, split_keys, output_dir / "splits" / f"{split_type}_seed{seed}.assignment.csv")
    _write_family_report(family_rows, output_dir / "splits" / f"{split_type}_seed{seed}.family_counts.csv")
    return split_path


def _write_assignment_report(
    df: pd.DataFrame,
    key_col: str,
    family_col: str,
    split_keys: dict[str, list[str]],
    output_path: Path,
) -> None:
    key_to_split = {key: split_name for split_name, keys in split_keys.items() for key in keys}
    rows = []
    for key, part in df.groupby(key_col, dropna=False):
        rows.append(
            {
                "split": key_to_split.get(str(key), ""),
                "family": _majority_family(part[family_col]),
                "ion_smiles_key": key,
                "rows": len(part),
                "unique_il_smiles": part["IL_SMILES"].nunique(dropna=True),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "family", "ion_smiles_key", "rows", "unique_il_smiles"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["family"], row["split"], str(row["ion_smiles_key"]))))


def _write_family_report(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create seen-family/unseen-ion splits by ion SMILES within each family.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split-types", default="cation_within_family,anion_within_family")
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
        paths[split_type] = create_within_family_split(
            clean_csv,
            processed_dir,
            split_type,
            cfg["data"]["train_ratio"],
            cfg["data"]["val_ratio"],
            cfg["data"]["test_ratio"],
            seed,
        )
    print({name: str(path) for name, path in paths.items()})


if __name__ == "__main__":
    main()
