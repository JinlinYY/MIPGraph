from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .ion_family import add_ion_family_columns, export_ion_family_report


PROPERTY_NAMES = [
    "Density",
    "ElectricalConductivity",
    "HeatCapacity",
    "SurfaceTension",
    "ThermalConductivity",
    "Viscosity",
]


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
    write_split_summary(df, split, out.with_suffix(".summary.csv"))
    return out


def create_il_pair_split(
    clean_csv: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Path:
    df = pd.read_csv(clean_csv)
    if {"Cation_ShortName", "Anion_ShortName"}.issubset(df.columns):
        keys = (
            df["Cation_ShortName"].fillna("").astype(str)
            + "||"
            + df["Anion_ShortName"].fillna("").astype(str)
        )
    else:
        keys = df["IL_SMILES"].fillna("").astype(str)
    groups = _split_list(keys.dropna().unique(), (train_ratio, val_ratio, test_ratio), seed)
    split = {}
    for name, group_keys in groups.items():
        split[name] = df.index[keys.isin(group_keys)].astype(int).tolist()
    out = Path(output_dir) / "splits" / f"il_pair_seed{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
    write_split_summary(df, split, out.with_suffix(".summary.csv"))
    return out


def create_family_split(
    clean_csv: str | Path,
    output_dir: str | Path,
    ion_type: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Path:
    if ion_type not in {"cation", "anion"}:
        raise ValueError("ion_type must be 'cation' or 'anion'")
    df = add_ion_family_columns(pd.read_csv(clean_csv))
    family_col = "Cation_Family" if ion_type == "cation" else "Anion_Family"
    families = df[family_col].dropna().unique()
    groups = _split_list(families, (train_ratio, val_ratio, test_ratio), seed)
    split = {}
    for name, family_names in groups.items():
        split[name] = df.index[df[family_col].isin(family_names)].astype(int).tolist()
    split_name = f"{ion_type}_family"
    out = Path(output_dir) / "splits" / f"{split_name}_seed{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
    write_split_summary(df, split, out.with_suffix(".summary.csv"))
    export_ion_family_report(df, Path(output_dir) / "ion_family_assignment.csv")
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
    write_split_summary(df, groups, out.with_suffix(".summary.csv"))
    return out


def create_random_point_split(
    clean_csv: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Path:
    df = pd.read_csv(clean_csv)
    split = _split_list(df.index.to_numpy(), (train_ratio, val_ratio, test_ratio), seed)
    out = Path(output_dir) / "splits" / f"random_point_seed{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
    write_split_summary(df, split, out.with_suffix(".summary.csv"))
    return out


def create_split(
    split_type: str,
    clean_csv: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Path:
    if split_type == "row_level":
        return create_row_level_split(clean_csv, output_dir, train_ratio, val_ratio, test_ratio, seed)
    if split_type == "random_point":
        return create_random_point_split(clean_csv, output_dir, train_ratio, val_ratio, test_ratio, seed)
    if split_type == "il_level":
        return create_il_level_split(clean_csv, output_dir, train_ratio, val_ratio, test_ratio, seed)
    if split_type == "il_pair":
        return create_il_pair_split(clean_csv, output_dir, train_ratio, val_ratio, test_ratio, seed)
    if split_type == "cation_family":
        return create_family_split(clean_csv, output_dir, "cation", train_ratio, val_ratio, test_ratio, seed)
    if split_type == "anion_family":
        return create_family_split(clean_csv, output_dir, "anion", train_ratio, val_ratio, test_ratio, seed)
    raise ValueError(f"Unknown split_type: {split_type}")


def create_standard_generalization_splits(
    clean_csv: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, Path]:
    return {
        name: create_split(name, clean_csv, output_dir, train_ratio, val_ratio, test_ratio, seed)
        for name in ["row_level", "random_point", "il_level", "il_pair", "cation_family", "anion_family"]
    }


def write_split_summary(df: pd.DataFrame, split: dict[str, list[int]], output_path: str | Path, properties: list[str] | None = None) -> Path:
    properties = properties or PROPERTY_NAMES
    tagged = add_ion_family_columns(df)
    rows = []
    for split_name in ["train", "val", "test"]:
        idx = split.get(split_name, [])
        part = tagged.iloc[idx]
        row = {
            "split": split_name,
            "rows": len(part),
            "unique_il_smiles": int(part["IL_SMILES"].nunique(dropna=True)) if "IL_SMILES" in part else 0,
            "unique_cations": int(part["Cation_ShortName"].nunique(dropna=True)) if "Cation_ShortName" in part else 0,
            "unique_anions": int(part["Anion_ShortName"].nunique(dropna=True)) if "Anion_ShortName" in part else 0,
            "cation_families": "|".join(sorted(part["Cation_Family"].dropna().unique())),
            "anion_families": "|".join(sorted(part["Anion_Family"].dropna().unique())),
        }
        for prop in properties:
            col = f"{prop}_ActualValue"
            if col in part.columns:
                row[f"{prop}_labels"] = int(pd.to_numeric(part[col], errors="coerce").notna().sum())
        rows.append(row)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def load_split(path: str | Path) -> dict[str, list[int]]:
    with Path(path).open("r", encoding="utf-8") as f:
        split = json.load(f)
    return {k: [int(i) for i in v] for k, v in split.items()}
