from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROPERTY_NAMES = ["Density", "ElectricalConductivity", "HeatCapacity", "SurfaceTension", "ThermalConductivity", "Viscosity"]


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


def _quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.asarray([-np.inf, np.inf])
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 2:
        return np.asarray([-np.inf, np.inf])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _balanced_group_features(df: pd.DataFrame, n_quantile_bins: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    groups = np.asarray(sorted(df["IL_SMILES"].dropna().astype(str).unique()))
    group_index = {smiles: idx for idx, smiles in enumerate(groups)}
    row_groups = df["IL_SMILES"].astype(str).map(group_index).to_numpy(dtype=np.int64)
    columns: list[np.ndarray] = []
    names: list[str] = []

    def add_feature(name: str, values: np.ndarray) -> None:
        columns.append(np.bincount(row_groups, weights=values.astype(np.float64), minlength=len(groups)))
        names.append(name)

    add_feature("row_count", np.ones(len(df), dtype=np.float64))
    temperature = pd.to_numeric(df["Temperature_K"], errors="coerce").to_numpy(dtype=float)
    for bin_idx in range(len(_quantile_bins(temperature, n_quantile_bins)) - 1):
        edges = _quantile_bins(temperature, n_quantile_bins)
        membership = np.isfinite(temperature) & (np.digitize(temperature, edges[1:-1]) == bin_idx)
        add_feature(f"temperature_bin_{bin_idx}", membership.astype(float))
    pressure = pd.to_numeric(df["Pressure_kPa"], errors="coerce").fillna(101.325).to_numpy(dtype=float)
    log_pressure = np.log(np.maximum(pressure, 1e-8))
    pressure_edges = _quantile_bins(log_pressure, n_quantile_bins)
    for bin_idx in range(len(pressure_edges) - 1):
        membership = np.digitize(log_pressure, pressure_edges[1:-1]) == bin_idx
        add_feature(f"pressure_bin_{bin_idx}", membership.astype(float))

    for prop in PROPERTY_NAMES:
        values = pd.to_numeric(df[f"{prop}_ActualValue"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(values) & (values > 0)
        label_counts = np.bincount(row_groups, weights=valid.astype(float), minlength=len(groups))
        columns.append((label_counts > 0).astype(float))
        names.append(f"{prop}_IL_presence")
        columns.append(label_counts.astype(float))
        names.append(f"{prop}_label_count")
        log_values = np.full(len(values), np.nan, dtype=float)
        log_values[valid] = np.log(values[valid])
        edges = _quantile_bins(log_values, n_quantile_bins)
        for bin_idx in range(len(edges) - 1):
            membership = valid & (np.digitize(log_values, edges[1:-1]) == bin_idx)
            add_feature(f"{prop}_log_bin_{bin_idx}", membership.astype(float))
    return groups, np.stack(columns, axis=1), names


def _split_objective(fold_sums: np.ndarray, ratios: np.ndarray, totals: np.ndarray) -> float:
    expected = ratios[:, None] * totals[None, :]
    relative_error = (fold_sums - expected) / np.maximum(expected, 1.0)
    return float(np.mean(relative_error**2))


def create_property_balanced_il_level_split(
    clean_csv: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    n_quantile_bins: int = 4,
    restarts: int = 8,
    swap_iterations: int = 30000,
) -> tuple[Path, Path]:
    """Balance label availability and condition/target quantiles without splitting an IL across folds."""
    df = pd.read_csv(clean_csv)
    groups, features, feature_names = _balanced_group_features(df, n_quantile_bins)
    ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    ratios = ratios / ratios.sum()
    n_groups = len(groups)
    fold_sizes = np.asarray(
        [int(round(n_groups * ratios[0])), int(round(n_groups * ratios[1]))], dtype=np.int64
    )
    fold_sizes = np.append(fold_sizes, n_groups - int(fold_sizes.sum()))
    totals = features.sum(axis=0)
    rng = np.random.default_rng(seed)
    best_assignment = None
    best_objective = float("inf")

    for _ in range(max(1, restarts)):
        permutation = rng.permutation(n_groups)
        assignment = np.empty(n_groups, dtype=np.int8)
        start = 0
        for fold_idx, size in enumerate(fold_sizes):
            assignment[permutation[start : start + size]] = fold_idx
            start += int(size)
        fold_sums = np.stack([features[assignment == idx].sum(axis=0) for idx in range(3)])
        objective = _split_objective(fold_sums, ratios, totals)
        for _ in range(max(0, swap_iterations)):
            first, second = rng.integers(0, n_groups, size=2)
            fold_a, fold_b = int(assignment[first]), int(assignment[second])
            if fold_a == fold_b:
                continue
            candidate_sums = fold_sums.copy()
            candidate_sums[fold_a] += features[second] - features[first]
            candidate_sums[fold_b] += features[first] - features[second]
            candidate_objective = _split_objective(candidate_sums, ratios, totals)
            if candidate_objective < objective:
                assignment[first], assignment[second] = fold_b, fold_a
                fold_sums = candidate_sums
                objective = candidate_objective
        if objective < best_objective:
            best_assignment = assignment.copy()
            best_objective = objective

    split = {}
    fold_names = ["train", "val", "test"]
    for fold_idx, fold_name in enumerate(fold_names):
        fold_groups = set(groups[best_assignment == fold_idx])
        split[fold_name] = df.index[df["IL_SMILES"].astype(str).isin(fold_groups)].astype(int).tolist()

    split_dir = Path(output_dir) / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_path = split_dir / f"il_level_property_balanced_seed{seed}.json"
    with split_path.open("w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
    diagnostics = {
        "method": "property_balanced_il_level",
        "seed": seed,
        "ratios": dict(zip(fold_names, ratios.tolist())),
        "objective": best_objective,
        "feature_names": feature_names,
        "folds": {},
    }
    for fold_name, indices in split.items():
        subset = df.loc[indices]
        diagnostics["folds"][fold_name] = {
            "rows": len(indices),
            "unique_ils": int(subset["IL_SMILES"].nunique()),
            "property_labels": {
                prop: int(
                    (
                        np.isfinite(pd.to_numeric(subset[f"{prop}_ActualValue"], errors="coerce"))
                        & (pd.to_numeric(subset[f"{prop}_ActualValue"], errors="coerce") > 0)
                    ).sum()
                )
                for prop in PROPERTY_NAMES
            },
        }
    diagnostics_path = split_dir / f"il_level_property_balanced_seed{seed}_diagnostics.json"
    with diagnostics_path.open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)
    return split_path, diagnostics_path


def create_train_val_balanced_split_keep_test(
    clean_csv: str | Path,
    base_split_path: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    n_quantile_bins: int = 4,
    restarts: int = 8,
    swap_iterations: int = 30000,
) -> tuple[Path, Path]:
    """Rebalance train/validation IL groups while preserving the test set exactly."""
    df = pd.read_csv(clean_csv)
    base_split = load_split(base_split_path)
    pool_indices = list(dict.fromkeys(base_split["train"] + base_split["val"]))
    pool_df = df.loc[pool_indices].copy()
    groups, features, feature_names = _balanced_group_features(pool_df, n_quantile_bins)
    train_group_count = int(df.loc[base_split["train"], "IL_SMILES"].nunique())
    val_group_count = len(groups) - train_group_count
    ratios = np.asarray([train_group_count, val_group_count], dtype=np.float64) / len(groups)
    totals = features.sum(axis=0)
    rng = np.random.default_rng(seed)
    best_assignment = None
    best_objective = float("inf")

    for _ in range(max(1, restarts)):
        permutation = rng.permutation(len(groups))
        assignment = np.ones(len(groups), dtype=np.int8)
        assignment[permutation[:train_group_count]] = 0
        fold_sums = np.stack([features[assignment == idx].sum(axis=0) for idx in range(2)])
        objective = _split_objective(fold_sums, ratios, totals)
        for _ in range(max(0, swap_iterations)):
            first, second = rng.integers(0, len(groups), size=2)
            fold_a, fold_b = int(assignment[first]), int(assignment[second])
            if fold_a == fold_b:
                continue
            candidate_sums = fold_sums.copy()
            candidate_sums[fold_a] += features[second] - features[first]
            candidate_sums[fold_b] += features[first] - features[second]
            candidate_objective = _split_objective(candidate_sums, ratios, totals)
            if candidate_objective < objective:
                assignment[first], assignment[second] = fold_b, fold_a
                fold_sums = candidate_sums
                objective = candidate_objective
        if objective < best_objective:
            best_assignment = assignment.copy()
            best_objective = objective

    train_groups = set(groups[best_assignment == 0])
    val_groups = set(groups[best_assignment == 1])
    pool_set = set(pool_indices)
    split = {
        "train": df.index[df["IL_SMILES"].astype(str).isin(train_groups) & df.index.to_series().isin(pool_set)].astype(int).tolist(),
        "val": df.index[df["IL_SMILES"].astype(str).isin(val_groups) & df.index.to_series().isin(pool_set)].astype(int).tolist(),
        "test": list(base_split["test"]),
    }
    split_dir = Path(output_dir) / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_path = split_dir / f"il_level_train_val_balanced_keep_test_seed{seed}.json"
    with split_path.open("w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
    diagnostics = {
        "method": "property_balanced_train_val_with_fixed_test",
        "seed": seed,
        "base_split": str(Path(base_split_path).resolve()),
        "test_indices_unchanged": split["test"] == base_split["test"],
        "objective": best_objective,
        "feature_names": feature_names,
        "folds": {},
    }
    for fold_name, indices in split.items():
        subset = df.loc[indices]
        diagnostics["folds"][fold_name] = {
            "rows": len(indices),
            "unique_ils": int(subset["IL_SMILES"].nunique()),
            "property_labels": {
                prop: int(
                    (
                        np.isfinite(pd.to_numeric(subset[f"{prop}_ActualValue"], errors="coerce"))
                        & (pd.to_numeric(subset[f"{prop}_ActualValue"], errors="coerce") > 0)
                    ).sum()
                )
                for prop in PROPERTY_NAMES
            },
        }
    diagnostics_path = split_dir / f"il_level_train_val_balanced_keep_test_seed{seed}_diagnostics.json"
    with diagnostics_path.open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)
    return split_path, diagnostics_path


def load_split(path: str | Path) -> dict[str, list[int]]:
    with Path(path).open("r", encoding="utf-8") as f:
        split = json.load(f)
    return {k: [int(i) for i in v] for k, v in split.items()}
