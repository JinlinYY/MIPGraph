"""Stage current-best MIPGraph outputs for manuscript plotting scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_ROOT = (
    REPO_ROOT / "il_property_prediction" / "outputs" / "mps_weak_merged_validation"
)
PROPERTIES = [
    "Density",
    "Viscosity",
    "ElectricalConductivity",
    "HeatCapacity",
    "SurfaceTension",
    "ThermalConductivity",
]
SPLITS = [
    "random_point",
    "random_il_level",
    "property_balanced_il_level",
    "ion_family",
]
KEY_COLUMNS = ["IL_SMILES", "Temperature_K", "Pressure_kPa"]
META_COLUMNS = ["sample_id", "IL_Name", "IL_SMILES", "Temperature_K", "Pressure_kPa"]


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if pd.isna(value):
        return None
    return value


def sample_keys(frame: pd.DataFrame) -> pd.Series:
    keys = frame[KEY_COLUMNS].copy()
    keys["Temperature_K"] = pd.to_numeric(keys["Temperature_K"], errors="coerce").round(6)
    keys["Pressure_kPa"] = pd.to_numeric(keys["Pressure_kPa"], errors="coerce").round(6)
    return keys.astype("string").fillna("__NA__").agg("|".join, axis=1)


def load_long_predictions(split_dir: Path) -> pd.DataFrame:
    frames = []
    for prop in PROPERTIES:
        path = split_dir / "final" / f"test_predictions_{prop}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame["property"] = prop
        frames.append(frame)

    long = pd.concat(frames, ignore_index=True)
    codes, _ = pd.factorize(sample_keys(long), sort=True)
    if "sample_id" in long.columns:
        long = long.drop(columns=["sample_id"])
    long.insert(0, "sample_id", codes.astype(int))
    return long


def make_wide_predictions(long: pd.DataFrame) -> pd.DataFrame:
    base = (
        long[META_COLUMNS]
        .drop_duplicates("sample_id")
        .sort_values("sample_id")
        .reset_index(drop=True)
    )
    for prop in PROPERTIES:
        sub = long.loc[
            long["property"] == prop,
            ["sample_id", "y_true", "y_pred"],
        ].copy()
        sub = sub.groupby("sample_id", as_index=False).mean(numeric_only=True)
        sub = sub.rename(
            columns={
                "y_true": f"{prop}_true",
                "y_pred": f"{prop}_pred",
            }
        )
        base = base.merge(sub, on="sample_id", how="left")
    return base


def load_metrics(split_dir: Path) -> pd.DataFrame:
    metrics_path = split_dir / "final" / "selected_test_metrics.csv"
    metrics_json_path = split_dir / "final" / "selected_test_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    if not metrics_json_path.exists():
        raise FileNotFoundError(metrics_json_path)

    metrics = pd.read_csv(metrics_path)
    with metrics_json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    average = raw.get("average")
    if average is None:
        numeric_cols = metrics.select_dtypes(include=["number"]).columns
        average = {col: float(metrics[col].mean()) for col in numeric_cols}
        average["property"] = "Average"
        average["source"] = "macro_average"
        average["checkpoint"] = ""
        average["val_score"] = np.nan
    else:
        average = dict(average)
        average["property"] = "Average"
        average.setdefault("source", "macro_average")
        average.setdefault("checkpoint", "")
        average.setdefault("val_score", np.nan)

    metrics = pd.concat([metrics, pd.DataFrame([average])], ignore_index=True)
    return metrics


def make_metrics_json(metrics: pd.DataFrame, long: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for prop in PROPERTIES:
        row = metrics.loc[metrics["property"] == prop].iloc[0].to_dict()
        row = {key: _json_scalar(value) for key, value in row.items()}
        row["label_count"] = int((long["property"] == prop).sum())
        out[prop] = row
    return out


def stage_split(run_root: Path, output_root: Path, split: str) -> None:
    split_dir = run_root / split
    output_dir = output_root / split
    output_dir.mkdir(parents=True, exist_ok=True)

    long = load_long_predictions(split_dir)
    wide = make_wide_predictions(long)
    metrics = load_metrics(split_dir)
    metrics_json = make_metrics_json(metrics, long)

    long.to_csv(output_dir / "test_predictions.csv", index=False)
    wide.to_csv(output_dir / "test_predictions_wide.csv", index=False)
    metrics.to_csv(output_dir / "test_metrics_log.csv", index=False)
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare current-best MIPGraph figure inputs."
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else run_root / "figure_inputs"
    )
    for split in SPLITS:
        stage_split(run_root, output_root, split)
    print(f"Wrote staged figure inputs to {output_root}")


if __name__ == "__main__":
    main()
