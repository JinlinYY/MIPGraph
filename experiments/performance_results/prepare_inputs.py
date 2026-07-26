"""Stage audited MIPGraph split outputs for the performance figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
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
    source_id_presence: list[bool] = []
    for prop in PROPERTIES:
        path = split_dir / "final" / f"test_predictions_{prop}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        has_source_id = "sample_id" in frame.columns
        source_id_presence.append(has_source_id)
        if has_source_id:
            if frame["sample_id"].isna().any():
                raise ValueError(f"{path} contains missing sample_id values.")
            if frame["sample_id"].duplicated().any():
                raise ValueError(f"{path} contains duplicate sample_id values.")
            frame = frame.rename(columns={"sample_id": "source_sample_id"})
            frame["_observation_token"] = (
                "source_id=" + frame["source_sample_id"].astype("string")
            )
        else:
            condition_keys = sample_keys(frame)
            if condition_keys.duplicated().any():
                raise ValueError(
                    f"{path} contains replicate observations at identical conditions "
                    "but has no sample_id; refusing to average them."
                )
            frame["_observation_token"] = "condition=" + condition_keys
        frame["property"] = prop
        frames.append(frame)

    if any(source_id_presence) and not all(source_id_presence):
        raise ValueError(
            "Prediction files inconsistently provide sample_id; all properties must "
            "use the same observation-identification scheme."
        )

    long = pd.concat(frames, ignore_index=True)
    metadata_variation = long.groupby("_observation_token", dropna=False)[
        KEY_COLUMNS
    ].nunique(dropna=False)
    inconsistent = metadata_variation.gt(1).any(axis=1)
    if inconsistent.any():
        examples = metadata_variation.index[inconsistent].astype(str).tolist()[:5]
        raise ValueError(
            "Observation identifiers map to conflicting IL/condition metadata: "
            + ", ".join(examples)
        )

    codes, _ = pd.factorize(long["_observation_token"], sort=True)
    long = long.drop(columns=["_observation_token"])
    long.insert(0, "sample_id", codes.astype(int))
    return long


def make_wide_predictions(long: pd.DataFrame) -> pd.DataFrame:
    if long.duplicated(["property", "sample_id"]).any():
        raise ValueError(
            "Each property must contain at most one row per sample_id; refusing "
            "to average duplicate observations."
        )
    metadata_columns = ["IL_Name", "IL_SMILES", "Temperature_K", "Pressure_kPa"]
    if "source_sample_id" in long.columns:
        metadata_columns.insert(0, "source_sample_id")
    metadata_variation = long.groupby("sample_id", dropna=False)[
        metadata_columns
    ].nunique(dropna=False)
    if metadata_variation.gt(1).any(axis=1).any():
        raise ValueError("sample_id maps to conflicting metadata across properties.")

    output_meta_columns = META_COLUMNS.copy()
    if "source_sample_id" in long.columns:
        output_meta_columns.insert(1, "source_sample_id")
    base = (
        long[output_meta_columns]
        .drop_duplicates("sample_id")
        .sort_values("sample_id")
        .reset_index(drop=True)
    )
    for prop in PROPERTIES:
        sub = long.loc[
            long["property"] == prop,
            ["sample_id", "y_true", "y_pred"],
        ].copy()
        sub = sub.rename(
            columns={
                "y_true": f"{prop}_true",
                "y_pred": f"{prop}_pred",
            }
        )
        base = base.merge(
            sub,
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
    return base


def validate_property_metrics(
    metrics: pd.DataFrame,
    *,
    allow_average: bool,
) -> None:
    """Require one and only one metric row for every plotted property."""

    if "property" not in metrics.columns:
        raise ValueError("Metrics table is missing the property column.")
    if metrics["property"].isna().any():
        raise ValueError("Metrics table contains missing property names.")

    average_count = int((metrics["property"] == "Average").sum())
    if not allow_average and average_count:
        raise ValueError(
            "selected_test_metrics.csv must not contain an Average row; "
            "the audited macro average is loaded from JSON."
        )
    if allow_average and average_count > 1:
        raise ValueError("Metrics table contains multiple Average rows.")

    property_rows = metrics.loc[metrics["property"] != "Average", "property"]
    duplicates = sorted(
        property_rows.loc[property_rows.duplicated(keep=False)].astype(str).unique()
    )
    found = set(property_rows.astype(str))
    expected = set(PROPERTIES)
    missing = sorted(expected - found)
    unexpected = sorted(found - expected)
    problems = []
    if duplicates:
        problems.append("duplicate properties: " + ", ".join(duplicates))
    if missing:
        problems.append("missing properties: " + ", ".join(missing))
    if unexpected:
        problems.append("unexpected properties: " + ", ".join(unexpected))
    if problems:
        raise ValueError("Invalid property metric rows (" + "; ".join(problems) + ").")


def load_metrics(split_dir: Path) -> pd.DataFrame:
    metrics_path = split_dir / "final" / "selected_test_metrics.csv"
    metrics_json_path = split_dir / "final" / "selected_test_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    if not metrics_json_path.exists():
        raise FileNotFoundError(metrics_json_path)

    metrics = pd.read_csv(metrics_path)
    validate_property_metrics(metrics, allow_average=False)
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
    validate_property_metrics(metrics, allow_average=True)
    return metrics


def make_metrics_json(metrics: pd.DataFrame, long: pd.DataFrame) -> dict[str, Any]:
    validate_property_metrics(metrics, allow_average=True)
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
        description="Prepare MIPGraph split outputs for the performance figure."
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
