from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import METRIC_COLUMNS, compute_basic_metrics, compute_metrics_by_groups


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    default_input = repo_root / "il_property_prediction" / "outputs" / "predictions" / "finetune_viscosity_from_weak_seed42"
    default_output = repo_root / "exp2_performance" / "outputs"
    parser = argparse.ArgumentParser(description="Evaluate IL property prediction results.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input,
        help="Directory containing prediction files such as test_predictions.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Base output directory to store metrics/tables/figures.",
    )
    parser.add_argument(
        "--prediction-file",
        default="test_predictions.csv",
        help="Prediction file name inside --input-dir.",
    )
    parser.add_argument(
        "--metric-space",
        choices=["log", "raw"],
        default="log",
        help="Compute metrics in log space by default; raw keeps the original physical-unit scale.",
    )
    parser.add_argument(
        "--log-eps",
        type=float,
        default=1e-8,
        help="Small positive offset used for log-space metrics.",
    )
    return parser.parse_args()


def ensure_output_dirs(output_dir: Path) -> tuple[Path, Path]:
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir, tables_dir


def load_predictions(input_dir: Path, prediction_file: str, metric_space: str, log_eps: float) -> pd.DataFrame:
    pred_path = input_dir / prediction_file
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    df = pd.read_csv(pred_path)
    required = {"property", "y_true", "y_pred"}
    if not required.issubset(df.columns):
        missing = sorted(required.difference(df.columns))
        raise ValueError(f"Missing required columns in {pred_path}: {missing}")
    if "split" not in df.columns:
        df["split"] = "test"
    df = df.dropna(subset=["property", "y_true", "y_pred"]).copy()
    df["y_true_raw"] = df["y_true"].astype(float)
    df["y_pred_raw"] = df["y_pred"].astype(float)
    if "absolute_error" in df.columns:
        df["absolute_error_raw"] = df["absolute_error"].astype(float)
    else:
        df["absolute_error_raw"] = (df["y_pred_raw"] - df["y_true_raw"]).abs()
    df["metric_space"] = metric_space
    if metric_space == "log":
        if {"y_true_log", "y_pred_log"}.issubset(df.columns):
            df["y_true_log"] = df["y_true_log"].astype(float)
            df["y_pred_log"] = df["y_pred_log"].astype(float)
        else:
            positive = (df["y_true_raw"] > 0) & (df["y_pred_raw"] > 0)
            df = df.loc[positive].copy()
            df["y_true_log"] = np.log(df["y_true_raw"] + log_eps)
            df["y_pred_log"] = np.log(df["y_pred_raw"] + log_eps)
        df["y_true"] = df["y_true_log"]
        df["y_pred"] = df["y_pred_log"]
        df["absolute_error"] = (df["y_pred_log"] - df["y_true_log"]).abs()
    else:
        df["absolute_error"] = df["absolute_error_raw"]
    return df


def build_metric_tables(pred_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_by_split_property = compute_metrics_by_groups(pred_df, ["split", "property"])
    metrics_by_split = compute_metrics_by_groups(pred_df, ["split"])

    overall = compute_basic_metrics(pred_df["y_true"].values, pred_df["y_pred"].values)
    overall_df = pd.DataFrame([{"split": "all", "property": "Average", **overall.to_dict()}])

    combined = pd.concat([metrics_by_split_property, overall_df], ignore_index=True, sort=False)
    if "metric_space" in pred_df.columns:
        metric_space = str(pred_df["metric_space"].dropna().iloc[0])
        for table in (metrics_by_split_property, metrics_by_split, combined):
            table["metric_space"] = metric_space
            if metric_space == "log":
                table["MAPE"] = np.nan
    return metrics_by_split_property, metrics_by_split, combined


def write_outputs(
    pred_df: pd.DataFrame,
    metrics_by_split_property: pd.DataFrame,
    metrics_by_split: pd.DataFrame,
    combined: pd.DataFrame,
    tables_dir: Path,
) -> None:
    pred_df.to_csv(tables_dir / "predictions_long.csv", index=False)
    metrics_by_split_property.to_csv(tables_dir / "metrics_by_split_property.csv", index=False)
    metrics_by_split.to_csv(tables_dir / "metrics_by_split.csv", index=False)
    combined.to_csv(tables_dir / "metrics_summary.csv", index=False)

    pivot = metrics_by_split_property.pivot(index="property", columns="split", values="MAE")
    pivot = pivot.sort_index()
    pivot_path = tables_dir / "mae_pivot.csv"
    try:
        pivot.to_csv(pivot_path)
    except PermissionError:
        print(f"Skipped locked auxiliary table: {pivot_path}")


def main() -> None:
    args = parse_args()
    _, tables_dir = ensure_output_dirs(args.output_dir)
    pred_df = load_predictions(args.input_dir, args.prediction_file, args.metric_space, args.log_eps)
    metrics_by_split_property, metrics_by_split, combined = build_metric_tables(pred_df)
    write_outputs(pred_df, metrics_by_split_property, metrics_by_split, combined, tables_dir)

    cols = ["split", "property", *METRIC_COLUMNS]
    display = combined[cols].copy()
    with pd.option_context("display.max_rows", 100, "display.width", 140):
        print(display.to_string(index=False))


if __name__ == "__main__":
    main()
