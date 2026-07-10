from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data.dataset import PROPERTY_NAMES
from src.utils.io import load_config, resolve_path, save_json


def _parse_csv_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _parse_target_from_run_name(run_name: str) -> str | None:
    prefix = "property_specialist_"
    if not run_name.startswith(prefix):
        return None
    suffix = run_name[len(prefix) :]
    for prop in PROPERTY_NAMES:
        marker = f"{prop}_seed"
        if suffix.startswith(marker):
            return prop
    return None


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _target_val_score(metrics: dict[str, Any], prop: str) -> float | None:
    item = metrics.get("log_space", {}).get(prop, {})
    r2 = item.get("log_R2")
    nmae = item.get("log_NMAE")
    if r2 is None or nmae is None:
        return None
    return float(r2) - 0.2 * float(nmae)


def _metric_item(metrics: dict[str, Any], prop: str, key: str) -> float | None:
    value = metrics.get(prop, {}).get(key)
    return float(value) if value is not None else None


def _log_metric_item(metrics: dict[str, Any], prop: str, key: str) -> float | None:
    value = metrics.get("log_space", {}).get(prop, {}).get(key)
    return float(value) if value is not None else None


def _discover_runs(root: Path) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    metric_root = root / "metrics"
    checkpoint_root = root / "checkpoints"
    rows = []
    for val_path in sorted(metric_root.glob("*/val_metrics.json")):
        run_name = val_path.parent.name
        prop = _parse_target_from_run_name(run_name)
        if prop is None:
            continue
        test_path = val_path.parent / "test_metrics.json"
        ckpt_path = checkpoint_root / run_name / "best_model.pt"
        if not test_path.exists() or not ckpt_path.exists():
            continue
        val_metrics = _load_json(val_path)
        test_metrics = _load_json(test_path)
        score = _target_val_score(val_metrics, prop)
        if score is None:
            continue
        seed_text = run_name.rsplit("_seed", 1)[-1]
        try:
            seed = int(seed_text)
        except ValueError:
            seed = None
        row = {
            "target_property": prop,
            "seed": seed,
            "run_name": run_name,
            "checkpoint": str(ckpt_path),
            "val_score": score,
            "val_log_R2": _log_metric_item(val_metrics, prop, "log_R2"),
            "val_log_MAE": _log_metric_item(val_metrics, prop, "log_MAE"),
            "val_log_RMSE": _log_metric_item(val_metrics, prop, "log_RMSE"),
            "val_log_NMAE": _log_metric_item(val_metrics, prop, "log_NMAE"),
            "test_log_R2": _log_metric_item(test_metrics, prop, "log_R2"),
            "test_log_MAE": _log_metric_item(test_metrics, prop, "log_MAE"),
            "test_log_RMSE": _log_metric_item(test_metrics, prop, "log_RMSE"),
            "test_log_NMAE": _log_metric_item(test_metrics, prop, "log_NMAE"),
            "test_raw_R2": _metric_item(test_metrics, prop, "R2"),
            "test_raw_NMAE": _metric_item(test_metrics, prop, "NMAE"),
        }
        rows.append(row)
    rows.sort(key=lambda item: (item["target_property"], -(item["val_score"])))
    return rows, pd.DataFrame(rows)


def _load_prediction_table(root: Path, run_name: str, split: str = "test") -> pd.DataFrame:
    pred_dir = root / "predictions" / run_name
    candidates = [
        pred_dir / f"{split}_predictions.csv",
        pred_dir / f"{split}_predictions_long.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "property" in df.columns:
            return df
    raise FileNotFoundError(f"Long-format {split} predictions not found for {run_name} in {pred_dir}")


def _prediction_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["IL_SMILES"].astype(str)
        + "|"
        + df["Temperature_K"].round(8).astype(str)
        + "|"
        + df["Pressure_kPa"].round(8).astype(str)
        + "|"
        + df["property"].astype(str)
        + "|"
        + df["y_true"].round(12).astype(str)
    )


def _average_property_predictions(root: Path, run_names: list[str], prop: str) -> pd.DataFrame:
    frames = []
    for run_name in run_names:
        df = _load_prediction_table(root, run_name)
        df = df[df["property"] == prop].copy()
        if df.empty:
            raise ValueError(f"No test predictions for {prop} in {run_name}")
        df["_key"] = _prediction_key(df)
        df = df.sort_values("_key").reset_index(drop=True)
        frames.append(df)
    base = frames[0][["IL_Name", "IL_SMILES", "Temperature_K", "Pressure_kPa", "property", "y_true", "_key"]].copy()
    pred_logs = []
    for frame in frames:
        if list(frame["_key"]) != list(base["_key"]):
            raise ValueError(f"Prediction rows do not align for {prop}: {run_names}")
        pred_logs.append(np.log(np.maximum(frame["y_pred"].to_numpy(dtype=float), 1e-12)))
    base["y_pred"] = np.exp(np.mean(np.stack(pred_logs, axis=0), axis=0))
    base["source_runs"] = ";".join(run_names)
    return base.drop(columns=["_key"])


def _compute_property_metrics(y_true: np.ndarray, y_pred: np.ndarray, normalization_std: float) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0) & (y_pred > 0)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    y_true_log = np.log(y_true)
    y_pred_log = np.log(y_pred)
    err_log = y_pred_log - y_true_log
    err = y_pred - y_true
    log_mae = float(np.mean(np.abs(err_log)))
    log_rmse = float(np.sqrt(np.mean(err_log**2)))
    log_r2 = float(r2_score(y_true_log, y_pred_log)) if len(y_true_log) > 1 else math.nan
    log_nmae = float(log_mae / max(abs(float(normalization_std)), 1e-12))
    log_nmae_range = float(log_mae / max(float(np.max(y_true_log) - np.min(y_true_log)), 1e-12))
    physical_nmae = float(np.mean(np.abs(err)) / max(float(np.mean(np.abs(y_true))), 1e-12))
    return {
        "label_count": int(len(y_true)),
        "log_MAE": log_mae,
        "log_RMSE": log_rmse,
        "log_R2": log_r2,
        "log_NMAE": log_nmae,
        "log_normalization_std": float(normalization_std),
        "log_NMAE_range": log_nmae_range,
        "NMAE": physical_nmae,
    }


def _build_mode_metrics(root: Path, selected: dict[str, list[str]], mode: str) -> list[dict[str, Any]]:
    rows = []
    for prop in PROPERTY_NAMES:
        pred_df = _average_property_predictions(root, selected[prop], prop)
        normalization_stds = []
        for run_name in selected[prop]:
            val_metrics = _load_json(root / "metrics" / run_name / "val_metrics.json")
            value = val_metrics.get("log_space", {}).get(prop, {}).get("log_normalization_std")
            if value is None:
                raise KeyError(f"Missing log_normalization_std for {prop} in {run_name}")
            normalization_stds.append(float(value))
        if not np.allclose(normalization_stds, normalization_stds[0], rtol=1e-6, atol=1e-8):
            raise ValueError(f"Inconsistent log normalization scales for {prop}: {normalization_stds}")
        item = _compute_property_metrics(
            pred_df["y_true"].to_numpy(),
            pred_df["y_pred"].to_numpy(),
            normalization_stds[0],
        )
        item.update({"mode": mode, "property": prop, "source_runs": ";".join(selected[prop])})
        rows.append(item)
    average = {"mode": mode, "property": "Average", "source_runs": ""}
    for key in ["log_MAE", "log_RMSE", "log_R2", "log_NMAE", "log_NMAE_range", "NMAE"]:
        average[key] = float(np.mean([row[key] for row in rows]))
    average["label_count"] = int(sum(row["label_count"] for row in rows))
    rows.append(average)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select property-specialized checkpoints by validation score and evaluate fixed single/ensemble predictions on test."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--root", default="outputs/property_specialists")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--extra-top-k",
        default="5",
        help="Optional additional fixed top-k ensembles to report, comma separated. Empty string disables extras.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = resolve_path(args.root, cfg["_base_dir"])
    root.mkdir(parents=True, exist_ok=True)

    discovered, results_df = _discover_runs(root)
    if results_df.empty:
        raise FileNotFoundError(f"No completed property specialist runs found under {root}")
    results_path = root / "results_all.csv"
    results_df.to_csv(results_path, index=False)

    by_prop: dict[str, list[dict[str, Any]]] = {}
    for prop in PROPERTY_NAMES:
        items = [row for row in discovered if row["target_property"] == prop]
        if not items:
            raise ValueError(f"No completed specialist runs found for {prop}")
        items.sort(key=lambda item: item["val_score"], reverse=True)
        by_prop[prop] = items

    top_ks = [args.top_k] + [k for k in _parse_csv_ints(args.extra_top_k) if k != args.top_k]
    selection_doc: dict[str, Any] = {
        "selection_rule": "val_score = val_log_R2_target - 0.2 * val_log_NMAE_target",
        "log_NMAE_definition": "log_MAE divided by the training-set log-target standard deviation stored in the checkpoint scaler",
        "note": "The test set is not used for checkpoint selection.",
        "single_best": {},
        "ensembles": {},
    }
    all_metric_rows = []

    single_selected = {prop: [by_prop[prop][0]["run_name"]] for prop in PROPERTY_NAMES}
    for prop in PROPERTY_NAMES:
        selection_doc["single_best"][prop] = by_prop[prop][0]
    all_metric_rows.extend(_build_mode_metrics(root, single_selected, "single_best"))

    for k in top_ks:
        mode = f"ensemble_top{k}"
        selected = {}
        selection_doc["ensembles"][mode] = {}
        for prop in PROPERTY_NAMES:
            chosen = by_prop[prop][: min(k, len(by_prop[prop]))]
            selected[prop] = [item["run_name"] for item in chosen]
            selection_doc["ensembles"][mode][prop] = chosen
        all_metric_rows.extend(_build_mode_metrics(root, selected, mode))

    save_json(selection_doc, root / "selected_checkpoints.json")
    out_df = pd.DataFrame(all_metric_rows)
    out_df.to_csv(root / "ensemble_test_metrics.csv", index=False)
    print(f"saved: {results_path}")
    print(f"saved: {root / 'selected_checkpoints.json'}")
    print(f"saved: {root / 'ensemble_test_metrics.csv'}")


if __name__ == "__main__":
    main()
