from __future__ import annotations

import numpy as np
from sklearn.metrics import r2_score


PROPERTY_NAMES = ["Density", "ElectricalConductivity", "HeatCapacity", "SurfaceTension", "ThermalConductivity", "Viscosity"]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray, property_names: list[str] | None = None) -> dict:
    property_names = property_names or PROPERTY_NAMES
    metrics: dict = {}
    maes, rmses, r2s, r2_counts, counts = [], [], [], [], []
    for j, name in enumerate(property_names):
        valid = (mask[:, j] > 0) & np.isfinite(y_true[:, j]) & np.isfinite(y_pred[:, j])
        count = int(valid.sum())
        if count == 0:
            metrics[name] = {"MAE": None, "RMSE": None, "R2": None, "MAPE": None, "label_count": 0}
            continue
        err = y_pred[valid, j] - y_true[valid, j]
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        denom = np.maximum(np.abs(y_true[valid, j]), 1e-8)
        y_range = float(np.max(y_true[valid, j]) - np.min(y_true[valid, j]))
        y_mean_abs = float(np.mean(np.abs(y_true[valid, j])))
        y_std = float(np.std(y_true[valid, j]))
        range_denom = max(y_range, 1e-8)
        mean_denom = max(y_mean_abs, 1e-8)
        std_denom = max(y_std, 1e-8)
        mape = float(np.mean(np.abs(err) / denom) * 100.0)
        r2 = float(r2_score(y_true[valid, j], y_pred[valid, j])) if count > 1 else None
        metrics[name] = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "MAPE": mape,
            "NMAE_range": float(mae / range_denom),
            "NRMSE_range": float(rmse / range_denom),
            "NMAE_mean": float(mae / mean_denom),
            "NRMSE_mean": float(rmse / mean_denom),
            "NMAE_std": float(mae / std_denom),
            "NRMSE_std": float(rmse / std_denom),
            "label_count": count,
        }
        maes.append(mae)
        rmses.append(rmse)
        counts.append(count)
        if r2 is not None:
            r2s.append(r2)
            r2_counts.append(count)
    norm_keys = ["NMAE_range", "NRMSE_range", "NMAE_mean", "NRMSE_mean", "NMAE_std", "NRMSE_std"]
    if maes:
        counts_arr = np.array(counts, dtype=np.float64)
        metrics["macro_MAE"] = float(np.mean(maes))
        metrics["macro_RMSE"] = float(np.mean(rmses))
        metrics["weighted_MAE"] = float(np.average(maes, weights=counts_arr))
        metrics["weighted_RMSE"] = float(np.average(rmses, weights=counts_arr))
        metrics["macro_R2"] = float(np.mean(r2s)) if r2s else None
        metrics["weighted_R2"] = float(np.average(r2s, weights=np.asarray(r2_counts, dtype=np.float64))) if r2s else None
        for key in norm_keys:
            vals = [metrics[name][key] for name in property_names if metrics[name]["label_count"] > 0]
            weights = [metrics[name]["label_count"] for name in property_names if metrics[name]["label_count"] > 0]
            metrics[f"macro_{key}"] = float(np.mean(vals))
            metrics[f"weighted_{key}"] = float(np.average(vals, weights=np.asarray(weights, dtype=np.float64)))
        metrics["normalized_summary"] = {
            "macro_NMAE_mean": metrics["macro_NMAE_mean"],
            "weighted_NMAE_mean": metrics["weighted_NMAE_mean"],
            "macro_NRMSE_mean": metrics["macro_NRMSE_mean"],
            "weighted_NRMSE_mean": metrics["weighted_NRMSE_mean"],
            "macro_NMAE_range": metrics["macro_NMAE_range"],
            "weighted_NMAE_range": metrics["weighted_NMAE_range"],
            "macro_NRMSE_range": metrics["macro_NRMSE_range"],
            "weighted_NRMSE_range": metrics["weighted_NRMSE_range"],
            "macro_R2": metrics["macro_R2"],
            "weighted_R2": metrics["weighted_R2"],
        }
    else:
        metrics["macro_MAE"] = metrics["macro_RMSE"] = metrics["weighted_MAE"] = metrics["weighted_RMSE"] = None
        metrics["macro_R2"] = metrics["weighted_R2"] = None
        for key in norm_keys:
            metrics[f"macro_{key}"] = None
            metrics[f"weighted_{key}"] = None
        metrics["normalized_summary"] = {}
    return metrics


def compute_log_space_metrics(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    mask: np.ndarray,
    property_names: list[str] | None = None,
) -> dict:
    property_names = property_names or PROPERTY_NAMES
    out: dict = {}
    maes, rmses, r2s, r2_counts, counts = [], [], [], [], []
    for j, name in enumerate(property_names):
        valid = (mask[:, j] > 0) & np.isfinite(y_true_log[:, j]) & np.isfinite(y_pred_log[:, j])
        count = int(valid.sum())
        if count == 0:
            out[name] = {"log_MAE": None, "log_RMSE": None, "log_R2": None, "label_count": 0}
            continue
        err = y_pred_log[valid, j] - y_true_log[valid, j]
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        r2 = float(r2_score(y_true_log[valid, j], y_pred_log[valid, j])) if count > 1 else None
        out[name] = {"log_MAE": mae, "log_RMSE": rmse, "log_R2": r2, "label_count": count}
        maes.append(mae)
        rmses.append(rmse)
        counts.append(count)
        if r2 is not None:
            r2s.append(r2)
            r2_counts.append(count)
    if maes:
        counts_arr = np.asarray(counts, dtype=np.float64)
        out["macro_log_MAE"] = float(np.mean(maes))
        out["macro_log_RMSE"] = float(np.mean(rmses))
        out["weighted_log_MAE"] = float(np.average(maes, weights=counts_arr))
        out["weighted_log_RMSE"] = float(np.average(rmses, weights=counts_arr))
        out["macro_log_R2"] = float(np.mean(r2s)) if r2s else None
        out["weighted_log_R2"] = float(np.average(r2s, weights=np.asarray(r2_counts, dtype=np.float64))) if r2s else None
    else:
        out["macro_log_MAE"] = out["macro_log_RMSE"] = None
        out["weighted_log_MAE"] = out["weighted_log_RMSE"] = None
        out["macro_log_R2"] = out["weighted_log_R2"] = None
    return out
