from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


METRIC_COLUMNS = ["MAE", "RMSE", "R2", "MAPE", "N"]


@dataclass(frozen=True)
class MetricResult:
    mae: float
    rmse: float
    r2: float
    mape: float | float("nan")
    n: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "MAE": self.mae,
            "RMSE": self.rmse,
            "R2": self.r2,
            "MAPE": self.mape,
            "N": self.n,
        }


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    nonzero_mask = np.abs(y_true) > 1e-12
    if not np.any(nonzero_mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100.0)


def compute_basic_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> MetricResult:
    y_true_arr = np.asarray(list(y_true), dtype=float)
    y_pred_arr = np.asarray(list(y_pred), dtype=float)
    n_samples = int(y_true_arr.shape[0])
    if n_samples == 0:
        return MetricResult(np.nan, np.nan, np.nan, np.nan, 0)
    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    r2 = float(r2_score(y_true_arr, y_pred_arr)) if n_samples > 1 else float("nan")
    mape = _safe_mape(y_true_arr, y_pred_arr)
    return MetricResult(mae, rmse, r2, mape, n_samples)


def compute_metrics_by_groups(
    df: pd.DataFrame,
    group_cols: list[str],
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = df.groupby(group_cols, dropna=False)
    for keys, frame in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        result = compute_basic_metrics(frame[y_true_col].values, frame[y_pred_col].values)
        row = {col: value for col, value in zip(group_cols, keys)}
        row.update(result.to_dict())
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(group_cols).reset_index(drop=True)
    return out

