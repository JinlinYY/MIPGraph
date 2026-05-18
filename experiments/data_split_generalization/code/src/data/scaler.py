from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ConditionScaler:
    temp_mean: float
    temp_std: float
    pressure_median: float
    pressure_mean: float
    pressure_std: float

    @classmethod
    def fit(cls, temperature: np.ndarray, pressure: np.ndarray) -> "ConditionScaler":
        t = np.asarray(temperature, dtype=np.float64)
        p = np.asarray(pressure, dtype=np.float64)
        p_med = float(np.nanmedian(p)) if np.isfinite(p).any() else 101.325
        p_filled = np.where(np.isfinite(p), p, p_med)
        return cls(
            temp_mean=float(np.nanmean(t)),
            temp_std=float(np.nanstd(t) + 1e-8),
            pressure_median=p_med,
            pressure_mean=float(np.nanmean(p_filled)),
            pressure_std=float(np.nanstd(p_filled) + 1e-8),
        )

    def transform(self, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        t = np.asarray(temperature, dtype=np.float32)
        p = np.asarray(pressure, dtype=np.float32)
        p = np.where(np.isfinite(p), p, self.pressure_median)
        out = np.stack([(t - self.temp_mean) / self.temp_std, (p - self.pressure_mean) / self.pressure_std], axis=-1)
        return out.astype(np.float32)


@dataclass
class TargetScaler:
    means: np.ndarray
    stds: np.ndarray
    eps: float = 1e-8

    @classmethod
    def fit(cls, y: np.ndarray, mask: np.ndarray, eps: float = 1e-8) -> "TargetScaler":
        means = np.zeros(y.shape[1], dtype=np.float32)
        stds = np.ones(y.shape[1], dtype=np.float32)
        for j in range(y.shape[1]):
            valid = (mask[:, j] > 0) & np.isfinite(y[:, j]) & (y[:, j] > 0)
            if valid.any():
                vals = np.log(y[valid, j] + eps)
                means[j] = float(vals.mean())
                stds[j] = float(vals.std() + 1e-8)
        return cls(means=means, stds=stds, eps=eps)

    def transform(self, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = np.zeros_like(y, dtype=np.float32)
        for j in range(y.shape[1]):
            valid = (mask[:, j] > 0) & np.isfinite(y[:, j]) & (y[:, j] > 0)
            out[valid, j] = (np.log(y[valid, j] + self.eps) - self.means[j]) / self.stds[j]
        return out

    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        log_y = y_scaled * self.stds[None, :] + self.means[None, :]
        return np.exp(log_y) - self.eps

    def error_weights(
        self,
        y: np.ndarray,
        y_error: np.ndarray,
        mask: np.ndarray,
        error_mask: np.ndarray,
        clip_min: float = 0.1,
        clip_max: float = 10.0,
    ) -> np.ndarray:
        weights = np.ones_like(y, dtype=np.float32)
        for j in range(y.shape[1]):
            valid = (mask[:, j] > 0) & (error_mask[:, j] > 0) & np.isfinite(y[:, j]) & np.isfinite(y_error[:, j]) & (y[:, j] > 0)
            sigma_log = np.zeros(y.shape[0], dtype=np.float32)
            sigma_log[valid] = y_error[valid, j] / (y[valid, j] + self.eps)
            sigma_scaled = sigma_log / self.stds[j]
            w = 1.0 / (sigma_scaled**2 + self.eps)
            weights[valid, j] = np.clip(w[valid], clip_min, clip_max)
        return weights


def fit_scalers(arrays: dict[str, Any], train_indices: list[int], clip_min: float = 0.1, clip_max: float = 10.0):
    idx = np.array(train_indices, dtype=np.int64)
    condition_scaler = ConditionScaler.fit(arrays["temperature"][idx], arrays["pressure"][idx])
    target_scaler = TargetScaler.fit(arrays["y"][idx], arrays["mask"][idx])
    y_scaled = target_scaler.transform(arrays["y"], arrays["mask"])
    condition = condition_scaler.transform(arrays["temperature"], arrays["pressure"])
    weights = target_scaler.error_weights(arrays["y"], arrays["y_error"], arrays["mask"], arrays["error_mask"], clip_min, clip_max)
    return condition_scaler, target_scaler, y_scaled, condition, weights
