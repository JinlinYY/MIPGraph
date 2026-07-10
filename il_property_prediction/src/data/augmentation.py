from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from .dataset import PROPERTY_NAMES


def build_temperature_interpolation_samples(
    clean_df: pd.DataFrame,
    arrays: dict[str, Any],
    train_indices: Sequence[int],
    condition_scaler,
    target_scaler,
    properties: Sequence[str],
    points_per_interval: int = 1,
    max_temperature_gap: float = 40.0,
    pressure_round_decimals: int = 1,
    sample_weight: float = 0.5,
    max_samples_per_property: int = 0,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Interpolate log-targets only between train observations of the same IL and pressure bin."""
    bad = [name for name in properties if name not in PROPERTY_NAMES]
    if bad:
        raise ValueError(f"Unknown augmentation properties: {bad}")
    if points_per_interval < 1:
        raise ValueError("points_per_interval must be at least 1")
    if max_temperature_gap <= 0:
        raise ValueError("max_temperature_gap must be positive")

    rng = np.random.default_rng(seed)
    train_set = set(int(i) for i in train_indices)
    if not train_set:
        raise ValueError("Cannot augment an empty training split")
    if min(train_set) < 0 or max(train_set) >= len(clean_df):
        raise IndexError("Training indices are outside the cleaned dataframe")
    samples: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "method": "same_il_pressure_temperature_log_interpolation",
        "train_row_count": len(train_set),
        "properties": {},
    }

    for prop in properties:
        prop_idx = PROPERTY_NAMES.index(prop)
        frame = clean_df.loc[sorted(train_set), ["IL_SMILES", "Temperature_K", "Pressure_kPa"]].copy()
        frame["source_idx"] = frame.index.astype(int)
        frame["target"] = np.asarray(arrays["y"])[frame.index.to_numpy(), prop_idx]
        frame = frame[
            np.isfinite(frame["Temperature_K"])
            & np.isfinite(frame["target"])
            & (frame["target"] > 0)
        ].copy()
        frame["Pressure_kPa"] = pd.to_numeric(frame["Pressure_kPa"], errors="coerce").fillna(
            float(condition_scaler.pressure_median)
        )
        frame["pressure_bin"] = frame["Pressure_kPa"].round(pressure_round_decimals)

        prop_samples: list[dict[str, Any]] = []
        candidate_intervals = 0
        skipped_large_gap = 0
        for (_, _), group in frame.groupby(["IL_SMILES", "pressure_bin"], sort=False):
            by_temperature = (
                group.assign(log_target=np.log(group["target"].to_numpy(dtype=float) + target_scaler.eps))
                .groupby("Temperature_K", as_index=False)
                .agg(
                    source_idx=("source_idx", "first"),
                    pressure=("Pressure_kPa", "mean"),
                    log_target=("log_target", "median"),
                )
                .sort_values("Temperature_K")
            )
            if len(by_temperature) < 2:
                continue
            existing_temperatures = by_temperature["Temperature_K"].to_numpy(dtype=float)
            records = by_temperature.to_dict("records")
            for left, right in zip(records[:-1], records[1:]):
                delta_t = float(right["Temperature_K"] - left["Temperature_K"])
                if delta_t <= 1e-8:
                    continue
                candidate_intervals += 1
                if delta_t > max_temperature_gap:
                    skipped_large_gap += 1
                    continue
                for point_idx in range(1, points_per_interval + 1):
                    alpha = point_idx / (points_per_interval + 1.0)
                    temperature = (1.0 - alpha) * float(left["Temperature_K"]) + alpha * float(right["Temperature_K"])
                    if np.min(np.abs(existing_temperatures - temperature)) < 1e-6:
                        continue
                    pressure = (1.0 - alpha) * float(left["pressure"]) + alpha * float(right["pressure"])
                    log_target = (1.0 - alpha) * float(left["log_target"]) + alpha * float(right["log_target"])
                    y_scaled = np.zeros(len(PROPERTY_NAMES), dtype=np.float32)
                    y_raw = np.zeros(len(PROPERTY_NAMES), dtype=np.float32)
                    mask = np.zeros(len(PROPERTY_NAMES), dtype=np.float32)
                    error_weight = np.ones(len(PROPERTY_NAMES), dtype=np.float32)
                    y_scaled[prop_idx] = (log_target - float(target_scaler.means[prop_idx])) / float(target_scaler.stds[prop_idx])
                    y_raw[prop_idx] = float(np.exp(log_target) - target_scaler.eps)
                    mask[prop_idx] = 1.0
                    error_weight[prop_idx] = float(sample_weight)
                    condition = condition_scaler.transform(
                        np.asarray([temperature], dtype=np.float32),
                        np.asarray([pressure], dtype=np.float32),
                    )[0]
                    prop_samples.append(
                        {
                            "source_idx": int(left["source_idx"]),
                            "property": prop,
                            "temperature": temperature,
                            "pressure": pressure,
                            "condition": condition,
                            "y_scaled": y_scaled,
                            "y_raw": y_raw,
                            "mask": mask,
                            "error_weight": error_weight,
                        }
                    )

        if max_samples_per_property > 0 and len(prop_samples) > max_samples_per_property:
            keep = np.sort(rng.choice(len(prop_samples), size=max_samples_per_property, replace=False))
            prop_samples = [prop_samples[int(i)] for i in keep]
        samples.extend(prop_samples)
        report["properties"][prop] = {
            "observed_train_labels": int(len(frame)),
            "candidate_intervals": int(candidate_intervals),
            "skipped_large_temperature_gap": int(skipped_large_gap),
            "generated_samples": int(len(prop_samples)),
        }

    report["generated_total"] = len(samples)
    return samples, report
