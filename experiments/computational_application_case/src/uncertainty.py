"""Truthful uncertainty-mode selection without fabricated intervals."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


PROPERTY_NAMES = [
    "Density",
    "ElectricalConductivity",
    "HeatCapacity",
    "SurfaceTension",
    "ThermalConductivity",
    "Viscosity",
]


def estimate_uncertainty(
    predictions: pd.DataFrame,
    proxy_table: pd.DataFrame,
    checkpoint_paths: Sequence[str],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Use compatible ensemble rows when present, otherwise report unavailable."""

    minimum = int(config["ensemble_min_checkpoints"])
    checkpoint_count = len(list(checkpoint_paths))
    key_columns = ["candidate_id", "temperature_K", "pressure_kPa"]
    has_ensemble_rows = (
        checkpoint_count >= minimum
        and "checkpoint_name" in predictions
        and predictions.groupby(key_columns)["checkpoint_name"].nunique().min() >= minimum
    )
    if has_ensemble_rows:
        aggregated = predictions.groupby(key_columns, as_index=False)[PROPERTY_NAMES].agg(
            ["mean", "std"]
        )
        aggregated.columns = [
            "_".join(str(item) for item in column if item)
            if isinstance(column, tuple)
            else str(column)
            for column in aggregated.columns
        ]
        property_uncertainty = aggregated
        status = {
            "uncertainty_status": "checkpoint_ensemble",
            "checkpoint_count": checkpoint_count,
            "propagation_status": "requires_per-checkpoint_proxy_rows",
        }
    else:
        property_uncertainty = predictions[key_columns].drop_duplicates().copy()
        property_uncertainty["uncertainty_status"] = "not_available"
        property_uncertainty["reason"] = (
            "Fewer than three compatible checkpoint predictions and no held-out residual calibration were configured."
        )
        status = {
            "uncertainty_status": "not_available",
            "checkpoint_count": checkpoint_count,
            "reason": property_uncertainty["reason"].iloc[0],
        }
    proxy_uncertainty = proxy_table[key_columns].drop_duplicates().copy()
    proxy_uncertainty["uncertainty_status"] = status["uncertainty_status"]
    feasibility = predictions[["candidate_id"]].drop_duplicates().copy()
    feasibility["constraint_pass_probability"] = np.nan
    feasibility["pareto_rank_1_probability"] = np.nan
    feasibility["uncertainty_status"] = status["uncertainty_status"]
    return property_uncertainty, proxy_uncertainty, feasibility, status

