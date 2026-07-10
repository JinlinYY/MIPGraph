from __future__ import annotations

import math

import numpy as np
import pandas as pd


TASKS = {
    "electrolyte_low_viscosity_high_conductivity": {
        "objectives": {
            "Viscosity_pred": "min",
            "ElectricalConductivity_pred": "max",
            "density_penalty": "min",
            "uncertainty_penalty": "min",
        },
        "density_range": (700.0, 1800.0),
    },
    "heat_transfer_fluid": {
        "objectives": {
            "ThermalConductivity_pred": "max",
            "HeatCapacity_pred": "max",
            "Viscosity_pred": "min",
            "uncertainty_penalty": "min",
        },
        "max_viscosity": 100.0,
    },
    "separation_solvent": {
        "objectives": {
            "surface_tension_deviation": "min",
            "viscosity_deviation": "min",
            "density_penalty": "min",
            "uncertainty_penalty": "min",
        },
        "surface_tension_target": 35.0,
        "viscosity_range": (1.0, 100.0),
        "density_range": (700.0, 1800.0),
    },
}


def get_task(name: str) -> dict:
    if name not in TASKS:
        raise ValueError(f"Unknown design task {name!r}. Choose from {sorted(TASKS)}")
    return TASKS[name]


def _value(row: pd.Series, name: str, default: float = 0.0) -> float:
    value = row.get(name, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _range_penalty(value: float, low: float, high: float) -> float:
    if value < low:
        return (low - value) / max(abs(low), 1.0)
    if value > high:
        return (value - high) / max(abs(high), 1.0)
    return 0.0


def _uncertainty_penalty(row: pd.Series) -> float:
    values = []
    for key, value in row.items():
        if key.endswith("_pred_std_log"):
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                pass
    return float(np.nanmean(values)) if values else 0.0


def apply_task_scoring(df: pd.DataFrame, task_name: str) -> pd.DataFrame:
    task = get_task(task_name)
    rows = []
    for _, row in df.iterrows():
        item = row.to_dict()
        uncertainty = _uncertainty_penalty(row)
        density = _value(row, "Density_pred", 1000.0)
        viscosity = max(_value(row, "Viscosity_pred", 1.0), 0.0)
        conductivity = max(_value(row, "ElectricalConductivity_pred", 0.0), 0.0)
        heat_capacity = max(_value(row, "HeatCapacity_pred", 0.0), 0.0)
        thermal = max(_value(row, "ThermalConductivity_pred", 0.0), 0.0)
        surface = max(_value(row, "SurfaceTension_pred", 0.0), 0.0)
        item["uncertainty_penalty"] = uncertainty
        item["density_penalty"] = _range_penalty(density, *task.get("density_range", (700.0, 1800.0)))

        if task_name == "electrolyte_low_viscosity_high_conductivity":
            item["passes_constraints"] = item["density_penalty"] <= 0.25
            item["objective_score"] = math.log1p(conductivity) - math.log1p(viscosity) - 2.0 * item["density_penalty"] - uncertainty
            item["selection_reason"] = "low predicted viscosity, high predicted conductivity, acceptable density, and low uncertainty penalty"
        elif task_name == "heat_transfer_fluid":
            viscosity_penalty = max(0.0, viscosity - float(task["max_viscosity"])) / float(task["max_viscosity"])
            item["viscosity_penalty"] = viscosity_penalty
            item["passes_constraints"] = viscosity_penalty <= 1.0
            item["objective_score"] = math.log1p(thermal) + math.log1p(heat_capacity) - viscosity_penalty - uncertainty
            item["selection_reason"] = "high predicted thermal transport and heat capacity with controlled viscosity and uncertainty"
        else:
            target = float(task["surface_tension_target"])
            low, high = task["viscosity_range"]
            item["surface_tension_deviation"] = abs(surface - target) / max(target, 1.0)
            item["viscosity_deviation"] = _range_penalty(viscosity, low, high)
            item["passes_constraints"] = item["density_penalty"] <= 0.25 and item["viscosity_deviation"] <= 1.0
            item["objective_score"] = -item["surface_tension_deviation"] - item["viscosity_deviation"] - item["density_penalty"] - uncertainty
            item["selection_reason"] = "surface tension near target with moderate viscosity, acceptable density, and low uncertainty"
        rows.append(item)
    return pd.DataFrame(rows)
