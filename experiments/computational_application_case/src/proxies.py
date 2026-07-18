"""Property-to-application proxy mapping and whole-window summaries."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from .units import (
    millilitres_to_cubic_metres,
    molar_heat_capacity_to_mass_specific,
    molar_mass_kg_per_mol,
    simplified_thermal_diffusion_timescale,
    thermal_diffusivity,
    volumetric_heat_capacity,
)


def log_iqr_standardize(
    value: float,
    reference_values: Iterable[float],
    epsilon: float,
) -> tuple[float, bool]:
    """Standardize log10(value) by the reference median and IQR.

    The second return value records whether the configured numerical floor was
    used for a zero or near-zero IQR.
    """

    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        return float("nan"), False
    reference = np.asarray(list(reference_values), dtype=float)
    reference = reference[np.isfinite(reference) & (reference > 0.0)]
    if reference.size == 0:
        return float("nan"), False
    log_reference = np.log10(reference)
    median = float(np.median(log_reference))
    q25, q75 = np.quantile(log_reference, [0.25, 0.75])
    raw_iqr = float(q75 - q25)
    protected = raw_iqr < float(epsilon)
    denominator = max(raw_iqr, float(epsilon))
    return (float(np.log10(scalar) - median) / denominator, protected)


def interfacial_window_deviation(
    surface_tension: float,
    lower_quantile: float,
    upper_quantile: float,
    reference_iqr: float,
    epsilon: float,
) -> float:
    """Return distance outside a reference surface-tension window in IQR units."""

    values = np.asarray(
        [surface_tension, lower_quantile, upper_quantile, reference_iqr], dtype=float
    )
    if not np.isfinite(values).all() or surface_tension <= 0.0:
        return float("nan")
    denominator = max(float(reference_iqr), float(epsilon))
    if surface_tension < lower_quantile:
        return float(lower_quantile - surface_tension) / denominator
    if surface_tension > upper_quantile:
        return float(surface_tension - upper_quantile) / denominator
    return 0.0


def transport_favorability(z_conductivity: float, z_viscosity: float) -> float:
    """Return the transport-favorability proxy z_sigma - z_eta."""

    if not np.isfinite(z_conductivity) or not np.isfinite(z_viscosity):
        return float("nan")
    return float(z_conductivity - z_viscosity)


def _safe_row_calculation(function, *values: float) -> float:
    try:
        return float(function(*values))
    except (TypeError, ValueError, FloatingPointError):
        return float("nan")


def compute_application_proxies(
    predictions: pd.DataFrame,
    proxy_config: Mapping[str, Any],
) -> pd.DataFrame:
    """Map six-property predictions to transport, interface, and thermal proxies."""

    required = {
        "candidate_id",
        "candidate_type",
        "il_smiles",
        "temperature_K",
        "Density",
        "Viscosity",
        "ElectricalConductivity",
        "HeatCapacity",
        "SurfaceTension",
        "ThermalConductivity",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction table lacks required columns: {missing}")
    epsilon = float(proxy_config["numerical_epsilon"])
    length = float(proxy_config["thermal_length_m"])
    volume_m3 = float(millilitres_to_cubic_metres(proxy_config["electrolyte_volume_mL"]))
    low_quantile = float(proxy_config["reference_gamma_low_quantile"])
    high_quantile = float(proxy_config["reference_gamma_high_quantile"])
    output = predictions.copy()
    mass_by_smiles: dict[str, float] = {}
    for smiles in output["il_smiles"].astype(str).unique():
        mass_by_smiles[smiles] = molar_mass_kg_per_mol(smiles)
    output["molar_mass_kg_per_mol"] = output["il_smiles"].map(mass_by_smiles)
    output["cp_mass_J_kg-1_K-1"] = [
        _safe_row_calculation(
            molar_heat_capacity_to_mass_specific,
            row.HeatCapacity,
            row.molar_mass_kg_per_mol,
        )
        for row in output.itertuples(index=False)
    ]
    output["volumetric_heat_capacity"] = [
        _safe_row_calculation(volumetric_heat_capacity, density, heat_capacity)
        for density, heat_capacity in zip(
            output["Density"], output["cp_mass_J_kg-1_K-1"]
        )
    ]
    output["thermal_diffusivity"] = [
        _safe_row_calculation(
            thermal_diffusivity,
            row.ThermalConductivity,
            row.volumetric_heat_capacity,
        )
        for row in output.itertuples(index=False)
    ]
    output["simplified_thermal_diffusion_timescale"] = [
        _safe_row_calculation(
            simplified_thermal_diffusion_timescale, length, row.thermal_diffusivity
        )
        for row in output.itertuples(index=False)
    ]
    output["electrolyte_mass_kg"] = pd.to_numeric(
        output["Density"], errors="coerce"
    ) * volume_m3
    output["thermal_effusivity"] = np.sqrt(
        pd.to_numeric(output["ThermalConductivity"], errors="coerce")
        * output["volumetric_heat_capacity"]
    )
    output["z_conductivity"] = np.nan
    output["z_viscosity"] = np.nan
    output["transport_favorability"] = np.nan
    output["interfacial_window_deviation"] = np.nan
    output["proxy_warnings"] = ""
    references = output[output["candidate_type"].eq("observed_reference")]
    for temperature, indices in output.groupby("temperature_K").groups.items():
        reference = references[np.isclose(references["temperature_K"], temperature)]
        conductivity_reference = reference["ElectricalConductivity"].to_numpy(dtype=float)
        viscosity_reference = reference["Viscosity"].to_numpy(dtype=float)
        gamma_reference = reference["SurfaceTension"].to_numpy(dtype=float)
        valid_gamma = gamma_reference[np.isfinite(gamma_reference) & (gamma_reference > 0.0)]
        if valid_gamma.size:
            gamma_low, gamma_high = np.quantile(
                valid_gamma, [low_quantile, high_quantile]
            )
            gamma_q25, gamma_q75 = np.quantile(valid_gamma, [0.25, 0.75])
            gamma_iqr = float(gamma_q75 - gamma_q25)
        else:
            gamma_low = gamma_high = gamma_iqr = float("nan")
        for index in indices:
            z_sigma, sigma_protected = log_iqr_standardize(
                float(output.at[index, "ElectricalConductivity"]),
                conductivity_reference,
                epsilon,
            )
            z_eta, eta_protected = log_iqr_standardize(
                float(output.at[index, "Viscosity"]), viscosity_reference, epsilon
            )
            output.at[index, "z_conductivity"] = z_sigma
            output.at[index, "z_viscosity"] = z_eta
            output.at[index, "transport_favorability"] = transport_favorability(
                z_sigma, z_eta
            )
            output.at[index, "interfacial_window_deviation"] = (
                interfacial_window_deviation(
                    float(output.at[index, "SurfaceTension"]),
                    float(gamma_low),
                    float(gamma_high),
                    gamma_iqr,
                    epsilon,
                )
            )
            warnings = []
            if len(reference) < 3:
                warnings.append("fewer_than_three_temperature_matched_references")
            if sigma_protected:
                warnings.append("conductivity_iqr_floor_used")
            if eta_protected:
                warnings.append("viscosity_iqr_floor_used")
            if np.isfinite(gamma_iqr) and gamma_iqr < epsilon:
                warnings.append("surface_tension_iqr_floor_used")
            output.at[index, "proxy_warnings"] = ";".join(warnings)
    return output


def _complete_extreme(
    group: pd.DataFrame,
    column: str,
    mode: str,
) -> tuple[float, float]:
    values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
    temperatures = pd.to_numeric(group["temperature_K"], errors="coerce").to_numpy(
        dtype=float
    )
    if values.size == 0 or not np.isfinite(values).all():
        return float("nan"), float("nan")
    index = int(np.argmin(values) if mode == "min" else np.argmax(values))
    return float(values[index]), float(temperatures[index])


def _temperature_diagnostics(group: pd.DataFrame, column: str) -> dict[str, float]:
    values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
    temperature = pd.to_numeric(group["temperature_K"], errors="coerce").to_numpy(
        dtype=float
    )
    valid = np.isfinite(values) & np.isfinite(temperature)
    if not valid.any():
        return {
            f"{column}_mean": float("nan"),
            f"{column}_slope": float("nan"),
            f"{column}_relative_change": float("nan"),
            f"{column}_coefficient_of_variation": float("nan"),
        }
    selected = values[valid]
    selected_temperature = temperature[valid]
    mean = float(np.mean(selected))
    slope = (
        float(np.polyfit(selected_temperature, selected, 1)[0])
        if selected.size >= 2 and np.ptp(selected_temperature) > 0.0
        else 0.0
    )
    relative_change = (
        float((selected[-1] - selected[0]) / abs(selected[0]))
        if selected[0] != 0.0
        else float("nan")
    )
    coefficient = float(np.std(selected) / abs(mean)) if mean != 0.0 else float("nan")
    return {
        f"{column}_mean": mean,
        f"{column}_slope": slope,
        f"{column}_relative_change": relative_change,
        f"{column}_coefficient_of_variation": coefficient,
    }


def summarize_whole_temperature_window(proxy_table: pd.DataFrame) -> pd.DataFrame:
    """Aggregate complete temperature curves using specified worst directions."""

    required = {
        "candidate_id",
        "temperature_K",
        "ElectricalConductivity",
        "Viscosity",
        "transport_favorability",
        "volumetric_heat_capacity",
        "thermal_diffusivity",
        "simplified_thermal_diffusion_timescale",
        "interfacial_window_deviation",
        "Density",
    }
    missing = sorted(required - set(proxy_table.columns))
    if missing:
        raise ValueError(f"Proxy table lacks required columns: {missing}")
    rows: list[dict[str, Any]] = []
    expected_temperature_count = int(proxy_table["temperature_K"].nunique())
    for candidate_id, group in proxy_table.groupby("candidate_id", sort=True):
        group = group.sort_values("temperature_K").reset_index(drop=True)
        first = group.iloc[0]
        row: dict[str, Any] = {"candidate_id": candidate_id}
        row["temperature_point_count"] = int(group["temperature_K"].nunique())
        row["expected_temperature_point_count"] = expected_temperature_count
        row["temperature_grid_complete"] = bool(
            row["temperature_point_count"] == expected_temperature_count
        )
        for metadata in [
            "candidate_type",
            "cation_smiles",
            "anion_smiles",
            "il_smiles",
            "canonical_il_key",
            "cation_support_count",
            "anion_support_count",
        ]:
            if metadata in group.columns:
                row[metadata] = first[metadata]
        definitions = {
            "conductivity_worst": ("ElectricalConductivity", "min"),
            "viscosity_worst": ("Viscosity", "max"),
            "transport_favorability_worst": ("transport_favorability", "min"),
            "volumetric_heat_capacity_worst": ("volumetric_heat_capacity", "min"),
            "thermal_diffusivity_worst": ("thermal_diffusivity", "min"),
            "thermal_timescale_worst": (
                "simplified_thermal_diffusion_timescale",
                "max",
            ),
            "interfacial_deviation_worst": (
                "interfacial_window_deviation",
                "max",
            ),
        }
        for output_name, (column, mode) in definitions.items():
            value, temperature = _complete_extreme(group, column, mode)
            row[output_name] = value
            row[f"{output_name}_temperature_K"] = temperature
        density = pd.to_numeric(group["Density"], errors="coerce").to_numpy(dtype=float)
        row["density_range"] = (
            float(np.max(density) - np.min(density))
            if density.size and np.isfinite(density).all()
            else float("nan")
        )
        for column in [
            "ElectricalConductivity",
            "Viscosity",
            "transport_favorability",
            "volumetric_heat_capacity",
            "thermal_diffusivity",
            "simplified_thermal_diffusion_timescale",
            "interfacial_window_deviation",
            "Density",
        ]:
            row.update(_temperature_diagnostics(group, column))
        row["curve_warning_count"] = int(
            pd.to_numeric(group.get("curve_warning_count", 0), errors="coerce")
            .fillna(0)
            .sum()
            if "curve_warning_count" in group
            else 0
        )
        row["severe_curve_failure_count"] = int(
            pd.to_numeric(group.get("severe_curve_failure_count", 0), errors="coerce")
            .fillna(0)
            .sum()
            if "severe_curve_failure_count" in group
            else 0
        )
        rows.append(row)
    return pd.DataFrame(rows)
