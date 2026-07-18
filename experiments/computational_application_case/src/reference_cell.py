"""Transparent conditional reference-cell calculations for electrolyte candidates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_SCENARIO_FIELDS = (
    "electrode_area_cm2",
    "separator_thickness_um",
    "electrolyte_volume_mL",
    "nominal_capacitance_F",
    "charge_discharge_current_A",
    "convective_heat_transfer_coefficient_W_m2_K",
    "exposed_face_count",
    "transient_duration_s",
    "reference_temperature_K",
    "risk_reference_quantiles",
)

REQUIRED_PROXY_COLUMNS = (
    "candidate_id",
    "candidate_type",
    "temperature_K",
    "ElectricalConductivity",
    "ThermalConductivity",
    "volumetric_heat_capacity",
)

RISK_BAND_ORDER = {
    "within_reference_envelope": 0,
    "elevated_reference_tail": 1,
    "beyond_reference_tail": 2,
}


def _validated_scenario(
    config: Mapping[str, Any],
) -> dict[str, float | int | str | list[float]]:
    missing = [field for field in REQUIRED_SCENARIO_FIELDS if field not in config]
    if missing:
        raise ValueError(f"Reference-cell scenario lacks fields: {missing}")
    positive_fields = REQUIRED_SCENARIO_FIELDS[:-1]
    scenario: dict[str, float | int | str | list[float]] = {}
    for field in positive_fields:
        value = float(config[field])
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"reference_cell.{field} must be finite and positive")
        scenario[field] = int(value) if field == "exposed_face_count" else value
    quantiles = [float(value) for value in config["risk_reference_quantiles"]]
    if len(quantiles) != 2 or not 0.0 < quantiles[0] < quantiles[1] < 1.0:
        raise ValueError(
            "reference_cell.risk_reference_quantiles must contain two ordered values in (0, 1)"
        )
    scenario["risk_reference_quantiles"] = quantiles
    scenario["scenario_name"] = str(
        config.get("scenario_name", "conditional_reference_cell")
    )
    return scenario


def _add_relative_resistance(
    metrics: pd.DataFrame,
    reference_temperature_K: float,
) -> pd.DataFrame:
    reference_rows = metrics[
        np.isclose(metrics["temperature_K"], reference_temperature_K, atol=1.0e-8)
    ][["candidate_id", "electrolyte_resistance_ohm"]].rename(
        columns={"electrolyte_resistance_ohm": "reference_temperature_resistance_ohm"}
    )
    if reference_rows["candidate_id"].duplicated().any():
        raise ValueError("Reference temperature occurs more than once for a candidate")
    missing = sorted(set(metrics["candidate_id"]) - set(reference_rows["candidate_id"]))
    if missing:
        raise ValueError(
            f"Configured reference_temperature_K is absent for candidates: {missing[:5]}"
        )
    output = metrics.merge(reference_rows, on="candidate_id", how="left", validate="many_to_one")
    output["relative_electrolyte_resistance"] = (
        output["electrolyte_resistance_ohm"]
        / output["reference_temperature_resistance_ohm"]
    )
    return output


def _annotate_comparative_risk(
    metrics: pd.DataFrame,
    quantiles: list[float],
) -> pd.DataFrame:
    reference = metrics[metrics["candidate_type"].eq("observed_reference")]
    reference_counts = reference.groupby("temperature_K")["candidate_id"].nunique()
    all_temperatures = set(metrics["temperature_K"].astype(float))
    missing = sorted(all_temperatures - set(reference_counts.index.astype(float)))
    if missing or reference_counts.min() < 2:
        raise ValueError(
            "At least two observed-reference candidates are required at every scenario temperature"
        )
    value_columns = [
        "electrolyte_resistance_ohm",
        "transient_temperature_rise_K",
    ]
    thresholds = reference.groupby("temperature_K")[value_columns].quantile(
        quantiles
    )
    thresholds = thresholds.unstack(level=-1).reset_index()
    threshold_names = ["temperature_K"]
    for metric, quantile in thresholds.columns.tolist()[1:]:
        label = "q75" if np.isclose(quantile, quantiles[0]) else "q95"
        threshold_names.append(f"reference_{metric}_{label}")
    thresholds.columns = threshold_names
    output = metrics.merge(thresholds, on="temperature_K", how="left", validate="many_to_one")
    electrical_q75 = output["reference_electrolyte_resistance_ohm_q75"]
    thermal_q75 = output["reference_transient_temperature_rise_K_q75"]
    output["electrical_risk_ratio_to_reference_q75"] = (
        output["electrolyte_resistance_ohm"] / electrical_q75
    )
    output["thermal_risk_ratio_to_reference_q75"] = (
        output["transient_temperature_rise_K"] / thermal_q75
    )
    output["reference_cell_risk_index"] = output[
        [
            "electrical_risk_ratio_to_reference_q75",
            "thermal_risk_ratio_to_reference_q75",
        ]
    ].max(axis=1)
    electrical_q95 = output["reference_electrolyte_resistance_ohm_q95"]
    thermal_q95 = output["reference_transient_temperature_rise_K_q95"]
    beyond_electrical = output["electrolyte_resistance_ohm"] > electrical_q95
    beyond_thermal = output["transient_temperature_rise_K"] > thermal_q95
    elevated_electrical = output["electrolyte_resistance_ohm"] > electrical_q75
    elevated_thermal = output["transient_temperature_rise_K"] > thermal_q75
    output["reference_cell_risk_band"] = np.select(
        [beyond_electrical | beyond_thermal, elevated_electrical | elevated_thermal],
        ["beyond_reference_tail", "elevated_reference_tail"],
        default="within_reference_envelope",
    )
    reasons: list[str] = []
    for electrical, thermal in zip(elevated_electrical, elevated_thermal):
        reason = []
        if electrical:
            reason.append("electrical")
        if thermal:
            reason.append("thermal")
        reasons.append(";".join(reason) if reason else "none")
    output["reference_cell_risk_reason"] = reasons
    return output


def _summarize_reference_cell_metrics(
    metrics: pd.DataFrame,
    reference_temperature_K: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate_id, group in metrics.groupby("candidate_id", sort=True):
        group = group.sort_values("temperature_K").reset_index(drop=True)
        low = group.iloc[0]
        high = group.iloc[-1]
        reference = group[
            np.isclose(group["temperature_K"], reference_temperature_K, atol=1.0e-8)
        ].iloc[0]
        risk_order = group["reference_cell_risk_band"].map(RISK_BAND_ORDER)
        worst_band_order = int(risk_order.max())
        worst_candidates = group[risk_order.eq(worst_band_order)].sort_values(
            ["reference_cell_risk_index", "temperature_K"],
            ascending=[False, True],
        )
        worst = worst_candidates.iloc[0]

        def maximum(metric: str) -> tuple[float, float]:
            index = int(group[metric].astype(float).idxmax())
            return float(group.loc[index, metric]), float(group.loc[index, "temperature_K"])

        maximum_resistance, maximum_resistance_temperature = maximum(
            "electrolyte_resistance_ohm"
        )
        maximum_rc, maximum_rc_temperature = maximum("electrolyte_RC_time_constant_s")
        maximum_power, maximum_power_temperature = maximum("joule_heating_power_W")
        maximum_steady_rise, maximum_steady_temperature = maximum(
            "steady_state_temperature_rise_K"
        )
        maximum_transient_rise, maximum_transient_temperature = maximum(
            "transient_temperature_rise_K"
        )
        reference_resistance = float(reference["electrolyte_resistance_ohm"])
        low_resistance = float(low["electrolyte_resistance_ohm"])
        high_resistance = float(high["electrolyte_resistance_ohm"])
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_type": str(group["candidate_type"].iloc[0]),
                "temperature_point_count": int(len(group)),
                "low_temperature_K": float(low["temperature_K"]),
                "reference_temperature_K": float(reference_temperature_K),
                "high_temperature_K": float(high["temperature_K"]),
                "electrolyte_resistance_ohm_at_low_temperature": low_resistance,
                "electrolyte_resistance_ohm_at_reference_temperature": reference_resistance,
                "electrolyte_resistance_ohm_at_high_temperature": high_resistance,
                "low_temperature_resistance_ratio_to_reference": low_resistance
                / reference_resistance,
                "high_temperature_resistance_ratio_to_reference": high_resistance
                / reference_resistance,
                "high_to_low_temperature_resistance_ratio": high_resistance / low_resistance,
                "low_temperature_conductivity_retention_pct": 100.0
                * reference_resistance
                / low_resistance,
                "high_temperature_conductivity_retention_pct": 100.0
                * reference_resistance
                / high_resistance,
                "electrolyte_resistance_ohm_worst": maximum_resistance,
                "electrolyte_resistance_worst_temperature_K": maximum_resistance_temperature,
                "electrolyte_RC_time_constant_s_worst": maximum_rc,
                "electrolyte_RC_worst_temperature_K": maximum_rc_temperature,
                "joule_heating_power_W_worst": maximum_power,
                "joule_heating_worst_temperature_K": maximum_power_temperature,
                "steady_state_temperature_rise_K_worst": maximum_steady_rise,
                "steady_state_temperature_rise_worst_temperature_K": maximum_steady_temperature,
                "transient_temperature_rise_K_worst": maximum_transient_rise,
                "transient_temperature_rise_worst_temperature_K": maximum_transient_temperature,
                "reference_cell_risk_index_worst": float(
                    worst["reference_cell_risk_index"]
                ),
                "reference_cell_risk_band_worst": str(
                    worst["reference_cell_risk_band"]
                ),
                "reference_cell_worst_temperature_K": float(worst["temperature_K"]),
                "reference_cell_risk_reason_worst": str(
                    worst["reference_cell_risk_reason"]
                ),
                "scenario_interpretation": "conditional_liquid_phase_not_verified",
            }
        )
    return pd.DataFrame(rows)


def simulate_reference_cell_scenario(
    proxy_table: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compute temperature-resolved and worst-window conditional cell metrics."""

    scenario = _validated_scenario(config)
    missing_columns = [column for column in REQUIRED_PROXY_COLUMNS if column not in proxy_table]
    if missing_columns:
        raise ValueError(f"Reference-cell input lacks columns: {missing_columns}")
    output = proxy_table.copy()
    positive_columns = [
        "ElectricalConductivity",
        "ThermalConductivity",
        "volumetric_heat_capacity",
    ]
    values = output[positive_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("Reference-cell inputs must be finite and positive")
    area_m2 = float(scenario["electrode_area_cm2"]) * 1.0e-4
    thickness_m = float(scenario["separator_thickness_um"]) * 1.0e-6
    volume_m3 = float(scenario["electrolyte_volume_mL"]) * 1.0e-6
    capacitance_F = float(scenario["nominal_capacitance_F"])
    current_A = float(scenario["charge_discharge_current_A"])
    heat_transfer_coefficient = float(
        scenario["convective_heat_transfer_coefficient_W_m2_K"]
    )
    heat_transfer_area_m2 = int(scenario["exposed_face_count"]) * area_m2
    duration_s = float(scenario["transient_duration_s"])
    output["scenario_electrode_area_m2"] = area_m2
    output["scenario_separator_thickness_m"] = thickness_m
    output["scenario_electrolyte_volume_m3"] = volume_m3
    output["scenario_heat_transfer_area_m2"] = heat_transfer_area_m2
    output["electrolyte_resistance_ohm"] = thickness_m / (
        output["ElectricalConductivity"] * area_m2
    )
    output["electrolyte_RC_time_constant_s"] = (
        output["electrolyte_resistance_ohm"] * capacitance_F
    )
    output["joule_heating_power_W"] = (
        current_A**2 * output["electrolyte_resistance_ohm"]
    )
    output["internal_thermal_conduction_resistance_K_per_W"] = thickness_m / (
        output["ThermalConductivity"] * area_m2
    )
    output["convective_thermal_resistance_K_per_W"] = 1.0 / (
        heat_transfer_coefficient * heat_transfer_area_m2
    )
    output["thermal_resistance_K_per_W"] = (
        output["internal_thermal_conduction_resistance_K_per_W"]
        + output["convective_thermal_resistance_K_per_W"]
    )
    output["electrolyte_thermal_capacitance_J_per_K"] = (
        output["volumetric_heat_capacity"] * volume_m3
    )
    output["lumped_thermal_time_constant_s"] = (
        output["thermal_resistance_K_per_W"]
        * output["electrolyte_thermal_capacitance_J_per_K"]
    )
    output["steady_state_temperature_rise_K"] = (
        output["joule_heating_power_W"] * output["thermal_resistance_K_per_W"]
    )
    output["transient_temperature_rise_K"] = output[
        "steady_state_temperature_rise_K"
    ] * (
        1.0
        - np.exp(-duration_s / output["lumped_thermal_time_constant_s"])
    )
    output["initial_adiabatic_temperature_rise_rate_K_per_s"] = (
        output["joule_heating_power_W"]
        / output["electrolyte_thermal_capacitance_J_per_K"]
    )
    output = _add_relative_resistance(
        output, float(scenario["reference_temperature_K"])
    )
    output = _annotate_comparative_risk(
        output, list(scenario["risk_reference_quantiles"])
    )
    output["scenario_interpretation"] = "conditional_liquid_phase_not_verified"
    output = output.sort_values(["candidate_id", "temperature_K"]).reset_index(drop=True)
    summary = _summarize_reference_cell_metrics(
        output, float(scenario["reference_temperature_K"])
    )
    metadata = {
        "model_scope": "conditional_reference_cell_scenario",
        "not_a_device_prediction": True,
        "liquid_phase_assumed_not_verified": True,
        "scenario": dict(scenario),
        "equations": {
            "electrolyte_resistance_ohm": "separator_thickness_m / (conductivity_S_m-1 * electrode_area_m2)",
            "electrolyte_RC_time_constant_s": "electrolyte_resistance_ohm * nominal_capacitance_F",
            "joule_heating_power_W": "charge_discharge_current_A^2 * electrolyte_resistance_ohm",
            "thermal_resistance_K_per_W": "separator_thickness_m / (thermal_conductivity_W_m-1_K-1 * electrode_area_m2) + 1 / (h_W_m-2_K-1 * exposed_face_count * electrode_area_m2)",
            "thermal_capacitance_J_per_K": "volumetric_heat_capacity_J_m-3_K-1 * electrolyte_volume_m3",
            "transient_temperature_rise_K": "P_Joule * R_thermal * (1 - exp(-duration_s / (R_thermal * C_thermal)))",
        },
        "risk_definition": (
            "At each temperature, electrical resistance and transient temperature rise "
            "are compared with observed-reference q75/q95 values. The worst band and "
            "maximum q75 ratio are comparative priorities, not safety limits."
        ),
    }
    return output, summary, metadata
