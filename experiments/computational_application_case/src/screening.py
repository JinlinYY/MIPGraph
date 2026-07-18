"""Curve diagnostics, data-driven hard constraints, and lead prioritization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .pareto import add_utopia_distance, non_dominated_sort


PROPERTY_NAMES = [
    "Density",
    "ElectricalConductivity",
    "HeatCapacity",
    "SurfaceTension",
    "ThermalConductivity",
    "Viscosity",
]


def _relative_differences(values: np.ndarray, epsilon: float = 1.0e-12) -> np.ndarray:
    if values.size < 2:
        return np.asarray([], dtype=float)
    return np.abs(np.diff(values)) / np.maximum(np.abs(values[:-1]), epsilon)


def audit_curve_quality(
    predictions: pd.DataFrame,
    benchmark: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Flag non-finite, non-positive, out-of-range, and discontinuous curves."""

    flags: list[dict[str, Any]] = []
    reference = predictions[predictions["candidate_type"].eq("observed_reference")]
    jump_quantile = float(config["adjacent_relative_jump_quantile"])
    explosion_factor = float(config["numerical_explosion_factor"])
    reference_jump_thresholds: dict[str, float] = {}
    for prop in PROPERTY_NAMES:
        jumps = []
        for _, group in reference.groupby("candidate_id"):
            values = group.sort_values("temperature_K")[prop].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            jumps.extend(_relative_differences(values).tolist())
        reference_jump_thresholds[prop] = (
            float(np.quantile(jumps, jump_quantile)) if jumps else float("inf")
        )
    temperature_min = float(pd.to_numeric(benchmark["Temperature_K"], errors="coerce").min())
    temperature_max = float(pd.to_numeric(benchmark["Temperature_K"], errors="coerce").max())
    for candidate_id, group in predictions.groupby("candidate_id", sort=True):
        group = group.sort_values("temperature_K")
        for prop in PROPERTY_NAMES:
            value_column = f"{prop}_ActualValue"
            reference_values = (
                pd.to_numeric(benchmark[value_column], errors="coerce")
                if value_column in benchmark
                else pd.Series(dtype=float)
            )
            reference_values = reference_values[np.isfinite(reference_values)]
            reference_range = (
                f"[{float(reference_values.min()):.8g}, {float(reference_values.max()):.8g}]"
                if not reference_values.empty
                else "not_available"
            )
            values = pd.to_numeric(group[prop], errors="coerce").to_numpy(dtype=float)
            temperatures = group["temperature_K"].to_numpy(dtype=float)
            for index, (temperature, value) in enumerate(zip(temperatures, values)):
                if not np.isfinite(value):
                    flags.append(
                        {
                            "candidate_id": candidate_id,
                            "property": prop,
                            "flag_type": "nonfinite",
                            "severity": "severe",
                            "temperature_K": temperature,
                            "value": value,
                            "reference_range": reference_range,
                            "explanation": "Prediction is NaN or infinity.",
                        }
                    )
                elif value <= 0.0:
                    flags.append(
                        {
                            "candidate_id": candidate_id,
                            "property": prop,
                            "flag_type": "nonpositive",
                            "severity": "severe",
                            "temperature_K": temperature,
                            "value": value,
                            "reference_range": reference_range,
                            "explanation": "Positive thermophysical property is non-positive.",
                        }
                    )
                elif not reference_values.empty and (
                    value < float(reference_values.min()) or value > float(reference_values.max())
                ):
                    distance_factor = max(
                        value / max(float(reference_values.max()), 1.0e-12),
                        float(reference_values.min()) / max(value, 1.0e-12),
                    )
                    flags.append(
                        {
                            "candidate_id": candidate_id,
                            "property": prop,
                            "flag_type": "outside_training_property_range",
                            "severity": "severe" if distance_factor > explosion_factor else "warning",
                            "temperature_K": temperature,
                            "value": value,
                            "reference_range": reference_range,
                            "explanation": "Prediction is outside the observed benchmark property range.",
                        }
                    )
                if temperature < temperature_min or temperature > temperature_max:
                    flags.append(
                        {
                            "candidate_id": candidate_id,
                            "property": prop,
                            "flag_type": "temperature_extrapolation",
                            "severity": "warning",
                            "temperature_K": temperature,
                            "value": value,
                            "reference_range": f"[{temperature_min:.4g}, {temperature_max:.4g}] K",
                            "explanation": "Condition is outside the raw benchmark temperature range.",
                        }
                    )
            differences = _relative_differences(values)
            threshold = reference_jump_thresholds[prop]
            for offset, difference in enumerate(differences, start=1):
                if np.isfinite(threshold) and difference > threshold:
                    flags.append(
                        {
                            "candidate_id": candidate_id,
                            "property": prop,
                            "flag_type": "adjacent_relative_jump",
                            "severity": "severe" if difference > explosion_factor * max(threshold, 1.0e-12) else "warning",
                            "temperature_K": temperatures[offset],
                            "value": values[offset],
                            "reference_range": f"reference q{jump_quantile:g}={threshold:.8g}",
                            "explanation": "Adjacent-temperature relative change exceeds the observed-reference rule.",
                        }
                    )
    return pd.DataFrame(
        flags,
        columns=[
            "candidate_id",
            "property",
            "flag_type",
            "severity",
            "temperature_K",
            "value",
            "reference_range",
            "explanation",
        ],
    )


def curve_counts(flags: pd.DataFrame) -> pd.DataFrame:
    """Aggregate warning and severe flag counts by candidate."""

    if flags.empty:
        return pd.DataFrame(
            columns=["candidate_id", "curve_warning_count", "severe_curve_failure_count"]
        )
    counts = flags.assign(
        warning=flags["severity"].eq("warning").astype(int),
        severe=flags["severity"].eq("severe").astype(int),
    ).groupby("candidate_id", as_index=False).agg(
        curve_warning_count=("warning", "sum"),
        severe_curve_failure_count=("severe", "sum"),
    )
    return counts


def derive_reference_thresholds(
    robust_summary: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, float]:
    """Freeze hard-constraint thresholds from observed-reference quantiles."""

    reference = robust_summary[
        robust_summary["candidate_type"].eq("observed_reference")
    ]
    if reference.empty:
        raise ValueError("Observed references are required to derive screening thresholds")
    definitions = {
        "conductivity_min": (
            "conductivity_worst",
            float(config["conductivity_min_reference_quantile"]),
        ),
        "viscosity_max": (
            "viscosity_worst",
            float(config["viscosity_max_reference_quantile"]),
        ),
        "volumetric_heat_capacity_min": (
            "volumetric_heat_capacity_worst",
            float(config["volumetric_heat_capacity_min_reference_quantile"]),
        ),
        "thermal_diffusivity_min": (
            "thermal_diffusivity_worst",
            float(config["thermal_diffusivity_min_reference_quantile"]),
        ),
    }
    thresholds = {
        name: float(np.nanquantile(reference[column].to_numpy(dtype=float), quantile))
        for name, (column, quantile) in definitions.items()
    }
    thresholds["interfacial_deviation_max"] = float(config["interfacial_deviation_max"])
    thresholds["reference_count"] = int(len(reference))
    thresholds.update(
        {
            "conductivity_reference_quantile": float(
                config["conductivity_min_reference_quantile"]
            ),
            "viscosity_reference_quantile": float(config["viscosity_max_reference_quantile"]),
            "volumetric_heat_capacity_reference_quantile": float(
                config["volumetric_heat_capacity_min_reference_quantile"]
            ),
            "thermal_diffusivity_reference_quantile": float(
                config["thermal_diffusivity_min_reference_quantile"]
            ),
        }
    )
    return thresholds


def screen_candidates(
    robust_summary: pd.DataFrame,
    applicability_domain: pd.DataFrame,
    candidate_library: pd.DataFrame,
    thresholds: Mapping[str, float],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Apply structure, inference, curve, AD, and thermophysical hard constraints."""

    ad_columns = ["candidate_id", "AD_status", "AD_reason"]
    merged = robust_summary.merge(
        applicability_domain[ad_columns], on="candidate_id", how="left", validate="one_to_one"
    )
    library_columns = [
        column
        for column in [
            "candidate_id",
            "cation_charge",
            "anion_charge",
            "generation_status",
        ]
        if column in candidate_library
    ]
    merged = merged.merge(
        candidate_library[library_columns], on="candidate_id", how="left", validate="one_to_one"
    )
    required_metrics = [
        "conductivity_worst",
        "viscosity_worst",
        "volumetric_heat_capacity_worst",
        "thermal_diffusivity_worst",
        "interfacial_deviation_worst",
    ]
    merged["pass_structure"] = (
        merged.get("cation_charge", 1).eq(1)
        & merged.get("anion_charge", -1).eq(-1)
        & merged.get("generation_status", "retained").astype(str).str.startswith("retained")
    )
    merged["pass_inference"] = np.isfinite(
        merged[required_metrics].to_numpy(dtype=float)
    ).all(axis=1)
    merged["pass_curve_quality"] = merged.get(
        "severe_curve_failure_count", 0
    ).fillna(0).eq(0)
    merged["pass_AD"] = ~merged["AD_status"].eq("out_of_domain")
    merged["pass_conductivity"] = merged["conductivity_worst"] >= float(
        thresholds["conductivity_min"]
    )
    merged["pass_viscosity"] = merged["viscosity_worst"] <= float(
        thresholds["viscosity_max"]
    )
    merged["pass_heat_capacity"] = merged[
        "volumetric_heat_capacity_worst"
    ] >= float(thresholds["volumetric_heat_capacity_min"])
    merged["pass_thermal_diffusivity"] = merged[
        "thermal_diffusivity_worst"
    ] >= float(thresholds["thermal_diffusivity_min"])
    merged["pass_interfacial_window"] = merged[
        "interfacial_deviation_worst"
    ] <= float(thresholds["interfacial_deviation_max"])
    pass_columns = [
        "pass_structure",
        "pass_inference",
        "pass_curve_quality",
        "pass_AD",
        "pass_conductivity",
        "pass_viscosity",
        "pass_heat_capacity",
        "pass_thermal_diffusivity",
        "pass_interfacial_window",
    ]
    if not bool(config.get("exclude_out_of_domain", True)):
        pass_columns.remove("pass_AD")
    if not bool(config.get("exclude_severe_curve_failures", True)):
        pass_columns.remove("pass_curve_quality")
    merged["final_feasible"] = merged[pass_columns].all(axis=1)
    reasons = []
    for row in merged.itertuples(index=False):
        failed = [
            column.removeprefix("pass_")
            for column in pass_columns
            if not bool(getattr(row, column))
        ]
        reasons.append(";".join(failed))
    merged["failure_reasons"] = reasons
    return merged


def prioritize_candidates(
    feasible_candidates: pd.DataFrame,
    pareto_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank feasible unseen pairs and assign cautious recommendation classes."""

    unseen = feasible_candidates[
        feasible_candidates["candidate_type"].eq("unseen_pair_recombination")
        & feasible_candidates["final_feasible"]
    ].copy()
    objectives = pareto_config["objectives"]
    ranked = add_utopia_distance(non_dominated_sort(unseen, objectives), objectives)
    if ranked.empty:
        for column in [
            "recommendation_class",
            "main_advantage",
            "main_limitation",
            "downstream_priority",
            "uncertainty_status",
        ]:
            ranked[column] = pd.Series(dtype="object")
        return ranked, ranked.copy()
    objective_columns = list(objectives["maximize"]) + list(objectives["minimize"])
    normalized = pd.DataFrame(index=ranked.index)
    for column in objectives["maximize"]:
        low, high = ranked[column].min(), ranked[column].max()
        normalized[column] = 1.0 if np.isclose(low, high) else (ranked[column] - low) / (high - low)
    for column in objectives["minimize"]:
        low, high = ranked[column].min(), ranked[column].max()
        normalized[column] = 1.0 if np.isclose(low, high) else (high - ranked[column]) / (high - low)
    balanced_pool = ranked[ranked["AD_status"].eq("in_domain") & ranked["Pareto_rank"].eq(1)]
    balanced_index = (
        int(balanced_pool["utopia_distance"].idxmin()) if not balanced_pool.empty else None
    )
    classes = []
    advantages = []
    limitations = []
    priorities = []
    for index, row in ranked.iterrows():
        scores = normalized.loc[index]
        best_metric = str(scores.idxmax())
        worst_metric = str(scores.idxmin())
        if str(row["AD_status"]) == "borderline":
            recommendation = "exploratory"
            priority = "AD-focused qualification before broader testing"
        elif balanced_index is not None and index == balanced_index:
            recommendation = "balanced"
            priority = "balanced thermophysical qualification"
        else:
            transport = float(
                np.mean(
                    [
                        scores.get("conductivity_worst", np.nan),
                        scores.get("viscosity_worst", np.nan),
                    ]
                )
            )
            thermal = float(
                np.mean(
                    [
                        scores.get("volumetric_heat_capacity_worst", np.nan),
                        scores.get("thermal_diffusivity_worst", np.nan),
                    ]
                )
            )
            if transport >= thermal:
                recommendation = "high_transport"
                priority = "transport and electrochemical-window qualification"
            else:
                recommendation = "thermal_robust"
                priority = "thermal-property and phase-window qualification"
        classes.append(recommendation)
        advantages.append(best_metric)
        limitations.append(worst_metric)
        priorities.append(priority)
    ranked["recommendation_class"] = classes
    ranked["main_advantage"] = advantages
    ranked["main_limitation"] = limitations
    ranked["downstream_priority"] = priorities
    ranked["uncertainty_status"] = ranked.get("uncertainty_status", "not_available")
    maximum = int(pareto_config["final_candidate_max"])
    minimum = int(pareto_config["final_candidate_min"])
    ranked["_ad_order"] = ranked["AD_status"].map(
        {"in_domain": 0, "borderline": 1, "out_of_domain": 2}
    ).fillna(3)
    ranked["_distance_order"] = ranked["utopia_distance"].fillna(float("inf"))
    ordered = ranked.sort_values(
        ["_ad_order", "Pareto_rank", "_distance_order", "candidate_id"]
    )
    selection_count = min(maximum, len(ordered))
    if selection_count < minimum:
        selection_count = len(ordered)
    final = ordered.head(selection_count).drop(columns=["_ad_order", "_distance_order"])
    ranked = ranked.drop(columns=["_ad_order", "_distance_order"])
    return ranked.reset_index(drop=True), final.reset_index(drop=True)

