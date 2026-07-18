"""Checkpoint-ensemble uncertainty and decision-stability propagation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .schema import PROPERTY_NAMES


PROXY_UNCERTAINTY_COLUMNS = [
    "cp_mass_J_kg-1_K-1",
    "volumetric_heat_capacity",
    "thermal_diffusivity",
    "simplified_thermal_diffusion_timescale",
    "electrolyte_mass_kg",
    "transport_favorability",
    "interfacial_window_deviation",
    "thermal_effusivity",
    "electrolyte_resistance_ohm",
    "electrolyte_RC_time_constant_s",
    "joule_heating_power_W",
    "steady_state_temperature_rise_K",
    "transient_temperature_rise_K",
    "reference_cell_risk_index",
]

ENSEMBLE_COMPATIBILITY_FIELDS = (
    "model_class",
    "model_structure_fingerprint",
    "property_order",
    "property_units",
    "graph_config_fingerprint",
    "graph_feature_dimension",
    "global_descriptor_dimension",
    "functional_group_dimension",
    "condition_scaler_class",
    "target_scaler_class",
    "target_inverse_transform",
    "target_epsilon",
)


def validate_ensemble_compatibility(
    member_metadata: Sequence[Mapping[str, Any]],
) -> None:
    """Fail before averaging checkpoints with incompatible runtime semantics."""

    if len(member_metadata) < 2:
        return
    reference = member_metadata[0]
    problems: list[str] = []
    for member_index, metadata in enumerate(member_metadata[1:], start=2):
        for field in ENSEMBLE_COMPATIBILITY_FIELDS:
            if field not in reference or field not in metadata:
                problems.append(f"member {member_index}: missing {field}")
            elif metadata[field] != reference[field]:
                problems.append(f"member {member_index}: incompatible {field}")
    if problems:
        raise ValueError(
            "Checkpoint ensemble members are incompatible: " + "; ".join(problems)
        )


def _mean_std_table(
    table: pd.DataFrame,
    value_columns: Sequence[str],
) -> pd.DataFrame:
    keys = ["candidate_id", "temperature_K", "pressure_kPa"]
    aggregated = table.groupby(keys)[list(value_columns)].agg(["mean", "std"]).reset_index()
    aggregated.columns = [
        str(column[0])
        if isinstance(column, tuple) and not column[1]
        else f"{column[0]}_{column[1]}"
        if isinstance(column, tuple)
        else str(column)
        for column in aggregated.columns
    ]
    return aggregated


def estimate_uncertainty(
    predictions: pd.DataFrame,
    proxy_table: pd.DataFrame,
    checkpoint_paths: Sequence[str],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Aggregate compatible ensemble rows, otherwise report unavailability."""

    minimum = int(config["ensemble_min_checkpoints"])
    checkpoint_count = len(list(checkpoint_paths))
    key_columns = ["candidate_id", "temperature_K", "pressure_kPa"]
    property_member_count = (
        predictions.groupby(key_columns)["checkpoint_name"].nunique()
        if "checkpoint_name" in predictions
        else pd.Series(dtype=int)
    )
    proxy_member_count = (
        proxy_table.groupby(key_columns)["checkpoint_name"].nunique()
        if "checkpoint_name" in proxy_table
        else pd.Series(dtype=int)
    )
    has_ensemble_rows = bool(
        checkpoint_count >= minimum
        and not property_member_count.empty
        and not proxy_member_count.empty
        and property_member_count.min() >= minimum
        and proxy_member_count.min() >= minimum
    )
    if has_ensemble_rows:
        property_uncertainty = _mean_std_table(predictions, PROPERTY_NAMES)
        proxy_columns = [
            column for column in PROXY_UNCERTAINTY_COLUMNS if column in proxy_table
        ]
        proxy_uncertainty = _mean_std_table(proxy_table, proxy_columns)
        status = {
            "uncertainty_status": "checkpoint_ensemble",
            "checkpoint_count": checkpoint_count,
            "minimum_member_count_per_condition": int(property_member_count.min()),
            "propagation_status": "properties_and_application_proxies_propagated",
        }
    else:
        property_uncertainty = predictions[key_columns].drop_duplicates().copy()
        property_uncertainty["uncertainty_status"] = "not_available"
        property_uncertainty["reason"] = (
            "Fewer than three compatible checkpoint predictions and no held-out residual calibration were configured."
        )
        proxy_uncertainty = proxy_table[key_columns].drop_duplicates().copy()
        proxy_uncertainty["uncertainty_status"] = "not_available"
        proxy_uncertainty["reason"] = property_uncertainty["reason"].iloc[0]
        status = {
            "uncertainty_status": "not_available",
            "checkpoint_count": checkpoint_count,
            "reason": property_uncertainty["reason"].iloc[0],
        }
    feasibility = predictions[["candidate_id"]].drop_duplicates().copy()
    feasibility["constraint_pass_probability"] = np.nan
    feasibility["pareto_rank_1_probability"] = np.nan
    feasibility["uncertainty_status"] = status["uncertainty_status"]
    return property_uncertainty, proxy_uncertainty, feasibility, status


def estimate_ensemble_decision_probabilities(
    proxy_members: pd.DataFrame,
    benchmark: pd.DataFrame,
    applicability_domain: pd.DataFrame,
    candidate_library: pd.DataFrame,
    fixed_thresholds: Mapping[str, float],
    curve_config: Mapping[str, Any],
    screening_config: Mapping[str, Any],
    pareto_config: Mapping[str, Any],
    reference_cell_config: Mapping[str, Any],
) -> pd.DataFrame:
    """Propagate ensemble members through full-window constraints and Pareto rank."""

    from .proxies import summarize_whole_temperature_window
    from .reference_cell import simulate_reference_cell_scenario
    from .screening import (
        audit_curve_quality,
        curve_counts,
        prioritize_candidates,
        screen_candidates,
    )

    member_rows: list[pd.DataFrame] = []
    for checkpoint_name, member in proxy_members.groupby("checkpoint_name", sort=True):
        if "analysis_window" in member:
            member = member[member["analysis_window"].eq("main")]
        flags = audit_curve_quality(member, benchmark, curve_config)
        counts = curve_counts(flags)
        robust = summarize_whole_temperature_window(member).merge(
            counts, on="candidate_id", how="left", suffixes=("", "_audit")
        )
        _, scenario_summary, _ = simulate_reference_cell_scenario(
            member, reference_cell_config
        )
        robust = robust.merge(
            scenario_summary.drop(columns=["candidate_type"], errors="ignore"),
            on="candidate_id",
            how="left",
            validate="one_to_one",
        )
        for column in ["curve_warning_count", "severe_curve_failure_count"]:
            audit_column = f"{column}_audit"
            if audit_column in robust:
                robust[column] = robust[audit_column].fillna(robust[column]).fillna(0).astype(int)
                robust = robust.drop(columns=audit_column)
            else:
                robust[column] = robust.get(column, 0)
        trace = screen_candidates(
            robust,
            applicability_domain,
            candidate_library,
            fixed_thresholds,
            screening_config,
        )
        ranked, _ = prioritize_candidates(trace, pareto_config)
        rank_one_ids = set(
            ranked.loc[ranked["Pareto_rank"].eq(1), "candidate_id"]
            if not ranked.empty
            else []
        )
        decisions = trace[["candidate_id"]].copy()
        decisions["checkpoint_name"] = checkpoint_name
        decisions["constraint_pass"] = trace["final_feasible"].astype(float)
        decisions["pareto_rank_1"] = decisions["candidate_id"].isin(rank_one_ids).astype(float)
        member_rows.append(decisions)
    combined = pd.concat(member_rows, ignore_index=True)
    probabilities = combined.groupby("candidate_id", as_index=False).agg(
        constraint_pass_probability=("constraint_pass", "mean"),
        pareto_rank_1_probability=("pareto_rank_1", "mean"),
        ensemble_member_count=("checkpoint_name", "nunique"),
    )
    probabilities["uncertainty_status"] = "checkpoint_ensemble"
    return probabilities
