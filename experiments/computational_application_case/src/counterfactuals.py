"""Matched one-ion substitution comparisons with non-causal language."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


COMPARISON_METRICS = [
    "Density",
    "Viscosity",
    "ElectricalConductivity",
    "HeatCapacity",
    "SurfaceTension",
    "ThermalConductivity",
    "transport_favorability",
    "volumetric_heat_capacity",
    "thermal_diffusivity",
    "simplified_thermal_diffusion_timescale",
    "surface_tension_reference_envelope_deviation",
]


def analyze_counterfactual_substitutions(
    proxy_table: pd.DataFrame,
    target_temperatures: Sequence[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare inferred pairs sharing exactly one canonical ion."""

    cation_key = (
        "cation_identity_key"
        if "cation_identity_key" in proxy_table.columns
        else "canonical_cation_smiles"
    )
    anion_key = (
        "anion_identity_key"
        if "anion_identity_key" in proxy_table.columns
        else "canonical_anion_smiles"
    )
    required = {
        "candidate_id",
        "candidate_type",
        "canonical_cation_smiles",
        "canonical_anion_smiles",
        "temperature_K",
        *COMPARISON_METRICS,
    }
    missing = sorted(required - set(proxy_table.columns))
    if missing:
        raise ValueError(f"Counterfactual input lacks required columns: {missing}")
    rows: list[dict[str, Any]] = []
    for target in target_temperatures:
        at_temperature = proxy_table[
            np.isclose(proxy_table["temperature_K"].to_numpy(dtype=float), float(target))
        ].drop_duplicates("candidate_id")
        for shared_column, replaced_column, substitution_type in [
            (
                cation_key,
                anion_key,
                "fixed_cation_replace_anion",
            ),
            (
                anion_key,
                cation_key,
                "fixed_anion_replace_cation",
            ),
        ]:
            for shared_value, group in at_temperature.groupby(shared_column):
                records = group.sort_values("candidate_id").reset_index(drop=True)
                for left_index in range(len(records)):
                    for right_index in range(left_index + 1, len(records)):
                        left = records.iloc[left_index]
                        right = records.iloc[right_index]
                        if left[replaced_column] == right[replaced_column]:
                            continue
                        if not (
                            left["candidate_type"] == "unseen_pair_recombination"
                            or right["candidate_type"] == "unseen_pair_recombination"
                        ):
                            continue
                        if int(left.get("severe_curve_failure_count", 0)) > 0 or int(
                            right.get("severe_curve_failure_count", 0)
                        ) > 0:
                            continue
                        row = {
                            "temperature_K": float(target),
                            "substitution_type": substitution_type,
                            "shared_ion": shared_value,
                            "candidate_from": left["candidate_id"],
                            "candidate_to": right["candidate_id"],
                            "replaced_ion_from": left[replaced_column],
                            "replaced_ion_to": right[replaced_column],
                            "AD_status_from": left.get("AD_status", "not_available"),
                            "AD_status_to": right.get("AD_status", "not_available"),
                            "interpretation": (
                                "The ion substitution is associated with a predicted shift and suggests a trade-off within the represented chemical domain."
                            ),
                        }
                        for metric in COMPARISON_METRICS:
                            left_value = float(left[metric])
                            right_value = float(right[metric])
                            row[f"delta_{metric}"] = (
                                right_value - left_value
                                if np.isfinite(left_value) and np.isfinite(right_value)
                                else np.nan
                            )
                        rows.append(row)
    comparisons = pd.DataFrame(rows)
    if comparisons.empty:
        return comparisons, pd.DataFrame(
            columns=["substitution_type", "comparison_count", "interpretation"]
        )
    delta_columns = [column for column in comparisons if column.startswith("delta_")]
    summary = comparisons.groupby("substitution_type", as_index=False).agg(
        comparison_count=("candidate_from", "size"),
        **{f"mean_{column}": (column, "mean") for column in delta_columns},
    )
    summary["interpretation"] = (
        "Mean shifts are associations from matched model predictions, not causal mechanisms."
    )
    return comparisons, summary
