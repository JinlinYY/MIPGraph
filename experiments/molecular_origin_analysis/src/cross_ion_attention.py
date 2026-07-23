"""Shared cross-ion attention profiles and non-causal property contrasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .feature_extractor import FeatureBundle


@dataclass
class CrossIonResults:
    interaction_statistics: pd.DataFrame
    property_conditioned_interactions: pd.DataFrame
    family_interaction_profiles: pd.DataFrame
    high_low_property_contrasts: pd.DataFrame
    metadata: dict[str, Any]


class CrossIonAnalyzer:
    """Analyse shared attention mass after per-pair-count correction."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.min_group_size = int(
            config["attention"].get("minimum_samples_per_group", 20)
        )

    @staticmethod
    def _aggregate(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
        return (
            frame.groupby(group_columns, as_index=False)
            .agg(
                sample_count=("sample_id", "nunique"),
                total_pair_count=("pair_count", "sum"),
                mean_attention_mass=("attention_mass", "mean"),
                mean_attention_per_pair=("attention_per_pair", "mean"),
                median_attention_per_pair=("attention_per_pair", "median"),
            )
        )

    @staticmethod
    def _condition_residuals(
        records: pd.DataFrame,
        property_name: str,
    ) -> pd.DataFrame:
        """Residualize an observed log-property before defining high/low groups."""

        value_column = f"{property_name}_ActualValue"
        mask_column = f"{property_name}_mask"
        valid = (
            (records[mask_column].to_numpy(dtype=float) > 0)
            & np.isfinite(records[value_column].to_numpy(dtype=float))
            & (records[value_column].to_numpy(dtype=float) > 0)
        )
        response = np.log(
            records.loc[valid, value_column].to_numpy(dtype=float)
        )
        numeric = records.loc[valid, ["Temperature_K", "Pressure_kPa"]].copy()
        pressure_median = numeric["Pressure_kPa"].median()
        numeric["Pressure_kPa"] = numeric["Pressure_kPa"].fillna(
            pressure_median if np.isfinite(pressure_median) else 101.325
        )
        families = pd.get_dummies(
            records.loc[valid, ["cation_family", "anion_family"]].astype(str),
            drop_first=True,
            dtype=float,
        )
        covariates = np.column_stack(
            [numeric.to_numpy(dtype=float), families.to_numpy(dtype=float)]
        )
        if len(response) <= covariates.shape[1] + 2:
            covariates = numeric.to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(response)), covariates])
        coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
        residuals = response - design @ coefficients
        return pd.DataFrame(
            {
                "sample_id": records.loc[valid, "sample_id"].to_numpy(),
                "condition_controlled_log_residual": residuals,
            }
        )

    def run(
        self,
        bundle: FeatureBundle,
        property_conditioned: pd.DataFrame | None = None,
    ) -> CrossIonResults:
        attention = bundle.auxiliary_tables.get("cross_ion_attention", pd.DataFrame())
        if attention.empty:
            raise ValueError("No extracted cross-ion attention summary is available")
        statistics = self._aggregate(attention, ["interaction_category"])
        identity = bundle.records[
            ["sample_id", "cation_family", "anion_family", "Temperature_K"]
        ]
        merged = attention.merge(identity, on="sample_id", how="inner", validate="many_to_one")
        family_profiles = self._aggregate(
            merged,
            ["cation_family", "anion_family", "interaction_category"],
        )
        family_profiles = family_profiles.loc[
            family_profiles["sample_count"] >= self.min_group_size
        ].reset_index(drop=True)
        contrast_rows: list[dict[str, Any]] = []
        for property_name in bundle.metadata["property_order"]:
            actual = self._condition_residuals(
                bundle.records,
                property_name,
            )
            if len(actual) < 2 * self.min_group_size:
                continue
            response_column = "condition_controlled_log_residual"
            low = float(actual[response_column].quantile(0.25))
            high = float(actual[response_column].quantile(0.75))
            actual["property_group"] = np.where(
                actual[response_column] <= low,
                "low",
                np.where(
                    actual[response_column] >= high,
                    "high",
                    "middle",
                ),
            )
            comparison = attention.merge(
                actual[["sample_id", "property_group"]],
                on="sample_id",
                how="inner",
            )
            comparison = comparison.loc[
                comparison["property_group"].isin(["low", "high"])
            ]
            grouped = comparison.groupby(
                ["interaction_category", "property_group"]
            )["attention_per_pair"].agg(["mean", "count"]).reset_index()
            pivot = grouped.pivot(
                index="interaction_category",
                columns="property_group",
                values=["mean", "count"],
            )
            for category in pivot.index:
                low_mean = float(pivot.loc[category, ("mean", "low")])
                high_mean = float(pivot.loc[category, ("mean", "high")])
                contrast_rows.append(
                    {
                        "property": property_name,
                        "interaction_category": category,
                        "low_group_mean_attention_per_pair": low_mean,
                        "high_group_mean_attention_per_pair": high_mean,
                        "high_minus_low": high_mean - low_mean,
                        "low_group_count": int(pivot.loc[category, ("count", "low")]),
                        "high_group_count": int(pivot.loc[category, ("count", "high")]),
                        "grouping_basis": (
                            "quartiles of observed log-property residual after "
                            "temperature, pressure, cation-family, and anion-family controls"
                        ),
                        "interpretation_scope": (
                            "association between shared attention profile and "
                            "condition-controlled observed-property group"
                        ),
                    }
                )
        conditioned = (
            property_conditioned.copy()
            if property_conditioned is not None
            else pd.DataFrame()
        )
        return CrossIonResults(
            interaction_statistics=statistics,
            property_conditioned_interactions=conditioned,
            family_interaction_profiles=family_profiles,
            high_low_property_contrasts=pd.DataFrame(contrast_rows),
            metadata={
                "attention_architecture": "shared structural encoder",
                "property_specific_claim": False,
                "normalization": (
                    "attention mass and attention per valid atom pair are both reported"
                ),
                "physical_interaction_energy_claim": False,
            },
        )
