"""Shared cross-ion attention profiles and non-causal property contrasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .feature_extractor import FeatureBundle
from .revision_analysis import aggregate_identity_responses, identity_key


@dataclass
class CrossIonResults:
    interaction_statistics: pd.DataFrame
    property_conditioned_interactions: pd.DataFrame
    family_interaction_profiles: pd.DataFrame
    high_low_property_contrasts: pd.DataFrame
    family_stratified_contrasts: pd.DataFrame
    metadata: dict[str, Any]


class CrossIonAnalyzer:
    """Analyse shared attention profiles with one contribution per IL identity."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.min_group_size = int(
            config["attention"].get("minimum_samples_per_group", 20)
        )
        revision = config.get("revision_analysis", {})
        self.bootstrap_repeats = int(
            revision.get(
                "identity_bootstrap_repeats",
                config["statistics"].get("bootstrap_repeats", 500),
            )
        )
        self.confidence = float(
            config["statistics"].get("confidence_level", 0.95)
        )
        self.seed = int(config["statistics"].get("random_seed", 42))
        self.temperature_knots = int(
            revision.get("temperature_spline_knots", 4)
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
        identity = bundle.records[
            [
                "sample_id",
                "cation_smiles",
                "anion_smiles",
                "cation_family",
                "anion_family",
            ]
        ].copy()
        identity["il_identity_key"] = identity_key(identity)
        merged = attention.merge(
            identity,
            on="sample_id",
            how="inner",
            validate="many_to_one",
        )
        identity_attention = (
            merged.groupby(
                [
                    "il_identity_key",
                    "cation_family",
                    "anion_family",
                    "interaction_category",
                ],
                as_index=False,
            )
            .agg(
                total_attention_mass=("attention_mass", "median"),
                attention_per_pair=("attention_per_pair", "median"),
                pair_count=("pair_count", "median"),
                n_condition_records=("sample_id", "nunique"),
            )
        )
        statistics = (
            identity_attention.groupby("interaction_category", as_index=False)
            .agg(
                n_unique_ils=("il_identity_key", "nunique"),
                mean_total_attention_mass=("total_attention_mass", "mean"),
                median_total_attention_mass=("total_attention_mass", "median"),
                mean_attention_per_pair=("attention_per_pair", "mean"),
                median_attention_per_pair=("attention_per_pair", "median"),
                mean_pair_count=("pair_count", "mean"),
            )
        )
        statistics["analysis_weighting"] = "one median profile per IL identity"
        family_profiles = self._aggregate(
            identity_attention.rename(
                columns={
                    "il_identity_key": "sample_id",
                    "total_attention_mass": "attention_mass",
                }
            ),
            ["cation_family", "anion_family", "interaction_category"],
        )
        family_profiles = family_profiles.loc[
            family_profiles["sample_count"] >= self.min_group_size
        ].reset_index(drop=True)
        contrast_rows: list[dict[str, Any]] = []
        family_contrast_rows: list[dict[str, Any]] = []
        reference_pressure = float(
            self.config["conditions"].get("reference_pressure_kpa", 101.325)
        )
        for property_index, property_name in enumerate(
            bundle.metadata["property_order"]
        ):
            actual, _, _, _ = aggregate_identity_responses(
                bundle.records,
                property_name,
                temperature_knots=self.temperature_knots,
                reference_pressure_kpa=reference_pressure,
            )
            if len(actual) < 2 * self.min_group_size:
                continue
            response_column = "condition_adjusted_log_response"
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
            comparison = identity_attention.merge(
                actual[
                    [
                        "il_identity_key",
                        "property_group",
                        "cation_family",
                        "anion_family",
                    ]
                ],
                on="il_identity_key",
                how="inner",
                suffixes=("", "_response"),
            )
            comparison = comparison.loc[
                comparison["property_group"].isin(["low", "high"])
            ]
            for category, category_frame in comparison.groupby(
                "interaction_category"
            ):
                family_low = (
                    category_frame.loc[
                        category_frame["property_group"] == "low",
                        ["cation_family", "anion_family"],
                    ]
                    .astype(str)
                    .agg("||".join, axis=1)
                    .value_counts(normalize=True)
                )
                family_high = (
                    category_frame.loc[
                        category_frame["property_group"] == "high",
                        ["cation_family", "anion_family"],
                    ]
                    .astype(str)
                    .agg("||".join, axis=1)
                    .value_counts(normalize=True)
                )
                all_families = family_low.index.union(family_high.index)
                family_tv = 0.5 * float(
                    np.abs(
                        family_low.reindex(all_families, fill_value=0.0)
                        - family_high.reindex(all_families, fill_value=0.0)
                    ).sum()
                )
                for metric in (
                    "total_attention_mass",
                    "attention_per_pair",
                    "pair_count",
                ):
                    low_values = category_frame.loc[
                        category_frame["property_group"] == "low",
                        metric,
                    ].to_numpy(dtype=float)
                    high_values = category_frame.loc[
                        category_frame["property_group"] == "high",
                        metric,
                    ].to_numpy(dtype=float)
                    if (
                        len(low_values) < self.min_group_size
                        or len(high_values) < self.min_group_size
                    ):
                        continue
                    difference = float(high_values.mean() - low_values.mean())
                    rng = np.random.default_rng(
                        self.seed
                        + property_index * 10000
                        + sum(ord(character) for character in str(category))
                        + {
                            "total_attention_mass": 0,
                            "attention_per_pair": 1,
                            "pair_count": 2,
                        }[metric]
                    )
                    bootstrap = np.asarray(
                        [
                            rng.choice(
                                high_values,
                                len(high_values),
                                replace=True,
                            ).mean()
                            - rng.choice(
                                low_values,
                                len(low_values),
                                replace=True,
                            ).mean()
                            for _ in range(self.bootstrap_repeats)
                        ],
                        dtype=float,
                    )
                    alpha = (1.0 - self.confidence) / 2.0
                    ci_low = float(np.quantile(bootstrap, alpha))
                    ci_high = float(
                        np.quantile(bootstrap, 1.0 - alpha)
                    )
                    contrast_rows.append(
                        {
                            "property": property_name,
                            "interaction_category": category,
                            "attention_metric": metric,
                            "low_group_mean": float(low_values.mean()),
                            "high_group_mean": float(high_values.mean()),
                            "high_minus_low": difference,
                            "bootstrap_ci_low": ci_low,
                            "bootstrap_ci_high": ci_high,
                            "ci_excludes_zero": bool(
                                ci_low > 0 or ci_high < 0
                            ),
                            "main_figure_eligible": bool(
                                metric == "attention_per_pair"
                                and (ci_low > 0 or ci_high < 0)
                            ),
                            "low_group_unique_ils": int(len(low_values)),
                            "high_group_unique_ils": int(len(high_values)),
                            "family_composition_total_variation": family_tv,
                            "grouping_basis": (
                                "unique-IL quartiles of condition-adjusted "
                                "observed log-property residual"
                            ),
                            "interpretation_scope": (
                                "non-causal association with shared encoder "
                                "attention; not interaction energy"
                            ),
                        }
                    )
                    for (
                        cation_family,
                        anion_family,
                    ), family_group in category_frame.groupby(
                        ["cation_family", "anion_family"]
                    ):
                        family_low_values = family_group.loc[
                            family_group["property_group"] == "low",
                            metric,
                        ].to_numpy(dtype=float)
                        family_high_values = family_group.loc[
                            family_group["property_group"] == "high",
                            metric,
                        ].to_numpy(dtype=float)
                        if (
                            len(family_low_values) < 5
                            or len(family_high_values) < 5
                        ):
                            continue
                        family_contrast_rows.append(
                            {
                                "property": property_name,
                                "interaction_category": category,
                                "attention_metric": metric,
                                "cation_family": cation_family,
                                "anion_family": anion_family,
                                "low_group_unique_ils": int(
                                    len(family_low_values)
                                ),
                                "high_group_unique_ils": int(
                                    len(family_high_values)
                                ),
                                "high_minus_low": float(
                                    family_high_values.mean()
                                    - family_low_values.mean()
                                ),
                                "minimum_per_quartile": 5,
                                "interpretation_scope": (
                                    "family-stratified sensitivity analysis; "
                                    "descriptive and non-causal"
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
            family_stratified_contrasts=pd.DataFrame(
                family_contrast_rows
            ),
            metadata={
                "attention_architecture": "shared structural encoder",
                "property_specific_claim": False,
                "normalization": (
                    "total attention mass, attention per valid atom pair, and "
                    "valid pair count are reported per unique IL identity"
                ),
                "bootstrap_unit": "IL identity",
                "physical_interaction_energy_claim": False,
            },
        )
