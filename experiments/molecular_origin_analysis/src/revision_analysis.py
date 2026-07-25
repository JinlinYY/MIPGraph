"""Identity-balanced molecular-structure--property revision analyses.

This module implements the reviewer-facing statistical corrections requested
for the molecular-origin case without changing the trained model, checkpoint,
data split, or original observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import SplineTransformer

from .feature_extractor import (
    DESCRIPTOR_NAMES,
    FUNCTIONAL_GROUP_NAMES,
    FeatureBundle,
)
from .structure_property import benjamini_hochberg


ALL_FEATURES = DESCRIPTOR_NAMES + FUNCTIONAL_GROUP_NAMES


def identity_key(records: pd.DataFrame) -> pd.Series:
    """Return a charge-aware ion-pair identity key already canonicalized upstream."""

    return (
        records["cation_smiles"].astype(str)
        + "||"
        + records["anion_smiles"].astype(str)
    )


def feature_scope(feature: str) -> str:
    prefix = feature.split("_", 1)[0]
    return {
        "cation": "Cation",
        "anion": "Anion",
        "pair": "Ion pair",
    }.get(prefix, "Other")


def _safe_corr(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "pearson",
) -> tuple[float, float]:
    finite = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, dtype=float)[finite]
    y = np.asarray(y, dtype=float)[finite]
    if len(x) < 4 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return np.nan, np.nan
    result = (
        stats.pearsonr(x, y)
        if method == "pearson"
        else stats.spearmanr(x, y)
    )
    return float(result.statistic), float(result.pvalue)


def _residualize(values: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    covariates = np.asarray(covariates, dtype=float)
    finite = np.isfinite(values)
    if covariates.size:
        finite &= np.isfinite(covariates).all(axis=1)
    residuals = np.full(len(values), np.nan, dtype=float)
    if finite.sum() < 4:
        return residuals
    design = np.ones((finite.sum(), 1), dtype=float)
    if covariates.size:
        design = np.column_stack([design, covariates[finite]])
    coefficients, *_ = np.linalg.lstsq(design, values[finite], rcond=None)
    residuals[finite] = values[finite] - design @ coefficients
    return residuals


def _drop_constant_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    keep = [
        column
        for column in frame
        if np.nanstd(frame[column].to_numpy(dtype=float)) > 1e-12
    ]
    return frame[keep]


def build_condition_covariates(
    records: pd.DataFrame,
    *,
    temperature_knots: int,
    reference_pressure_kpa: float,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Build a full-rank condition/family design with pressure availability."""

    temperature = records["Temperature_K"].to_numpy(dtype=float).reshape(-1, 1)
    spline = SplineTransformer(
        n_knots=max(int(temperature_knots), 3),
        degree=3,
        include_bias=False,
    )
    spline_values = spline.fit_transform(temperature)
    covariates = pd.DataFrame(
        spline_values,
        index=records.index,
        columns=[f"temperature_spline_{i + 1}" for i in range(spline_values.shape[1])],
    )

    pressure_available = records["Pressure_kPa"].notna().astype(float)
    available_pressure = pd.to_numeric(
        records.loc[pressure_available.astype(bool), "Pressure_kPa"],
        errors="coerce",
    )
    pressure_fill = (
        float(available_pressure.median())
        if available_pressure.notna().any()
        else float(reference_pressure_kpa)
    )
    pressure = (
        pd.to_numeric(records["Pressure_kPa"], errors="coerce")
        .fillna(pressure_fill)
        .astype(float)
    )
    pressure_std = float(pressure.std(ddof=0))
    covariates["pressure_standardized"] = (
        (pressure - float(pressure.mean())) / pressure_std
        if pressure_std > 1e-12
        else 0.0
    )
    covariates["pressure_available"] = pressure_available

    family = pd.get_dummies(
        records[["cation_family", "anion_family"]].astype(str),
        prefix=["cation_family", "anion_family"],
        drop_first=True,
        dtype=float,
    )
    covariates = pd.concat([covariates, family], axis=1)

    source_columns = [
        name
        for name in ("data_source", "source", "reference_id")
        if name in records.columns
    ]
    source_used = None
    if source_columns:
        source_used = source_columns[0]
        source = pd.get_dummies(
            records[source_used].fillna("missing").astype(str),
            prefix="data_source",
            drop_first=True,
            dtype=float,
        )
        covariates = pd.concat([covariates, source], axis=1)

    covariates = _drop_constant_columns(covariates)
    return (
        covariates.to_numpy(dtype=float),
        covariates.columns.tolist(),
        {
            "pressure_fill_kpa": pressure_fill,
            "pressure_missing_count": int((pressure_available == 0).sum()),
            "pressure_availability_in_design": (
                "pressure_available" in covariates.columns
            ),
            "data_source_covariate": source_used or "unavailable_in_source_table",
        },
    )


def aggregate_identity_responses(
    records: pd.DataFrame,
    property_name: str,
    *,
    response_column: str | None = None,
    temperature_knots: int = 4,
    reference_pressure_kpa: float = 101.325,
    extra_response_covariates: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    """Residualize log response on records, then aggregate once per IL identity."""

    value_column = response_column or f"{property_name}_ActualValue"
    values = pd.to_numeric(records[value_column], errors="coerce").to_numpy(dtype=float)
    mask_column = f"{property_name}_mask"
    valid = np.isfinite(values) & (values > 0)
    if mask_column in records:
        valid &= records[mask_column].to_numpy(dtype=float) > 0
    selected = records.loc[valid].copy()
    response = np.log(values[valid])
    covariates, covariate_names, metadata = build_condition_covariates(
        selected,
        temperature_knots=temperature_knots,
        reference_pressure_kpa=reference_pressure_kpa,
    )
    if extra_response_covariates is not None:
        extra = extra_response_covariates.loc[selected.index].to_numpy(dtype=float)
        covariates = np.column_stack([covariates, extra])
        covariate_names.extend(extra_response_covariates.columns.tolist())
    residual = _residualize(response, covariates)
    selected["il_identity_key"] = identity_key(selected)
    selected["condition_adjusted_log_response"] = residual
    selected["observed_log_response"] = response
    identity = (
        selected.groupby("il_identity_key", as_index=False)
        .agg(
            cation_smiles=("cation_smiles", "first"),
            anion_smiles=("anion_smiles", "first"),
            cation_family=("cation_family", "first"),
            anion_family=("anion_family", "first"),
            n_records=("sample_id", "size"),
            condition_adjusted_log_response=(
                "condition_adjusted_log_response",
                "median",
            ),
            observed_log_response=("observed_log_response", "median"),
        )
    )
    metadata.update(
        {
            "property": property_name,
            "n_records": int(valid.sum()),
            "n_unique_ils": int(len(identity)),
            "covariates": covariate_names,
            "aggregation": "median of record-level residuals per IL identity",
        }
    )
    return identity, valid, covariates, metadata


def _family_proxy_diagnostics(
    feature_values: np.ndarray,
    identity_frame: pd.DataFrame,
) -> dict[str, float]:
    values = np.asarray(feature_values, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 4 or np.nanstd(values[finite]) <= 1e-12:
        return {
            "raw_feature_std": float(np.nanstd(values)),
            "family_residual_std": 0.0,
            "family_residual_std_ratio": 0.0,
            "family_predictability_r2": 1.0,
            "variance_inflation_factor": np.inf,
            "condition_number": np.inf,
        }
    family = pd.get_dummies(
        identity_frame.loc[
            finite,
            ["cation_family", "anion_family"],
        ].astype(str),
        drop_first=True,
        dtype=float,
    )
    family = _drop_constant_columns(family)
    x = family.to_numpy(dtype=float)
    y = values[finite]
    residual = _residualize(y, x)
    total = float(np.sum((y - y.mean()) ** 2))
    error = float(np.nansum(residual**2))
    r_squared = 1.0 - error / total if total > 1e-12 else 1.0
    vif = 1.0 / max(1.0 - r_squared, 1e-12)
    raw_std = float(np.std(y))
    residual_std = float(np.nanstd(residual))
    standardized_y = (y - y.mean()) / max(raw_std, 1e-12)
    design = np.column_stack([np.ones(len(y)), x, standardized_y])
    condition_number = float(np.linalg.cond(design))
    return {
        "raw_feature_std": raw_std,
        "family_residual_std": residual_std,
        "family_residual_std_ratio": residual_std / max(raw_std, 1e-12),
        "family_predictability_r2": float(np.clip(r_squared, 0.0, 1.0)),
        "variance_inflation_factor": float(vif),
        "condition_number": condition_number,
    }


def _bootstrap_correlations(
    x: np.ndarray,
    y: np.ndarray,
    repeats: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Identity-level bootstrap correlations for a common complete matrix."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    output = np.full((repeats, x.shape[1]), np.nan, dtype=float)
    for repeat in range(repeats):
        indices = rng.integers(0, len(y), len(y))
        sampled_x = x[indices]
        sampled_y = y[indices]
        sampled_x = sampled_x - sampled_x.mean(axis=0, keepdims=True)
        sampled_y = sampled_y - sampled_y.mean()
        denominator = np.sqrt(
            np.sum(sampled_x**2, axis=0) * np.sum(sampled_y**2)
        )
        valid = denominator > 1e-12
        output[repeat, valid] = (
            sampled_x[:, valid].T @ sampled_y
        ) / denominator[valid]
    return output


def _correlation_clusters(
    frame: pd.DataFrame,
    features: list[str],
    threshold: float,
) -> dict[str, str]:
    """Deterministic connected components of absolute Spearman correlations."""

    if not features:
        return {}
    rank = frame[features].rank(method="average")
    correlation = rank.corr(method="pearson").abs().fillna(0.0)
    parent = {feature: feature for feature in features}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    for left_index, left in enumerate(features):
        for right in features[left_index + 1 :]:
            if float(correlation.loc[left, right]) >= threshold:
                union(left, right)
    groups: dict[str, list[str]] = {}
    for feature in features:
        groups.setdefault(find(feature), []).append(feature)
    ordered = sorted(
        (sorted(members) for members in groups.values()),
        key=lambda members: members[0],
    )
    return {
        feature: f"cluster_{index + 1:03d}"
        for index, members in enumerate(ordered)
        for feature in members
    }


def _family_sign_consistency(
    frame: pd.DataFrame,
    feature: str,
    global_r: float,
    minimum_group_size: int,
) -> tuple[float, int]:
    comparisons: list[bool] = []
    for role in ("cation_family", "anion_family"):
        for _, group in frame.groupby(role):
            if len(group) < minimum_group_size:
                continue
            rho, _ = _safe_corr(
                group[feature].to_numpy(dtype=float),
                group["condition_adjusted_log_response"].to_numpy(dtype=float),
                "pearson",
            )
            if np.isfinite(rho) and abs(rho) > 1e-12:
                comparisons.append(bool(np.sign(rho) == np.sign(global_r)))
    return (
        float(np.mean(comparisons)) if comparisons else np.nan,
        len(comparisons),
    )


@dataclass
class RevisionAnalysisResults:
    associations: pd.DataFrame
    record_weighted_associations: pd.DataFrame
    comparison: pd.DataFrame
    diagnostics: pd.DataFrame
    identity_responses: pd.DataFrame
    response_shapes: pd.DataFrame
    robust_factors: pd.DataFrame
    structural_themes: pd.DataFrame
    heat_capacity_size_control: pd.DataFrame
    heat_capacity_identity_data: pd.DataFrame
    data_support: pd.DataFrame


class IdentityBalancedStructurePropertyAnalyzer:
    """Primary IL-identity analysis plus record-weighted sensitivity analysis."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        stats_config = config["statistics"]
        revision = config.get("revision_analysis", {})
        self.bootstrap_repeats = int(
            revision.get(
                "identity_bootstrap_repeats",
                stats_config.get("bootstrap_repeats", 500),
            )
        )
        self.confidence = float(stats_config.get("confidence_level", 0.95))
        self.seed = int(stats_config.get("random_seed", 42))
        self.min_group_size = int(stats_config.get("min_group_size", 20))
        self.family_minimum = int(revision.get("family_minimum_unique_ils", 8))
        self.temperature_knots = int(revision.get("temperature_spline_knots", 4))
        self.cluster_threshold = float(
            revision.get("descriptor_cluster_abs_spearman", 0.85)
        )
        self.family_proxy_r2 = float(
            revision.get("family_proxy_r2_threshold", 0.98)
        )
        self.residual_std_ratio = float(
            revision.get("minimum_residual_std_ratio", 0.05)
        )
        self.main_top_k = int(revision.get("main_top_factors", 3))
        self.reference_pressure = float(
            config["conditions"].get("reference_pressure_kpa", 101.325)
        )

    def _property_analysis(
        self,
        bundle: FeatureBundle,
        property_name: str,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        records = bundle.records.copy()
        records["il_identity_key"] = identity_key(records)
        descriptors = bundle.descriptors.set_index("sample_id").loc[
            records["sample_id"]
        ].set_index(records.index)
        response_identity, valid, covariates, response_metadata = (
            aggregate_identity_responses(
                records,
                property_name,
                temperature_knots=self.temperature_knots,
                reference_pressure_kpa=self.reference_pressure,
            )
        )
        selected = records.loc[valid].copy()
        selected["il_identity_key"] = identity_key(selected)
        observed_values = np.log(
            pd.to_numeric(
                selected[f"{property_name}_ActualValue"],
                errors="coerce",
            ).to_numpy(dtype=float)
        )
        response_record_residual = _residualize(observed_values, covariates)
        raw_model_values = pd.to_numeric(
            selected[f"prediction_{property_name}"],
            errors="coerce",
        ).to_numpy(dtype=float)
        model_values = np.where(
            raw_model_values > 0,
            np.log(raw_model_values),
            np.nan,
        )
        model_record_residual = _residualize(model_values, covariates)

        identity_index = response_identity["il_identity_key"].tolist()
        identity_features = pd.DataFrame(index=identity_index)
        identity_model_response = (
            pd.DataFrame(
                {
                    "il_identity_key": selected["il_identity_key"].to_numpy(),
                    "model_response_residual": model_record_residual,
                }
            )
            .groupby("il_identity_key")["model_response_residual"]
            .median()
            .reindex(identity_index)
        )
        experimental_response = response_identity.set_index("il_identity_key")[
            "condition_adjusted_log_response"
        ].reindex(identity_index)

        rows: list[dict[str, Any]] = []
        record_rows: list[dict[str, Any]] = []
        diagnostic_rows: list[dict[str, Any]] = []
        feature_record_residuals: dict[str, np.ndarray] = {}
        for feature in ALL_FEATURES:
            feature_values = descriptors.loc[valid, feature].to_numpy(dtype=float)
            feature_residual = _residualize(feature_values, covariates)
            feature_record_residuals[feature] = feature_residual
            aggregated = (
                pd.DataFrame(
                    {
                        "il_identity_key": selected["il_identity_key"].to_numpy(),
                        "feature_residual": feature_residual,
                        "feature_raw": feature_values,
                    }
                )
                .groupby("il_identity_key")
                .median()
                .reindex(identity_index)
            )
            identity_features[feature] = aggregated["feature_residual"]
            partial_r, partial_p = _safe_corr(
                aggregated["feature_residual"].to_numpy(dtype=float),
                experimental_response.to_numpy(dtype=float),
                "pearson",
            )
            model_r, model_p = _safe_corr(
                aggregated["feature_residual"].to_numpy(dtype=float),
                identity_model_response.to_numpy(dtype=float),
                "pearson",
            )
            record_r, record_p = _safe_corr(
                feature_residual,
                response_record_residual,
                "pearson",
            )
            family_frame = response_identity.copy()
            family_frame[feature] = aggregated["feature_residual"].to_numpy()
            consistency, family_comparisons = _family_sign_consistency(
                family_frame,
                feature,
                partial_r,
                self.family_minimum,
            )
            diagnostics = _family_proxy_diagnostics(
                aggregated["feature_raw"].to_numpy(dtype=float),
                response_identity,
            )
            ineligible_proxy = bool(
                diagnostics["family_predictability_r2"] >= self.family_proxy_r2
                or diagnostics["family_residual_std_ratio"]
                < self.residual_std_ratio
            )
            finite_identity = (
                np.isfinite(aggregated["feature_residual"].to_numpy(dtype=float))
                & np.isfinite(experimental_response.to_numpy(dtype=float))
            )
            row = {
                "property": property_name,
                "feature": feature,
                "structural_scope": feature_scope(feature),
                "data_type": "experimental",
                "analysis_weighting": "identity_balanced",
                "n_records": int(valid.sum()),
                "n_unique_ils": int(finite_identity.sum()),
                "n_cation_families": int(
                    response_identity.loc[finite_identity, "cation_family"].nunique()
                ),
                "n_anion_families": int(
                    response_identity.loc[finite_identity, "anion_family"].nunique()
                ),
                "partial_correlation": partial_r,
                "partial_p": partial_p,
                "model_partial_correlation": model_r,
                "model_partial_p": model_p,
                "family_consistency": consistency,
                "family_comparison_count": family_comparisons,
                "bootstrap_ci_low": np.nan,
                "bootstrap_ci_high": np.nan,
                "selection_stability": 0.0,
                "eligibility_status": (
                    "ineligible_family_proxy"
                    if ineligible_proxy
                    else "eligible"
                ),
                "exclusion_reason": (
                    "family predictability or residual-variance threshold"
                    if ineligible_proxy
                    else ""
                ),
                "pressure_missing_records": response_metadata[
                    "pressure_missing_count"
                ],
                "pressure_availability_in_design": response_metadata[
                    "pressure_availability_in_design"
                ],
                "data_source_covariate": response_metadata[
                    "data_source_covariate"
                ],
                "response_scale": (
                    "condition-adjusted residual in natural-log property"
                ),
                "fdr_scope": (
                    "all experimental feature-property hypotheses"
                ),
                "causal_interpretation": False,
            }
            rows.append(row)
            record_rows.append(
                {
                    "property": property_name,
                    "feature": feature,
                    "structural_scope": feature_scope(feature),
                    "n_records": int(valid.sum()),
                    "n_unique_ils": int(finite_identity.sum()),
                    "record_weighted_partial_r": record_r,
                    "record_weighted_raw_p": record_p,
                }
            )
            diagnostic_rows.append(
                {
                    "property": property_name,
                    "feature": feature,
                    "structural_scope": feature_scope(feature),
                    **diagnostics,
                    "eligibility_status": row["eligibility_status"],
                    "exclusion_reason": row["exclusion_reason"],
                }
            )

        association = pd.DataFrame(rows)
        record_weighted = pd.DataFrame(record_rows)
        diagnostics = pd.DataFrame(diagnostic_rows)
        eligible_features = association.loc[
            association["eligibility_status"] == "eligible",
            "feature",
        ].tolist()
        cluster_map = _correlation_clusters(
            identity_features,
            eligible_features,
            self.cluster_threshold,
        )
        association["feature_cluster"] = association["feature"].map(
            cluster_map
        ).fillna("ineligible")
        diagnostics["feature_cluster"] = diagnostics["feature"].map(
            cluster_map
        ).fillna("ineligible")

        complete = identity_features.copy()
        complete["response"] = experimental_response.to_numpy(dtype=float)
        complete = complete.dropna(subset=["response"])
        x = complete[ALL_FEATURES].to_numpy(dtype=float)
        y = complete["response"].to_numpy(dtype=float)
        for column_index in range(x.shape[1]):
            finite = np.isfinite(x[:, column_index])
            fill = (
                float(np.median(x[finite, column_index]))
                if finite.any()
                else 0.0
            )
            x[~finite, column_index] = fill
        rng = np.random.default_rng(
            self.seed
            + 10000 * bundle.metadata["property_order"].index(property_name)
        )
        bootstrap = _bootstrap_correlations(
            x,
            y,
            self.bootstrap_repeats,
            rng,
        )
        alpha = (1.0 - self.confidence) / 2.0
        ci_low = np.nanquantile(bootstrap, alpha, axis=0)
        ci_high = np.nanquantile(bootstrap, 1.0 - alpha, axis=0)
        association["bootstrap_ci_low"] = ci_low
        association["bootstrap_ci_high"] = ci_high

        selected_counts = {feature: 0 for feature in ALL_FEATURES}
        for bootstrap_row in bootstrap:
            representatives: list[tuple[str, float]] = []
            for cluster in sorted(set(cluster_map.values())):
                members = [
                    feature
                    for feature in eligible_features
                    if cluster_map[feature] == cluster
                ]
                if not members:
                    continue
                representative = sorted(
                    members,
                    key=lambda feature: (
                        -abs(bootstrap_row[ALL_FEATURES.index(feature)])
                        if np.isfinite(
                            bootstrap_row[ALL_FEATURES.index(feature)]
                        )
                        else np.inf,
                        feature,
                    ),
                )[0]
                representatives.append(
                    (
                        representative,
                        abs(bootstrap_row[ALL_FEATURES.index(representative)]),
                    )
                )
            for feature, _ in sorted(
                representatives,
                key=lambda item: (-item[1], item[0]),
            )[: self.main_top_k]:
                selected_counts[feature] += 1
        association["selection_stability"] = association["feature"].map(
            {
                feature: selected_counts[feature] / self.bootstrap_repeats
                for feature in ALL_FEATURES
            }
        )
        association["unique_feature_values"] = association["feature"].map(
            {
                feature: int(
                    identity_features[feature].dropna().nunique()
                )
                for feature in ALL_FEATURES
            }
        )

        identity_long = response_identity.copy()
        identity_long["property"] = property_name
        identity_long["analysis_weighting"] = "identity_balanced"
        identity_long["response_definition"] = (
            "median record-level condition-adjusted log-property residual"
        )
        identity_feature_table = identity_features.reset_index().rename(
            columns={"index": "il_identity_key"}
        )
        identity_long = identity_long.merge(
            identity_feature_table,
            on="il_identity_key",
            how="left",
            validate="one_to_one",
        )
        return (
            association,
            record_weighted,
            diagnostics,
            identity_long,
            pd.DataFrame(
                {
                    "property": [property_name],
                    "n_records": [response_metadata["n_records"]],
                    "n_unique_ils": [response_metadata["n_unique_ils"]],
                    "pressure_missing_records": [
                        response_metadata["pressure_missing_count"]
                    ],
                    "pressure_availability_in_design": [
                        response_metadata["pressure_availability_in_design"]
                    ],
                    "condition_covariates": [
                        ";".join(response_metadata["covariates"])
                    ],
                    "data_source_covariate": [
                        response_metadata["data_source_covariate"]
                    ],
                }
            ),
        )

    def _select_nonredundant_top(
        self,
        association: pd.DataFrame,
    ) -> pd.DataFrame:
        eligible = association.loc[
            (association["eligibility_status"] == "eligible")
            & (association["unique_feature_values"] >= 5)
        ].copy()
        eligible["ci_excludes_zero"] = (
            (eligible["bootstrap_ci_low"] > 0)
            | (eligible["bootstrap_ci_high"] < 0)
        )
        eligible["statistical_support"] = (
            (eligible["fdr_q"] <= self.config["statistics"]["fdr_alpha"])
            & eligible["ci_excludes_zero"]
        )
        representatives = (
            eligible.sort_values(
                [
                    "property",
                    "feature_cluster",
                    "statistical_support",
                    "selection_stability",
                    "partial_correlation",
                    "feature",
                ],
                ascending=[True, True, False, False, False, True],
                key=lambda series: (
                    series.abs()
                    if series.name == "partial_correlation"
                    else series
                ),
            )
            .groupby(["property", "feature_cluster"], as_index=False)
            .head(1)
        )
        selected = (
            representatives.sort_values(
                [
                    "property",
                    "statistical_support",
                    "selection_stability",
                    "partial_correlation",
                    "feature",
                ],
                ascending=[True, False, False, False, True],
                key=lambda series: (
                    series.abs()
                    if series.name == "partial_correlation"
                    else series
                ),
            )
            .groupby("property", as_index=False)
            .head(self.main_top_k)
            .copy()
        )
        selected["selection_rank"] = (
            selected.groupby("property").cumcount() + 1
        )
        selected["main_figure_eligible"] = selected["statistical_support"]
        return selected

    def _response_shapes(
        self,
        identity_responses: pd.DataFrame,
        selected: pd.DataFrame,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for rule in selected.itertuples(index=False):
            source = identity_responses.loc[
                identity_responses["property"] == rule.property,
                [
                    "il_identity_key",
                    rule.feature,
                    "condition_adjusted_log_response",
                ],
            ].dropna()
            if len(source) < self.min_group_size or source[rule.feature].nunique() < 5:
                continue
            source["quantile_bin"] = pd.qcut(
                source[rule.feature],
                q=5,
                labels=False,
                duplicates="drop",
            )
            if source["quantile_bin"].nunique() < 5:
                continue
            medians = source.groupby("quantile_bin")[
                "condition_adjusted_log_response"
            ].median()
            monotonic_rho, _ = _safe_corr(
                np.arange(1, 6, dtype=float),
                medians.to_numpy(dtype=float),
                "spearman",
            )
            turning_points = int(
                np.sum(
                    np.diff(np.sign(np.diff(medians.to_numpy(dtype=float))))
                    != 0
                )
            )
            for bin_index, group in source.groupby("quantile_bin"):
                values = group[
                    "condition_adjusted_log_response"
                ].to_numpy(dtype=float)
                rng = np.random.default_rng(
                    self.seed
                    + 100000 * int(rule.selection_rank)
                    + 1000
                    * list(self.config["properties"]).index(
                        {
                            "Density": "density",
                            "ElectricalConductivity": "electrical_conductivity",
                            "HeatCapacity": "heat_capacity",
                            "SurfaceTension": "surface_tension",
                            "ThermalConductivity": "thermal_conductivity",
                            "Viscosity": "viscosity",
                        }[rule.property]
                    )
                    + int(bin_index)
                )
                bootstrap_medians = np.asarray(
                    [
                        np.median(
                            rng.choice(values, size=len(values), replace=True)
                        )
                        for _ in range(self.bootstrap_repeats)
                    ],
                    dtype=float,
                )
                alpha = (1.0 - self.confidence) / 2.0
                rows.append(
                    {
                        "property": rule.property,
                        "feature": rule.feature,
                        "feature_cluster": rule.feature_cluster,
                        "structural_scope": rule.structural_scope,
                        "selection_rank": int(rule.selection_rank),
                        "partial_r": float(rule.partial_correlation),
                        "fdr_q": float(rule.fdr_q),
                        "selection_stability": float(
                            rule.selection_stability
                        ),
                        "main_figure_eligible": bool(
                            rule.main_figure_eligible
                        ),
                        "quantile_bin": int(bin_index) + 1,
                        "quantile_label": f"Q{int(bin_index) + 1}",
                        "sample_count": int(len(group)),
                        "n_unique_ils": int(len(source)),
                        "feature_bin_median": float(
                            group[rule.feature].median()
                        ),
                        "response_log_mean": float(np.median(values)),
                        "bootstrap_ci_low": float(
                            np.quantile(bootstrap_medians, alpha)
                        ),
                        "bootstrap_ci_high": float(
                            np.quantile(
                                bootstrap_medians,
                                1.0 - alpha,
                            )
                        ),
                        "uncertainty_definition": (
                            "identity-level bootstrap 95% CI of bin median"
                        ),
                        "monotonic_bin_spearman": monotonic_rho,
                        "turning_point_count": turning_points,
                        "nonlinearity_flag": turning_points > 0,
                        "response_shape_support": bool(
                            np.isfinite(monotonic_rho)
                            and np.sign(monotonic_rho)
                            == np.sign(rule.partial_correlation)
                        ),
                    }
                )
        return pd.DataFrame(rows)

    def _heat_capacity_size_control(
        self,
        bundle: FeatureBundle,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        records = bundle.records.copy()
        records["il_identity_key"] = identity_key(records)
        descriptors = bundle.descriptors.set_index("sample_id").loc[
            records["sample_id"]
        ].set_index(records.index)
        molar_mass = (
            descriptors["pair_total_molecular_weight_scaled"].astype(float)
            * 1000.0
        )
        valid = (
            (records["HeatCapacity_mask"].to_numpy(dtype=float) > 0)
            & np.isfinite(records["HeatCapacity_ActualValue"])
            & (records["HeatCapacity_ActualValue"] > 0)
            & np.isfinite(molar_mass)
            & (molar_mass > 0)
        )
        records["HeatCapacityMassSpecific_ActualValue"] = np.nan
        records.loc[valid, "HeatCapacityMassSpecific_ActualValue"] = (
            records.loc[valid, "HeatCapacity_ActualValue"].to_numpy(dtype=float)
            / molar_mass.loc[valid].to_numpy(dtype=float)
        )
        records["HeatCapacityMassSpecific_mask"] = valid.astype(float)
        mass_specific, _, _, _ = aggregate_identity_responses(
            records,
            "HeatCapacityMassSpecific",
            response_column="HeatCapacityMassSpecific_ActualValue",
            temperature_knots=self.temperature_knots,
            reference_pressure_kpa=self.reference_pressure,
        )
        molar, _, _, _ = aggregate_identity_responses(
            records,
            "HeatCapacity",
            temperature_knots=self.temperature_knots,
            reference_pressure_kpa=self.reference_pressure,
        )
        extra = pd.DataFrame(
            {"log_molar_mass": np.log(molar_mass.clip(lower=1e-12))},
            index=records.index,
        )
        adjusted, _, _, _ = aggregate_identity_responses(
            records,
            "HeatCapacity",
            temperature_knots=self.temperature_knots,
            reference_pressure_kpa=self.reference_pressure,
            extra_response_covariates=extra,
        )
        mass_by_identity = (
            pd.DataFrame(
                {
                    "il_identity_key": records.loc[valid, "il_identity_key"],
                    "molar_mass_g_mol": molar_mass.loc[valid],
                }
            )
            .groupby("il_identity_key", as_index=False)
            .median()
        )
        identity_data = mass_by_identity.merge(
            molar[
                [
                    "il_identity_key",
                    "condition_adjusted_log_response",
                    "observed_log_response",
                ]
            ].rename(
                columns={
                    "condition_adjusted_log_response": (
                        "molar_condition_adjusted_log_response"
                    ),
                    "observed_log_response": "molar_observed_log_response",
                }
            ),
            on="il_identity_key",
            how="inner",
        ).merge(
            mass_specific[
                ["il_identity_key", "condition_adjusted_log_response"]
            ].rename(
                columns={
                    "condition_adjusted_log_response": (
                        "mass_specific_condition_adjusted_log_response"
                    )
                }
            ),
            on="il_identity_key",
            how="inner",
        ).merge(
            adjusted[
                ["il_identity_key", "condition_adjusted_log_response"]
            ].rename(
                columns={
                    "condition_adjusted_log_response": (
                        "molecular_weight_adjusted_log_response"
                    )
                }
            ),
            on="il_identity_key",
            how="inner",
        )
        rows: list[dict[str, Any]] = []
        x = np.log(
            identity_data["molar_mass_g_mol"].to_numpy(dtype=float)
        )
        for analysis_type, column in (
            (
                "molar_heat_capacity",
                "molar_condition_adjusted_log_response",
            ),
            (
                "mass_specific_heat_capacity",
                "mass_specific_condition_adjusted_log_response",
            ),
            (
                "molar_heat_capacity_molecular_weight_adjusted",
                "molecular_weight_adjusted_log_response",
            ),
        ):
            y = identity_data[column].to_numpy(dtype=float)
            correlation, p_value = _safe_corr(x, y, "pearson")
            rng = np.random.default_rng(
                self.seed + 700000 + len(rows)
            )
            bootstrap = []
            for _ in range(self.bootstrap_repeats):
                indices = rng.integers(0, len(y), len(y))
                value, _ = _safe_corr(x[indices], y[indices], "pearson")
                bootstrap.append(value)
            bootstrap = np.asarray(bootstrap, dtype=float)
            alpha = (1.0 - self.confidence) / 2.0
            rows.append(
                {
                    "analysis_type": analysis_type,
                    "feature": "log_molar_mass_g_mol",
                    "n_unique_ils": int(len(y)),
                    "partial_r": correlation,
                    "raw_p": p_value,
                    "bootstrap_ci_low": float(
                        np.nanquantile(bootstrap, alpha)
                    ),
                    "bootstrap_ci_high": float(
                        np.nanquantile(bootstrap, 1.0 - alpha)
                    ),
                    "response_definition": column,
                }
            )
        table = pd.DataFrame(rows)
        table["fdr_q"] = benjamini_hochberg(table["raw_p"])
        return table, identity_data

    def _data_support(self, bundle: FeatureBundle) -> pd.DataFrame:
        records = bundle.records.copy()
        rows: list[dict[str, Any]] = []
        for property_name in bundle.metadata["property_order"]:
            values = pd.to_numeric(
                records[f"{property_name}_ActualValue"],
                errors="coerce",
            )
            valid = (
                (records[f"{property_name}_mask"] > 0)
                & values.notna()
                & (values > 0)
            )
            rows.append(
                {
                    "property": property_name,
                    "split": records["data_split"].iloc[0],
                    "split_path": records["split_path"].iloc[0],
                    "checkpoint_type": records["checkpoint_type"].iloc[0],
                    "n_records": int(valid.sum()),
                    "n_unique_ils": int(
                        identity_key(records.loc[valid]).nunique()
                    ),
                    "n_unique_cations": int(
                        records.loc[valid, "cation_smiles"].nunique()
                    ),
                    "n_unique_anions": int(
                        records.loc[valid, "anion_smiles"].nunique()
                    ),
                    "pressure_missing_records": int(
                        records.loc[valid, "Pressure_kPa"].isna().sum()
                    ),
                    "label_source_filter": (
                        "original six-property workbook; no copied/interpolated "
                        "ValueSource columns and training augmentation disabled"
                    ),
                }
            )
        return pd.DataFrame(rows)

    def run(self, bundle: FeatureBundle) -> RevisionAnalysisResults:
        association_parts: list[pd.DataFrame] = []
        record_parts: list[pd.DataFrame] = []
        diagnostic_parts: list[pd.DataFrame] = []
        identity_parts: list[pd.DataFrame] = []
        support_parts: list[pd.DataFrame] = []
        for property_name in bundle.metadata["property_order"]:
            (
                association,
                record_weighted,
                diagnostics,
                identity_response,
                support,
            ) = self._property_analysis(bundle, property_name)
            association_parts.append(association)
            record_parts.append(record_weighted)
            diagnostic_parts.append(diagnostics)
            identity_parts.append(identity_response)
            support_parts.append(support)
        associations = pd.concat(association_parts, ignore_index=True)
        associations["fdr_q"] = benjamini_hochberg(
            associations["partial_p"]
        )
        associations["model_fdr_q"] = benjamini_hochberg(
            associations["model_partial_p"]
        )
        associations["effect_direction"] = np.where(
            associations["partial_correlation"] > 0,
            "positive",
            np.where(
                associations["partial_correlation"] < 0,
                "negative",
                "undetermined",
            ),
        )
        record_weighted = pd.concat(record_parts, ignore_index=True)
        record_weighted["record_weighted_fdr_q"] = benjamini_hochberg(
            record_weighted["record_weighted_raw_p"]
        )
        record_weighted = record_weighted.merge(
            associations[
                [
                    "property",
                    "feature",
                    "partial_correlation",
                    "fdr_q",
                    "feature_cluster",
                    "eligibility_status",
                ]
            ].rename(
                columns={
                    "partial_correlation": (
                        "partial_r_identity_balanced"
                    ),
                    "fdr_q": "identity_balanced_fdr_q",
                }
            ),
            on=["property", "feature"],
            how="left",
        )
        record_weighted["direction_agreement"] = (
            np.sign(record_weighted["record_weighted_partial_r"])
            == np.sign(record_weighted["partial_r_identity_balanced"])
        )
        comparison_rows: list[dict[str, Any]] = []
        for property_name, group in record_weighted.groupby("property"):
            eligible = group.loc[
                group["eligibility_status"] == "eligible"
            ].copy()
            rank_rho, _ = _safe_corr(
                eligible["record_weighted_partial_r"].abs().to_numpy(),
                eligible["partial_r_identity_balanced"].abs().to_numpy(),
                "spearman",
            )
            identity_top = set(
                eligible.assign(
                    abs_identity=eligible[
                        "partial_r_identity_balanced"
                    ].abs()
                ).nlargest(self.main_top_k, "abs_identity")["feature"]
            )
            record_top = set(
                eligible.assign(
                    abs_record=eligible[
                        "record_weighted_partial_r"
                    ].abs()
                ).nlargest(self.main_top_k, "abs_record")["feature"]
            )
            comparison_rows.append(
                {
                    "property": property_name,
                    "eligible_feature_count": int(len(eligible)),
                    "direction_agreement_rate": float(
                        eligible["direction_agreement"].mean()
                    ),
                    "absolute_rank_spearman": rank_rho,
                    "top_factor_overlap_count": int(
                        len(identity_top & record_top)
                    ),
                    "top_factor_overlap_fraction": float(
                        len(identity_top & record_top)
                        / max(len(identity_top | record_top), 1)
                    ),
                    "identity_balanced_top_features": ";".join(
                        sorted(identity_top)
                    ),
                    "record_weighted_top_features": ";".join(
                        sorted(record_top)
                    ),
                }
            )
        selected = self._select_nonredundant_top(associations)
        associations = associations.merge(
            selected[
                [
                    "property",
                    "feature",
                    "selection_rank",
                    "main_figure_eligible",
                    "statistical_support",
                ]
            ],
            on=["property", "feature"],
            how="left",
        )
        identity_responses = pd.concat(identity_parts, ignore_index=True)
        response_shapes = self._response_shapes(
            identity_responses,
            selected,
        )
        response_support = (
            response_shapes.groupby(["property", "feature"])
            ["response_shape_support"]
            .all()
            .rename("response_shape_support")
            .reset_index()
        )
        robust = associations.merge(
            response_support,
            on=["property", "feature"],
            how="left",
        )
        robust["response_shape_support"] = robust[
            "response_shape_support"
        ].fillna(False)
        robust["experimental_model_direction_agreement"] = (
            np.sign(robust["partial_correlation"])
            == np.sign(robust["model_partial_correlation"])
        )
        robust["statistical_support"] = (
            (robust["fdr_q"] <= self.config["statistics"]["fdr_alpha"])
            & (
                (robust["bootstrap_ci_low"] > 0)
                | (robust["bootstrap_ci_high"] < 0)
            )
            & (robust["eligibility_status"] == "eligible")
        )
        robust["robustness_level"] = np.select(
            [
                robust["statistical_support"]
                & robust["experimental_model_direction_agreement"]
                & (robust["family_consistency"].fillna(0.0) >= 0.75)
                & robust["response_shape_support"],
                robust["statistical_support"]
                & robust["experimental_model_direction_agreement"],
                robust["statistical_support"],
            ],
            ["strong", "moderate", "tentative"],
            default="unsupported",
        )
        themes = (
            robust.loc[robust["eligibility_status"] == "eligible"]
            .groupby(
                ["property", "feature_cluster"],
                as_index=False,
            )
            .agg(
                representative_feature=(
                    "feature",
                    lambda values: sorted(values)[0],
                ),
                member_count=("feature", "size"),
                member_features=(
                    "feature",
                    lambda values: ";".join(sorted(values)),
                ),
                maximum_absolute_partial_r=(
                    "partial_correlation",
                    lambda values: float(np.max(np.abs(values))),
                ),
            )
        )
        heat_table, heat_identity = self._heat_capacity_size_control(
            bundle
        )
        return RevisionAnalysisResults(
            associations=associations,
            record_weighted_associations=record_weighted,
            comparison=pd.DataFrame(comparison_rows),
            diagnostics=pd.concat(diagnostic_parts, ignore_index=True),
            identity_responses=identity_responses,
            response_shapes=response_shapes,
            robust_factors=robust,
            structural_themes=themes,
            heat_capacity_size_control=heat_table,
            heat_capacity_identity_data=heat_identity,
            data_support=self._data_support(bundle),
        )
