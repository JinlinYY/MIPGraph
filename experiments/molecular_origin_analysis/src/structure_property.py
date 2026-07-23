"""Condition-controlled and family-stratified structure-property analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

from .feature_extractor import (
    DESCRIPTOR_NAMES,
    FUNCTIONAL_GROUP_NAMES,
    FeatureBundle,
)


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    """Return BH-adjusted q-values while preserving missing entries and order."""

    output = pd.Series(np.nan, index=values.index, dtype=float)
    finite = values.notna() & np.isfinite(values)
    if not finite.any():
        return output
    p = values.loc[finite].to_numpy(dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    output.loc[finite] = restored
    return output


def _safe_corr(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
) -> tuple[float, float]:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 4 or np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
        return np.nan, np.nan
    result = stats.pearsonr(x, y) if method == "pearson" else stats.spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def _residualize(values: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values) & np.isfinite(covariates).all(axis=1)
    residuals = np.full(values.shape, np.nan, dtype=float)
    if finite.sum() <= covariates.shape[1] + 2:
        return residuals
    design = np.column_stack([np.ones(finite.sum()), covariates[finite]])
    coefficients, *_ = np.linalg.lstsq(design, values[finite], rcond=None)
    residuals[finite] = values[finite] - design @ coefficients
    return residuals


def _family_stratified_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    families: np.ndarray,
    repeats: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    finite = np.isfinite(x) & np.isfinite(y) & pd.notna(families)
    x = x[finite]
    y = y[finite]
    families = families[finite]
    if len(x) < 10:
        return np.nan, np.nan
    x = stats.rankdata(x)
    y = stats.rankdata(y)
    groups = [np.flatnonzero(families == family) for family in pd.unique(families)]
    estimates: list[float] = []
    for _ in range(repeats):
        sampled = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in groups if len(group)]
        )
        sampled_x = x[sampled]
        sampled_y = y[sampled]
        centered_x = sampled_x - sampled_x.mean()
        centered_y = sampled_y - sampled_y.mean()
        denominator = np.sqrt(
            np.dot(centered_x, centered_x) * np.dot(centered_y, centered_y)
        )
        if denominator > 1e-12:
            estimates.append(float(np.dot(centered_x, centered_y) / denominator))
    if not estimates:
        return np.nan, np.nan
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(estimates, alpha)),
        float(np.quantile(estimates, 1.0 - alpha)),
    )


def _feature_category(name: str) -> str:
    kind = "functional_group" if name in FUNCTIONAL_GROUP_NAMES else "molecular_descriptor"
    role = name.split("_", 1)[0]
    return f"{role}_{kind}"


@dataclass
class StructurePropertyResults:
    associations: pd.DataFrame
    partial_correlations: pd.DataFrame
    family_stratified: pd.DataFrame
    nonlinear_trends: pd.DataFrame
    robust_factors: pd.DataFrame


class StructurePropertyAnalyzer:
    """Separate observed-data associations from model-derived response trends."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        stats_config = config["statistics"]
        self.bootstrap_repeats = int(stats_config.get("bootstrap_repeats", 500))
        self.confidence = float(stats_config.get("confidence_level", 0.95))
        self.min_group_size = int(stats_config.get("min_group_size", 20))
        self.top_k = int(stats_config.get("top_features_per_property", 15))
        self.seed = int(stats_config.get("random_seed", 42))
        self.reference_temperatures = [
            float(value) for value in config["conditions"]["reference_temperatures_k"]
        ]
        self.temperature_tolerance = float(
            config["conditions"].get("temperature_tolerance_k", 2.0)
        )

    @staticmethod
    def _covariates(records: pd.DataFrame) -> np.ndarray:
        numeric = records[["Temperature_K", "Pressure_kPa"]].copy()
        numeric["Pressure_kPa"] = numeric["Pressure_kPa"].fillna(
            numeric["Pressure_kPa"].median()
        )
        families = pd.get_dummies(
            records[["cation_family", "anion_family"]].astype(str),
            drop_first=True,
            dtype=float,
        )
        values = np.column_stack([numeric.to_numpy(dtype=float), families.to_numpy()])
        means = np.nanmean(values, axis=0)
        stds = np.nanstd(values, axis=0)
        return (values - means) / np.where(stds > 1e-12, stds, 1.0)

    def _response(
        self,
        records: pd.DataFrame,
        property_name: str,
        data_type: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        mask = records[f"{property_name}_mask"].to_numpy(dtype=float) > 0
        column = (
            f"{property_name}_ActualValue"
            if data_type == "experimental"
            else f"prediction_{property_name}"
        )
        values = records[column].to_numpy(dtype=float)
        valid = mask & np.isfinite(values) & (values > 0)
        response = np.full(len(records), np.nan, dtype=float)
        response[valid] = np.log(values[valid])
        return response, valid

    def _all_condition_associations(
        self,
        bundle: FeatureBundle,
    ) -> pd.DataFrame:
        records = bundle.records
        descriptors = bundle.descriptors.set_index("sample_id").loc[
            records["sample_id"]
        ]
        covariates = self._covariates(records)
        rows: list[dict[str, Any]] = []
        for property_name in bundle.metadata["property_order"]:
            for data_type in ("experimental", "model_prediction"):
                response, valid = self._response(records, property_name, data_type)
                response_residual = _residualize(response, covariates)
                group_rows: list[dict[str, Any]] = []
                residual_cache: dict[str, np.ndarray] = {}
                for feature in DESCRIPTOR_NAMES + FUNCTIONAL_GROUP_NAMES:
                    values = descriptors[feature].to_numpy(dtype=float)
                    feature_residual = _residualize(values, covariates)
                    residual_cache[feature] = feature_residual
                    pearson_r, pearson_p = _safe_corr(
                        values[valid], response[valid], "pearson"
                    )
                    spearman_rho, spearman_p = _safe_corr(
                        values[valid], response[valid], "spearman"
                    )
                    partial_r, partial_p = _safe_corr(
                        feature_residual[valid],
                        response_residual[valid],
                        "pearson",
                    )
                    finite = valid & np.isfinite(values) & np.isfinite(response)
                    mutual_information = np.nan
                    if finite.sum() >= self.min_group_size and np.nanstd(values[finite]) > 1e-12:
                        mutual_information = float(
                            mutual_info_regression(
                                values[finite].reshape(-1, 1),
                                response[finite],
                                random_state=self.seed,
                            )[0]
                        )
                    group_rows.append(
                        {
                            "property": property_name,
                            "feature": feature,
                            "feature_category": _feature_category(feature),
                            "data_type": data_type,
                            "temperature": "all_condition_controlled",
                            "sample_count": int(valid.sum()),
                            "pearson_r": pearson_r,
                            "pearson_p": pearson_p,
                            "spearman_rho": spearman_rho,
                            "spearman_p": spearman_p,
                            "partial_correlation": partial_r,
                            "partial_p": partial_p,
                            "mutual_information": mutual_information,
                            "bootstrap_ci_low": np.nan,
                            "bootstrap_ci_high": np.nan,
                            "bootstrap_scope": "not_selected_for_top_k_bootstrap",
                            "response_scale": "natural_log_physical_unit",
                            "causal_interpretation": False,
                        }
                    )
                top_features = {
                    row["feature"]
                    for row in sorted(
                        group_rows,
                        key=lambda item: (
                            -abs(item["partial_correlation"])
                            if np.isfinite(item["partial_correlation"])
                            else np.inf
                        ),
                    )[: self.top_k]
                }
                rng = np.random.default_rng(
                    self.seed
                    + 1000 * bundle.metadata["property_order"].index(property_name)
                    + (1 if data_type == "model_prediction" else 0)
                )
                for row in group_rows:
                    if row["feature"] not in top_features:
                        continue
                    ci_low, ci_high = _family_stratified_bootstrap(
                        residual_cache[row["feature"]][valid],
                        response_residual[valid],
                        records.loc[valid, "cation_family"].astype(str).to_numpy(),
                        self.bootstrap_repeats,
                        self.confidence,
                        rng,
                    )
                    row["bootstrap_ci_low"] = ci_low
                    row["bootstrap_ci_high"] = ci_high
                    row["bootstrap_scope"] = (
                        f"family_stratified_top_{self.top_k}; "
                        f"{self.bootstrap_repeats}_replicates"
                    )
                rows.extend(group_rows)
        frame = pd.DataFrame(rows)
        frame["fdr_q"] = frame.groupby(
            ["property", "data_type"], group_keys=False
        )["partial_p"].apply(benjamini_hochberg)
        frame["effect_direction"] = np.where(
            frame["partial_correlation"] > 0,
            "positive",
            np.where(frame["partial_correlation"] < 0, "negative", "undetermined"),
        )
        return frame

    def _temperature_stratified(
        self,
        bundle: FeatureBundle,
        associations: pd.DataFrame,
    ) -> pd.DataFrame:
        records = bundle.records
        descriptors = bundle.descriptors.set_index("sample_id").loc[
            records["sample_id"]
        ]
        selected = (
            associations.sort_values(
                ["property", "data_type", "partial_correlation"],
                key=lambda series: series.abs()
                if series.name == "partial_correlation"
                else series,
                ascending=[True, True, False],
            )
            .groupby(["property", "data_type"])
            .head(self.top_k)
        )
        rows: list[dict[str, Any]] = []
        for item in selected.itertuples(index=False):
            response, valid_response = self._response(
                records,
                item.property,
                item.data_type,
            )
            for temperature in self.reference_temperatures:
                mask = (
                    valid_response
                    & (
                        np.abs(
                            records["Temperature_K"].to_numpy(dtype=float)
                            - temperature
                        )
                        <= self.temperature_tolerance
                    )
                )
                values = descriptors[item.feature].to_numpy(dtype=float)
                rho, p_value = _safe_corr(values[mask], response[mask], "spearman")
                rows.append(
                    {
                        "property": item.property,
                        "feature": item.feature,
                        "data_type": item.data_type,
                        "temperature_K": temperature,
                        "temperature_tolerance_K": self.temperature_tolerance,
                        "sample_count": int(mask.sum()),
                        "stratified_spearman_rho": rho,
                        "stratified_p": p_value,
                    }
                )
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame["fdr_q"] = frame.groupby(
                ["property", "data_type", "temperature_K"], group_keys=False
            )["stratified_p"].apply(benjamini_hochberg)
        return frame

    def _family_stratified(
        self,
        bundle: FeatureBundle,
        associations: pd.DataFrame,
    ) -> pd.DataFrame:
        records = bundle.records
        descriptors = bundle.descriptors.set_index("sample_id").loc[
            records["sample_id"]
        ]
        selected = (
            associations.loc[associations["data_type"] == "experimental"]
            .assign(abs_effect=lambda frame: frame["partial_correlation"].abs())
            .sort_values(["property", "abs_effect"], ascending=[True, False])
            .groupby("property")
            .head(self.top_k)
        )
        rows: list[dict[str, Any]] = []
        for item in selected.itertuples(index=False):
            response, valid_response = self._response(
                records,
                item.property,
                "experimental",
            )
            values = descriptors[item.feature].to_numpy(dtype=float)
            for family_role in ("cation_family", "anion_family"):
                counts = records.loc[valid_response, family_role].value_counts()
                for family, count in counts.items():
                    if count < self.min_group_size:
                        continue
                    family_values = records[family_role].to_numpy()
                    for analysis_scope, mask in (
                        (
                            "within_family",
                            valid_response & (family_values == family),
                        ),
                        (
                            "leave_one_family_out",
                            valid_response & (family_values != family),
                        ),
                    ):
                        if int(mask.sum()) < self.min_group_size:
                            continue
                        rho, p_value = _safe_corr(
                            values[mask],
                            response[mask],
                            "spearman",
                        )
                        rows.append(
                            {
                                "property": item.property,
                                "feature": item.feature,
                                "family_role": family_role,
                                "family": family,
                                "analysis_scope": analysis_scope,
                                "sample_count": int(mask.sum()),
                                "spearman_rho": rho,
                                "spearman_p": p_value,
                                "global_direction": item.effect_direction,
                                "direction_matches_global": (
                                    np.sign(rho)
                                    == np.sign(item.partial_correlation)
                                    if np.isfinite(rho)
                                    else False
                                ),
                            }
                        )
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame["fdr_q"] = frame.groupby(
                ["property", "family_role"], group_keys=False
            )["spearman_p"].apply(benjamini_hochberg)
        return frame

    def _nonlinear_trends(
        self,
        bundle: FeatureBundle,
        associations: pd.DataFrame,
    ) -> pd.DataFrame:
        records = bundle.records
        descriptors = bundle.descriptors.set_index("sample_id").loc[
            records["sample_id"]
        ]
        selected = (
            associations.loc[associations["data_type"] == "experimental"]
            .assign(abs_effect=lambda frame: frame["partial_correlation"].abs())
            .sort_values(["property", "abs_effect"], ascending=[True, False])
            .groupby("property")
            .head(min(5, self.top_k))
        )
        rows: list[dict[str, Any]] = []
        for item in selected.itertuples(index=False):
            response, valid = self._response(records, item.property, "experimental")
            values = descriptors[item.feature].to_numpy(dtype=float)
            finite = valid & np.isfinite(values) & np.isfinite(response)
            if finite.sum() < self.min_group_size or np.unique(values[finite]).size < 4:
                continue
            bins = pd.qcut(values[finite], q=min(10, np.unique(values[finite]).size), duplicates="drop")
            temp = pd.DataFrame(
                {
                    "feature_value": values[finite],
                    "response": response[finite],
                    "bin": bins,
                }
            )
            grouped = temp.groupby("bin", observed=True)
            means = grouped[["feature_value", "response"]].mean()
            turning_points = int(
                np.sum(
                    np.diff(np.sign(np.diff(means["response"].to_numpy(dtype=float))))
                    != 0
                )
            )
            monotonic_rho, _ = _safe_corr(
                means["feature_value"].to_numpy(dtype=float),
                means["response"].to_numpy(dtype=float),
                "spearman",
            )
            for bin_index, (_, group) in enumerate(grouped):
                rows.append(
                    {
                        "property": item.property,
                        "feature": item.feature,
                        "quantile_bin": bin_index + 1,
                        "sample_count": len(group),
                        "feature_mean": float(group["feature_value"].mean()),
                        "response_log_mean": float(group["response"].mean()),
                        "response_log_sem": float(group["response"].sem()),
                        "monotonic_bin_spearman": monotonic_rho,
                        "turning_point_count": turning_points,
                        "nonlinearity_flag": turning_points > 0,
                    }
                )
        return pd.DataFrame(rows)

    def _robust_factors(
        self,
        associations: pd.DataFrame,
        family: pd.DataFrame,
    ) -> pd.DataFrame:
        experimental = associations.loc[
            associations["data_type"] == "experimental"
        ].copy()
        predicted = associations.loc[
            associations["data_type"] == "model_prediction",
            ["property", "feature", "partial_correlation", "fdr_q"],
        ].rename(
            columns={
                "partial_correlation": "model_partial_correlation",
                "fdr_q": "model_fdr_q",
            }
        )
        merged = experimental.merge(predicted, on=["property", "feature"], how="left")
        family_summary = pd.DataFrame(
            columns=["property", "feature", "family_consistency"]
        )
        if not family.empty:
            family_summary = (
                family.groupby(["property", "feature"])["direction_matches_global"]
                .mean()
                .rename("family_consistency")
                .reset_index()
            )
        merged = merged.merge(family_summary, on=["property", "feature"], how="left")
        merged["family_consistency"] = merged["family_consistency"].fillna(0.0)
        merged["experimental_model_direction_agreement"] = (
            np.sign(merged["partial_correlation"])
            == np.sign(merged["model_partial_correlation"])
        )
        significant = merged["fdr_q"] <= float(
            self.config["statistics"].get("fdr_alpha", 0.05)
        )
        model_supported = merged["model_fdr_q"] <= float(
            self.config["statistics"].get("fdr_alpha", 0.05)
        )
        ci_excludes_zero = (
            (merged["bootstrap_ci_low"] > 0)
            | (merged["bootstrap_ci_high"] < 0)
        )
        strong = (
            significant
            & model_supported
            & ci_excludes_zero
            & merged["experimental_model_direction_agreement"]
            & (merged["family_consistency"] >= 0.75)
        )
        moderate = (
            significant
            & merged["experimental_model_direction_agreement"]
            & ci_excludes_zero
        )
        tentative = significant | model_supported
        merged["robustness_level"] = np.select(
            [strong, moderate, tentative],
            ["strong", "moderate", "tentative"],
            default="unsupported",
        )
        return merged.sort_values(
            ["property", "robustness_level", "partial_correlation"],
            ascending=[True, True, False],
            key=lambda series: series.abs()
            if series.name == "partial_correlation"
            else series,
        )

    def run(self, bundle: FeatureBundle) -> StructurePropertyResults:
        associations = self._all_condition_associations(bundle)
        partial = self._temperature_stratified(bundle, associations)
        family = self._family_stratified(bundle, associations)
        nonlinear = self._nonlinear_trends(bundle, associations)
        robust = self._robust_factors(associations, family)
        return StructurePropertyResults(
            associations=associations,
            partial_correlations=partial,
            family_stratified=family,
            nonlinear_trends=nonlinear,
            robust_factors=robust,
        )
