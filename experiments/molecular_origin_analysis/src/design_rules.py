"""Evidence-gated design rules and unified Top-8 structural interpretation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem

from il_property_prediction.src.chem.functional_groups import (
    ion_pair_functional_group_descriptors,
)
from il_property_prediction.src.chem.global_descriptors import ion_pair_descriptors

from .feature_extractor import DESCRIPTOR_NAMES, FUNCTIONAL_GROUP_NAMES


@dataclass
class DesignRuleResults:
    design_rules: pd.DataFrame
    unsupported_hypotheses: pd.DataFrame
    evidence_table: pd.DataFrame
    property_design_rules: dict[str, Any]
    candidate_profiles: pd.DataFrame
    top8_vs_nonfeasible: pd.DataFrame
    candidate_rule_consistency: pd.DataFrame


def _screening_implication(property_name: str, direction: str) -> str:
    desired = {
        "Viscosity": "negative",
        "ElectricalConductivity": "positive",
        "HeatCapacity": "positive",
        "ThermalConductivity": "positive",
    }
    if property_name == "SurfaceTension":
        return "No monotonic preference; qualify against a reference envelope."
    if property_name == "Density":
        return "Interpret jointly with heat capacity for volumetric heat storage."
    if desired.get(property_name) == direction:
        return "Direction is aligned with the thermophysical pre-screening objective."
    return "Direction is a potential trade-off against the thermophysical pre-screening objective."


class DesignRuleSynthesizer:
    """Promote a trend only when independent evidence gates are satisfied."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.fdr_alpha = float(config["statistics"].get("fdr_alpha", 0.05))
        self.top_k = int(config["statistics"].get("top_features_per_property", 15))

    def _rules(
        self,
        robust: pd.DataFrame,
        importance: pd.DataFrame,
        counterfactual_summary: pd.DataFrame | None,
        checkpoint_count: int,
        record_weighted: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        direct = importance.loc[
            importance["method"] == "direct_gradient_x_input"
        ].copy()
        direct["attribution_rank"] = direct.groupby("property")[
            "normalized_importance"
        ].rank(ascending=False, method="min")
        attribution = direct[
            ["property", "feature", "attribution_rank", "normalized_importance"]
        ]
        merged = robust.merge(attribution, on=["property", "feature"], how="left")
        merged["attribution_support"] = merged["attribution_rank"] <= self.top_k
        merged["statistical_support"] = (
            (merged["fdr_q"] <= self.fdr_alpha)
            & (
                (merged["bootstrap_ci_low"] > 0)
                | (merged["bootstrap_ci_high"] < 0)
            )
        )
        merged["model_response_support"] = (
            merged["experimental_model_direction_agreement"]
            & (merged["model_fdr_q"] <= self.fdr_alpha)
        )
        supported_modifications = set()
        if counterfactual_summary is not None and not counterfactual_summary.empty:
            supported_modifications = set(
                counterfactual_summary.loc[
                    counterfactual_summary["temperature_direction_consistent"] == True,  # noqa: E712
                    "linked_feature",
                ].dropna()
            )
        merged["counterfactual_support"] = merged["feature"].isin(
            supported_modifications
        )
        merged["cross_checkpoint_support"] = False
        merged["cross_checkpoint_status"] = (
            "not_assessable_single_compatible_checkpoint"
            if checkpoint_count < 2
            else "not_evaluated_on_an_identical_identity_set"
        )
        merged["checkpoint_consistency"] = False
        merged["confidence_level"] = "Unsupported"
        level_c = merged["statistical_support"]
        level_b = (
            level_c
            & merged["model_response_support"]
            & merged["attribution_support"]
            & merged["response_shape_support"].fillna(False)
        )
        merged.loc[level_c, "confidence_level"] = "Level C"
        merged.loc[level_b, "confidence_level"] = "Level B"
        merged["ion_role"] = merged["feature"].str.split("_", n=1).str[0]
        merged["effect_direction"] = np.where(
            merged["partial_correlation"] > 0,
            "positive",
            "negative",
        )
        merged["structural_factor"] = merged["feature"]
        merged["statistical_evidence"] = merged.apply(
            lambda row: (
                f"partial r={row.partial_correlation:.3f}; q={row.fdr_q:.3g}; "
                "IL-identity bootstrap "
                f"CI=[{row.bootstrap_ci_low:.3f}, {row.bootstrap_ci_high:.3f}]"
            ),
            axis=1,
        )
        merged["attribution_evidence"] = merged.apply(
            lambda row: (
                f"direct gradient×input rank={row.attribution_rank:.0f}"
                if np.isfinite(row.attribution_rank)
                else "not supported by direct gradient×input top features"
            ),
            axis=1,
        )
        merged["counterfactual_evidence"] = np.where(
            merged["counterfactual_support"],
            "supported by a linked valid-SMILES counterfactual trend",
            "no feature-specific counterfactual confirmation",
        )
        merged["chemical_interpretation"] = (
            "Model-derived association requiring chemical and experimental qualification; "
            "no causal claim is made."
        )
        merged["screening_implication"] = [
            _screening_implication(prop, direction)
            for prop, direction in zip(
                merged["property"],
                merged["effect_direction"],
            )
        ]
        merged["tradeoff_properties"] = ""
        merged["partial_r_identity_balanced"] = merged["partial_correlation"]
        merged["partial_r_ci_low"] = merged["bootstrap_ci_low"]
        merged["partial_r_ci_high"] = merged["bootstrap_ci_high"]
        if record_weighted is not None and not record_weighted.empty:
            record_columns = record_weighted[
                ["property", "feature", "record_weighted_partial_r"]
            ].drop_duplicates(["property", "feature"])
            merged = merged.merge(
                record_columns,
                on=["property", "feature"],
                how="left",
                validate="one_to_one",
            )
        else:
            merged["record_weighted_partial_r"] = np.nan
        merged["evidence_eligibility"] = np.where(
            merged["eligibility_status"].eq("eligible")
            & merged["statistical_support"],
            "eligible",
            "excluded",
        )
        merged["evidence_exclusion_reason"] = np.where(
            merged["eligibility_status"].ne("eligible"),
            merged["exclusion_reason"],
            np.where(
                ~merged["statistical_support"],
                "global BH-FDR and/or identity-bootstrap gate not passed",
                "",
            ),
        )
        columns = [
            "property",
            "structural_factor",
            "ion_role",
            "effect_direction",
            "family_consistency",
            "statistical_evidence",
            "attribution_evidence",
            "counterfactual_evidence",
            "checkpoint_consistency",
            "chemical_interpretation",
            "confidence_level",
            "screening_implication",
            "tradeoff_properties",
        ]
        formal = merged.loc[
            merged["confidence_level"] != "Unsupported",
            columns,
        ].copy()
        unsupported = merged.loc[
            merged["confidence_level"] == "Unsupported",
            columns
            + [
                "statistical_support",
                "model_response_support",
                "attribution_support",
                "counterfactual_support",
            ],
        ].copy()
        evidence_columns = [
            "property",
            "feature",
            "feature_cluster",
            "structural_scope",
            "n_records",
            "n_unique_ils",
            "n_cation_families",
            "n_anion_families",
            "partial_r_identity_balanced",
            "partial_r_ci_low",
            "partial_r_ci_high",
            "fdr_q",
            "family_consistency",
            "family_comparison_count",
            "selection_stability",
            "record_weighted_partial_r",
            "attribution_rank",
            "normalized_importance",
            "attribution_support",
            "response_shape_support",
            "model_partial_correlation",
            "model_fdr_q",
            "model_response_support",
            "cross_checkpoint_support",
            "cross_checkpoint_status",
            "confidence_level",
            "evidence_eligibility",
            "evidence_exclusion_reason",
            "analysis_weighting",
            "fdr_scope",
            "data_source_covariate",
            "causal_interpretation",
        ]
        evidence = merged[evidence_columns].copy()
        return formal, unsupported, evidence

    @staticmethod
    def _candidate_profiles(trajectory: pd.DataFrame, top8: pd.DataFrame) -> pd.DataFrame:
        top_ids = set(top8["candidate_id"].astype(str))
        rows: list[dict[str, Any]] = []
        for source in trajectory.to_dict("records"):
            cation = Chem.MolFromSmiles(str(source["cation_smiles"]))
            anion = Chem.MolFromSmiles(str(source["anion_smiles"]))
            if cation is None or anion is None:
                continue
            values = ion_pair_descriptors(cation, anion, None, None)
            functional = ion_pair_functional_group_descriptors(cation, anion)
            rows.append(
                {
                    "candidate_id": source["candidate_id"],
                    "cation_smiles": source["cation_smiles"],
                    "anion_smiles": source["anion_smiles"],
                    "formal_shortlist": str(source["candidate_id"]) in top_ids,
                    "hard_feasible": bool(source.get("final_feasible", False)),
                    "AD_status": source.get("AD_status"),
                    "feature_scope": (
                        "2D identity/functional descriptors; 3D shape entries are zero "
                        "because no candidate conformer is reused here"
                    ),
                    **dict(zip(DESCRIPTOR_NAMES, values)),
                    **dict(zip(FUNCTIONAL_GROUP_NAMES, functional)),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _top8_comparison(profiles: pd.DataFrame) -> pd.DataFrame:
        shortlist = profiles["formal_shortlist"]
        nonfeasible = ~profiles["hard_feasible"]
        rows: list[dict[str, Any]] = []
        for feature in DESCRIPTOR_NAMES + FUNCTIONAL_GROUP_NAMES:
            left = profiles.loc[shortlist, feature].to_numpy(dtype=float)
            right = profiles.loc[nonfeasible, feature].to_numpy(dtype=float)
            pooled = np.sqrt(
                (
                    max(len(left) - 1, 0) * np.nanvar(left, ddof=1)
                    + max(len(right) - 1, 0) * np.nanvar(right, ddof=1)
                )
                / max(len(left) + len(right) - 2, 1)
            )
            rows.append(
                {
                    "feature": feature,
                    "top8_count": len(left),
                    "nonfeasible_count": len(right),
                    "top8_mean": float(np.nanmean(left)),
                    "nonfeasible_mean": float(np.nanmean(right)),
                    "standardized_mean_difference": (
                        float((np.nanmean(left) - np.nanmean(right)) / pooled)
                        if pooled > 1e-12
                        else np.nan
                    ),
                    "interpretation_scope": "descriptive structural contrast; no selection causality",
                }
            )
        return pd.DataFrame(rows).sort_values(
            "standardized_mean_difference",
            key=lambda series: series.abs(),
            ascending=False,
        )

    @staticmethod
    def _candidate_rule_consistency(
        profiles: pd.DataFrame,
        rules: pd.DataFrame,
    ) -> pd.DataFrame:
        shortlist = profiles.loc[profiles["formal_shortlist"]].copy()
        rows: list[dict[str, Any]] = []
        for rule in rules.to_dict("records"):
            feature = rule["structural_factor"]
            if feature not in profiles:
                continue
            mean = float(profiles[feature].mean())
            std = float(profiles[feature].std(ddof=0))
            for candidate in shortlist.to_dict("records"):
                z_score = (
                    (float(candidate[feature]) - mean) / std if std > 1e-12 else 0.0
                )
                expected_sign = 1 if rule["effect_direction"] == "positive" else -1
                rows.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "property": rule["property"],
                        "structural_factor": feature,
                        "rule_confidence": rule["confidence_level"],
                        "candidate_standardized_feature": z_score,
                        "rule_consistent": np.sign(z_score) == expected_sign,
                        "interpretation_scope": (
                            "post hoc consistency with a precomputed rule; not a reason for selection"
                        ),
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _json_rules(rules: pd.DataFrame) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for property_name, group in rules.groupby("property"):
            output[property_name] = {
                "increase_factors": group.loc[
                    group["effect_direction"] == "positive"
                ].to_dict("records"),
                "decrease_factors": group.loc[
                    group["effect_direction"] == "negative"
                ].to_dict("records"),
                "scope": "structure-property association and frozen-model sensitivity",
                "causal_claim": False,
            }
        return json.loads(json.dumps(output, default=str))

    def run(
        self,
        robust_factors: pd.DataFrame,
        property_importance: pd.DataFrame,
        screening_tables: dict[str, pd.DataFrame],
        counterfactual_summary: pd.DataFrame | None = None,
        checkpoint_count: int = 1,
        record_weighted: pd.DataFrame | None = None,
    ) -> DesignRuleResults:
        rules, unsupported, evidence = self._rules(
            robust_factors,
            property_importance,
            counterfactual_summary,
            checkpoint_count,
            record_weighted,
        )
        trajectory = screening_tables.get("candidate_trajectory_608", pd.DataFrame())
        top8 = screening_tables.get("top8", pd.DataFrame())
        if trajectory.empty or top8.empty:
            profiles = pd.DataFrame()
            comparison = pd.DataFrame()
            consistency = pd.DataFrame()
        else:
            profiles = self._candidate_profiles(trajectory, top8)
            comparison = self._top8_comparison(profiles)
            consistency = self._candidate_rule_consistency(profiles, rules)
        return DesignRuleResults(
            design_rules=rules,
            unsupported_hypotheses=unsupported,
            evidence_table=evidence,
            property_design_rules=self._json_rules(rules),
            candidate_profiles=profiles,
            top8_vs_nonfeasible=comparison,
            candidate_rule_consistency=consistency,
        )
