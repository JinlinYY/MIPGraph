"""Direct gradient sensitivity and audited descriptor-surrogate attribution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from .data_adapter import AnalysisData
from .feature_extractor import (
    DESCRIPTOR_NAMES,
    FUNCTIONAL_GROUP_NAMES,
    FeatureBundle,
)
from .model_adapter import ModelAdapter


@dataclass
class AttributionResults:
    property_feature_importance: pd.DataFrame
    grouped_feature_importance: pd.DataFrame
    module_level_importance: pd.DataFrame
    temperature_conditioned_importance: pd.DataFrame
    method_agreement: pd.DataFrame
    metadata: dict[str, Any]


def _module_name(feature: str) -> str:
    ion_role = feature.split("_", 1)[0]
    source = "functional_group" if feature in FUNCTIONAL_GROUP_NAMES else "global_descriptor"
    return f"{ion_role}_{source}"


class AttributionAnalyzer:
    """Keep direct MIPGraph sensitivities distinct from surrogate explanations."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        attribution = config["attribution"]
        self.repeats = int(attribution.get("permutation_repeats", 20))
        self.max_samples = int(attribution.get("maximum_samples", 512))
        self.seed = int(config["model"].get("seed", 42))
        self.reference_temperatures = [
            float(value) for value in config["conditions"]["reference_temperatures_k"]
        ]
        self.temperature_tolerance = float(
            config["conditions"].get("temperature_tolerance_k", 2.0)
        )

    def direct_gradient_x_input(
        self,
        adapter: ModelAdapter,
        data: AnalysisData,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        loader = adapter.make_loader(data)
        feature_names = DESCRIPTOR_NAMES + FUNCTIONAL_GROUP_NAMES
        totals = {
            property_name: np.zeros(len(feature_names), dtype=np.float64)
            for property_name in adapter.property_names
        }
        counts = {property_name: 0 for property_name in adapter.property_names}
        temperature_totals: dict[tuple[str, float], np.ndarray] = {}
        temperature_counts: dict[tuple[str, float], int] = {}
        adapter.model.eval()
        for batch in loader:
            batch = batch.to(adapter.device)
            batch.global_desc = (
                batch.global_desc.detach().clone().requires_grad_(True)
            )
            batch.functional_group_desc = (
                batch.functional_group_desc.detach().clone().requires_grad_(True)
            )
            scaled, _ = adapter.model(batch)
            raw_temperature = batch.raw_condition.view(-1, 2)[:, 0].detach()
            for property_index, property_name in enumerate(adapter.property_names):
                global_gradient, fg_gradient = torch.autograd.grad(
                    scaled[:, property_index].sum(),
                    [batch.global_desc, batch.functional_group_desc],
                    retain_graph=property_index < len(adapter.property_names) - 1,
                    allow_unused=False,
                )
                global_score = (
                    global_gradient
                    * batch.global_desc
                ).abs().view(len(raw_temperature), -1)
                fg_score = (
                    fg_gradient
                    * batch.functional_group_desc
                ).abs().view(len(raw_temperature), -1)
                score = torch.cat([global_score, fg_score], dim=1).detach().cpu().numpy()
                totals[property_name] += score.sum(axis=0)
                counts[property_name] += len(score)
                raw_np = raw_temperature.cpu().numpy()
                for temperature in self.reference_temperatures:
                    selected = np.abs(raw_np - temperature) <= self.temperature_tolerance
                    if not selected.any():
                        continue
                    key = (property_name, temperature)
                    temperature_totals.setdefault(
                        key,
                        np.zeros(len(feature_names), dtype=np.float64),
                    )
                    temperature_totals[key] += score[selected].sum(axis=0)
                    temperature_counts[key] = temperature_counts.get(key, 0) + int(
                        selected.sum()
                    )
        rows: list[dict[str, Any]] = []
        temperature_rows: list[dict[str, Any]] = []
        for property_name, values in totals.items():
            mean_values = values / max(counts[property_name], 1)
            normalized = mean_values / max(mean_values.sum(), 1e-12)
            for feature_index, feature in enumerate(feature_names):
                rows.append(
                    {
                        "property": property_name,
                        "feature": feature,
                        "method": "direct_gradient_x_input",
                        "importance": float(mean_values[feature_index]),
                        "normalized_importance": float(normalized[feature_index]),
                        "sample_count": counts[property_name],
                        "target_space": "checkpoint_standardized_log_property",
                        "interpretation_scope": "MIPGraph local model sensitivity; not causal",
                    }
                )
        for (property_name, temperature), values in temperature_totals.items():
            count = temperature_counts[(property_name, temperature)]
            mean_values = values / max(count, 1)
            normalized = mean_values / max(mean_values.sum(), 1e-12)
            for feature_index, feature in enumerate(feature_names):
                temperature_rows.append(
                    {
                        "property": property_name,
                        "feature": feature,
                        "temperature_K": temperature,
                        "temperature_tolerance_K": self.temperature_tolerance,
                        "sample_count": count,
                        "normalized_importance": float(normalized[feature_index]),
                        "method": "direct_gradient_x_input",
                    }
                )
        return pd.DataFrame(rows), pd.DataFrame(temperature_rows)

    def surrogate_permutation(
        self,
        bundle: FeatureBundle,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
        feature_names = DESCRIPTOR_NAMES + FUNCTIONAL_GROUP_NAMES
        x = bundle.descriptors[feature_names].to_numpy(dtype=np.float32)
        if len(x) > self.max_samples:
            rng = np.random.default_rng(self.seed)
            selected = np.sort(rng.choice(len(x), self.max_samples, replace=False))
            x = x[selected]
            records = bundle.records.iloc[selected].reset_index(drop=True)
        else:
            records = bundle.records.reset_index(drop=True)
        individual_rows: list[dict[str, Any]] = []
        grouped_rows: list[dict[str, Any]] = []
        fidelities: dict[str, float] = {}
        group_map = {
            module: [
                index
                for index, feature in enumerate(feature_names)
                if _module_name(feature) == module
            ]
            for module in sorted({_module_name(feature) for feature in feature_names})
        }
        for property_name in bundle.metadata["property_order"]:
            y = np.log(
                records[f"prediction_{property_name}"].to_numpy(dtype=float).clip(
                    min=1e-12
                )
            )
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=0.3,
                random_state=self.seed,
            )
            surrogate = ExtraTreesRegressor(
                n_estimators=200,
                min_samples_leaf=3,
                random_state=self.seed,
                n_jobs=1,
            ).fit(x_train, y_train)
            baseline_prediction = surrogate.predict(x_test)
            baseline = float(r2_score(y_test, baseline_prediction))
            fidelities[property_name] = baseline
            permutation = permutation_importance(
                surrogate,
                x_test,
                y_test,
                scoring="r2",
                n_repeats=self.repeats,
                random_state=self.seed,
                n_jobs=1,
            )
            nonnegative = np.clip(permutation.importances_mean, 0.0, None)
            normalized = nonnegative / max(nonnegative.sum(), 1e-12)
            for index, feature in enumerate(feature_names):
                individual_rows.append(
                    {
                        "property": property_name,
                        "feature": feature,
                        "method": "descriptor_surrogate_permutation",
                        "importance": float(permutation.importances_mean[index]),
                        "importance_std": float(permutation.importances_std[index]),
                        "normalized_importance": float(normalized[index]),
                        "sample_count": len(x),
                        "surrogate_test_r2": baseline,
                        "interpretation_scope": (
                            "ExtraTrees approximation to MIPGraph response; not direct model attribution"
                        ),
                    }
                )
            rng = np.random.default_rng(self.seed)
            for group_name, indices in group_map.items():
                drops = []
                for _ in range(self.repeats):
                    perturbed = x_test.copy()
                    order = rng.permutation(len(perturbed))
                    perturbed[:, indices] = perturbed[order][:, indices]
                    drops.append(
                        baseline
                        - float(r2_score(y_test, surrogate.predict(perturbed)))
                    )
                grouped_rows.append(
                    {
                        "property": property_name,
                        "feature_group": group_name,
                        "method": "grouped_descriptor_surrogate_permutation",
                        "importance": float(np.mean(drops)),
                        "importance_std": float(np.std(drops, ddof=1)),
                        "surrogate_test_r2": baseline,
                        "sample_count": len(x),
                    }
                )
        return pd.DataFrame(individual_rows), pd.DataFrame(grouped_rows), fidelities

    @staticmethod
    def _agreement(importance: pd.DataFrame) -> pd.DataFrame:
        direct = importance.loc[
            importance["method"] == "direct_gradient_x_input"
        ]
        surrogate = importance.loc[
            importance["method"] == "descriptor_surrogate_permutation"
        ]
        rows: list[dict[str, Any]] = []
        for property_name in sorted(set(direct["property"]) & set(surrogate["property"])):
            merged = direct.loc[
                direct["property"] == property_name,
                ["feature", "normalized_importance"],
            ].merge(
                surrogate.loc[
                    surrogate["property"] == property_name,
                    ["feature", "normalized_importance"],
                ],
                on="feature",
                suffixes=("_direct", "_surrogate"),
            )
            rho = stats.spearmanr(
                merged["normalized_importance_direct"],
                merged["normalized_importance_surrogate"],
            ).statistic
            top_direct = set(
                merged.nlargest(10, "normalized_importance_direct")["feature"]
            )
            top_surrogate = set(
                merged.nlargest(10, "normalized_importance_surrogate")["feature"]
            )
            rows.append(
                {
                    "property": property_name,
                    "method_a": "direct_gradient_x_input",
                    "method_b": "descriptor_surrogate_permutation",
                    "spearman_rank_correlation": float(rho),
                    "top10_overlap_count": len(top_direct & top_surrogate),
                    "top10_jaccard": len(top_direct & top_surrogate)
                    / max(len(top_direct | top_surrogate), 1),
                }
            )
        return pd.DataFrame(rows)

    def run(
        self,
        bundle: FeatureBundle,
        adapter: ModelAdapter,
        attribution_data: AnalysisData,
    ) -> AttributionResults:
        direct, temperature = self.direct_gradient_x_input(adapter, attribution_data)
        surrogate, grouped, fidelities = self.surrogate_permutation(bundle)
        combined = pd.concat([direct, surrogate], ignore_index=True)
        module = (
            direct.assign(module=direct["feature"].map(_module_name))
            .groupby(["property", "module"], as_index=False)
            .agg(
                importance=("importance", "sum"),
                normalized_importance=("normalized_importance", "sum"),
                sample_count=("sample_count", "max"),
            )
        )
        module["method"] = "direct_gradient_x_input_aggregated"
        return AttributionResults(
            property_feature_importance=combined,
            grouped_feature_importance=grouped,
            module_level_importance=module,
            temperature_conditioned_importance=temperature,
            method_agreement=self._agreement(combined),
            metadata={
                "direct_method": "absolute gradient-times-input on frozen MIPGraph",
                "surrogate_method": (
                    "ExtraTrees fitted only to approximate frozen MIPGraph outputs for "
                    "descriptor-level permutation analysis"
                ),
                "surrogate_fidelity_r2": fidelities,
                "shap_status": "not_run_optional_third_priority",
                "integrated_gradients_status": "not_run_optional_third_priority",
                "causal_interpretation": False,
            },
        )
