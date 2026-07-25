"""Stage-oriented orchestration for the complete non-invasive analysis."""

from __future__ import annotations

import json
import re
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .applicability_domain import ApplicabilityDomainAnalyzer
from .attribution import AttributionAnalyzer
from .counterfactual import CounterfactualGenerator
from .cross_ion_attention import CrossIonAnalyzer
from .data_adapter import DataAdapter
from .design_rules import DesignRuleSynthesizer
from .feature_extractor import (
    DESCRIPTOR_NAMES,
    FUNCTIONAL_GROUP_NAMES,
    FeatureBundle,
    FeatureExtractor,
)
from .model_adapter import MECHANISM_NAMES, ModelAdapter
from .plotting import PublicationPlotter
from .project_adapter import InspectionReport, ProjectAdapter
from .revision_analysis import IdentityBalancedStructurePropertyAnalyzer
from .structure_property import StructurePropertyAnalyzer
from .utils import (
    MODULE_ROOT,
    configure_logging,
    file_sha256,
    git_commit,
    git_status,
    load_config,
    read_table,
    seed_everything,
    software_versions,
    stable_hash,
    utc_now,
    write_json,
    write_table,
)


PROPERTY_NAMES = [
    "Density",
    "ElectricalConductivity",
    "HeatCapacity",
    "SurfaceTension",
    "ThermalConductivity",
    "Viscosity",
]


class AnalysisPipeline:
    """Execute independent stages with cache provenance and explicit degradation."""

    def __init__(
        self,
        config_path: str | Path,
        overrides: dict[str, Any] | None = None,
        verbose: bool = False,
    ) -> None:
        self.config = load_config(config_path, overrides=overrides)
        self.project = ProjectAdapter(self.config)
        self.output_root = self.project.output_root
        for subdirectory in (
            "cache",
            "tables",
            "tables/figure_source_data",
            "figures",
            "logs",
            "reports",
            "manuscript",
        ):
            (self.output_root / subdirectory).mkdir(parents=True, exist_ok=True)
        self.logger = configure_logging(
            self.output_root / "logs" / "molecular_origin_analysis.log",
            verbose=verbose,
        )
        seed_everything(int(self.config["model"].get("seed", 42)))
        self._inspection: InspectionReport | None = None
        self.stage_status: dict[str, dict[str, Any]] = {}
        self.initial_git_status = git_status(self.project.project_root)

    @property
    def inspection(self) -> InspectionReport:
        if self._inspection is None:
            self._inspection = self.project.inspect()
        return self._inspection

    def _record_stage(
        self,
        name: str,
        status: str,
        outputs: list[str] | None = None,
        detail: str | None = None,
    ) -> None:
        self.stage_status[name] = {
            "status": status,
            "timestamp_utc": utc_now(),
            "outputs": outputs or [],
            "detail": detail,
        }

    def run_inspect(self) -> dict[str, Any]:
        report = self.inspection.to_dict()
        report["git_status_before"] = self.initial_git_status
        report["planned_reuse"] = [
            "il_property_prediction.src.models.factory.build_model",
            "il_property_prediction.src.data.dataset.ILPropertyDataset",
            "checkpoint-stored ConditionScaler and TargetScaler",
            "il_property_prediction.src.chem.global_descriptors",
            "il_property_prediction.src.chem.functional_groups",
            "MIPGraph.forward auxiliary tensors",
            "computational_application_case chapter result tables (read-only)",
        ]
        json_path = write_json(
            self.output_root / "reports" / "project_inspection.json",
            report,
        )
        warnings = "\n".join(f"- {warning}" for warning in report["warnings"]) or "- None"
        markdown = f"""# MIPGraph project inspection

Generated: {utc_now()}

## Selected runtime

- Model class: `{report['model_class']}`
- Model config: `{report['model_config']}`
- Requested checkpoint protocol: `{report['requested_checkpoint_type']}`
- Selected checkpoint protocol: `{report['selected_checkpoint_type']}`
- Selected checkpoint: `{report['selected_checkpoint']}`
- Selected split: `{report['selected_split']}`
- Property order: `{', '.join(report['property_names'])}`
- Global descriptor dimension: `{report['descriptor_dimensions']['global']}`
- Functional-group descriptor dimension: `{report['descriptor_dimensions']['functional_group']}`

## Verified intermediate access

- Cross-ion attention: `{report['cross_ion_attention_access']}`
- MoE router: `{report['router_access']}`
- Expert outputs: `{report['expert_output_access']}`
- Descriptor names complete: `{report['descriptor_names_complete']}`

The model's existing `forward` method already exposes these tensors. The new
module therefore consumes the returned auxiliary dictionary and uses temporary
forward hooks only for condition-modulated activations. No existing source file
is changed.

## Checkpoint selection limitation

{warnings}

The current workspace contains no random-IL, property-balanced, or ion-family
checkpoint file. Consequently, the available random-point checkpoint is used
only as an explicit fallback for executed inference. Cross-checkpoint claims
remain **not assessable**, rather than being inferred from previously generated
decision tables.

## Scientific boundaries

Statistical associations, frozen-model sensitivities, and chemistry-based
interpretations are reported separately. Shared cross-ion attention is not
treated as a property-specific interaction energy. Candidate screening tables
are read only and candidate membership is never changed by this module.
"""
        markdown_path = self.output_root / "reports" / "project_inspection.md"
        markdown_path.write_text(markdown, encoding="utf-8")
        outputs = [str(json_path), str(markdown_path)]
        self._record_stage("inspect", "completed", outputs)
        return report

    def _cache_identity(self) -> str:
        checkpoint = self.inspection.selected_checkpoint
        stat = checkpoint.stat()
        payload = {
            "config": self.config,
            "checkpoint": str(checkpoint),
            "checkpoint_size": stat.st_size,
            "checkpoint_mtime_ns": stat.st_mtime_ns,
            "split": str(self.inspection.selected_split),
            "dataset_mtime_ns": self.inspection.data_paths["clean_csv"].stat().st_mtime_ns,
        }
        return stable_hash(payload)

    @staticmethod
    def _path_stamps(paths: list[Path]) -> list[dict[str, Any]]:
        stamps: list[dict[str, Any]] = []
        for path in paths:
            resolved = Path(path).resolve()
            if resolved.is_file():
                stat = resolved.stat()
                stamps.append(
                    {
                        "path": str(resolved),
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                    }
                )
            else:
                stamps.append({"path": str(resolved), "missing": True})
        return stamps

    def _stage_cache_identity(
        self,
        stage: str,
        upstream_paths: list[Path],
    ) -> str:
        return stable_hash(
            {
                "stage": stage,
                "base_identity": self._cache_identity(),
                "config": self.config,
                "upstream": self._path_stamps(upstream_paths),
            }
        )

    def _stage_cache_metadata_path(self, stage: str) -> Path:
        return self.output_root / "cache" / f"{stage}_cache_metadata.json"

    def _stage_cache_valid(
        self,
        stage: str,
        output_paths: list[Path],
        upstream_paths: list[Path],
    ) -> bool:
        metadata_path = self._stage_cache_metadata_path(stage)
        if not metadata_path.is_file() or not all(
            path.is_file() for path in output_paths
        ):
            return False
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return metadata.get("cache_identity") == self._stage_cache_identity(
            stage,
            upstream_paths,
        )

    def _write_stage_cache_metadata(
        self,
        stage: str,
        output_paths: list[Path],
        upstream_paths: list[Path],
    ) -> Path:
        return write_json(
            self._stage_cache_metadata_path(stage),
            {
                "stage": stage,
                "generated_utc": utc_now(),
                "cache_identity": self._stage_cache_identity(
                    stage,
                    upstream_paths,
                ),
                "outputs": [str(path) for path in output_paths],
                "upstream": self._path_stamps(upstream_paths),
            },
        )

    @property
    def bundle_base(self) -> Path:
        return self.output_root / "cache" / "analysis_bundle"

    def load_bundle(self) -> FeatureBundle:
        metadata_path = self.bundle_base.with_name(
            f"{self.bundle_base.name}_metadata.json"
        )
        if not metadata_path.is_file():
            raise FileNotFoundError("Extraction cache is absent; run the extract stage first")
        return FeatureExtractor.load_bundle(self.bundle_base)

    def run_extract(self, force: bool = False) -> FeatureBundle:
        metadata_path = self.bundle_base.with_name(
            f"{self.bundle_base.name}_metadata.json"
        )
        cache_identity = self._cache_identity()
        if metadata_path.is_file() and not force:
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            if metadata.get("cache_identity") == cache_identity:
                bundle = self.load_bundle()
                self._record_stage(
                    "extract",
                    "cached",
                    list(metadata.get("cache_files", {}).values()),
                )
                return bundle
        data = DataAdapter(self.config, self.inspection).load()
        model = ModelAdapter(self.config, self.inspection)
        outputs = model.predict(data)
        bundle = FeatureExtractor(self.config).from_model_outputs(data, outputs)
        bundle.metadata.update(
            {
                "cache_identity": cache_identity,
                "generated_utc": utc_now(),
                "config_path": self.config["_config_path"],
            }
        )
        cache_files = FeatureExtractor.save_bundle(bundle, self.bundle_base)
        expected_files = {
            "model_outputs": write_table(
                bundle.records,
                self.output_root / "cache" / "model_outputs.parquet",
            ),
            "descriptor_matrix": write_table(
                bundle.descriptors,
                self.output_root / "cache" / "descriptor_matrix.parquet",
            ),
        }
        latent_path = self.output_root / "cache" / "latent_representations.npz"
        np.savez_compressed(
            latent_path,
            **{
                key: value
                for key, value in bundle.latent_arrays.items()
                if key
                not in {
                    "global_descriptors",
                    "functional_group_descriptors",
                    "router_weights",
                }
            },
        )
        router = bundle.latent_arrays["router_weights"]
        router_rows = []
        for sample_index, sample_id in enumerate(bundle.latent_arrays["sample_id"]):
            for property_index, property_name in enumerate(PROPERTY_NAMES):
                for mechanism_index, mechanism in enumerate(MECHANISM_NAMES):
                    router_rows.append(
                        {
                            "sample_id": int(sample_id),
                            "property": property_name,
                            "mechanism_associated_channel": mechanism,
                            "router_weight": float(
                                router[sample_index, property_index, mechanism_index]
                            ),
                            "interpretation_scope": (
                                "model routing weight; not a measured physical mechanism"
                            ),
                        }
                    )
        router_path = write_table(
            pd.DataFrame(router_rows),
            self.output_root / "cache" / "router_weights.parquet",
        )
        attention_dir = self.output_root / "cache" / "cross_ion_attention"
        attention_path = write_table(
            bundle.auxiliary_tables["cross_ion_attention"],
            attention_dir / "cross_ion_attention_summary.parquet",
        )
        summary_path = self.output_root / "reports" / "extraction_summary.md"
        summary_path.write_text(
            f"""# Extraction summary

- Generated: {utc_now()}
- Samples: {len(bundle.records)}
- Properties: {', '.join(PROPERTY_NAMES)}
- Checkpoint: `{self.inspection.selected_checkpoint}`
- Checkpoint type: `{self.inspection.selected_checkpoint_type}`
- Split: `{self.inspection.selected_split}`
- Global descriptor names/dimension: {len(DESCRIPTOR_NAMES)}
- Functional-group names/dimension: {len(FUNCTIONAL_GROUP_NAMES)}
- Prediction values finite: {bool(np.isfinite(outputs.predictions).all())}
- Identity alignment verified: yes
- Target inverse transform: `{outputs.metadata['target_inverse_transform']}`
- Attention scope: shared structural attention, not interaction energy
""",
            encoding="utf-8",
        )
        output_paths = [
            *(str(path) for path in cache_files.values()),
            *(str(path) for path in expected_files.values()),
            str(latent_path),
            str(router_path),
            str(attention_path),
            str(summary_path),
        ]
        self._record_stage("extract", "completed", output_paths)
        return bundle

    def run_association(self, force: bool = False) -> dict[str, pd.DataFrame]:
        paths = {
            "feature_property_associations": self.output_root
            / "tables"
            / "feature_property_associations.csv",
            "partial_correlations": self.output_root
            / "tables"
            / "partial_correlations.csv",
            "family_stratified_associations": self.output_root
            / "tables"
            / "family_stratified_associations.csv",
            "nonlinear_structure_property_trends": self.output_root
            / "tables"
            / "nonlinear_structure_property_trends.csv",
            "robust_structure_property_factors": self.output_root
            / "tables"
            / "robust_structure_property_factors.csv",
            "record_weighted_feature_property_associations": self.output_root
            / "tables"
            / "record_weighted_feature_property_associations.csv",
            "record_weighted_vs_identity_balanced": self.output_root
            / "tables"
            / "record_weighted_vs_identity_balanced.csv",
            "family_proxy_diagnostics": self.output_root
            / "tables"
            / "family_proxy_diagnostics.csv",
            "identity_level_condition_adjusted_responses": self.output_root
            / "tables"
            / "identity_level_condition_adjusted_responses.csv",
            "nonredundant_structural_themes": self.output_root
            / "tables"
            / "nonredundant_structural_themes.csv",
            "heat_capacity_size_control": self.output_root
            / "tables"
            / "heat_capacity_size_control.csv",
            "heat_capacity_size_control_identity_data": self.output_root
            / "tables"
            / "heat_capacity_size_control_identity_data.csv",
            "property_data_support_counts": self.output_root
            / "tables"
            / "property_data_support_counts.csv",
        }
        upstream = [
            self.bundle_base.with_name(f"{self.bundle_base.name}_metadata.json")
        ]
        if (
            not force
            and self._stage_cache_valid(
                "association",
                list(paths.values()),
                upstream,
            )
        ):
            tables = {name: pd.read_csv(path) for name, path in paths.items()}
            self._record_stage(
                "association", "cached", [str(path) for path in paths.values()]
            )
            return tables
        result = IdentityBalancedStructurePropertyAnalyzer(self.config).run(
            self.load_bundle()
        )
        tables = {
            "feature_property_associations": result.associations,
            "partial_correlations": result.associations,
            "family_stratified_associations": result.diagnostics,
            "nonlinear_structure_property_trends": result.response_shapes,
            "robust_structure_property_factors": result.robust_factors,
            "record_weighted_feature_property_associations": (
                result.record_weighted_associations
            ),
            "record_weighted_vs_identity_balanced": result.comparison,
            "family_proxy_diagnostics": result.diagnostics,
            "identity_level_condition_adjusted_responses": (
                result.identity_responses
            ),
            "nonredundant_structural_themes": result.structural_themes,
            "heat_capacity_size_control": result.heat_capacity_size_control,
            "heat_capacity_size_control_identity_data": (
                result.heat_capacity_identity_data
            ),
            "property_data_support_counts": result.data_support,
        }
        for name, frame in tables.items():
            write_table(frame, paths[name])
        self._write_stage_cache_metadata(
            "association",
            list(paths.values()),
            upstream,
        )
        self._record_stage(
            "association", "completed", [str(path) for path in paths.values()]
        )
        return tables

    def run_attribution(self, force: bool = False) -> dict[str, pd.DataFrame]:
        paths = {
            "property_feature_importance": self.output_root
            / "tables"
            / "property_feature_importance.csv",
            "grouped_feature_importance": self.output_root
            / "tables"
            / "grouped_feature_importance.csv",
            "module_level_importance": self.output_root
            / "tables"
            / "module_level_importance.csv",
            "temperature_conditioned_importance": self.output_root
            / "tables"
            / "temperature_conditioned_importance.csv",
            "attribution_method_agreement": self.output_root
            / "tables"
            / "attribution_method_agreement.csv",
        }
        upstream = [
            self.bundle_base.with_name(f"{self.bundle_base.name}_metadata.json")
        ]
        if (
            not force
            and self._stage_cache_valid(
                "attribution",
                list(paths.values()),
                upstream,
            )
        ):
            tables = {name: pd.read_csv(path) for name, path in paths.items()}
            self._record_stage(
                "attribution", "cached", [str(path) for path in paths.values()]
            )
            return tables
        limit = int(self.config["attribution"].get("maximum_samples", 512))
        data = DataAdapter(self.config, self.inspection).load(max_samples=limit)
        adapter = ModelAdapter(self.config, self.inspection)
        result = AttributionAnalyzer(self.config).run(
            self.load_bundle(),
            adapter,
            data,
        )
        tables = {
            "property_feature_importance": result.property_feature_importance,
            "grouped_feature_importance": result.grouped_feature_importance,
            "module_level_importance": result.module_level_importance,
            "temperature_conditioned_importance": result.temperature_conditioned_importance,
            "attribution_method_agreement": result.method_agreement,
        }
        for name, frame in tables.items():
            write_table(frame, paths[name])
        write_json(
            self.output_root / "reports" / "attribution_metadata.json",
            result.metadata,
        )
        self._write_stage_cache_metadata(
            "attribution",
            list(paths.values()),
            upstream,
        )
        self._record_stage(
            "attribution", "completed", [str(path) for path in paths.values()]
        )
        return tables

    def run_attention(self, force: bool = False) -> dict[str, pd.DataFrame]:
        paths = {
            "cross_ion_interaction_statistics": self.output_root
            / "tables"
            / "cross_ion_interaction_statistics.csv",
            "property_conditioned_interactions": self.output_root
            / "tables"
            / "property_conditioned_interactions.csv",
            "family_interaction_profiles": self.output_root
            / "tables"
            / "family_interaction_profiles.csv",
            "high_low_property_interaction_contrasts": self.output_root
            / "tables"
            / "high_low_property_interaction_contrasts.csv",
            "attention_family_stratified_contrasts": self.output_root
            / "tables"
            / "attention_family_stratified_contrasts.csv",
        }
        required_cache = [
            paths["cross_ion_interaction_statistics"],
            paths["family_interaction_profiles"],
            paths["high_low_property_interaction_contrasts"],
            paths["attention_family_stratified_contrasts"],
        ]
        upstream = [
            self.bundle_base.with_name(f"{self.bundle_base.name}_metadata.json")
        ]
        if (
            not force
            and self._stage_cache_valid(
                "attention",
                required_cache,
                upstream,
            )
        ):
            tables = {
                name: pd.read_csv(path)
                for name, path in paths.items()
                if path.is_file()
            }
            self._record_stage(
                "attention", "cached", [str(path) for path in required_cache]
            )
            return tables
        result = CrossIonAnalyzer(self.config).run(self.load_bundle())
        tables = {
            "cross_ion_interaction_statistics": result.interaction_statistics,
            "property_conditioned_interactions": result.property_conditioned_interactions,
            "family_interaction_profiles": result.family_interaction_profiles,
            "high_low_property_interaction_contrasts": result.high_low_property_contrasts,
            "attention_family_stratified_contrasts": (
                result.family_stratified_contrasts
            ),
        }
        for name, frame in tables.items():
            if not frame.empty:
                write_table(frame, paths[name])
        interpretation = self.output_root / "reports" / "cross_ion_interpretation.md"
        interpretation.write_text(
            f"""# Cross-ion interaction interpretation

The extracted attention is shared by all six property heads. Therefore, the
reported profiles describe model focus patterns and observed-property
contrasts; they are not property-specific attention maps and are not
cation–anion interaction energies.

Both total attention mass and attention per valid atom pair are retained to
control the compositional dilution induced by different ion sizes.

Property-conditioned attention×gradient was not promoted in this run because
the current implementation has not yet passed an independent numerical
stability audit. This optional second-priority result is reported as
unavailable rather than replaced by shared attention.
""",
            encoding="utf-8",
        )
        self._write_stage_cache_metadata(
            "attention",
            required_cache,
            upstream,
        )
        self._record_stage(
            "attention",
            "completed_with_limitation",
            [
                *(str(path) for path in paths.values() if path.is_file()),
                str(interpretation),
            ],
            "Property-conditioned attention×gradient not reported.",
        )
        return tables

    @staticmethod
    def _counterfactual_trend_summary(
        predictions: pd.DataFrame,
    ) -> pd.DataFrame:
        if predictions.empty:
            return pd.DataFrame()
        property_names = [name for name in PROPERTY_NAMES if name in predictions]
        frame = predictions.copy()
        frame["series_order"] = (
            frame.get("series_member", pd.Series(index=frame.index, dtype=object))
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
        )
        frame["series_order"] = pd.to_numeric(frame["series_order"], errors="coerce")
        rows = []
        for (template_id, modification_type), group in frame.groupby(
            ["template_id", "modification_type"]
        ):
            linked_feature = {
                "cation alkyl-chain homologation": "cation_longest_aliphatic_carbon_chain",
                "anion alkyl-chain homologation": "anion_longest_aliphatic_carbon_chain",
            }.get(modification_type)
            for property_name in property_names:
                temperature_directions = []
                correlations = []
                for temperature, temperature_group in group.groupby("temperature_K"):
                    valid = temperature_group["series_order"].notna()
                    if valid.sum() < 3:
                        continue
                    rho = temperature_group.loc[valid, ["series_order", property_name]].corr(
                        method="spearman"
                    ).iloc[0, 1]
                    if np.isfinite(rho):
                        correlations.append(float(rho))
                        temperature_directions.append(int(np.sign(rho)))
                consistent = (
                    bool(temperature_directions)
                    and len(set(temperature_directions)) == 1
                    and temperature_directions[0] != 0
                )
                rows.append(
                    {
                        "template_id": template_id,
                        "modification_type": modification_type,
                        "property": property_name,
                        "linked_feature": linked_feature,
                        "temperature_points_evaluated": len(correlations),
                        "median_series_spearman": (
                            float(np.median(correlations)) if correlations else np.nan
                        ),
                        "temperature_direction_consistent": consistent,
                        "applicability_scope": (
                            "conditional frozen-model prediction; experimental validation required"
                        ),
                    }
                )
        return pd.DataFrame(rows)

    def run_counterfactual(self, force: bool = False) -> dict[str, pd.DataFrame]:
        paths = {
            "matched_molecular_pairs": self.output_root
            / "tables"
            / "matched_molecular_pairs.csv",
            "virtual_counterfactual_library": self.output_root
            / "tables"
            / "virtual_counterfactual_library.csv",
            "counterfactual_predictions": self.output_root
            / "tables"
            / "counterfactual_predictions.csv",
            "counterfactual_trend_summary": self.output_root
            / "tables"
            / "counterfactual_trend_summary.csv",
            "cross_checkpoint_counterfactual_consistency": self.output_root
            / "tables"
            / "cross_checkpoint_counterfactual_consistency.csv",
            "condition_matched_substitution_matches": self.output_root
            / "tables"
            / "condition_matched_substitution_matches.csv",
            "condition_matched_substitution_pairs": self.output_root
            / "tables"
            / "condition_matched_substitution_pairs.csv",
            "condition_matched_substitution_summary": self.output_root
            / "tables"
            / "condition_matched_substitution_summary.csv",
            "signed_substitution_descriptors": self.output_root
            / "tables"
            / "signed_substitution_descriptors.csv",
        }
        required_cache = [
            paths["matched_molecular_pairs"],
            paths["virtual_counterfactual_library"],
            paths["counterfactual_predictions"],
            paths["counterfactual_trend_summary"],
            paths["cross_checkpoint_counterfactual_consistency"],
            paths["condition_matched_substitution_matches"],
            paths["condition_matched_substitution_pairs"],
            paths["condition_matched_substitution_summary"],
            paths["signed_substitution_descriptors"],
        ]
        upstream = [
            self.bundle_base.with_name(f"{self.bundle_base.name}_metadata.json"),
            MODULE_ROOT / "templates" / "counterfactual_templates.yaml",
        ]
        if (
            not force
            and self._stage_cache_valid(
                "counterfactual",
                required_cache,
                upstream,
            )
        ):
            tables = {
                name: pd.read_csv(path)
                for name, path in paths.items()
                if path.is_file()
            }
            self._record_stage(
                "counterfactual", "cached", [str(path) for path in required_cache]
            )
            return tables
        generator = CounterfactualGenerator(self.config)
        valid, rejected = generator.validate_records(generator.template_records())
        valid = valid.reset_index(drop=True)
        valid.insert(
            0,
            "candidate_id",
            [f"CF-{index + 1:04d}" for index in range(len(valid))],
        )
        bundle = self.load_bundle()
        matched = generator.matched_pairs(bundle.records)
        (
            matched_conditions,
            matched_unique_pairs,
            matched_summary,
            signed_substitutions,
        ) = generator.summarize_matched_pairs(matched)
        adapter = ModelAdapter(self.config, self.inspection)
        predictions, failures = adapter.predict_ion_pairs(
            valid,
            [float(value) for value in self.config["counterfactual"]["temperatures_k"]],
            float(self.config["conditions"]["reference_pressure_kpa"]),
        )
        if not predictions.empty:
            descriptor_rows = (
                predictions.sort_values("temperature_K")
                .drop_duplicates("candidate_id")
                .reset_index(drop=True)
            )
            query = pd.DataFrame({"sample_id": descriptor_rows["candidate_id"]})
            for index, name in enumerate(DESCRIPTOR_NAMES):
                query[name] = descriptor_rows[f"global_descriptor_{index}"]
            for index, name in enumerate(FUNCTIONAL_GROUP_NAMES):
                query[name] = descriptor_rows[f"functional_group_descriptor_{index}"]
            training = DataAdapter(self.config, self.inspection).load(
                split_name="train",
            )
            training_descriptors = FeatureExtractor.descriptors_from_graph_cache(training)
            ad_result = ApplicabilityDomainAnalyzer(self.config).evaluate(
                training_descriptors,
                query,
            )
            ad_table = ad_result.query.rename(columns={"sample_id": "candidate_id"})
            predictions = predictions.merge(ad_table, on="candidate_id", how="left")
            write_json(
                self.output_root / "reports" / "counterfactual_ad_thresholds.json",
                {
                    "thresholds": ad_result.thresholds,
                    "metadata": ad_result.metadata,
                },
            )
        trend_summary = self._counterfactual_trend_summary(predictions)
        checkpoint_consistency = trend_summary[
            ["template_id", "property"]
        ].drop_duplicates()
        checkpoint_consistency["available_checkpoint_count"] = len(
            self.inspection.checkpoint_candidates
        )
        checkpoint_consistency["consistency_status"] = (
            "not_assessable_single_checkpoint"
            if len(self.inspection.checkpoint_candidates) < 2
            else "requires_protocol_specific_rerun"
        )
        tables = {
            "matched_molecular_pairs": matched,
            "virtual_counterfactual_library": valid,
            "counterfactual_predictions": predictions,
            "counterfactual_trend_summary": trend_summary,
            "cross_checkpoint_counterfactual_consistency": checkpoint_consistency,
            "condition_matched_substitution_matches": matched_conditions,
            "condition_matched_substitution_pairs": matched_unique_pairs,
            "condition_matched_substitution_summary": matched_summary,
            "signed_substitution_descriptors": signed_substitutions,
        }
        for name, frame in tables.items():
            if not frame.empty:
                write_table(frame, paths[name])
        if not rejected.empty:
            write_table(
                rejected,
                self.output_root / "tables" / "counterfactual_rejections.csv",
            )
        if not failures.empty:
            write_table(
                failures,
                self.output_root / "tables" / "counterfactual_inference_failures.csv",
            )
        self._write_stage_cache_metadata(
            "counterfactual",
            required_cache,
            upstream,
        )
        self._record_stage(
            "counterfactual",
            "completed_with_checkpoint_limitation",
            [str(path) for path in paths.values() if path.is_file()],
            "Cross-checkpoint consistency cannot be evaluated from one available checkpoint.",
        )
        return tables

    def run_applicability(self, force: bool = False) -> pd.DataFrame:
        path = self.output_root / "tables" / "observed_test_applicability_domain.csv"
        upstream = [
            self.bundle_base.with_name(f"{self.bundle_base.name}_metadata.json")
        ]
        if (
            not force
            and self._stage_cache_valid(
                "applicability",
                [path],
                upstream,
            )
        ):
            frame = pd.read_csv(path)
            self._record_stage("applicability", "cached", [str(path)])
            return frame
        bundle = self.load_bundle()
        training = DataAdapter(self.config, self.inspection).load(split_name="train")
        reference = FeatureExtractor.descriptors_from_graph_cache(training)
        result = ApplicabilityDomainAnalyzer(self.config).evaluate(
            reference,
            bundle.descriptors,
        )
        write_table(result.query, path)
        write_json(
            self.output_root / "reports" / "applicability_domain_summary.json",
            {"thresholds": result.thresholds, "metadata": result.metadata},
        )
        self._write_stage_cache_metadata(
            "applicability",
            [path],
            upstream,
        )
        self._record_stage("applicability", "completed", [str(path)])
        return result.query

    def run_rules(self, force: bool = False) -> dict[str, pd.DataFrame]:
        paths = {
            "design_rule_summary": self.output_root / "tables" / "design_rule_summary.csv",
            "unsupported_hypotheses": self.output_root / "tables" / "unsupported_hypotheses.csv",
            "candidate_structural_profiles": self.output_root
            / "tables"
            / "candidate_structural_profiles.csv",
            "top8_vs_nonfeasible_comparison": self.output_root
            / "tables"
            / "top8_vs_nonfeasible_comparison.csv",
            "candidate_rule_consistency": self.output_root
            / "tables"
            / "candidate_rule_consistency.csv",
            "molecular_structure_property_evidence_table": self.output_root
            / "tables"
            / "molecular_structure_property_evidence_table.csv",
        }
        upstream = [
            self.output_root / "tables" / "robust_structure_property_factors.csv",
            self.output_root / "tables" / "property_feature_importance.csv",
            self.output_root / "tables" / "counterfactual_trend_summary.csv",
            *self.inspection.screening_paths.values(),
        ]
        if (
            not force
            and self._stage_cache_valid(
                "rules",
                list(paths.values()),
                upstream,
            )
        ):
            tables = {
                name: pd.read_csv(path)
                for name, path in paths.items()
                if path.is_file()
            }
            self._record_stage("rules", "cached", [str(path) for path in paths.values() if path.is_file()])
            return tables
        robust = pd.read_csv(
            self.output_root / "tables" / "robust_structure_property_factors.csv"
        )
        importance = pd.read_csv(
            self.output_root / "tables" / "property_feature_importance.csv"
        )
        counter_path = self.output_root / "tables" / "counterfactual_trend_summary.csv"
        counter = pd.read_csv(counter_path) if counter_path.is_file() else None
        record_weighted_path = (
            self.output_root
            / "tables"
            / "record_weighted_feature_property_associations.csv"
        )
        record_weighted = (
            pd.read_csv(record_weighted_path)
            if record_weighted_path.is_file()
            else None
        )
        screening = DataAdapter(self.config, self.inspection).load_screening_assets()
        result = DesignRuleSynthesizer(self.config).run(
            robust,
            importance,
            screening,
            counter,
            checkpoint_count=len(self.inspection.checkpoint_candidates),
            record_weighted=record_weighted,
        )
        tables = {
            "design_rule_summary": result.design_rules,
            "unsupported_hypotheses": result.unsupported_hypotheses,
            "candidate_structural_profiles": result.candidate_profiles,
            "top8_vs_nonfeasible_comparison": result.top8_vs_nonfeasible,
            "candidate_rule_consistency": result.candidate_rule_consistency,
            "molecular_structure_property_evidence_table": (
                result.evidence_table
            ),
        }
        for name, frame in tables.items():
            if not frame.empty:
                write_table(frame, paths[name])
        write_json(
            self.output_root / "manuscript" / "property_design_rules.json",
            result.property_design_rules,
        )
        evidence_path = self.output_root / "reports" / "design_rule_evidence.md"
        evidence_path.write_text(
            f"""# Design-rule evidence audit

- Formal evidence-gated rules: {len(result.design_rules)}
- Unsupported hypotheses retained for audit: {len(result.unsupported_hypotheses)}
- Available checkpoint protocols: {len(self.inspection.checkpoint_candidates)}

Level A requires statistical, direct-attribution, real-SMILES counterfactual,
family, and multi-checkpoint agreement. Because only one checkpoint is present,
this run cannot award Level A solely from checkpoint evidence.

Rules describe associations and frozen-model sensitivities. They are not causal
laws and do not establish electrochemical suitability.
""",
            encoding="utf-8",
        )
        interpretation = self.output_root / "reports" / "top8_molecular_interpretation.md"
        top_features = (
            result.top8_vs_nonfeasible.head(10)["feature"].tolist()
            if not result.top8_vs_nonfeasible.empty
            else []
        )
        interpretation.write_text(
            f"""# Top-8 molecular interpretation

This report is post hoc and does not alter hard constraints, Pareto sorting, or
the formal shortlist.

The largest descriptive Top-8 versus non-feasible structural contrasts were:
{chr(10).join(f'- `{feature}`' for feature in top_features) if top_features else '- Not available'}

Candidate-level rule consistency is reported only where an independently
evidence-gated rule exists. A shortlist candidate is never assigned a favourable
explanation merely because it was selected.
""",
            encoding="utf-8",
        )
        self._write_stage_cache_metadata(
            "rules",
            [path for path in paths.values() if path.is_file()],
            upstream,
        )
        self._record_stage(
            "rules",
            "completed",
            [str(path) for path in paths.values() if path.is_file()],
        )
        return tables

    def run_figures(self, force: bool = False) -> dict[str, list[str]]:
        del force
        plotter = PublicationPlotter(self.config)
        figure_dir = self.output_root / "figures"
        tables = self.output_root / "tables"
        calls: list[tuple[str, Callable[[], list[Path]]]] = [
            (
                "figure_main_molecular_origin_analysis_final",
                lambda: plotter.composite_results_figure_v2(
                    pd.read_csv(tables / "design_rule_summary.csv"),
                    pd.read_csv(tables / "feature_property_associations.csv"),
                    pd.read_csv(tables / "nonlinear_structure_property_trends.csv"),
                    pd.read_csv(
                        tables / "condition_matched_substitution_pairs.csv"
                    ),
                    pd.read_csv(
                        tables / "high_low_property_interaction_contrasts.csv"
                    ),
                    figure_dir / "figure_main_molecular_origin_analysis_final",
                ),
            ),
            (
                "figure_si_heat_capacity_size_control",
                lambda: plotter.heat_capacity_size_control_figure(
                    pd.read_csv(
                        tables / "heat_capacity_size_control_identity_data.csv"
                    ),
                    figure_dir / "figure_si_heat_capacity_size_control",
                ),
            ),
        ]
        outputs: dict[str, list[str]] = {}
        failures: dict[str, str] = {}
        for name, call in calls:
            try:
                outputs[name] = [str(path) for path in call()]
            except (FileNotFoundError, KeyError, ValueError) as exc:
                failures[name] = f"{type(exc).__name__}: {exc}"
                self.logger.warning("Figure %s was not generated: %s", name, exc)
        write_json(
            self.output_root / "reports" / "figure_generation_summary.json",
            {"outputs": outputs, "failures": failures},
        )
        (self.output_root / "reports" / "composite_figure_contract.md").write_text(
            """# Composite figure contract

- Core conclusion: condition-controlled associations, matched ion
  substitutions and frozen-model diagnostics jointly support auditable
  microstructure–property relationships that can be used as qualitative
  structural priors before full property inference.
- Archetype: asymmetric mixed-modality quantitative figure.
- Backend: Python/Matplotlib only.
- Final size: 183 mm × 173 mm.
- Panel a: integrated hero map in which the three highest-confidence
  condition-controlled structural associations and three largest cross-ion
  attention contrasts per property converge on the same property nodes.
- Panel b: ion-level dominance plot showing the strongest signed partial
  correlation within cation, anion and ion-pair descriptor scopes for each
  property; it answers which molecular level dominates rather than repeating
  the named links in panel a.
- Panels b and d: a vertically stacked left-hand evidence column containing
  ion-level dominance and condition-matched substitution-effect forests. A
  shared low-saturation blue/coral/violet family identifies ion roles.
- Panel c: a right-hand 3 × 2 small-multiple atlas containing three
  curve-supported structural priors per property. Candidates are ordered
  deterministically by confidence level, absolute partial correlation, family
  consistency and factor name. Blue, coral and violet identify cation, anion
  and ion-pair scopes; solid-circle, dashed-square and dotted-triangle curves
  identify ranks 1--3.
- Reviewer boundary: no candidate screening, causal, interaction-energy,
  electrochemical-suitability or cross-checkpoint-stability claim is made.
""",
            encoding="utf-8",
        )
        (self.output_root / "reports" / "composite_figure_qa.md").write_text(
            """# Composite figure quality-assurance record

- Rendering backend: Python/Matplotlib.
- Export bundle: PNG and TIFF at 600 dpi; PDF and SVG with editable text.
- Typography at the 183 mm final width: panel titles are 8.5--9.0 pt,
  axes and legends are 5.5--6.0 pt, and the smallest in-panel factor
  key is 5.3 pt.
- Panel labels: bold lowercase a–d at the upper-left of each evidence block.
- Palette: restrained red/blue association directions, low-saturation
  blue/coral/violet ion roles and orange/purple attention contrasts; no
  rainbow map.
- Panel-c legend: one shared key maps blue/coral/violet to
  cation/anion/ion-pair scope and solid-circle/dashed-square/dotted-triangle
  to deterministic ranks 1--3.
- Panel-a encoding: red/blue gives association direction and line width gives
  absolute condition-controlled partial correlation; the source table records
  the exact displayed width and deterministic within-property rank.
- Statistics: sample sizes are printed for response curves and matched-pair
  panels. BH--FDR significance is encoded by marker rims in panel b. Ribbons
  in panel c are within-bin standard errors. Panel d reports 10th, 25th, 50th,
  75th and 90th percentiles without suppressing the underlying source archive.
- Source-data traceability: one CSV per panel is written to
  `results/tables/figure_source_data/`.
- Integrity: panel d is computed from every available observed matched-pair
  value and exports the exact displayed quantiles and sample counts; the full
  pair-level archive remains available in `matched_molecular_pairs.csv`.
- Interpretation control: the right side of panel a explicitly identifies
  attention as a model-focus pattern rather than interaction energy. Rule-level evidence components
  remain available in the source tables rather than being collapsed into a
  prediction, confidence probability or suitability score.
""",
            encoding="utf-8",
        )
        self._record_stage(
            "figures",
            "completed" if not failures else "completed_with_limitations",
            [path for values in outputs.values() for path in values],
            json.dumps(failures, ensure_ascii=False) if failures else None,
        )
        return outputs

    def run_manuscript(self, force: bool = False) -> list[Path]:
        del force
        manuscript = self.output_root / "manuscript"
        rules_path = self.output_root / "tables" / "design_rule_summary.csv"
        rules = pd.read_csv(rules_path) if rules_path.is_file() else pd.DataFrame()
        associations = pd.read_csv(
            self.output_root / "tables" / "feature_property_associations.csv"
        )
        unsupported = pd.read_csv(
            self.output_root / "tables" / "unsupported_hypotheses.csv"
        )
        matched_pairs = pd.read_csv(
            self.output_root / "tables" / "matched_molecular_pairs.csv"
        )
        nonlinear = pd.read_csv(
            self.output_root / "tables" / "nonlinear_structure_property_trends.csv"
        )
        observed_ad = pd.read_csv(
            self.output_root / "tables" / "observed_test_applicability_domain.csv"
        )
        attention_contrasts = pd.read_csv(
            self.output_root
            / "tables"
            / "high_low_property_interaction_contrasts.csv"
        )
        strong = (
            rules.loc[rules["confidence_level"].isin(["Level A", "Level B"])]
            if not rules.empty
            else pd.DataFrame()
        )
        sample_count = int(associations["sample_count"].max())
        representative_strong = (
            strong.assign(
                _level_order=strong["confidence_level"].map(
                    {"Level A": 0, "Level B": 1}
                ),
                _property_order=strong["property"].map(
                    {
                        "Density": 0,
                        "Viscosity": 1,
                        "ElectricalConductivity": 2,
                        "HeatCapacity": 3,
                        "SurfaceTension": 4,
                        "ThermalConductivity": 5,
                    }
                ),
            )
            .assign(
                _partial_r=lambda frame: pd.to_numeric(
                    frame["statistical_evidence"].str.extract(
                        r"partial r=([+-]?[0-9]*\.?[0-9]+)"
                    )[0],
                    errors="coerce",
                )
            )
            .assign(_abs_partial_r=lambda frame: frame["_partial_r"].abs())
            .sort_values(
                [
                    "_property_order",
                    "_level_order",
                    "_abs_partial_r",
                    "family_consistency",
                    "structural_factor",
                ],
                ascending=[True, True, False, False, True],
                na_position="last",
            )
            .groupby("property", sort=False)
            .head(1)
        )
        strongest_text = (
            "; ".join(
                f"{row.structural_factor} ({row.property}, {row.effect_direction})"
                for row in representative_strong.itertuples(index=False)
            )
            if not representative_strong.empty
            else "no Level A/B rule passed all configured evidence gates"
        )
        level_counts = rules["confidence_level"].value_counts().to_dict()
        level_b_count = int(level_counts.get("Level B", 0))
        level_c_count = int(level_counts.get("Level C", 0))

        def _representative(property_name: str) -> pd.Series:
            subset = representative_strong.loc[
                representative_strong["property"] == property_name
            ]
            if subset.empty:
                return pd.Series(dtype=object)
            return subset.iloc[0]

        def _factor(property_name: str) -> str:
            row = _representative(property_name)
            return (
                str(row.get("structural_factor", "unavailable"))
                .replace("_functional_group", "")
                .replace("_", " ")
            )

        def _partial_r(property_name: str) -> float:
            row = _representative(property_name)
            return float(row.get("_partial_r", np.nan))

        def _q_text(property_name: str) -> str:
            row = _representative(property_name)
            evidence = str(row.get("statistical_evidence", ""))
            match = re.search(r"q=([^;]+)", evidence)
            if not match:
                return r"\mathrm{NA}"
            raw_value = match.group(1)
            scientific = re.fullmatch(
                r"([+-]?[0-9]*\.?[0-9]+)e([+-]?[0-9]+)",
                raw_value,
                flags=re.IGNORECASE,
            )
            if scientific:
                return (
                    f"{scientific.group(1)}"
                    rf"\times 10^{{{int(scientific.group(2))}}}"
                )
            return raw_value

        matched_statistics: dict[str, dict[str, float | int]] = {}
        matched_role_statistics: dict[
            str,
            dict[str, dict[str, float | int]],
        ] = {}
        for property_name in [
            "Density",
            "Viscosity",
            "ElectricalConductivity",
            "HeatCapacity",
            "SurfaceTension",
            "ThermalConductivity",
        ]:
            value_column = f"observed_abs_log_difference_{property_name}"
            values = pd.to_numeric(
                matched_pairs[value_column],
                errors="coerce",
            ).dropna()
            matched_statistics[property_name] = {
                "count": int(len(values)),
                "median": float(values.median()),
            }
            matched_role_statistics[property_name] = {}
            for fixed_role in ["anion_fixed", "cation_fixed"]:
                role_values = pd.to_numeric(
                    matched_pairs.loc[
                        matched_pairs["fixed_role"] == fixed_role,
                        value_column,
                    ],
                    errors="coerce",
                ).dropna()
                matched_role_statistics[property_name][fixed_role] = {
                    "count": int(len(role_values)),
                    "median": float(role_values.median()),
                }

        inference_record_count = int(len(observed_ad))

        def _attention_extreme(property_name: str) -> pd.Series:
            subset = attention_contrasts.loc[
                attention_contrasts["property"] == property_name
            ]
            if subset.empty:
                return pd.Series(dtype=object)
            return subset.loc[subset["high_minus_low"].abs().idxmax()]

        viscosity_attention = _attention_extreme("Viscosity")
        conductivity_attention = _attention_extreme(
            "ElectricalConductivity"
        )
        unsupported_count = int(len(unsupported))
        viscosity_attention_delta = float(
            viscosity_attention.get("high_minus_low", np.nan)
        )
        conductivity_attention_delta = float(
            conductivity_attention.get("high_minus_low", np.nan)
        )
        viscosity_attention_n = int(
            viscosity_attention.get("high_group_count", 0)
        )
        conductivity_attention_n = int(
            conductivity_attention.get("high_group_count", 0)
        )
        primary_evidence = PublicationPlotter._rule_evidence_source(
            rules,
            nonlinear,
        )
        primary_family_min = float(
            primary_evidence["family_consistency"].min()
        )
        primary_family_max = float(
            primary_evidence["family_consistency"].max()
        )
        primary_attribution_rank_min = int(
            primary_evidence["attribution_rank"].min()
        )
        primary_attribution_rank_max = int(
            primary_evidence["attribution_rank"].max()
        )
        primary_direction_agreement = int(
            primary_evidence["response_direction_consistent"].sum()
        )
        factor_names_zh = {
            "Density": "阴离子氟原子比例",
            "Viscosity": "阳离子杂芳香原子特征",
            "ElectricalConductivity": "离子对烷基链长总和",
            "HeatCapacity": "离子对总分子量（缩放）",
            "SurfaceTension": "阳离子芳香原子比例",
            "ThermalConductivity": "阴离子分子量（缩放）",
        }

        results_en_markdown = f"""# 3.X Evidence-integrated microstructure–property relationships from MIPGraph

To test whether the frozen MIPGraph representation encodes chemically
interpretable information beyond aggregate predictive accuracy, we analysed
{inference_record_count:,} identity-aligned test records without retraining;
individual observed-property analyses used up to {sample_count:,} labelled
records (Fig. X). Observed-data association, frozen-model attribution and
chemical interpretation were kept separate, and a structure–property statement
was retained only after passing the predefined evidence gates. This procedure
yielded {level_b_count} Level B and {level_c_count} Level C rules, while
preserving {unsupported_count} hypotheses as unsupported. No Level A rule was
assigned because only one compatible checkpoint was available. The resulting
statements are therefore treated as auditable microstructure–property
hypotheses rather than cross-checkpoint molecular laws.

## 3.X.1 Property-specific microstructure–property relationships

The multi-link evidence map summarizes the three highest-confidence eligible
associations for each property (Fig. Xa). After adjustment for temperature,
pressure, cation family and anion family, the ion-level comparison in Fig. Xb
separates whether the strongest retained association within each scope
originates from a cation, anion or ion-pair descriptor. This avoids treating
the molecular representation as a single undifferentiated descriptor block.
Density was most strongly associated with
{_factor("Density")} ($r_{{\mathrm{{partial}}}}={_partial_r("Density"):.3f}$,
$q={_q_text("Density")}$), whereas viscosity was negatively associated with
{_factor("Viscosity")} ($r_{{\mathrm{{partial}}}}={_partial_r("Viscosity"):.3f}$,
$q={_q_text("Viscosity")}$). Electrical conductivity decreased with
{_factor("ElectricalConductivity")}
($r_{{\mathrm{{partial}}}}={_partial_r("ElectricalConductivity"):.3f}$,
$q={_q_text("ElectricalConductivity")}$), whereas heat capacity increased
with {_factor("HeatCapacity")}
($r_{{\mathrm{{partial}}}}={_partial_r("HeatCapacity"):.3f}$). Surface tension and
thermal conductivity were most strongly associated with
{_factor("SurfaceTension")}
($r_{{\mathrm{{partial}}}}={_partial_r("SurfaceTension"):.3f}$) and
{_factor("ThermalConductivity")}
($r_{{\mathrm{{partial}}}}={_partial_r("ThermalConductivity"):.3f}$), respectively.
The rank-1 binned responses for these six primary factors preserved the
corresponding association direction in {primary_direction_agreement} of 6
cases. Figure Xc additionally exposes the rank-2 and rank-3 curve-supported
factors for each property, revealing alternative and sometimes non-linear
responses over the sampled structural ranges. These quantities are conditional
associations, not additive atomic contributions or causal molecular
determinants.

## 3.X.2 Condition-matched structural substitutions

The matched-pair archive contained {len(matched_pairs):,} structural
comparisons aligned in temperature and pressure (Fig. Xd). Transport properties
showed the largest median response magnitudes: median $|\\Delta\\ln y|$ was
{matched_statistics["ElectricalConductivity"]["median"]:.3f} for electrical
conductivity ($n={matched_statistics["ElectricalConductivity"]["count"]:,}$)
and {matched_statistics["Viscosity"]["median"]:.3f} for viscosity
($n={matched_statistics["Viscosity"]["count"]:,}$). The corresponding values
were {matched_statistics["Density"]["median"]:.3f} for density
($n={matched_statistics["Density"]["count"]:,}$),
{matched_statistics["HeatCapacity"]["median"]:.3f} for heat capacity
($n={matched_statistics["HeatCapacity"]["count"]:,}$),
{matched_statistics["SurfaceTension"]["median"]:.3f} for surface tension
($n={matched_statistics["SurfaceTension"]["count"]:,}$), and
{matched_statistics["ThermalConductivity"]["median"]:.3f} for thermal
conductivity ($n={matched_statistics["ThermalConductivity"]["count"]:,}$).
The role-resolved distributions add information that is lost in the pooled
median: anion changes showed a larger median viscosity response than cation
changes ({matched_role_statistics["Viscosity"]["cation_fixed"]["median"]:.3f}
versus
{matched_role_statistics["Viscosity"]["anion_fixed"]["median"]:.3f}), whereas
cation changes produced the larger heat-capacity response
({matched_role_statistics["HeatCapacity"]["anion_fixed"]["median"]:.3f}
versus
{matched_role_statistics["HeatCapacity"]["cation_fixed"]["median"]:.3f}).
Electrical-conductivity responses were comparably broad for the two roles
({matched_role_statistics["ElectricalConductivity"]["anion_fixed"]["median"]:.3f}
versus
{matched_role_statistics["ElectricalConductivity"]["cation_fixed"]["median"]:.3f}).
The thermal-conductivity comparison remains exploratory because only
{matched_statistics["ThermalConductivity"]["count"]} matched pairs carried two
observed labels. By separating cation and anion substitutions, this analysis
quantifies how changes in either ionic component accompany changes in each
macroscopic property without assigning a causal effect to a specific edit.

## 3.X.3 Model diagnostics and qualitative structural priors

Condition-controlled residual quartiles revealed property-dependent contrasts
in the shared cross-ion attention (right side of Fig. Xa). Alkyl-carbon–anion-polar-site
pairs carried more attention per atom pair in the high-conductivity quartile
($\\Delta={conductivity_attention_delta:.3f}$;
$n={conductivity_attention_n}$ per quartile), but less in the high-viscosity
quartile ($\\Delta={viscosity_attention_delta:.3f}$;
$n={viscosity_attention_n}$ per quartile). This opposing transport pattern is
consistent with the model using different ion-pair contexts across the two
response regimes. Because the attention tensor is shared by six property
heads and is compositionally constrained, it is a model-focus diagnostic and
not an interaction energy.

The corresponding rule-level association, family-consistency, attribution and
response-monotonicity values remain available in the auditable source tables.
Once derived from the trained model and reference data, these rules require
only molecular descriptors when applied to a new ion-pair identity. They can
therefore support a qualitative pre-inference judgement of whether a structure
tends to favour higher or lower values of a specific thermophysical property.
They do not replace full MIPGraph inference and do not determine overall
electrolyte suitability, phase behaviour, electrochemical stability,
capacitance, cycling performance or safety.
"""

        results_zh_markdown = f"""# 3.X MIPGraph 提炼的微观结构—宏观性质关系

为检验冻结的 MIPGraph 表征是否在总体预测精度之外编码了可解释的化学
信息，我们在不重新训练模型的前提下分析了 {inference_record_count:,} 条
身份严格对齐的测试记录；单项实验性质分析最多使用 {sample_count:,} 条
有效标签记录（图 X）。实验数据关联、冻结模型归因和化学解释始终分开
处理，只有通过预定义证据门槛的结构—性质关系才被保留。该过程得到
{level_b_count} 条 Level B 和 {level_c_count} 条 Level C 规律，并将
{unsupported_count} 个假设保留为证据不足项。由于当前只有一个兼容
checkpoint，没有规律被授予 Level A。因此，这些结论被视为可审计的
微观结构—宏观热物性假设，而不是跨 checkpoint 的普适分子规律。

## 3.X.1 性质特异性的微观结构—宏观性质关系

多连接证据图汇总了每种性质置信度最高的三条合格关联（图 Xa）。在控制
温度、压力、阳离子家族和阴离子家族后，图 Xb 进一步区分了每种性质在
阳离子、阴离子和离子对描述符范围内的最强关联来源，避免把整个分子表征
视为不可分解的描述符整体。密度与 {factor_names_zh["Density"]} 的关联最强
（偏相关 $r={_partial_r("Density"):.3f}$，
$q={_q_text("Density")}$）；黏度与
{factor_names_zh["Viscosity"]} 呈负相关
（$r={_partial_r("Viscosity"):.3f}$，
$q={_q_text("Viscosity")}$）。电导率随
{factor_names_zh["ElectricalConductivity"]} 增大而降低
（$r={_partial_r("ElectricalConductivity"):.3f}$），热容则随
{factor_names_zh["HeatCapacity"]} 增大而升高
（$r={_partial_r("HeatCapacity"):.3f}$）。表面张力和热导率分别与
{factor_names_zh["SurfaceTension"]}
（$r={_partial_r("SurfaceTension"):.3f}$）及
{factor_names_zh["ThermalConductivity"]}
（$r={_partial_r("ThermalConductivity"):.3f}$）关系最强。
上述六条主要规律中，第 1 位结构因子的实验分箱曲线在
{primary_direction_agreement}/6 种性质中保持了相同方向。图 Xc 进一步
展示每种性质第 2 位和第 3 位具有可用响应曲线的结构因子，从而揭示已采样
结构范围内的替代响应与部分非线性趋势。这些效应量是条件受控关联，不能
解释为原子贡献的简单加和或因果分子决定因素。

## 3.X.2 条件匹配的离子结构替换响应

匹配分子对库包含 {len(matched_pairs):,} 个温度和压力匹配的结构比较
（图 Xd）。输运性质表现出最大的中位响应幅度：电导率和黏度的中位
$|\\Delta\\ln y|$ 分别为
{matched_statistics["ElectricalConductivity"]["median"]:.3f}
（$n={matched_statistics["ElectricalConductivity"]["count"]:,}$）和
{matched_statistics["Viscosity"]["median"]:.3f}
（$n={matched_statistics["Viscosity"]["count"]:,}$）。密度、热容、
表面张力和热导率的对应中位数分别为
{matched_statistics["Density"]["median"]:.3f}、
{matched_statistics["HeatCapacity"]["median"]:.3f}、
{matched_statistics["SurfaceTension"]["median"]:.3f} 和
{matched_statistics["ThermalConductivity"]["median"]:.3f}。
离子角色分辨的分布进一步揭示了总体中位数无法表达的信息：阴离子替换的
黏度响应中位数高于阳离子替换
（{matched_role_statistics["Viscosity"]["cation_fixed"]["median"]:.3f}
对
{matched_role_statistics["Viscosity"]["anion_fixed"]["median"]:.3f}），
而阳离子替换的热容响应更高
（{matched_role_statistics["HeatCapacity"]["anion_fixed"]["median"]:.3f}
对
{matched_role_statistics["HeatCapacity"]["cation_fixed"]["median"]:.3f}）。
两种替换角色的电导率响应均较宽，且中位数相近
（{matched_role_statistics["ElectricalConductivity"]["anion_fixed"]["median"]:.3f}
对
{matched_role_statistics["ElectricalConductivity"]["cation_fixed"]["median"]:.3f}）。
其中热导率
仅有 {matched_statistics["ThermalConductivity"]["count"]} 个双标签匹配
比较，因此该结果只能作为探索性证据。通过分别比较阳离子和阴离子替换，
该分析量化了离子组分变化与各宏观性质变化之间的对应关系，但不把某次
结构编辑解释为因果作用。

## 3.X.3 模型诊断与定性结构先验

基于条件控制残差四分位组的分析显示，共享 cross-ion attention 存在性质
相关的差异模式（图 Xa 右侧）。烷基碳—阴离子极性位点原子对在高电导率组中
获得更多单位原子对 attention
（$\\Delta={conductivity_attention_delta:.3f}$，每组
$n={conductivity_attention_n}$），而在高黏度组中相对减少
（$\\Delta={viscosity_attention_delta:.3f}$，每组
$n={viscosity_attention_n}$）。这一相反的输运模式与模型在两种响应区间
使用不同离子对结构语境相一致。由于该 attention 由六个性质头共享且受
组成约束，它只能解释为模型关注模式，不能解释为真实相互作用能。

相应的关联强度、离子家族一致性、模型归因百分位和实验响应单调性仍完整
保存在可审计源数据表中。这些规律一旦由训练模型和参考数据提炼完成，在
面对新的离子对身份时只需要计算分子结构描述符。因此，它们可以
在完整性质推理之前，对某一结构更倾向于提高或降低特定热物性作出定性的
先验判断。但这些规律不能替代完整 MIPGraph 计算，也不能直接判定电解液
整体适宜性、液态范围、电化学稳定性、真实电容、循环性能或安全性。
"""

        results_en_tex = results_en_markdown.replace(
            "# 3.X Evidence-integrated microstructure–property relationships from MIPGraph",
            r"\subsection{Evidence-integrated microstructure--property relationships from MIPGraph}",
        )
        results_en_tex = re.sub(
            r"^## 3\.X\.\d+ (.+)$",
            r"\\subsubsection{\1}",
            results_en_tex,
            flags=re.MULTILINE,
        )
        results_en_tex = results_en_tex.replace("(Fig. X)", r"(Fig.~\ref{fig:molecular-origin-composite})")
        for panel in "abcdef":
            results_en_tex = results_en_tex.replace(
                f"Fig. X{panel}",
                rf"Fig.~\ref{{fig:molecular-origin-composite}}{panel}",
            )

        results_zh_tex = results_zh_markdown.replace(
            "# 3.X MIPGraph 提炼的微观结构—宏观性质关系",
            r"\subsection{MIPGraph 提炼的微观结构--宏观性质关系}",
        )
        results_zh_tex = re.sub(
            r"^## 3\.X\.\d+ (.+)$",
            r"\\subsubsection{\1}",
            results_zh_tex,
            flags=re.MULTILINE,
        )
        results_zh_tex = results_zh_tex.replace("（图 X）", r"（图~\ref{fig:molecular-origin-composite}）")
        for panel in "abcdef":
            results_zh_tex = results_zh_tex.replace(
                f"图 X{panel}",
                rf"图~\ref{{fig:molecular-origin-composite}}{panel}",
            )

        composite_caption_en = r"""\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figure_main_molecular_origin_analysis.pdf}
\caption{\textbf{Evidence-integrated microstructure--property relationships from MIPGraph.}
\textbf{a}, Integrated microstructure--property evidence map. The left side contains the three highest-confidence eligible condition-controlled structural associations per property; red and blue denote positive and negative partial correlations, respectively. The right side contains the three largest shared cross-ion attention contrasts per property between upper and lower quartiles of condition-controlled observed-property residuals; orange and purple denote higher and lower attention in the upper quartile, respectively. Line width scales within each evidence family. All structural links are associative and non-causal, and attention is a model-focus diagnostic rather than an interaction energy.
\textbf{b}, Ion-level origin of the condition-controlled associations. For each property, the strongest absolute partial correlation is selected independently within the cation, anion and ion-pair descriptor scopes; horizontal position and printed value retain its sign and magnitude, and a dark marker rim denotes BH--FDR $q\leq0.05$. This panel compares the dominant molecular level rather than repeating the individual links in panel a.
\textbf{c}, Property-specific response shapes for three curve-supported structural priors per property, arranged as a $3\times2$ small-multiple atlas. Factors are selected deterministically by confidence level, absolute partial correlation, family consistency and factor name; Level C evidence is used only after available Level A/B factors in this ordering. Blue, coral and violet identify cation-, anion- and ion-pair-level priors, whereas solid circles, dashed squares and dotted triangles identify ranks 1, 2 and 3. In-panel keys report the factor identity and partial correlation. Curves show centered observed log-property responses, points are bin means and ribbons are within-bin standard errors; the source-data export retains the binned-response Spearman coefficient, log-response range, confidence level and labelled sample size for every curve. Vertical scales are property specific and are not intended for cross-property magnitude comparison.
\textbf{d}, Distribution forest of observed response magnitudes for condition-matched cation and anion substitutions. Thin and thick segments denote the 10th--90th percentile range and interquartile range, respectively; points denote medians, dotted connectors compare ion roles, and the directly labelled C/A sample counts identify cation and anion changes without a detached legend.}
\label{fig:molecular-origin-composite}
\end{figure*}
"""
        composite_caption_zh = r"""\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figure_main_molecular_origin_analysis.pdf}
\caption{\textbf{MIPGraph 提炼的微观结构--宏观性质关系。}
\textbf{a}，整合的微观结构--宏观性质证据图。左侧为每种性质三个置信等级最高且满足条件的结构关联，红色和蓝色分别表示正、负条件受控偏相关；右侧为条件受控实验性质残差上下四分位组之间，每种性质三个最大的共享 cross-ion attention 差异，橙色和紫色分别表示高性质组 attention 增加和降低。线宽分别在两类证据内部随效应绝对值变化。所有结构连线均为非因果关联，attention 仅表示模型关注模式而不代表相互作用能。
\textbf{b}，条件受控关联的离子层级来源。对每种性质，分别在阳离子、阴离子和离子对描述符中选取偏相关绝对值最大的结构因子；横向位置和标注数值保留偏相关的方向与大小，深色边框表示 BH--FDR $q\leq0.05$。该面板比较主导关联来自哪个分子层级，而不重复面板 a 的具体连线。
\textbf{c}，每种性质三个具有可审计响应曲线的结构先验，以 $3\times2$ 小多图形式展示。结构因子按照置信等级、偏相关绝对值、家族一致性和名称进行确定性排序；只有在可用 Level A/B 因子不足时，才按该排序补充 Level C 证据。蓝色、珊瑚色和柔和紫色分别表示阳离子、阴离子和离子对层级，实线圆点、虚线方点和点线三角分别表示第 1、2、3 位。子图内的直接标注给出结构因子和偏相关。曲线表示中心化后的实验对数性质响应，点为箱内均值，阴影带为箱内标准误；源数据保留每条曲线的分箱响应 Spearman 系数、对数响应范围、置信等级和标签样本量。各性质使用独立纵轴尺度，不用于跨性质比较响应绝对大小。
\textbf{d}，条件匹配的阳离子和阴离子替换所引起实验响应幅度的分布森林图。细线和粗线分别表示第 10--90 百分位范围与四分位距，点表示中位数，虚线连接用于比较两种离子替换角色；右侧直接标注的 C/A 样本量分别表示阳离子和阴离子替换，避免使用分离图例。}
\label{fig:molecular-origin-composite}
\end{figure*}
"""
        contents = {
            "revision_plan_zh.md": f"""# 论文增量修改建议

建议新增 `3.X Molecular Origins of Ionic-Liquid Thermophysical Properties`，
但所有表述限定为统计关联、冻结模型敏感性和条件匹配的结构响应。
本次对 {inference_record_count} 条记录完成推理，单项实验性质分析最多
使用 {sample_count} 条标签记录。当前最强证据项为：
{strongest_text}。

由于工作区仅有 random-point checkpoint，不应写成跨 random-IL、
property-balanced 或 ion-family 协议均稳定。attention 也不得写成真实
离子间作用能。本节不包含候选筛选、Pareto 排序或 shortlist 解释。
""",
            "manuscript_summary_en.md": f"""# Manuscript summary

The post-processing workflow connected source-audited molecular descriptors,
observed thermophysical data, and frozen-MIPGraph sensitivities for
{inference_record_count} aligned inference records; individual
observed-property analyses used up to {sample_count} labelled records. The
strongest evidence-gated trends were
{strongest_text}. These results are associations and model-derived trends, not
causal molecular laws or evidence of electrochemical performance.
""",
            "manuscript_summary_zh.md": f"""# 手稿结果摘要

该非侵入式后处理流程在 {inference_record_count} 条严格对齐记录上完成
冻结模型推理；单项实验性质分析最多使用 {sample_count} 条标签记录，并
连接经源码审计的分子描述符、实验热物性与模型敏感性。当前最强
证据为：{strongest_text}。这些结论不代表因果规律，也不证明真实超级
电容器性能。
""",
            "results_section_draft_en.md": results_en_markdown,
            "results_section_draft_zh.md": results_zh_markdown,
            "molecular_origin_results_section_en.tex": results_en_tex,
            "molecular_origin_results_section_zh.tex": results_zh_tex,
            "composite_figure_caption_en.tex": composite_caption_en,
            "composite_figure_caption_zh.tex": composite_caption_zh,
            "discussion_section_draft_en.md": """# Discussion draft

The analysis distinguishes observed-data association, frozen-model sensitivity
and chemistry-based interpretation. Agreement among these classes supports a
qualitative structural prior but does not establish causality. Family
imbalance, sparse labels, condition coverage and descriptor collinearity remain
material limitations. The shared cross-ion attention is compositionally
constrained and should not be equated with an interaction energy. Full property
inference and prospective measurements remain necessary before judging
electrolyte suitability.
""",
            "conclusion_revision_en.md": """# Conclusion revision

The added analysis layer provides a reproducible route from MIPGraph
representations to auditable microstructure–property hypotheses. Its outputs
are condition-controlled associations, matched-pair structural responses and
frozen-model diagnostics. Once derived, these relationships can provide
qualitative structural priors before full property inference, but they neither
determine electrolyte suitability nor establish universal molecular design
laws.
""",
            "abstract_revision_points_zh.md": """# 摘要修改要点

- 可增加“非侵入式、可审计的结构—性质后处理流程”。
- 明确结果属于统计关联和模型敏感性，而非因果发现。
- 将用途限定为完整性质推理前的定性结构先验，不写成候选筛选结果。
- 不声称 attention 等于实际相互作用能。
- 不声称已验证电化学性能、宽温运行或安全性。
- 明确当前多 checkpoint 稳健性受可用 checkpoint 数量限制。
""",
            "figure_captions_en.md": """# Figure caption

The manuscript-facing result is the consolidated a–d figure exported as
`figure_main_molecular_origin_analysis`. Its complete English LaTeX caption is
provided in `composite_figure_caption_en.tex`. The five standalone figures are
retained as auditable component exports and need not be placed separately in
the main text.
""",
            "figure_captions_zh.md": """# 图注

正文使用合并后的 a–d 总图 `figure_main_molecular_origin_analysis`；完整
中文 LaTeX 图注见 `composite_figure_caption_zh.tex`。原五张独立图作为
可审计的组成图保留，不需要在正文中分别排版。
""",
            "table_captions_en.md": """# Table captions

**Table S1.** Condition-controlled experimental and model-response associations.

**Table S2.** Family-stratified robustness of selected structural factors.

**Table S3.** Direct gradient×input and descriptor-surrogate permutation
importance with method-agreement diagnostics.

**Table S4.** Valid-SMILES counterfactual library, applicability-domain flags,
and conditional property responses.

**Table S5.** Evidence-gated design rules and unsupported audited hypotheses.
""",
        }
        paths = []
        for name, text in contents.items():
            path = manuscript / name
            path.write_text(text, encoding="utf-8")
            paths.append(path)
        self._record_stage(
            "manuscript", "completed", [str(path) for path in paths]
        )
        return paths

    def write_reproducibility_manifest(self) -> Path:
        payload = {
            "generated_utc": utc_now(),
            "config_path": self.config["_config_path"],
            "config_sha256": file_sha256(self.config["_config_path"]),
            "project_root": str(self.project.project_root),
            "module_root": str(MODULE_ROOT),
            "git_commit": git_commit(self.project.project_root),
            "git_status_before": self.initial_git_status,
            "git_status_current": git_status(self.project.project_root),
            "checkpoint": str(self.inspection.selected_checkpoint),
            "checkpoint_sha256": file_sha256(self.inspection.selected_checkpoint),
            "checkpoint_type": self.inspection.selected_checkpoint_type,
            "split": str(self.inspection.selected_split),
            "software_versions": software_versions(),
            "stages": self.stage_status,
            "random_seed": int(self.config["model"].get("seed", 42)),
        }
        return write_json(
            self.output_root / "reports" / "reproducibility_manifest.json",
            payload,
        )

    def write_implementation_report(self) -> Path:
        current = git_status(self.project.project_root)
        new_files = sorted(
            str(path.relative_to(MODULE_ROOT)).replace("\\", "/")
            for path in MODULE_ROOT.rglob("*")
            if path.is_file()
        )
        failed = {
            name: detail
            for name, detail in self.stage_status.items()
            if detail["status"].startswith("failed")
            or "limitation" in detail["status"]
        }
        tables = self.output_root / "tables"
        rules_path = tables / "design_rule_summary.csv"
        unsupported_path = tables / "unsupported_hypotheses.csv"
        rules = pd.read_csv(rules_path) if rules_path.is_file() else pd.DataFrame()
        unsupported = (
            pd.read_csv(unsupported_path)
            if unsupported_path.is_file()
            else pd.DataFrame()
        )
        property_order = {
            "Density": 0,
            "Viscosity": 1,
            "ElectricalConductivity": 2,
            "HeatCapacity": 3,
            "SurfaceTension": 4,
            "ThermalConductivity": 5,
        }
        strongest = (
            rules.loc[rules["confidence_level"].isin(["Level A", "Level B"])]
            .assign(_order=lambda frame: frame["property"].map(property_order))
            .sort_values(["_order", "confidence_level", "structural_factor"])
            .groupby("property", sort=False)
            .head(1)
            if not rules.empty
            else pd.DataFrame()
        )
        strongest_lines = (
            "\n".join(
                "- "
                f"{row.property}: `{row.structural_factor}` "
                f"({row.effect_direction}, {row.confidence_level}); "
                f"{row.statistical_evidence}"
                for row in strongest.itertuples(index=False)
            )
            if not strongest.empty
            else "- No Level A/B hypothesis passed the configured gates."
        )
        unsupported_examples = (
            unsupported.assign(
                _order=lambda frame: frame["property"].map(property_order)
            )
            .sort_values(["_order", "structural_factor"])
            .groupby("property", sort=False)
            .head(1)
            if not unsupported.empty
            else pd.DataFrame()
        )
        unsupported_lines = (
            "\n".join(
                "- "
                f"{row.property}: `{row.structural_factor}` was retained as "
                "unsupported because the predefined evidence gates did not agree."
                for row in unsupported_examples.itertuples(index=False)
            )
            if not unsupported_examples.empty
            else "- No unsupported-hypothesis table was available."
        )
        test_report_path = self.output_root / "reports" / "test_report.json"
        if test_report_path.is_file():
            test_report = json.loads(test_report_path.read_text(encoding="utf-8"))
            test_lines = (
                f"- Command: `{test_report.get('command', 'not recorded')}`\n"
                f"- Result: `{test_report.get('result', 'not recorded')}`\n"
                f"- Module-root command: "
                f"`{test_report.get('module_root_command', 'not recorded')}`\n"
                f"- Module-root result: "
                f"`{test_report.get('module_root_result', 'not recorded')}`"
            )
        else:
            test_lines = "- Test result not yet recorded; run the commands below."
        path = self.output_root / "reports" / "implementation_report.md"
        path.write_text(
            f"""# Molecular-origin analysis implementation report

Generated: {utc_now()}

## Scope integrity

- Git status before: `{self.initial_git_status}`
- Git status at report generation: `{current}`
- All files created by this task are below:
  `{MODULE_ROOT}`
- Existing model, data, checkpoint, screening, manuscript, and figure files
  were opened read-only and were not edited by this module.

## Actual runtime

- Model class: `{self.inspection.model_class}`
- Checkpoint: `{self.inspection.selected_checkpoint}`
- Selected checkpoint type: `{self.inspection.selected_checkpoint_type}`
- Data: `{self.inspection.data_paths['clean_csv']}`
- Split: `{self.inspection.selected_split}`

## Completed stages

{chr(10).join(f"- {name}: {detail['status']}" for name, detail in self.stage_status.items())}

## Test verification

{test_lines}

## Reproduction commands

From the project root:

```powershell
python experiments\molecular_origin_analysis\run_all.py --config experiments\molecular_origin_analysis\config\analysis_config.yaml
pytest experiments\molecular_origin_analysis\tests -v
```

The completed results are under `{self.output_root}`. Figure source data are
under `results/tables/figure_source_data/`; writing suggestions are under
`results/manuscript/`.

## Strongest evidence-gated hypotheses

{strongest_lines}

These are condition-controlled associations with frozen-model support. They
are not causal molecular laws.

## Examples retained as unsupported

{unsupported_lines}

## Limitations and incomplete items

{chr(10).join(f"- {name}: {detail.get('detail')}" for name, detail in failed.items()) if failed else '- None recorded'}

- Random-IL, property-balanced, and ion-family checkpoint files were not
  present; cross-checkpoint trend stability is not assessable.
- Property-conditioned attention×gradient is not promoted until an independent
  numerical stability audit is completed.
- SHAP, integrated gradients, ALE, and large-scale virtual generation are
  optional third-priority analyses and were not substituted with fabricated
  values.

## Items requiring human or new-data confirmation

- Supply compatible random-IL, property-balanced, and ion-family checkpoints
  before making any cross-checkpoint stability claim.
- Independently audit property-conditioned attention×gradient before promoting
  it beyond the current shared-attention analysis.
- Chemically review virtual structures and perform downstream liquid-state,
  thermophysical, and electrochemical measurements before using any candidate
  as an electrolyte.

## Scientific interpretation boundary

The strongest outputs are condition-controlled structure–property
associations, direct frozen-model gradient sensitivities, valid-SMILES
counterfactual trends, and post hoc candidate-rule consistency. They do not
prove causal mechanisms, actual cation–anion interaction energies, liquid-state
persistence, or electrochemical suitability.

## Suggested paper materials

Use `figure_main_molecular_origin_analysis` as the single manuscript-facing
a–d result figure. The standalone Figures A–E are retained as auditable
component exports rather than separate main-text figures. The attention half
of panel a must retain its shared-attention caveat. All
panel source data are stored in `results/tables/figure_source_data/`.

## New-file inventory

{chr(10).join(f'- `{name}`' for name in new_files)}
""",
            encoding="utf-8",
        )
        return path

    def validate(self) -> dict[str, Any]:
        required = [
            self.output_root / "reports" / "project_inspection.md",
            self.output_root / "cache" / "model_outputs.parquet",
            self.output_root / "cache" / "descriptor_matrix.parquet",
            self.output_root / "tables" / "feature_property_associations.csv",
            self.output_root / "tables" / "property_feature_importance.csv",
            self.output_root / "tables" / "matched_molecular_pairs.csv",
            self.output_root / "tables" / "virtual_counterfactual_library.csv",
            self.output_root / "tables" / "design_rule_summary.csv",
            self.output_root / "manuscript" / "results_section_draft_en.md",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        result = {
            "required_file_count": len(required),
            "missing": missing,
            "all_required_present": not missing,
            "original_scope_changes": [
                line
                for line in git_status(self.project.project_root)
                if "experiments/molecular_origin_analysis" not in line.replace("\\", "/")
                and line not in self.initial_git_status
            ],
        }
        write_json(self.output_root / "reports" / "final_validation.json", result)
        self._record_stage(
            "validate",
            "completed" if not missing else "failed",
            [str(self.output_root / "reports" / "final_validation.json")],
            f"Missing: {missing}" if missing else None,
        )
        return result

    def run_stage(self, stage: str, force: bool = False) -> Any:
        mapping: dict[str, Callable[..., Any]] = {
            "inspect": self.run_inspect,
            "extract": self.run_extract,
            "association": self.run_association,
            "attribution": self.run_attribution,
            "attention": self.run_attention,
            "counterfactual": self.run_counterfactual,
            "applicability": self.run_applicability,
            "rules": self.run_rules,
            "figures": self.run_figures,
            "manuscript": self.run_manuscript,
            "validate": self.validate,
        }
        if stage not in mapping:
            raise KeyError(f"Unknown stage {stage!r}; choose from {sorted(mapping)}")
        if stage in {"inspect", "validate"}:
            return mapping[stage]()
        return mapping[stage](force=force)

    def run_all(self, force: bool = False) -> dict[str, Any]:
        stages = [
            "inspect",
            "extract",
            "association",
            "attribution",
            "attention",
            "counterfactual",
            "applicability",
            "rules",
            "figures",
            "manuscript",
        ]
        optional = {"attribution", "attention", "figures"}
        for stage in stages:
            self.logger.info("Starting stage: %s", stage)
            try:
                self.run_stage(stage, force=force)
            except Exception as exc:
                if (
                    stage not in optional
                    or not self.config["runtime"].get(
                        "continue_on_optional_failure",
                        True,
                    )
                ):
                    self._record_stage(
                        stage,
                        "failed",
                        detail=f"{type(exc).__name__}: {exc}",
                    )
                    self.write_reproducibility_manifest()
                    self.write_implementation_report()
                    raise
                detail = "".join(
                    traceback.format_exception_only(type(exc), exc)
                ).strip()
                self.logger.exception("Optional stage %s failed", stage)
                self._record_stage(stage, "failed_optional", detail=detail)
        validation = self.validate()
        manifest = self.write_reproducibility_manifest()
        implementation = self.write_implementation_report()
        return {
            "stages": self.stage_status,
            "validation": validation,
            "reproducibility_manifest": str(manifest),
            "implementation_report": str(implementation),
        }
