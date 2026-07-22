"""Stepwise, resumable computational application-case pipeline."""

from __future__ import annotations

import json
import hashlib
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import torch

from .applicability_domain import assess_applicability_domain
from .chemistry import (
    CandidateGenerationSettings,
    build_candidate_tables,
    ion_family,
    parse_monovalent_pair,
)
from .config import temperature_grid
from .counterfactuals import analyze_counterfactual_substitutions
from .io_utils import (
    prepare_output_tree,
    validate_columns,
    write_csv,
    write_json,
    write_step_marker,
)
from .model_adapter import MIPGraphModelAdapter, PROPERTY_UNITS
from .paths import resolve_project_path
from .proxies import compute_application_proxies, summarize_whole_temperature_window
from .reference_cell import simulate_reference_cell_scenario
from .screening import (
    audit_curve_quality,
    curve_counts,
    derive_reference_thresholds,
    prioritize_candidates,
    screen_candidates,
)
from .uncertainty import (
    estimate_ensemble_decision_probabilities,
    estimate_uncertainty,
    validate_ensemble_compatibility,
)


LOGGER = logging.getLogger("computational_application_case")

STEP_ORDER = [
    "repository_audit",
    "unit_audit",
    "candidate_generation",
    "inference",
    "proxies",
    "reference_cell",
    "curve_quality",
    "applicability_domain",
    "uncertainty",
    "screening",
    "pareto",
    "counterfactuals",
    "figures",
    "tables",
    "report",
]


def _safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class CasePipeline:
    """Execute named application steps with output markers and schema checks."""

    def __init__(
        self,
        config: dict[str, Any],
        force: bool = False,
        resume: bool = False,
        skip_figures: bool = False,
        skip_report: bool = False,
    ) -> None:
        self.config = config
        self.root = Path(config["_project_root"])
        self.paths = prepare_output_tree(config["_output_dir"])
        self.force = bool(force)
        self.resume = bool(resume)
        self.skip_figures = bool(skip_figures)
        self.skip_report = bool(skip_report)
        self._adapter: MIPGraphModelAdapter | None = None
        self.run_fingerprint = self._compute_run_fingerprint()
        self.step_functions: dict[str, Callable[[], dict[str, Any]]] = {
            "repository_audit": self.repository_audit,
            "unit_audit": self.unit_audit,
            "candidate_generation": self.candidate_generation,
            "inference": self.inference,
            "proxies": self.proxies,
            "reference_cell": self.reference_cell,
            "curve_quality": self.curve_quality,
            "applicability_domain": self.applicability_domain,
            "uncertainty": self.uncertainty,
            "screening": self.screening,
            "pareto": self.pareto,
            "counterfactuals": self.counterfactuals,
            "figures": self.figures,
            "tables": self.tables,
            "report": self.report,
        }

    def marker(self, step: str) -> Path:
        return self.paths["steps"] / f"{step}.json"

    def _compute_run_fingerprint(self) -> str:
        """Fingerprint configuration, checkpoint identities, and case source code."""

        digest = hashlib.sha256(
            json.dumps(self.config, sort_keys=True, default=str).encode("utf-8")
        )
        checkpoint_values = list(self.config["model"].get("checkpoint_paths", [])) or [
            self.config["model"]["checkpoint_path"]
        ]
        input_values = checkpoint_values + [
            self.config["model"]["config_path"],
            self.config["model"]["graph_cache_path"],
            self.config["model"]["unimol2_feature_cache_path"],
            self.config["data"]["benchmark_path"],
            self.config["data"]["arrays_path"],
            self.config["data"]["split_path"],
        ]
        for value in input_values:
            path = resolve_project_path(self.root, value)
            digest.update(str(path).encode("utf-8"))
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
        case_root = self.root / "experiments" / "computational_application_case"
        source_paths = sorted((case_root / "src").glob("*.py")) + [case_root / "run_all.py"]
        for path in source_paths:
            digest.update(path.relative_to(case_root).as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def _invalidate_downstream_markers(self, completed_step: str) -> None:
        """Invalidate later markers after an explicitly forced upstream rerun."""

        start = STEP_ORDER.index(completed_step) + 1
        for downstream in STEP_ORDER[start:]:
            marker = self.marker(downstream)
            if marker.exists():
                marker.unlink()
                LOGGER.info(
                    "[%s] invalidated downstream marker after forced %s rerun",
                    downstream,
                    completed_step,
                )

    def _required_artifacts(self, step: str) -> list[Path]:
        """Return output files whose presence makes a step resumable."""

        data = self.paths["data"]
        audit = self.paths["audit"]
        mapping = {
            "repository_audit": [audit / "repository_audit.json", audit / "repository_audit.md"],
            "unit_audit": [audit / "unit_audit.json", audit / "unit_audit.md"],
            "candidate_generation": [
                data / "cation_library.csv",
                data / "anion_library.csv",
                data / "observed_reference_library.csv",
                data / "candidate_library.csv",
                data / "candidate_generation_trace.csv",
                data / "candidate_generation_failures.csv",
            ],
            "inference": [
                data / "property_predictions_long.csv",
                data / "property_predictions_wide.csv",
                data / "model_features.csv",
                data / "inference_failures.csv",
                audit / "inference_pipeline.json",
                audit / "inference_pipeline.md",
                self.paths["cache"] / "candidate_graphs.pt",
            ] + (
                [data / "property_predictions_by_checkpoint.csv"]
                if len(self.config["model"].get("checkpoint_paths", []))
                >= int(self.config["uncertainty"]["ensemble_min_checkpoints"])
                else []
            ),
            "proxies": [
                data / "application_proxies_temperature.csv",
                data / "application_proxies_wide.csv",
            ] + (
                [data / "application_proxies_by_checkpoint.csv"]
                if len(self.config["model"].get("checkpoint_paths", []))
                >= int(self.config["uncertainty"]["ensemble_min_checkpoints"])
                else []
            ),
            "reference_cell": [
                data / "reference_cell_metrics_temperature.csv",
                data / "reference_cell_candidate_summary.csv",
                audit / "reference_cell_scenario.json",
            ] + (
                [data / "reference_cell_metrics_by_checkpoint.csv"]
                if len(self.config["model"].get("checkpoint_paths", []))
                >= int(self.config["uncertainty"]["ensemble_min_checkpoints"])
                else []
            ),
            "curve_quality": [
                data / "curve_quality_flags.csv",
                data / "candidate_robust_summary.csv",
            ],
            "applicability_domain": [
                data / "applicability_domain.csv",
                audit / "applicability_domain.json",
            ],
            "uncertainty": [
                data / "property_uncertainty.csv",
                data / "proxy_uncertainty.csv",
                data / "feasibility_probability.csv",
                audit / "uncertainty.json",
            ],
            "screening": [data / "reference_thresholds.json", data / "screening_trace.csv"],
            "pareto": [data / "pareto_candidates.csv", data / "final_prioritized_candidates.csv"],
            "counterfactuals": [
                data / "counterfactual_ion_substitutions.csv",
                data / "counterfactual_summary.csv",
            ],
            "tables": [
                self.paths["tables"] / "candidate_generation_summary.csv",
                self.paths["tables"] / "final_candidate_table.csv",
                self.paths["tables"] / "final_candidate_table.tex",
                self.paths["tables"] / "reference_electrolyte_summary.csv",
                self.paths["tables"] / "screening_thresholds.csv",
                self.paths["tables"] / "screening_thresholds.tex",
                self.paths["tables"] / "reference_cell_scenario_parameters.csv",
                self.paths["tables"] / "reference_cell_scenario_parameters.tex",
                self.paths["tables"] / "reference_cell_candidate_summary.csv",
                self.paths["tables"] / "reference_cell_candidate_summary.tex",
            ],
            "report": [
                self.paths["report"] / "computational_application_case_results.md",
                self.paths["report"] / "computational_application_case_results.tex",
                self.paths["report"] / "computational_application_case_summary.json",
            ],
        }
        if step == "figures":
            names = [
                "figure5_computational_application_case",
                "figure6_reference_cell_scenario",
            ]
            if bool(self.config["figures"].get("make_individual_panels", True)):
                names.extend(
                    [
                        "panel_a_workflow",
                        "panel_b_funnel",
                        "panel_c_properties",
                        "panel_d_proxies",
                        "panel_e_applicability_domain",
                        "panel_f_constraints",
                        "panel_g_pareto",
                        "panel_h_candidates",
                        "cell_panel_a_scenario",
                        "cell_panel_b_resistance",
                        "cell_panel_c_rc_time",
                        "cell_panel_d_joule_heating",
                        "cell_panel_e_steady_temperature_rise",
                        "cell_panel_f_transient_temperature_rise",
                        "cell_panel_g_temperature_retention",
                        "cell_panel_h_worst_temperature_risk",
                    ]
                )
            return [
                self.paths["figures"] / f"{name}.{extension}"
                for name in names
                for extension in self.config["figures"]["formats"]
            ]
        return mapping.get(step, [])

    def run(self, only_step: str | None = None) -> dict[str, Any]:
        """Run all steps or one explicitly selected step."""

        if only_step is not None and only_step not in STEP_ORDER:
            raise ValueError(f"Unknown step {only_step!r}; choose from {STEP_ORDER}")
        selected = [only_step] if only_step else STEP_ORDER
        results: dict[str, Any] = {}
        for step in selected:
            if step == "figures" and self.skip_figures:
                LOGGER.info("Skipping figure generation by request")
                continue
            if step == "report" and self.skip_report:
                LOGGER.info("Skipping report generation by request")
                continue
            marker = self.marker(step)
            if marker.exists() and not self.force:
                if self.resume:
                    marker_payload = json.loads(marker.read_text(encoding="utf-8"))
                    if marker_payload.get("run_fingerprint") != self.run_fingerprint:
                        raise RuntimeError(
                            f"Cannot resume {step}: configuration, checkpoint, or case code changed. Use --force."
                        )
                    missing_artifacts = [
                        path for path in self._required_artifacts(step) if not path.exists()
                    ]
                    if missing_artifacts:
                        raise FileNotFoundError(
                            f"Cannot resume {step}; marker exists but artifacts are missing: {missing_artifacts}"
                        )
                    LOGGER.info("[%s] already completed; resume skips it", step)
                    results[step] = marker_payload
                    continue
                raise FileExistsError(
                    f"Step {step} already completed at {marker}. Use --resume or --force."
                )
            LOGGER.info("[%s] starting", step)
            payload = self.step_functions[step]()
            marker_payload = {
                **payload,
                "run_fingerprint": self.run_fingerprint,
                "artifacts": [str(path) for path in self._required_artifacts(step)],
            }
            write_step_marker(self.paths["steps"], step, marker_payload)
            if self.force:
                self._invalidate_downstream_markers(step)
            results[step] = payload
            LOGGER.info("[%s] completed", step)
        return results

    def _data_path(self, name: str) -> Path:
        return self.paths["data"] / name

    def _benchmark(self) -> pd.DataFrame:
        path = resolve_project_path(self.root, self.config["data"]["benchmark_path"])
        frame = pd.read_csv(path)
        validate_columns(frame, ["IL_SMILES", "Temperature_K", "Pressure_kPa"], "benchmark")
        return frame

    def _split(self) -> dict[str, list[int]]:
        path = resolve_project_path(self.root, self.config["data"]["split_path"])
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return {key: [int(value) for value in values] for key, values in payload.items()}

    def repository_audit(self) -> dict[str, Any]:
        """Write a runtime artefact inventory without entering deprecated code."""

        static_audit = self.root / "experiments" / "computational_application_case" / "repository_audit.md"
        if not static_audit.exists():
            raise FileNotFoundError(f"Static repository audit is missing: {static_audit}")
        artefacts = {
            "benchmark": str(resolve_project_path(self.root, self.config["data"]["benchmark_path"])),
            "arrays": str(resolve_project_path(self.root, self.config["data"]["arrays_path"])),
            "split": str(resolve_project_path(self.root, self.config["data"]["split_path"])),
            "model_config": str(resolve_project_path(self.root, self.config["model"]["config_path"])),
            "checkpoint": str(resolve_project_path(self.root, self.config["model"]["checkpoint_path"])),
            "graph_cache": str(resolve_project_path(self.root, self.config["model"]["graph_cache_path"])),
            "unimol2_cache": str(resolve_project_path(self.root, self.config["model"]["unimol2_feature_cache_path"])),
            "deprecated_archive_used": False,
        }
        write_json(artefacts, self.paths["audit"] / "repository_audit.json")
        lines = [
            "# Runtime repository audit",
            "",
            "web/ is a deprecated implementation and was deliberately excluded from model loading, preprocessing, inference, and result generation.",
            "",
            *[f"- {key}: `{value}`" for key, value in artefacts.items()],
        ]
        (self.paths["audit"] / "repository_audit.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        return {"artefacts": artefacts, "static_audit": str(static_audit)}

    def unit_audit(self) -> dict[str, Any]:
        """Audit configured units against current dataset ranges and positivity."""

        benchmark = self._benchmark()
        audit: dict[str, Any] = {
            "property_order": list(PROPERTY_UNITS),
            "units": dict(PROPERTY_UNITS),
            "heat_capacity_basis": "molar",
            "unit_source": "current dataset-statistics exporter and source magnitudes",
            "properties": {},
            "passed": True,
        }
        for prop, unit in PROPERTY_UNITS.items():
            column = f"{prop}_ActualValue"
            if column not in benchmark:
                raise ValueError(f"Benchmark lacks unit-audit column {column}")
            values = pd.to_numeric(benchmark[column], errors="coerce")
            finite = values[np.isfinite(values)]
            positive = finite[finite > 0.0]
            audit["properties"][prop] = {
                "unit": unit,
                "finite_count": int(len(finite)),
                "nonpositive_count": int((finite <= 0.0).sum()),
                "minimum": float(finite.min()) if len(finite) else None,
                "maximum": float(finite.max()) if len(finite) else None,
                "positive_range_check": bool(len(finite) and len(positive) == len(finite)),
            }
            audit["passed"] = audit["passed"] and bool(
                audit["properties"][prop]["positive_range_check"]
            )
        write_json(audit, self.paths["audit"] / "unit_audit.json")
        lines = ["# Unit audit", "", f"Overall positivity audit: **{'PASS' if audit['passed'] else 'FAIL'}**", ""]
        for prop, item in audit["properties"].items():
            lines.append(
                f"- {prop}: {item['unit']}; n={item['finite_count']}; range={item['minimum']} to {item['maximum']}; non-positive={item['nonpositive_count']}."
            )
        lines.extend(
            [
                "",
                "HeatCapacity is molar (J mol^-1 K^-1) and is converted once using the complete RDKit ion-pair molar mass in kg mol^-1.",
            ]
        )
        (self.paths["audit"] / "unit_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        if not audit["passed"]:
            raise ValueError("Unit audit found non-positive training property values")
        return audit

    def candidate_generation(self) -> dict[str, Any]:
        """Extract supported ions and generate unseen pair recombinations."""

        benchmark = self._benchmark()
        split = self._split()
        name = str(self.config["data"]["training_split_name"])
        if name not in split:
            raise KeyError(f"Training split {name!r} is absent from split file")
        cfg = self.config["candidate_generation"]
        settings = CandidateGenerationSettings(
            min_cation_support=int(cfg["min_cation_support"]),
            min_anion_support=int(cfg["min_anion_support"]),
            max_cations=int(cfg["max_cations"]),
            max_anions=int(cfg["max_anions"]),
            max_candidates=int(cfg["max_candidates"]),
            max_observed_references=int(cfg["max_observed_references"]),
            stable_id_anchor_count=int(cfg.get("stable_id_anchor_count", 0)),
            require_monovalent_1to1=bool(cfg["require_monovalent_1to1"]),
            exclude_observed_pairs=bool(cfg["exclude_observed_pairs"]),
            descriptor_prefilter_multiplier=int(cfg["descriptor_prefilter_multiplier"]),
            random_seed=int(self.config["project"]["random_seed"]),
        )
        result = build_candidate_tables(benchmark, split[name], settings)
        if result.candidates.empty:
            raise RuntimeError("No unseen pair recombination survived candidate generation")
        observed = result.observed_references.copy()
        candidates = result.candidates.copy()
        union_columns = sorted(set(observed.columns) | set(candidates.columns))
        library = pd.concat(
            [observed.reindex(columns=union_columns), candidates.reindex(columns=union_columns)],
            ignore_index=True,
        )
        write_csv(result.cations, self._data_path("cation_library.csv"))
        write_csv(result.anions, self._data_path("anion_library.csv"))
        write_csv(observed, self._data_path("observed_reference_library.csv"))
        write_csv(library, self._data_path("candidate_library.csv"))
        write_csv(result.trace, self._data_path("candidate_generation_trace.csv"))
        write_csv(result.failures, self._data_path("candidate_generation_failures.csv"))
        theoretical = int(len(result.cations) * len(result.anions))
        trace_by_step = result.trace.set_index("step")
        return {
            "settings": asdict(settings),
            "initial_cations": int(len(result.cations)),
            "initial_anions": int(len(result.anions)),
            "theoretical_combinations": theoretical,
            "valid_parsed_rows": int(trace_by_step.loc["parse_benchmark", "retained_count"]),
            "valid_unique_pairs": int(trace_by_step.loc["canonical_pair_deduplication", "retained_count"]),
            "valid_unseen_pool": int(trace_by_step.loc["exclude_observed_pairs", "retained_count"]),
            "observed_references": int(len(observed)),
            "unseen_candidates": int(len(candidates)),
            "parse_failures": int(len(result.failures)),
        }

    def _get_adapter(self) -> MIPGraphModelAdapter:
        if self._adapter is None:
            self._adapter = MIPGraphModelAdapter(self.config)
        return self._adapter

    def inference(self) -> dict[str, Any]:
        """Run the current MIPGraph model on observed and unseen pairs."""

        library_path = self._data_path("candidate_library.csv")
        if not library_path.exists():
            raise FileNotFoundError("Candidate generation output is required before inference")
        library = pd.read_csv(library_path)
        validate_columns(
            library,
            ["candidate_id", "candidate_type", "cation_smiles", "anion_smiles", "il_smiles"],
            "candidate library",
        )
        main_temperatures = temperature_grid(self.config["conditions"])
        temperatures = main_temperatures
        if bool(self.config["conditions"].get("run_extended_sensitivity", False)):
            temperatures = np.unique(
                np.concatenate([temperatures, temperature_grid(self.config["conditions"], extended=True)])
            )
        benchmark = self._benchmark()
        training = benchmark.iloc[self._split()[self.config["data"]["training_split_name"]]]
        training_range = (
            float(pd.to_numeric(training["Temperature_K"], errors="coerce").min()),
            float(pd.to_numeric(training["Temperature_K"], errors="coerce").max()),
        )
        ensemble_paths = list(self.config["model"].get("checkpoint_paths", []))
        minimum_members = int(self.config["uncertainty"]["ensemble_min_checkpoints"])
        ensemble_enabled = len(ensemble_paths) >= minimum_members
        if ensemble_enabled:
            member_results = []
            for member_index, checkpoint_path in enumerate(ensemble_paths, start=1):
                adapter = MIPGraphModelAdapter(self.config, checkpoint_path=checkpoint_path)
                member = adapter.predict(
                    library,
                    temperatures,
                    float(self.config["conditions"]["pressure_kPa"]),
                    self.paths["cache"] / "candidate_graphs.pt",
                    training_range,
                    force=self.force and member_index == 1,
                )
                member_name = f"member_{member_index:02d}_{Path(checkpoint_path).name}"
                member.predictions["checkpoint_name"] = member_name
                member.predictions["analysis_window"] = np.where(
                    member.predictions["temperature_K"].isin(main_temperatures),
                    "main",
                    "extended_sensitivity",
                )
                member.failures["checkpoint_name"] = member_name
                member_results.append(member)
                del adapter
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            validate_ensemble_compatibility(
                [member.metadata for member in member_results]
            )
            by_checkpoint = pd.concat(
                [member.predictions for member in member_results], ignore_index=True
            )
            key_columns = ["candidate_id", "temperature_K", "pressure_kPa"]
            member_counts = by_checkpoint.groupby(key_columns)["checkpoint_name"].nunique()
            complete_keys = member_counts[member_counts.eq(len(ensemble_paths))].reset_index()[
                key_columns
            ]
            if complete_keys.empty:
                raise RuntimeError(
                    "No candidate-condition row has complete coverage across all ensemble checkpoints"
                )
            complete_members = by_checkpoint.merge(
                complete_keys, on=key_columns, how="inner", validate="many_to_one"
            )
            base_columns = [
                column
                for column in complete_members.columns
                if column not in list(PROPERTY_UNITS) + ["checkpoint_name"]
            ]
            base = complete_members.sort_values("checkpoint_name").drop_duplicates(
                key_columns
            )[base_columns]
            means = complete_members.groupby(key_columns, as_index=False)[
                list(PROPERTY_UNITS)
            ].mean()
            point_predictions = base.merge(
                means, on=key_columns, how="inner", validate="one_to_one"
            )
            point_predictions["checkpoint_name"] = f"ensemble_mean_{len(ensemble_paths)}"
            result = member_results[0]
            result.predictions = point_predictions.sort_values(
                ["candidate_id", "temperature_K"]
            ).reset_index(drop=True)
            result.predictions_wide = result.predictions.pivot(
                index=[
                    "candidate_id",
                    "candidate_type",
                    "cation_smiles",
                    "anion_smiles",
                    "il_smiles",
                ],
                columns="temperature_K",
                values=list(PROPERTY_UNITS),
            )
            result.predictions_wide.columns = [
                f"{name}_{temperature:g}K"
                for name, temperature in result.predictions_wide.columns
            ]
            result.predictions_wide = result.predictions_wide.reset_index()
            result.failures = pd.concat(
                [member.failures for member in member_results], ignore_index=True
            )
            result.metadata.update(
                {
                    "ensemble_enabled": True,
                    "checkpoint_paths": [
                        str(resolve_project_path(self.root, value))
                        for value in ensemble_paths
                    ],
                    "checkpoint_count": len(ensemble_paths),
                    "point_prediction": "arithmetic ensemble mean in physical units",
                    "ensemble_member_scalers": [
                        {
                            "checkpoint_path": member.metadata["checkpoint_path"],
                            "condition_scaler": member.metadata["condition_scaler"],
                            "target_means": member.metadata["target_means"],
                            "target_stds": member.metadata["target_stds"],
                        }
                        for member in member_results
                    ],
                }
            )
            write_csv(
                by_checkpoint,
                self._data_path("property_predictions_by_checkpoint.csv"),
            )
        else:
            result = self._get_adapter().predict(
                library,
                temperatures,
                float(self.config["conditions"]["pressure_kPa"]),
                self.paths["cache"] / "candidate_graphs.pt",
                training_range,
                force=self.force,
            )
            result.predictions["analysis_window"] = np.where(
                result.predictions["temperature_K"].isin(main_temperatures),
                "main",
                "extended_sensitivity",
            )
            result.metadata.update(
                {
                    "ensemble_enabled": False,
                    "checkpoint_paths": [str(result.checkpoint_path)]
                    if hasattr(result, "checkpoint_path")
                    else [result.metadata["checkpoint_path"]],
                    "checkpoint_count": 1,
                    "point_prediction": "single checkpoint",
                }
            )
        write_csv(result.predictions, self._data_path("property_predictions_long.csv"))
        write_csv(result.predictions_wide, self._data_path("property_predictions_wide.csv"))
        write_csv(result.features, self._data_path("model_features.csv"))
        write_csv(result.failures, self._data_path("inference_failures.csv"))
        write_json(result.metadata, self.paths["audit"] / "inference_pipeline.json")
        metadata_lines = [
            "# Inference pipeline",
            "",
            "- Imported model factory: `src.models.factory.build_model` from current `il_property_prediction`.",
            f"- Model class: `{result.metadata['model_class']}`.",
            f"- Checkpoint: `{result.metadata['checkpoint_path']}`.",
            f"- Checkpoint ensemble enabled: `{result.metadata['ensemble_enabled']}`; members: `{result.metadata['checkpoint_paths']}`.",
            f"- Ignored legacy dormant checkpoint keys: `{result.metadata['ignored_legacy_checkpoint_keys']}`.",
            f"- Property order: `{result.metadata['property_order']}`.",
            f"- Units: `{result.metadata['property_units']}`.",
            f"- Inverse transform: `{result.metadata['target_inverse_transform']}`.",
            f"- Uni-Mol2 cache: `{result.metadata['unimol2_feature_cache']}`.",
            f"- Application graph cache: `{result.metadata['application_graph_cache']}`.",
            "- Tensor schema: graph x[*,45], edge_attr[*,12], condition[batch,2], global_desc[batch,56], functional_group_desc[batch,80], prediction[batch,6].",
            "- Deprecated archive used: No.",
        ]
        (self.paths["audit"] / "inference_pipeline.md").write_text(
            "\n".join(metadata_lines) + "\n", encoding="utf-8"
        )
        expected = len(library) * len(temperatures)
        return {
            **result.metadata,
            "requested_candidate_conditions": int(expected),
            "main_temperature_points": int(len(main_temperatures)),
            "extended_sensitivity_points": int(len(temperatures) - len(main_temperatures)),
            "successful_predictions": int(len(result.predictions)),
            "successful_candidates": int(result.predictions["candidate_id"].nunique()),
            "successful_unseen_candidates": int(
                result.predictions.loc[
                    result.predictions["candidate_type"].eq("unseen_pair_recombination"),
                    "candidate_id",
                ].nunique()
            ),
            "inference_failures": int(len(result.failures)),
            "ensemble_member_prediction_rows": int(len(by_checkpoint))
            if ensemble_enabled
            else 0,
        }

    def proxies(self) -> dict[str, Any]:
        """Compute temperature-resolved application proxy mappings."""

        prediction_path = self._data_path("property_predictions_long.csv")
        if not prediction_path.exists():
            raise FileNotFoundError("Inference predictions are required before proxy calculation")
        predictions = pd.read_csv(prediction_path)
        proxies = compute_application_proxies(predictions, self.config["proxies"])
        write_csv(proxies, self._data_path("application_proxies_temperature.csv"))
        inference_metadata = json.loads(
            (self.paths["audit"] / "inference_pipeline.json").read_text(encoding="utf-8")
        )
        ensemble_proxy_rows = 0
        if bool(inference_metadata.get("ensemble_enabled", False)):
            member_predictions = pd.read_csv(
                self._data_path("property_predictions_by_checkpoint.csv")
            )
            member_proxy_frames = [
                compute_application_proxies(group.copy(), self.config["proxies"])
                for _, group in member_predictions.groupby("checkpoint_name", sort=True)
            ]
            member_proxies = pd.concat(member_proxy_frames, ignore_index=True)
            write_csv(
                member_proxies,
                self._data_path("application_proxies_by_checkpoint.csv"),
            )
            ensemble_proxy_rows = len(member_proxies)
        proxy_columns = [
            "cp_mass_J_kg-1_K-1",
            "volumetric_heat_capacity",
            "thermal_diffusivity",
            "simplified_thermal_diffusion_timescale",
            "electrolyte_mass_kg",
            "z_conductivity",
            "z_viscosity",
            "transport_favorability",
            "surface_tension_reference_envelope_deviation",
            "thermal_effusivity",
        ]
        wide = proxies.pivot(
            index=["candidate_id", "candidate_type", "cation_smiles", "anion_smiles", "il_smiles"],
            columns="temperature_K",
            values=proxy_columns,
        )
        wide.columns = [f"{name}_{temperature:g}K" for name, temperature in wide.columns]
        write_csv(wide.reset_index(), self._data_path("application_proxies_wide.csv"))
        return {
            "candidate_conditions": int(len(proxies)),
            "ensemble_member_proxy_rows": int(ensemble_proxy_rows),
            "proxy_warning_rows": int(proxies["proxy_warnings"].fillna("").ne("").sum()),
        }

    def reference_cell(self) -> dict[str, Any]:
        """Evaluate the explicit conditional reference-cell scenario."""

        proxy_path = self._data_path("application_proxies_temperature.csv")
        if not proxy_path.exists():
            raise FileNotFoundError(
                "Application proxies are required before reference-cell simulation"
            )
        proxies = pd.read_csv(proxy_path)
        metrics, summary, metadata = simulate_reference_cell_scenario(
            proxies, self.config["reference_cell"]
        )
        write_csv(metrics, self._data_path("reference_cell_metrics_temperature.csv"))
        write_csv(summary, self._data_path("reference_cell_candidate_summary.csv"))
        ensemble_rows = 0
        member_proxy_path = self._data_path("application_proxies_by_checkpoint.csv")
        inference_metadata = json.loads(
            (self.paths["audit"] / "inference_pipeline.json").read_text(
                encoding="utf-8"
            )
        )
        if bool(inference_metadata.get("ensemble_enabled", False)):
            if not member_proxy_path.exists():
                raise FileNotFoundError(
                    "Ensemble inference requires per-checkpoint proxy rows"
                )
            member_proxies = pd.read_csv(member_proxy_path)
            member_frames: list[pd.DataFrame] = []
            for checkpoint_name, group in member_proxies.groupby(
                "checkpoint_name", sort=True
            ):
                member_metrics, _, _ = simulate_reference_cell_scenario(
                    group.copy(), self.config["reference_cell"]
                )
                member_metrics["checkpoint_name"] = checkpoint_name
                member_frames.append(member_metrics)
            member_metrics = pd.concat(member_frames, ignore_index=True)
            write_csv(
                member_metrics,
                self._data_path("reference_cell_metrics_by_checkpoint.csv"),
            )
            ensemble_rows = len(member_metrics)
        metadata.update(
            {
                "candidate_condition_rows": int(len(metrics)),
                "candidate_summary_rows": int(len(summary)),
                "ensemble_member_rows": int(ensemble_rows),
                "exceedance_band_counts": {
                    str(key): int(value)
                    for key, value in summary[
                        "reference_cell_exceedance_band_worst"
                    ].value_counts().items()
                },
            }
        )
        write_json(metadata, self.paths["audit"] / "reference_cell_scenario.json")
        return metadata

    def curve_quality(self) -> dict[str, Any]:
        """Audit curves and build complete whole-window robust summaries."""

        proxy_path = self._data_path("application_proxies_temperature.csv")
        if not proxy_path.exists():
            raise FileNotFoundError("Proxy output is required before curve-quality audit")
        proxies = pd.read_csv(proxy_path)
        if "analysis_window" not in proxies:
            proxies["analysis_window"] = "main"
        main_proxies = proxies[proxies["analysis_window"].eq("main")].copy()
        sensitivity_proxies = proxies[
            proxies["analysis_window"].eq("extended_sensitivity")
        ].copy()
        main_flags = audit_curve_quality(
            main_proxies, self._benchmark(), self.config["curve_quality"]
        )
        main_flags["analysis_window"] = "main"
        sensitivity_flags = audit_curve_quality(
            sensitivity_proxies, self._benchmark(), self.config["curve_quality"]
        ) if not sensitivity_proxies.empty else main_flags.iloc[0:0].copy()
        if not sensitivity_flags.empty:
            sensitivity_flags["analysis_window"] = "extended_sensitivity"
        flags = pd.concat([main_flags, sensitivity_flags], ignore_index=True)
        counts = curve_counts(main_flags)
        summary = summarize_whole_temperature_window(main_proxies)
        scenario_summary_path = self._data_path("reference_cell_candidate_summary.csv")
        if not scenario_summary_path.exists():
            raise FileNotFoundError(
                "Reference-cell scenario output is required before curve-quality audit"
            )
        scenario_summary = pd.read_csv(scenario_summary_path).drop(
            columns=["candidate_type"], errors="ignore"
        )
        summary = summary.merge(
            scenario_summary, on="candidate_id", how="left", validate="one_to_one"
        )
        summary = summary.merge(counts, on="candidate_id", how="left", suffixes=("", "_audit"))
        for column in ["curve_warning_count", "severe_curve_failure_count"]:
            audit_column = f"{column}_audit"
            if audit_column in summary:
                summary[column] = summary[audit_column].fillna(summary.get(column, 0)).fillna(0).astype(int)
                summary = summary.drop(columns=audit_column)
            else:
                summary[column] = summary.get(column, 0)
        write_csv(flags, self._data_path("curve_quality_flags.csv"))
        write_csv(summary, self._data_path("candidate_robust_summary.csv"))
        return {
            "flags": int(len(flags)),
            "warning_flags": int(flags["severity"].eq("warning").sum()) if not flags.empty else 0,
            "severe_flags": int(flags["severity"].eq("severe").sum()) if not flags.empty else 0,
            "summarized_candidates": int(len(summary)),
        }

    def _reference_descriptor_frame(self, descriptor_columns: Sequence[str]) -> pd.DataFrame:
        benchmark = self._benchmark()
        train_indices = self._split()[self.config["data"]["training_split_name"]]
        training_smiles = benchmark.iloc[train_indices]["IL_SMILES"].dropna().astype(str).unique()
        graph_cache_path = resolve_project_path(self.root, self.config["model"]["graph_cache_path"])
        graph_cache = _safe_torch_load(graph_cache_path)
        rows = []
        for smiles in sorted(training_smiles):
            graph = graph_cache.get(smiles)
            if graph is None:
                continue
            values = np.concatenate(
                [
                    graph.global_desc.detach().cpu().numpy().reshape(-1),
                    graph.functional_group_desc.detach().cpu().numpy().reshape(-1),
                ]
            )
            row = {"reference_id": smiles}
            row.update({column: float(value) for column, value in zip(descriptor_columns, values)})
            rows.append(row)
        reference = pd.DataFrame(rows)
        if len(reference) < 2:
            raise RuntimeError("Fewer than two training-domain descriptor references were recovered")
        return reference

    def _ad_metadata(self, candidates: pd.DataFrame) -> pd.DataFrame:
        benchmark = self._benchmark()
        training = benchmark.iloc[self._split()[self.config["data"]["training_split_name"]]].copy()
        unique_smiles = training["IL_SMILES"].dropna().astype(str).unique()
        parsed = {}
        for smiles in unique_smiles:
            try:
                parsed[smiles] = parse_monovalent_pair(smiles)
            except ValueError:
                continue
        training["_cation"] = training["IL_SMILES"].map(
            lambda value: parsed.get(str(value)).cation_identity_key if str(value) in parsed else None
        )
        training["_anion"] = training["IL_SMILES"].map(
            lambda value: parsed.get(str(value)).anion_identity_key if str(value) in parsed else None
        )
        training_pairs = (
            training.dropna(subset=["_cation", "_anion"])
            .drop_duplicates(["_cation", "_anion"])
            .copy()
        )
        training_pair_keys = set(
            training_pairs["_cation"].astype(str)
            + "||"
            + training_pairs["_anion"].astype(str)
        )
        cation_support = training_pairs.groupby("_cation").size()
        anion_support = training_pairs.groupby("_anion").size()
        cation_family_map = {
            pair.cation_identity_key: ion_family(pair.cation_smiles, "cation")
            for pair in parsed.values()
        }
        anion_family_map = {
            pair.anion_identity_key: ion_family(pair.anion_smiles, "anion")
            for pair in parsed.values()
        }
        training["_cation_family"] = training["_cation"].map(cation_family_map)
        training["_anion_family"] = training["_anion"].map(anion_family_map)
        cation_family_support = training.groupby("_cation_family")["_cation"].nunique()
        anion_family_support = training.groupby("_anion_family")["_anion"].nunique()
        property_columns = [f"{name}_ActualValue" for name in PROPERTY_UNITS]
        training["_label_count"] = training[property_columns].notna().sum(axis=1)
        metadata_rows = []
        temperatures = temperature_grid(self.config["conditions"])
        for row in candidates.itertuples(index=False):
            cation_rows = training[training["_cation"].eq(row.cation_identity_key)]
            anion_rows = training[training["_anion"].eq(row.anion_identity_key)]
            component_min = max(
                float(cation_rows["Temperature_K"].min()) if not cation_rows.empty else float("inf"),
                float(anion_rows["Temperature_K"].min()) if not anion_rows.empty else float("inf"),
            )
            component_max = min(
                float(cation_rows["Temperature_K"].max()) if not cation_rows.empty else -float("inf"),
                float(anion_rows["Temperature_K"].max()) if not anion_rows.empty else -float("inf"),
            )
            temperature_status = (
                "in_domain"
                if component_min <= float(np.min(temperatures))
                and component_max >= float(np.max(temperatures))
                else "extrapolation"
            )
            metadata_rows.append(
                {
                    "candidate_id": row.candidate_id,
                    "candidate_type": row.candidate_type,
                    "cation_seen": not cation_rows.empty,
                    "anion_seen": not anion_rows.empty,
                    "pair_seen": (
                        f"{row.cation_identity_key}||{row.anion_identity_key}"
                        in training_pair_keys
                    ),
                    "cation_support_count": int(
                        cation_support.get(str(row.cation_identity_key), 0)
                    ),
                    "anion_support_count": int(
                        anion_support.get(str(row.anion_identity_key), 0)
                    ),
                    "cation_family_support": int(
                        cation_family_support.get(str(row.cation_family), 0)
                    ),
                    "anion_family_support": int(
                        anion_family_support.get(str(row.anion_family), 0)
                    ),
                    "property_support_count": int(
                        min(cation_rows["_label_count"].sum(), anion_rows["_label_count"].sum())
                    ),
                    "temperature_support_min_K": component_min,
                    "temperature_support_max_K": component_max,
                    "temperature_domain_status": temperature_status,
                }
            )
        return pd.DataFrame(metadata_rows)

    def applicability_domain(self) -> dict[str, Any]:
        """Fit reference-only descriptor AD and score all predicted candidates."""

        feature_path = self._data_path("model_features.csv")
        library_path = self._data_path("candidate_library.csv")
        if not feature_path.exists() or not library_path.exists():
            raise FileNotFoundError("Inference features and candidate library are required for AD")
        features = pd.read_csv(feature_path)
        candidates = pd.read_csv(library_path)
        descriptor_columns = [
            column
            for column in features.columns
            if column.startswith("global_desc_") or column.startswith("fg_desc_")
        ]
        if len(descriptor_columns) != 136:
            raise ValueError(f"Expected 136 current descriptors, found {len(descriptor_columns)}")
        reference = self._reference_descriptor_frame(descriptor_columns)
        write_csv(
            reference,
            self._data_path("training_domain_descriptor_reference.csv"),
        )
        metadata = self._ad_metadata(candidates)
        ad, model = assess_applicability_domain(
            features,
            reference,
            metadata,
            descriptor_columns,
            self.config["applicability_domain"],
        )
        write_csv(ad, self._data_path("applicability_domain.csv"))
        write_json(
            {
                "reference_domain": self.config["applicability_domain"]["reference_domain"],
                "reference_count": len(reference),
                "descriptor_columns_used": int(model.kept_columns.sum()),
                "constant_descriptor_columns_removed": int((~model.kept_columns).sum()),
                "k": model.k,
                "in_domain_threshold": model.in_domain_threshold,
                "borderline_threshold": model.borderline_threshold,
                "embedding_distance": "not_available",
            },
            self.paths["audit"] / "applicability_domain.json",
        )
        counts = ad["AD_status"].value_counts().to_dict()
        return {"reference_count": len(reference), **{str(key): int(value) for key, value in counts.items()}}

    def uncertainty(self) -> dict[str, Any]:
        """Select the strongest supported uncertainty mode and persist all tables."""

        predictions = pd.read_csv(self._data_path("property_predictions_long.csv"))
        proxies = pd.read_csv(self._data_path("application_proxies_temperature.csv"))
        inference_metadata = json.loads(
            (self.paths["audit"] / "inference_pipeline.json").read_text(encoding="utf-8")
        )
        paths = list(self.config["model"].get("checkpoint_paths", []))
        if not paths and self.config["model"].get("checkpoint_path"):
            paths = [self.config["model"]["checkpoint_path"]]
        ensemble_enabled = bool(inference_metadata.get("ensemble_enabled", False))
        uncertainty_predictions = (
            pd.read_csv(self._data_path("property_predictions_by_checkpoint.csv"))
            if ensemble_enabled
            else predictions
        )
        uncertainty_proxies = (
            pd.read_csv(self._data_path("application_proxies_by_checkpoint.csv"))
            if ensemble_enabled
            else proxies
        )
        if ensemble_enabled:
            member_cell = pd.read_csv(
                self._data_path("reference_cell_metrics_by_checkpoint.csv")
            )
            keys = [
                "candidate_id",
                "temperature_K",
                "pressure_kPa",
                "checkpoint_name",
            ]
            scenario_columns = [
                "electrolyte_resistance_ohm",
                "joule_heating_power_W",
                "steady_state_temperature_rise_K",
                "transient_temperature_rise_K",
                "reference_cell_exceedance_index",
            ]
            uncertainty_proxies = uncertainty_proxies.merge(
                member_cell[keys + scenario_columns],
                on=keys,
                how="left",
                validate="one_to_one",
            )
        property_table, proxy_table, feasibility, status = estimate_uncertainty(
            uncertainty_predictions,
            uncertainty_proxies,
            paths,
            self.config["uncertainty"],
        )
        if status["uncertainty_status"] == "checkpoint_ensemble":
            robust = pd.read_csv(self._data_path("candidate_robust_summary.csv"))
            ad = pd.read_csv(self._data_path("applicability_domain.csv"))
            library = pd.read_csv(self._data_path("candidate_library.csv"))
            fixed_thresholds = derive_reference_thresholds(
                robust, self.config["screening"]
            )
            feasibility = estimate_ensemble_decision_probabilities(
                uncertainty_proxies,
                self._benchmark(),
                ad,
                library,
                fixed_thresholds,
                self.config["curve_quality"],
                self.config["screening"],
                self.config["pareto"],
                self.config["reference_cell"],
            )
            status["decision_probability_status"] = (
                "full_window_constraints_and_pareto_propagated"
            )
            status["fixed_thresholds"] = fixed_thresholds
        write_csv(property_table, self._data_path("property_uncertainty.csv"))
        write_csv(proxy_table, self._data_path("proxy_uncertainty.csv"))
        write_csv(feasibility, self._data_path("feasibility_probability.csv"))
        write_json(status, self.paths["audit"] / "uncertainty.json")
        return status

    def screening(self) -> dict[str, Any]:
        """Freeze observed-reference thresholds and apply all hard constraints."""

        robust = pd.read_csv(self._data_path("candidate_robust_summary.csv"))
        ad = pd.read_csv(self._data_path("applicability_domain.csv"))
        library = pd.read_csv(self._data_path("candidate_library.csv"))
        thresholds = derive_reference_thresholds(robust, self.config["screening"])
        trace = screen_candidates(robust, ad, library, thresholds, self.config["screening"])
        write_json(thresholds, self._data_path("reference_thresholds.json"))
        write_csv(trace, self._data_path("screening_trace.csv"))
        all_pass = int(trace["final_feasible"].sum())
        unseen_pass = int(
            trace.loc[
                trace["candidate_type"].eq("unseen_pair_recombination"),
                "final_feasible",
            ].sum()
        )
        return {
            "thresholds": thresholds,
            "screened": int(len(trace)),
            "hard_constraint_pass": unseen_pass,
            "hard_constraint_pass_unseen": unseen_pass,
            "hard_constraint_pass_all": all_pass,
        }

    def pareto(self) -> dict[str, Any]:
        """Perform multi-objective non-dominated sorting and prioritization."""

        trace = pd.read_csv(self._data_path("screening_trace.csv"))
        ranked, final = prioritize_candidates(trace, self.config["pareto"])
        write_csv(ranked, self._data_path("pareto_candidates.csv"))
        write_csv(final, self._data_path("final_prioritized_candidates.csv"))
        rank_one_record = ranked[ranked["Pareto_rank"].eq(1)].sort_values(
            [
                "utopia_distance",
                "descriptor_knn_distance",
                "cation_support_count",
                "anion_support_count",
                "canonical_il_key",
            ],
            ascending=[True, True, False, False, True],
            kind="mergesort",
            na_position="last",
        )
        write_csv(
            rank_one_record,
            self._data_path("pareto_rank1_top8_selection.csv"),
        )
        return {
            "feasible_unseen": int(len(ranked)),
            "pareto_rank_1": int(ranked["Pareto_rank"].eq(1).sum()) if not ranked.empty else 0,
            "final_recommendations": int(len(final)),
            "recommendation_classes": sorted(final["recommendation_class"].dropna().unique().tolist()) if not final.empty else [],
        }

    def counterfactuals(self) -> dict[str, Any]:
        """Compute matched cation-fixed and anion-fixed substitution shifts."""

        proxies = pd.read_csv(self._data_path("application_proxies_temperature.csv"))
        ad = pd.read_csv(self._data_path("applicability_domain.csv"))[
            ["candidate_id", "AD_status"]
        ]
        flags = pd.read_csv(self._data_path("curve_quality_flags.csv"))
        main_flags = (
            flags[flags["analysis_window"].eq("main")]
            if "analysis_window" in flags
            else flags
        )
        counts = curve_counts(main_flags)
        merged = proxies.merge(ad, on="candidate_id", how="left").merge(
            counts, on="candidate_id", how="left"
        )
        merged["severe_curve_failure_count"] = merged[
            "severe_curve_failure_count"
        ].fillna(0)
        comparisons, summary = analyze_counterfactual_substitutions(
            merged, self.config["counterfactuals"]["temperatures_K"]
        )
        write_csv(comparisons, self._data_path("counterfactual_ion_substitutions.csv"))
        write_csv(summary, self._data_path("counterfactual_summary.csv"))
        return {"comparisons": int(len(comparisons)), "summary_rows": int(len(summary))}

    def figures(self) -> dict[str, Any]:
        """Generate thermophysical-screening and reference-cell figures."""

        from .plotting import generate_figure5, generate_reference_cell_figure

        figure5 = generate_figure5(self.paths, self.config)
        figure6 = generate_reference_cell_figure(self.paths, self.config)
        return {**figure5, **figure6}

    def tables(self) -> dict[str, Any]:
        """Generate paper-ready CSV and LaTeX result tables."""

        from .reporting import generate_tables

        return generate_tables(self.paths, self.config)

    def report(self) -> dict[str, Any]:
        """Generate Markdown, LaTeX, JSON, and terminal summary from actual outputs."""

        from .reporting import generate_report

        return generate_report(self.paths, self.config)
