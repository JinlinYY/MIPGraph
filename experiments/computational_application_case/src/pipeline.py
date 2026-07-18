"""Stepwise, resumable computational application-case pipeline."""

from __future__ import annotations

import json
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
from .screening import (
    audit_curve_quality,
    curve_counts,
    derive_reference_thresholds,
    prioritize_candidates,
    screen_candidates,
)
from .uncertainty import estimate_uncertainty


LOGGER = logging.getLogger("computational_application_case")

STEP_ORDER = [
    "repository_audit",
    "unit_audit",
    "candidate_generation",
    "inference",
    "proxies",
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
        self.step_functions: dict[str, Callable[[], dict[str, Any]]] = {
            "repository_audit": self.repository_audit,
            "unit_audit": self.unit_audit,
            "candidate_generation": self.candidate_generation,
            "inference": self.inference,
            "proxies": self.proxies,
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
                    LOGGER.info("[%s] already completed; resume skips it", step)
                    results[step] = json.loads(marker.read_text(encoding="utf-8"))
                    continue
                raise FileExistsError(
                    f"Step {step} already completed at {marker}. Use --resume or --force."
                )
            LOGGER.info("[%s] starting", step)
            payload = self.step_functions[step]()
            write_step_marker(self.paths["steps"], step, payload)
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
            "units": PROPERTY_UNITS,
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
        temperatures = temperature_grid(self.config["conditions"])
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
        result = self._get_adapter().predict(
            library,
            temperatures,
            float(self.config["conditions"]["pressure_kPa"]),
            self.paths["cache"] / "candidate_graphs.pt",
            training_range,
            force=self.force,
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
            "successful_predictions": int(len(result.predictions)),
            "successful_candidates": int(result.predictions["candidate_id"].nunique()),
            "successful_unseen_candidates": int(
                result.predictions.loc[
                    result.predictions["candidate_type"].eq("unseen_pair_recombination"),
                    "candidate_id",
                ].nunique()
            ),
            "inference_failures": int(len(result.failures)),
        }

    def proxies(self) -> dict[str, Any]:
        """Compute temperature-resolved application proxy mappings."""

        prediction_path = self._data_path("property_predictions_long.csv")
        if not prediction_path.exists():
            raise FileNotFoundError("Inference predictions are required before proxy calculation")
        predictions = pd.read_csv(prediction_path)
        proxies = compute_application_proxies(predictions, self.config["proxies"])
        write_csv(proxies, self._data_path("application_proxies_temperature.csv"))
        proxy_columns = [
            "cp_mass_J_kg-1_K-1",
            "volumetric_heat_capacity",
            "thermal_diffusivity",
            "simplified_thermal_diffusion_timescale",
            "electrolyte_mass_kg",
            "z_conductivity",
            "z_viscosity",
            "transport_favorability",
            "interfacial_window_deviation",
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
            "proxy_warning_rows": int(proxies["proxy_warnings"].fillna("").ne("").sum()),
        }

    def curve_quality(self) -> dict[str, Any]:
        """Audit curves and build complete whole-window robust summaries."""

        proxy_path = self._data_path("application_proxies_temperature.csv")
        if not proxy_path.exists():
            raise FileNotFoundError("Proxy output is required before curve-quality audit")
        proxies = pd.read_csv(proxy_path)
        flags = audit_curve_quality(proxies, self._benchmark(), self.config["curve_quality"])
        counts = curve_counts(flags)
        summary = summarize_whole_temperature_window(proxies)
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
            lambda value: parsed.get(str(value)).canonical_cation_smiles if str(value) in parsed else None
        )
        training["_anion"] = training["IL_SMILES"].map(
            lambda value: parsed.get(str(value)).canonical_anion_smiles if str(value) in parsed else None
        )
        property_columns = [f"{name}_ActualValue" for name in PROPERTY_UNITS]
        training["_label_count"] = training[property_columns].notna().sum(axis=1)
        metadata_rows = []
        temperatures = temperature_grid(self.config["conditions"])
        for row in candidates.itertuples(index=False):
            cation_rows = training[training["_cation"].eq(row.canonical_cation_smiles)]
            anion_rows = training[training["_anion"].eq(row.canonical_anion_smiles)]
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
                    "pair_seen": bool(row.pair_seen_in_training),
                    "cation_support_count": int(row.cation_support_count),
                    "anion_support_count": int(row.anion_support_count),
                    "cation_family_support": int(row.cation_support_count),
                    "anion_family_support": int(row.anion_support_count),
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
        paths = list(self.config["model"].get("checkpoint_paths", []))
        if not paths and self.config["model"].get("checkpoint_path"):
            paths = [self.config["model"]["checkpoint_path"]]
        property_table, proxy_table, feasibility, status = estimate_uncertainty(
            predictions, proxies, paths, self.config["uncertainty"]
        )
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
        counts = curve_counts(flags)
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
        """Generate the data-driven eight-panel Figure 5 and panel files."""

        from .plotting import generate_figure5

        return generate_figure5(self.paths, self.config)

    def tables(self) -> dict[str, Any]:
        """Generate paper-ready CSV and LaTeX result tables."""

        from .reporting import generate_tables

        return generate_tables(self.paths, self.config)

    def report(self) -> dict[str, Any]:
        """Generate Markdown, LaTeX, JSON, and terminal summary from actual outputs."""

        from .reporting import generate_report

        return generate_report(self.paths, self.config)
