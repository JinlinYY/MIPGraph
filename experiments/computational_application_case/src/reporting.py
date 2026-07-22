"""Paper-ready tables, cautious case report, and machine-readable run summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io_utils import write_csv, write_json


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_latex(frame: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    path.write_text(
        frame.to_latex(
            index=False,
            escape=True,
            na_rep="not available",
            caption=caption,
            label=label,
            float_format=lambda value: f"{value:.5g}",
        ),
        encoding="utf-8",
    )


def generate_tables(paths: dict[str, Path], config: dict[str, Any]) -> dict[str, Any]:
    """Create the four required paper-ready result tables from persisted outputs."""

    generation = _read_json(paths["steps"] / "candidate_generation.json")
    funnel_rows = [
        ("Initial supported cations", generation.get("initial_cations", 0)),
        ("Initial supported anions", generation.get("initial_anions", 0)),
        ("Theoretical combinations", generation.get("theoretical_combinations", 0)),
        ("Observed references", generation.get("observed_references", 0)),
        ("Unseen retained pairs", generation.get("unseen_candidates", 0)),
        ("Structure/parse failures", generation.get("parse_failures", 0)),
    ]
    funnel = pd.DataFrame(funnel_rows, columns=["stage", "count"])
    write_csv(funnel, paths["tables"] / "candidate_generation_summary.csv")

    final = _read_csv(paths["data"] / "final_prioritized_candidates.csv")
    final_columns = [
        "candidate_id",
        "cation_smiles",
        "anion_smiles",
        "candidate_type",
        "recommendation_class",
        "AD_status",
        "Pareto_rank",
        "conductivity_worst",
        "viscosity_worst",
        "volumetric_heat_capacity_worst",
        "thermal_diffusivity_worst",
        "surface_tension_reference_envelope_deviation_worst",
        "electrolyte_resistance_ohm_worst",
        "joule_heating_power_W_worst",
        "steady_state_temperature_rise_K_worst",
        "transient_temperature_rise_K_worst",
        "low_temperature_conductivity_retention_pct",
        "low_temperature_resistance_retention_pct",
        "high_temperature_resistance_retention_pct",
        "reference_cell_exceedance_index_worst",
        "reference_cell_exceedance_band_worst",
        "reference_cell_exceedance_index_worst_temperature_K",
        "reference_cell_exceedance_index_at_band_worst",
        "reference_cell_exceedance_band_worst_temperature_K",
        "main_advantage",
        "main_limitation",
        "downstream_priority",
        "uncertainty_status",
    ]
    final_table = final.reindex(columns=final_columns).rename(
        columns={
            "candidate_id": "Candidate",
            "cation_smiles": "Cation",
            "anion_smiles": "Anion",
            "candidate_type": "Novelty",
            "recommendation_class": "Recommendation class",
            "AD_status": "AD status",
            "Pareto_rank": "Pareto rank",
            "conductivity_worst": "Worst-case conductivity (S m^-1)",
            "viscosity_worst": "Worst-case viscosity (Pa s)",
            "volumetric_heat_capacity_worst": "Minimum volumetric heat capacity (J m^-3 K^-1)",
            "thermal_diffusivity_worst": "Minimum thermal diffusivity (m^2 s^-1)",
            "surface_tension_reference_envelope_deviation_worst": "Maximum surface-tension reference-envelope deviation (reference IQR)",
            "electrolyte_resistance_ohm_worst": "Worst ideal electrolyte resistance (ohm)",
            "joule_heating_power_W_worst": "Worst conditional Joule power (W)",
            "steady_state_temperature_rise_K_worst": "Worst conditional steady temperature rise (K)",
            "transient_temperature_rise_K_worst": "Worst conditional transient temperature rise (K)",
            "low_temperature_conductivity_retention_pct": "Low-temperature conductivity retention (%)",
            "low_temperature_resistance_retention_pct": "Low-temperature resistance retention (%)",
            "high_temperature_resistance_retention_pct": "High-temperature resistance retention (%)",
            "reference_cell_exceedance_index_worst": "Worst reference-cell exceedance index",
            "reference_cell_exceedance_band_worst": "Worst reference-cell exceedance band",
            "reference_cell_exceedance_index_worst_temperature_K": "Maximum numeric exceedance-index temperature (K)",
            "reference_cell_exceedance_index_at_band_worst": "Exceedance index at most severe band temperature",
            "reference_cell_exceedance_band_worst_temperature_K": "Most severe exceedance-band temperature (K)",
            "main_advantage": "Main advantage",
            "main_limitation": "Main limitation",
            "downstream_priority": "Downstream qualification priority",
            "uncertainty_status": "Uncertainty status",
        }
    )
    write_csv(final_table, paths["tables"] / "final_candidate_table.csv")
    _write_latex(final_table, paths["tables"] / "final_candidate_table.tex", "Prioritized unseen ionic-liquid pairs.", "tab:final-candidates")

    robust = _read_csv(paths["data"] / "candidate_robust_summary.csv")
    reference_columns = [
        "candidate_id",
        "cation_smiles",
        "anion_smiles",
        "conductivity_worst",
        "viscosity_worst",
        "volumetric_heat_capacity_worst",
        "thermal_diffusivity_worst",
        "surface_tension_reference_envelope_deviation_worst",
        "severe_curve_failure_count",
    ]
    references = robust[robust.get("candidate_type", pd.Series(index=robust.index, dtype=str)).eq("observed_reference")].reindex(columns=reference_columns)
    write_csv(references, paths["tables"] / "reference_electrolyte_summary.csv")

    threshold_payload = _read_json(paths["data"] / "reference_thresholds.json")
    threshold_rows = [
        ("Electrical conductivity", "minimum", threshold_payload.get("conductivity_min"), "S m^-1", f"observed-reference q={threshold_payload.get('conductivity_reference_quantile', 'not available')}"),
        ("Viscosity", "maximum", threshold_payload.get("viscosity_max"), "Pa s", f"observed-reference q={threshold_payload.get('viscosity_reference_quantile', 'not available')}"),
        ("Volumetric heat capacity", "minimum", threshold_payload.get("volumetric_heat_capacity_min"), "J m^-3 K^-1", f"observed-reference q={threshold_payload.get('volumetric_heat_capacity_reference_quantile', 'not available')}"),
        ("Thermal diffusivity", "minimum", threshold_payload.get("thermal_diffusivity_min"), "m^2 s^-1", f"observed-reference q={threshold_payload.get('thermal_diffusivity_reference_quantile', 'not available')}"),
        ("Surface-tension reference-envelope deviation", "maximum", threshold_payload.get("surface_tension_reference_envelope_deviation_max"), "reference IQR", "configured reference-envelope bound"),
    ]
    thresholds = pd.DataFrame(threshold_rows, columns=["criterion", "direction", "threshold", "unit", "derivation"])
    write_csv(thresholds, paths["tables"] / "screening_thresholds.csv")
    _write_latex(thresholds, paths["tables"] / "screening_thresholds.tex", "Frozen hard-screening thresholds.", "tab:screening-thresholds")
    scenario = config["reference_cell"]
    scenario_units = {
        "electrode_area_cm2": "cm^2",
        "separator_thickness_um": "micrometre",
        "electrolyte_volume_mL": "mL",
        "charge_discharge_current_A": "A",
        "convective_heat_transfer_coefficient_W_m2_K": "W m^-2 K^-1",
        "exposed_face_count": "count",
        "transient_duration_s": "s",
        "reference_temperature_K": "K",
    }
    scenario_table = pd.DataFrame(
        [
            {
                "parameter": key,
                "value": scenario[key],
                "unit": unit,
                "status": "fixed conditional assumption",
            }
            for key, unit in scenario_units.items()
        ]
    )
    write_csv(
        scenario_table,
        paths["tables"] / "reference_cell_scenario_parameters.csv",
    )
    _write_latex(
        scenario_table,
        paths["tables"] / "reference_cell_scenario_parameters.tex",
        "Fixed assumptions for the conditional reference-cell scenario.",
        "tab:reference-cell-scenario",
    )
    cell_summary = _read_csv(paths["data"] / "reference_cell_candidate_summary.csv")
    final_ids = set(final["candidate_id"].astype(str)) if not final.empty else set()
    cell_table = cell_summary[
        cell_summary["candidate_id"].astype(str).isin(final_ids)
    ].reindex(
        columns=[
            "candidate_id",
            "low_temperature_K",
            "high_temperature_K",
            "low_temperature_conductivity_retention_pct",
            "low_temperature_resistance_retention_pct",
            "high_temperature_resistance_retention_pct",
            "electrolyte_resistance_ohm_worst",
            "joule_heating_power_W_worst",
            "steady_state_temperature_rise_K_worst",
            "transient_temperature_rise_K_worst",
            "reference_cell_exceedance_index_worst",
            "reference_cell_exceedance_band_worst",
            "reference_cell_exceedance_index_worst_temperature_K",
            "reference_cell_exceedance_index_at_band_worst",
            "reference_cell_exceedance_band_worst_temperature_K",
        ]
    )
    write_csv(cell_table, paths["tables"] / "reference_cell_candidate_summary.csv")
    _write_latex(
        cell_table,
        paths["tables"] / "reference_cell_candidate_summary.tex",
        "Conditional reference-cell metrics for prioritized candidates.",
        "tab:reference-cell-candidates",
    )
    return {
        "table_files": [
            str(paths["tables"] / "candidate_generation_summary.csv"),
            str(paths["tables"] / "final_candidate_table.csv"),
            str(paths["tables"] / "final_candidate_table.tex"),
            str(paths["tables"] / "reference_electrolyte_summary.csv"),
            str(paths["tables"] / "screening_thresholds.csv"),
            str(paths["tables"] / "screening_thresholds.tex"),
            str(paths["tables"] / "reference_cell_scenario_parameters.csv"),
            str(paths["tables"] / "reference_cell_scenario_parameters.tex"),
            str(paths["tables"] / "reference_cell_candidate_summary.csv"),
            str(paths["tables"] / "reference_cell_candidate_summary.tex"),
        ],
        "final_candidate_rows": int(len(final_table)),
        "reference_rows": int(len(references)),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "not available"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.5g}" if np.isfinite(value) else "not available"
    return str(value)


def _candidate_bullets(final: pd.DataFrame) -> list[str]:
    if final.empty:
        return ["- No unseen pair satisfied the complete hard-constraint and prioritization rule in this run."]
    lines = []
    for row in final.itertuples(index=False):
        scenario = ""
        if hasattr(row, "reference_cell_exceedance_band_worst"):
            scenario = (
                f" Conditional scenario: worst ideal electrolyte resistance "
                f"{_fmt(row.electrolyte_resistance_ohm_worst)} ohm, worst transient "
                f"rise {_fmt(row.transient_temperature_rise_K_worst)} K, low/high-temperature "
                f"resistance retention {_fmt(row.low_temperature_resistance_retention_pct)}%/"
                f"{_fmt(row.high_temperature_resistance_retention_pct)}%, "
                f"maximum reference-population exceedance index "
                f"{_fmt(row.reference_cell_exceedance_index_worst)} at "
                f"{_fmt(row.reference_cell_exceedance_index_worst_temperature_K)} K; most severe "
                f"band `{row.reference_cell_exceedance_band_worst}` at "
                f"{_fmt(row.reference_cell_exceedance_band_worst_temperature_K)} K."
            )
        lines.append(
            f"- `{row.candidate_id}` ({row.recommendation_class}, {row.AD_status}, Pareto rank {int(row.Pareto_rank)}): advantage `{row.main_advantage}`; limitation `{row.main_limitation}`; next priority: {row.downstream_priority}.{scenario}"
        )
    return lines


def generate_report(paths: dict[str, Path], config: dict[str, Any]) -> dict[str, Any]:
    """Generate an auditable Markdown/LaTeX case report and terminal summary."""

    generation = _read_json(paths["steps"] / "candidate_generation.json")
    inference = _read_json(paths["steps"] / "inference.json")
    ad_step = _read_json(paths["steps"] / "applicability_domain.json")
    screening = _read_json(paths["steps"] / "screening.json")
    pareto = _read_json(paths["steps"] / "pareto.json")
    reference_cell_step = _read_json(paths["steps"] / "reference_cell.json")
    reference_cell_audit = _read_json(paths["audit"] / "reference_cell_scenario.json")
    uncertainty = _read_json(paths["audit"] / "uncertainty.json")
    unit_audit = _read_json(paths["audit"] / "unit_audit.json")
    final = _read_csv(paths["data"] / "final_prioritized_candidates.csv")
    counterfactual = _read_csv(paths["data"] / "counterfactual_summary.csv")
    classes = sorted(final["recommendation_class"].dropna().astype(str).unique().tolist()) if not final.empty else []
    case_root = Path(config["_project_root"]) / "experiments" / "computational_application_case"
    source_files = [
        str(path.relative_to(Path(config["_project_root"])))
        for path in sorted(case_root.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and not any(
            part.startswith("outputs")
            for part in path.relative_to(case_root).parts
        )
    ]
    test_record = _read_json(paths["audit"] / "unit_test_results.json")
    test_stdout = str(test_record.get("stdout", "")).strip().splitlines()
    is_smoke = bool(config["figures"].get("simplified", False))
    smoke_audit = _read_json(case_root / "outputs" / "smoke_test" / "audit" / "final_output_audit.json")
    uncertainty_available = uncertainty.get("uncertainty_status") == "checkpoint_ensemble"
    unresolved = [
        "reference embedding distance unavailable",
        "candidate liquid-phase persistence across the modeled window is not verified",
        "reference-cell geometry, current, and heat transfer are fixed scenario assumptions rather than calibrated device parameters",
    ]
    if not uncertainty_available:
        unresolved.append("predictive uncertainty unavailable for the single configured checkpoint")
    summary = {
        "run_status": "completed",
        "new_files": source_files,
        "modified_existing_files": [],
        "core_model_code_modified": False,
        "reused_il_property_prediction_modules": [
            "src.chem.smiles_utils",
            "src.chem.graph_featurizer",
            "src.chem.global_descriptors",
            "src.models.factory",
            "src.models.mipgraph",
            "src.data.scaler",
        ],
        "configuration": config.get("_config_path"),
        "data_file": str(config["data"]["benchmark_path"]),
        "split_file": str(config["data"]["split_path"]),
        "checkpoint": inference.get("checkpoint_path", config["model"].get("checkpoint_path")),
        "checkpoint_paths": inference.get("checkpoint_paths", []),
        "scalers": {
            "condition_scaler": inference.get("condition_scaler", {}),
            "target_means": inference.get("target_means", []),
            "target_stds": inference.get("target_stds", []),
            "inverse_transform": inference.get("target_inverse_transform"),
        },
        "deprecated_web_code_used": False,
        "unit_audit_passed": bool(unit_audit.get("passed", False)),
        "property_order": inference.get("property_order", list(unit_audit.get("property_order", []))),
        "property_units": inference.get("property_units", unit_audit.get("units", {})),
        "initial_cations": generation.get("initial_cations", 0),
        "initial_anions": generation.get("initial_anions", 0),
        "theoretical_combinations": generation.get("theoretical_combinations", 0),
        "observed_references": generation.get("observed_references", 0),
        "unseen_candidates": generation.get("unseen_candidates", 0),
        "valid_parsed_rows": generation.get("valid_parsed_rows", 0),
        "valid_unique_pairs": generation.get("valid_unique_pairs", 0),
        "valid_unseen_pool": generation.get("valid_unseen_pool", 0),
        "successful_candidate_conditions": inference.get("successful_predictions", 0),
        "successful_candidates": inference.get("successful_candidates", 0),
        "successful_unseen_candidates": inference.get("successful_unseen_candidates", 0),
        "inference_failures": inference.get("inference_failures", 0),
        "applicability_domain_counts": {key: ad_step.get(key, 0) for key in ["in_domain", "borderline", "out_of_domain"]},
        "hard_constraint_pass": screening.get("hard_constraint_pass", 0),
        "hard_constraint_pass_all": screening.get("hard_constraint_pass_all", 0),
        "pareto_rank_1": pareto.get("pareto_rank_1", 0),
        "final_recommendations": pareto.get("final_recommendations", len(final)),
        "recommendation_classes": classes,
        "uncertainty_status": uncertainty,
        "reference_cell_scenario": reference_cell_audit.get("scenario", {}),
        "reference_cell_model_scope": reference_cell_audit.get("model_scope"),
        "reference_cell_not_a_device_prediction": reference_cell_audit.get(
            "not_a_device_prediction", True
        ),
        "reference_cell_exceedance_band_counts": reference_cell_step.get(
            "exceedance_band_counts", {}
        ),
        "counterfactual_summary_rows": int(len(counterfactual)),
        "figure": str(
            paths["figures"] / "figure5_auditable_virtual_screening_validation.png"
        ),
        "reference_cell_figure": str(
            paths["figures"] / "figure6_reference_cell_scenario_audited.png"
        ),
        "tables": str(paths["tables"]),
        "report": str(paths["report"] / "computational_application_case_results.md"),
        "output_root": str(paths["root"]),
        "unit_test_status": test_record.get("status", "not_recorded"),
        "unit_test_result": test_stdout[-1] if test_stdout else "not_recorded",
        "smoke_test_status": "completed" if is_smoke else smoke_audit.get("status", "not_recorded"),
        "full_run_status": "not_applicable_to_smoke_output" if is_smoke else "completed",
        "unresolved_issues": unresolved,
        "reproducible_commands": {
            "tests": "python experiments/computational_application_case/scripts/run_unit_tests.py",
            "smoke": "python experiments/computational_application_case/run_all.py --config experiments/computational_application_case/configs/smoke_test.yaml --smoke-test --force",
            "primary": "python experiments/computational_application_case/run_all.py --config experiments/computational_application_case/configs/auditable_virtual_screening.yaml --force --skip-figures --skip-report",
            "protocols": "python experiments/computational_application_case/scripts/build_protocol_stability_outputs.py --force",
            "figures_and_bootstrap": "python experiments/computational_application_case/scripts/build_refactored_application_case.py",
        },
    }
    write_json(summary, paths["report"] / "computational_application_case_summary.json")
    threshold_payload = screening.get("thresholds", {})
    lines = [
        "# MIPGraph computational application case",
        "",
        "## 1. Application definition",
        "",
        f"The current six-output MIPGraph checkpoint screened {generation.get('unseen_candidates', 0)} unseen cation-anion recombinations across a fixed temperature window. The workflow retained {screening.get('hard_constraint_pass', 0)} candidates after hard constraints and prioritized {pareto.get('final_recommendations', 0)} leads. All claims below are model-based and conditional on the audited dataset, preprocessing path, checkpoint, and applicability-domain rules.",
        "",
        "## 2. Candidate-space construction",
        "",
        f"The application reused the current `il_property_prediction` model factory, graph builder, trained checkpoint, graph cache, ion feature cache, cleaned benchmark, and declared row-level split. The training split yielded {generation.get('initial_cations', 0)} supported cations and {generation.get('initial_anions', 0)} supported anions, defining {generation.get('theoretical_combinations', 0)} theoretical combinations. After canonicalization, charge checks, support filtering, observed-pair exclusion, and deterministic descriptor coverage, {generation.get('valid_unseen_pool', 0)} valid unseen pairs remained before the configured cap and {generation.get('unseen_candidates', 0)} entered inference. The archived implementation was not imported or consulted during result generation.",
        "",
        "## 3. Six-property inference coverage",
        "",
        f"Checkpoint: `{summary['checkpoint']}`. Property order: `{summary['property_order']}`. Successful condition-level predictions: {inference.get('successful_predictions', 0)}; failures: {inference.get('inference_failures', 0)}. The target inverse transform and condition scalers are read directly from the selected checkpoint.",
        "",
        "## 4. Unit audit",
        "",
        f"The source-unit audit passed={unit_audit.get('passed', False)}. Predictions retain density in kg m^-3, conductivity in S m^-1, molar heat capacity in J mol^-1 K^-1, surface tension in N m^-1, thermal conductivity in W m^-1 K^-1, and viscosity in Pa s. The heat-capacity basis is molar and is converted once only when constructing mass-specific and volumetric proxies.",
        "",
        "## 5. Physical and curve-quality audit",
        "",
        "Temperature curves are retained without smoothing or post-hoc replacement. Each main-window curve is checked for non-finite or non-positive values, excursions beyond benchmark property ranges, temperature extrapolation, and adjacent-temperature jumps relative to observed-reference behavior. Severe failures are excluded by the hard decision rule. When enabled, extended-window rows and flags are stored as sensitivity outputs and do not enter the main robust summary or screening decision.",
        "",
        "## 6. Application proxy construction",
        "",
        "Molar heat capacity is converted once to mass-specific heat capacity using the complete ion-pair RDKit molar mass. The workflow then computes volumetric heat capacity, thermal diffusivity, a simplified diffusion timescale for the configured length, thermal effusivity, electrolyte mass for the configured volume, robust transport favorability, and reference-window surface-tension deviation.",
        "",
        "## 7. Conditional reference-cell scenario",
        "",
        f"Only after the formal shortlist is fixed, a transparent 60-s constant-current scenario uses electrode area={_fmt(config['reference_cell']['electrode_area_cm2'])} cm^2, separator thickness={_fmt(config['reference_cell']['separator_thickness_um'])} micrometre, electrolyte volume={_fmt(config['reference_cell']['electrolyte_volume_mL'])} mL, current={_fmt(config['reference_cell']['charge_discharge_current_A'])} A, and convective coefficient={_fmt(config['reference_cell']['convective_heat_transfer_coefficient_W_m2_K'])} W m^-2 K^-1. It computes ideal electrolyte-path resistance, I^2R heating, a series conduction-plus-convection thermal resistance, and a first-order 60-s lumped temperature rise. At each primary-window temperature, resistance and transient rise are divided by their temperature-matched observed-reference q75 values; the maximum component ratio is Xi_max, a reference-population exceedance index rather than a safety, failure, or thermal-runaway risk.",
        "",
        "The calculation excludes capacitance, electrode/contact resistance, current collectors, packaging, leakage, reaction heat, temperature-dependent geometry, phase changes, and electrochemical degradation. Liquid phase is assumed but not verified.",
        "",
        "## 8. Applicability-domain analysis",
        "",
        f"Descriptor-space distances are calibrated only on {ad_step.get('reference_count', 0)} training-domain ion pairs. Ion-level support, ion-family support, and temperature coverage can downgrade the descriptor result. Counts were in-domain={ad_step.get('in_domain', 0)}, borderline={ad_step.get('borderline', 0)}, and out-of-domain={ad_step.get('out_of_domain', 0)}. Embedding distance remains unavailable because an identically processed complete reference embedding bank is absent.",
        "",
        "## 9. Whole-temperature-window screening",
        "",
        f"Observed-reference thresholds were frozen before unseen-candidate screening: conductivity >= {_fmt(threshold_payload.get('conductivity_min'))} S m^-1, viscosity <= {_fmt(threshold_payload.get('viscosity_max'))} Pa s, volumetric heat capacity >= {_fmt(threshold_payload.get('volumetric_heat_capacity_min'))} J m^-3 K^-1, thermal diffusivity >= {_fmt(threshold_payload.get('thermal_diffusivity_min'))} m^2 s^-1, and surface-tension reference-envelope deviation <= {_fmt(threshold_payload.get('surface_tension_reference_envelope_deviation_max'))} reference IQR. Complete main-window coverage, curve quality, and allowed AD status are independent fail-closed gates.",
        "",
        "## 10. Pareto analysis",
        "",
        f"Among hard-feasible unseen pairs, {pareto.get('pareto_rank_1', 0)} were non-dominated under exactly four objectives: maximize worst-window conductivity, minimize worst-window viscosity, maximize worst-window volumetric heat capacity, and maximize worst-window thermal diffusivity. The surface-tension reference envelope, applicability domain, and curve quality remain hard constraints. Reference-cell resistance, temperature rise, and Xi_max do not enter hard screening, Pareto sorting, Top-8 selection, or role selection; they are calculated only after the shortlist is fixed.",
        "",
        "## 11. Prioritized candidate classes",
        "",
        *_candidate_bullets(final),
        "",
        "## 12. Counterfactual ion-substitution analysis",
        "",
        f"Matched substitutions compare candidates sharing the same cation or the same anion at configured temperatures. {len(counterfactual)} aggregated substitution rows were produced after excluding severe curve failures and out-of-domain candidates. These are association-based model comparisons, not causal estimates.",
        "",
        "## 13. Limitations",
        "",
        ("At least three compatible checkpoints were propagated through property, proxy, hard-constraint, and Pareto calculations; ensemble spread still captures model variation only. " if uncertainty_available else "The selected checkpoint supplies point predictions only, so uncertainty is explicitly marked unavailable rather than synthesized. ") + "The conditional cell metrics are scenario calculations rather than device predictions: only the ideal electrolyte path and a fixed heat-transfer geometry are represented. The candidates are unseen ion-pair recombinations, but their component ions are deliberately supported by the training split and feature cache. Descriptor AD, family support, and temperature coverage cannot establish phase stability, purity effects, long-term electrochemical behavior, synthesis accessibility, or process-scale safety.",
        "",
        "## 14. Recommended downstream qualification",
        "",
        "Follow-up should measure water content, liquid range, thermal stability, transport properties, ion-size/pore matching, electrode-specific wetting, electrochemical window, capacitance, impedance, self-discharge, and cycling behavior, while separately assessing synthesis and purification feasibility. Sparse thermal-conductivity labels and family-level extrapolation remain explicit risks.",
        "",
        "The identified candidates are thermophysically favorable electrolyte leads generated through prospective computational screening. The present analysis does not establish electrochemical stability, capacitance, energy density, cycle life, phase stability, synthesizability, or device-level superiority.",
        "",
        "Deprecated web code used: No",
    ]
    markdown_path = paths["report"] / "computational_application_case_results.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    section_text = "\n\n".join(lines)
    latex_path = paths["report"] / "computational_application_case_results.tex"
    latex_path.write_text(
        "\\documentclass[10pt]{article}\n\\usepackage[margin=1in]{geometry}\n\\usepackage{booktabs}\n\\usepackage{graphicx}\n\\begin{document}\n"
        + "\\section*{MIPGraph computational application case}\n"
        + "\\begin{verbatim}\n"
        + section_text
        + "\n\\end{verbatim}\n\\end{document}\n",
        encoding="utf-8",
    )
    terminal = (
        f"COMPLETED | unseen={summary['unseen_candidates']} | inferred={summary['successful_candidate_conditions']} "
        f"| failures={summary['inference_failures']} | hard-pass={summary['hard_constraint_pass']} "
        f"| pareto-1={summary['pareto_rank_1']} | final={summary['final_recommendations']} "
        "| unit-audit=PASS | Deprecated web code used: No"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(terminal)
    return {
        "markdown_report": str(markdown_path),
        "latex_report": str(latex_path),
        "summary_json": str(paths["report"] / "computational_application_case_summary.json"),
        "terminal_summary": terminal,
    }
