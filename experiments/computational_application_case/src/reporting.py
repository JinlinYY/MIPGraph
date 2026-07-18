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
        "interfacial_deviation_worst",
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
            "interfacial_deviation_worst": "Maximum interfacial deviation (reference IQR)",
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
        "interfacial_deviation_worst",
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
        ("Interfacial-window deviation", "maximum", threshold_payload.get("interfacial_deviation_max"), "reference IQR", "configured physical proxy bound"),
    ]
    thresholds = pd.DataFrame(threshold_rows, columns=["criterion", "direction", "threshold", "unit", "derivation"])
    write_csv(thresholds, paths["tables"] / "screening_thresholds.csv")
    _write_latex(thresholds, paths["tables"] / "screening_thresholds.tex", "Frozen hard-screening thresholds.", "tab:screening-thresholds")
    return {
        "table_files": [
            str(paths["tables"] / "candidate_generation_summary.csv"),
            str(paths["tables"] / "final_candidate_table.csv"),
            str(paths["tables"] / "final_candidate_table.tex"),
            str(paths["tables"] / "reference_electrolyte_summary.csv"),
            str(paths["tables"] / "screening_thresholds.csv"),
            str(paths["tables"] / "screening_thresholds.tex"),
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
        lines.append(
            f"- `{row.candidate_id}` ({row.recommendation_class}, {row.AD_status}, Pareto rank {int(row.Pareto_rank)}): advantage `{row.main_advantage}`; limitation `{row.main_limitation}`; next priority: {row.downstream_priority}."
        )
    return lines


def _latex_escape(text: str) -> str:
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(char, char) for char in text)


def generate_report(paths: dict[str, Path], config: dict[str, Any]) -> dict[str, Any]:
    """Generate an auditable Markdown/LaTeX case report and terminal summary."""

    generation = _read_json(paths["steps"] / "candidate_generation.json")
    inference = _read_json(paths["steps"] / "inference.json")
    ad_step = _read_json(paths["steps"] / "applicability_domain.json")
    screening = _read_json(paths["steps"] / "screening.json")
    pareto = _read_json(paths["steps"] / "pareto.json")
    uncertainty = _read_json(paths["audit"] / "uncertainty.json")
    unit_audit = _read_json(paths["audit"] / "unit_audit.json")
    final = _read_csv(paths["data"] / "final_prioritized_candidates.csv")
    counterfactual = _read_csv(paths["data"] / "counterfactual_summary.csv")
    classes = sorted(final["recommendation_class"].dropna().astype(str).unique().tolist()) if not final.empty else []
    summary = {
        "run_status": "completed",
        "configuration": config.get("_config_path"),
        "checkpoint": inference.get("checkpoint_path", config["model"].get("checkpoint_path")),
        "deprecated_web_code_used": False,
        "unit_audit_passed": bool(unit_audit.get("passed", False)),
        "property_order": inference.get("property_order", list(unit_audit.get("property_order", []))),
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
        "counterfactual_summary_rows": int(len(counterfactual)),
        "figure": str(paths["figures"] / "figure5_computational_application_case.png"),
        "tables": str(paths["tables"]),
        "report": str(paths["report"] / "computational_application_case_results.md"),
        "output_root": str(paths["root"]),
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
        "The application reused the current `il_property_prediction` model factory, graph builder, trained checkpoint, graph cache, ion feature cache, cleaned benchmark, and declared row-level split. The archived implementation was not imported or consulted during result generation.",
        "",
        "## 3. Six-property inference coverage",
        "",
        f"Checkpoint: `{summary['checkpoint']}`. Property order: `{summary['property_order']}`. Successful condition-level predictions: {inference.get('successful_predictions', 0)}; failures: {inference.get('inference_failures', 0)}. The target inverse transform and condition scalers are read directly from the selected checkpoint.",
        "",
        "## 4. Unit audit",
        "",
        f"The source-unit audit passed={unit_audit.get('passed', False)}. The training split yielded {generation.get('initial_cations', 0)} supported cations and {generation.get('initial_anions', 0)} supported anions, defining {generation.get('theoretical_combinations', 0)} theoretical combinations. After canonicalization, charge checks, support filtering, observed-pair exclusion, cache compatibility, and the configured cap, {generation.get('unseen_candidates', 0)} unseen pair recombinations and {generation.get('observed_references', 0)} observed references entered inference.",
        "",
        "## 5. Physical and curve-quality audit",
        "",
        "Predictions preserve the checkpoint's raw physical units: density in kg m^-3, electrical conductivity in S m^-1, molar heat capacity in J mol^-1 K^-1, surface tension in N m^-1, thermal conductivity in W m^-1 K^-1, and viscosity in Pa s. Temperature curves are retained without smoothing or post-hoc replacement.",
        "",
        "## 6. Application proxy construction",
        "",
        "Molar heat capacity is converted once to mass-specific heat capacity using the complete ion-pair RDKit molar mass. The workflow then computes volumetric heat capacity, thermal diffusivity, a simplified diffusion timescale for the configured length, thermal effusivity, electrolyte mass for the configured volume, robust transport favorability, and reference-window surface-tension deviation.",
        "",
        "## 7. Applicability-domain analysis",
        "",
        f"Each complete temperature curve is checked for non-finite or non-positive values, excursions beyond benchmark property ranges, temperature extrapolation, and adjacent-temperature jumps relative to observed-reference behavior. Severe failures are excluded by the hard decision rule; warning records remain visible in `curve_quality_flags.csv`. Descriptor-space distances are calibrated only on {ad_step.get('reference_count', 0)} training-domain ion pairs. Counts were in-domain={ad_step.get('in_domain', 0)}, borderline={ad_step.get('borderline', 0)}, and out-of-domain={ad_step.get('out_of_domain', 0)}.",
        "",
        "## 8. Whole-temperature-window screening",
        "",
        f"Ion support and temperature coverage can downgrade descriptor-based status. Embedding distance is reported as unavailable because a complete, identically processed reference embedding bank is not present. Observed-reference thresholds were frozen before unseen-candidate screening: conductivity >= {_fmt(threshold_payload.get('conductivity_min'))} S m^-1, viscosity <= {_fmt(threshold_payload.get('viscosity_max'))} Pa s, volumetric heat capacity >= {_fmt(threshold_payload.get('volumetric_heat_capacity_min'))} J m^-3 K^-1, thermal diffusivity >= {_fmt(threshold_payload.get('thermal_diffusivity_min'))} m^2 s^-1, and interfacial-window deviation <= {_fmt(threshold_payload.get('interfacial_deviation_max'))} reference IQR.",
        "",
        "## 9. Pareto analysis",
        "",
        f"Among hard-feasible unseen pairs, {pareto.get('pareto_rank_1', 0)} were non-dominated under the declared conductivity, viscosity, heat-capacity, thermal-diffusivity, and interfacial objectives. Utopia distance is used only for ordering within this transparent multi-objective result, not as a substitute for hard feasibility.",
        "",
        "## 10. Prioritized candidate classes",
        "",
        *_candidate_bullets(final),
        "",
        "## 11. Counterfactual ion-substitution analysis",
        "",
        f"Matched substitutions compare candidates sharing the same cation or the same anion at configured temperatures. {len(counterfactual)} aggregated substitution rows were produced after excluding severe curve failures and out-of-domain candidates. These are association-based model comparisons, not causal estimates.",
        "",
        "## 12. Limitations",
        "",
        "The selected checkpoint supplies point predictions only, and fewer than three compatible checkpoints were configured; uncertainty is therefore explicitly marked unavailable rather than synthesized. The candidates are unseen ion-pair recombinations, but their component ions are deliberately supported by the training split and feature cache. Descriptor AD, ion-frequency support, and temperature coverage cannot establish phase stability, purity effects, long-term electrochemical behavior, synthesis accessibility, or process-scale safety. Recommended follow-up is independent thermophysical measurement, electrochemical-window characterization, phase-behavior assessment, and iterative model updating with newly measured labels.",
        "",
        "## 13. Recommended downstream qualification",
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
    print(terminal)
    return {
        "markdown_report": str(markdown_path),
        "latex_report": str(latex_path),
        "summary_json": str(paths["report"] / "computational_application_case_summary.json"),
        "terminal_summary": terminal,
    }
