"""Audit completed application outputs for scientific and schema consistency."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CASE_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.computational_application_case.src.config import (  # noqa: E402
    load_case_config,
    temperature_grid,
)
from experiments.computational_application_case.src.io_utils import write_json  # noqa: E402
from experiments.computational_application_case.src.schema import PROPERTY_UNITS  # noqa: E402


def _assert_close(actual: pd.Series, expected: pd.Series, label: str) -> None:
    if not np.allclose(actual.to_numpy(float), expected.to_numpy(float), rtol=1.0e-9, atol=1.0e-12):
        raise AssertionError(f"Output formula audit failed for {label}")


def audit_outputs(output_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Validate a completed result tree without modifying scientific results."""

    data = output_dir / "data"
    library = pd.read_csv(data / "candidate_library.csv")
    predictions = pd.read_csv(data / "property_predictions_long.csv")
    proxies = pd.read_csv(data / "application_proxies_temperature.csv")
    cell_metrics = pd.read_csv(
        data / "reference_cell_metrics_temperature.csv", low_memory=False
    )
    cell_summary = pd.read_csv(data / "reference_cell_candidate_summary.csv")
    robust = pd.read_csv(data / "candidate_robust_summary.csv")
    ad = pd.read_csv(data / "applicability_domain.csv")
    trace = pd.read_csv(data / "screening_trace.csv")
    final = pd.read_csv(data / "final_prioritized_candidates.csv")
    failures = pd.read_csv(data / "inference_failures.csv")
    with (output_dir / "audit" / "inference_pipeline.json").open("r", encoding="utf-8") as handle:
        inference_metadata = json.load(handle)
    with (data / "reference_thresholds.json").open("r", encoding="utf-8") as handle:
        thresholds = json.load(handle)

    checks: dict[str, bool] = {}
    property_names = list(PROPERTY_UNITS)
    property_matrix = predictions[property_names].to_numpy(dtype=float)
    checks["property_order_exact"] = inference_metadata["property_order"] == property_names
    checks["all_property_predictions_finite_positive"] = bool(
        np.isfinite(property_matrix).all() and (property_matrix > 0.0).all()
    )
    temperature_count = predictions["temperature_K"].nunique()
    checks["complete_candidate_temperature_grid"] = bool(
        len(predictions) == len(library) * temperature_count
        and predictions.groupby("candidate_id").size().eq(temperature_count).all()
    )
    checks["primary_summary_uses_main_window_only"] = bool(
        robust["expected_temperature_point_count"]
        .eq(len(temperature_grid(config["conditions"])))
        .all()
    )
    checks["zero_inference_failures"] = failures.empty
    checks["all_pairs_monovalent"] = bool(
        library["cation_charge"].eq(1).all() and library["anion_charge"].eq(-1).all()
    )
    observed_keys = set(
        library.loc[library["candidate_type"].eq("observed_reference"), "canonical_il_key"]
    )
    unseen_keys = set(
        library.loc[
            library["candidate_type"].eq("unseen_pair_recombination"),
            "canonical_il_key",
        ]
    )
    checks["observed_and_unseen_disjoint"] = observed_keys.isdisjoint(unseen_keys)
    checks["unseen_pairs_not_seen_in_benchmark"] = bool(
        ~library.loc[
            library["candidate_type"].eq("unseen_pair_recombination"),
            "pair_seen_in_benchmark",
        ].astype(bool).any()
    )
    _assert_close(
        proxies["cp_mass_J_kg-1_K-1"],
        proxies["HeatCapacity"] / proxies["molar_mass_kg_per_mol"],
        "mass-specific heat capacity",
    )
    _assert_close(
        proxies["volumetric_heat_capacity"],
        proxies["Density"] * proxies["cp_mass_J_kg-1_K-1"],
        "volumetric heat capacity",
    )
    _assert_close(
        proxies["thermal_diffusivity"],
        proxies["ThermalConductivity"] / proxies["volumetric_heat_capacity"],
        "thermal diffusivity",
    )
    length = float(config["proxies"]["thermal_length_m"])
    _assert_close(
        proxies["simplified_thermal_diffusion_timescale"],
        length**2 / proxies["thermal_diffusivity"],
        "simplified thermal diffusion timescale",
    )
    checks["proxy_unit_formulas_exact"] = True
    scenario = config["reference_cell"]
    area_m2 = float(scenario["electrode_area_cm2"]) * 1.0e-4
    thickness_m = float(scenario["separator_thickness_um"]) * 1.0e-6
    volume_m3 = float(scenario["electrolyte_volume_mL"]) * 1.0e-6
    heat_transfer_area_m2 = float(scenario["exposed_face_count"]) * area_m2
    resistance = thickness_m / (
        cell_metrics["ElectricalConductivity"] * area_m2
    )
    _assert_close(
        cell_metrics["electrolyte_resistance_ohm"],
        resistance,
        "reference-cell electrolyte resistance",
    )
    _assert_close(
        cell_metrics["electrolyte_RC_time_constant_s"],
        resistance * float(scenario["nominal_capacitance_F"]),
        "reference-cell RC contribution",
    )
    power = float(scenario["charge_discharge_current_A"]) ** 2 * resistance
    _assert_close(
        cell_metrics["joule_heating_power_W"],
        power,
        "reference-cell Joule power",
    )
    thermal_resistance = thickness_m / (
        cell_metrics["ThermalConductivity"] * area_m2
    ) + 1.0 / (
        float(scenario["convective_heat_transfer_coefficient_W_m2_K"])
        * heat_transfer_area_m2
    )
    _assert_close(
        cell_metrics["thermal_resistance_K_per_W"],
        thermal_resistance,
        "reference-cell thermal resistance",
    )
    thermal_capacitance = cell_metrics["volumetric_heat_capacity"] * volume_m3
    _assert_close(
        cell_metrics["electrolyte_thermal_capacitance_J_per_K"],
        thermal_capacitance,
        "reference-cell thermal capacitance",
    )
    thermal_time = thermal_resistance * thermal_capacitance
    steady_rise = power * thermal_resistance
    transient_rise = steady_rise * (
        1.0
        - np.exp(-float(scenario["transient_duration_s"]) / thermal_time)
    )
    _assert_close(
        cell_metrics["steady_state_temperature_rise_K"],
        steady_rise,
        "reference-cell steady temperature rise",
    )
    _assert_close(
        cell_metrics["transient_temperature_rise_K"],
        transient_rise,
        "reference-cell transient temperature rise",
    )
    _assert_close(
        cell_metrics["relative_electrolyte_resistance"],
        resistance / cell_metrics["reference_temperature_resistance_ohm"],
        "relative electrolyte resistance",
    )
    checks["conditional_reference_cell_equations_exact"] = True
    checks["reference_cell_summary_complete"] = bool(
        cell_summary["candidate_id"].nunique() == library["candidate_id"].nunique()
        and set(cell_summary["reference_cell_risk_band_worst"]).issubset(
            {
                "within_reference_envelope",
                "elevated_reference_tail",
                "beyond_reference_tail",
            }
        )
        and np.isfinite(
            cell_summary[
                [
                    "reference_cell_risk_index_worst",
                    "reference_cell_worst_temperature_K",
                ]
            ].to_numpy(float)
        ).all()
    )
    reference = robust[robust["candidate_type"].eq("observed_reference")]
    threshold_definitions = {
        "conductivity_min": ("conductivity_worst", "conductivity_reference_quantile"),
        "viscosity_max": ("viscosity_worst", "viscosity_reference_quantile"),
        "volumetric_heat_capacity_min": (
            "volumetric_heat_capacity_worst",
            "volumetric_heat_capacity_reference_quantile",
        ),
        "thermal_diffusivity_min": (
            "thermal_diffusivity_worst",
            "thermal_diffusivity_reference_quantile",
        ),
    }
    checks["thresholds_equal_observed_reference_quantiles"] = all(
        np.isclose(
            thresholds[name],
            np.nanquantile(reference[column], thresholds[quantile_name]),
            rtol=1.0e-12,
            atol=1.0e-15,
        )
        for name, (column, quantile_name) in threshold_definitions.items()
    )
    final_ids = set(final["candidate_id"])
    feasible_unseen = trace[
        trace["candidate_type"].eq("unseen_pair_recombination")
        & trace["final_feasible"].astype(bool)
    ]
    checks["final_candidates_are_hard_feasible_unseen_pairs"] = final_ids.issubset(
        set(feasible_unseen["candidate_id"])
    )
    checks["no_out_of_domain_final_candidate"] = bool(
        final.empty or ~final["AD_status"].eq("out_of_domain").any()
    )
    checks["ad_statuses_are_declared"] = set(ad["AD_status"]).issubset(
        {"in_domain", "borderline", "out_of_domain"}
    )
    checks["ad_includes_family_support"] = {
        "cation_family_support",
        "anion_family_support",
    }.issubset(ad.columns)
    report_path = output_dir / "report" / "computational_application_case_results.md"
    report_text = report_path.read_text(encoding="utf-8").lower()
    prohibited = [
        "experimental validation",
        "experimentally confirmed",
        "device performance prediction",
        "optimal electrolyte",
        "best supercapacitor electrolyte",
        "predicted capacitance",
        "predicted cycle life",
        "predicted energy density",
        "predicted electrochemical stability",
    ]
    checks["report_avoids_prohibited_claims"] = not any(term in report_text for term in prohibited)
    checks["report_contains_required_boundary"] = (
        "the identified candidates are thermophysically favorable electrolyte leads generated through prospective computational screening. the present analysis does not establish electrochemical stability, capacitance, energy density, cycle life, phase stability, synthesizability, or device-level superiority."
        in report_text
    )
    source_files = list((CASE_DIR / "src").glob("*.py")) + list((CASE_DIR / "scripts").glob("*.py"))
    imported_roots = []
    for path in source_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.extend(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.append(node.module.split(".")[0])
    checks["no_deprecated_package_import"] = "web" not in imported_roots
    required_files = [
        output_dir / "figures" / "figure5_computational_application_case.png",
        output_dir / "figures" / "figure5_computational_application_case.pdf",
        output_dir / "figures" / "figure6_reference_cell_scenario.png",
        output_dir / "figures" / "figure6_reference_cell_scenario.pdf",
        output_dir / "tables" / "final_candidate_table.csv",
        output_dir / "tables" / "final_candidate_table.tex",
        output_dir / "tables" / "reference_cell_scenario_parameters.csv",
        output_dir / "tables" / "reference_cell_scenario_parameters.tex",
        output_dir / "tables" / "reference_cell_candidate_summary.csv",
        output_dir / "tables" / "reference_cell_candidate_summary.tex",
        output_dir / "report" / "computational_application_case_results.tex",
        output_dir / "report" / "computational_application_case_summary.json",
    ]
    checks["required_outputs_exist_nonempty"] = all(
        path.exists() and path.stat().st_size > 0 for path in required_files
    )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"Final output audit failed: {failed}")
    return {
        "status": "passed",
        "checks": checks,
        "candidate_count": int(len(library)),
        "prediction_rows": int(len(predictions)),
        "inference_failure_rows": int(len(failures)),
        "hard_feasible_unseen": int(len(feasible_unseen)),
        "final_candidate_count": int(len(final)),
        "deprecated_web_code_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CASE_DIR / "configs" / "default.yaml")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = load_case_config(args.config)
    output_dir = args.output_dir.resolve() if args.output_dir else Path(config["_output_dir"])
    result = audit_outputs(output_dir, config)
    write_json(result, output_dir / "audit" / "final_output_audit.json")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
