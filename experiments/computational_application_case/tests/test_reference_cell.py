from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.computational_application_case.src.reference_cell import (
    simulate_reference_cell_scenario,
)


SCENARIO = {
    "electrode_area_cm2": 100.0,
    "separator_thickness_um": 100.0,
    "electrolyte_volume_mL": 1.0,
    "charge_discharge_current_A": 2.0,
    "convective_heat_transfer_coefficient_W_m2_K": 10.0,
    "exposed_face_count": 2,
    "transient_duration_s": 10.1,
    "reference_temperature_K": 298.15,
    "exceedance_reference_quantiles": [0.75, 0.95],
}


def _row(
    candidate_id: str,
    candidate_type: str,
    temperature: float,
    conductivity: float,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "candidate_type": candidate_type,
        "temperature_K": temperature,
        "analysis_window": "main",
        "ElectricalConductivity": conductivity,
        "ThermalConductivity": 0.2,
        "Density": 1000.0,
        "cp_mass_J_kg-1_K-1": 2000.0,
        "volumetric_heat_capacity": 2.0e6,
    }


def test_reference_cell_metrics_follow_the_declared_lumped_model() -> None:
    rows = [
        _row("R1", "observed_reference", 298.15, 1.0),
        _row("R2", "observed_reference", 298.15, 0.5),
        _row("U", "unseen_pair_recombination", 298.15, 1.0),
    ]
    metrics, _, metadata = simulate_reference_cell_scenario(
        pd.DataFrame(rows), SCENARIO
    )
    result = metrics.loc[metrics["candidate_id"].eq("U")].iloc[0]

    assert result["electrolyte_resistance_ohm"] == pytest.approx(0.01)
    assert result["relative_electrolyte_resistance"] == pytest.approx(1.0)
    assert result["joule_heating_power_W"] == pytest.approx(0.04)
    assert result["thermal_resistance_K_per_W"] == pytest.approx(5.05)
    assert result["thermal_resistance_conduction_fraction"] == pytest.approx(
        0.05 / 5.05
    )
    assert result["thermal_resistance_convection_fraction"] == pytest.approx(
        5.0 / 5.05
    )
    assert (
        result["thermal_resistance_conduction_fraction"]
        + result["thermal_resistance_convection_fraction"]
    ) == pytest.approx(1.0)
    assert result["electrolyte_thermal_capacitance_J_per_K"] == pytest.approx(2.0)
    assert result["lumped_thermal_time_constant_s"] == pytest.approx(10.1)
    assert result["steady_state_temperature_rise_K"] == pytest.approx(0.202)
    assert result["transient_temperature_rise_K"] == pytest.approx(
        0.202 * (1.0 - np.exp(-1.0))
    )
    assert metadata["model_scope"] == "conditional_reference_cell_scenario"


def test_reference_cell_summary_reports_retention_and_worst_temperature() -> None:
    temperatures = [278.15, 298.15, 318.15]
    rows: list[dict[str, object]] = []
    for temperature, conductivities in zip(
        temperatures,
        [(1.0, 0.5, 0.25), (2.0, 1.0, 0.5), (4.0, 2.0, 1.0)],
    ):
        rows.extend(
            [
                _row("R1", "observed_reference", temperature, conductivities[0]),
                _row("R2", "observed_reference", temperature, conductivities[1]),
                _row("U", "unseen_pair_recombination", temperature, conductivities[2]),
            ]
        )

    metrics, summary, _ = simulate_reference_cell_scenario(
        pd.DataFrame(rows), SCENARIO
    )
    candidate = summary.loc[summary["candidate_id"].eq("U")].iloc[0]
    low_row = metrics.loc[
        metrics["candidate_id"].eq("U") & metrics["temperature_K"].eq(278.15)
    ].iloc[0]

    assert candidate["low_temperature_resistance_ratio_to_reference"] == pytest.approx(2.0)
    assert candidate["high_temperature_resistance_ratio_to_reference"] == pytest.approx(0.5)
    assert candidate["low_temperature_resistance_retention_pct"] == pytest.approx(200.0)
    assert candidate["high_temperature_resistance_retention_pct"] == pytest.approx(50.0)
    assert candidate["low_temperature_conductivity_retention_pct"] == pytest.approx(50.0)
    assert candidate["high_temperature_conductivity_retention_pct"] == pytest.approx(200.0)
    assert candidate["reference_cell_exceedance_index_worst_temperature_K"] == pytest.approx(278.15)
    assert candidate["reference_cell_exceedance_band_worst"] == "beyond_reference_tail"
    assert low_row["reference_cell_exceedance_component"] == "electrical;thermal"


def test_reference_cell_rejects_nonphysical_scenario_parameters() -> None:
    invalid = dict(SCENARIO, electrode_area_cm2=0.0)
    with pytest.raises(ValueError, match="electrode_area_cm2"):
        simulate_reference_cell_scenario(
            pd.DataFrame([_row("R1", "observed_reference", 298.15, 1.0)]),
            invalid,
        )


def test_reference_cell_requires_integer_face_count_and_declared_exceedance_quantiles() -> None:
    with pytest.raises(ValueError, match="exposed_face_count must be an integer"):
        simulate_reference_cell_scenario(
            pd.DataFrame([_row("R1", "observed_reference", 298.15, 1.0)]),
            dict(SCENARIO, exposed_face_count=2.5),
        )
    with pytest.raises(ValueError, match=r"exactly \[0.75, 0.95\]"):
        simulate_reference_cell_scenario(
            pd.DataFrame([_row("R1", "observed_reference", 298.15, 1.0)]),
            dict(SCENARIO, exceedance_reference_quantiles=[0.8, 0.9]),
        )


def test_extended_sensitivity_rows_do_not_change_primary_summary() -> None:
    rows = [
        _row("R1", "observed_reference", 298.15, 1.0),
        _row("R2", "observed_reference", 298.15, 0.5),
        _row("U", "unseen_pair_recombination", 298.15, 0.75),
    ]
    for candidate_id, candidate_type, conductivity in [
        ("R1", "observed_reference", 1.0),
        ("R2", "observed_reference", 0.5),
        ("U", "unseen_pair_recombination", 1.0e-6),
    ]:
        row = _row(candidate_id, candidate_type, 398.15, conductivity)
        row["analysis_window"] = "extended_sensitivity"
        rows.append(row)
    metrics, summary, _ = simulate_reference_cell_scenario(
        pd.DataFrame(rows), SCENARIO
    )
    candidate = summary.loc[summary["candidate_id"].eq("U")].iloc[0]

    assert len(metrics) == 6
    assert candidate["temperature_point_count"] == 1
    assert candidate["high_temperature_K"] == pytest.approx(298.15)


def test_numeric_worst_exceedance_and_worst_band_are_reported_independently() -> None:
    rows: list[dict[str, object]] = []
    for temperature, reference_resistances, unseen_resistance in [
        (298.15, [1.0, 1.0, 1.0, 100.0], 50.0),
        (318.15, [1.0, 1.0, 1.0, 1.0], 1.5),
    ]:
        for index, resistance in enumerate(reference_resistances, start=1):
            rows.append(
                _row(
                    f"R{index}",
                    "observed_reference",
                    temperature,
                    0.01 / resistance,
                )
            )
        rows.append(
            _row(
                "U",
                "unseen_pair_recombination",
                temperature,
                0.01 / unseen_resistance,
            )
        )
    _, summary, _ = simulate_reference_cell_scenario(pd.DataFrame(rows), SCENARIO)
    candidate = summary.loc[summary["candidate_id"].eq("U")].iloc[0]

    assert candidate["reference_cell_exceedance_index_worst"] == pytest.approx(
        50.0 / 25.75
    )
    assert candidate["reference_cell_exceedance_index_worst_temperature_K"] == pytest.approx(
        298.15
    )
    assert candidate["reference_cell_exceedance_band_worst"] == "beyond_reference_tail"
    assert candidate["reference_cell_exceedance_band_worst_temperature_K"] == pytest.approx(
        318.15
    )
    assert candidate["reference_cell_exceedance_index_at_band_worst"] == pytest.approx(1.5)
