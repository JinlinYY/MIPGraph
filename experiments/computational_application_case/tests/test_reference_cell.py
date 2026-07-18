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
    "nominal_capacitance_F": 10.0,
    "charge_discharge_current_A": 2.0,
    "convective_heat_transfer_coefficient_W_m2_K": 10.0,
    "exposed_face_count": 2,
    "transient_duration_s": 10.1,
    "reference_temperature_K": 298.15,
    "risk_reference_quantiles": [0.75, 0.95],
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
    assert result["electrolyte_RC_time_constant_s"] == pytest.approx(0.1)
    assert result["joule_heating_power_W"] == pytest.approx(0.04)
    assert result["thermal_resistance_K_per_W"] == pytest.approx(5.05)
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
    assert candidate["low_temperature_conductivity_retention_pct"] == pytest.approx(50.0)
    assert candidate["high_temperature_conductivity_retention_pct"] == pytest.approx(200.0)
    assert candidate["reference_cell_worst_temperature_K"] == pytest.approx(278.15)
    assert candidate["reference_cell_risk_band_worst"] == "beyond_reference_tail"
    assert low_row["reference_cell_risk_reason"] == "electrical;thermal"


def test_reference_cell_rejects_nonphysical_scenario_parameters() -> None:
    invalid = dict(SCENARIO, electrode_area_cm2=0.0)
    with pytest.raises(ValueError, match="electrode_area_cm2"):
        simulate_reference_cell_scenario(
            pd.DataFrame([_row("R1", "observed_reference", 298.15, 1.0)]),
            invalid,
        )
