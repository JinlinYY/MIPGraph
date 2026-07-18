from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.computational_application_case.src.proxies import (
    interfacial_window_deviation,
    log_iqr_standardize,
    summarize_whole_temperature_window,
    transport_favorability,
)
from experiments.computational_application_case.src.screening import screen_candidates
from experiments.computational_application_case.src.units import (
    simplified_thermal_diffusion_timescale,
    thermal_diffusivity,
    volumetric_heat_capacity,
)


def test_conductivity_log_iqr_standardization_uses_log10() -> None:
    score, protected = log_iqr_standardize(100.0, [1.0, 10.0, 100.0], 1.0e-12)
    assert score == pytest.approx(1.0)
    assert not protected


def test_viscosity_log_iqr_standardization_has_same_orientation() -> None:
    score, _ = log_iqr_standardize(1.0, [0.01, 0.1, 1.0], 1.0e-12)
    assert score == pytest.approx(1.0)


def test_zero_iqr_uses_numerical_protection_and_reports_it() -> None:
    score, protected = log_iqr_standardize(10.0, [1.0, 1.0, 1.0], 0.5)
    assert score == pytest.approx(2.0)
    assert protected


def test_interfacial_window_deviation_is_zero_inside_and_scaled_outside() -> None:
    assert interfacial_window_deviation(30.0, 20.0, 40.0, 10.0, 1.0e-12) == 0.0
    assert interfacial_window_deviation(15.0, 20.0, 40.0, 10.0, 1.0e-12) == pytest.approx(0.5)
    assert interfacial_window_deviation(50.0, 20.0, 40.0, 10.0, 1.0e-12) == pytest.approx(1.0)


def test_transport_favorability_subtracts_viscosity_penalty() -> None:
    assert transport_favorability(1.5, 0.4) == pytest.approx(1.1)


def test_thermal_proxy_chain_matches_independent_worked_values() -> None:
    c_vol = volumetric_heat_capacity(1200.0, 1500.0)
    alpha = thermal_diffusivity(0.18, c_vol)
    tau = simplified_thermal_diffusion_timescale(1.0e-3, alpha)
    assert c_vol == pytest.approx(1.8e6)
    assert alpha == pytest.approx(1.0e-7)
    assert tau == pytest.approx(10.0)


def test_whole_window_summary_uses_required_worst_case_directions() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["A", "A", "A"],
            "temperature_K": [298.15, 303.15, 308.15],
            "ElectricalConductivity": [1.0, 2.0, 3.0],
            "Viscosity": [0.3, 0.2, 0.1],
            "transport_favorability": [0.0, 1.0, 2.0],
            "volumetric_heat_capacity": [2.0e6, 1.9e6, 1.8e6],
            "thermal_diffusivity": [1.0e-7, 1.1e-7, 1.2e-7],
            "simplified_thermal_diffusion_timescale": [10.0, 9.0, 8.0],
            "interfacial_window_deviation": [0.1, 0.2, 0.3],
            "Density": [1200.0, 1180.0, 1160.0],
        }
    )
    summary = summarize_whole_temperature_window(frame).iloc[0]
    assert summary["conductivity_worst"] == 1.0
    assert summary["viscosity_worst"] == 0.3
    assert summary["volumetric_heat_capacity_worst"] == 1.8e6
    assert summary["thermal_diffusivity_worst"] == 1.0e-7
    assert summary["thermal_timescale_worst"] == 10.0
    assert summary["interfacial_deviation_worst"] == 0.3
    assert summary["density_range"] == 40.0
    assert summary["conductivity_worst_temperature_K"] == 298.15
    assert summary["viscosity_worst_temperature_K"] == 298.15


def test_nonpositive_or_nan_values_propagate_as_nan() -> None:
    zero_score, _ = log_iqr_standardize(0.0, [1.0, 10.0, 100.0], 1.0e-12)
    nan_score, _ = log_iqr_standardize(np.nan, [1.0, 10.0, 100.0], 1.0e-12)
    assert np.isnan(zero_score)
    assert np.isnan(nan_score)


def test_whole_window_summary_marks_incomplete_candidate_grid() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["complete", "complete", "partial"],
            "candidate_type": ["unseen_pair_recombination"] * 3,
            "temperature_K": [298.15, 303.15, 298.15],
            "ElectricalConductivity": [1.0, 2.0, 1.0],
            "Viscosity": [0.2, 0.1, 0.2],
            "transport_favorability": [1.0, 2.0, 1.0],
            "volumetric_heat_capacity": [2.0e6] * 3,
            "thermal_diffusivity": [1.0e-7] * 3,
            "simplified_thermal_diffusion_timescale": [10.0] * 3,
            "interfacial_window_deviation": [0.0] * 3,
            "Density": [1200.0] * 3,
        }
    )
    summary = summarize_whole_temperature_window(frame).set_index("candidate_id")
    assert bool(summary.loc["complete", "temperature_grid_complete"])
    assert not bool(summary.loc["partial", "temperature_grid_complete"])


def test_screening_fails_closed_for_missing_summary_or_ad() -> None:
    robust = pd.DataFrame(
        {
            "candidate_id": ["complete"],
            "candidate_type": ["unseen_pair_recombination"],
            "temperature_grid_complete": [True],
            "conductivity_worst": [2.0],
            "viscosity_worst": [0.1],
            "volumetric_heat_capacity_worst": [2.0e6],
            "thermal_diffusivity_worst": [1.0e-7],
            "interfacial_deviation_worst": [0.0],
            "severe_curve_failure_count": [0],
        }
    )
    library = pd.DataFrame(
        {
            "candidate_id": ["complete", "missing"],
            "candidate_type": ["unseen_pair_recombination"] * 2,
            "cation_charge": [1, 1],
            "anion_charge": [-1, -1],
            "generation_status": ["retained", "retained"],
        }
    )
    ad = pd.DataFrame(
        {
            "candidate_id": ["complete"],
            "AD_status": ["in_domain"],
            "AD_reason": ["descriptor_distance=in_domain"],
        }
    )
    thresholds = {
        "conductivity_min": 1.0,
        "viscosity_max": 1.0,
        "volumetric_heat_capacity_min": 1.0e6,
        "thermal_diffusivity_min": 1.0e-8,
        "interfacial_deviation_max": 1.0,
    }
    result = screen_candidates(robust, ad, library, thresholds, {}).set_index(
        "candidate_id"
    )
    assert bool(result.loc["complete", "final_feasible"])
    assert not bool(result.loc["missing", "pass_inference"])
    assert not bool(result.loc["missing", "pass_AD"])
    assert not bool(result.loc["missing", "final_feasible"])
