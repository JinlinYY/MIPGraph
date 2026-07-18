from __future__ import annotations

import pandas as pd
import pytest

from experiments.computational_application_case.src.schema import PROPERTY_NAMES
from experiments.computational_application_case.src.uncertainty import (
    estimate_uncertainty,
    validate_ensemble_compatibility,
)


def test_three_checkpoint_rows_produce_property_and_proxy_statistics() -> None:
    prediction_rows = []
    proxy_rows = []
    for index, checkpoint in enumerate(["a", "b", "c"], start=1):
        prediction = {
            "candidate_id": "A",
            "temperature_K": 298.15,
            "pressure_kPa": 101.325,
            "checkpoint_name": checkpoint,
        }
        prediction.update({name: float(index) for name in PROPERTY_NAMES})
        prediction_rows.append(prediction)
        proxy_rows.append(
            {
                "candidate_id": "A",
                "temperature_K": 298.15,
                "pressure_kPa": 101.325,
                "checkpoint_name": checkpoint,
                "thermal_diffusivity": float(index) * 1.0e-7,
            }
        )
    properties, proxies, probabilities, status = estimate_uncertainty(
        pd.DataFrame(prediction_rows),
        pd.DataFrame(proxy_rows),
        ["a.pt", "b.pt", "c.pt"],
        {"ensemble_min_checkpoints": 3},
    )
    assert status["uncertainty_status"] == "checkpoint_ensemble"
    assert properties.loc[0, "Density_mean"] == pytest.approx(2.0)
    assert properties.loc[0, "Density_std"] == pytest.approx(1.0)
    assert proxies.loc[0, "thermal_diffusivity_mean"] == pytest.approx(2.0e-7)
    assert probabilities["constraint_pass_probability"].isna().all()


def test_incomplete_ensemble_rows_fail_to_unavailable_status() -> None:
    predictions = pd.DataFrame(
        {
            "candidate_id": ["A", "A"],
            "temperature_K": [298.15, 298.15],
            "pressure_kPa": [101.325, 101.325],
            "checkpoint_name": ["a", "b"],
            **{name: [1.0, 2.0] for name in PROPERTY_NAMES},
        }
    )
    proxies = predictions[
        ["candidate_id", "temperature_K", "pressure_kPa", "checkpoint_name"]
    ].copy()
    proxies["thermal_diffusivity"] = 1.0e-7
    _, _, probabilities, status = estimate_uncertainty(
        predictions,
        proxies,
        ["a.pt", "b.pt", "c.pt"],
        {"ensemble_min_checkpoints": 3},
    )
    assert status["uncertainty_status"] == "not_available"
    assert probabilities["constraint_pass_probability"].isna().all()


def test_ensemble_compatibility_rejects_preprocessing_mismatch() -> None:
    reference = {
        "model_class": "m.Model",
        "model_structure_fingerprint": "model-a",
        "property_order": list(PROPERTY_NAMES),
        "property_units": {name: "unit" for name in PROPERTY_NAMES},
        "graph_config_fingerprint": "graph-a",
        "graph_feature_dimension": 8,
        "global_descriptor_dimension": 4,
        "functional_group_dimension": 2,
        "condition_scaler_class": "scaler.ConditionScaler",
        "target_scaler_class": "scaler.TargetScaler",
        "target_inverse_transform": "exp",
        "target_epsilon": 1.0e-8,
    }
    incompatible = dict(reference, graph_config_fingerprint="graph-b")
    with pytest.raises(ValueError, match="incompatible graph_config_fingerprint"):
        validate_ensemble_compatibility([reference, incompatible])


def test_ensemble_compatibility_allows_different_scaler_parameters() -> None:
    shared = {
        "model_class": "m.Model",
        "model_structure_fingerprint": "model-a",
        "property_order": list(PROPERTY_NAMES),
        "property_units": {name: "unit" for name in PROPERTY_NAMES},
        "graph_config_fingerprint": "graph-a",
        "graph_feature_dimension": 8,
        "global_descriptor_dimension": 4,
        "functional_group_dimension": 2,
        "condition_scaler_class": "scaler.ConditionScaler",
        "target_scaler_class": "scaler.TargetScaler",
        "target_inverse_transform": "exp",
        "target_epsilon": 1.0e-8,
    }
    validate_ensemble_compatibility(
        [
            dict(shared, target_means=[1.0], target_stds=[2.0]),
            dict(shared, target_means=[3.0], target_stds=[4.0]),
        ]
    )
