"""Regression tests for the refactored application-study decision contract."""

from pathlib import Path

import numpy as np
import pytest

from experiments.computational_application_case.scripts.build_protocol_stability_outputs import (
    validate_protocol_manifest,
)
from experiments.computational_application_case.src.config import load_case_config


ROOT = Path(__file__).resolve().parents[3]
CONFIG = (
    ROOT
    / "experiments"
    / "computational_application_case"
    / "configs"
    / "auditable_virtual_screening.yaml"
)


def _config() -> dict:
    return load_case_config(CONFIG)


def test_formal_predictions_use_one_primary_checkpoint() -> None:
    config = _config()
    assert config["model"]["checkpoint_paths"] == []
    assert "il_level_random_seed42" in config["model"]["checkpoint_path"]
    assert config["data"]["split_path"].endswith("splits/il_level_seed42.json")
    assert config["uncertainty"]["mode"] == "single_primary_checkpoint"


def test_main_window_and_stress_endpoints_are_separate() -> None:
    conditions = _config()["conditions"]
    assert conditions["temperature_start_K"] == 298.15
    assert conditions["temperature_end_K"] == 353.15
    assert conditions["extended_temperature_start_K"] == 278.15
    assert conditions["extended_temperature_end_K"] == 373.15
    assert conditions["run_extended_sensitivity"] is True


def test_pareto_has_only_four_declared_objectives() -> None:
    objectives = _config()["pareto"]["objectives"]
    assert objectives["maximize"] == [
        "conductivity_worst",
        "volumetric_heat_capacity_worst",
        "thermal_diffusivity_worst",
    ]
    assert objectives["minimize"] == ["viscosity_worst"]
    flattened = set(objectives["maximize"] + objectives["minimize"])
    assert "interfacial_deviation_worst" not in flattened
    assert "reference_cell_risk_index_worst" not in flattened


def test_protocol_configs_remain_sensitivity_only() -> None:
    case_dir = CONFIG.parents[1]
    expected_splits = {
        "protocol_stability_balanced.yaml": "il_level_property_balanced_seed42.json",
        "protocol_stability_ion_family.yaml": "il_level_family_pair_seed42.json",
    }
    for filename, split_name in expected_splits.items():
        config = load_case_config(case_dir / "configs" / filename)
        assert config["model"]["checkpoint_paths"] == []
        assert config["outputs"]["output_dir"] != _config()["outputs"]["output_dir"]
        assert config["data"]["split_path"].endswith(f"splits/{split_name}")


def test_protocol_manifest_rejects_a_stale_training_split() -> None:
    config_path = ROOT / "protocol.yaml"
    split_path = ROOT / "correct_split.json"
    temperatures = np.asarray([278.15, 298.15, 353.15, 373.15])
    manifest = {
        "config_path": str(config_path.resolve()),
        "training_split_path": str(split_path.resolve()),
        "candidate_identity_sha256": "identity-digest",
        "checkpoint_sha256": "checkpoint-digest",
        "candidate_count": 3,
        "temperature_grid_K": temperatures.tolist(),
        "prediction_rows": 12,
        "prediction_aggregation": "none; single checkpoint",
    }
    validate_protocol_manifest(
        manifest,
        config_path=config_path,
        training_split_path=split_path,
        candidate_identity_digest="identity-digest",
        checkpoint_digest="checkpoint-digest",
        candidate_count=3,
        temperature_grid_K=temperatures,
    )
    stale = dict(manifest)
    stale["training_split_path"] = str((ROOT / "wrong_split.json").resolve())
    with pytest.raises(RuntimeError, match="training_split_path"):
        validate_protocol_manifest(
            stale,
            config_path=config_path,
            training_split_path=split_path,
            candidate_identity_digest="identity-digest",
            checkpoint_digest="checkpoint-digest",
            candidate_count=3,
            temperature_grid_K=temperatures,
        )
