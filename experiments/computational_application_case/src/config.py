"""YAML loading, inheritance, command-line overrides, and schema checks."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from .paths import ensure_output_within_case, locate_project_root, resolve_project_path
from .schema import PROPERTY_NAMES


EXPECTED_PROPERTIES = list(PROPERTY_NAMES)


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_with_inheritance(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"Circular YAML inheritance detected at {path}")
    seen.add(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    parent = payload.get("extends")
    if parent is None:
        return payload
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return _deep_merge(_read_with_inheritance(parent_path, seen), payload)


def temperature_grid(config: Mapping[str, Any], extended: bool = False) -> np.ndarray:
    """Return an inclusive, validated temperature grid from configuration."""

    if extended:
        start = float(config["extended_temperature_start_K"])
        end = float(config["extended_temperature_end_K"])
    else:
        start = float(config["temperature_start_K"])
        end = float(config["temperature_end_K"])
    step = float(config["temperature_step_K"])
    if not np.isfinite([start, end, step]).all() or start <= 0 or end < start or step <= 0:
        raise ValueError("Temperature bounds and step must be finite, positive, and ordered")
    count = int(np.floor((end - start) / step + 1.0e-9)) + 1
    values = start + np.arange(count, dtype=float) * step
    if values[-1] < end - 1.0e-8:
        values = np.append(values, end)
    return np.round(values, 10)


def apply_overrides(config: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Apply supported CLI overrides without mutating the loaded mapping."""

    output = copy.deepcopy(config)
    mapping = {
        "checkpoint": ("model", "checkpoint_path"),
        "checkpoints": ("model", "checkpoint_paths"),
        "device": ("model", "device"),
        "batch_size": ("model", "batch_size"),
        "output_dir": ("outputs", "output_dir"),
    }
    for argument, path in mapping.items():
        value = overrides.get(argument)
        if value is None:
            continue
        output[path[0]][path[1]] = value
    return output


def validate_config(config: dict[str, Any], root: Path) -> dict[str, Any]:
    """Validate required sections, values, artefact paths, and output boundary."""

    sections = [
        "project",
        "model",
        "data",
        "candidate_generation",
        "conditions",
        "proxies",
        "reference_cell",
        "applicability_domain",
        "uncertainty",
        "screening",
        "pareto",
        "figures",
        "outputs",
    ]
    missing_sections = [section for section in sections if section not in config]
    if missing_sections:
        raise ValueError(f"Configuration lacks required sections: {missing_sections}")
    temperature_grid(config["conditions"])
    if float(config["conditions"]["pressure_kPa"]) <= 0.0:
        raise ValueError("Pressure must be positive")
    for key in [
        "electrode_area_cm2",
        "separator_thickness_um",
        "electrolyte_volume_mL",
        "nominal_capacitance_F",
        "charge_discharge_current_A",
        "convective_heat_transfer_coefficient_W_m2_K",
        "exposed_face_count",
        "transient_duration_s",
        "reference_temperature_K",
    ]:
        value = float(config["reference_cell"].get(key, float("nan")))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"reference_cell.{key} must be finite and positive")
    reference_temperature = float(config["reference_cell"]["reference_temperature_K"])
    if not np.isclose(temperature_grid(config["conditions"]), reference_temperature).any():
        raise ValueError(
            "reference_cell.reference_temperature_K must occur on the main temperature grid"
        )
    for key in ["batch_size"]:
        if int(config["model"][key]) <= 0:
            raise ValueError(f"model.{key} must be positive")
    for key in ["max_candidates", "max_observed_references"]:
        if int(config["candidate_generation"][key]) <= 0:
            raise ValueError(f"candidate_generation.{key} must be positive")
    audited_units = config["data"].get("audited_units", {})
    if set(audited_units) != set(EXPECTED_PROPERTIES):
        raise ValueError("data.audited_units must define exactly the six model properties")
    output_dir = ensure_output_within_case(root, config["outputs"]["output_dir"])
    config["_project_root"] = str(root)
    config["_output_dir"] = str(output_dir)
    for section, keys in {
        "model": [
            "config_path",
            "checkpoint_path",
            "graph_cache_path",
            "unimol2_feature_cache_path",
        ],
        "data": ["benchmark_path", "arrays_path", "split_path"],
    }.items():
        for key in keys:
            value = config[section].get(key)
            if value is None:
                if key == "checkpoint_path" and config["model"].get("checkpoint_paths"):
                    continue
                raise ValueError(f"Explicit configuration is required for {section}.{key}")
            path = resolve_project_path(root, value)
            if not path.exists():
                raise FileNotFoundError(f"Configured artefact does not exist: {path}")
    for checkpoint in config["model"].get("checkpoint_paths", []):
        path = resolve_project_path(root, checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"Configured ensemble checkpoint does not exist: {path}")
    return config


def load_case_config(
    path: str | Path,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load an inherited case YAML and resolve it against the project root."""

    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Case configuration not found: {path}")
    config = _read_with_inheritance(path)
    config = apply_overrides(config, overrides or {})
    requested_root = config.get("project", {}).get("root", "auto")
    root = locate_project_root(path) if requested_root == "auto" else Path(requested_root).resolve()
    config["_config_path"] = str(path)
    return validate_config(config, root)
