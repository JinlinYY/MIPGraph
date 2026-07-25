"""Behavioural tests for the read-only project-discovery seam."""

from __future__ import annotations

import hashlib
from pathlib import Path

from experiments.molecular_origin_analysis.src.project_adapter import ProjectAdapter
from experiments.molecular_origin_analysis.src.utils import load_config, resolve_path


EXPECTED_PROPERTIES = [
    "Density",
    "ElectricalConductivity",
    "HeatCapacity",
    "SurfaceTension",
    "ThermalConductivity",
    "Viscosity",
]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_discovery_finds_real_model_data_and_checkpoint_without_mutation(
    config_path: Path,
) -> None:
    config = load_config(config_path)
    project_root = resolve_path(config["project"]["root"], config["_module_root"])
    protected = (
        project_root
        / "il_property_prediction"
        / "src"
        / "models"
        / "mipgraph.py"
    )
    before = _digest(protected)

    report = ProjectAdapter(config).inspect()

    assert report.model_class == "src.models.mipgraph.MIPGraph"
    assert report.property_names == EXPECTED_PROPERTIES
    assert report.descriptor_dimensions == {"global": 56, "functional_group": 80}
    assert report.selected_checkpoint.is_file()
    assert report.data_paths["clean_csv"].is_file()
    assert report.data_paths["arrays"].is_file()
    assert report.data_paths["graph_cache"].is_file()
    assert report.split_paths["row_level"].is_file()
    assert report.cross_ion_attention_access == "forward_auxiliary"
    assert report.router_access == "forward_auxiliary"
    assert _digest(protected) == before


def test_all_resolved_output_paths_remain_inside_new_module(config_path: Path) -> None:
    config = load_config(config_path)
    adapter = ProjectAdapter(config)
    output_root = adapter.output_root.resolve()
    module_root = adapter.module_root.resolve()

    assert output_root.is_relative_to(module_root)
