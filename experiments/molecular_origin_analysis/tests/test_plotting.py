"""Regression tests for deterministic, property-balanced figure selection."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from experiments.molecular_origin_analysis.src.plotting import (
    PROPERTY_ORDER,
    PublicationPlotter,
)
from experiments.molecular_origin_analysis.src.utils import load_config


def test_evidence_map_represents_every_property(config_path, module_root) -> None:
    config = load_config(config_path)
    config["figures"]["formats"] = ["png"]
    config["figures"]["dpi"] = 72
    records = []
    for property_name in PROPERTY_ORDER:
        for rank in range(6):
            records.append(
                {
                    "property": property_name,
                    "structural_factor": f"{property_name}_factor_{rank}",
                    "effect_direction": "positive" if rank % 2 == 0 else "negative",
                    "confidence_level": "Level B" if rank < 2 else "Level C",
                    "family_consistency": 1.0 - 0.1 * rank,
                }
            )
    rules = pd.DataFrame.from_records(records)

    with tempfile.TemporaryDirectory(prefix="plot_", dir=module_root) as directory:
        stem = Path(directory) / "figures" / "evidence_map"
        PublicationPlotter(config).evidence_map(rules, stem)
        source_path = (
            Path(directory)
            / "tables"
            / "figure_source_data"
            / "evidence_map_source_data.csv"
        )
        source = pd.read_csv(source_path)

        assert set(source["property"]) == set(PROPERTY_ORDER)
        assert source.groupby("property").size().eq(4).all()


def test_composite_figure_exports_six_traceable_panels(
    config_path,
    module_root,
) -> None:
    config = load_config(config_path)
    config["figures"]["formats"] = ["png"]
    config["figures"]["dpi"] = 72
    rules = pd.DataFrame(
        [
            {
                "property": property_name,
                "structural_factor": f"factor_{index}",
                "effect_direction": "positive" if index % 2 == 0 else "negative",
                "confidence_level": "Level B",
                "family_consistency": 0.8,
                "statistical_evidence": (
                    f"partial r={0.2 + 0.05 * index:.3f}; q=1e-4"
                ),
            }
            for index, property_name in enumerate(PROPERTY_ORDER)
        ]
    )
    association_records = []
    for feature_index in range(12):
        for property_index, property_name in enumerate(PROPERTY_ORDER):
            association_records.append(
                {
                    "feature": f"feature_{feature_index}",
                    "property": property_name,
                    "data_type": "experimental",
                    "partial_correlation": (
                        ((feature_index + property_index) % 7 - 3) / 8
                    ),
                    "fdr_q": 0.01,
                }
            )
    associations = pd.DataFrame(association_records)
    nonlinear = pd.DataFrame(
        [
            {
                "property": property_name,
                "feature": f"response_factor_{property_index}",
                "monotonic_bin_spearman": 0.9,
                "feature_mean": float(bin_index),
                "response_log_mean": 1.0 + 0.1 * bin_index,
                "response_log_sem": 0.02,
                "sample_count": 10,
            }
            for property_index, property_name in enumerate(PROPERTY_ORDER)
            for bin_index in range(3)
        ]
    )
    temperatures = [298.15, 323.15, 348.15]
    predictions = pd.DataFrame(
        [
            {
                "candidate_id": f"CF-{candidate_index:04d}",
                "temperature_K": temperature,
                "Viscosity": np.exp(2.0 - 0.01 * (temperature - 298.15)),
                "ElectricalConductivity": np.exp(
                    -1.0 + 0.01 * (temperature - 298.15)
                ),
                "AD_status": "in_domain",
            }
            for candidate_index in range(3)
            for temperature in temperatures
        ]
    )
    matched_pairs = pd.DataFrame(
        [
            {
                "fixed_role": role,
                "left_sample_id": f"L{index}",
                "right_sample_id": f"R{index}",
                "observed_abs_log_difference_Viscosity": 0.2 + 0.1 * index,
                "observed_abs_log_difference_ElectricalConductivity": (
                    0.3 + 0.1 * index
                ),
            }
            for index, role in enumerate(
                ["anion_fixed", "cation_fixed"] * 3
            )
        ]
    )
    contrasts = pd.DataFrame(
        [
            {
                "interaction_category": f"category-{category_index}",
                "property": property_name,
                "high_minus_low": (
                    ((category_index + property_index) % 5 - 2) / 20
                ),
            }
            for category_index in range(8)
            for property_index, property_name in enumerate(PROPERTY_ORDER)
        ]
    )
    trajectory = pd.DataFrame(
        [
            {
                "candidate_id": f"UPR-{index:04d}",
                "viscosity_worst": 0.01 * (index + 1),
                "conductivity_worst": 0.1 * (index + 1),
                "thermal_diffusivity_worst": 1e-7 * (1 + 0.02 * index),
                "volumetric_heat_capacity_worst": 1e6 * (1 + 0.03 * index),
            }
            for index in range(10)
        ]
    )
    top8 = trajectory.head(8)[["candidate_id"]].copy()

    with tempfile.TemporaryDirectory(
        prefix="composite_",
        dir=module_root,
    ) as directory:
        stem = Path(directory) / "figures" / "composite"
        outputs = PublicationPlotter(config).composite_results_figure(
            rules,
            associations,
            nonlinear,
            predictions,
            matched_pairs,
            contrasts,
            trajectory,
            top8,
            stem,
        )
        source_root = Path(directory) / "tables" / "figure_source_data"

        assert {path.suffix for path in outputs} == {".png", ".tiff"}
        assert all(path.is_file() for path in outputs)
        assert all(
            (source_root / f"composite_panel_{panel}_source_data.csv").is_file()
            for panel in "abcdef"
        )
