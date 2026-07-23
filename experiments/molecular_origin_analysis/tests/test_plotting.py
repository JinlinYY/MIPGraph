"""Regression tests for deterministic, property-balanced figure selection."""

from __future__ import annotations

from pathlib import Path
import tempfile

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
