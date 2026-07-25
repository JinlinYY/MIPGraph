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


def test_composite_figure_exports_four_traceable_panels(
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
                "structural_factor": f"factor_{index}_{rank}",
                "effect_direction": (
                    "positive" if (index + rank) % 2 == 0 else "negative"
                ),
                "confidence_level": "Level B",
                "family_consistency": 0.8 - 0.05 * rank,
                    "statistical_evidence": (
                        f"partial r="
                        f"{(-1 if (index + rank) % 2 else 1) * (0.2 + 0.05 * index + 0.02 * rank):.3f}; "
                        "q=1e-4"
                    ),
                    "attribution_evidence": (
                        f"direct gradient×input rank={rank + 1}"
                    ),
                }
            for index, property_name in enumerate(PROPERTY_ORDER)
            for rank in range(3)
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
                "feature": f"factor_{property_index}_2",
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
    with tempfile.TemporaryDirectory(
        prefix="composite_",
        dir=module_root,
    ) as directory:
        stem = Path(directory) / "figures" / "composite"
        outputs = PublicationPlotter(config).composite_results_figure(
            rules,
            associations,
            nonlinear,
            matched_pairs,
            contrasts,
            stem,
        )
        source_root = Path(directory) / "tables" / "figure_source_data"

        assert {path.suffix for path in outputs} == {".png", ".tiff"}
        assert all(path.is_file() for path in outputs)
        assert all(
            (source_root / f"composite_panel_{panel}_source_data.csv").is_file()
            for panel in "abcd"
        )
        panel_a = pd.read_csv(
            source_root / "composite_panel_a_source_data.csv"
        )
        assert set(panel_a["evidence_family"]) == {
            "condition_controlled_association",
            "cross_ion_attention_contrast",
        }
        association_a = panel_a.loc[
            panel_a["evidence_family"]
            == "condition_controlled_association"
        ]
        attention_a = panel_a.loc[
            panel_a["evidence_family"]
            == "cross_ion_attention_contrast"
        ]
        assert association_a.groupby("property").size().eq(3).all()
        assert attention_a.groupby("property").size().eq(3).all()
        assert panel_a["line_width_pt"].nunique() > 1
        assert association_a["selection_rule"].str.contains(
            "top three Level A/B links"
        ).all()
        panel_c = pd.read_csv(
            source_root / "composite_panel_c_source_data.csv"
        )
        assert "response_log_centered" in panel_c
        centered_means = panel_c.groupby(
            ["property", "feature"]
        )["response_log_centered"].mean()
        assert centered_means.abs().lt(1e-10).all()
        assert set(panel_c["structural_scope"]).issubset(
            {"Cation", "Anion", "Ion pair"}
        )
        assert panel_c["plot_color_hex"].notna().all()
        assert panel_c["selection_rank"].between(1, 3).all()
        assert {
            "plot_line_style",
            "plot_marker",
            "confidence_level",
            "selection_rule",
        }.issubset(panel_c.columns)
        assert panel_c.groupby("structural_scope")[
            "plot_color_hex"
        ].nunique().eq(1).all()
        assert not (
            source_root / "composite_panel_e_source_data.csv"
        ).exists()
        assert not (
            source_root / "composite_panel_f_source_data.csv"
        ).exists()
