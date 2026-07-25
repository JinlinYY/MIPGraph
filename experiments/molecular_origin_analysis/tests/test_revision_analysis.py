"""Smoke tests for the identity-balanced revision workflow."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from experiments.molecular_origin_analysis.src.plotting import (
    PROPERTY_ORDER,
    PublicationPlotter,
)
from experiments.molecular_origin_analysis.src.revision_analysis import (
    aggregate_identity_responses,
    build_condition_covariates,
)
from experiments.molecular_origin_analysis.src.utils import load_config


def _identity_records() -> pd.DataFrame:
    rows = []
    sample_id = 0
    for identity_index in range(8):
        for temperature in (298.15, 323.15):
            rows.append(
                {
                    "sample_id": sample_id,
                    "cation_smiles": f"C{identity_index}",
                    "anion_smiles": f"A{identity_index % 3}",
                    "cation_family": f"CF{identity_index % 2}",
                    "anion_family": f"AF{identity_index % 3}",
                    "Temperature_K": temperature,
                    "Pressure_kPa": (
                        np.nan if identity_index % 4 == 0 else 101.325
                    ),
                    "Density_ActualValue": (
                        900.0 + 3.0 * identity_index + temperature / 10
                    ),
                    "Density_mask": 1.0,
                }
            )
            sample_id += 1
    return pd.DataFrame(rows)


def test_identity_aggregation_and_pressure_indicator() -> None:
    records = _identity_records()
    covariates, names, metadata = build_condition_covariates(
        records,
        temperature_knots=4,
        reference_pressure_kpa=101.325,
    )
    identity, valid, _, aggregation_metadata = aggregate_identity_responses(
        records,
        "Density",
        temperature_knots=4,
        reference_pressure_kpa=101.325,
    )

    assert covariates.shape[0] == len(records)
    assert "pressure_available" in names
    assert metadata["pressure_missing_count"] == 4
    assert valid.sum() == len(records)
    assert len(identity) == 8
    assert identity["il_identity_key"].is_unique
    assert identity["n_records"].eq(2).all()
    assert aggregation_metadata["n_unique_ils"] == 8


def test_v2_composite_exports_auditable_panels(
    config_path,
    module_root,
) -> None:
    config = load_config(config_path)
    config["figures"]["formats"] = ["png"]
    config["figures"]["dpi"] = 72
    associations = []
    nonlinear = []
    for property_index, property_name in enumerate(PROPERTY_ORDER):
        for rank, scope in enumerate(("Cation", "Anion", "Ion pair"), start=1):
            feature = f"{scope.lower().replace(' ', '_')}_{property_index}_{rank}"
            effect = (-1 if rank % 2 else 1) * (0.25 + rank / 10)
            associations.append(
                {
                    "property": property_name,
                    "feature": feature,
                    "structural_scope": scope,
                    "partial_correlation": effect,
                    "main_figure_eligible": True,
                    "eligibility_status": "eligible",
                    "selection_rank": rank,
                }
            )
            for quantile in range(1, 6):
                value = effect * (quantile - 3) / 4
                nonlinear.append(
                    {
                        "property": property_name,
                        "feature": feature,
                        "feature_cluster": f"cluster_{rank}",
                        "structural_scope": scope,
                        "selection_rank": rank,
                        "main_figure_eligible": True,
                        "quantile_bin": quantile,
                        "n_unique_ils": 40 + property_index,
                        "response_log_mean": value,
                        "bootstrap_ci_low": value - 0.05,
                        "bootstrap_ci_high": value + 0.05,
                    }
                )
    matched = []
    for property_name in PROPERTY_ORDER:
        for role in ("anion_fixed", "cation_fixed"):
            for pair_index in range(6):
                matched.append(
                    {
                        "property": property_name,
                        "fixed_role": role,
                        "substitution_pair_id": (
                            f"{property_name}-{role}-{pair_index}"
                        ),
                        "observed_abs_log_difference": 0.02
                        + 0.03 * pair_index,
                    }
                )
    contrasts = pd.DataFrame(
        [
            {
                "property": property_name,
                "interaction_category": f"category-{property_index}",
                "attention_metric": "attention_per_pair",
                "high_minus_low": 0.01 * (-1) ** property_index,
                "main_figure_eligible": True,
            }
            for property_index, property_name in enumerate(PROPERTY_ORDER)
        ]
    )
    with tempfile.TemporaryDirectory(
        prefix="revision_plot_",
        dir=module_root,
    ) as directory:
        stem = Path(directory) / "figures" / "revision"
        outputs = PublicationPlotter(config).composite_results_figure_v2(
            pd.DataFrame(),
            pd.DataFrame(associations),
            pd.DataFrame(nonlinear),
            pd.DataFrame(matched),
            contrasts,
            stem,
        )
        source_root = Path(directory) / "tables" / "figure_source_data"

        assert {path.suffix for path in outputs} == {".png", ".tiff"}
        assert all(path.is_file() for path in outputs)
        assert all(
            (source_root / f"revision_panel_{panel}_source_data.csv").is_file()
            for panel in "abcd"
        )
        panel_c = pd.read_csv(
            source_root / "revision_panel_c_source_data.csv"
        )
        assert panel_c.groupby("property")["feature_cluster"].nunique().eq(3).all()
        assert set(panel_c["quantile_bin"]) == {1, 2, 3, 4, 5}
