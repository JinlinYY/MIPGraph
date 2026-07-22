from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.computational_application_case.src.applicability_domain import (
    classify_ad_distance,
    assess_applicability_domain,
    fit_descriptor_ad,
    score_descriptor_ad,
)
from experiments.computational_application_case.src.chemistry import (
    ion_family,
    parse_monovalent_pair,
)
from experiments.computational_application_case.src.pipeline import CasePipeline


def test_standard_scaler_is_fit_only_on_reference_rows() -> None:
    reference = np.asarray([[0.0], [1.0], [2.0]])
    model = fit_descriptor_ad(reference, k=1, in_domain_quantile=0.5, borderline_quantile=0.9)
    score_descriptor_ad(model, np.asarray([[100.0]]))
    assert model.scaler.mean_[0] == pytest.approx(1.0)


def test_reference_distances_are_leave_one_out() -> None:
    model = fit_descriptor_ad(
        np.asarray([[0.0], [1.0], [2.0]]),
        k=1,
        in_domain_quantile=0.5,
        borderline_quantile=0.9,
    )
    assert model.reference_distances.tolist() == pytest.approx([1.22474487] * 3)
    assert np.all(model.reference_distances > 0.0)


def test_q90_and_q95_thresholds_are_calibrated_from_reference_distances() -> None:
    model = fit_descriptor_ad(
        np.asarray([[0.0], [1.0], [3.0], [10.0]]),
        k=1,
        in_domain_quantile=0.90,
        borderline_quantile=0.95,
    )
    assert model.in_domain_threshold == pytest.approx(
        np.quantile(model.reference_distances, 0.90)
    )
    assert model.borderline_threshold == pytest.approx(
        np.quantile(model.reference_distances, 0.95)
    )


def test_distance_classification_has_three_ordered_regions() -> None:
    assert classify_ad_distance(0.9, 1.0, 2.0) == "in_domain"
    assert classify_ad_distance(1.5, 1.0, 2.0) == "borderline"
    assert classify_ad_distance(2.1, 1.0, 2.0) == "out_of_domain"


def test_scoring_returns_in_borderline_and_out_of_domain() -> None:
    model = fit_descriptor_ad(
        np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0]]),
        k=1,
        in_domain_quantile=0.5,
        borderline_quantile=0.9,
    )
    model.in_domain_threshold = 1.0
    model.borderline_threshold = 2.0
    scored = score_descriptor_ad(model, np.asarray([[2.0], [6.0], [10.0]]))
    assert scored["AD_status"].tolist() == ["in_domain", "borderline", "out_of_domain"]


def test_constant_descriptor_columns_are_removed_before_scaling() -> None:
    reference = np.asarray([[5.0, 0.0], [5.0, 1.0], [5.0, 2.0]])
    model = fit_descriptor_ad(reference, k=1, in_domain_quantile=0.9, borderline_quantile=0.95)
    assert model.kept_columns.tolist() == [False, True]
    scored = score_descriptor_ad(model, np.asarray([[999.0, 1.0]]))
    assert np.isfinite(scored.loc[0, "descriptor_knn_distance"])


def test_low_ion_family_support_downgrades_in_domain_candidate() -> None:
    reference = pd.DataFrame(
        {"reference_id": ["r1", "r2", "r3"], "d": [0.0, 1.0, 2.0]}
    )
    candidates = pd.DataFrame({"candidate_id": ["A"], "d": [1.0]})
    metadata = pd.DataFrame(
        {
            "candidate_id": ["A"],
            "candidate_type": ["unseen_pair_recombination"],
            "cation_seen": [True],
            "anion_seen": [True],
            "cation_support_count": [10],
            "anion_support_count": [10],
            "cation_family_support": [1],
            "anion_family_support": [3],
            "temperature_domain_status": ["in_domain"],
        }
    )
    output, _ = assess_applicability_domain(
        candidates,
        reference,
        metadata,
        ["d"],
        {
            "descriptor_knn_k": 1,
            "in_domain_quantile": 0.9,
            "borderline_quantile": 0.95,
            "minimum_ion_support_for_in_domain": 5,
            "minimum_family_support_for_in_domain": 2,
        },
    )
    assert output.loc[0, "AD_status"] == "borderline"
    assert "low_ion_family_support" in output.loc[0, "AD_reason"]


def test_ad_metadata_recomputes_pair_and_ion_support_from_active_split() -> None:
    chloride = "C[N+](C)(C)C.[Cl-]"
    bromide = "C[N+](C)(C)C.[Br-]"
    parsed_chloride = parse_monovalent_pair(chloride)
    benchmark = pd.DataFrame(
        {
            "IL_SMILES": [chloride, chloride, bromide],
            "Temperature_K": [298.15, 303.15, 308.15],
            **{
                f"{name}_ActualValue": [1.0, 1.0, 1.0]
                for name in [
                    "Density",
                    "ElectricalConductivity",
                    "HeatCapacity",
                    "SurfaceTension",
                    "ThermalConductivity",
                    "Viscosity",
                ]
            },
        }
    )
    candidates = pd.DataFrame(
        {
            "candidate_id": ["A"],
            "candidate_type": ["unseen_pair_recombination"],
            "cation_identity_key": [parsed_chloride.cation_identity_key],
            "anion_identity_key": [parsed_chloride.anion_identity_key],
            "cation_family": [ion_family(parsed_chloride.cation_smiles, "cation")],
            "anion_family": [ion_family(parsed_chloride.anion_smiles, "anion")],
            "pair_seen_in_training": [False],
            "cation_support_count": [999],
            "anion_support_count": [999],
        }
    )
    pipeline = object.__new__(CasePipeline)
    pipeline.config = {
        "data": {"training_split_name": "train"},
        "conditions": {
            "temperature_start_K": 298.15,
            "temperature_end_K": 308.15,
            "temperature_step_K": 5.0,
        },
    }
    pipeline._benchmark = lambda: benchmark
    pipeline._split = lambda: {"train": [0, 1, 2]}
    metadata = pipeline._ad_metadata(candidates)
    assert bool(metadata.loc[0, "pair_seen"]) is True
    assert metadata.loc[0, "cation_support_count"] == 2
    assert metadata.loc[0, "anion_support_count"] == 1
