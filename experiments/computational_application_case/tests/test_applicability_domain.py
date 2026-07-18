from __future__ import annotations

import numpy as np
import pytest

from experiments.computational_application_case.src.applicability_domain import (
    classify_ad_distance,
    fit_descriptor_ad,
    score_descriptor_ad,
)


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
