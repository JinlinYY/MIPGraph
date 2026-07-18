"""Reference-calibrated descriptor-space applicability-domain analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass
class DescriptorADModel:
    """Fitted descriptor AD with leave-one-out calibration distances."""

    scaler: StandardScaler
    kept_columns: np.ndarray
    reference_scaled: np.ndarray
    reference_distances: np.ndarray
    k: int
    in_domain_quantile: float
    borderline_quantile: float
    in_domain_threshold: float
    borderline_threshold: float
    all_columns_constant: bool = False


def _as_finite_matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if matrix.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return matrix


def fit_descriptor_ad(
    reference_descriptors: np.ndarray,
    k: int = 5,
    in_domain_quantile: float = 0.90,
    borderline_quantile: float = 0.95,
) -> DescriptorADModel:
    """Fit reference-only scaling and leave-one-out kNN thresholds."""

    reference = _as_finite_matrix(reference_descriptors, "reference descriptors")
    if reference.shape[0] < 2:
        raise ValueError("Descriptor AD needs at least two reference rows")
    if not 0.0 < in_domain_quantile < borderline_quantile <= 1.0:
        raise ValueError("AD quantiles must satisfy 0 < in < borderline <= 1")
    variance = np.var(reference, axis=0)
    kept_columns = np.isfinite(variance) & (variance > 1.0e-12)
    all_constant = not bool(kept_columns.any())
    selected = (
        reference[:, kept_columns]
        if not all_constant
        else np.zeros((reference.shape[0], 1), dtype=float)
    )
    scaler = StandardScaler().fit(selected)
    reference_scaled = scaler.transform(selected)
    effective_k = min(max(int(k), 1), reference.shape[0] - 1)
    neighbours = NearestNeighbors(n_neighbors=effective_k + 1, metric="euclidean")
    neighbours.fit(reference_scaled)
    distances, indices = neighbours.kneighbors(reference_scaled)
    loo_rows = []
    for row_index, (row_distances, row_indices) in enumerate(zip(distances, indices)):
        nonself = row_distances[row_indices != row_index][:effective_k]
        if nonself.size != effective_k:
            raise RuntimeError("Leave-one-out neighbour calibration failed")
        loo_rows.append(float(np.mean(nonself)))
    reference_distances = np.asarray(loo_rows, dtype=float)
    in_threshold = float(np.quantile(reference_distances, in_domain_quantile))
    borderline_threshold = float(
        np.quantile(reference_distances, borderline_quantile)
    )
    return DescriptorADModel(
        scaler=scaler,
        kept_columns=kept_columns,
        reference_scaled=reference_scaled,
        reference_distances=reference_distances,
        k=effective_k,
        in_domain_quantile=float(in_domain_quantile),
        borderline_quantile=float(borderline_quantile),
        in_domain_threshold=in_threshold,
        borderline_threshold=borderline_threshold,
        all_columns_constant=all_constant,
    )


def classify_ad_distance(
    distance: float,
    in_domain_threshold: float,
    borderline_threshold: float,
) -> str:
    """Classify one distance against nested reference quantile thresholds."""

    if not np.isfinite(distance):
        return "out_of_domain"
    if distance <= in_domain_threshold:
        return "in_domain"
    if distance <= borderline_threshold:
        return "borderline"
    return "out_of_domain"


def score_descriptor_ad(
    model: DescriptorADModel,
    candidate_descriptors: np.ndarray,
) -> pd.DataFrame:
    """Score candidate descriptors using a fitted reference AD model."""

    candidates = _as_finite_matrix(candidate_descriptors, "candidate descriptors")
    if candidates.shape[1] != model.kept_columns.size:
        raise ValueError(
            "Candidate descriptor width does not match the reference descriptor width"
        )
    selected = (
        candidates[:, model.kept_columns]
        if not model.all_columns_constant
        else np.zeros((candidates.shape[0], 1), dtype=float)
    )
    candidate_scaled = model.scaler.transform(selected)
    neighbours = NearestNeighbors(n_neighbors=model.k, metric="euclidean")
    neighbours.fit(model.reference_scaled)
    distances, _ = neighbours.kneighbors(candidate_scaled)
    mean_distance = distances.mean(axis=1)
    percentiles = np.asarray(
        [
            float(np.mean(model.reference_distances <= distance))
            for distance in mean_distance
        ],
        dtype=float,
    )
    statuses = [
        classify_ad_distance(
            float(distance), model.in_domain_threshold, model.borderline_threshold
        )
        for distance in mean_distance
    ]
    return pd.DataFrame(
        {
            "descriptor_knn_distance": mean_distance,
            "descriptor_distance_percentile": percentiles,
            "AD_status": statuses,
        }
    )


def assess_applicability_domain(
    candidate_features: pd.DataFrame,
    reference_features: pd.DataFrame,
    metadata: pd.DataFrame,
    descriptor_columns: Sequence[str],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, DescriptorADModel]:
    """Assess descriptor AD and combine it with ion-support risk metadata."""

    if "candidate_id" not in candidate_features or "candidate_id" not in metadata:
        raise ValueError("Candidate features and metadata must contain candidate_id")
    missing_features = [
        column
        for column in descriptor_columns
        if column not in candidate_features or column not in reference_features
    ]
    if missing_features:
        raise ValueError(f"Missing AD descriptor columns: {missing_features}")
    model = fit_descriptor_ad(
        reference_features[list(descriptor_columns)].to_numpy(dtype=float),
        k=int(config["descriptor_knn_k"]),
        in_domain_quantile=float(config["in_domain_quantile"]),
        borderline_quantile=float(config["borderline_quantile"]),
    )
    scored = score_descriptor_ad(
        model, candidate_features[list(descriptor_columns)].to_numpy(dtype=float)
    )
    scored.insert(0, "candidate_id", candidate_features["candidate_id"].to_numpy())
    output = metadata.merge(scored, on="candidate_id", how="inner", validate="one_to_one")
    minimum_support = int(config.get("minimum_ion_support_for_in_domain", 1))
    minimum_family_support = int(
        config.get("minimum_family_support_for_in_domain", 0)
    )
    reasons: list[str] = []
    final_statuses: list[str] = []
    for row in output.itertuples(index=False):
        status = str(row.AD_status)
        row_reasons = [f"descriptor_distance={status}"]
        cation_seen = bool(getattr(row, "cation_seen", True))
        anion_seen = bool(getattr(row, "anion_seen", True))
        cation_support = int(getattr(row, "cation_support_count", 0))
        anion_support = int(getattr(row, "anion_support_count", 0))
        cation_family_support = int(getattr(row, "cation_family_support", 0))
        anion_family_support = int(getattr(row, "anion_family_support", 0))
        temperature_status = str(getattr(row, "temperature_domain_status", "in_domain"))
        candidate_type = str(getattr(row, "candidate_type", ""))
        if not cation_seen or not anion_seen or candidate_type == "one_ion_extrapolation":
            status = "out_of_domain"
            row_reasons.append("unseen_ion_component")
        elif status == "in_domain" and (
            cation_support < minimum_support or anion_support < minimum_support
        ):
            status = "borderline"
            row_reasons.append("low_ion_support")
        if status == "in_domain" and (
            cation_family_support < minimum_family_support
            or anion_family_support < minimum_family_support
        ):
            status = "borderline"
            row_reasons.append("low_ion_family_support")
        if temperature_status != "in_domain" and status == "in_domain":
            status = "borderline"
            row_reasons.append("temperature_extrapolation")
        final_statuses.append(status)
        reasons.append(";".join(row_reasons))
    output["AD_status"] = final_statuses
    output["AD_reason"] = reasons
    output["embedding_knn_distance"] = np.nan
    output["embedding_distance_percentile"] = np.nan
    output["embedding_distance_status"] = "not_available"
    output["descriptor_in_domain_threshold"] = model.in_domain_threshold
    output["descriptor_borderline_threshold"] = model.borderline_threshold
    return output, model
