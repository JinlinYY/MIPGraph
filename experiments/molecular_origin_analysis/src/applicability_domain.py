"""Descriptor-space applicability-domain audit fitted on training data only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass
class ApplicabilityDomainResult:
    query: pd.DataFrame
    thresholds: dict[str, float]
    metadata: dict[str, Any]


class ApplicabilityDomainAnalyzer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        ad = config["applicability_domain"]
        self.k = int(ad.get("descriptor_knn_k", 5))
        self.in_domain_quantile = float(ad.get("in_domain_quantile", 0.90))
        self.borderline_quantile = float(ad.get("borderline_quantile", 0.95))

    @staticmethod
    def _feature_columns(frame: pd.DataFrame) -> list[str]:
        return [
            column
            for column in frame.columns
            if column != "sample_id" and pd.api.types.is_numeric_dtype(frame[column])
        ]

    def evaluate(
        self,
        reference_descriptors: pd.DataFrame,
        query_descriptors: pd.DataFrame,
    ) -> ApplicabilityDomainResult:
        feature_columns = self._feature_columns(reference_descriptors)
        if feature_columns != self._feature_columns(query_descriptors):
            raise ValueError("Reference and query descriptor columns are not identical")
        reference = (
            reference_descriptors[feature_columns]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .drop_duplicates()
        )
        query = (
            query_descriptors[feature_columns]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        if len(reference) <= self.k:
            raise ValueError("Training reference set is too small for configured kNN")
        scaler = StandardScaler().fit(reference)
        reference_scaled = scaler.transform(reference)
        query_scaled = scaler.transform(query)
        reference_neighbors = NearestNeighbors(
            n_neighbors=min(self.k + 1, len(reference)),
        ).fit(reference_scaled)
        reference_distances, _ = reference_neighbors.kneighbors(reference_scaled)
        reference_score = reference_distances[:, 1:].mean(axis=1)
        in_threshold = float(np.quantile(reference_score, self.in_domain_quantile))
        borderline_threshold = float(
            np.quantile(reference_score, self.borderline_quantile)
        )
        query_neighbors = NearestNeighbors(
            n_neighbors=min(self.k, len(reference)),
        ).fit(reference_scaled)
        distances, indices = query_neighbors.kneighbors(query_scaled)
        scores = distances.mean(axis=1)
        status = np.where(
            scores <= in_threshold,
            "in_domain",
            np.where(scores <= borderline_threshold, "borderline", "out_of_domain"),
        )
        result = pd.DataFrame(
            {
                "sample_id": query_descriptors["sample_id"].to_numpy(),
                "descriptor_knn_distance": scores,
                "AD_status": status,
                "nearest_reference_row": indices[:, 0],
                "nearest_reference_distance": distances[:, 0],
            }
        )
        return ApplicabilityDomainResult(
            query=result,
            thresholds={
                "in_domain": in_threshold,
                "borderline": borderline_threshold,
            },
            metadata={
                "reference_rows_before_identity_deduplication": len(
                    reference_descriptors
                ),
                "reference_unique_descriptor_rows": len(reference),
                "query_rows": len(query),
                "k": self.k,
                "scaler_fit_scope": "training reference only",
                "feature_count": len(feature_columns),
            },
        )
