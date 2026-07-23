"""Behavioural tests for descriptor naming, alignment, and cache round-trips."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from experiments.molecular_origin_analysis.src.feature_extractor import (
    DESCRIPTOR_NAMES,
    FUNCTIONAL_GROUP_NAMES,
    FeatureBundle,
    FeatureExtractor,
)


def test_descriptor_names_match_confirmed_model_dimensions() -> None:
    assert len(DESCRIPTOR_NAMES) == 56
    assert len(FUNCTIONAL_GROUP_NAMES) == 80
    assert DESCRIPTOR_NAMES[0] == "cation_molecular_weight_scaled"
    assert DESCRIPTOR_NAMES[-1] == "pair_radius_of_gyration_difference"
    assert FUNCTIONAL_GROUP_NAMES[0] == "cation_hbond_donor"
    assert FUNCTIONAL_GROUP_NAMES[-1] == "pair_functional_group_dot"


def test_feature_bundle_round_trip_preserves_identity_and_values(module_root) -> None:
    bundle = FeatureBundle(
        records=pd.DataFrame(
            {
                "sample_id": [17, 31],
                "IL_SMILES": ["[Na+].[Cl-]", "C[N+](C)(C)C.[Br-]"],
                "prediction_Density": [1000.0, 1100.0],
            }
        ),
        descriptors=pd.DataFrame(
            {
                "sample_id": [17, 31],
                DESCRIPTOR_NAMES[0]: [0.058, 0.074],
            }
        ),
        latent_arrays={"ion_pair_embedding": np.arange(8, dtype=np.float32).reshape(2, 4)},
        metadata={"property_order": ["Density"]},
    )
    with tempfile.TemporaryDirectory(
        prefix="round_trip_",
        dir=module_root,
    ) as temporary_directory:
        path = Path(temporary_directory) / "test_feature_bundle"
        FeatureExtractor.save_bundle(bundle, path)
        restored = FeatureExtractor.load_bundle(path)

        assert restored.records["sample_id"].tolist() == [17, 31]
        assert (
            restored.records["IL_SMILES"].tolist()
            == bundle.records["IL_SMILES"].tolist()
        )
        np.testing.assert_allclose(
            restored.latent_arrays["ion_pair_embedding"],
            bundle.latent_arrays["ion_pair_embedding"],
        )
        assert restored.metadata["property_order"] == ["Density"]
