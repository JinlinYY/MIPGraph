"""Small real-checkpoint smoke test for the adapter/inference seam."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from experiments.molecular_origin_analysis.src.counterfactual import CounterfactualGenerator
from experiments.molecular_origin_analysis.src.data_adapter import DataAdapter
from experiments.molecular_origin_analysis.src.feature_extractor import (
    FeatureExtractor,
)
from experiments.molecular_origin_analysis.src.model_adapter import ModelAdapter
from experiments.molecular_origin_analysis.src.plotting import PublicationPlotter
from experiments.molecular_origin_analysis.src.project_adapter import ProjectAdapter
from experiments.molecular_origin_analysis.src.utils import (
    load_config,
    write_json,
)


def test_real_checkpoint_smoke_pipeline(
    config_path,
    module_root,
) -> None:
    sample_count = 16
    config = load_config(
        config_path,
        overrides={
            "model.device": "cpu",
            "model.batch_size": 8,
            "data.max_samples": sample_count,
        },
    )
    project = ProjectAdapter(config)
    inspection = project.inspect()
    data = DataAdapter(config, inspection).load(
        split_name="test",
        max_samples=sample_count,
    )
    model = ModelAdapter(config, inspection)
    outputs = model.predict(data)
    bundle = FeatureExtractor(config).from_model_outputs(data, outputs)

    assert outputs.predictions.shape == (sample_count, 6)
    assert outputs.property_names == inspection.property_names
    assert np.isfinite(outputs.predictions).all()
    assert bundle.records["sample_id"].tolist() == data.frame["sample_id"].tolist()
    assert bundle.descriptors.shape[0] == sample_count
    assert bundle.descriptors.shape[1] == 1 + 56 + 80
    assert bundle.latent_arrays["cation_embedding"].shape[0] == sample_count
    assert bundle.latent_arrays["anion_embedding"].shape[0] == sample_count
    assert bundle.latent_arrays["ion_pair_embedding"].shape[0] == sample_count

    counterfactual = CounterfactualGenerator(config).generate_virtual_library()
    assert not counterfactual.empty
    assert (counterfactual["net_charge"] == 0).all()

    with tempfile.TemporaryDirectory(
        prefix="smoke_",
        dir=module_root,
    ) as temporary_directory:
        output_root = Path(temporary_directory)
        cache_base = output_root / "cache" / "bundle"
        cache_paths = FeatureExtractor.save_bundle(bundle, cache_base)
        restored = FeatureExtractor.load_bundle(cache_base)
        assert restored.records["sample_id"].tolist() == bundle.records[
            "sample_id"
        ].tolist()
        assert all(Path(path).is_file() for path in cache_paths.values())

        config["figures"]["formats"] = ["png"]
        config["figures"]["dpi"] = 72
        trend = pd.DataFrame(
            {
                "property": ["Density"] * 4,
                "feature": ["smoke_descriptor"] * 4,
                "quantile_bin": [0, 1, 2, 3],
                "sample_count": [4, 4, 4, 4],
                "feature_mean": [0.0, 1.0, 2.0, 3.0],
                "response_log_mean": [6.8, 6.9, 7.0, 7.1],
                "response_log_sem": [0.02] * 4,
                "monotonic_bin_spearman": [1.0] * 4,
            }
        )
        figure_paths = PublicationPlotter(config).response_curves(
            trend,
            output_root / "figures" / "smoke_response",
        )
        report_path = write_json(
            output_root / "reports" / "smoke_report.json",
            {
                "sample_count": sample_count,
                "checkpoint": str(inspection.selected_checkpoint),
                "counterfactual_count": len(counterfactual),
            },
        )
        assert all(path.is_file() for path in figure_paths)
        assert report_path.is_file()
