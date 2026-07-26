"""Tests for the audited performance-results figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from PIL import Image

from experiments.performance_results.plot_performance_results import (
    PROPERTY_ORDER,
    REPO_ROOT,
    create_figure,
    create_figure_from_saved_source,
    load_saved_source_data,
)
from experiments.performance_results.prepare_inputs import (
    PROPERTIES,
    SPLITS,
    load_long_predictions,
    load_metrics,
    sample_keys,
    stage_split,
)


SOURCE_DATA = (
    REPO_ROOT
    / "experiments"
    / "manuscript_figure_source_data"
    / "performance_results"
)


def test_saved_source_data_reconstructs_all_protocol_property_metrics() -> None:
    predictions, metrics = load_saved_source_data(SOURCE_DATA)

    assert set(predictions["property"]) == set(PROPERTY_ORDER)
    assert set(predictions["split_strategy"]) == {
        "Random point",
        "Random IL",
        "Balanced IL",
        "Ion-family",
    }
    property_metrics = metrics.loc[metrics["property"] != "Average"]
    assert len(property_metrics) == 4 * len(PROPERTY_ORDER)
    assert not property_metrics.duplicated(["split_strategy", "property"]).any()
    assert metrics.loc[metrics["property"] == "Average", "log_R2"].notna().all()


def test_redraw_from_saved_source_uses_fixed_journal_canvas(tmp_path: Path) -> None:
    create_figure_from_saved_source(
        SOURCE_DATA,
        tmp_path,
        "performance_results_test",
        72,
    )

    expected = {
        "performance_results_test.pdf",
        "performance_results_test.png",
        "performance_results_test.svg",
        "performance_results_test.tiff",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected
    with Image.open(tmp_path / "performance_results_test.png") as image:
        assert image.size == (518, 272)


def test_sample_keys_are_stable_to_small_condition_roundoff() -> None:
    frame = pd.DataFrame(
        {
            "IL_SMILES": ["cation.anion", "cation.anion"],
            "Temperature_K": [298.15, 298.1500001],
            "Pressure_kPa": [101.325, 101.3250001],
        }
    )
    keys = sample_keys(frame)
    assert keys.iloc[0] == keys.iloc[1]


def _write_raw_split(run_root: Path, split: str) -> None:
    final = run_root / split / "final"
    final.mkdir(parents=True)
    for property_index, prop in enumerate(PROPERTIES):
        values = [1.0 + property_index, 2.0 + property_index]
        pd.DataFrame(
            {
                "sample_id": [10, 11],
                "IL_Name": ["IL-1", "IL-1"],
                "IL_SMILES": ["cation.anion", "cation.anion"],
                "Temperature_K": [298.15, 298.15],
                "Pressure_kPa": [101.325, 101.325],
                "y_true": values,
                "y_pred": [value * 1.01 for value in values],
            }
        ).to_csv(final / f"test_predictions_{prop}.csv", index=False)

    metrics = pd.DataFrame(
        {
            "property": PROPERTIES,
            "log_MAE": [0.01] * len(PROPERTIES),
            "log_RMSE": [0.02] * len(PROPERTIES),
            "log_R2": [0.90] * len(PROPERTIES),
            "log_NMAE": [0.10] * len(PROPERTIES),
        }
    )
    metrics.to_csv(final / "selected_test_metrics.csv", index=False)
    (final / "selected_test_metrics.json").write_text(
        json.dumps(
            {
                "average": {
                    "log_MAE": 0.01,
                    "log_RMSE": 0.02,
                    "log_R2": 0.90,
                    "log_NMAE": 0.10,
                }
            }
        ),
        encoding="utf-8",
    )


def test_refresh_path_preserves_replicates_and_exports_a_to_i(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    staged_root = tmp_path / "staged"
    for split in SPLITS:
        _write_raw_split(run_root, split)
        stage_split(run_root, staged_root, split)

    long = pd.read_csv(staged_root / SPLITS[0] / "test_predictions.csv")
    wide = pd.read_csv(staged_root / SPLITS[0] / "test_predictions_wide.csv")
    assert long["source_sample_id"].nunique() == 2
    assert long["sample_id"].nunique() == 2
    assert len(wide) == 2
    assert set(wide["source_sample_id"]) == {10, 11}

    figure_dir = tmp_path / "figure"
    create_figure(figure_dir, "refreshed", 72, staged_root)
    expected_sources = {
        f"performance_results_source_data_{panel}.csv" for panel in "ABCDEFGHI"
    }
    assert expected_sources.issubset({path.name for path in figure_dir.iterdir()})


def test_duplicate_conditions_without_observation_ids_are_rejected(
    tmp_path: Path,
) -> None:
    final = tmp_path / "final"
    final.mkdir()
    pd.DataFrame(
        {
            "IL_Name": ["IL-1", "IL-1"],
            "IL_SMILES": ["cation.anion", "cation.anion"],
            "Temperature_K": [298.15, 298.15],
            "Pressure_kPa": [101.325, 101.325],
            "y_true": [1.0, 1.1],
            "y_pred": [0.9, 1.2],
        }
    ).to_csv(final / f"test_predictions_{PROPERTIES[0]}.csv", index=False)

    with pytest.raises(ValueError, match="refusing to average"):
        load_long_predictions(tmp_path)


def test_all_exports_receive_requested_dpi_and_figure_is_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved: list[tuple[str, int]] = []

    def record_savefig(self, path, **kwargs):
        saved.append((Path(path).suffix, kwargs["dpi"]))

    monkeypatch.setattr(plt.Figure, "savefig", record_savefig)
    create_figure_from_saved_source(SOURCE_DATA, Path("unused"), "test", 600)

    assert saved == [
        (".pdf", 600),
        (".svg", 600),
        (".png", 600),
        (".tiff", 600),
    ]
    assert plt.get_fignums() == []


def test_figure_is_closed_when_plotting_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plt.close("all")

    def raise_plotting_error(*args, **kwargs):
        raise RuntimeError("forced plotting failure")

    monkeypatch.setattr(
        "experiments.performance_results.plot_performance_results."
        "plot_multi_split_parity",
        raise_plotting_error,
    )
    with pytest.raises(RuntimeError, match="forced plotting failure"):
        create_figure_from_saved_source(
            SOURCE_DATA,
            tmp_path,
            "failure",
            72,
        )
    assert plt.get_fignums() == []


def test_duplicate_or_incomplete_metric_rows_are_rejected(tmp_path: Path) -> None:
    final = tmp_path / "final"
    final.mkdir()
    invalid_properties = PROPERTIES[:-1] + [PROPERTIES[0]]
    pd.DataFrame(
        {
            "property": invalid_properties,
            "log_MAE": [0.01] * len(invalid_properties),
            "log_RMSE": [0.02] * len(invalid_properties),
            "log_R2": [0.90] * len(invalid_properties),
            "log_NMAE": [0.10] * len(invalid_properties),
        }
    ).to_csv(final / "selected_test_metrics.csv", index=False)
    (final / "selected_test_metrics.json").write_text(
        json.dumps({"average": {"log_R2": 0.90, "log_NMAE": 0.10}}),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="duplicate properties.*missing properties",
    ):
        load_metrics(tmp_path)
