from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

from experiments.result_analysis.figures.manuscript_figure_source_data import (
    rebuild_manifest,
)
from experiments.result_analysis.scripts import package_manuscript_results


PROJECT_ROOT = Path(__file__).absolute().parents[3]
RESULT_ROOT = PROJECT_ROOT / "experiments" / "result_analysis"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_figure_links_use_single_canonical_csvs() -> None:
    figure_ids = {
        path.name
        for path in (RESULT_ROOT / "figures").iterdir()
        if path.is_dir() and path.name != "manuscript_figure_source_data"
    }
    assert not (RESULT_ROOT / "source_data").exists()
    assert not (RESULT_ROOT / "source_data_manifest.csv").exists()

    map_rows = _rows(RESULT_ROOT / "figure_source_map.csv")
    assert map_rows
    assert {row["figure_id"] for row in map_rows} == figure_ids

    for row in map_rows:
        packaged = (PROJECT_ROOT / row["csv_file"]).resolve()
        canonical = (PROJECT_ROOT / row["canonical_csv_file"]).resolve()
        assert packaged == canonical
        canonical.relative_to(
            (
                PROJECT_ROOT
                / "experiments"
                / "result_analysis" / "figures" / "manuscript_figure_source_data"
            ).resolve()
        )
        assert packaged.is_file()
        assert canonical.is_file()
        assert _sha256(packaged) == _sha256(canonical) == row["sha256"]


def test_stale_csv_is_rejected_before_any_packaged_file_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale = tmp_path / "source_data" / "stale.csv"
    stale.parent.mkdir(parents=True)
    stale.write_text("old,content\n", encoding="utf-8")

    def fail_if_copy_starts(source: Path, destination: Path) -> None:
        raise AssertionError(f"copy started before stale preflight: {source}, {destination}")

    monkeypatch.setattr(
        package_manuscript_results,
        "copy_if_different",
        fail_if_copy_starts,
    )

    with pytest.raises(RuntimeError, match="Duplicate source-data"):
        package_manuscript_results.package_source_data(tmp_path)

    assert stale.read_text(encoding="utf-8") == "old,content\n"
    assert not (tmp_path / "figure_source_map.csv").exists()
    assert not (tmp_path / "source_data_manifest.csv").exists()


def test_authoritative_manifest_rebuild_does_not_require_current_result_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_result_copy_is_checked() -> None:
        raise AssertionError("derived result bundle must not block canonical rebuild")

    monkeypatch.setattr(
        rebuild_manifest,
        "assert_result_analysis_source_bundle",
        fail_if_result_copy_is_checked,
    )
    monkeypatch.setattr(rebuild_manifest, "assert_data_only", lambda: None)
    monkeypatch.setattr(
        rebuild_manifest,
        "assert_no_unmanaged_manuscript_source_copies",
        lambda: None,
    )
    monkeypatch.setattr(rebuild_manifest, "panel_source_files", lambda: [])
    monkeypatch.setattr(
        rebuild_manifest,
        "assert_no_duplicate_source_tables",
        lambda paths: None,
    )
    monkeypatch.setattr(
        rebuild_manifest,
        "rebuild_column_dictionary",
        lambda paths: None,
    )
    monkeypatch.setattr(rebuild_manifest, "rebuild_manifest", lambda paths: None)

    rebuild_manifest.main()
