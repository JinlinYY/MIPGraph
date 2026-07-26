"""Rebuild the authoritative manuscript source-data catalogs.

This directory is the only manuscript-facing home for panel-level CSV files.
The aggregate manifest and column dictionary are rebuilt from the files that
are physically present below this directory; final figure files are rejected.
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from field_definitions import definition_for


PROJECT_ROOT = ROOT.parents[1]
MANIFEST_PATH = ROOT / "manifest.csv"
COLUMN_DICTIONARY_PATH = ROOT / "column_dictionary.csv"
FIGURE_SUFFIXES = {".pdf", ".png", ".svg", ".tif", ".tiff"}

PRODUCERS = {
    "Intro-method": "manual figure inventory",
    "computational_application_case": (
        "experiments/computational_application_case/"
        "scripts/build_refactored_application_case.py"
    ),
    "dataset_statistics": (
        "experiments/dataset_analysis/scripts/"
        "export_dataset_statistics_source_data.py"
    ),
    "molecular_origin_analysis": (
        "experiments/molecular_origin_analysis/package_source_data.py"
    ),
    "performance_results": (
        "experiments/performance_results/plot_performance_results.py"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_shape(path: Path) -> tuple[int, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        return sum(1 for _ in reader), len(header)


def csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return next(csv.reader(handle), [])


def csv_value_types(path: Path) -> dict[str, str]:
    """Infer the physical CSV storage type of each field."""

    precedence = {
        "empty": 0,
        "boolean": 1,
        "integer": 2,
        "number": 3,
        "string": 4,
    }

    def scalar_type(value: str) -> str:
        stripped = value.strip()
        if not stripped:
            return "empty"
        if stripped.lower() in {"true", "false"}:
            return "boolean"
        try:
            integer = int(stripped)
        except ValueError:
            integer = None
        if integer is not None and str(integer) == stripped:
            return "integer"
        try:
            float(stripped)
        except ValueError:
            return "string"
        return "number"

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        inferred = {field: "empty" for field in fields}
        for row in reader:
            for field in fields:
                candidate = scalar_type(row.get(field, ""))
                if precedence[candidate] > precedence[inferred[field]]:
                    inferred[field] = candidate
    return inferred


def write_csv(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: Iterable[dict[str, object]],
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def bundle_name(path: Path) -> str:
    relative = path.relative_to(ROOT)
    if len(relative.parts) < 2:
        return "_catalog"
    return relative.parts[0]


def producer_path(bundle: str, source_file: str) -> str:
    if bundle == "_catalog":
        return (
            "experiments/manuscript_figure_source_data/"
            "rebuild_manifest.py"
        )
    if bundle == "interpretability_feature_importance_4x3":
        if source_file.startswith("interpretability_results_source_data_"):
            return (
                "experiments/interpretability/scripts/"
                "plot_interpretability_results_current.py"
            )
        if source_file in {
            "feature_importance_heatmap_source_data_nodes.csv",
            "feature_importance_heatmap_source_data_edges.csv",
            "feature_importance_heatmap_source_data_functional_groups.csv",
        }:
            return (
                "il_property_prediction/scripts/"
                "compute_feature_importance_heatmap.py"
            )
        if source_file.startswith("feature_importance_heatmap"):
            return (
                "experiments/interpretability/scripts/"
                "plot_feature_importance_summary.py"
            )
        raise KeyError(
            "No interpretability producer is registered for "
            f"{source_file!r}"
        )
    try:
        return PRODUCERS[bundle]
    except KeyError as error:
        raise KeyError(f"No producer is registered for bundle {bundle!r}") from error


def panel_source_files() -> list[Path]:
    excluded = {MANIFEST_PATH.resolve(), COLUMN_DICTIONARY_PATH.resolve()}
    return sorted(
        (
            path
            for path in ROOT.rglob("*.csv")
            if path.resolve() not in excluded
            and path.name != "manifest.csv"
        ),
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )


def assert_data_only() -> None:
    figure_files = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in FIGURE_SUFFIXES
    )
    if figure_files:
        details = "\n".join(f"- {path}" for path in figure_files)
        raise RuntimeError(
            "Figure files are not allowed in the authoritative source-data "
            f"directory:\n{details}"
        )


def assert_no_legacy_manuscript_source_copies() -> None:
    """Reject known former manuscript source-data publication locations."""

    forbidden_directories = (
        PROJECT_ROOT / "experiments" / "result_analysis" / "source_data",
        PROJECT_ROOT / "result_fig" / "source_data",
        PROJECT_ROOT / "LaTex-MIPGraph" / "Fig" / "source_data",
    )
    offenders = [
        path
        for path in forbidden_directories
        if path.exists() and any(item.is_file() for item in path.rglob("*"))
    ]
    latex_figure_dir = PROJECT_ROOT / "LaTex-MIPGraph" / "Fig"
    if latex_figure_dir.exists():
        offenders.extend(sorted(latex_figure_dir.glob("*.csv")))
    if offenders:
        details = "\n".join(f"- {path}" for path in offenders)
        raise RuntimeError(
            "Legacy manuscript source-data copies were detected outside the "
            f"authoritative directory:\n{details}"
        )


def assert_no_duplicate_source_tables(paths: list[Path]) -> None:
    by_hash: dict[str, list[Path]] = {}
    for path in paths:
        if path.name == "column_dictionary.csv":
            continue
        by_hash.setdefault(sha256(path), []).append(path)
    duplicates = [items for items in by_hash.values() if len(items) > 1]
    if duplicates:
        details = "\n".join(
            " = ".join(str(path.relative_to(ROOT)) for path in items)
            for items in duplicates
        )
        raise RuntimeError(f"Duplicate canonical source tables detected:\n{details}")


def existing_field_definitions() -> dict[tuple[str, str, str], tuple[str, str]]:
    definitions: dict[tuple[str, str, str], tuple[str, str]] = {}
    for dictionary in ROOT.rglob("column_dictionary.csv"):
        if dictionary.resolve() == COLUMN_DICTIONARY_PATH.resolve():
            continue
        bundle = bundle_name(dictionary)
        with dictionary.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                definitions[
                    (
                        bundle,
                        row["source_data_file"],
                        row["column"],
                    )
                ] = (
                    row["description"],
                    row["unit_or_scale"],
                )
    return definitions


def rebuild_column_dictionary(paths: list[Path]) -> None:
    definitions = existing_field_definitions()
    rows: list[dict[str, object]] = []
    for path in paths:
        if path.name == "column_dictionary.csv":
            continue
        bundle = bundle_name(path)
        relative_to_bundle = path.relative_to(ROOT / bundle).as_posix()
        value_types = csv_value_types(path)
        for column in csv_header(path):
            local_definition = definitions.get((bundle, path.name, column))
            if local_definition is None:
                description, unit, definition_status = definition_for(
                    bundle,
                    column,
                )
            else:
                description, unit = local_definition
                definition_status = "curated-local"
            rows.append(
                {
                    "bundle": bundle,
                    "source_data_file": relative_to_bundle,
                    "column": column,
                    "description": description,
                    "unit_or_scale": unit,
                    "storage_type": value_types[column],
                    "definition_status": definition_status,
                    "producer_path": producer_path(
                        bundle,
                        relative_to_bundle,
                    ),
                }
            )
    write_csv(
        COLUMN_DICTIONARY_PATH,
        (
            "bundle",
            "source_data_file",
            "column",
            "description",
            "unit_or_scale",
            "storage_type",
            "definition_status",
            "producer_path",
        ),
        rows,
    )


def rebuild_manifest(paths: list[Path]) -> None:
    catalog_paths = [*paths, COLUMN_DICTIONARY_PATH]
    records: list[dict[str, object]] = []
    for path in catalog_paths:
        bundle = bundle_name(path)
        rows, columns = csv_shape(path)
        records.append(
            {
                "artifact_type": (
                    "field_dictionary"
                    if path.name == "column_dictionary.csv"
                    else "panel_source_data"
                ),
                "bundle": bundle,
                "file": path.relative_to(ROOT).as_posix(),
                "destination_relative_path": path.relative_to(
                    PROJECT_ROOT
                ).as_posix(),
                "producer_path": producer_path(
                    bundle,
                    path.relative_to(ROOT / bundle).as_posix()
                    if bundle != "_catalog"
                    else path.name,
                ),
                "bytes": path.stat().st_size,
                "rows": rows,
                "columns": columns,
                "sha256": sha256(path),
            }
        )
    write_csv(
        MANIFEST_PATH,
        (
            "artifact_type",
            "bundle",
            "file",
            "destination_relative_path",
            "producer_path",
            "bytes",
            "rows",
            "columns",
            "sha256",
        ),
        records,
    )


def main() -> None:
    assert_data_only()
    assert_no_legacy_manuscript_source_copies()
    paths = panel_source_files()
    assert_no_duplicate_source_tables(paths)
    rebuild_column_dictionary(paths)
    rebuild_manifest(paths)
    print(
        f"Wrote {len(paths) + 1} authoritative source-data records "
        f"under {ROOT}"
    )


if __name__ == "__main__":
    main()
