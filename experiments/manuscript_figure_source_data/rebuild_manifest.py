"""Rebuild checksummed manifests for manuscript figure source-data bundles."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]


@dataclass(frozen=True)
class Bundle:
    name: str
    figure_file: str
    producer_path: str


BUNDLES = (
    Bundle(
        "dataset_statistics",
        "dataset_statistics.png",
        "experiments/dataset_analysis/scripts/plot_dataset_statistics_nature.py",
    ),
    Bundle(
        "interpretability_feature_importance_4x3",
        "interpretability_feature_importance_4x3.png",
        "experiments/interpretability/scripts/compose_interpretability_four_by_three.py",
    ),
    Bundle(
        "performance_results",
        "performance_results.png",
        "experiments/performance_results/plot_performance_results.py",
    ),
)


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


def write_csv(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: list[dict[str, object]],
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    records: list[dict[str, object]] = []
    for bundle in BUNDLES:
        bundle_dir = ROOT / bundle.name
        figure = bundle_dir / bundle.figure_file
        if not figure.is_file():
            raise FileNotFoundError(figure)

        csv_files = sorted(
            path
            for path in bundle_dir.glob("*.csv")
            if path.name != "source_figure_audit.csv"
        )
        for path in [*csv_files, figure]:
            rows: int | str = ""
            columns: int | str = ""
            if path.suffix.lower() == ".csv":
                rows, columns = csv_shape(path)
            records.append(
                {
                    "bundle": bundle.name,
                    "file": path.name,
                    "destination_relative_path": path.relative_to(
                        PROJECT_ROOT
                    ).as_posix(),
                    "producer_path": bundle.producer_path,
                    "bytes": path.stat().st_size,
                    "rows": rows,
                    "columns": columns,
                    "sha256": sha256(path),
                }
            )

        write_csv(
            bundle_dir / "source_figure_audit.csv",
            (
                "figure",
                "producer_path",
                "packaged_figure_relative_path",
                "bytes",
                "sha256",
                "csv_file_count",
            ),
            [
                {
                    "figure": bundle.name,
                    "producer_path": bundle.producer_path,
                    "packaged_figure_relative_path": figure.relative_to(
                        PROJECT_ROOT
                    ).as_posix(),
                    "bytes": figure.stat().st_size,
                    "sha256": sha256(figure),
                    "csv_file_count": len(csv_files),
                }
            ],
        )

    write_csv(
        ROOT / "manifest.csv",
        (
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
    print(f"Wrote {len(records)} manifest records under {ROOT}")


if __name__ == "__main__":
    main()
