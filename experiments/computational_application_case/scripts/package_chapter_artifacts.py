"""Assemble the reproducible application-chapter result bundle.

The script copies, rather than transforms, current application CSV data.
Figures are expected to have been exported by
``build_refactored_application_case.py`` as both PNG and native Matplotlib SVG.
"""

from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path


CASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CASE_DIR.parents[1]
PAPER_FIG_DIR = PROJECT_ROOT / "LaTex-MIPGraph" / "Fig"
PRIMARY_DATA_DIR = CASE_DIR / "outputs_primary_audited" / "data"

CHAPTER_RESULTS_DIR = CASE_DIR / "chapter_results"
CHAPTER_FIGURE_DIR = CHAPTER_RESULTS_DIR / "figures"
CHAPTER_CSV_DIR = CHAPTER_RESULTS_DIR / "csv"

APPLICATION_FIGURE_NAMES = (
    ("figure5_auditable_virtual_screening_validation", "auditable_virtual_screening_validation"),
    ("figure6_reference_cell_scenario_audited", "reference_cell_scenario_audited"),
    ("figureS_application_decision_stability", "figureS_application_decision_stability"),
    ("figureS_application_derived_metrics", "figureS_application_derived_metrics"),
)

APPLICATION_CSV_NAMES = (
    "chemical_identity_audit_old_unseen_pool.csv",
    "chemical_identity_audit_old_shortlist.csv",
    "chemical_identity_audit_current_shortlist.csv",
    "standard_inchikey_identity_audit_608.csv",
    "observed_reference_selection_audit.csv",
    "threshold_sensitivity.csv",
    "candidate_selection_stability.csv",
    "reference_bootstrap_iterations.csv",
    "reference_bootstrap_candidate_selection.csv",
    "reference_bootstrap_summary.csv",
    "cross_protocol_decision_matrix.csv",
    "cross_protocol_summary.csv",
    "downstream_qualification_priorities.csv",
    "qualification_role_selection_audit.csv",
    "pareto_rank1_all_candidates.csv",
    "pareto_rank1_top8_selection.csv",
    "final_candidate_constraint_margins.csv",
    "candidate_screening_trajectory_608.csv",
    "extended_endpoint_transport_tradeoff.csv",
    "final_prioritized_candidates.csv",
    "reference_cell_candidate_summary.csv",
    "reference_cell_metrics_temperature.csv",
    "reference_cell_heat_resistance_contribution_audit.csv",
    "reference_cell_heat_resistance_contribution_summary.csv",
    "reference_cell_heat_resistance_population_summary.csv",
    "extreme_property_nearest_neighbors.csv",
    "extreme_property_audit.csv",
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
        rows = sum(1 for _ in reader)
    return rows, len(header)


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact is missing: {path}")


def copy_record(source: Path, destination: Path, bundle: str) -> dict[str, object]:
    require_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    rows: int | str = ""
    columns: int | str = ""
    if destination.suffix.lower() == ".csv":
        rows, columns = csv_shape(destination)
    return {
        "bundle": bundle,
        "file": destination.name,
        "destination_relative_path": destination.relative_to(PROJECT_ROOT).as_posix(),
        "source_relative_path": source.relative_to(PROJECT_ROOT).as_posix(),
        "bytes": destination.stat().st_size,
        "rows": rows,
        "columns": columns,
        "sha256": sha256(destination),
    }


def write_manifest(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "bundle",
        "file",
        "destination_relative_path",
        "source_relative_path",
        "bytes",
        "rows",
        "columns",
        "sha256",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def package_application_results() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for source_stem, destination_stem in APPLICATION_FIGURE_NAMES:
        for suffix in (".png", ".svg"):
            source = PAPER_FIG_DIR / f"{source_stem}{suffix}"
            destination = CHAPTER_FIGURE_DIR / f"{destination_stem}{suffix}"
            records.append(copy_record(source, destination, "application_chapter_results"))
    for name in APPLICATION_CSV_NAMES:
        source = PRIMARY_DATA_DIR / name
        destination = CHAPTER_CSV_DIR / name
        records.append(copy_record(source, destination, "application_chapter_results"))
    write_manifest(CHAPTER_RESULTS_DIR / "manifest.csv", records)
    return records


def write_code_inventory() -> None:
    included_suffixes = {".py", ".yaml", ".yml", ".txt", ".md"}
    excluded_roots = {
        "outputs",
        "outputs_auditable",
        "outputs_primary_audited",
        "outputs_protocol_balanced",
        "outputs_protocol_ion_family",
        "chapter_results",
        "__pycache__",
    }
    records = []
    for path in sorted(CASE_DIR.rglob("*")):
        relative = path.relative_to(CASE_DIR)
        if not path.is_file() or path.suffix.lower() not in included_suffixes:
            continue
        if relative.parts and relative.parts[0] in excluded_roots:
            continue
        records.append({
            "relative_path": relative.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    inventory = CASE_DIR / "CODE_INVENTORY.csv"
    with inventory.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("relative_path", "bytes", "sha256"))
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    application_records = package_application_results()
    write_code_inventory()
    print(f"Application result files packaged: {len(application_records)}")
    print(f"Application results: {CHAPTER_RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
