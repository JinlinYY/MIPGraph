"""Package the exact manuscript figures and their auditable source-data CSVs.

The author-supplied PNG files are treated as the canonical manuscript views.
Native PDF/SVG exports are copied when the matching plotting workflow provides
them. Raster-derived PDF/SVG files are explicitly identified in the manifest
when no matching native vector export exists.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODULE_ROOT = PROJECT_ROOT / "experiments" / "result_analysis"
DEFAULT_MANUSCRIPT_FIGURE_DIR = (
    Path.home() / "Downloads" / "OD-TwoColumn (June 2026)" / "Fig"
)


@dataclass(frozen=True)
class FigureSpec:
    figure_id: str
    manuscript_png: str
    native_pdf: str | None
    native_svg: str | None
    vector_provenance: str


@dataclass(frozen=True)
class SourceSpec:
    path: Path
    panels: str
    description: str


FIGURES = (
    FigureSpec(
        "auditable_virtual_screening_validation",
        "auditable_virtual_screening_validation.png",
        (
            "experiments/result_analysis/figures/"
            "auditable_virtual_screening_validation/"
            "auditable_virtual_screening_validation.pdf"
        ),
        (
            "experiments/computational_application_case/chapter_results/"
            "figures/auditable_virtual_screening_validation.svg"
        ),
        "native Matplotlib export from the matching application workflow",
    ),
    FigureSpec(
        "dataset_statistics",
        "dataset_statistics.png",
        (
            "experiments/result_analysis/figures/dataset_statistics/"
            "dataset_statistics.pdf"
        ),
        (
            "experiments/result_analysis/figures/dataset_statistics/"
            "dataset_statistics.svg"
        ),
        "native Matplotlib export from the matching dataset workflow",
    ),
    FigureSpec(
        "figureS_application_decision_stability",
        "figureS_application_decision_stability.png",
        (
            "experiments/result_analysis/figures/"
            "figureS_application_decision_stability/"
            "figureS_application_decision_stability.pdf"
        ),
        (
            "experiments/computational_application_case/chapter_results/"
            "figures/figureS_application_decision_stability.svg"
        ),
        "native Matplotlib export from the matching application workflow",
    ),
    FigureSpec(
        "figureS_application_derived_metrics",
        "figureS_application_derived_metrics.png",
        (
            "experiments/result_analysis/figures/"
            "figureS_application_derived_metrics/"
            "figureS_application_derived_metrics.pdf"
        ),
        (
            "experiments/computational_application_case/chapter_results/"
            "figures/figureS_application_derived_metrics.svg"
        ),
        "native Matplotlib export from the matching application workflow",
    ),
    FigureSpec(
        "interpretability",
        "interpretability.png",
        (
            "experiments/result_analysis/figures/interpretability/"
            "interpretability.pdf"
        ),
        None,
        "the submitted composite is raster-assembled; no native composite SVG",
    ),
    FigureSpec(
        "figureS_heat_capacity_size_control",
        "figureS_heat_capacity_size_control.png",
        (
            "experiments/manuscript_figure_source_data/molecular_origin_analysis/"
            "figures/figure_si_heat_capacity_size_control.pdf"
        ),
        (
            "experiments/manuscript_figure_source_data/molecular_origin_analysis/"
            "figures/figure_si_heat_capacity_size_control.svg"
        ),
        "native Matplotlib export from the matching molecular-origin workflow",
    ),
    FigureSpec(
        "Intro-method",
        "Intro-method.png",
        None,
        None,
        "conceptual schematic supplied only as the exact manuscript PNG",
    ),
    FigureSpec(
        "performance_results",
        "performance_results.png",
        (
            "experiments/result_analysis/figures/performance_results/"
            "performance_results.pdf"
        ),
        (
            "experiments/result_analysis/figures/performance_results/"
            "performance_results.svg"
        ),
        "native Matplotlib export from the matching performance workflow",
    ),
    FigureSpec(
        "reference_cell_scenario_audited",
        "reference_cell_scenario_audited.png",
        (
            "experiments/result_analysis/figures/"
            "reference_cell_scenario_audited/"
            "reference_cell_scenario_audited.pdf"
        ),
        (
            "experiments/computational_application_case/chapter_results/"
            "figures/reference_cell_scenario_audited.svg"
        ),
        "native Matplotlib export from the matching application workflow",
    ),
    FigureSpec(
        "molecular_origin_analysis",
        "molecular_origin_analysis.png",
        None,
        None,
        "the compact submitted layout is available only as the exact manuscript PNG",
    ),
)


def project_path(relative: str) -> Path:
    return PROJECT_ROOT / Path(relative)


def application_csv(name: str) -> Path:
    return (
        PROJECT_ROOT
        / "experiments"
        / "computational_application_case"
        / "chapter_results"
        / "csv"
        / name
    )


def source_specs() -> dict[str, list[SourceSpec]]:
    sources: dict[str, list[SourceSpec]] = {
        "auditable_virtual_screening_validation": [
            SourceSpec(
                application_csv("candidate_screening_trajectory_608.csv"),
                "a",
                "Complete 608-candidate screening trajectory and hard constraints.",
            ),
            SourceSpec(
                application_csv("chemical_identity_audit_old_unseen_pool.csv"),
                "b",
                "Legacy SMILES-based novelty audit.",
            ),
            SourceSpec(
                application_csv("standard_inchikey_identity_audit_608.csv"),
                "b",
                "Charge-aware InChIKey identity audit for the evaluated pool.",
            ),
            SourceSpec(
                application_csv("pareto_rank1_all_candidates.csv"),
                "a",
                "All twelve Pareto-rank-1 candidates.",
            ),
            SourceSpec(
                application_csv("pareto_rank1_top8_selection.csv"),
                "a,f",
                "Deterministic Pareto-rank-1 to Top-8 selection record.",
            ),
            SourceSpec(
                application_csv("final_candidate_constraint_margins.csv"),
                "c",
                "Hard-constraint margins for the formal shortlist.",
            ),
            SourceSpec(
                application_csv("reference_bootstrap_candidate_selection.csv"),
                "d",
                "Candidate-level bootstrap selection frequencies.",
            ),
            SourceSpec(
                application_csv("cross_protocol_decision_matrix.csv"),
                "e",
                "Primary and sensitivity-protocol decisions.",
            ),
            SourceSpec(
                application_csv("downstream_qualification_priorities.csv"),
                "f",
                "Automatically selected qualification roles.",
            ),
            SourceSpec(
                application_csv("qualification_role_selection_audit.csv"),
                "f",
                "Qualification-role eligibility and deterministic selection audit.",
            ),
            SourceSpec(
                application_csv("final_prioritized_candidates.csv"),
                "a,c,e,f",
                "Formal eight-candidate shortlist.",
            ),
        ],
        "figureS_application_decision_stability": [
            SourceSpec(
                application_csv("threshold_sensitivity.csv"),
                "a",
                "Prespecified threshold-grid sensitivity results.",
            ),
            SourceSpec(
                application_csv("reference_bootstrap_iterations.csv"),
                "b",
                "Bootstrap replicate Top-8 overlap results.",
            ),
            SourceSpec(
                application_csv("reference_bootstrap_candidate_selection.csv"),
                "c",
                "Candidate-level bootstrap selection frequencies.",
            ),
        ],
        "figureS_application_derived_metrics": [
            SourceSpec(
                application_csv("reference_cell_metrics_temperature.csv"),
                "a,b",
                "Temperature-resolved Joule and conditional temperature-rise metrics.",
            ),
            SourceSpec(
                application_csv("downstream_qualification_priorities.csv"),
                "a,b",
                "Candidate identities displayed in the SI curves.",
            ),
        ],
        "reference_cell_scenario_audited": [
            SourceSpec(
                application_csv("reference_cell_metrics_temperature.csv"),
                "b,d,e",
                "Temperature-resolved electrolyte-path engineering metrics.",
            ),
            SourceSpec(
                application_csv("candidate_screening_trajectory_608.csv"),
                "c",
                "Screened-population thermal-property coordinates.",
            ),
            SourceSpec(
                application_csv("downstream_qualification_priorities.csv"),
                "b,c,d",
                "Qualification leads highlighted in the post hoc mapping.",
            ),
            SourceSpec(
                application_csv("extended_endpoint_transport_tradeoff.csv"),
                "e",
                "Cold-hot endpoint resistance ratios.",
            ),
            SourceSpec(
                application_csv("reference_cell_candidate_summary.csv"),
                "f",
                "Reference-population exceedance context.",
            ),
            SourceSpec(
                application_csv("final_prioritized_candidates.csv"),
                "b,c,d,e,f",
                "Formal shortlist fixed before reference-cell mapping.",
            ),
        ],
        "figureS_heat_capacity_size_control": [
            SourceSpec(
                project_path(
                    "experiments/manuscript_figure_source_data/"
                    "molecular_origin_analysis/source_data/"
                    "figure_si_heat_capacity_size_control_source_data.csv"
                ),
                "a-c",
                "Identity-level heat-capacity size-control records.",
            )
        ],
        "molecular_origin_analysis": [
            SourceSpec(
                project_path(
                    "experiments/manuscript_figure_source_data/"
                    "molecular_origin_analysis/source_data/"
                    "figure_main_molecular_origin_analysis_final_panel_a_source_data.csv"
                ),
                "a",
                "Association and attention evidence-map records.",
            ),
            SourceSpec(
                project_path(
                    "experiments/manuscript_figure_source_data/"
                    "molecular_origin_analysis/source_data/"
                    "figure_main_molecular_origin_analysis_final_panel_b_source_data.csv"
                ),
                "b",
                "Leading association within each structural scope.",
            ),
            SourceSpec(
                project_path(
                    "experiments/manuscript_figure_source_data/"
                    "molecular_origin_analysis/source_data/"
                    "figure_main_molecular_origin_analysis_final_panel_c_source_data.csv"
                ),
                "c",
                "Condition-adjusted response curves and bootstrap intervals.",
            ),
            SourceSpec(
                project_path(
                    "experiments/manuscript_figure_source_data/"
                    "molecular_origin_analysis/source_data/"
                    "figure_main_molecular_origin_analysis_final_panel_d_source_data.csv"
                ),
                "d",
                "Condition-matched ion-substitution effects.",
            ),
        ],
    }

    dataset_dir = (
        PROJECT_ROOT
        / "experiments"
        / "manuscript_figure_source_data"
        / "dataset_statistics"
    )
    sources["dataset_statistics"] = [
        SourceSpec(
            path,
            path.stem.rsplit("_", 1)[-1].lower(),
            "Exact panel-level dataset-statistics source data.",
        )
        for path in sorted(dataset_dir.glob("*.csv"))
    ]

    performance_dir = (
        PROJECT_ROOT
        / "experiments"
        / "manuscript_figure_source_data"
        / "performance_results"
    )
    sources["performance_results"] = [
        SourceSpec(
            path,
            (
                "g-i"
                if path.name == "final_test_metrics_summary_source_data.csv"
                else path.stem.rsplit("_", 1)[-1].lower()
            ),
            "Exact panel-level performance source data.",
        )
        for path in sorted(performance_dir.glob("*.csv"))
    ]

    interpretability_dir = (
        PROJECT_ROOT
        / "experiments"
        / "manuscript_figure_source_data"
        / "interpretability_feature_importance_4x3"
    )
    interpretability_sources: list[SourceSpec] = []
    for path in sorted(interpretability_dir.glob("*.csv")):
        if path.name == "source_figure_audit.csv":
            continue
        if "interpretability_results_source_data_" in path.name:
            suffix = path.stem.rsplit("_", 1)[-1].lower()
            panel = "g-i (master table)" if suffix == "m" else suffix
        elif "nodes" in path.name:
            panel = "j"
        elif "edges" in path.name:
            panel = "k"
        elif "functional_groups" in path.name:
            panel = "l"
        else:
            panel = "j-l"
        interpretability_sources.append(
            SourceSpec(
                path,
                panel,
                "Interpretability or feature-attribution panel source data.",
            )
        )
    sources["interpretability"] = interpretability_sources
    sources["Intro-method"] = []
    return sources


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_rows(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, object]],
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def csv_shape(path: Path) -> tuple[int, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        return sum(1 for _ in reader), len(header)


def relative_origin(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def write_raster_svg(png_path: Path, svg_path: Path) -> None:
    with Image.open(png_path) as image:
        width, height = image.size
    encoded = base64.b64encode(png_path.read_bytes()).decode("ascii")
    svg_path.write_text(
        (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">\n'
            "  <metadata>Raster-embedded exact manuscript PNG; not a native "
            "vector export.</metadata>\n"
            f'  <image width="{width}" height="{height}" '
            f'href="data:image/png;base64,{encoded}"/>\n'
            "</svg>\n"
        ),
        encoding="utf-8",
    )


def write_raster_pdf(png_path: Path, pdf_path: Path) -> None:
    with Image.open(png_path) as image:
        rgb = image.convert("RGB")
        rgb.save(pdf_path, "PDF", resolution=600.0)


def write_tiff(png_path: Path, tiff_path: Path) -> None:
    with Image.open(png_path) as image:
        rgb = image.convert("RGB")
        rgb.save(
            tiff_path,
            "TIFF",
            compression="tiff_lzw",
            dpi=(600, 600),
        )


def copy_if_different(source: Path, destination: Path) -> None:
    """Copy an artifact unless the packaged file is also its canonical source."""

    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)


def add_generated_source_tables(
    figure_id: str,
    target_dir: Path,
) -> list[tuple[Path, str, str, str]]:
    generated: list[tuple[Path, str, str, str]] = []
    if figure_id == "auditable_virtual_screening_validation":
        path = target_dir / "panel_a_candidate_space_funnel.csv"
        rows = [
            {
                "stage_order": 1,
                "stage": "combinatorial_pairs",
                "count": 900,
                "provenance": "30 cations x 30 anions",
            },
            {
                "stage_order": 2,
                "stage": "InChI_identity_new_pool",
                "count": 608,
                "provenance": "standard_inchikey_identity_audit_608.csv",
            },
            {
                "stage_order": 3,
                "stage": "evaluated_by_primary_model",
                "count": 608,
                "provenance": "candidate_screening_trajectory_608.csv",
            },
            {
                "stage_order": 4,
                "stage": "hard_feasible",
                "count": 26,
                "provenance": "candidate_screening_trajectory_608.csv",
            },
            {
                "stage_order": 5,
                "stage": "Pareto_rank_1",
                "count": 12,
                "provenance": "pareto_rank1_all_candidates.csv",
            },
            {
                "stage_order": 6,
                "stage": "formal_shortlist",
                "count": 8,
                "provenance": "final_prioritized_candidates.csv",
            },
        ]
        write_rows(path, ["stage_order", "stage", "count", "provenance"], rows)
        generated.append(
            (
                path,
                "a",
                "Figure-ready candidate-space funnel counts.",
                "derived from copied application CSVs and the prespecified 30 x 30 pool",
            )
        )
    elif figure_id == "reference_cell_scenario_audited":
        path = target_dir / "panel_a_reference_cell_scenario_parameters.csv"
        rows = [
            {"parameter": "electrode_area", "symbol": "A", "value": 100, "unit": "cm^2"},
            {"parameter": "separator_thickness", "symbol": "L", "value": 100, "unit": "um"},
            {"parameter": "electrolyte_volume", "symbol": "V", "value": 1.0, "unit": "mL"},
            {"parameter": "constant_current", "symbol": "I", "value": 2.0, "unit": "A"},
            {
                "parameter": "convection_coefficient",
                "symbol": "h",
                "value": 10,
                "unit": "W m^-2 K^-1",
            },
            {"parameter": "exposed_faces", "symbol": "N_f", "value": 2, "unit": "count"},
            {"parameter": "stress_duration", "symbol": "t", "value": 60, "unit": "s"},
        ]
        write_rows(path, ["parameter", "symbol", "value", "unit"], rows)
        generated.append(
            (
                path,
                "a",
                "Standardized 60-s constant-current scenario parameters.",
                "transcribed from the audited plotting protocol",
            )
        )
    elif figure_id == "Intro-method":
        path = target_dir / "Intro-method_panel_inventory.csv"
        rows = [
            {
                "panel": "a",
                "content": "ionic-liquid application contexts",
                "quantitative_source_data": "not applicable",
            },
            {
                "panel": "b",
                "content": "ion-pair concept and six target properties",
                "quantitative_source_data": "not applicable",
            },
            {
                "panel": "c",
                "content": "experiment, simulation, descriptors, and machine learning",
                "quantitative_source_data": "not applicable",
            },
            {
                "panel": "d",
                "content": "ILThermo query, API, matching, and sparse-label merge",
                "quantitative_source_data": "not applicable",
            },
            {
                "panel": "e",
                "content": "MIPGraph architecture schematic",
                "quantitative_source_data": "not applicable",
            },
        ]
        write_rows(
            path,
            ["panel", "content", "quantitative_source_data"],
            rows,
        )
        generated.append(
            (
                path,
                "a-e",
                "Non-quantitative panel inventory; the schematic has no plotted CSV data.",
                "manual figure inventory",
            )
        )
    return generated


def package(
    manuscript_figure_dir: Path,
    destination: Path,
) -> None:
    figure_root = destination / "figures"
    source_root = destination / "source_data"
    figure_root.mkdir(parents=True, exist_ok=True)
    source_root.mkdir(parents=True, exist_ok=True)

    source_map = source_specs()
    manifest_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []

    for spec in FIGURES:
        manuscript_png = manuscript_figure_dir / spec.manuscript_png
        if not manuscript_png.is_file():
            raise FileNotFoundError(manuscript_png)
        figure_dir = figure_root / spec.figure_id
        data_dir = source_root / spec.figure_id
        figure_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)

        png_target = figure_dir / f"{spec.figure_id}.png"
        shutil.copy2(manuscript_png, png_target)
        with Image.open(png_target) as image:
            width, height = image.size

        figure_outputs: list[tuple[Path, str, str, str]] = [
            (
                png_target,
                "png",
                "exact author-supplied manuscript PNG",
                f"author_supplied_manuscript_png/{spec.manuscript_png}",
            )
        ]

        pdf_target = figure_dir / f"{spec.figure_id}.pdf"
        if spec.native_pdf is not None:
            native_pdf = project_path(spec.native_pdf)
            if not native_pdf.is_file():
                raise FileNotFoundError(native_pdf)
            copy_if_different(native_pdf, pdf_target)
            pdf_provenance = spec.vector_provenance
            pdf_origin = relative_origin(native_pdf)
        else:
            write_raster_pdf(png_target, pdf_target)
            pdf_provenance = "raster-derived from exact manuscript PNG"
            pdf_origin = relative_origin(png_target)
        figure_outputs.append((pdf_target, "pdf", pdf_provenance, pdf_origin))

        svg_target = figure_dir / f"{spec.figure_id}.svg"
        if spec.native_svg is not None:
            native_svg = project_path(spec.native_svg)
            if not native_svg.is_file():
                raise FileNotFoundError(native_svg)
            copy_if_different(native_svg, svg_target)
            svg_provenance = spec.vector_provenance
            svg_origin = relative_origin(native_svg)
        else:
            write_raster_svg(png_target, svg_target)
            svg_provenance = "raster-embedded exact manuscript PNG; not native vector"
            svg_origin = relative_origin(png_target)
        figure_outputs.append((svg_target, "svg", svg_provenance, svg_origin))

        tiff_target = figure_dir / f"{spec.figure_id}.tiff"
        write_tiff(png_target, tiff_target)
        figure_outputs.append(
            (
                tiff_target,
                "tiff",
                "lossless LZW 600-dpi conversion from exact manuscript PNG",
                relative_origin(png_target),
            )
        )

        for output, file_format, provenance, origin in figure_outputs:
            manifest_rows.append(
                {
                    "artifact_type": "figure",
                    "figure_id": spec.figure_id,
                    "panel_mapping": "all",
                    "format": file_format,
                    "relative_path": output.relative_to(destination).as_posix(),
                    "origin": origin,
                    "provenance": provenance,
                    "bytes": output.stat().st_size,
                    "rows": "",
                    "columns": "",
                    "width_px": width if file_format in {"png", "tiff"} else "",
                    "height_px": height if file_format in {"png", "tiff"} else "",
                    "sha256": sha256(output),
                }
            )

        for source in source_map[spec.figure_id]:
            if not source.path.is_file():
                raise FileNotFoundError(source.path)
            target = data_dir / source.path.name
            shutil.copy2(source.path, target)
            rows, columns = csv_shape(target)
            origin = relative_origin(source.path)
            source_rows.append(
                {
                    "figure_id": spec.figure_id,
                    "panel_mapping": source.panels,
                    "csv_file": target.relative_to(destination).as_posix(),
                    "description": source.description,
                    "origin": origin,
                    "rows": rows,
                    "columns": columns,
                    "sha256": sha256(target),
                }
            )
            manifest_rows.append(
                {
                    "artifact_type": "source_data",
                    "figure_id": spec.figure_id,
                    "panel_mapping": source.panels,
                    "format": "csv",
                    "relative_path": target.relative_to(destination).as_posix(),
                    "origin": origin,
                    "provenance": source.description,
                    "bytes": target.stat().st_size,
                    "rows": rows,
                    "columns": columns,
                    "width_px": "",
                    "height_px": "",
                    "sha256": sha256(target),
                }
            )

        for target, panels, description, origin in add_generated_source_tables(
            spec.figure_id,
            data_dir,
        ):
            rows, columns = csv_shape(target)
            source_rows.append(
                {
                    "figure_id": spec.figure_id,
                    "panel_mapping": panels,
                    "csv_file": target.relative_to(destination).as_posix(),
                    "description": description,
                    "origin": origin,
                    "rows": rows,
                    "columns": columns,
                    "sha256": sha256(target),
                }
            )
            manifest_rows.append(
                {
                    "artifact_type": "source_data",
                    "figure_id": spec.figure_id,
                    "panel_mapping": panels,
                    "format": "csv",
                    "relative_path": target.relative_to(destination).as_posix(),
                    "origin": origin,
                    "provenance": description,
                    "bytes": target.stat().st_size,
                    "rows": rows,
                    "columns": columns,
                    "width_px": "",
                    "height_px": "",
                    "sha256": sha256(target),
                }
            )

    write_rows(
        destination / "manifest.csv",
        [
            "artifact_type",
            "figure_id",
            "panel_mapping",
            "format",
            "relative_path",
            "origin",
            "provenance",
            "bytes",
            "rows",
            "columns",
            "width_px",
            "height_px",
            "sha256",
        ],
        manifest_rows,
    )
    write_rows(
        destination / "figure_source_map.csv",
        [
            "figure_id",
            "panel_mapping",
            "csv_file",
            "description",
            "origin",
            "rows",
            "columns",
            "sha256",
        ],
        source_rows,
    )

    expected_formats = {"png", "pdf", "svg", "tiff"}
    for spec in FIGURES:
        formats = {
            path.suffix.lower().lstrip(".")
            for path in (figure_root / spec.figure_id).iterdir()
            if path.is_file()
        }
        if formats != expected_formats:
            raise RuntimeError(
                f"{spec.figure_id} formats are {sorted(formats)}, "
                f"expected {sorted(expected_formats)}."
            )
        if not any((source_root / spec.figure_id).glob("*.csv")):
            raise RuntimeError(f"No CSV mapping exists for {spec.figure_id}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package manuscript figures and panel-level source data."
    )
    parser.add_argument(
        "--manuscript-figure-dir",
        type=Path,
        default=DEFAULT_MANUSCRIPT_FIGURE_DIR,
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=MODULE_ROOT,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package(
        args.manuscript_figure_dir.resolve(),
        args.destination.resolve(),
    )
    print(f"Packaged manuscript results in {args.destination.resolve()}")


if __name__ == "__main__":
    main()
