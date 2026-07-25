"""Package manuscript figures and their panel-level CSV source data.

The analysis pipeline writes regenerable artifacts to ``results/``.  This
utility copies only the manuscript-facing figures and exact plotted data into
``experiments/manuscript_figure_source_data/molecular_origin_analysis/`` and
adds a checksummed manifest plus a compact column dictionary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import Iterable


MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_ROOT.parents[1]

FIGURES = {
    "figure_main_molecular_origin_analysis_final.pdf": (
        "Main Figure",
        "Vector manuscript figure",
    ),
    "figure_main_molecular_origin_analysis_final.png": (
        "Main Figure",
        "Raster manuscript figure (600 dpi)",
    ),
    "figure_main_molecular_origin_analysis_final.svg": (
        "Main Figure",
        "Editable vector manuscript figure",
    ),
    "figure_main_molecular_origin_analysis_final.tiff": (
        "Main Figure",
        "Lossless 600-dpi raster manuscript figure",
    ),
    "figure_main_molecular_origin_analysis_final_17p8cm.png": (
        "Main Figure",
        "Final-width readability check at 17.8 cm",
    ),
    "figure_si_heat_capacity_size_control.pdf": (
        "Supplementary Figure",
        "Vector heat-capacity size-control figure",
    ),
    "figure_si_heat_capacity_size_control.png": (
        "Supplementary Figure",
        "Raster heat-capacity size-control figure",
    ),
    "figure_si_heat_capacity_size_control.svg": (
        "Supplementary Figure",
        "Editable vector heat-capacity size-control figure",
    ),
}

SOURCE_TABLES = {
    "figure_main_molecular_origin_analysis_final_panel_a_source_data.csv": (
        "Main Figure, panel a",
        "Identity-balanced association and attention evidence-map records",
    ),
    "figure_main_molecular_origin_analysis_final_panel_b_source_data.csv": (
        "Main Figure, panel b",
        "Property-specific leading structure-property associations",
    ),
    "figure_main_molecular_origin_analysis_final_panel_c_source_data.csv": (
        "Main Figure, panel c",
        "Binned condition-adjusted response curves and bootstrap intervals",
    ),
    "figure_main_molecular_origin_analysis_final_panel_d_source_data.csv": (
        "Main Figure, panel d",
        "Observed and predicted matched ion-substitution effects",
    ),
    "figure_si_heat_capacity_size_control_source_data.csv": (
        "Supplementary Figure",
        "Identity-level heat-capacity size-control records",
    ),
}

DESCRIPTIONS = {
    "property": "Thermophysical property represented by the row.",
    "feature": "Machine-readable structural descriptor name.",
    "full_feature_label": "Complete display label used for the descriptor.",
    "structural_scope": "Cation, anion, or ion-pair scope of the descriptor.",
    "data_type": "Experimental or model-derived evidence source.",
    "analysis_weighting": "Weighting unit used in the statistical analysis.",
    "n_records": "Number of condition-level records.",
    "n_unique_ils": "Number of unique ionic-liquid identities.",
    "n_cation_families": "Number of represented cation families.",
    "n_anion_families": "Number of represented anion families.",
    "partial_correlation": "Condition-adjusted experimental partial correlation.",
    "partial_r": "Condition-adjusted experimental partial correlation.",
    "model_partial_correlation": "Condition-adjusted model-response partial correlation.",
    "partial_p": "Unadjusted P value for the experimental partial correlation.",
    "model_partial_p": "Unadjusted P value for the model-response partial correlation.",
    "fdr_q": "False-discovery-rate-adjusted q value.",
    "model_fdr_q": "FDR-adjusted q value for the model-response association.",
    "bootstrap_ci_low": "Lower bound of the stated bootstrap confidence interval.",
    "bootstrap_ci_high": "Upper bound of the stated bootstrap confidence interval.",
    "selection_stability": "Bootstrap selection frequency.",
    "confidence_level": "Audited evidence level used in the figure.",
    "line_alpha": "Plotting opacity used for the evidence level.",
    "line_width_pt": "Plotting line width encoding effect magnitude.",
    "signed_effect": "Signed statistic visualized in the evidence map.",
    "abs_effect": "Absolute magnitude of the visualized statistic.",
    "quantile_bin": "Ordinal descriptor bin.",
    "quantile_label": "Display label of the descriptor bin.",
    "sample_count": "Number of identity-level observations in the bin.",
    "feature_bin_median": "Median descriptor value in the bin.",
    "response_log_mean": "Mean condition-adjusted response on the natural-log scale.",
    "substitution_pair_id": "Stable identifier for one unordered ion-substitution pair.",
    "fixed_role": "Whether the cation or anion is held fixed.",
    "fixed_ion_smiles": "SMILES of the ion held fixed.",
    "left_changed_ion_smiles": "SMILES of the first substituted ion.",
    "right_changed_ion_smiles": "SMILES of the second substituted ion.",
    "n_condition_matches": "Number of matched experimental conditions.",
    "observed_log_delta": "Signed observed property difference on the natural-log scale.",
    "observed_abs_log_difference": "Absolute observed log-property difference.",
    "predicted_log_delta": "Signed model-predicted property difference on the natural-log scale.",
    "predicted_abs_log_difference": "Absolute model-predicted log-property difference.",
    "il_identity_key": "Charge-aware ionic-liquid identity key.",
    "molar_mass_g_mol": "Ion-pair molar mass.",
    "condition_adjusted_response": "Condition-adjusted identity-level response.",
    "analysis_panel": "Supplementary panel membership.",
    "exploratory": "Whether the result is explicitly marked exploratory.",
}

UNITS = {
    "partial_correlation": "dimensionless",
    "partial_r": "dimensionless",
    "model_partial_correlation": "dimensionless",
    "partial_p": "dimensionless",
    "model_partial_p": "dimensionless",
    "fdr_q": "dimensionless",
    "model_fdr_q": "dimensionless",
    "bootstrap_ci_low": "same scale as associated statistic",
    "bootstrap_ci_high": "same scale as associated statistic",
    "selection_stability": "fraction",
    "line_alpha": "fraction",
    "line_width_pt": "pt",
    "signed_effect": "dimensionless",
    "abs_effect": "dimensionless",
    "response_log_mean": "natural-log response",
    "observed_log_delta": "natural-log response",
    "observed_abs_log_difference": "natural-log response",
    "predicted_log_delta": "natural-log response",
    "predicted_abs_log_difference": "natural-log response",
    "molar_mass_g_mol": "g mol^-1",
    "condition_adjusted_response": "natural-log response",
}


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_shape(path: Path) -> tuple[int, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = sum(1 for _ in reader)
    return rows, len(header)


def csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return next(csv.reader(handle))


def humanize(name: str) -> str:
    return name.replace("_", " ").strip().capitalize() + "."


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_readme(path: Path) -> None:
    figure_lines = "\n".join(
        f"- `figures/{filename}` — {mapping}"
        for filename, (mapping, _) in FIGURES.items()
    )
    table_lines = "\n".join(
        f"- `source_data/{filename}` — {mapping}"
        for filename, (mapping, _) in SOURCE_TABLES.items()
    )
    path.write_text(
        f"""# Molecular-origin analysis: manuscript figure source data

## Scope

This bundle contains the exact manuscript-facing figures and panel-level CSV
tables exported by `experiments/molecular_origin_analysis/`.  The tables
support inspection and redrawing of the plotted results; they are processed
figure source data, not the raw thermophysical-property database.

## Figure-to-data map

### Figures

{figure_lines}

### Source-data tables

{table_lines}

## Interpretation boundaries

The association, attention-contrast and matched-substitution results are
non-causal.  Thermal-conductivity results marked `exploratory=True` retain that
qualification.  SMILES strings are included only where they define the
matched-substitution analysis unit.

## Provenance and integrity

`manifest.csv` records the source path, manuscript mapping, file size, table
shape and SHA-256 checksum.  `column_dictionary.csv` defines the exported
fields and units.  Rebuild the analysis outputs first, then refresh this bundle:

```powershell
python experiments\\molecular_origin_analysis\\run_all.py --stage all
python experiments\\molecular_origin_analysis\\package_source_data.py
```
""",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package molecular-origin manuscript figures and source-data CSVs."
    )
    return parser.parse_args()


def main() -> int:
    parse_args()
    results_dir = (MODULE_ROOT / "results").resolve()
    destination = (
        PROJECT_ROOT
        / "experiments"
        / "manuscript_figure_source_data"
        / "molecular_origin_analysis"
    ).resolve()
    figures_dir = results_dir / "figures"
    source_dir = results_dir / "tables" / "figure_source_data"

    inputs: list[tuple[Path, str, str, str]] = []
    for filename, (mapping, description) in FIGURES.items():
        inputs.append((figures_dir / filename, f"figures/{filename}", mapping, description))
    for filename, (mapping, description) in SOURCE_TABLES.items():
        inputs.append(
            (source_dir / filename, f"source_data/{filename}", mapping, description)
        )

    missing = [source for source, _, _, _ in inputs if not source.is_file()]
    if missing:
        details = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Required analysis artifacts are missing:\n{details}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary)
        (staging / "figures").mkdir()
        (staging / "source_data").mkdir()

        manifest_rows: list[dict[str, object]] = []
        dictionary_rows: list[dict[str, object]] = []
        for source, relative_name, mapping, description in inputs:
            target = staging / relative_name
            shutil.copy2(source, target)
            rows: int | str = ""
            columns: int | str = ""
            if target.suffix.lower() == ".csv":
                rows, columns = csv_shape(target)
                for column in csv_header(target):
                    dictionary_rows.append(
                        {
                            "source_data_file": target.name,
                            "column": column,
                            "description": DESCRIPTIONS.get(column, humanize(column)),
                            "unit_or_scale": UNITS.get(column, "not applicable / categorical"),
                        }
                    )
            manifest_rows.append(
                {
                    "artifact_type": "source_data"
                    if target.suffix.lower() == ".csv"
                    else "figure",
                    "manuscript_mapping": mapping,
                    "description": description,
                    "destination_relative_path": relative_name.replace("\\", "/"),
                    "source_relative_path": source.relative_to(PROJECT_ROOT)
                    .as_posix(),
                    "bytes": target.stat().st_size,
                    "rows": rows,
                    "columns": columns,
                    "sha256": sha256(target),
                }
            )

        write_csv(
            staging / "manifest.csv",
            [
                "artifact_type",
                "manuscript_mapping",
                "description",
                "destination_relative_path",
                "source_relative_path",
                "bytes",
                "rows",
                "columns",
                "sha256",
            ],
            manifest_rows,
        )
        write_csv(
            staging / "column_dictionary.csv",
            ["source_data_file", "column", "description", "unit_or_scale"],
            dictionary_rows,
        )
        write_readme(staging / "README.md")

        if destination.exists():
            shutil.rmtree(destination)
        # Create the final directory under its real parent so it inherits the
        # repository ACL. Moving TemporaryDirectory itself can retain its
        # intentionally restrictive temporary-directory permissions on Windows.
        destination.mkdir(parents=True)
        shutil.copytree(staging, destination, dirs_exist_ok=True)

    print(f"Packaged {len(inputs)} artifacts in {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
