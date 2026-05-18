"""Statistical analysis of the multi-property ionic-liquid dataset.

This script:

1. Loads the curated Excel file
   ``ionic_liquid_6_properties_values_errors_ilthermo_strict.xlsx``.
2. Counts the number of measurements available for each of the six
   thermophysical properties.
3. Builds the IL × Property label-presence matrix and per-IL coverage.
4. Summarises the temperature and pressure distributions.
5. Summarises the value distribution of every property.
6. Assigns a coarse cation / anion family label (via
   :mod:`family_classifier`) to each unique ionic liquid.
7. Writes all intermediate tables to ``outputs/tables/`` so the
   plotting script can be run without re-doing any work.

Run::

    python analyze_dataset.py \
        --input data/processed/ionic_liquid_6_properties_values_errors_ilthermo_strict.xlsx \
        --output-dir exp1_dataset_analysis/outputs
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

# Allow running the script directly from the command line by extending
# ``sys.path`` with its containing directory before importing the local
# helper module.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from family_classifier import (  # noqa: E402  (import after sys.path tweak)
    classify_anion,
    classify_cation,
    split_il_smiles,
)

logger = logging.getLogger("analyze_dataset")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
PROPERTIES: Sequence[str] = (
    "Density",
    "ElectricalConductivity",
    "HeatCapacity",
    "SurfaceTension",
    "ThermalConductivity",
    "Viscosity",
)

# Pretty labels and SI units used for tables / figures.
PROPERTY_DISPLAY: Dict[str, str] = {
    "Density":                "Density",
    "ElectricalConductivity": "Electrical conductivity",
    "HeatCapacity":           "Heat capacity",
    "SurfaceTension":         "Surface tension",
    "ThermalConductivity":    "Thermal conductivity",
    "Viscosity":              "Viscosity",
}

PROPERTY_UNITS: Dict[str, str] = {
    "Density":                "kg m$^{-3}$",
    "ElectricalConductivity": "S m$^{-1}$",
    "HeatCapacity":           "J K$^{-1}$ mol$^{-1}$",
    "SurfaceTension":         "N m$^{-1}$",
    "ThermalConductivity":    "W m$^{-1}$ K$^{-1}$",
    "Viscosity":              "Pa s",
}

# Properties whose values span many orders of magnitude → show on log scale.
LOG_SCALE_PROPERTIES = frozenset({"ElectricalConductivity", "Viscosity"})


# --------------------------------------------------------------------------- #
# Configuration container
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Paths:
    input_xlsx: Path
    output_dir: Path
    sheet_name: str = "Merged"

    @property
    def tables_dir(self) -> Path:
        return self.output_dir / "tables"

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    def ensure(self) -> None:
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_dataset(paths: Paths) -> pd.DataFrame:
    """Load the merged dataset from the Excel workbook."""

    logger.info("Loading %s (sheet=%s)", paths.input_xlsx, paths.sheet_name)
    df = pd.read_excel(paths.input_xlsx, sheet_name=paths.sheet_name)
    logger.info("Loaded %d rows × %d columns", *df.shape)
    return df


# --------------------------------------------------------------------------- #
# Property level statistics
# --------------------------------------------------------------------------- #
def property_value_columns() -> List[str]:
    return [f"{p}_ActualValue" for p in PROPERTIES]


def property_sample_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Number of non-NaN measurements per property and # unique ILs covered."""

    rows = []
    for prop in PROPERTIES:
        col = f"{prop}_ActualValue"
        present = df[col].notna()
        n_meas = int(present.sum())
        n_il = int(df.loc[present, "IL_SMILES"].nunique())
        rows.append({
            "Property":           prop,
            "Property_Display":   PROPERTY_DISPLAY[prop],
            "Unit":               PROPERTY_UNITS[prop],
            "N_Measurements":     n_meas,
            "N_Unique_IL":        n_il,
            "Pct_Total_Rows":     100.0 * n_meas / len(df),
        })
    return pd.DataFrame(rows)


def property_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Min / quantile / max statistics per property."""

    rows = []
    for prop in PROPERTIES:
        col = f"{prop}_ActualValue"
        s = df[col].dropna()
        if s.empty:
            continue
        q = s.quantile([0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0])
        rows.append({
            "Property":     prop,
            "Unit":         PROPERTY_UNITS[prop],
            "N":            int(s.size),
            "Mean":         float(s.mean()),
            "Std":          float(s.std()),
            "Min":          float(q.loc[0.00]),
            "Q01":          float(q.loc[0.01]),
            "Q25":          float(q.loc[0.25]),
            "Median":       float(q.loc[0.50]),
            "Q75":          float(q.loc[0.75]),
            "Q99":          float(q.loc[0.99]),
            "Max":          float(q.loc[1.00]),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# IL-level label coverage
# --------------------------------------------------------------------------- #
def il_label_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean matrix (IL × Property) marking whether ≥1 measurement exists."""

    presence_per_row = pd.DataFrame(
        {p: df[f"{p}_ActualValue"].notna() for p in PROPERTIES},
        index=df.index,
    )
    presence_per_row["IL_SMILES"] = df["IL_SMILES"].values
    presence_per_row["IL_Name"] = df["IL_Name"].values
    matrix = (presence_per_row
              .groupby(["IL_SMILES", "IL_Name"], dropna=False)[list(PROPERTIES)]
              .any()
              .reset_index())
    return matrix


def il_label_coverage(label_matrix: pd.DataFrame) -> pd.DataFrame:
    """Per-IL: how many of the six properties are labelled."""

    coverage = label_matrix.copy()
    coverage["N_Labels"] = coverage[list(PROPERTIES)].sum(axis=1).astype(int)
    return coverage[["IL_SMILES", "IL_Name", "N_Labels"] + list(PROPERTIES)]


def coverage_histogram(coverage: pd.DataFrame) -> pd.DataFrame:
    """Number of ILs with k labels (k = 1..6)."""

    rows = []
    for k in range(1, len(PROPERTIES) + 1):
        n = int((coverage["N_Labels"] == k).sum())
        rows.append({"N_Labels": k,
                     "N_IL": n,
                     "Pct_IL": 100.0 * n / len(coverage)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# T / P statistics
# --------------------------------------------------------------------------- #
def temperature_pressure_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, unit in (("Temperature_K", "K"), ("Pressure_kPa", "kPa")):
        s = df[col].dropna()
        rows.append({
            "Variable":  col,
            "Unit":      unit,
            "N":         int(s.size),
            "N_Missing": int(df[col].isna().sum()),
            "Mean":      float(s.mean()),
            "Std":       float(s.std()),
            "Min":       float(s.min()),
            "Q25":       float(s.quantile(0.25)),
            "Median":    float(s.median()),
            "Q75":       float(s.quantile(0.75)),
            "Max":       float(s.max()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Family classification
# --------------------------------------------------------------------------- #
def assign_families(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-IL DataFrame with split SMILES + family labels.

    Classification is performed once per unique IL_SMILES rather than per
    measurement row.
    """

    unique_ils = (df[["IL_SMILES", "IL_Name",
                      "Cation_ShortName", "Anion_ShortName"]]
                  .drop_duplicates(subset=["IL_SMILES"])
                  .reset_index(drop=True))

    splits = unique_ils["IL_SMILES"].apply(split_il_smiles)
    unique_ils["Cation_SMILES"] = splits.apply(lambda t: t[0])
    unique_ils["Anion_SMILES"] = splits.apply(lambda t: t[1])
    unique_ils["Cation_Family"] = unique_ils["Cation_SMILES"].apply(classify_cation)
    unique_ils["Anion_Family"] = unique_ils["Anion_SMILES"].apply(classify_anion)
    return unique_ils


def family_counts(per_il: pd.DataFrame, column: str) -> pd.DataFrame:
    counts = (per_il[column]
              .value_counts(dropna=False)
              .rename_axis(column)
              .reset_index(name="N_IL"))
    counts["Pct_IL"] = 100.0 * counts["N_IL"] / counts["N_IL"].sum()
    return counts


def merge_family_into_data(df: pd.DataFrame, per_il: pd.DataFrame) -> pd.DataFrame:
    keep = ["IL_SMILES", "Cation_SMILES", "Anion_SMILES",
            "Cation_Family", "Anion_Family"]
    return df.merge(per_il[keep], on="IL_SMILES", how="left")


# --------------------------------------------------------------------------- #
# Combined summary table
# --------------------------------------------------------------------------- #
def build_summary(df: pd.DataFrame,
                  prop_counts: pd.DataFrame,
                  prop_values: pd.DataFrame,
                  coverage: pd.DataFrame,
                  per_il: pd.DataFrame) -> pd.DataFrame:
    """A single CSV summarising the key numbers for the paper."""

    rows: List[Dict[str, object]] = []

    rows.append({"Section": "Dataset", "Item": "Total measurements",
                 "Value": int(len(df))})
    rows.append({"Section": "Dataset", "Item": "Unique ionic liquids",
                 "Value": int(df["IL_SMILES"].nunique())})
    rows.append({"Section": "Dataset", "Item": "Unique cations (short name)",
                 "Value": int(df["Cation_ShortName"].nunique())})
    rows.append({"Section": "Dataset", "Item": "Unique anions (short name)",
                 "Value": int(df["Anion_ShortName"].nunique())})
    rows.append({"Section": "Dataset", "Item": "Unique cation families",
                 "Value": int(per_il["Cation_Family"].nunique())})
    rows.append({"Section": "Dataset", "Item": "Unique anion families",
                 "Value": int(per_il["Anion_Family"].nunique())})

    for _, r in prop_counts.iterrows():
        rows.append({"Section": "Property counts",
                     "Item":    f"{r['Property']} (N measurements)",
                     "Value":   int(r["N_Measurements"])})
        rows.append({"Section": "Property counts",
                     "Item":    f"{r['Property']} (N unique ILs)",
                     "Value":   int(r["N_Unique_IL"])})

    for _, r in prop_values.iterrows():
        rows.append({"Section": "Property values",
                     "Item":    f"{r['Property']} median [{r['Unit']}]",
                     "Value":   float(r["Median"])})
        rows.append({"Section": "Property values",
                     "Item":    f"{r['Property']} min [{r['Unit']}]",
                     "Value":   float(r["Min"])})
        rows.append({"Section": "Property values",
                     "Item":    f"{r['Property']} max [{r['Unit']}]",
                     "Value":   float(r["Max"])})

    rows.append({"Section": "Coverage",
                 "Item":    "Mean labels per IL",
                 "Value":   float(coverage["N_Labels"].mean())})
    rows.append({"Section": "Coverage",
                 "Item":    "Median labels per IL",
                 "Value":   float(coverage["N_Labels"].median())})
    rows.append({"Section": "Coverage",
                 "Item":    "ILs with all 6 labels",
                 "Value":   int((coverage["N_Labels"] == 6).sum())})
    rows.append({"Section": "Coverage",
                 "Item":    "ILs with only 1 label",
                 "Value":   int((coverage["N_Labels"] == 1).sum())})

    s_T = df["Temperature_K"].dropna()
    s_P = df["Pressure_kPa"].dropna()
    rows.append({"Section": "Conditions",
                 "Item":    "Temperature min [K]",
                 "Value":   float(s_T.min())})
    rows.append({"Section": "Conditions",
                 "Item":    "Temperature max [K]",
                 "Value":   float(s_T.max())})
    rows.append({"Section": "Conditions",
                 "Item":    "Temperature median [K]",
                 "Value":   float(s_T.median())})
    rows.append({"Section": "Conditions",
                 "Item":    "Pressure min [kPa]",
                 "Value":   float(s_P.min())})
    rows.append({"Section": "Conditions",
                 "Item":    "Pressure max [kPa]",
                 "Value":   float(s_P.max())})
    rows.append({"Section": "Conditions",
                 "Item":    "Pressure median [kPa]",
                 "Value":   float(s_P.median())})
    rows.append({"Section": "Conditions",
                 "Item":    "Pressure missing rows",
                 "Value":   int(df["Pressure_kPa"].isna().sum())})

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_analysis(paths: Paths) -> Dict[str, pd.DataFrame]:
    paths.ensure()
    df = load_dataset(paths)

    prop_counts = property_sample_counts(df)
    prop_values = property_value_summary(df)
    label_matrix = il_label_matrix(df)
    coverage = il_label_coverage(label_matrix)
    cov_hist = coverage_histogram(coverage)
    tp_summary = temperature_pressure_summary(df)
    per_il = assign_families(df)

    cation_counts = family_counts(per_il, "Cation_Family")
    anion_counts = family_counts(per_il, "Anion_Family")

    df_with_family = merge_family_into_data(df, per_il)
    summary = build_summary(df_with_family, prop_counts, prop_values,
                            coverage, per_il)

    outputs = {
        "raw_with_family":           df_with_family,
        "property_sample_counts":    prop_counts,
        "property_value_summary":    prop_values,
        "il_label_matrix":           label_matrix,
        "il_label_coverage":         coverage,
        "il_coverage_histogram":     cov_hist,
        "temperature_pressure_summary": tp_summary,
        "per_il_family":             per_il,
        "cation_family_counts":      cation_counts,
        "anion_family_counts":       anion_counts,
        "summary_statistics":        summary,
    }

    write_outputs(outputs, paths)
    log_summary(summary)
    return outputs


def write_outputs(outputs: Dict[str, pd.DataFrame], paths: Paths) -> None:
    paths.tables_dir.mkdir(parents=True, exist_ok=True)
    out_files = {
        "property_sample_counts":      "property_sample_counts.csv",
        "property_value_summary":      "property_value_summary.csv",
        "il_label_matrix":             "il_label_matrix.csv",
        "il_label_coverage":           "il_label_coverage.csv",
        "il_coverage_histogram":       "il_coverage_histogram.csv",
        "temperature_pressure_summary": "temperature_pressure_summary.csv",
        "per_il_family":               "per_il_family_assignment.csv",
        "cation_family_counts":        "cation_family_counts.csv",
        "anion_family_counts":         "anion_family_counts.csv",
        "summary_statistics":          "summary_statistics.csv",
    }
    for key, fname in out_files.items():
        path = paths.tables_dir / fname
        outputs[key].to_csv(path, index=False)
        logger.info("Wrote %s (%d rows)", path, len(outputs[key]))

    raw_path = paths.tables_dir / "dataset_with_family.parquet"
    try:
        outputs["raw_with_family"].to_parquet(raw_path, index=False)
        logger.info("Wrote %s", raw_path)
    except Exception as exc:  # pragma: no cover - depends on optional deps
        logger.warning("Could not write parquet (%s); falling back to CSV.", exc)
        csv_path = paths.tables_dir / "dataset_with_family.csv"
        outputs["raw_with_family"].to_csv(csv_path, index=False)
        logger.info("Wrote %s", csv_path)


def log_summary(summary: pd.DataFrame) -> None:
    logger.info("---- Dataset summary ----")
    for _, r in summary.iterrows():
        logger.info("[%-15s] %-40s = %s",
                    r["Section"], r["Item"], r["Value"])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Statistical analysis of the multi-property ionic "
                     "liquid dataset."))
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/processed/"
                     "ionic_liquid_6_properties_values_errors_ilthermo_strict.xlsx"),
        help="Path to the merged Excel file.")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("exp1_dataset_analysis/outputs"),
        help="Directory for tables/ and figures/ subfolders.")
    parser.add_argument(
        "--sheet", default="Merged",
        help="Excel sheet name to read (default: Merged).")
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    paths = Paths(input_xlsx=args.input,
                  output_dir=args.output_dir,
                  sheet_name=args.sheet)
    run_analysis(paths)


if __name__ == "__main__":
    main()
