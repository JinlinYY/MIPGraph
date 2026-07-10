"""Export source data behind Fig/dataset_statistics.png as panel-wise CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

PROPERTIES = (
    "Density",
    "ElectricalConductivity",
    "HeatCapacity",
    "SurfaceTension",
    "ThermalConductivity",
    "Viscosity",
)
PROPERTY_UNITS = {
    "Density": "kg m^-3",
    "ElectricalConductivity": "S m^-1",
    "HeatCapacity": "J K^-1 mol^-1",
    "SurfaceTension": "N m^-1",
    "ThermalConductivity": "W m^-1 K^-1",
    "Viscosity": "Pa s",
}
LOG_SCALE_PROPERTIES = {"ElectricalConductivity", "Viscosity"}
SOURCE_ORDER = (
    "observed",
    "exact_condition_copy",
    "temperature_interpolation",
)


def load_dataset(tables_dir: Path, input_xlsx: Path) -> pd.DataFrame:
    parquet = tables_dir / "dataset_with_family.parquet"
    csv = tables_dir / "dataset_with_family.csv"
    if parquet.exists():
        try:
            return pd.read_parquet(parquet)
        except Exception:
            pass
    if csv.exists():
        return pd.read_csv(csv)
    return pd.read_excel(input_xlsx, sheet_name="Merged")


def append_record(rows: list[dict], panel: str, panel_title: str,
                  measure: str, value: object, **kwargs: object) -> None:
    record = {
        "figure": "dataset_statistics",
        "panel": panel,
        "panel_title": panel_title,
        "measure": measure,
        "value": value,
    }
    record.update(kwargs)
    rows.append(record)


def collapse_top_families(counts: pd.DataFrame,
                          column: str,
                          top_n: int,
                          ion_type: str) -> pd.DataFrame:
    ordered = counts.sort_values("N_IL", ascending=False).reset_index(drop=True)
    keep = ordered.iloc[:top_n].copy()
    if len(ordered) > top_n:
        other = pd.DataFrame([{
            column: "Other",
            "N_IL": int(ordered.iloc[top_n:]["N_IL"].sum()),
            "Pct_IL": float(ordered.iloc[top_n:]["Pct_IL"].sum()),
        }])
        keep = pd.concat([keep, other], ignore_index=True)
    keep["ion_type"] = ion_type
    keep["family"] = keep[column].astype(str)
    return (keep[["ion_type", "family", "N_IL", "Pct_IL"]]
            .groupby(["ion_type", "family"], as_index=False)
            .agg({"N_IL": "sum", "Pct_IL": "sum"})
            .sort_values("N_IL", ascending=False)
            .reset_index(drop=True))


def build_source_data(tables_dir: Path, input_xlsx: Path) -> pd.DataFrame:
    df = load_dataset(tables_dir, input_xlsx)
    prop_counts = pd.read_csv(tables_dir / "property_sample_counts.csv")
    source_counts = pd.read_csv(tables_dir / "label_source_counts.csv")
    coverage = pd.read_csv(tables_dir / "il_label_coverage.csv")
    coverage_hist = pd.read_csv(tables_dir / "il_coverage_histogram.csv")
    cation_counts = pd.read_csv(tables_dir / "cation_family_counts.csv")
    anion_counts = pd.read_csv(tables_dir / "anion_family_counts.csv")

    rows: list[dict] = []

    # Panel A: summary numbers and source-aware expansion totals.
    panel = "A"
    title = "Dataset scale and source-aware label expansion"
    append_record(rows, panel, title, "condition_records", len(df), unit="records")
    append_record(rows, panel, title, "unique_ionic_liquids",
                  int(df["IL_SMILES"].nunique()), unit="ILs")
    append_record(rows, panel, title, "available_property_labels",
                  int(prop_counts["N_Measurements"].sum()), unit="labels")
    source_totals = (source_counts.groupby("Source_Category")["N_Labels"]
                     .sum()
                     .reindex(SOURCE_ORDER, fill_value=0))
    for category, value in source_totals.items():
        append_record(rows, panel, title, "label_source_total", int(value),
                      source_category=category, unit="labels")

    # Panel B: property-wise stacked label availability.
    panel = "B"
    title = "Property-wise label availability after curation"
    for _, row in source_counts.iterrows():
        append_record(rows, panel, title, "property_label_count",
                      int(row["N_Labels"]),
                      property=row["Property"],
                      source_category=row["Source_Category"],
                      unit="labels")
    for _, row in prop_counts.iterrows():
        append_record(rows, panel, title, "property_total_labels",
                      int(row["N_Measurements"]),
                      property=row["Property"], unit="labels")
        append_record(rows, panel, title, "property_unique_ils",
                      int(row["N_Unique_IL"]),
                      property=row["Property"], unit="ILs")

    # Panel C: sorted sparse IL-property matrix.
    panel = "C"
    title = "Sorted sparse IL-property label-presence matrix"
    sorted_cov = coverage.sort_values(
        ["N_Labels", *PROPERTIES],
        ascending=[False, *([False] * len(PROPERTIES))]
    ).reset_index(drop=True)
    for rank, row in sorted_cov.iterrows():
        for prop in PROPERTIES:
            append_record(rows, panel, title, "label_presence",
                          int(row[prop]),
                          property=prop,
                          il_rank=rank + 1,
                          il_smiles=row["IL_SMILES"],
                          il_name=row.get("IL_Name", ""),
                          n_labels=int(row["N_Labels"]),
                          unit="binary")

    # Panel D: per-IL label coverage histogram.
    panel = "D"
    title = "Distribution of available property labels per IL"
    for _, row in coverage_hist.iterrows():
        append_record(rows, panel, title, "il_count_by_label_number",
                      int(row["N_IL"]),
                      n_labels=int(row["N_Labels"]),
                      pct_il=float(row["Pct_IL"]),
                      unit="ILs")

    # Panel E: temperature-bin counts by property.
    panel = "E"
    title = "Temperature coverage of the six target properties"
    all_t = df["Temperature_K"].dropna()
    lo = max(150.0, float(np.floor(all_t.quantile(0.005) / 10.0) * 10.0))
    hi = min(750.0, float(np.ceil(all_t.quantile(0.995) / 10.0) * 10.0))
    bins = np.linspace(lo, hi, 38)
    centers = 0.5 * (bins[:-1] + bins[1:])
    for prop in PROPERTIES:
        vals = df.loc[df[f"{prop}_ActualValue"].notna(), "Temperature_K"].dropna()
        counts, _ = np.histogram(vals, bins=bins)
        for i, count in enumerate(counts):
            append_record(rows, panel, title, "temperature_bin_label_count",
                          int(count),
                          property=prop,
                          bin_index=i + 1,
                          bin_left=float(bins[i]),
                          bin_right=float(bins[i + 1]),
                          bin_center=float(centers[i]),
                          unit="labels")

    # Panel F: displayed major ion families.
    panel = "F"
    title = "Major cation and anion families"
    family_rows = pd.concat([
        collapse_top_families(cation_counts, "Cation_Family", 4, "cation"),
        collapse_top_families(anion_counts, "Anion_Family", 4, "anion"),
    ], ignore_index=True)
    for _, row in family_rows.iterrows():
        append_record(rows, panel, title, "family_il_count",
                      int(row["N_IL"]),
                      ion_type=row["ion_type"],
                      ion_family=row["family"],
                      pct_il=float(row["Pct_IL"]),
                      unit="ILs")

    # Panel G: all values used for robust-scaled property box plots.
    panel = "G"
    title = "Robust-scaled property-value ranges"
    for prop in PROPERTIES:
        value_col = f"{prop}_ActualValue"
        valid = df[df[value_col].notna()].copy()
        if prop in LOG_SCALE_PROPERTIES:
            valid = valid[valid[value_col] > 0].copy()
            transformed = np.log10(valid[value_col].astype(float))
            transform = "log10"
        else:
            transformed = valid[value_col].astype(float)
            transform = "identity"
        q_lo, q_hi = transformed.quantile([0.01, 0.99])
        if q_hi <= q_lo:
            q_lo, q_hi = transformed.min(), transformed.max()
        scaled = ((transformed - q_lo) / (q_hi - q_lo)).clip(-0.15, 1.15)
        for row_idx, (idx, row) in enumerate(valid.iterrows(), start=1):
            append_record(rows, panel, title, "robust_scaled_property_value",
                          float(scaled.loc[idx]),
                          property=prop,
                          unit=PROPERTY_UNITS[prop],
                          transform=transform,
                          raw_value=float(row[value_col]),
                          transformed_value=float(transformed.loc[idx]),
                          q01=float(q_lo),
                          q99=float(q_hi),
                          property_row_index=row_idx,
                          dataset_row_index=int(idx),
                          il_smiles=row.get("IL_SMILES", ""),
                          il_name=row.get("IL_Name", ""),
                          temperature_k=float(row["Temperature_K"])
                          if pd.notna(row.get("Temperature_K")) else np.nan,
                          pressure_kpa=float(row["Pressure_kPa"])
                          if pd.notna(row.get("Pressure_kPa")) else np.nan)

    # Panel H: pressure coverage categories.
    panel = "H"
    title = "Pressure coverage"
    pressure = df["Pressure_kPa"].dropna()
    positive = pressure[pressure > 0]
    ambient = int(np.isclose(positive.to_numpy(), 101.325, atol=1e-6).sum())
    non_ambient = int(len(pressure) - ambient)
    missing = int(df["Pressure_kPa"].isna().sum())
    pressure_min = float(positive.min()) if len(positive) else np.nan
    pressure_max = float(positive.max()) if len(positive) else np.nan
    for category, value in (
        ("ambient_101.325_kpa", ambient),
        ("non_ambient_reported", non_ambient),
        ("missing_pressure", missing),
    ):
        append_record(rows, panel, title, "pressure_record_count",
                      value,
                      pressure_category=category,
                      pressure_min_kpa=pressure_min,
                      pressure_max_kpa=pressure_max,
                      unit="records")

    out = pd.DataFrame(rows)
    return out


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export panel-wise source data for Fig/dataset_statistics.png.")
    parser.add_argument(
        "--tables-dir", type=Path,
        default=Path("experiments/dataset_analysis/outputs_v2_interpolated_nature/tables"))
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/processed/"
                     "ionic_liquid_6_properties_values_errors_ilthermo_strict_v2_interpolated.xlsx"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("LaTex-MIPGraph/Fig"))
    parser.add_argument(
        "--prefix", default="dataset_statistics_source_data",
        help="Output filename prefix. Files are written as <prefix>_<panel>.csv.")
    return parser.parse_args(list(argv) if argv is not None else None)


def _drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that are entirely empty within a panel-specific table."""

    keep = []
    for col in df.columns:
        series = df[col]
        non_empty = series.notna()
        if series.dtype == object:
            non_empty &= series.astype(str).str.len() > 0
        if non_empty.any():
            keep.append(col)
    return df[keep]


def write_panel_csvs(source_data: pd.DataFrame,
                     output_dir: Path,
                     prefix: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for panel in sorted(source_data["panel"].dropna().unique()):
        panel_df = source_data[source_data["panel"] == panel].copy()
        panel_df = _drop_empty_columns(panel_df)
        out_path = output_dir / f"{prefix}_{panel}.csv"
        panel_df.to_csv(out_path, index=False)
        written.append(out_path)
    return written


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    source_data = build_source_data(args.tables_dir, args.input)
    written = write_panel_csvs(source_data, args.output_dir, args.prefix)
    for path in written:
        n_rows = sum(1 for _ in path.open("r", encoding="utf-8")) - 1
        print(f"Wrote {path} with {n_rows:,} rows")


if __name__ == "__main__":
    main()
