from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROPERTY_NAMES = [
    "Density",
    "ElectricalConductivity",
    "HeatCapacity",
    "SurfaceTension",
    "ThermalConductivity",
    "Viscosity",
]

META_COLUMNS = [
    "IL_Name",
    "IL_SMILES",
    "Cation_FullName",
    "Cation_ShortName",
    "Anion_FullName",
    "Anion_ShortName",
    "Temperature_K",
    "Pressure_kPa",
]


def _to_float_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _stats(series: pd.Series) -> dict[str, float | None]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"min": None, "max": None, "mean": None, "median": None}
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(values.median()),
    }


def preprocess_excel(
    raw_path: str | Path,
    processed_dir: str | Path,
    sheet_name: str = "Merged",
    properties: list[str] | None = None,
    evaluation_reference_path: str | Path | None = None,
) -> dict[str, Any]:
    properties = properties or PROPERTY_NAMES
    raw_path = Path(raw_path)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(raw_path, sheet_name=sheet_name)
    evaluation_reference = None
    if evaluation_reference_path is not None:
        evaluation_reference = pd.read_excel(evaluation_reference_path, sheet_name=sheet_name)
        if len(evaluation_reference) != len(df):
            raise ValueError("Evaluation reference and augmented workbook have different row counts")
        for key in ("IL_Name", "Temperature_K"):
            if key not in evaluation_reference.columns or not evaluation_reference[key].fillna("").equals(df[key].fillna("")):
                raise ValueError(f"Evaluation reference row alignment failed for {key}")
    failed_rows: list[dict[str, Any]] = []
    for col in META_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    before = len(df)
    df["Temperature_K"] = _to_float_series(df["Temperature_K"])
    df["Pressure_kPa"] = _to_float_series(df["Pressure_kPa"])
    missing_temp = df["Temperature_K"].isna()
    for idx in df.index[missing_temp]:
        failed_rows.append({"row_index": int(idx), "reason": "missing Temperature_K"})
    df = df.loc[~missing_temp].copy()
    if evaluation_reference is not None:
        evaluation_reference = evaluation_reference.loc[~missing_temp].copy()

    y = np.full((len(df), len(properties)), np.nan, dtype=np.float32)
    y_error = np.full_like(y, np.nan)
    mask = np.zeros_like(y, dtype=np.float32)
    error_mask = np.zeros_like(y, dtype=np.float32)
    label_weight = np.ones_like(y, dtype=np.float32)
    evaluation_mask = np.zeros_like(y, dtype=np.float32)

    report: dict[str, Any] = {
        "raw_rows": int(before),
        "total_samples": int(len(df)),
        "unique_ils": int(df["IL_SMILES"].nunique(dropna=True)),
        "properties": {},
        "Temperature_K": _stats(df["Temperature_K"]),
        "Pressure_kPa": _stats(df["Pressure_kPa"]),
    }

    for p_idx, prop in enumerate(properties):
        value_col = f"{prop}_ActualValue"
        error_col = f"{prop}_ErrorValue"
        if value_col not in df.columns:
            raise ValueError(f"Missing required column: {value_col}")
        if error_col not in df.columns:
            raise ValueError(f"Missing required column: {error_col}")
        values = _to_float_series(df[value_col])
        errors = _to_float_series(df[error_col])
        source_col = f"{prop}_ValueSource"
        source_counts: dict[str, int] = {}
        if source_col in df.columns:
            sources = df[source_col].fillna("").astype(str).str.strip().str.lower()
            exact_copy = sources.eq("exact_condition_copy")
            interpolated = sources.isin(
                ["temperature_linear_interpolation", "temperature_log_interpolation"]
            )
            label_weight[exact_copy.to_numpy(), p_idx] = 0.5
            label_weight[interpolated.to_numpy(), p_idx] = 0.25
            source_counts = {
                "exact_condition_copy": int(exact_copy.sum()),
                "temperature_interpolation": int(interpolated.sum()),
            }
        non_positive = values.notna() & (values <= 0)
        if non_positive.any():
            warnings.warn(f"{prop} has {int(non_positive.sum())} non-positive values; kept by default.")
        y[:, p_idx] = values.to_numpy(dtype=np.float32, na_value=np.nan)
        y_error[:, p_idx] = errors.to_numpy(dtype=np.float32, na_value=np.nan)
        mask[:, p_idx] = values.notna().to_numpy(dtype=np.float32)
        error_mask[:, p_idx] = errors.notna().to_numpy(dtype=np.float32)
        if evaluation_reference is not None:
            reference_values = _to_float_series(evaluation_reference[value_col])
            evaluation_mask[:, p_idx] = reference_values.notna().to_numpy(dtype=np.float32)
        else:
            evaluation_mask[:, p_idx] = mask[:, p_idx] * (label_weight[:, p_idx] >= 1.0)
        valid = int(mask[:, p_idx].sum())
        report["properties"][prop] = {
            "label_count": valid,
            "missing_rate": float(1.0 - valid / max(len(df), 1)),
            "ActualValue": _stats(values),
            "ErrorValue": _stats(errors),
            "value_sources": source_counts,
        }
        df[value_col] = values
        df[error_col] = errors

    df.insert(0, "sample_id", np.arange(len(df), dtype=np.int64))
    for p_idx, prop in enumerate(properties):
        df[f"{prop}_mask"] = mask[:, p_idx]
        df[f"{prop}_error_mask"] = error_mask[:, p_idx]

    clean_csv = processed_dir / "il_multiprop_clean.csv"
    arrays_path = processed_dir / "il_multiprop_arrays.npz"
    report_path = processed_dir / "preprocessing_report.json"
    unique_path = processed_dir / "unique_ils.csv"
    failed_path = processed_dir / "failed_rows.csv"

    df.to_csv(clean_csv, index=False)
    np.savez_compressed(
        arrays_path,
        y=y,
        mask=mask,
        y_error=y_error,
        error_mask=error_mask,
        label_weight=label_weight,
        evaluation_mask=evaluation_mask,
        temperature=df["Temperature_K"].to_numpy(dtype=np.float32),
        pressure=df["Pressure_kPa"].to_numpy(dtype=np.float32),
        sample_id=df["sample_id"].to_numpy(dtype=np.int64),
        property_names=np.array(properties),
    )
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    df[["IL_Name", "IL_SMILES"]].drop_duplicates("IL_SMILES").to_csv(unique_path, index=False)
    pd.DataFrame(failed_rows).to_csv(failed_path, index=False)
    return {
        "clean_csv": str(clean_csv),
        "arrays_path": str(arrays_path),
        "report_path": str(report_path),
        "unique_ils": str(unique_path),
        "failed_rows": str(failed_path),
        "report": report,
    }
