from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "Supporting Information_2" / "data set"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "ionic_liquid_6_properties_values_errors.xlsx"


DATASETS = [
    ("Density.xlsx", "Density", "ActualValue", "ErrorValue"),
    ("ElectricalConductivity.xlsx", "ElectricalConductivity", "ActualValue", "ErrorValue"),
    ("HeatCapacity.xlsx", "HeatCapacity", "ActualValue", "ErrorValue"),
    ("SurfaceTension.xlsx", "SurfaceTension", "Surface_tension", "ErrorValue"),
    ("ThermalConductivity.xlsx", "ThermalConductivity", "ActualValue", "ErrorValue"),
    ("Viscosity.xlsx", "Viscosity", "ActualValue", "ErrorValue"),
]


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\xa0", " ").strip())


def normalize_key(value: Any) -> str:
    return normalize_text(value).lower()


def compact(value: Any) -> str:
    return re.sub(r"[^0-9a-z]+", "", normalize_key(value))


def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    columns = {compact(column): str(column) for column in df.columns}
    for candidate in candidates:
        column = columns.get(compact(candidate))
        if column:
            return column
    raise ValueError(f"Missing column from candidates: {candidates}")


def load_ion_map(sheet_name: str) -> dict[str, dict[str, str]]:
    ion_map: dict[str, dict[str, str]] = {}
    for file_name, *_ in DATASETS:
        df = pd.read_excel(RAW_DIR / file_name, sheet_name=sheet_name)
        for _, row in df.iterrows():
            full_name = normalize_text(row.get("FullName"))
            short_name = normalize_text(row.get("ShortName"))
            if full_name and short_name:
                ion_map.setdefault(
                    normalize_key(full_name),
                    {"FullName": full_name, "ShortName": short_name},
                )
    return ion_map


def name_variants(name: str) -> list[str]:
    variants = [name]
    no_formula = re.sub(r"\s+chemical formula:\s*.*$", "", name, flags=re.IGNORECASE)
    variants.append(no_formula)
    no_ratio = re.sub(r"\s*\(\s*\d+\s*:\s*\d+\s*\)\s*$", "", no_formula)
    variants.append(no_ratio)
    return list(dict.fromkeys(variants))


def parse_ions(
    il_name: Any,
    cation_map: dict[str, dict[str, str]],
    anion_map: dict[str, dict[str, str]],
    cation_keys: list[str],
    anion_keys: list[str],
) -> dict[str, str]:
    for name in name_variants(normalize_key(il_name)):
        for cation_key in cation_keys:
            if not name.startswith(cation_key + " "):
                continue
            rest = name[len(cation_key) + 1 :]
            for anion_key in anion_keys:
                if (
                    rest == anion_key
                    or rest.startswith(anion_key + " ")
                    or compact(rest) == compact(anion_key)
                ):
                    cation = cation_map[cation_key]
                    anion = anion_map[anion_key]
                    return {
                        "Cation_FullName": cation["FullName"],
                        "Cation_ShortName": cation["ShortName"],
                        "Anion_FullName": anion["FullName"],
                        "Anion_ShortName": anion["ShortName"],
                    }
            cation = cation_map[cation_key]
            return {
                "Cation_FullName": cation["FullName"],
                "Cation_ShortName": cation["ShortName"],
                "Anion_FullName": rest,
                "Anion_ShortName": "",
            }
        for anion_key in anion_keys:
            if not name.endswith(" " + anion_key):
                continue
            cation_key = name[: -(len(anion_key) + 1)]
            if cation_key in cation_map:
                cation = cation_map[cation_key]
                anion = anion_map[anion_key]
                return {
                    "Cation_FullName": cation["FullName"],
                    "Cation_ShortName": cation["ShortName"],
                    "Anion_FullName": anion["FullName"],
                    "Anion_ShortName": anion["ShortName"],
                }
    return {
        "Cation_FullName": "",
        "Cation_ShortName": "",
        "Anion_FullName": "",
        "Anion_ShortName": "",
    }


def first_nonempty(values: pd.Series) -> str:
    for value in values:
        text = normalize_text(value)
        if text:
            return text
    return ""


def read_property_dataset(
    file_name: str,
    property_name: str,
    value_column: str,
    error_column: str,
    cation_map: dict[str, dict[str, str]],
    anion_map: dict[str, dict[str, str]],
    cation_keys: list[str],
    anion_keys: list[str],
) -> pd.DataFrame:
    df = pd.read_excel(RAW_DIR / file_name, sheet_name="Sheet1")
    name_col = find_column(df, ["Names", "Name"])
    temp_col = find_column(df, ["Temperature", "Temperature, K", "Temperature/K"])
    value_col = find_column(df, [value_column])
    error_col = find_column(df, [error_column])

    try:
        pressure_col = find_column(df, ["Pressure", "Pressure, kPa"])
        pressure = pd.to_numeric(df[pressure_col], errors="coerce")
    except ValueError:
        pressure = pd.Series([pd.NA] * len(df))

    ions = pd.DataFrame(
        [
            parse_ions(name, cation_map, anion_map, cation_keys, anion_keys)
            for name in df[name_col]
        ]
    )
    out = pd.DataFrame(
        {
            "IL_Name": df[name_col].map(normalize_text),
            "IL_Name_Key": df[name_col].map(normalize_key),
            "Temperature_K": pd.to_numeric(df[temp_col], errors="coerce"),
            "Pressure_kPa": pressure,
            f"{property_name}_ActualValue": pd.to_numeric(df[value_col], errors="coerce"),
            f"{property_name}_ErrorValue": pd.to_numeric(df[error_col], errors="coerce"),
        }
    )
    out = pd.concat([out, ions], axis=1)
    out["IL_Key"] = out.apply(
        lambda row: (
            f"{row['Cation_ShortName']}-{row['Anion_ShortName']}"
            if row["Cation_ShortName"] and row["Anion_ShortName"]
            else row["IL_Name_Key"]
        ),
        axis=1,
    )

    group_columns = ["IL_Key", "Temperature_K", "Pressure_kPa"]
    value_columns = [f"{property_name}_ActualValue", f"{property_name}_ErrorValue"]
    meta_columns = [
        "IL_Name",
        "Cation_FullName",
        "Cation_ShortName",
        "Anion_FullName",
        "Anion_ShortName",
    ]
    return (
        out.groupby(group_columns, dropna=False)
        .agg(
            **{column: (column, first_nonempty) for column in meta_columns},
            **{column: (column, "mean") for column in value_columns},
        )
        .reset_index()
    )


def merge_all() -> pd.DataFrame:
    cation_map = load_ion_map("Sheet2")
    anion_map = load_ion_map("Sheet3")
    cation_keys = sorted(cation_map, key=len, reverse=True)
    anion_keys = sorted(anion_map, key=len, reverse=True)

    merged: pd.DataFrame | None = None
    for file_name, property_name, value_column, error_column in DATASETS:
        current = read_property_dataset(
            file_name,
            property_name,
            value_column,
            error_column,
            cation_map,
            anion_map,
            cation_keys,
            anion_keys,
        )
        if merged is None:
            merged = current
            continue
        merged = merged.merge(
            current,
            on=["IL_Key", "Temperature_K", "Pressure_kPa"],
            how="outer",
            suffixes=("", "_new"),
        )
        for column in [
            "IL_Name",
            "Cation_FullName",
            "Cation_ShortName",
            "Anion_FullName",
            "Anion_ShortName",
        ]:
            merged[column] = merged[column].combine_first(merged[f"{column}_new"])
            merged.drop(columns=[f"{column}_new"], inplace=True)

    if merged is None:
        raise RuntimeError("No datasets were merged")

    columns = [
        "IL_Name",
        "Cation_FullName",
        "Cation_ShortName",
        "Anion_FullName",
        "Anion_ShortName",
        "Temperature_K",
        "Pressure_kPa",
    ]
    for _, property_name, _, _ in DATASETS:
        columns.extend([f"{property_name}_ActualValue", f"{property_name}_ErrorValue"])

    return merged[columns].sort_values(["IL_Name", "Temperature_K", "Pressure_kPa"])


def write_excel(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Merged", index=False)
        worksheet = writer.sheets["Merged"]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        for idx, column in enumerate(df.columns, start=1):
            sample = [str(column), *df[column].head(300).fillna("").astype(str).tolist()]
            width = min(max(max(len(value) for value in sample) + 2, 10), 45)
            worksheet.column_dimensions[worksheet.cell(row=1, column=idx).column_letter].width = width


def main() -> None:
    merged = merge_all()
    write_excel(merged)
    print(f"wrote: {OUTPUT_PATH}")
    print(f"rows: {len(merged)}")
    print(f"columns: {len(merged.columns)}")


if __name__ == "__main__":
    main()
